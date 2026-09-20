//! Native Vulkan off-screen raster conformance evidence.
//!
//! The test deliberately crosses three lowerings in one accepted batch:
//! graphics-pipeline creation, a render-pass scope containing a draw, and a
//! subsequent texture readback.  The transition from color attachment to copy
//! source is therefore evidence for the image-layout tracker as well as the
//! draw itself.  Its shaders are inline SPIR-V generated once with
//! `glslangValidator`; test execution never depends on a shader compiler.

use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
    RecorderDescriptor, StoreOp,
};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceInstanceId, Label};
use crate::api::pipeline::{
    ColorTargetState, PipelineInterfaceDescriptor, RasterPipelineDescriptor,
};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, Device, PlatformProvider};
use crate::api::resource::subresource::{Origin3d, TextureAspect, TextureSubresourceLayers};
use crate::api::resource::texture::{Extent3d, TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{ReadbackRequest, ReadbackViewData};
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ShaderAbiVersion, ShaderArtifact, ShaderCode,
    ShaderInterface, ShaderLocation, ShaderLocationInterface, ShaderNumericType,
    ShaderRequirements, ShaderStage,
};
use crate::api::submission::{
    CompletionState, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder,
};

use super::platform::VulkanProvider;

const VERT_SPIRV: &[u32] = &[
    0x07230203, 0x00010000, 0x0008000B, 0x00000028, 0x00000000, 0x00020011, 0x00000001, 0x0006000B,
    0x00000001, 0x4C534C47, 0x6474732E, 0x3035342E, 0x00000000, 0x0003000E, 0x00000000, 0x00000001,
    0x0007000F, 0x00000000, 0x00000004, 0x6E69616D, 0x00000000, 0x00000018, 0x0000001C, 0x00030003,
    0x00000002, 0x000001C2, 0x00040005, 0x00000004, 0x6E69616D, 0x00000000, 0x00050005, 0x0000000C,
    0x69736F70, 0x6E6F6974, 0x00000073, 0x00060005, 0x00000016, 0x505F6C67, 0x65567265, 0x78657472,
    0x00000000, 0x00060006, 0x00000016, 0x00000000, 0x505F6C67, 0x7469736F, 0x006E6F69, 0x00070006,
    0x00000016, 0x00000001, 0x505F6C67, 0x746E696F, 0x657A6953, 0x00000000, 0x00070006, 0x00000016,
    0x00000002, 0x435F6C67, 0x4470696C, 0x61747369, 0x0065636E, 0x00070006, 0x00000016, 0x00000003,
    0x435F6C67, 0x446C6C75, 0x61747369, 0x0065636E, 0x00030005, 0x00000018, 0x00000000, 0x00060005,
    0x0000001C, 0x565F6C67, 0x65747265, 0x646E4978, 0x00007865, 0x00030047, 0x00000016, 0x00000002,
    0x00050048, 0x00000016, 0x00000000, 0x0000000B, 0x00000000, 0x00050048, 0x00000016, 0x00000001,
    0x0000000B, 0x00000001, 0x00050048, 0x00000016, 0x00000002, 0x0000000B, 0x00000003, 0x00050048,
    0x00000016, 0x00000003, 0x0000000B, 0x00000004, 0x00040047, 0x0000001C, 0x0000000B, 0x0000002A,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000002, 0x00040015, 0x00000008, 0x00000020, 0x00000000,
    0x0004002B, 0x00000008, 0x00000009, 0x00000003, 0x0004001C, 0x0000000A, 0x00000007, 0x00000009,
    0x00040020, 0x0000000B, 0x00000006, 0x0000000A, 0x0004003B, 0x0000000B, 0x0000000C, 0x00000006,
    0x0004002B, 0x00000006, 0x0000000D, 0xBF800000, 0x0005002C, 0x00000007, 0x0000000E, 0x0000000D,
    0x0000000D, 0x0004002B, 0x00000006, 0x0000000F, 0x40400000, 0x0005002C, 0x00000007, 0x00000010,
    0x0000000F, 0x0000000D, 0x0005002C, 0x00000007, 0x00000011, 0x0000000D, 0x0000000F, 0x0006002C,
    0x0000000A, 0x00000012, 0x0000000E, 0x00000010, 0x00000011, 0x00040017, 0x00000013, 0x00000006,
    0x00000004, 0x0004002B, 0x00000008, 0x00000014, 0x00000001, 0x0004001C, 0x00000015, 0x00000006,
    0x00000014, 0x0006001E, 0x00000016, 0x00000013, 0x00000006, 0x00000015, 0x00000015, 0x00040020,
    0x00000017, 0x00000003, 0x00000016, 0x0004003B, 0x00000017, 0x00000018, 0x00000003, 0x00040015,
    0x00000019, 0x00000020, 0x00000001, 0x0004002B, 0x00000019, 0x0000001A, 0x00000000, 0x00040020,
    0x0000001B, 0x00000001, 0x00000019, 0x0004003B, 0x0000001B, 0x0000001C, 0x00000001, 0x00040020,
    0x0000001E, 0x00000006, 0x00000007, 0x0004002B, 0x00000006, 0x00000021, 0x00000000, 0x0004002B,
    0x00000006, 0x00000022, 0x3F800000, 0x00040020, 0x00000026, 0x00000003, 0x00000013, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200F8, 0x00000005, 0x0003003E, 0x0000000C,
    0x00000012, 0x0004003D, 0x00000019, 0x0000001D, 0x0000001C, 0x00050041, 0x0000001E, 0x0000001F,
    0x0000000C, 0x0000001D, 0x0004003D, 0x00000007, 0x00000020, 0x0000001F, 0x00050051, 0x00000006,
    0x00000023, 0x00000020, 0x00000000, 0x00050051, 0x00000006, 0x00000024, 0x00000020, 0x00000001,
    0x00070050, 0x00000013, 0x00000025, 0x00000023, 0x00000024, 0x00000021, 0x00000022, 0x00050041,
    0x00000026, 0x00000027, 0x00000018, 0x0000001A, 0x0003003E, 0x00000027, 0x00000025, 0x000100FD,
    0x00010038,
];

const FRAG_SPIRV: &[u32] = &[
    0x07230203, 0x00010000, 0x0008000B, 0x0000000F, 0x00000000, 0x00020011, 0x00000001, 0x0006000B,
    0x00000001, 0x4C534C47, 0x6474732E, 0x3035342E, 0x00000000, 0x0003000E, 0x00000000, 0x00000001,
    0x0006000F, 0x00000004, 0x00000004, 0x6E69616D, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
    0x00000007, 0x00030003, 0x00000002, 0x000001C2, 0x00040005, 0x00000004, 0x6E69616D, 0x00000000,
    0x00050005, 0x00000009, 0x5F74756F, 0x6F6C6F63, 0x00000072, 0x00040047, 0x00000009, 0x0000001E,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003,
    0x00000007, 0x0004003B, 0x00000008, 0x00000009, 0x00000003, 0x0004002B, 0x00000006, 0x0000000A,
    0x3E000000, 0x0004002B, 0x00000006, 0x0000000B, 0x3F000000, 0x0004002B, 0x00000006, 0x0000000C,
    0x3F600000, 0x0004002B, 0x00000006, 0x0000000D, 0x3F800000, 0x0007002C, 0x00000007, 0x0000000E,
    0x0000000A, 0x0000000B, 0x0000000C, 0x0000000D, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200F8, 0x00000005, 0x0003003E, 0x00000009, 0x0000000E, 0x000100FD, 0x00010038,
];

fn ready<T>(future: impl Future<Output = T>) -> T {
    let mut future = Box::pin(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the native Vulkan request unexpectedly deferred"),
    }
}

fn provider() -> Option<PlatformProvider> {
    let identity = DeviceInstanceId::new(0x0A57);
    VulkanProvider::new(identity)
        .ok()
        .map(|native| PlatformProvider::new(BackendKind::Vulkan, identity, Box::new(native)))
}

fn raster_lane(device: &Device) -> SubmissionLaneId {
    device
        .capabilities()
        .submission()
        .lanes()
        .iter()
        .find(|lane| lane.domains().contains(LaneWorkDomains::RASTER))
        .map(|lane| lane.id())
        .expect("the advertised Vulkan raster path needs a RASTER-capable lane")
}

fn artifact(
    stage: ShaderStage,
    words: &[u32],
    interface: ShaderInterface,
    hash: u8,
) -> ShaderArtifact {
    ShaderArtifact::new(
        stage,
        "main",
        ShaderCode::SpirV(Arc::from(words)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        ShaderRequirements::new(),
        ArtifactHash([hash; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

#[test]
fn offscreen_raster_draw_transitions_to_readback_and_retains_native_objects() {
    let Some(provider) = provider() else {
        return;
    };
    if ready(provider.enumerate_adapters())
        .expect("Vulkan enumeration failed")
        .as_ref()
        .is_none_or(Vec::is_empty)
    {
        return;
    }
    let device = ready(provider.request_device(DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new(),
    )))
    .expect("default Vulkan device request failed");

    let vertex = ready(device.create_shader(&artifact(
        ShaderStage::Vertex,
        VERT_SPIRV,
        ShaderInterface::new().with_writes_position(true),
        0xA1,
    )))
    .expect("inline Vulkan vertex SPIR-V must create");
    let fragment = ready(device.create_shader(&artifact(
        ShaderStage::Fragment,
        FRAG_SPIRV,
        ShaderInterface::new().with_output(ShaderLocationInterface {
            location: ShaderLocation::new(0),
            numeric_type: ShaderNumericType::Float32,
            components: 4,
            interpolation: None,
        }),
        0xA2,
    )))
    .expect("inline Vulkan fragment SPIR-V must create");
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(Vec::new()))
        .expect("an empty pipeline interface is valid for a vertex-index triangle");
    let pipeline = ready(
        device.create_raster_pipeline(
            &RasterPipelineDescriptor::new(vertex.clone(), interface.clone())
                .with_fragment(fragment.clone())
                .with_color_target(
                    ShaderLocation::new(0),
                    ColorTargetState::new(TextureFormat::Rgba8Unorm),
                ),
        ),
    )
    .expect("the advertised Vulkan RGBA8 raster pipeline must lower");

    let texture = device
        .create_texture(&TextureDescriptor::new_2d(
            8,
            8,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ))
        .expect("the advertised Vulkan offscreen attachment must create");
    let descriptor = TextureViewDescriptor::whole(&texture, TextureViewDimension::D2).unwrap();
    let view = device
        .create_texture_view(&texture, &descriptor)
        .expect("the offscreen attachment view must create");

    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("Vulkan recorder creation failed");
    {
        let mut raster = recorder
            .begin_raster(&RasterScopeDescriptor::new().with_color(
                ShaderLocation::new(0),
                ColorAttachment {
                    view: ColorAttachmentView::Texture(view.clone()),
                    load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                    store: StoreOp::Store,
                    resolve: None,
                    depth_slice: None,
                },
            ))
            .expect("Vulkan facts advertise this color attachment");
        raster.set_pipeline(&pipeline).unwrap();
        raster.draw(0..3, 0..1).unwrap();
        raster.end().unwrap();
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some("Vulkan offscreen raster output".into())),
            src: texture.clone(),
            subresource: TextureSubresourceLayers {
                aspect: TextureAspect::Color,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent: Extent3d::d2(8, 8),
        })
        .expect("the raster attachment is a declared COPY_SRC texture");
    let work = recorder.finish().unwrap();

    // The recorded work and accepted-batch retention, rather than these caller
    // handles, own the pipeline, shader modules, attachment/view and native
    // render-pass objects until the GPU completion publishes the readback.
    drop(view);
    drop(texture);
    drop(pipeline);
    drop(interface);
    drop(fragment);
    drop(vertex);

    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan.add_batch(raster_lane(&device), vec![work]).unwrap();
    let receipt = ready(device.submit(plan.build().unwrap()))
        .expect("the complete raster/readback batch must lower before native acceptance");
    let completion = receipt.completion_for(point).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while matches!(
        device.completion_state(completion).unwrap(),
        CompletionState::Pending
    ) && Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    assert!(matches!(
        device.completion_state(completion).unwrap(),
        CompletionState::Complete
    ));
    let view = ticket
        .try_read()
        .expect("completed Vulkan raster readback must be terminally readable")
        .expect("completion must publish the raster texels");
    let ReadbackViewData::Texture { bytes, layout } = view.data() else {
        panic!("raster texture readback returned buffer data");
    };
    let pixel = &bytes[..4];
    assert_eq!(pixel, &[32, 128, 223, 255]);
    assert!(layout.total_size >= 4);
}
