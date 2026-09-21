//! Native Vulkan compute image-binding conformance evidence.
//!
//! This deliberately crosses the entire image-binding path in one accepted
//! workload: CPU upload, tracked transfer-to-shader transition, separate sampled
//! image and filtering-sampler descriptors, storage-image write, and readback.
//! The shader is inline Vulkan 1.0 SPIR-V generated once with
//! `glslangValidator`; executing this test never depends on a shader compiler.

use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;

use crate::api::binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayoutDescriptor, BindingCount,
    BindingKind, BindingResource, BindingSlot, BindingSlotId, SamplerKind, StorageAccess,
    TextureSampleType,
};
use crate::api::command::{ComputeScopeDescriptor, RecorderDescriptor};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceInstanceId, Label};
use crate::api::pipeline::{ComputePipelineDescriptor, PipelineInterfaceDescriptor};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, Device, PlatformProvider};
use crate::api::resource::sampler::{FilterMode, SamplerDescriptor};
use crate::api::resource::subresource::{
    HostTexelLayout, Origin3d, TextureAspect, TextureSubresourceLayers,
};
use crate::api::resource::texture::{Extent3d, TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{ReadbackRequest, ReadbackViewData, TextureUploadDescriptor};
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ComputeWorkgroupSize, ShaderAbiVersion, ShaderArtifact,
    ShaderCode, ShaderInterface, ShaderRequirements, ShaderResourceRequirement, ShaderStage,
    ShaderStages,
};
use crate::api::submission::{LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder};
use crate::backend::test_harness::{block_on, require_complete};

use super::platform::VulkanProvider;

// GLSL source used once to produce these Vulkan 1.0 words:
//
// #version 450
// layout(local_size_x = 1) in;
// layout(set = 0, binding = 0) uniform texture2D source_texture;
// layout(set = 0, binding = 1) uniform sampler filtering_sampler;
// layout(set = 0, binding = 2, rgba8) uniform writeonly image2D destination_texture;
// void main() {
//     vec4 sampled = texture(sampler2D(source_texture, filtering_sampler), vec2(0.5));
//     imageStore(destination_texture, ivec2(0), sampled);
// }
const IMAGE_BINDING_SPIRV: &[u32] = &[
    0x0723_0203,
    0x0001_0000,
    0x0008_000B,
    0x0000_0026,
    0x0000_0000,
    0x0002_0011,
    0x0000_0001,
    0x0006_000B,
    0x0000_0001,
    0x4C53_4C47,
    0x6473_742E,
    0x3035_342E,
    0x0000_0000,
    0x0003_000E,
    0x0000_0000,
    0x0000_0001,
    0x0005_000F,
    0x0000_0005,
    0x0000_0004,
    0x6E69_616D,
    0x0000_0000,
    0x0006_0010,
    0x0000_0004,
    0x0000_0011,
    0x0000_0001,
    0x0000_0001,
    0x0000_0001,
    0x0003_0003,
    0x0000_0002,
    0x0000_01C2,
    0x0004_0005,
    0x0000_0004,
    0x6E69_616D,
    0x0000_0000,
    0x0004_0005,
    0x0000_0009,
    0x706D_6173,
    0x0064_656C,
    0x0006_0005,
    0x0000_000C,
    0x7275_6F73,
    0x745F_6563,
    0x7574_7865,
    0x0000_6572,
    0x0007_0005,
    0x0000_0010,
    0x746C_6966,
    0x6E69_7265,
    0x6173_5F67,
    0x656C_706D,
    0x0000_0072,
    0x0007_0005,
    0x0000_001B,
    0x7473_6564,
    0x7461_6E69,
    0x5F6E_6F69,
    0x7478_6574,
    0x0065_7275,
    0x0004_0047,
    0x0000_000C,
    0x0000_0021,
    0x0000_0000,
    0x0004_0047,
    0x0000_000C,
    0x0000_0022,
    0x0000_0000,
    0x0004_0047,
    0x0000_0010,
    0x0000_0021,
    0x0000_0001,
    0x0004_0047,
    0x0000_0010,
    0x0000_0022,
    0x0000_0000,
    0x0003_0047,
    0x0000_001B,
    0x0000_0019,
    0x0004_0047,
    0x0000_001B,
    0x0000_0021,
    0x0000_0002,
    0x0004_0047,
    0x0000_001B,
    0x0000_0022,
    0x0000_0000,
    0x0004_0047,
    0x0000_0025,
    0x0000_000B,
    0x0000_0019,
    0x0002_0013,
    0x0000_0002,
    0x0003_0021,
    0x0000_0003,
    0x0000_0002,
    0x0003_0016,
    0x0000_0006,
    0x0000_0020,
    0x0004_0017,
    0x0000_0007,
    0x0000_0006,
    0x0000_0004,
    0x0004_0020,
    0x0000_0008,
    0x0000_0007,
    0x0000_0007,
    0x0009_0019,
    0x0000_000A,
    0x0000_0006,
    0x0000_0001,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0001,
    0x0000_0000,
    0x0004_0020,
    0x0000_000B,
    0x0000_0000,
    0x0000_000A,
    0x0004_003B,
    0x0000_000B,
    0x0000_000C,
    0x0000_0000,
    0x0002_001A,
    0x0000_000E,
    0x0004_0020,
    0x0000_000F,
    0x0000_0000,
    0x0000_000E,
    0x0004_003B,
    0x0000_000F,
    0x0000_0010,
    0x0000_0000,
    0x0003_001B,
    0x0000_0012,
    0x0000_000A,
    0x0004_0017,
    0x0000_0014,
    0x0000_0006,
    0x0000_0002,
    0x0004_002B,
    0x0000_0006,
    0x0000_0015,
    0x3F00_0000,
    0x0005_002C,
    0x0000_0014,
    0x0000_0016,
    0x0000_0015,
    0x0000_0015,
    0x0004_002B,
    0x0000_0006,
    0x0000_0017,
    0x0000_0000,
    0x0009_0019,
    0x0000_0019,
    0x0000_0006,
    0x0000_0001,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0002,
    0x0000_0004,
    0x0004_0020,
    0x0000_001A,
    0x0000_0000,
    0x0000_0019,
    0x0004_003B,
    0x0000_001A,
    0x0000_001B,
    0x0000_0000,
    0x0004_0015,
    0x0000_001D,
    0x0000_0020,
    0x0000_0001,
    0x0004_0017,
    0x0000_001E,
    0x0000_001D,
    0x0000_0002,
    0x0004_002B,
    0x0000_001D,
    0x0000_001F,
    0x0000_0000,
    0x0005_002C,
    0x0000_001E,
    0x0000_0020,
    0x0000_001F,
    0x0000_001F,
    0x0004_0015,
    0x0000_0022,
    0x0000_0020,
    0x0000_0000,
    0x0004_0017,
    0x0000_0023,
    0x0000_0022,
    0x0000_0003,
    0x0004_002B,
    0x0000_0022,
    0x0000_0024,
    0x0000_0001,
    0x0006_002C,
    0x0000_0023,
    0x0000_0025,
    0x0000_0024,
    0x0000_0024,
    0x0000_0024,
    0x0005_0036,
    0x0000_0002,
    0x0000_0004,
    0x0000_0000,
    0x0000_0003,
    0x0002_00F8,
    0x0000_0005,
    0x0004_003B,
    0x0000_0008,
    0x0000_0009,
    0x0000_0007,
    0x0004_003D,
    0x0000_000A,
    0x0000_000D,
    0x0000_000C,
    0x0004_003D,
    0x0000_000E,
    0x0000_0011,
    0x0000_0010,
    0x0005_0056,
    0x0000_0012,
    0x0000_0013,
    0x0000_000D,
    0x0000_0011,
    0x0007_0058,
    0x0000_0007,
    0x0000_0018,
    0x0000_0013,
    0x0000_0016,
    0x0000_0002,
    0x0000_0017,
    0x0003_003E,
    0x0000_0009,
    0x0000_0018,
    0x0004_003D,
    0x0000_0019,
    0x0000_001C,
    0x0000_001B,
    0x0004_003D,
    0x0000_0007,
    0x0000_0021,
    0x0000_0009,
    0x0004_0063,
    0x0000_001C,
    0x0000_0020,
    0x0000_0021,
    0x0001_00FD,
    0x0001_0038,
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
    let identity = DeviceInstanceId::new(0x1A6E);
    VulkanProvider::new(identity)
        .ok()
        .map(|native| PlatformProvider::new(BackendKind::Vulkan, identity, Box::new(native)))
}

fn compute_lane(device: &Device) -> SubmissionLaneId {
    device
        .capabilities()
        .submission()
        .lanes()
        .iter()
        .find(|lane| lane.domains().contains(LaneWorkDomains::COMPUTE))
        .map(|lane| lane.id())
        .expect("the advertised Vulkan compute path needs a COMPUTE-capable lane")
}

fn image_artifact() -> ShaderArtifact {
    let resource = |slot, kind| ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(slot),
        kind,
        count: BindingCount::One,
    };
    ShaderArtifact::new(
        ShaderStage::Compute,
        "main",
        ShaderCode::SpirV(Arc::from(IMAGE_BINDING_SPIRV)),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new()
            .with_compute_workgroup_size(ComputeWorkgroupSize::new(1, 1, 1))
            .with_resource(resource(
                0,
                BindingKind::SampledTexture {
                    dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float,
                    multisampled: false,
                },
            ))
            .with_resource(resource(
                1,
                BindingKind::Sampler {
                    kind: SamplerKind::Filtering,
                },
            ))
            .with_resource(resource(
                2,
                BindingKind::StorageTexture {
                    dimension: TextureViewDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    access: StorageAccess::WriteOnly,
                },
            )),
        ShaderRequirements::new(),
        ArtifactHash([0x1A; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

#[test]
fn compute_sampled_filtering_and_storage_images_round_trip_through_vulkan() {
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

    let source = device
        .create_texture(&TextureDescriptor::new_2d(
            2,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_DST.union(TextureUsage::SAMPLED),
        ))
        .expect("the advertised sampled RGBA8 texture must create");
    let destination = device
        .create_texture(&TextureDescriptor::new_2d(
            1,
            1,
            TextureFormat::Rgba8Unorm,
            TextureUsage::STORAGE.union(TextureUsage::COPY_SRC),
        ))
        .expect("the advertised storage RGBA8 texture must create");
    let source_view = device
        .create_texture_view(
            &source,
            &TextureViewDescriptor::whole(&source, TextureViewDimension::D2).unwrap(),
        )
        .expect("the sampled texture view must create");
    let destination_view = device
        .create_texture_view(
            &destination,
            &TextureViewDescriptor::whole(&destination, TextureViewDimension::D2).unwrap(),
        )
        .expect("the storage texture view must create");
    let sampler = device
        .create_sampler(&SamplerDescriptor::new().with_filters(
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        ))
        .expect("the filtering sampler must create");

    let shader = ready(device.create_shader(&image_artifact()))
        .expect("the inline Vulkan image SPIR-V must create");
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::COMPUTE,
                BindingKind::SampledTexture {
                    dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float,
                    multisampled: false,
                },
            ),
            BindingSlot::new(
                BindingSlotId::new(1),
                ShaderStages::COMPUTE,
                BindingKind::Sampler {
                    kind: SamplerKind::Filtering,
                },
            ),
            BindingSlot::new(
                BindingSlotId::new(2),
                ShaderStages::COMPUTE,
                BindingKind::StorageTexture {
                    dimension: TextureViewDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    access: StorageAccess::WriteOnly,
                },
            ),
        ]))
        .expect("the image bind-group layout must validate");
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))
        .expect("the image layout must make a compute pipeline interface");
    let pipeline = ready(
        device.create_compute_pipeline(&ComputePipelineDescriptor::new(
            shader.clone(),
            interface.clone(),
        )),
    )
    .expect("the Vulkan image compute pipeline must create");
    let group = device
        .create_bind_group(
            &BindGroupDescriptor::new(layout.clone())
                .with_entry(BindGroupEntry::new(
                    BindingSlotId::new(0),
                    BindingResource::Texture(source_view.clone()),
                ))
                .with_entry(BindGroupEntry::new(
                    BindingSlotId::new(1),
                    BindingResource::Sampler(sampler.clone()),
                ))
                .with_entry(BindGroupEntry::new(
                    BindingSlotId::new(2),
                    BindingResource::Texture(destination_view.clone()),
                )),
        )
        .expect("the Vulkan sampled, sampler, and storage descriptors must create");

    let color = TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let upload = device
        .create_texture_upload(TextureUploadDescriptor {
            label: Label(Some("Vulkan sampled-image upload".into())),
            dst: source.clone(),
            subresource: color,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent: Extent3d::d2(2, 2),
            source_layout: HostTexelLayout {
                bytes_per_row: 8,
                rows_per_image: 2,
            },
            // Bilinear sampling in the centre must average these four texels.
            bytes: Arc::from(
                &[
                    0, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255, 255, 255, 255, 255,
                ][..],
            ),
        })
        .expect("the sampled image upload job must validate");

    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("Vulkan recorder creation failed");
    recorder.encode_upload(&upload).unwrap();
    {
        let mut compute = recorder
            .begin_compute(&ComputeScopeDescriptor::new())
            .expect("Vulkan facts advertise this compute image route");
        compute.set_pipeline(&pipeline).unwrap();
        compute
            .set_bind_group(BindGroupIndex::new(0), &group, &[])
            .unwrap();
        compute.dispatch(1, 1, 1).unwrap();
        compute.end().unwrap();
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some("Vulkan storage-image readback".into())),
            src: destination.clone(),
            subresource: color,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent: Extent3d::d2(1, 1),
        })
        .expect("the storage image has COPY_SRC for readback");
    let work = recorder.finish().unwrap();

    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan.add_batch(compute_lane(&device), vec![work]).unwrap();
    let receipt = ready(device.submit(plan.build().unwrap()))
        .expect("the complete image workload must lower before native work is accepted");
    let completion = receipt.completion_for(point).unwrap();
    block_on(require_complete(
        &device,
        completion,
        "Vulkan sampled/storage image readback",
    ));
    let view = block_on(ticket.read()).expect("completed Vulkan storage-image readback failed");
    let ReadbackViewData::Texture { bytes, .. } = view.data() else {
        panic!("storage-image readback returned buffer data");
    };
    assert_eq!(&bytes[..4], &[128, 128, 64, 255]);
}
