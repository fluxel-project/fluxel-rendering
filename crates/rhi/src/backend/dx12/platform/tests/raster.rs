//! Real D3D12 raster evidence.
//!
//! These tests deliberately read pixels back.  A successful `DrawInstanced` is
//! weaker evidence than an output: it cannot distinguish a graphics PSO that
//! ran from one with a broken vertex fetch, root table, sampler heap, RTV, or
//! resource-state transition.

use std::sync::Arc;

use super::{block_on, copy_lane, portable_device, settle};
use crate::api::binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayoutDescriptor, BindingCount,
    BindingKind, BindingResource, BindingSlot, BindingSlotId, SamplerKind, TextureSampleType,
};
use crate::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
    RecorderDescriptor, StoreOp,
};
use crate::api::format::TextureFormat;
use crate::api::identity::Label;
use crate::api::pipeline::{
    ColorTargetState, PipelineInterfaceDescriptor, RasterPipelineDescriptor, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexInputState, VertexStepMode,
};
use crate::api::resource::buffer::{BufferBinding, BufferDescriptor, BufferRange, BufferUsage};
use crate::api::resource::sampler::SamplerDescriptor;
use crate::api::resource::subresource::{
    HostTexelLayout, Origin3d, TextureAspect, TextureSubresourceLayers,
};
use crate::api::resource::texture::{Extent3d, TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{ReadbackRequest, ReadbackViewData, TextureUploadDescriptor};
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ShaderAbiVersion, ShaderArtifact, ShaderCode,
    ShaderInterface, ShaderLocation, ShaderLocationInterface, ShaderNumericType,
    ShaderResourceRequirement, ShaderStage, ShaderStages,
};
use crate::api::submission::{CompletionState, SubmissionPlanBuilder};

const TRIANGLE_VS_DXIL: &[u8] =
    include_bytes!("../../../../../../../scripts/tests/data/dxil/triangle_vs.dxil");
const SOLID_PS_DXIL: &[u8] =
    include_bytes!("../../../../../../../scripts/tests/data/dxil/solid_ps.dxil");
const SAMPLED_PS_DXIL: &[u8] =
    include_bytes!("../../../../../../../scripts/tests/data/dxil/sampled_ps.dxil");

const LAYERS: TextureSubresourceLayers = TextureSubresourceLayers {
    aspect: TextureAspect::Color,
    mip_level: 0,
    base_layer: 0,
    layer_count: 1,
};

fn artifact(
    stage: ShaderStage,
    entry: &'static str,
    code: &'static [u8],
    interface: ShaderInterface,
    hash: u8,
) -> ShaderArtifact {
    ShaderArtifact::new(
        stage,
        entry,
        ShaderCode::Dxil(Arc::from(code)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        crate::api::shader::ShaderRequirements::new(),
        ArtifactHash([hash; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

fn vertex_artifact() -> ShaderArtifact {
    artifact(
        ShaderStage::Vertex,
        "vs_main",
        TRIANGLE_VS_DXIL,
        ShaderInterface::new()
            .with_input(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 2,
                interpolation: None,
            })
            .with_writes_position(true),
        0x71,
    )
}

fn solid_fragment_artifact() -> ShaderArtifact {
    artifact(
        ShaderStage::Fragment,
        "ps_solid",
        SOLID_PS_DXIL,
        ShaderInterface::new().with_output(ShaderLocationInterface {
            location: ShaderLocation::new(0),
            numeric_type: ShaderNumericType::Float32,
            components: 4,
            interpolation: None,
        }),
        0x72,
    )
}

fn sampled_fragment_artifact() -> ShaderArtifact {
    let texture = BindingKind::SampledTexture {
        dimension: TextureViewDimension::D2,
        sample_type: TextureSampleType::Float,
        multisampled: false,
    };
    artifact(
        ShaderStage::Fragment,
        "ps_sampled",
        SAMPLED_PS_DXIL,
        ShaderInterface::new()
            .with_resource(ShaderResourceRequirement {
                group: BindGroupIndex::new(0),
                slot: BindingSlotId::new(0),
                kind: texture,
                count: BindingCount::One,
            })
            .with_resource(ShaderResourceRequirement {
                group: BindGroupIndex::new(0),
                slot: BindingSlotId::new(1),
                kind: BindingKind::Sampler {
                    kind: SamplerKind::Filtering,
                },
                count: BindingCount::One,
            })
            .with_output(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 4,
                interpolation: None,
            }),
        0x73,
    )
}

fn target(
    device: &crate::api::platform::Device,
) -> (
    crate::api::resource::texture::Texture,
    crate::api::resource::view::TextureView,
) {
    let texture = device
        .create_texture(&TextureDescriptor::new_2d(
            8,
            8,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ))
        .expect("offscreen RGBA8 render target");
    let view = device
        .create_texture_view(
            &texture,
            &TextureViewDescriptor::whole(&texture, TextureViewDimension::D2)
                .expect("whole 2D target view"),
        )
        .expect("native RTV-compatible texture view");
    (texture, view)
}

fn scope(view: crate::api::resource::view::TextureView) -> RasterScopeDescriptor {
    RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            depth_slice: None,
            view: ColorAttachmentView::Texture(view),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
            resolve: None,
        },
    )
}

fn vertex_buffer(device: &crate::api::platform::Device) -> crate::api::resource::buffer::Buffer {
    let vertices: [u8; 24] = [
        0xCD, 0xCC, 0x4C, 0xBF, 0xCD, 0xCC, 0x4C, 0xBF, // (-0.8, -0.8)
        0x00, 0x00, 0x00, 0x00, 0xCD, 0xCC, 0x4C, 0x3F, // ( 0.0,  0.8)
        0xCD, 0xCC, 0x4C, 0x3F, 0xCD, 0xCC, 0x4C, 0xBF, // ( 0.8, -0.8)
    ];
    let buffer = device
        .create_buffer(&BufferDescriptor::new(
            vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))
        .expect("triangle vertex buffer");
    let upload = device
        .create_buffer_upload(crate::api::resource::transfer::BufferUploadDescriptor {
            label: Label(Some("dx12 raster triangle vertices".into())),
            dst: buffer.clone(),
            dst_offset: 0,
            bytes: Arc::from(vertices.as_slice()),
        })
        .expect("vertex upload");
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    recorder.encode_upload(&upload).unwrap();
    let work = recorder.finish().unwrap();
    let mut plan = SubmissionPlanBuilder::new(device);
    let point = plan.add_batch(copy_lane(device), vec![work]).unwrap();
    let receipt = device.submit(plan.build().unwrap()).unwrap();
    assert!(matches!(
        settle(device, receipt.completion_for(point).unwrap()),
        CompletionState::Complete
    ));
    buffer
}

fn raster_readback(
    device: &crate::api::platform::Device,
    target: crate::api::resource::texture::Texture,
    scope: RasterScopeDescriptor,
    pipeline: &crate::api::pipeline::RasterPipeline,
    vertex: &crate::api::resource::buffer::Buffer,
    group: Option<&crate::api::binding::BindGroup>,
) -> Vec<u8> {
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    {
        let mut raster = recorder.begin_raster(&scope).unwrap();
        raster.set_pipeline(pipeline).unwrap();
        if let Some(group) = group {
            raster
                .set_bind_group(BindGroupIndex::new(0), group, &[])
                .unwrap();
        }
        raster
            .set_vertex_buffer(
                0,
                &BufferBinding::new(vertex.clone(), BufferRange::new(0, 24)),
            )
            .unwrap();
        raster.draw(0..3, 0..1).unwrap();
        raster.end().unwrap();
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some("dx12 raster output".into())),
            src: target,
            subresource: LAYERS,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent: Extent3d::d2(8, 8),
        })
        .unwrap();
    let work = recorder.finish().unwrap();
    let mut plan = SubmissionPlanBuilder::new(device);
    let point = plan.add_batch(copy_lane(device), vec![work]).unwrap();
    let receipt = device.submit(plan.build().unwrap()).unwrap();
    assert!(matches!(
        settle(device, receipt.completion_for(point).unwrap()),
        CompletionState::Complete
    ));
    let view = block_on(ticket.read()).expect("raster readback failed");
    let ReadbackViewData::Texture { bytes, .. } = view.data() else {
        panic!("texture readback");
    };
    bytes.to_vec()
}

fn raster_pipeline(
    device: &crate::api::platform::Device,
    fragment: ShaderArtifact,
    interface: crate::api::pipeline::PipelineInterface,
) -> crate::api::pipeline::RasterPipeline {
    let vertex = block_on(device.create_shader(&vertex_artifact())).unwrap();
    let fragment = block_on(device.create_shader(&fragment)).unwrap();
    let input =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(8, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x2, 0),
            ),
        );
    block_on(
        device.create_raster_pipeline(
            &RasterPipelineDescriptor::new(vertex, interface)
                .with_fragment(fragment)
                .with_vertex_input(input)
                .with_color_target(
                    ShaderLocation::new(0),
                    ColorTargetState::new(TextureFormat::Rgba8Unorm),
                ),
        ),
    )
    .expect("real DX12 graphics PSO")
}

fn texel(bytes: &[u8], x: usize, y: usize) -> [u8; 4] {
    // Texture readback rows use D3D12's 256-byte footprint alignment.
    bytes[y * 256 + x * 4..y * 256 + x * 4 + 4]
        .try_into()
        .unwrap()
}

#[test]
fn an_offscreen_triangle_writes_deterministic_rgba8_pixels() {
    let device = portable_device();
    let (target, view) = target(&device);
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![]))
        .unwrap();
    let pipeline = raster_pipeline(&device, solid_fragment_artifact(), interface);
    let output = raster_readback(
        &device,
        target,
        scope(view),
        &pipeline,
        &vertex_buffer(&device),
        None,
    );
    assert_eq!(texel(&output, 4, 4), [64, 128, 191, 255]);
    assert_eq!(texel(&output, 0, 0), [0, 0, 0, 255]);
}

#[test]
fn a_fragment_sampled_texture_and_sampler_reach_the_raster_output() {
    let device = portable_device();
    let source = device
        .create_texture(&TextureDescriptor::new_2d(
            2,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
        ))
        .unwrap();
    let upload = device
        .create_texture_upload(TextureUploadDescriptor {
            label: Label(Some("dx12 sampled source".into())),
            dst: source.clone(),
            subresource: LAYERS,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent: Extent3d::d2(2, 2),
            source_layout: HostTexelLayout {
                bytes_per_row: 8,
                rows_per_image: 2,
            },
            bytes: Arc::from(
                [
                    16u8, 32, 64, 255, 16, 32, 64, 255, 16, 32, 64, 255, 16, 32, 64, 255,
                ]
                .as_slice(),
            ),
        })
        .unwrap();
    let mut upload_recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    upload_recorder.encode_upload(&upload).unwrap();
    let mut upload_plan = SubmissionPlanBuilder::new(&device);
    let point = upload_plan
        .add_batch(copy_lane(&device), vec![upload_recorder.finish().unwrap()])
        .unwrap();
    let receipt = device.submit(upload_plan.build().unwrap()).unwrap();
    assert!(matches!(
        settle(&device, receipt.completion_for(point).unwrap()),
        CompletionState::Complete
    ));

    let source_view = device
        .create_texture_view(
            &source,
            &TextureViewDescriptor::whole(&source, TextureViewDimension::D2).unwrap(),
        )
        .unwrap();
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::FRAGMENT,
                BindingKind::SampledTexture {
                    dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float,
                    multisampled: false,
                },
            ),
            BindingSlot::new(
                BindingSlotId::new(1),
                ShaderStages::FRAGMENT,
                BindingKind::Sampler {
                    kind: SamplerKind::Filtering,
                },
            ),
        ]))
        .unwrap();
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))
        .unwrap();
    let sampler = device.create_sampler(&SamplerDescriptor::new()).unwrap();
    let group = device
        .create_bind_group(&BindGroupDescriptor::new(layout).with_entries([
            BindGroupEntry::new(BindingSlotId::new(0), BindingResource::Texture(source_view)),
            BindGroupEntry::new(BindingSlotId::new(1), BindingResource::Sampler(sampler)),
        ]))
        .unwrap();
    let (target, view) = target(&device);
    let pipeline = raster_pipeline(&device, sampled_fragment_artifact(), interface);
    let output = raster_readback(
        &device,
        target,
        scope(view),
        &pipeline,
        &vertex_buffer(&device),
        Some(&group),
    );
    assert_eq!(texel(&output, 4, 4), [16, 32, 64, 255]);
}

const SPARSE_MRT_PS_DXIL: &[u8] =
    include_bytes!("../../../../../../../scripts/tests/data/dxil/sparse_mrt_ps.dxil");

fn sparse_mrt_fragment_artifact() -> ShaderArtifact {
    // One pixel shader with two outputs at locations 0 and 3, the sparse shape
    // that leaves MRT holes at locations 1 and 2.  The interface must declare
    // both outputs so the pipeline is built with a four-slot MRT span and the
    // hardware test can read each real target back separately.
    artifact(
        ShaderStage::Fragment,
        "ps_sparse_mrt",
        SPARSE_MRT_PS_DXIL,
        ShaderInterface::new()
            .with_output(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 4,
                interpolation: None,
            })
            .with_output(ShaderLocationInterface {
                location: ShaderLocation::new(3),
                numeric_type: ShaderNumericType::Float32,
                components: 4,
                interpolation: None,
            }),
        0x74,
    )
}

/// A render target whose backing texture is a plain offscreen RGBA8 target,
/// distinct from the `target()` helper so both sparse slots read independently.
fn sparse_target(
    device: &crate::api::platform::Device,
) -> (
    crate::api::resource::texture::Texture,
    crate::api::resource::view::TextureView,
) {
    let texture = device
        .create_texture(&TextureDescriptor::new_2d(
            8,
            8,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ))
        .expect("offscreen RGBA8 sparse-MRT render target");
    let view = device
        .create_texture_view(
            &texture,
            &TextureViewDescriptor::whole(&texture, TextureViewDimension::D2)
                .expect("whole 2D sparse-MRT target view"),
        )
        .expect("native RTV-compatible sparse-MRT texture view");
    (texture, view)
}

/// A sparse color attachment (clear-to-black, store) at the given location.
fn sparse_attachment(view: crate::api::resource::view::TextureView) -> ColorAttachment {
    ColorAttachment {
        depth_slice: None,
        view: ColorAttachmentView::Texture(view),
        load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
        resolve: None,
    }
}

/// The two-location (0 and 3) raster scope with MRT holes at 1 and 2.
fn sparse_scope(
    low: crate::api::resource::view::TextureView,
    high: crate::api::resource::view::TextureView,
) -> RasterScopeDescriptor {
    RasterScopeDescriptor::new()
        .with_color(ShaderLocation::new(0), sparse_attachment(low))
        .with_color(ShaderLocation::new(3), sparse_attachment(high))
}

/// A two-target (locations 0 and 3) raster pipeline for the sparse MRT PS.
fn sparse_raster_pipeline(
    device: &crate::api::platform::Device,
    interface: crate::api::pipeline::PipelineInterface,
) -> crate::api::pipeline::RasterPipeline {
    let vertex = block_on(device.create_shader(&vertex_artifact())).unwrap();
    let fragment = block_on(device.create_shader(&sparse_mrt_fragment_artifact())).unwrap();
    let input =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(8, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x2, 0),
            ),
        );
    block_on(
        device.create_raster_pipeline(
            &RasterPipelineDescriptor::new(vertex, interface)
                .with_fragment(fragment)
                .with_vertex_input(input)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TextureFormat::Rgba8Unorm))
                .with_color_target(ShaderLocation::new(3), ColorTargetState::new(TextureFormat::Rgba8Unorm)),
        ),
    )
    .expect("real DX12 sparse-MRT graphics PSO")
}

/// Reads one RGBA8 render target back over the copy lane after a sparse scope.
fn sparse_raster_readback(
    device: &crate::api::platform::Device,
    target: crate::api::resource::texture::Texture,
    scope: RasterScopeDescriptor,
    pipeline: &crate::api::pipeline::RasterPipeline,
    vertex: &crate::api::resource::buffer::Buffer,
) -> Vec<u8> {
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    {
        let mut raster = recorder.begin_raster(&scope).unwrap();
        raster.set_pipeline(pipeline).unwrap();
        raster
            .set_vertex_buffer(
                0,
                &BufferBinding::new(vertex.clone(), BufferRange::new(0, 24)),
            )
            .unwrap();
        raster.draw(0..3, 0..1).unwrap();
        raster.end().unwrap();
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some("dx12 sparse-MRT output".into())),
            src: target,
            subresource: LAYERS,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent: Extent3d::d2(8, 8),
        })
        .unwrap();
    let work = recorder.finish().unwrap();
    let mut plan = SubmissionPlanBuilder::new(device);
    let point = plan.add_batch(copy_lane(device), vec![work]).unwrap();
    let receipt = device.submit(plan.build().unwrap()).unwrap();
    assert!(matches!(
        settle(device, receipt.completion_for(point).unwrap()),
        CompletionState::Complete
    ));
    let view = block_on(ticket.read()).expect("raster readback failed");
    let ReadbackViewData::Texture { bytes, .. } = view.data() else {
        panic!("texture readback");
    };
    bytes.to_vec()
}

#[test]
fn a_sparse_mrt_scope_writes_both_real_locations_and_ignores_the_holes() {
    let device = portable_device();
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![]))
        .unwrap();
    let pipeline = sparse_raster_pipeline(&device, interface);
    let vertex = vertex_buffer(&device);
    let (low_texture, low_view) = sparse_target(&device);
    let (high_texture, high_view) = sparse_target(&device);

    // Location 0 is the "low" target; location 3 is the "high" target.  The
    // holes at 1 and 2 are filled with null RTVs and must not receive writes.
    let low_output = sparse_raster_readback(
        &device,
        low_texture,
        sparse_scope(low_view.clone(), high_view.clone()),
        &pipeline,
        &vertex,
    );
    let high_output = sparse_raster_readback(
        &device,
        high_texture,
        sparse_scope(low_view.clone(), high_view.clone()),
        &pipeline,
        &vertex,
    );
    // Low target (location 0): opaque mid-blue, exactly the solid PS output.
    assert_eq!(texel(&low_output, 4, 4), [64, 128, 191, 255]);
    // High target (location 3): a distinct color, proving the hole at 1/2 did
    // not shift or drop the second real attachment.
    assert_eq!(texel(&high_output, 4, 4), [128, 64, 32, 255]);
}

/// A 4x-MSSA color target that resolves into a single-sampled target, proving
/// the whole MSAA raster/resolve lifecycle reads back real pixels.
#[test]
fn a_4x_msaa_scope_resolves_into_a_single_sampled_target() {
    let device = portable_device();
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![]))
        .unwrap();
    let fragment = block_on(device.create_shader(&solid_fragment_artifact())).unwrap();
    let vertex = block_on(device.create_shader(&vertex_artifact())).unwrap();
    let input =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(8, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x2, 0),
            ),
        );
    let pipeline = block_on(
        device.create_raster_pipeline(
            &RasterPipelineDescriptor::new(vertex, interface)
                .with_fragment(fragment)
                .with_vertex_input(input)
                .with_multisample(crate::api::pipeline::MultisampleState::new(4))
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TextureFormat::Rgba8Unorm)),
        ),
    )
    .expect("real DX12 4x-MSSA graphics PSO");

    // Multisampled source: written by the 4x pipeline, resolved at scope end.
    let msaa = device
        .create_texture(
            &TextureDescriptor::new_2d(
                8,
                8,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT,
            )
            .with_sample_count(4),
        )
        .expect("4x MSAA source render target");
    let msaa_view = device
        .create_texture_view(
            &msaa,
            &TextureViewDescriptor::whole(&msaa, TextureViewDimension::D2)
                .expect("whole 4x MSAA view"),
        )
        .expect("MSAA RTV-compatible view");

    // Single-sampled resolve destination.
    let resolved = device
        .create_texture(
            &TextureDescriptor::new_2d(
                8,
                8,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
            )
            .with_sample_count(1),
        )
        .expect("single-sample resolve destination");
    let resolved_view = device
        .create_texture_view(
            &resolved,
            &TextureViewDescriptor::whole(&resolved, TextureViewDimension::D2)
                .expect("whole 1x resolve view"),
        )
        .expect("resolve target view");

    // The scope renders into the 4x source and resolves it into the 1x target.
    let scope = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            depth_slice: None,
            view: ColorAttachmentView::Texture(msaa_view),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
            resolve: Some(ColorAttachmentView::Texture(resolved_view)),
        },
    );

    let output = raster_readback(
        &device,
        resolved,
        scope,
        &pipeline,
        &vertex_buffer(&device),
        None,
    );
    // The resolved pixel is the solid PS color (0.25, 0.5, 0.75), confirming
    // ResolveSubresource moved the 4x coverage into the 1x target.
    assert_eq!(texel(&output, 4, 4), [64, 128, 191, 255]);
}
