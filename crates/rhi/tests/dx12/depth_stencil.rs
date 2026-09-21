//! D3D12 fixture for the portable strict-depth-boundary conformance case.
//!
//! This module deliberately supplies only DXIL, graphics-PSO construction, and
//! the one-time vertex upload. Attachment allocation, command recording,
//! readback-layout handling, and CPU-visible assertions live in
//! `backend::conformance::cases::depth_stencil`, so this is the same RHI
//! workload every backend must eventually execute.

use std::sync::Arc;

use super::{block_on, copy_lane, portable_device, settle};
use crate::api::command::RecorderDescriptor;
use crate::api::pipeline::{
    ColorTargetState, DepthState, DepthStencilState, PipelineInterfaceDescriptor,
    RasterPipelineDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat, VertexInputState,
    VertexStepMode,
};
use crate::api::resource::buffer::{Buffer, BufferDescriptor, BufferUsage};
use crate::api::resource::sampler::CompareFunction;
use crate::api::resource::transfer::BufferUploadDescriptor;
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ShaderAbiVersion, ShaderArtifact, ShaderCode,
    ShaderInterface, ShaderLocation, ShaderLocationInterface, ShaderNumericType, ShaderStage,
};
use crate::api::submission::{CompletionState, SubmissionPlanBuilder};
use crate::backend::conformance::cases::depth_stencil::{
    DepthCompareCase, assert_depth_compare_boundaries, record_depth_compare_boundaries,
};

const TRIANGLE_VS_DXIL: &[u8] =
    include_bytes!("../../../../scripts/tests/data/dxil/triangle_vs.dxil");
const SOLID_PS_DXIL: &[u8] = include_bytes!("../../../../scripts/tests/data/dxil/solid_ps.dxil");

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

/// DXIL-specific fixture preparation. The portable case specifies the depth
/// comparison and asserted color; this function only creates its native-code
/// form of the pipeline.
fn raster_pipeline(device: &crate::api::platform::Device) -> crate::api::pipeline::RasterPipeline {
    let vertex = block_on(
        device.create_shader(&artifact(
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
            0x81,
        )),
    )
    .expect("triangle vertex shader");
    let fragment = block_on(device.create_shader(&artifact(
        ShaderStage::Fragment,
        "ps_solid",
        SOLID_PS_DXIL,
        ShaderInterface::new().with_output(ShaderLocationInterface {
            location: ShaderLocation::new(0),
            numeric_type: ShaderNumericType::Float32,
            components: 4,
            interpolation: None,
        }),
        0x82,
    )))
    .expect("solid fragment shader");
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![]))
        .expect("empty pipeline interface");
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
                    ColorTargetState::new(crate::api::format::TextureFormat::Rgba8Unorm),
                )
                .with_depth_stencil(
                    DepthStencilState::new(crate::api::format::TextureFormat::Depth32Float)
                        .with_depth(
                            DepthState::new(CompareFunction::Less).with_write_enabled(true),
                        ),
                ),
        ),
    )
    .expect("published D3D12 depth route must create a graphics PSO")
}

/// DXIL's position input is two little-endian `f32`s per vertex. Uploading it
/// is fixture setup; the common case owns how that buffer is bound and drawn.
fn vertex_buffer(device: &crate::api::platform::Device) -> Buffer {
    let vertices: [u8; 24] = [
        0xCD, 0xCC, 0x4C, 0xBF, 0xCD, 0xCC, 0x4C, 0xBF, 0x00, 0x00, 0x00, 0x00, 0xCD, 0xCC, 0x4C,
        0x3F, 0xCD, 0xCC, 0x4C, 0x3F, 0xCD, 0xCC, 0x4C, 0xBF,
    ];
    let buffer = device
        .create_buffer(&BufferDescriptor::new(
            vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))
        .expect("triangle vertex buffer");
    let upload = device
        .create_buffer_upload(BufferUploadDescriptor {
            label: Default::default(),
            dst: buffer.clone(),
            dst_offset: 0,
            bytes: Arc::from(vertices.as_slice()),
        })
        .expect("vertex upload");
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("recorder");
    recorder.encode_upload(&upload).expect("encode upload");
    let mut plan = SubmissionPlanBuilder::new(device);
    let point = plan
        .add_batch(
            copy_lane(device),
            vec![recorder.finish().expect("upload work")],
        )
        .expect("upload batch");
    let receipt =
        block_on(device.submit(plan.build().expect("upload plan"))).expect("upload submit");
    assert!(matches!(
        settle(
            device,
            receipt.completion_for(point).expect("upload completion")
        ),
        CompletionState::Complete
    ));
    buffer
}

#[test]
fn depth_stencil_fixture_runs_the_portable_strict_depth_boundary_case() {
    let device = portable_device();
    let pipeline = raster_pipeline(&device);
    let vertex = vertex_buffer(&device);
    let mut recording = record_depth_compare_boundaries(
        &device,
        &pipeline,
        &vertex,
        DepthCompareCase::depth32_float([64, 128, 191, 255]),
        "D3D12 strict depth boundaries",
    )
    .unwrap_or_else(|skip| {
        panic!("D3D12 published an incomplete depth conformance route: {skip:?}")
    });
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(copy_lane(&device), vec![recording.take_work()])
        .expect("D3D12 direct lane accepts the portable raster/copy workload");
    let receipt = block_on(device.submit(plan.build().expect("plan"))).expect("submit");
    assert!(matches!(
        settle(&device, receipt.completion_for(point).expect("completion")),
        CompletionState::Complete
    ));
    super::block_on(assert_depth_compare_boundaries(
        &recording,
        "D3D12 strict depth boundaries",
    ));
}
