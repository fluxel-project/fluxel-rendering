//! Native raster witnesses.

use super::native_common::Resources;
use super::native_compute::{build_k01, run_compute_case_bundle};
use crate::*;
use fluxel_rendergraph::*;
struct ComputeData {
    pipeline: ComputePipelineId,
    bindings: BindingSetId,
    storage: BufferReadWrite,
    groups: [u32; 3],
}

pub(super) struct ComputeCase {
    pub(super) compiled: fluxel_rendergraph::CompiledGraph,
    pub(super) export: ExportBufferSlot,
    pub(super) slot: fluxel_rendergraph::ImportBufferSlot,
    pub(super) input: Vec<u8>,
    pub(super) expected: Vec<u8>,
    pub(super) artifacts: Vec<(ComputePipelineId, crate::ComputeKernel, BindingSetId)>,
    pub(super) input_description: &'static str,
}

pub(super) struct RasterTextureCase {
    pub(super) compiled: fluxel_rendergraph::CompiledGraph,
    pub(super) export: fluxel_rendergraph::ExportTextureSlot,
    pipeline: RasterPipelineId,
    pub(super) expected: Vec<u8>,
}

pub(super) struct RasterIndexedCase {
    pub(super) compiled: fluxel_rendergraph::CompiledGraph,
    pub(super) export: fluxel_rendergraph::ExportTextureSlot,
    vertex_slot: fluxel_rendergraph::ImportBufferSlot,
    index_slot: fluxel_rendergraph::ImportBufferSlot,
    vertices: Vec<u8>,
    indices: Vec<u8>,
    pipeline: RasterPipelineId,
    pub(super) expected: Vec<u8>,
}

pub(super) struct RasterBufferCase {
    pub(super) compiled: fluxel_rendergraph::CompiledGraph,
    pub(super) export: ExportBufferSlot,
    pub(super) raster_pipeline: RasterPipelineId,
    pub(super) compute_pipeline: ComputePipelineId,
    pub(super) bindings: BindingSetId,
    pub(super) expected: Vec<u8>,
}

pub(super) fn add_compute(
    graph: &mut RenderGraph,
    name: &str,
    input: fluxel_rendergraph::BufferVersion,
    pipeline: ComputePipelineId,
    bindings: BindingSetId,
    range: BufferRange,
    groups: [u32; 3],
) -> fluxel_rendergraph::BufferVersion {
    graph
        .add_compute_pass(
            name,
            move |pass| {
                let (output, storage) =
                    pass.read_write_buffer(input, BufferReadWriteUse::Storage, range);
                (
                    output,
                    ComputeData {
                        pipeline,
                        bindings,
                        storage,
                        groups,
                    },
                )
            },
            |commands, resolver, data, _| {
                commands.set_pipeline(data.pipeline)?;
                let resolved = resolver.resolve_bindings(
                    data.bindings,
                    &[BindingResource::BufferReadWrite(&data.storage)],
                    &[],
                )?;
                commands.set_bindings(&resolved)?;
                commands.dispatch(data.groups)
            },
        )
        .output
}

pub(super) fn run_k01(backend: crate::Backend) {
    let case = build_k01();
    run_compute_case_bundle("K01", backend, &case);
}

pub(super) fn run_r01(backend: crate::Backend) {
    run_r01_case(backend, &build_r01(), "independent");
}

pub(super) fn build_r01() -> RasterTextureCase {
    let descriptor = TextureDesc {
        dimension: fluxel_rendergraph::TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let pipeline_id = RasterPipelineId::new(401);
    let mut graph = RenderGraph::new();
    let target = graph.create_texture("r01-target", descriptor);
    let pass = graph.add_raster_pass(
        "r01-clear-triangle",
        |pass| {
            let output = pass.color_attachment(
                target,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            (output, pipeline_id)
        },
        |commands, _, pipeline, _| {
            commands.set_pipeline(*pipeline)?;
            commands.draw(0..3, 0..1)
        },
    );
    let export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            // The graph itself requests CopySource; the external helper merely
            // consumes this reported fact and cannot repair an omitted usage.
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph
        .compile(&RasterBackend::portable_capabilities())
        .unwrap()
        .graph;
    RasterTextureCase {
        compiled,
        export,
        pipeline: pipeline_id,
        expected: r01_oracle(),
    }
}

pub(super) fn run_r01_case(backend: crate::Backend, case: &RasterTextureCase, mode: &str) {
    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let pipeline = device
        .create_raster_pipeline(crate::RasterKernel::Triangle)
        .unwrap();
    let mut provider = RasterObjectProvider::new(&device);
    provider
        .register_raster_pipeline(case.pipeline, pipeline)
        .unwrap();
    let resources = Resources {
        device: device.identity(),
        buffers: HashMap::new(),
        textures: HashMap::new(),
    };
    let executor = fluxel_rendergraph::FrameExecutor::new(RasterBackend::new(device.clone()));
    let mut frame = executor
        .execute(
            &case.compiled,
            case.compiled.instantiate_local(FrameInputs::new(())),
            &resources,
            &provider,
        )
        .unwrap();
    let completion = frame.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&completion, Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    let exported = frame.exports.texture(case.export).unwrap();
    let outgoing_state = exported.outgoing_state;
    let descriptor = exported.descriptor;
    let actual = readback_raster_exported_texture_for_test(&device, exported)
        .unwrap()
        .tight;
    assert_eq!(
        actual,
        case.expected,
        "{backend:?} first difference: {:?}",
        actual
            .iter()
            .zip(&case.expected)
            .position(|(left, right)| left != right)
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "R01/{backend:?} diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    let first_difference = actual
        .iter()
        .zip(&case.expected)
        .position(|(left, right)| left != right);
    eprintln!(
        "artifact case=R01 mode={mode} backend={backend:?} commit={commit} os={}; hardware={:?}; canonical_plan_label=R01-raster-v1; execution_plan={:?}; raster_artifact={:?}; input=draw vertices=0..3 instances=0..1 target=8x8 clear=[0,0,0,1]; resource=texture descriptor={descriptor:?}; expected={:?}; actual={actual:?}; first_difference={first_difference:?}; outgoing_state={outgoing_state:?}; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        case.compiled.execution_plan(),
        crate::RasterKernel::Triangle.portable_identity(),
        case.expected,
    );
}

pub(super) fn r01_oracle() -> Vec<u8> {
    let mut output = vec![0_u8; 8 * 8 * 4];
    for pixel in output.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0, 0, 0, 255]);
    }
    let points = [(1.2_f32, 6.4_f32), (6.8, 6.4), (4.0, 1.2)];
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    for y in 0..8 {
        for x in 0..8 {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            let edges = [
                edge(points[0], points[1], point),
                edge(points[1], points[2], point),
                edge(points[2], points[0], point),
            ];
            if edges.iter().all(|edge| *edge <= 0.0) || edges.iter().all(|edge| *edge >= 0.0) {
                output[(y * 8 + x) * 4..(y * 8 + x + 1) * 4].copy_from_slice(&[64, 160, 255, 255]);
            }
        }
    }
    output
}

struct IndexedRasterData {
    pipeline: RasterPipelineId,
    vertices: BufferRead,
    indices: BufferRead,
}

pub(super) fn run_r02(backend: crate::Backend) {
    run_r02_case(backend, &build_r02(), "independent");
}

pub(super) fn build_r02() -> RasterIndexedCase {
    let target = TextureDesc {
        dimension: fluxel_rendergraph::TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let vertices = r02_vertices();
    let indices = r02_indices();
    let mut graph = RenderGraph::new();
    let vertex_slot = graph.import_buffer_slot(
        "r02-vertices",
        ImportBufferContract {
            descriptor: BufferDesc {
                size: vertices.len() as u64,
            },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let index_slot = graph.import_buffer_slot(
        "r02-indices",
        ImportBufferContract {
            descriptor: BufferDesc {
                size: indices.len() as u64,
            },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let image = graph.create_texture("r02-target", target);
    let pipeline_id = RasterPipelineId::new(402);
    let pass = graph.add_raster_pass(
        "r02-indexed-viewport-scissor",
        |pass| {
            let output = pass.color_attachment(
                image,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let vertices = pass.read_buffer(
                &vertex_slot.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let indices = pass.read_buffer(
                &index_slot.version,
                fluxel_rendergraph::BufferReadUse::Index,
                BufferRange::Whole,
            );
            (
                output,
                IndexedRasterData {
                    pipeline: pipeline_id,
                    vertices,
                    indices,
                },
            )
        },
        |commands, _, data, _| {
            commands.set_pipeline(data.pipeline)?;
            commands.set_vertex_buffer(0, &data.vertices)?;
            commands.set_index_buffer(&data.indices, IndexFormat::Uint16)?;
            commands.set_viewport(Viewport {
                x: 1.0,
                y: 1.0,
                width: 6.0,
                height: 6.0,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            commands.set_scissor(ScissorRect {
                x: 2,
                y: 2,
                width: 4,
                height: 3,
            })?;
            commands.draw_indexed(0..6, 0, 0..1)
        },
    );
    let export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph
        .compile(&RasterBackend::portable_capabilities())
        .unwrap()
        .graph;
    RasterIndexedCase {
        compiled,
        export,
        vertex_slot: vertex_slot.slot,
        index_slot: index_slot.slot,
        vertices,
        indices,
        pipeline: pipeline_id,
        expected: r02_oracle(),
    }
}

pub(super) fn run_r02_case(backend: crate::Backend, case: &RasterIndexedCase, mode: &str) {
    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let vertex = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc {
                size: case.vertices.len() as u64,
            },
            usage: BufferUsage::from_kinds([
                BufferUsageKind::CopyDestination,
                BufferUsageKind::Vertex,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let index = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc {
                size: case.indices.len() as u64,
            },
            usage: BufferUsage::from_kinds([
                BufferUsageKind::CopyDestination,
                BufferUsageKind::Index,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let vertex_state = crate::imp::upload_buffer_for_test(
        &device.inner,
        vertex.native(),
        vertex.lease().into(),
        &case.vertices,
    )
    .unwrap();
    let index_state = crate::imp::upload_buffer_for_test(
        &device.inner,
        index.native(),
        index.lease().into(),
        &case.indices,
    )
    .unwrap();
    let resources = Resources {
        device: device.identity(),
        buffers: HashMap::from([
            (BufferBindingId::new(11), (vertex, vertex_state)),
            (BufferBindingId::new(12), (index, index_state)),
        ]),
        textures: HashMap::new(),
    };
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(case.vertex_slot, BufferBindingId::new(11));
    inputs.bind_buffer(case.index_slot, BufferBindingId::new(12));
    let mut provider = RasterObjectProvider::new(&device);
    provider
        .register_raster_pipeline(
            case.pipeline,
            device
                .create_raster_pipeline(crate::RasterKernel::IndexedPositionColor)
                .unwrap(),
        )
        .unwrap();
    let executor = fluxel_rendergraph::FrameExecutor::new(RasterBackend::new(device.clone()));
    let mut frame = executor
        .execute(
            &case.compiled,
            case.compiled.instantiate_local(inputs),
            &resources,
            &provider,
        )
        .unwrap();
    let completion = frame.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&completion, Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    let exported = frame.exports.texture(case.export).unwrap();
    let outgoing_state = exported.outgoing_state;
    let descriptor = exported.descriptor;
    let actual = readback_raster_exported_texture_for_test(&device, exported)
        .unwrap()
        .tight;
    assert_eq!(
        actual,
        case.expected,
        "{backend:?} first difference: {:?}",
        actual
            .iter()
            .zip(&case.expected)
            .position(|(left, right)| left != right)
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "R02/{backend:?} diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    let first_difference = actual
        .iter()
        .zip(&case.expected)
        .position(|(left, right)| left != right);
    eprintln!(
        "artifact case=R02 mode={mode} backend={backend:?} commit={commit} os={}; hardware={:?}; canonical_plan_label=R02-indexed-raster-v1; execution_plan={:?}; raster_artifact={:?}; input=vertex_layout=float32x2+unorm8x4 index_format=uint16 indices=0..6 viewport=[1,1,6,6,0,1] scissor=[2,2,4,3] target=8x8; resource=texture descriptor={descriptor:?}; expected={:?}; actual={actual:?}; first_difference={first_difference:?}; outgoing_state={outgoing_state:?}; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        case.compiled.execution_plan(),
        crate::RasterKernel::IndexedPositionColor.portable_identity(),
        case.expected,
    );
}

fn r02_vertices() -> Vec<u8> {
    let mut bytes = Vec::new();
    for (x, y) in [(-1.0_f32, -1.0_f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes.extend_from_slice(&[64, 160, 255, 255]);
    }
    bytes
}
fn r02_indices() -> Vec<u8> {
    [0_u16, 1, 2, 2, 1, 3]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect()
}
fn r02_oracle() -> Vec<u8> {
    let mut output = vec![0_u8; 8 * 8 * 4];
    for pixel in output.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0, 0, 0, 255]);
    }
    for y in 2..5 {
        for x in 2..6 {
            output[(y * 8 + x) * 4..(y * 8 + x + 1) * 4].copy_from_slice(&[64, 160, 255, 255]);
        }
    }
    output
}
