//! Fixed S1 raster consumer using the public Renderer facade.
//!
//! This is deliberately a fixture: it proves a vertex snapshot, fixed WGSL,
//! offscreen target, and readback through the private Renderer Graph. It does
//! not define a general Renderer pipeline or Shader API.

use std::{error::Error, fs, path::Path, sync::Arc};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_renderer::{Renderer, RendererError};
use fluxel_rhi::api::Error as RhiError;
use fluxel_rhi::{
    api::{
        ColorTargetState, CullMode, Device, PipelineLayoutDesc, PipelineStage, PolygonMode,
        PrimitiveState, PrimitiveTopology, RasterPipeline, RasterPipelineDesc, ShaderDesc,
        ShaderEntryPoint, ShaderFormat, ShaderStage, TextureFormat, VertexAttribute,
        VertexBufferLayout, VertexFormat, VertexStepMode,
    },
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use crate::block_on;

const EXTENT: [u32; 2] = [128, 128];
const VERTEX_STRIDE: usize = 24;
const VERTEX_COUNT: usize = 3;
const WGSL: &[u8] = include_bytes!("../assets/raster.wgsl");

struct RasterBytes;
impl AssetKind for RasterBytes {}

/// Runs three fixed color raster generations before the first completion wait.
pub(crate) fn run<B>(device: Device<B>, output: &Path) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    fs::create_dir_all(output)?;
    fs::write(output.join("raster.wgsl"), WGSL)?;
    let pipeline = make_pipeline(&device, TextureFormat::Rgba8Unorm)?;

    let store = AssetStore::<RasterBytes, Vec<u8>, String>::new();
    let handle = store.create()?;
    let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    fs::write(
        output.join("raster-expected.json"),
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"extent\": [{}, {}],\n",
                "  \"clearRgba\": [0, 0, 0, 0],\n",
                "  \"triangleColors\": [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]],\n",
                "  \"edgeBandPixelsExcluded\": 2\n",
                "}}\n"
            ),
            EXTENT[0],
            EXTENT[1],
            colors[0][0],
            colors[0][1],
            colors[0][2],
            colors[0][3],
            colors[1][0],
            colors[1][1],
            colors[1][2],
            colors[1][3],
            colors[2][0],
            colors[2][1],
            colors[2][2],
            colors[2][3],
        ),
    )?;
    let first = produce(&store, &handle, vertices(colors[0]))?;
    let second = replace(&store, &handle, vertices(colors[1]))?;
    let third = replace(&store, &handle, vertices(colors[2]))?;
    let generations = [
        first.generation().get(),
        second.generation().get(),
        third.generation().get(),
    ];
    if first.id() != second.id()
        || second.id() != third.id()
        || !(first.generation() < second.generation() && second.generation() < third.generation())
    {
        return Err("raster replacement did not preserve identity and advance generations".into());
    }
    for (index, color) in colors.iter().copied().enumerate() {
        fs::write(
            output.join(format!("raster-input-{index}.bin")),
            vertices(color),
        )?;
    }
    let weak = [
        Arc::downgrade(&first.value_arc()),
        Arc::downgrade(&second.value_arc()),
        Arc::downgrade(&third.value_arc()),
    ];

    let mut renderer = Renderer::new(device)?;
    let submissions = [
        block_on(renderer.submit_raster(first, pipeline.clone(), EXTENT))?,
        block_on(renderer.submit_raster(second, pipeline.clone(), EXTENT))?,
        block_on(renderer.submit_raster(third, pipeline, EXTENT))?,
    ];
    if renderer.pending() != 3 {
        return Err("raster fixture did not retain three in-flight submissions".into());
    }
    if renderer.compilations() != 1 {
        return Err("raster fixture did not reuse one compiled plan".into());
    }
    drop(handle);
    drop(store);
    if weak.iter().any(|value| value.upgrade().is_none()) {
        return Err("raster source snapshots were not retained before completion".into());
    }

    let mut oracle_rows = Vec::new();
    for (index, submission) in submissions.iter().enumerate() {
        let bytes = block_on(renderer.readback(submission))?;
        fs::write(output.join(format!("raster-readback-{index}.rgba")), &bytes)?;
        fs::write(
            output.join(format!("raster-readback-{index}.ppm")),
            ppm(&bytes)?,
        )?;
        verify_image(&bytes, colors[index])?;
        if weak[index].upgrade().is_some() {
            return Err(format!("raster snapshot {index} survived its covering readback").into());
        }
        if weak[index + 1..]
            .iter()
            .any(|value| value.upgrade().is_none())
        {
            return Err(format!(
                "a later raster snapshot retired before its own readback at frame {index}"
            )
            .into());
        }
        oracle_rows.push(format!(
            "{{\"generation\":{},\"rgba\":[{},{},{},{}]}}",
            generations[index],
            colors[index][0],
            colors[index][1],
            colors[index][2],
            colors[index][3]
        ));
    }
    if renderer.pending() != 0 {
        return Err("raster submissions remained pending after all readbacks".into());
    }
    fs::write(
        output.join("raster-workload.json"),
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"workload\": \"s1-fixed-raster-three-generations\",\n",
                "  \"extent\": [{}, {}],\n",
                "  \"vertexStride\": {},\n",
                "  \"vertexCount\": {},\n",
                "  \"clearRgba\": [0, 0, 0, 0],\n",
                "  \"generations\": [{}],\n",
                "  \"snapshotRetentionVerified\": true,\n",
                "  \"perFrameRetirementVerified\": true,\n",
                "  \"planCompilations\": {}\n",
                "}}\n"
            ),
            EXTENT[0],
            EXTENT[1],
            VERTEX_STRIDE,
            VERTEX_COUNT,
            oracle_rows.join(", "),
            renderer.compilations(),
        ),
    )?;
    Ok(())
}

/// A replacement device must reject executable products from the old device
/// before it allocates or records backend work. The logical asset snapshot
/// remains valid CPU data and stays alive through that refusal.
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
pub(crate) fn replacement_rejects_old_pipeline<B>(
    old_pipeline: RasterPipeline<B>,
    replacement: Device<B>,
    old_submission: &fluxel_renderer::Submission,
    output: &Path,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let old_device_id = old_pipeline.device();
    let replacement_device_id = replacement.id();
    if old_device_id == replacement_device_id {
        return Err("replacement fixture did not create a distinct RHI device".into());
    }
    let store = AssetStore::<RasterBytes, Vec<u8>, String>::new();
    let handle = store.create()?;
    let expected = [255, 255, 0, 255];
    let source = vertices(expected);
    fs::write(output.join("device-replacement-input.bin"), &source)?;
    fs::write(
        output.join("device-replacement-expected.json"),
        "{\"rgba\":[255,255,0,255],\"extent\":[128,128]}\n",
    )?;
    let snapshot = produce(&store, &handle, source)?;
    let weak = Arc::downgrade(&snapshot.value_arc());
    let before = replacement.stats().snapshot();
    let mut renderer = Renderer::new(replacement.clone())?;
    if !matches!(
        block_on(renderer.submit_raster(snapshot.clone(), old_pipeline, EXTENT)),
        Err(RendererError::Rhi(RhiError::WrongDevice))
    ) {
        return Err("replacement device accepted an old-device raster pipeline".into());
    }
    if replacement.stats().snapshot() != before {
        return Err(
            "old-device pipeline rejection made backend requests on replacement device".into(),
        );
    }
    if weak.upgrade().is_none() {
        return Err("logical snapshot did not survive old-device refusal".into());
    }
    if !matches!(
        block_on(renderer.readback(old_submission)),
        Err(RendererError::ForeignSubmission)
    ) || replacement.stats().snapshot() != before
    {
        return Err(
            "replacement device did not reject the old submission before a backend wait".into(),
        );
    }
    drop(handle);
    drop(store);
    let replacement_pipeline = make_pipeline(&replacement, TextureFormat::Rgba8Unorm)?;
    let submission = block_on(renderer.submit_raster(snapshot, replacement_pipeline, EXTENT))?;
    let bytes = block_on(renderer.readback(&submission))?;
    fs::write(output.join("device-replacement-readback.rgba"), &bytes)?;
    fs::write(output.join("device-replacement-readback.ppm"), ppm(&bytes)?)?;
    verify_image(&bytes, expected)?;
    if renderer.pending() != 0 {
        return Err("replacement-device raster submission remained pending after readback".into());
    }
    if weak.upgrade().is_some() {
        return Err("logical snapshot survived its replacement-device covering readback".into());
    }
    fs::write(
        output.join("device-replacement.json"),
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"oldDeviceId\": {:?},\n",
                "  \"replacementDeviceId\": {:?},\n",
                "  \"foreignPipelineSubmissionRejectedBeforeBackendWork\": true,\n",
                "  \"rejectionBackendStatsUnchanged\": true,\n",
                "  \"foreignSubmissionRejectedBeforeBackendWait\": true,\n",
                "  \"logicalSnapshotRetainedThroughRefusal\": true,\n",
                "  \"replacementPipelineSubmissionReadbackVerified\": true,\n",
                "  \"input\": \"device-replacement-input.bin\",\n",
                "  \"expected\": \"device-replacement-expected.json\",\n",
                "  \"readback\": \"device-replacement-readback.rgba\",\n",
                "  \"readbackBytes\": {},\n",
                "  \"readbackFnv1a64\": \"{:016x}\"\n",
                "}}\n"
            ),
            format!("{old_device_id:?}"),
            format!("{replacement_device_id:?}"),
            bytes.len(),
            fnv1a64(&bytes),
        ),
    )?;
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn make_pipeline<B>(
    device: &Device<B>,
    target_format: TextureFormat,
) -> Result<RasterPipeline<B>, Box<dyn Error>>
where
    B: ResourceBackend + PipelineBackend + ShaderBackend,
{
    let module = block_on(device.shader(ShaderDesc {
        label: Some("fluxel integration raster WGSL".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(WGSL),
        entries: vec![
            ShaderEntryPoint {
                name: "vs_main".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            },
            ShaderEntryPoint {
                name: "fs_main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            },
        ],
        required_features: vec![],
    }))?;
    let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: Some("fluxel integration raster layout"),
        groups: &[],
        push_constants: &[],
    })?;
    Ok(block_on(device.raster_pipeline(RasterPipelineDesc {
        label: Some("fluxel integration raster pipeline".into()),
        layout,
        vertex: PipelineStage {
            module: module.clone(),
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module,
            entry: "fs_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: VERTEX_STRIDE as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32(2),
                },
                VertexAttribute {
                    location: 1,
                    offset: 8,
                    format: VertexFormat::Float32(4),
                },
            ],
        }],
        color_targets: vec![Some(ColorTargetState {
            format: target_format,
            blend: None,
            write_mask: 0x0f,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: fluxel_rhi::api::FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: fluxel_rhi::api::MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    }))?)
}

fn produce(
    store: &AssetStore<RasterBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<RasterBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<RasterBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        _ => Err("unexpected non-producer initial raster asset state".into()),
    }
}

fn replace(
    store: &AssetStore<RasterBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<RasterBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<RasterBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.request_replacement(handle)? {
        Production::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        Production::Waiter(_) => Err("unexpected raster replacement waiter".into()),
    }
}

pub(crate) fn vertices(color: [u8; 4]) -> Vec<u8> {
    let rgba = color.map(|component| f32::from(component) / 255.0);
    let positions = [[-0.75_f32, -0.65], [0.55, -0.50], [-0.20, 0.75]];
    let mut bytes = Vec::with_capacity(VERTEX_STRIDE * VERTEX_COUNT);
    for position in positions {
        for value in position.into_iter().chain(rgba) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn verify_image(bytes: &[u8], color: [u8; 4]) -> Result<(), Box<dyn Error>> {
    let expected_len = usize::try_from(EXTENT[0])? * usize::try_from(EXTENT[1])? * 4;
    if bytes.len() != expected_len {
        return Err(format!(
            "raster readback had {} bytes, expected {expected_len}",
            bytes.len()
        )
        .into());
    }
    let mut checked = 0_u32;
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let Some(inside) = stable_interior(x, y) else {
                continue;
            };
            let offset =
                (usize::try_from(y)? * usize::try_from(EXTENT[0])? + usize::try_from(x)?) * 4;
            let actual = [
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ];
            let expected = if inside { color } else { [0, 0, 0, 0] };
            if actual != expected {
                return Err(format!(
                    "raster pixel ({x}, {y}) was {actual:?}, expected {expected:?}"
                )
                .into());
            }
            checked += 1;
        }
    }
    if checked < 1_000 {
        return Err("raster oracle had too few edge-independent samples".into());
    }
    Ok(())
}

/// Returns `None` for a two-pixel edge band where rasterization tie rules vary.
fn stable_interior(x: u32, y: u32) -> Option<bool> {
    let vertices = [[16.0_f32, 105.6], [99.2, 96.0], [51.2, 16.0]];
    let point = [x as f32 + 0.5, y as f32 + 0.5];
    let mut signs = [0.0_f32; 3];
    for index in 0..3 {
        let a = vertices[index];
        let b = vertices[(index + 1) % 3];
        let edge_x = b[0] - a[0];
        let edge_y = b[1] - a[1];
        signs[index] = edge_x * (point[1] - a[1]) - edge_y * (point[0] - a[0]);
        let distance = signs[index].abs() / (edge_x * edge_x + edge_y * edge_y).sqrt();
        if distance <= 2.0 {
            return None;
        }
    }
    Some(signs.iter().all(|sign| *sign > 0.0) || signs.iter().all(|sign| *sign < 0.0))
}

fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let expected_len = usize::try_from(EXTENT[0])? * usize::try_from(EXTENT[1])? * 4;
    if rgba.len() != expected_len {
        return Err("cannot write PPM from malformed raster readback".into());
    }
    let mut result = format!("P6\n{} {}\n255\n", EXTENT[0], EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        result.extend_from_slice(&pixel[..3]);
    }
    Ok(result)
}
