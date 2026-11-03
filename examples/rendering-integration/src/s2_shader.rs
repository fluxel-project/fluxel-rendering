//! S2 direct Shader artifact consumed by the existing raster Graph workload.

use std::{error::Error, fmt, fs, path::Path, sync::Arc};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_renderer::{Renderer, RendererError};
use fluxel_rhi::{
    api::{
        Device, Error as RhiError, RasterPipeline, ShaderDesc, ShaderEntryPoint, ShaderFormat,
        ShaderRoute, ShaderStage, VertexBufferLayout, VertexFormat, VertexStepMode,
    },
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::{
    EntryPoint, ReflectedVertexInput, ScalarType, ShaderArtifactKey, ShaderComposer,
    ShaderCompositionRequest, ShaderStage as LogicalShaderStage, SourceUnit,
    VertexInputDeclaration, VertexType,
};

use crate::{
    block_on,
    executable_cache::{ExecutableCache, ExecutableKey, RouteFingerprint},
    raster,
    shader_admission::{self, PipelineCapabilityFacts},
};

const EXTENT: [u32; 2] = [128, 128];
const SOURCE: &str = include_str!("../assets/raster.wgsl");

struct ShaderBytes;
impl AssetKind for ShaderBytes {}

/// A consumer-side rejection that occurs before RHI receives shader bytes.
#[derive(Clone, Debug)]
enum VertexLayoutPreflightError {
    MissingLocation {
        location: u32,
        shader: VertexType,
    },
    DuplicateLocation {
        location: u32,
    },
    TypeMismatch {
        location: u32,
        shader: VertexType,
        layout: VertexFormat,
    },
    UnsupportedLayoutFormat {
        location: u32,
        layout: VertexFormat,
    },
}

impl fmt::Display for VertexLayoutPreflightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingLocation { location, shader } => {
                write!(
                    formatter,
                    "vertex layout misses shader location {location} ({shader:?})"
                )
            }
            Self::DuplicateLocation { location } => {
                write!(formatter, "vertex layout duplicates location {location}")
            }
            Self::TypeMismatch {
                location,
                shader,
                layout,
            } => write!(
                formatter,
                "vertex layout location {location} has {layout:?}, shader requires {shader:?}"
            ),
            Self::UnsupportedLayoutFormat { location, layout } => write!(
                formatter,
                "vertex layout location {location} has unsupported format {layout:?}"
            ),
        }
    }
}

impl Error for VertexLayoutPreflightError {}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct S2CapabilityFacts {
    max_color_attachments: u32,
    max_vertex_buffers: u32,
    max_vertex_attributes: u32,
}

impl S2CapabilityFacts {
    fn from_device<B: fluxel_rhi::backend::Backend>(device: &Device<B>) -> Self {
        let limits = &device.capabilities().limits;
        Self {
            max_color_attachments: limits.max_color_attachments,
            max_vertex_buffers: limits.max_vertex_buffers,
            max_vertex_attributes: limits.max_vertex_attributes,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct S2VertexAttributeFacts {
    location: u32,
    offset: u64,
    format: S2VertexFormat,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum S2VertexFormat {
    Float32(u8),
    Float16(u8),
    Uint32(u8),
    Sint32(u8),
    Uint16(u8),
    Sint16(u8),
    Uint8(u8),
    Sint8(u8),
    Unorm8(u8),
    Snorm8(u8),
    Unorm16(u8),
    Snorm16(u8),
    Float64(u8),
    Uint64(u8),
}

impl From<VertexFormat> for S2VertexFormat {
    fn from(format: VertexFormat) -> Self {
        match format {
            VertexFormat::Float32(width) => Self::Float32(width),
            VertexFormat::Float16(width) => Self::Float16(width),
            VertexFormat::Uint32(width) => Self::Uint32(width),
            VertexFormat::Sint32(width) => Self::Sint32(width),
            VertexFormat::Uint16(width) => Self::Uint16(width),
            VertexFormat::Sint16(width) => Self::Sint16(width),
            VertexFormat::Uint8(width) => Self::Uint8(width),
            VertexFormat::Sint8(width) => Self::Sint8(width),
            VertexFormat::Unorm8(width) => Self::Unorm8(width),
            VertexFormat::Snorm8(width) => Self::Snorm8(width),
            VertexFormat::Unorm16(width) => Self::Unorm16(width),
            VertexFormat::Snorm16(width) => Self::Snorm16(width),
            VertexFormat::Float64(width) => Self::Float64(width),
            VertexFormat::Uint64(width) => Self::Uint64(width),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum S2VertexStepMode {
    Vertex,
    Instance,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct S2RasterPipelineFacts {
    vertex_buffers: Vec<(u64, S2VertexStepMode, Vec<S2VertexAttributeFacts>)>,
    color_target: &'static str,
    primitive: &'static str,
    multisample_count: u32,
}

impl S2RasterPipelineFacts {
    fn from_layouts(layouts: &[VertexBufferLayout]) -> Self {
        Self {
            vertex_buffers: layouts
                .iter()
                .map(|layout| {
                    (
                        layout.stride,
                        match layout.step_mode {
                            VertexStepMode::Vertex => S2VertexStepMode::Vertex,
                            VertexStepMode::Instance => S2VertexStepMode::Instance,
                        },
                        layout
                            .attributes
                            .iter()
                            .map(|attribute| S2VertexAttributeFacts {
                                location: attribute.location,
                                offset: attribute.offset,
                                format: attribute.format.into(),
                            })
                            .collect(),
                    )
                })
                .collect(),
            color_target: "Rgba8Unorm",
            primitive: "TriangleList/Ccw/None/Fill",
            multisample_count: 1,
        }
    }
}

type S2PipelineCache<B> =
    ExecutableCache<ShaderArtifactKey, S2CapabilityFacts, S2RasterPipelineFacts, RasterPipeline<B>>;

/// Long-lived S2 consumer state for one integration run.
///
/// The cache belongs to the caller rather than an individual workload
/// invocation. Its key keeps native executable products inside the selected
/// device and shader-route domain, while Renderer retains submission-owned
/// pipeline clones until completion is observed.
pub(crate) struct S2ShaderConsumer<B: fluxel_rhi::backend::Backend> {
    executable_cache: S2PipelineCache<B>,
    // Successful consumer calls into `make_pipeline_with_wgsl`; this is not a
    // backend-native allocation counter.
    consumer_pipeline_build_successes: u64,
}

impl<B: fluxel_rhi::backend::Backend> S2ShaderConsumer<B> {
    pub(crate) fn new() -> Self {
        Self {
            executable_cache: S2PipelineCache::new(),
            consumer_pipeline_build_successes: 0,
        }
    }

    /// Retires reusable products from a Device that this consumer is replacing.
    /// This is deliberately distinct from a terminal DeviceLost invalidation.
    pub(crate) fn retire_device(&mut self, device: fluxel_rhi::api::DeviceId) {
        self.executable_cache.evict_device(device);
    }

    pub(crate) fn has_device(&self, device: fluxel_rhi::api::DeviceId) -> bool {
        self.executable_cache.contains_device(device)
    }
}

/// Composes a direct Shader artifact, validates its RHI route, and renders it through Renderer.
pub(crate) fn run<B>(
    consumer: &mut S2ShaderConsumer<B>,
    device: Device<B>,
    output: &Path,
    expect_cached_pipeline: bool,
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
    fs::create_dir_all(output)?;
    fs::write(output.join("s2-shader-input.wgsl"), SOURCE)?;
    let colors = [[255, 96, 32, 255], [32, 192, 255, 255], [224, 48, 176, 255]];
    let inputs = colors.map(vertices);
    let mask: Vec<u8> = (0..128)
        .flat_map(|y| (0..128).map(move |x| u8::from(raster::stable_interior(x, y).is_some())))
        .collect();
    fs::write(output.join("s2-shader-oracle-mask.bin"), mask)?;
    for (index, input) in inputs.iter().enumerate() {
        fs::write(output.join(format!("s2-shader-input-{index}.bin")), input)?;
        fs::write(
            output.join(format!("s2-shader-expected-{index}.rgba")),
            expected_image(colors[index]),
        )?;
    }
    let request = ShaderCompositionRequest::new(vec![SourceUnit::new("s2-raster", SOURCE)])
        .with_entry_point(EntryPoint::new("vs_main", LogicalShaderStage::Vertex))
        .with_entry_point(EntryPoint::new("fs_main", LogicalShaderStage::Fragment))
        .with_vertex_input(VertexInputDeclaration::new(
            "POSITION",
            0,
            VertexType::Vec2(ScalarType::F32),
        ))
        .with_vertex_input(VertexInputDeclaration::new(
            "COLOR_0",
            1,
            VertexType::Vec4(ScalarType::F32),
        ));
    let mut composer = ShaderComposer::new();
    let artifact = composer.compose(&request)?;
    fs::write(output.join("s2-shader-artifact.wgsl"), artifact.source())?;
    fs::write(
        output.join("s2-shader-interface.txt"),
        format!("{:#?}\n", artifact.interface()),
    )?;

    let layouts = raster::vertex_buffer_layouts();
    let before_preflight = device.stats().snapshot();
    let mut incompatible_layouts = layouts.clone();
    incompatible_layouts[0].attributes[0].format = VertexFormat::Float32(3);
    if !matches!(
        preflight_vertex_layout(artifact.interface().vertex_inputs(), &incompatible_layouts),
        Err(VertexLayoutPreflightError::TypeMismatch { location: 0, .. })
    ) {
        return Err(
            "S2 vertex-layout negative fixture did not reject the actual incompatible descriptor"
                .into(),
        );
    }
    if device.stats().snapshot() != before_preflight {
        return Err("S2 vertex-layout preflight made a backend request".into());
    }
    preflight_vertex_layout(artifact.interface().vertex_inputs(), &layouts)?;
    fs::write(
        output.join("s2-shader-vertex-preflight.json"),
        "{\"schemaVersion\":1,\"negativeActualLayoutRejectedBeforeBackend\":true,\"backendStatsUnchanged\":true}\n",
    )?;
    let naga_demands = artifact
        .requirements()
        .naga_capabilities
        .iter()
        .map(|capability| capability.name().to_owned())
        .collect::<Vec<_>>();
    let rhi_feature_names = artifact
        .requirements()
        .names
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let pipeline_facts = PipelineCapabilityFacts {
        multiview_views: None,
        push_constant_bytes: 0,
        multisample_count: 1,
        nonuniform: Default::default(),
    };
    let descriptor = ShaderDesc {
        label: Some("fluxel S2 direct Shader artifact".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(artifact.source().as_bytes()),
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
        required_features: rhi_feature_names,
    };
    let admission = shader_admission::admit(&naga_demands, device.capabilities(), pipeline_facts);
    shader_admission::write_evidence(
        &output.join("shader-admission.json"),
        &naga_demands,
        &descriptor.required_features,
        &descriptor.entries,
        device.capabilities(),
        pipeline_facts,
        &admission,
    )?;
    admission?;

    let route = device.shader_route(&descriptor);
    fs::write(output.join("s2-shader-route.json"), route_json(&route))?;
    if matches!(route, ShaderRoute::Unsupported) {
        fs::write(
            output.join("s2-shader-workload.json"),
            unsupported_workload(),
        )?;
        return Ok(());
    }

    let key = ExecutableKey::new(
        artifact.cache_key().clone(),
        device.id(),
        S2CapabilityFacts::from_device(&device),
        RouteFingerprint::from_route(&route)?,
        S2RasterPipelineFacts::from_layouts(&layouts),
    );
    let consumer_pipeline_build_successes_before = consumer.consumer_pipeline_build_successes;
    let (pipeline, cache_hit) = match consumer.executable_cache.get(&key)? {
        Some(pipeline) => (pipeline.clone(), true),
        None => match raster::make_pipeline_with_wgsl(
            &device,
            fluxel_rhi::api::TextureFormat::Rgba8Unorm,
            descriptor.clone(),
            "fluxel S2 direct Shader raster pipeline",
        ) {
            Ok(pipeline) => {
                consumer.consumer_pipeline_build_successes += 1;
                consumer
                    .executable_cache
                    .insert(key.clone(), pipeline.clone())?;
                (pipeline, false)
            }
            Err(error) => {
                invalidate_cache_on_rhi_loss(
                    &mut consumer.executable_cache,
                    device.id(),
                    error.as_ref(),
                );
                return Err(error);
            }
        },
    };
    if cache_hit != expect_cached_pipeline {
        return Err(if expect_cached_pipeline {
            "S2 executable cache missed on a repeated same-device consumer invocation"
        } else {
            "S2 executable cache reused a pipeline across a new device domain"
        }
        .into());
    }
    if pipeline.device() != device.id() {
        return Err("S2 executable cache returned a pipeline from another device".into());
    }
    let store = AssetStore::<ShaderBytes, Vec<u8>, String>::new();
    let handle = store.create()?;
    let snapshots = [
        produce(&store, &handle, inputs[0].clone())?,
        replace(&store, &handle, inputs[1].clone())?,
        replace(&store, &handle, inputs[2].clone())?,
    ];
    let generations: [u64; 3] = std::array::from_fn(|index| snapshots[index].generation().get());
    if !(generations[0] < generations[1] && generations[1] < generations[2]) {
        return Err("S2 Shader asset generations did not advance".into());
    }
    let device_id = device.id();
    let mut renderer = Renderer::new(device).map_err(|error| {
        invalidate_cache_on_renderer_loss(&mut consumer.executable_cache, device_id, &error);
        Box::new(error) as Box<dyn Error>
    })?;
    let submissions = [
        submit_with_loss_invalidation(
            &mut renderer,
            snapshots[0].clone(),
            pipeline.clone(),
            &mut consumer.executable_cache,
        )?,
        submit_with_loss_invalidation(
            &mut renderer,
            snapshots[1].clone(),
            pipeline.clone(),
            &mut consumer.executable_cache,
        )?,
        submit_with_loss_invalidation(
            &mut renderer,
            snapshots[2].clone(),
            pipeline,
            &mut consumer.executable_cache,
        )?,
    ];
    for (index, submission) in submissions.iter().enumerate() {
        let readback = readback_with_loss_invalidation(
            &mut renderer,
            submission,
            device_id,
            &mut consumer.executable_cache,
        )?;
        fs::write(
            output.join(format!("s2-shader-readback-{index}.rgba")),
            &readback,
        )?;
        raster::verify_image(&readback, colors[index])?;
    }
    if renderer.pending() != 0 || renderer.compilations() != 1 {
        return Err("S2 Shader workload did not retire one reused Graph plan".into());
    }
    fs::write(
        output.join("s2-shader-workload.json"),
        format!(
            "{{\n  \"schemaVersion\": 1,\n  \"workload\": \"s2-direct-shader-raster-three-generations\",\n  \"status\": \"executed\",\n  \"deviceId\": {:?},\n  \"extent\": [128, 128],\n  \"vertexLayout\": {{\"stride\":24,\"attributes\":[{{\"semantic\":\"POSITION\",\"location\":0}},{{\"semantic\":\"COLOR_0\",\"location\":1}}]}},\n  \"consumerInvocationCacheHit\": {},\n  \"consumerInvocationExpectedCacheHit\": {},\n  \"consumerPipelineBuildSuccessesBefore\": {},\n  \"consumerPipelineBuildSuccessesThisInvocation\": {},\n  \"consumerPipelineBuildSuccessesAfter\": {},\n  \"assetGenerations\": [{}, {}, {}],\n  \"planCompilations\": 1\n}}\n",
            format!("{device_id:?}"),
            cache_hit,
            expect_cached_pipeline,
            consumer_pipeline_build_successes_before,
            consumer.consumer_pipeline_build_successes - consumer_pipeline_build_successes_before,
            consumer.consumer_pipeline_build_successes,
            generations[0],
            generations[1],
            generations[2],
        ),
    )?;
    Ok(())
}

fn invalidate_cache_on_rhi_loss<Logical, Capabilities, Pipeline, Value>(
    cache: &mut ExecutableCache<Logical, Capabilities, Pipeline, Value>,
    device: fluxel_rhi::api::DeviceId,
    error: &(dyn Error + 'static),
) where
    Logical: Eq + std::hash::Hash,
    Capabilities: Eq + std::hash::Hash,
    Pipeline: Eq + std::hash::Hash,
{
    if error.downcast_ref::<RhiError>() == Some(&RhiError::DeviceLost) {
        cache.invalidate_device(device);
    }
}

fn invalidate_cache_on_renderer_loss<B>(
    cache: &mut S2PipelineCache<B>,
    device: fluxel_rhi::api::DeviceId,
    error: &RendererError,
) where
    B: fluxel_rhi::backend::Backend,
{
    if matches!(error, RendererError::Rhi(RhiError::DeviceLost))
        || matches!(
            error,
            RendererError::RetainedExecution {
                recording_error: Some(RhiError::DeviceLost),
                ..
            } | RendererError::RetainedExecution {
                execution_error: Some(RhiError::DeviceLost),
                ..
            }
        )
    {
        cache.invalidate_device(device);
    }
}

fn submit_with_loss_invalidation<B>(
    renderer: &mut Renderer<B>,
    snapshot: AssetSnapshot<ShaderBytes, Vec<u8>>,
    pipeline: RasterPipeline<B>,
    cache: &mut S2PipelineCache<B>,
) -> Result<fluxel_renderer::Submission, Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend + RasterBackend,
{
    let device = pipeline.device();
    block_on(renderer.submit_raster(snapshot, pipeline, EXTENT)).map_err(|error| {
        invalidate_cache_on_renderer_loss(cache, device, &error);
        Box::new(error) as Box<dyn Error>
    })
}

fn readback_with_loss_invalidation<B>(
    renderer: &mut Renderer<B>,
    submission: &fluxel_renderer::Submission,
    device: fluxel_rhi::api::DeviceId,
    cache: &mut S2PipelineCache<B>,
) -> Result<Vec<u8>, Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    block_on(renderer.readback(submission)).map_err(|error| {
        invalidate_cache_on_renderer_loss(cache, device, &error);
        Box::new(error) as Box<dyn Error>
    })
}

fn preflight_vertex_layout(
    inputs: &[ReflectedVertexInput],
    layouts: &[VertexBufferLayout],
) -> Result<(), VertexLayoutPreflightError> {
    for input in inputs {
        let mut matching = layouts
            .iter()
            .flat_map(|layout| layout.attributes.iter())
            .filter(|attribute| attribute.location == input.location);
        let Some(attribute) = matching.next() else {
            return Err(VertexLayoutPreflightError::MissingLocation {
                location: input.location,
                shader: input.ty,
            });
        };
        if matching.next().is_some() {
            return Err(VertexLayoutPreflightError::DuplicateLocation {
                location: input.location,
            });
        }
        let Some(actual) = vertex_type(attribute.format) else {
            return Err(VertexLayoutPreflightError::UnsupportedLayoutFormat {
                location: input.location,
                layout: attribute.format,
            });
        };
        if actual != input.ty {
            return Err(VertexLayoutPreflightError::TypeMismatch {
                location: input.location,
                shader: input.ty,
                layout: attribute.format,
            });
        }
    }
    Ok(())
}

fn vertex_type(format: VertexFormat) -> Option<VertexType> {
    let vector = |count, scalar| match count {
        1 => Some(VertexType::Scalar(scalar)),
        2 => Some(VertexType::Vec2(scalar)),
        3 => Some(VertexType::Vec3(scalar)),
        4 => Some(VertexType::Vec4(scalar)),
        _ => None,
    };
    match format {
        VertexFormat::Float32(count) => vector(count, ScalarType::F32),
        VertexFormat::Float16(count) => vector(count, ScalarType::F16),
        VertexFormat::Sint32(count) => vector(count, ScalarType::I32),
        VertexFormat::Uint32(count) => vector(count, ScalarType::U32),
        _ => None,
    }
}

fn produce(
    store: &AssetStore<ShaderBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<ShaderBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<ShaderBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        _ => Err("unexpected S2 producer state".into()),
    }
}

fn replace(
    store: &AssetStore<ShaderBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<ShaderBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<ShaderBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.request_replacement(handle)? {
        fluxel_assets::Production::Producer(permit) => {
            Ok(permit.commit(bytes, ResidentBytes::new(size))?)
        }
        fluxel_assets::Production::Waiter(_) => Err("unexpected S2 replacement waiter".into()),
    }
}

fn vertices(color: [u8; 4]) -> Vec<u8> {
    let color = color.map(|value| f32::from(value) / 255.0);
    [(-0.75_f32, -0.65_f32), (0.55, -0.50), (-0.20, 0.75)]
        .into_iter()
        .flat_map(|(x, y)| [x, y, color[0], color[1], color[2], color[3]])
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn expected_image(color: [u8; 4]) -> Vec<u8> {
    let mut image = vec![0; 128 * 128 * 4];
    for y in 0..128 {
        for x in 0..128 {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            let vertices = [(16.0_f32, 105.6_f32), (99.2, 96.0), (51.2, 16.0)];
            let signs = (0..3)
                .map(|index| {
                    let (ax, ay) = vertices[index];
                    let (bx, by) = vertices[(index + 1) % 3];
                    (bx - ax) * (point.1 - ay) - (by - ay) * (point.0 - ax)
                })
                .collect::<Vec<_>>();
            if signs.iter().all(|sign| *sign > 0.0) || signs.iter().all(|sign| *sign < 0.0) {
                let offset = (y * 128 + x) * 4;
                image[offset..offset + 4].copy_from_slice(&color);
            }
        }
    }
    image
}

fn route_json(route: &ShaderRoute) -> String {
    match route {
        ShaderRoute::Direct(target) => format!(
            "{{\"route\":\"direct\",\"format\":{},\"environment\":{},\"abiKey\":{}}}\n",
            crate::json_string(&format!("{:?}", target.format)),
            crate::json_string(&target.environment),
            crate::json_string(&target.abi_key)
        ),
        ShaderRoute::Naga(target) => format!(
            "{{\"route\":\"naga\",\"format\":{},\"environment\":{},\"abiKey\":{}}}\n",
            crate::json_string(&format!("{:?}", target.format)),
            crate::json_string(&target.environment),
            crate::json_string(&target.abi_key)
        ),
        ShaderRoute::Unsupported => "{\"route\":\"unsupported\"}\n".into(),
    }
}

fn unsupported_workload() -> &'static str {
    "{\"schemaVersion\":1,\"workload\":\"s2-direct-shader-raster-three-generations\",\"status\":\"unsupported-before-native\"}\n"
}
