//! A fixed material-like read-set fixture consumed through Renderer Graph.
//!
//! The fixture binds one dynamic uniform range and one sampled texture. It is
//! deliberately private integration evidence, not a Material or Shader API.

use std::{
    collections::BTreeMap, error::Error, fmt, fs, ops::Range, path::Path, sync::Arc, time::Instant,
};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_renderer::Renderer;
use fluxel_rhi::{
    api::{
        BindGroupLayout, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingCount, BindingType,
        ColorTargetState, CullMode, Device, PipelineLayoutDesc, PipelineStage, PolygonMode,
        PrimitiveState, PrimitiveTopology, RasterPipeline, RasterPipelineDesc, ShaderDesc,
        ShaderEntryPoint, ShaderFormat, ShaderStage, TextureFormat, TextureSampleType,
        TextureViewDimension,
    },
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use crate::block_on;
use fluxel_shader::{BindingKind, ReflectedBinding, ShaderStage as LogicalShaderStage};

const EXTENT: [u32; 2] = [8, 8];
const UNIFORM_BYTES: u64 = 16;
const FRAME_RANGES: [Range<u64>; 3] = [0..256, 256..512, 512..768];
const WGSL: &[u8] = include_bytes!("../assets/read-set.wgsl");

struct ReadSetBytes;
impl AssetKind for ReadSetBytes {}

/// A structured mismatch between Shader reflection, a material resource token, and the fixed read-set protocol.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum MaterialBindingPreflightError {
    /// Reflection omitted a required fixed binding or described it incompatibly.
    Reflection { binding: u32 },
    /// A material instance did not resolve a required resource to this consumer's actual token.
    Token {
        binding: u32,
        expected: &'static str,
        actual: Option<String>,
    },
}

impl fmt::Display for MaterialBindingPreflightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl Error for MaterialBindingPreflightError {}

/// Checks reflection and material resource tokens against the concrete read-set binding contract before RHI layout creation.
pub(crate) fn preflight_material_bindings(
    reflected: &[ReflectedBinding],
    resources: &BTreeMap<String, String>,
) -> Result<(), MaterialBindingPreflightError> {
    for (binding, kind, token) in [
        (0, BindingKind::Uniform, "read-set-uniform"),
        (1, BindingKind::Texture, "read-set-texture"),
    ] {
        let reflection_matches = reflected.iter().any(|value| {
            value.group == 0
                && value.binding == binding
                && value.kind == kind
                && value.visibility.len() == 1
                && value.visibility.contains(&LogicalShaderStage::Fragment)
                && value.read
                && !value.write
                && !value.query
                && !value.atomic
        });
        if !reflection_matches {
            return Err(MaterialBindingPreflightError::Reflection { binding });
        }
        let name = if binding == 0 {
            "material_tint"
        } else {
            "material_texture"
        };
        let actual = resources.get(name).cloned();
        if actual.as_deref() != Some(token) {
            return Err(MaterialBindingPreflightError::Token {
                binding,
                expected: token,
                actual,
            });
        }
    }
    if reflected.len() != 2 {
        return Err(MaterialBindingPreflightError::Reflection { binding: 2 });
    }
    Ok(())
}

/// Executes three source generations through the fixed Graph read-set path.
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
    run_with_shader(
        device,
        output,
        "s1-private-read-set-raster-three-generations",
        ShaderDesc {
            label: Some("fluxel integration read-set WGSL".into()),
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
            required_features: Vec::new(),
        },
        [
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ],
        [[17, 129, 0, 255], [0, 33, 125, 255], [67, 0, 211, 255]],
    )
}

/// Executes the fixed read-set Graph workload with the routed shader descriptor.
pub(crate) fn run_with_shader<B>(
    device: Device<B>,
    output: &Path,
    workload: &str,
    descriptor: ShaderDesc,
    tints: [[f32; 4]; 3],
    expected: [[u8; 4]; 3],
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
    fs::write(output.join("read-set.wgsl"), &descriptor.bytes)?;
    let inputs = save_reference_inputs(output, tints, expected)?;
    let (pipeline, layout) = make_pipeline(&device, descriptor)?;

    let store = AssetStore::<ReadSetBytes, Vec<u8>, String>::new();
    let handle = store.create()?;
    let first = produce(&store, &handle, inputs[0].clone())?;
    let second = replace(&store, &handle, inputs[1].clone())?;
    let third = replace(&store, &handle, inputs[2].clone())?;
    let generations = [
        first.generation().get(),
        second.generation().get(),
        third.generation().get(),
    ];
    if first.id() != second.id()
        || second.id() != third.id()
        || !(first.generation() < second.generation() && second.generation() < third.generation())
    {
        return Err(
            "read-set replacement did not preserve identity and advance generations".into(),
        );
    }
    let weak = [
        Arc::downgrade(&first.value_arc()),
        Arc::downgrade(&second.value_arc()),
        Arc::downgrade(&third.value_arc()),
    ];

    let mut renderer = Renderer::new(device.clone())?;
    let invalid_input = inputs[0][..19].to_vec();
    fs::write(output.join("read-set-invalid-payload.bin"), &invalid_input)?;
    let invalid_handle = store.create()?;
    let invalid_snapshot = produce(&store, &invalid_handle, invalid_input)?;
    drop(invalid_handle);
    let mut refusal_rows = Vec::new();
    for (name, source, range) in [
        ("unaligned-range", first.clone(), 1..257),
        ("short-range", first.clone(), 0..255),
        ("short-payload", invalid_snapshot, 0..256),
    ] {
        let before = device.stats().snapshot();
        let result = block_on(renderer.submit_read_set_raster(
            source,
            pipeline.clone(),
            layout.clone(),
            range.clone(),
        ));
        let unchanged = before == device.stats().snapshot();
        refusal_rows.push(format!(
            "{{\"case\":{},\"range\":[{},{}],\"result\":{},\"backendStatsUnchanged\":{}}}",
            crate::json_string(name),
            range.start,
            range.end,
            crate::json_string(&format!("{result:?}")),
            unchanged,
        ));
        fs::write(
            output.join("read-set-refusals.jsonl"),
            refusal_rows.join("\n") + "\n",
        )?;
        if !matches!(
            result,
            Err(fluxel_renderer::RendererError::Rhi(
                fluxel_rhi::api::Error::InvalidRange
            ))
        ) || !unchanged
            || renderer.pending() != 0
            || renderer.compilations() != 0
        {
            return Err(format!(
                "read-set {name} did not fail before native work and cache selection"
            )
            .into());
        }
    }
    let started = Instant::now();
    let first_submission = block_on(renderer.submit_read_set_raster(
        first,
        pipeline.clone(),
        layout.clone(),
        FRAME_RANGES[0].clone(),
    ))?;
    let plan = renderer
        .plan_id()
        .ok_or("read-set plan identity was not retained")?;
    let submissions = [
        first_submission,
        block_on(renderer.submit_read_set_raster(
            second,
            pipeline.clone(),
            layout.clone(),
            FRAME_RANGES[1].clone(),
        ))?,
        block_on(renderer.submit_read_set_raster(
            third,
            pipeline,
            layout,
            FRAME_RANGES[2].clone(),
        ))?,
    ];
    let submission_elapsed = started.elapsed();
    if renderer.pending() != 3
        || renderer.compilations() != 1
        || renderer.plan_id() != Some(plan)
        || renderer.binding_refreshes() != 3
    {
        return Err("read-set fixture did not retain three frames behind one fresh plan".into());
    }
    drop(handle);
    drop(store);
    if weak.iter().any(|snapshot| snapshot.upgrade().is_none()) {
        return Err("read-set snapshots retired before covering completion".into());
    }

    let readback_started = Instant::now();
    for (index, submission) in submissions.iter().enumerate() {
        let actual = block_on(renderer.readback(submission))?;
        fs::write(
            output.join(format!("read-set-readback-{index}.rgba")),
            &actual,
        )?;
        verify_image(&actual, expected[index])?;
        if weak[index].upgrade().is_some() {
            return Err(format!("read-set snapshot {index} survived its covering readback").into());
        }
        if weak[index + 1..]
            .iter()
            .any(|snapshot| snapshot.upgrade().is_none())
        {
            return Err(format!("later read-set snapshot retired before frame {index}").into());
        }
    }
    let readback_elapsed = readback_started.elapsed();
    if renderer.pending() != 0 {
        return Err("read-set owners survived all readbacks".into());
    }
    fs::write(
        output.join("read-set-workload.json"),
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"workload\": {},\n",
                "  \"extent\": [{}, {}],\n",
                "  \"assetGenerations\": [{}, {}, {}],\n",
                "  \"uniformDestinationRanges\": [[{}, {}], [{}, {}], [{}, {}]],\n",
                "  \"sourceByteLengths\": [{}, {}, {}],\n",
                "  \"plan\": {{\"device\": \"{:?}\", \"ordinal\": {}}},\n",
                "  \"planReuseVerified\": true,\n",
                "  \"snapshotRetentionVerified\": true,\n",
                "  \"perFrameRetirementVerified\": true,\n",
                "  \"timingsNanoseconds\": {{\"submitThree\": {}, \"readbackThree\": {}}}\n",
                "}}\n"
            ),
            crate::json_string(workload),
            EXTENT[0],
            EXTENT[1],
            generations[0],
            generations[1],
            generations[2],
            FRAME_RANGES[0].start,
            FRAME_RANGES[0].end,
            FRAME_RANGES[1].start,
            FRAME_RANGES[1].end,
            FRAME_RANGES[2].start,
            FRAME_RANGES[2].end,
            inputs[0].len(),
            inputs[1].len(),
            inputs[2].len(),
            plan.device,
            plan.ordinal,
            submission_elapsed.as_nanos(),
            readback_elapsed.as_nanos(),
        ),
    )?;
    Ok(())
}

/// Retains independent source inputs and expectations before shader/backend work.
pub(crate) fn save_reference_inputs(
    output: &Path,
    tints: [[f32; 4]; 3],
    expected: [[u8; 4]; 3],
) -> Result<[Vec<u8>; 3], Box<dyn Error>> {
    fs::create_dir_all(output)?;
    let texels = [
        [17, 129, 241, 255],
        [231, 33, 125, 255],
        [67, 199, 211, 255],
    ];
    let inputs = std::array::from_fn(|index| input(tints[index], texels[index]));
    for index in 0..inputs.len() {
        fs::write(
            output.join(format!("read-set-input-{index}.bin")),
            &inputs[index],
        )?;
        fs::write(
            output.join(format!("read-set-expected-{index}.rgba")),
            expected_image(expected[index]),
        )?;
    }
    Ok(inputs)
}

fn make_pipeline<B>(
    device: &Device<B>,
    descriptor: ShaderDesc,
) -> Result<(RasterPipeline<B>, BindGroupLayout<B>), Box<dyn Error>>
where
    B: ResourceBackend + PipelineBackend + ShaderBackend,
{
    let module = block_on(device.shader(descriptor))?;
    let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("fluxel integration read-set bindings".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: true,
                    min_size: UNIFORM_BYTES,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            },
        ],
    })?;
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: Some("fluxel integration read-set pipeline layout"),
        groups: &[&layout],
        push_constants: &[],
    })?;
    let pipeline = block_on(device.raster_pipeline(RasterPipelineDesc {
        label: Some("fluxel integration read-set raster pipeline".into()),
        layout: pipeline_layout,
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
        vertex_buffers: vec![],
        color_targets: vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
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
    }))?;
    Ok((pipeline, layout))
}

fn input(tint: [f32; 4], texel: [u8; 4]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(20);
    for value in tint {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&texel);
    bytes
}

fn expected_image(color: [u8; 4]) -> Vec<u8> {
    let count = usize::try_from(EXTENT[0]).unwrap() * usize::try_from(EXTENT[1]).unwrap();
    color.into_iter().cycle().take(count * 4).collect()
}

fn verify_image(actual: &[u8], color: [u8; 4]) -> Result<(), Box<dyn Error>> {
    let expected = expected_image(color);
    if actual != expected {
        return Err(format!(
            "read-set raster differed from its independent expected RGBA image; got {} bytes",
            actual.len()
        )
        .into());
    }
    Ok(())
}

fn produce(
    store: &AssetStore<ReadSetBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<ReadSetBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<ReadSetBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        _ => Err("unexpected non-producer initial read-set asset state".into()),
    }
}

fn replace(
    store: &AssetStore<ReadSetBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<ReadSetBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<ReadSetBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.request_replacement(handle)? {
        Production::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        Production::Waiter(_) => Err("unexpected read-set replacement waiter".into()),
    }
}
