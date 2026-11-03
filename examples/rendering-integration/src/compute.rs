//! Fixed S1 compute consumer using only the public Renderer and RHI facades.

use std::{error::Error, fs, path::Path, sync::Arc};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_renderer::Renderer;
use fluxel_rhi::{
    api::{
        BindGroupLayout, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingCount, BindingType,
        ComputePipeline, ComputePipelineDesc, Device, DeviceCapabilities, PipelineLayoutDesc,
        PipelineStage, ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderRoute, ShaderStage,
        StorageAccess,
    },
    backend::{
        CommandBackend, ExecutionBackend, MappingBackend, PipelineBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderComposer, ShaderCompositionRequest,
    ShaderStage as LogicalShaderStage, SourceUnit,
};

use crate::{
    block_on,
    shader_admission::{self, PipelineCapabilityFacts},
};

const ELEMENTS: usize = 96;
const WORKGROUP_SIZE: usize = 64;
const WGSL: &[u8] = include_bytes!("../assets/compute.wgsl");

struct ComputeBytes;
impl AssetKind for ComputeBytes {}

fn requirements(caps: &DeviceCapabilities, compute_route: bool) -> [(&'static str, bool); 11] {
    let limits = caps.limits;
    [
        ("compute", caps.compute && compute_route),
        ("storage_buffers", caps.storage_buffers),
        (
            "buffer_bytes",
            limits.max_buffer_bytes >= (ELEMENTS * 4) as u64,
        ),
        ("bind_groups", limits.max_bind_groups >= 1),
        ("bindings_per_group", limits.max_bindings_per_group >= 1),
        (
            "workgroups_x",
            limits.max_compute_workgroups[0] >= ELEMENTS.div_ceil(WORKGROUP_SIZE) as u32,
        ),
        ("workgroups_y", limits.max_compute_workgroups[1] >= 1),
        ("workgroups_z", limits.max_compute_workgroups[2] >= 1),
        (
            "workgroup_size_x",
            limits.max_compute_workgroup_size[0] >= WORKGROUP_SIZE as u32,
        ),
        (
            "workgroup_size_y",
            limits.max_compute_workgroup_size[1] >= 1,
        ),
        (
            "workgroup_size_z",
            limits.max_compute_workgroup_size[2] >= 1,
        ),
    ]
}

#[cfg(test)]
#[path = "tests/compute_requirements.rs"]
mod tests;

/// Runs three immutable generations of the fixed in-place `u32` transform.
pub(crate) fn run<B>(device: Device<B>, output: &Path) -> Result<(), Box<dyn Error>>
where
    B: CommandBackend
        + ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + PipelineBackend
        + ShaderBackend,
{
    fs::create_dir_all(output)?;
    fs::write(output.join("compute.wgsl"), WGSL)?;
    let original = [input(0), input(1), input(2)];
    for (index, values) in original.iter().enumerate() {
        fs::write(
            output.join(format!("compute-input-{index}.bin")),
            encode(values),
        )?;
        fs::write(
            output.join(format!("compute-expected-{index}.bin")),
            encode(&expected(values)),
        )?;
    }
    let request = ShaderCompositionRequest::new(vec![SourceUnit::new(
        "s1-compute",
        String::from_utf8_lossy(WGSL),
    )])
    .with_entry_point(EntryPoint::new("cs_main", LogicalShaderStage::Compute))
    .with_binding(BindingDeclaration::new(0, 0, BindingKind::StorageReadWrite));
    let mut composer = ShaderComposer::new();
    let artifact = composer.compose(&request)?;
    fs::write(output.join("compute-composed.wgsl"), artifact.source())?;
    fs::write(
        output.join("compute-interface.txt"),
        format!("{:#?}\n", artifact.interface()),
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
        // Compute has no raster multisampling; `1` is the concrete
        // non-multisampled execution fact used by the admission layer.
        multisample_count: 1,
        nonuniform: Default::default(),
    };
    let descriptor = shader_descriptor(artifact.source(), rhi_feature_names);
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
    fs::write(output.join("compute-route.json"), route_json(&route))?;
    let checks = requirements(
        device.capabilities(),
        device.compute().is_some() && !matches!(route, ShaderRoute::Unsupported),
    );
    let supported = checks.iter().all(|(_, passed)| *passed);
    let fields = checks
        .iter()
        .map(|(name, passed)| format!("\"{name}\":{passed}"))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        output.join("compute-requirements.json"),
        format!("{{\"schemaVersion\":1,\"supported\":{supported},\"checks\":{{{fields}}}}}\n"),
    )?;
    if !supported {
        fs::write(
            output.join("compute-workload.json"),
            concat!(
                "{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"workload\": \"s1-fixed-compute-three-generations\",\n",
                "  \"status\": \"unsupported-before-submission\"\n",
                "}\n"
            ),
        )?;
        return Ok(());
    }
    let (pipeline, layout) = make_pipeline(&device, descriptor)?;
    let store = AssetStore::<ComputeBytes, Vec<u8>, String>::new();
    let handle = store.create()?;
    let first = produce(&store, &handle, encode(&original[0]))?;
    let second = replace(&store, &handle, encode(&original[1]))?;
    let third = replace(&store, &handle, encode(&original[2]))?;
    let generations = [
        first.generation().get(),
        second.generation().get(),
        third.generation().get(),
    ];
    if first.id() != second.id()
        || second.id() != third.id()
        || !(first.generation() < second.generation() && second.generation() < third.generation())
    {
        return Err("compute replacement did not preserve identity and advance generations".into());
    }
    let weak = [
        Arc::downgrade(&first.value_arc()),
        Arc::downgrade(&second.value_arc()),
        Arc::downgrade(&third.value_arc()),
    ];

    let before = device.stats().snapshot();
    let mut renderer = Renderer::new(device.clone())?;
    let submissions = [
        block_on(renderer.submit_compute(first, pipeline.clone(), layout.clone()))?,
        block_on(renderer.submit_compute(second, pipeline.clone(), layout.clone()))?,
        block_on(renderer.submit_compute(third, pipeline, layout))?,
    ];
    if renderer.pending() != 3 || renderer.compilations() != 1 {
        return Err("compute fixture did not retain three submissions behind one plan".into());
    }
    drop(handle);
    drop(store);
    if weak.iter().any(|value| value.upgrade().is_none()) {
        return Err("compute source snapshots were not retained before completion".into());
    }

    let mut rows = Vec::new();
    for (index, submission) in submissions.iter().enumerate() {
        let actual = block_on(renderer.readback(submission))?;
        fs::write(
            output.join(format!("compute-readback-{index}.bin")),
            &actual,
        )?;
        verify(&actual, &original[index])?;
        if weak[index].upgrade().is_some() {
            return Err(format!("compute snapshot {index} survived its covering readback").into());
        }
        if weak[index + 1..]
            .iter()
            .any(|value| value.upgrade().is_none())
        {
            return Err(format!(
                "a later compute snapshot retired before its own readback at frame {index}"
            )
            .into());
        }
        let original_values = original[index]
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        rows.push(format!(
            "{{\"generation\":{},\"original\":[{}]}}",
            generations[index], original_values
        ));
    }
    if renderer.pending() != 0 {
        return Err("compute submissions remained pending after all readbacks".into());
    }
    let after = device.stats().snapshot();
    fs::write(
        output.join("compute-workload.json"),
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"workload\": \"s1-fixed-compute-three-generations\",\n",
                "  \"elements\": {},\n",
                "  \"workgroupSize\": {},\n",
                "  \"operation\": \"u32.wrapping_mul(3).wrapping_add(7)\",\n",
                "  \"originalsByGeneration\": [{}],\n",
                "  \"snapshotRetentionVerified\": true,\n",
                "  \"perFrameRetirementVerified\": true,\n",
                "  \"planCompilations\": {},\n",
                "  \"rhiCounters\": {{\"dispatches\": {}, \"submits\": {}}}\n",
                "}}\n"
            ),
            ELEMENTS,
            WORKGROUP_SIZE,
            rows.join(", "),
            renderer.compilations(),
            after.dispatch.saturating_sub(before.dispatch),
            after.submits.saturating_sub(before.submits),
        ),
    )?;
    Ok(())
}

fn make_pipeline<B>(
    device: &Device<B>,
    descriptor: ShaderDesc,
) -> Result<(ComputePipeline<B>, BindGroupLayout<B>), Box<dyn Error>>
where
    B: ResourceBackend + PipelineBackend + ShaderBackend,
{
    let module = block_on(device.shader(descriptor))?;
    let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("fluxel integration compute storage".into()),
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size: 4,
                access: StorageAccess::ReadWrite,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        }],
    })?;
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: Some("fluxel integration compute layout"),
        groups: &[&layout],
        push_constants: &[],
    })?;
    let compute = device
        .compute()
        .ok_or("compute was unavailable after the pre-submission capability check")?;
    let pipeline = block_on(compute.create_pipeline(ComputePipelineDesc {
        label: Some("fluxel integration compute pipeline".into()),
        layout: pipeline_layout,
        compute: PipelineStage {
            module,
            entry: "cs_main".into(),
            constants: vec![],
        },
        required_subgroup_size: None,
        cache: None,
    }))?;
    Ok((pipeline, layout))
}

fn produce(
    store: &AssetStore<ComputeBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<ComputeBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<ComputeBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        _ => Err("unexpected non-producer initial compute asset state".into()),
    }
}

fn replace(
    store: &AssetStore<ComputeBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<ComputeBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<ComputeBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.request_replacement(handle)? {
        Production::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        Production::Waiter(_) => Err("unexpected compute replacement waiter".into()),
    }
}

fn input(generation: u32) -> [u32; ELEMENTS] {
    std::array::from_fn(|index| match index {
        0 => u32::MAX,
        1 => u32::MAX - generation,
        2 => 0,
        3 => 1,
        _ => (index as u32)
            .wrapping_mul(0x9e37_79b9)
            .wrapping_add(generation.wrapping_mul(0x7f4a_7c15)),
    })
}

fn encode(values: &[u32; ELEMENTS]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Builds the oracle input before submission, independently from readback.
fn expected(values: &[u32; ELEMENTS]) -> [u32; ELEMENTS] {
    values.map(|value| value.wrapping_mul(3).wrapping_add(7))
}

fn verify(bytes: &[u8], original: &[u32; ELEMENTS]) -> Result<(), Box<dyn Error>> {
    if bytes.len() != ELEMENTS * std::mem::size_of::<u32>() {
        return Err(format!("compute readback had {} bytes", bytes.len()).into());
    }
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let actual = u32::from_le_bytes(chunk.try_into()?);
        let expected = expected(original)[index];
        if actual != expected {
            return Err(
                format!("compute element {index} was {actual}, expected {expected}").into(),
            );
        }
    }
    Ok(())
}

fn shader_descriptor(source: &str, required_features: Vec<String>) -> ShaderDesc {
    ShaderDesc {
        label: Some("fluxel integration composed compute WGSL".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_bytes()),
        entries: vec![ShaderEntryPoint {
            name: "cs_main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([WORKGROUP_SIZE as u32, 1, 1]),
        }],
        required_features,
    }
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
