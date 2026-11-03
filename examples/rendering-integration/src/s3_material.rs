//! S3 Material contribution composed into the existing read-set Graph workload.

use std::{collections::BTreeMap, error::Error, fmt, fs, path::Path, sync::Arc};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, DynamicValue, FragmentCall, MaterialAsset, MaterialCompiler,
    MaterialDefinition, MaterialFragment, MaterialInstance, MaterialParameter, MaterialResource,
    MaterialResourceKind, MaterialType, StaticValue, SurfaceOutput,
};
use fluxel_rhi::{
    api::{Device, ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderRoute, ShaderStage},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::{
    BindingKind, EntryPoint, ShaderComposer, ShaderCompositionRequest,
    ShaderStage as LogicalShaderStage, SourceUnit,
};

use crate::{
    read_set,
    shader_admission::{self, PipelineCapabilityFacts},
};

const TEMPLATE: &str = r#"
struct VertexOut { @builtin(position) position: vec4<f32>, };
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: VertexOut;
    output.position = vec4<f32>(positions[index], 0.0, 1.0);
    return output;
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return evaluate_material(); }
"#;

/// A selected RHI shader route could not execute this S3 workload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum S3RouteError {
    Unsupported { stage: &'static str },
}

impl fmt::Display for S3RouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { stage } => write!(
                formatter,
                "S3 {stage} shader route is unsupported before native creation"
            ),
        }
    }
}

impl Error for S3RouteError {}

/// Builds a Material contribution, derives every uploaded tint from its instances, and renders it.
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
    let tints = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ];
    let expected = [[17, 0, 0, 255], [0, 33, 0, 255], [0, 0, 211, 255]];
    let replacement_expected = [[241, 0, 0, 255], [0, 231, 0, 255], [0, 0, 199, 255]];
    read_set::save_reference_inputs(output, tints, expected)?;
    read_set::save_reference_inputs(&output.join("replacement"), tints, replacement_expected)?;
    fs::write(output.join("s3-material-input.wgsl"), TEMPLATE)?;
    fs::write(
        output.join("s3-material-definition.txt"),
        format!("{:#?}\n", definition(false)),
    )?;
    fs::write(
        output.join("replacement/s3-material-definition.txt"),
        format!("{:#?}\n", definition(true)),
    )?;
    let asset_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let handle = asset_store.create()?;
    let snapshot = match asset_store.acquire(&handle)? {
        // This fixture does not exercise CPU cache budgeting; zero is a logical
        // fixture charge, not a measurement of the definition's heap memory.
        Acquire::Producer(permit) => permit.commit(definition(false), ResidentBytes::new(0))?,
        _ => return Err("unexpected S3 material asset state".into()),
    };
    let variant = MaterialCompiler::compile_for(&snapshot)?;
    let replacement_snapshot = match asset_store.request_replacement(&handle)? {
        fluxel_assets::Production::Producer(permit) => {
            permit.commit(definition(true), ResidentBytes::new(0))?
        }
        _ => return Err("unexpected S3 material replacement state".into()),
    };
    if snapshot.id() != replacement_snapshot.id()
        || snapshot.generation() >= replacement_snapshot.generation()
    {
        return Err("S3 replacement did not retain identity and advance generation".into());
    }
    let replacement_variant = MaterialCompiler::compile_for(&replacement_snapshot)?;
    if variant.key() == replacement_variant.key() {
        return Err("S3 material replacement reused the prior static variant identity".into());
    }
    let instances: [MaterialInstance; 3] = [
        instance(&snapshot, &variant, tints[0])?,
        instance(&snapshot, &variant, tints[1])?,
        instance(&snapshot, &variant, tints[2])?,
    ];
    let uploaded: [[f32; 4]; 3] = [
        tint_from_instance(&instances[0], &variant)?,
        tint_from_instance(&instances[1], &variant)?,
        tint_from_instance(&instances[2], &variant)?,
    ];
    if instances
        .iter()
        .any(|instance| instance.variant() != variant.key())
    {
        return Err("S3 dynamic material instances changed the static shader variant".into());
    }
    let contribution = variant.contribution();
    fs::write(
        output.join("s3-material-contribution.wgsl"),
        &contribution.source,
    )?;
    fs::write(
        output.join("s3-material-variant.txt"),
        variant.key().as_str(),
    )?;
    let mut request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("s3-read-set-template", TEMPLATE),
        contribution.source_unit(),
    ])
    .with_entry_point(EntryPoint::new("vs_main", LogicalShaderStage::Vertex))
    .with_entry_point(EntryPoint::new("fs_main", LogicalShaderStage::Fragment));
    request.bindings = contribution.binding_declarations();
    request.callables = contribution.callable_declarations();
    request.static_options = contribution.shader_static_options();
    let mut composer = ShaderComposer::new();
    let artifact = composer.compose(&request)?;
    fs::write(output.join("s3-material-artifact.wgsl"), artifact.source())?;
    fs::write(
        output.join("s3-material-interface.txt"),
        format!("{:#?}\n", artifact.interface()),
    )?;
    let before_binding_preflight = device.stats().snapshot();
    for instance in &instances {
        read_set::preflight_material_bindings(
            artifact.interface().bindings(),
            instance.resources(),
        )?;
    }
    let mut incompatible_reflection = artifact.interface().bindings().to_vec();
    incompatible_reflection
        .iter_mut()
        .find(|binding| binding.group == 0 && binding.binding == 1)
        .ok_or("S3 texture binding was not reflected")?
        .kind = BindingKind::StorageRead;
    if !matches!(
        read_set::preflight_material_bindings(&incompatible_reflection, instances[0].resources()),
        Err(read_set::MaterialBindingPreflightError::Reflection { binding: 1 })
    ) {
        return Err("S3 incompatible reflected binding was not refused before RHI work".into());
    }
    let mut invalid_tokens = instances[0].resources().clone();
    invalid_tokens.insert("material_texture".into(), "foreign-texture-token".into());
    if !matches!(
        read_set::preflight_material_bindings(artifact.interface().bindings(), &invalid_tokens),
        Err(read_set::MaterialBindingPreflightError::Token { binding: 1, .. })
    ) {
        return Err(
            "S3 incompatible material resource token was not refused before RHI work".into(),
        );
    }
    if device.stats().snapshot() != before_binding_preflight {
        return Err("S3 material binding preflight made a backend request".into());
    }
    fs::write(
        output.join("s3-material-binding-preflight.json"),
        "{\"schemaVersion\":1,\"actualReadSetLayoutChecked\":true,\"fragmentReadOnlyAccessChecked\":true,\"incompatibleReflectionRejectedBeforeBackend\":true,\"foreignTokenRejectedBeforeBackend\":true}\n",
    )?;
    let descriptor = ShaderDesc {
        label: Some("fluxel S3 Material Shader artifact".into()),
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
        required_features: artifact.requirements().names.iter().cloned().collect(),
    };
    admit_capabilities(&artifact, &descriptor.entries, &device, output)?;
    let route = device.shader_route(&descriptor);
    fs::write(output.join("s3-material-route.json"), route_json(&route))?;
    if matches!(route, ShaderRoute::Unsupported) {
        fs::write(
            output.join("s3-material-workload.json"),
            unsupported_workload("original"),
        )?;
        return Err(S3RouteError::Unsupported { stage: "original" }.into());
    }
    read_set::run_with_shader(
        device.clone(),
        output,
        "s3-material-read-set-raster-three-generations",
        descriptor,
        uploaded,
        expected,
    )?;
    let replacement_contribution = replacement_variant.contribution();
    let replacement_output = output.join("replacement");
    fs::write(
        replacement_output.join("s3-material-contribution.wgsl"),
        &replacement_contribution.source,
    )?;
    fs::write(
        replacement_output.join("s3-material-variant.txt"),
        replacement_variant.key().as_str(),
    )?;
    let mut replacement_request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("s3-read-set-template", TEMPLATE),
        replacement_contribution.source_unit(),
    ])
    .with_entry_point(EntryPoint::new("vs_main", LogicalShaderStage::Vertex))
    .with_entry_point(EntryPoint::new("fs_main", LogicalShaderStage::Fragment));
    replacement_request.bindings = replacement_contribution.binding_declarations();
    replacement_request.callables = replacement_contribution.callable_declarations();
    replacement_request.static_options = replacement_contribution.shader_static_options();
    let replacement_artifact = composer.compose(&replacement_request)?;
    if artifact.cache_key() == replacement_artifact.cache_key() {
        return Err("S3 changed Material source reused a stale Shader artifact".into());
    }
    fs::write(
        replacement_output.join("s3-material-artifact.wgsl"),
        replacement_artifact.source(),
    )?;
    fs::write(
        replacement_output.join("s3-material-interface.txt"),
        format!("{:#?}\n", replacement_artifact.interface()),
    )?;
    let replacement_descriptor = ShaderDesc {
        label: Some("fluxel S3 replacement Material artifact".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(replacement_artifact.source().as_bytes()),
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
        required_features: replacement_artifact
            .requirements()
            .names
            .iter()
            .cloned()
            .collect(),
    };
    admit_capabilities(
        &replacement_artifact,
        &replacement_descriptor.entries,
        &device,
        &replacement_output,
    )?;
    let replacement_route = device.shader_route(&replacement_descriptor);
    fs::write(
        replacement_output.join("s3-material-route.json"),
        route_json(&replacement_route),
    )?;
    if matches!(replacement_route, ShaderRoute::Unsupported) {
        fs::write(
            replacement_output.join("s3-material-workload.json"),
            unsupported_workload("replacement"),
        )?;
        return Err(S3RouteError::Unsupported {
            stage: "replacement",
        }
        .into());
    }
    let replacement_instances = [
        instance(&replacement_snapshot, &replacement_variant, tints[0])?,
        instance(&replacement_snapshot, &replacement_variant, tints[1])?,
        instance(&replacement_snapshot, &replacement_variant, tints[2])?,
    ];
    let replacement_uploaded = [
        tint_from_instance(&replacement_instances[0], &replacement_variant)?,
        tint_from_instance(&replacement_instances[1], &replacement_variant)?,
        tint_from_instance(&replacement_instances[2], &replacement_variant)?,
    ];
    for instance in &replacement_instances {
        read_set::preflight_material_bindings(
            replacement_artifact.interface().bindings(),
            instance.resources(),
        )?;
    }
    fs::write(
        output.join("s3-material-generations.json"),
        format!(
            "{{\"original\":{},\"replacement\":{},\"sameAsset\":true,\"variantChanged\":true}}\n",
            snapshot.generation().get(),
            replacement_snapshot.generation().get()
        ),
    )?;
    read_set::run_with_shader(
        device,
        &replacement_output,
        "s3-material-replacement-read-set-raster-three-generations",
        replacement_descriptor,
        replacement_uploaded,
        replacement_expected,
    )
}

fn admit_capabilities<B>(
    artifact: &fluxel_shader::ShaderArtifact,
    target_entries: &[ShaderEntryPoint],
    device: &Device<B>,
    output: &Path,
) -> Result<(), Box<dyn Error>>
where
    B: fluxel_rhi::backend::Backend,
{
    let demands = artifact
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
    let admission = shader_admission::admit(&demands, device.capabilities(), pipeline_facts);
    shader_admission::write_evidence(
        &output.join("shader-admission.json"),
        &demands,
        &rhi_feature_names,
        target_entries,
        device.capabilities(),
        pipeline_facts,
        &admission,
    )?;
    admission?;
    Ok(())
}

fn definition(replacement: bool) -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment { name: "sample_color".into(), inputs: vec![], output: MaterialType::Vec4, calls: vec![], wgsl: "fn sample_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0); }".into() },
            MaterialFragment { name: "custom_tint".into(), inputs: vec![], output: MaterialType::Vec4, calls: vec![FragmentCall { fragment: "sample_color".into(), arguments: vec![] }], wgsl: if replacement { "fn custom_tint() -> vec4<f32> { let c = sample_color(); return vec4<f32>(c.b, c.r, c.g, c.a) * tint; }" } else { "fn custom_tint() -> vec4<f32> { return sample_color() * tint; }" }.into() },
        ],
        "custom_tint", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }],
        vec![
            MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
        ], BTreeMap::from([("sample_level".into(), StaticValue::U32(0))]), SurfaceOutput::Color,
    )
}

fn instance(
    snapshot: &fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition>,
    variant: &fluxel_material::CompiledMaterialVariant,
    tint: [f32; 4],
) -> Result<MaterialInstance, fluxel_material::MaterialError> {
    MaterialInstance::new(
        fluxel_material::MaterialAssetRef::from_snapshot(snapshot),
        variant,
        BTreeMap::from([("tint".into(), DynamicValue::Vec4(tint))]),
        BTreeMap::from([
            ("material_tint".into(), "read-set-uniform".into()),
            ("material_texture".into(), "read-set-texture".into()),
        ]),
    )
}

fn tint_from_instance(
    instance: &MaterialInstance,
    variant: &fluxel_material::CompiledMaterialVariant,
) -> Result<[f32; 4], Box<dyn Error>> {
    let bindings = &variant.contribution().parameter_bindings;
    if bindings.len() != 1 {
        return Err("S3 read-set requires one dynamic uniform parameter".into());
    }
    let binding = &bindings[0];
    if binding.location
        != (BindingLocation {
            group: 0,
            binding: 0,
        })
        || binding.ty != MaterialType::Vec4
        || instance.resource(&binding.resource) != Some("read-set-uniform")
    {
        return Err("S3 dynamic parameter does not resolve to the read-set uniform owner".into());
    }
    match instance.parameter(&binding.parameter) {
        Some(DynamicValue::Vec4(value)) => Ok(*value),
        _ => Err("S3 material instance did not expose a vec4 tint".into()),
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

fn unsupported_workload(stage: &str) -> String {
    format!(
        "{{\"schemaVersion\":1,\"stage\":{},\"route\":\"unsupported\",\"status\":\"unsupported-before-native\"}}\n",
        crate::json_string(stage),
    )
}
