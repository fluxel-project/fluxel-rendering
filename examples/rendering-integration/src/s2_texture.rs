//! Material-independent S2 textured raster Shader consumed through the read-set Graph.

use std::{error::Error, fs, path::Path, sync::Arc};

use fluxel_rhi::{
    api::{Device, ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderRoute, ShaderStage},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderComposer, ShaderCompositionRequest,
    ShaderStage as LogicalShaderStage, SourceUnit,
};

use crate::{
    read_set,
    shader_admission::{self, PipelineCapabilityFacts},
};

const SOURCE: &str = r#"
struct Tint { value: vec4<f32>, };
@group(0) @binding(0) var<uniform> s2_tint: Tint;
@group(0) @binding(1) var s2_texture: texture_2d<f32>;
struct VertexOut { @builtin(position) position: vec4<f32>, };
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: VertexOut;
    output.position = vec4<f32>(positions[index], 0.0, 1.0);
    return output;
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return textureLoad(s2_texture, vec2<i32>(0, 0), 0) * s2_tint.value;
}
"#;

/// Composes and executes a standalone texture Shader without Material involvement.
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
        [1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let expected = [[17, 0, 241, 255], [0, 33, 125, 255], [67, 199, 0, 255]];
    read_set::save_reference_inputs(output, tints, expected)?;
    fs::write(output.join("s2-texture-input.wgsl"), SOURCE)?;
    let request = ShaderCompositionRequest::new(vec![SourceUnit::new("s2-texture", SOURCE)])
        .with_entry_point(EntryPoint::new("vs_main", LogicalShaderStage::Vertex))
        .with_entry_point(EntryPoint::new("fs_main", LogicalShaderStage::Fragment))
        .with_binding(BindingDeclaration::new(0, 0, BindingKind::Uniform))
        .with_binding(BindingDeclaration::new(0, 1, BindingKind::Texture));
    let mut composer = ShaderComposer::new();
    let artifact = composer.compose(&request)?;
    fs::write(output.join("s2-texture-artifact.wgsl"), artifact.source())?;
    fs::write(
        output.join("s2-texture-interface.txt"),
        format!("{:#?}\n", artifact.interface()),
    )?;
    let naga_demands = artifact
        .requirements()
        .naga_capabilities
        .iter()
        .map(|capability| capability.name().to_owned())
        .collect::<Vec<_>>();
    let pipeline_facts = PipelineCapabilityFacts {
        multiview_views: None,
        push_constant_bytes: 0,
        multisample_count: 1,
        nonuniform: Default::default(),
    };
    let required_features = artifact
        .requirements()
        .names
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let descriptor = ShaderDesc {
        label: Some("fluxel S2 texture Shader artifact".into()),
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
        required_features,
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
    fs::write(output.join("s2-texture-route.json"), route_json(&route))?;
    if matches!(route, ShaderRoute::Unsupported) {
        fs::write(
            output.join("s2-texture-workload.json"),
            "{\"schemaVersion\":1,\"workload\":\"s2-direct-shader-texture-read-set-three-generations\",\"status\":\"unsupported-before-native\"}\n",
        )?;
        return Ok(());
    }
    read_set::run_with_shader(
        device,
        output,
        "s2-direct-shader-texture-read-set-three-generations",
        descriptor,
        tints,
        expected,
    )
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
