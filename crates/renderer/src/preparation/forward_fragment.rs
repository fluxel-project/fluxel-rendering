//! Renderer-owned ordinary Forward fragment entries.
//!
//! Material contributions only define evaluation helpers.  The selected
//! renderer interface names the input passed to `renderer_forward_shade`, so
//! changing a draw stage never rewrites contributor source text.

use fluxel_shader::{ShaderCompositionRequest, SourceUnit};

const ENTRY_UNIT: &str = "renderer-forward-fragment-entry";
const INTERFACE_OPTION: &str = "renderer.forward_fragment_interface";

/// Renderer-owned input ABI for a Forward material body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ForwardFragmentInterface {
    /// Unlit position-only Color surface.
    Position,
    /// Unlit UV0 Color surface.
    PositionUv0,
    /// Unlit legacy vertex-color Color surface.
    PositionUvColor,
    /// Metallic/roughness PBR without UV0.
    Pbr,
    /// Metallic/roughness PBR with UV0.
    PbrUv0,
    /// Metallic/roughness PBR with UV0 tangent-space normal.
    PbrUv0Normal,
    /// View-dependent specular PBR (with or without UV0 normal mapping).
    PbrSpecular,
}

impl ForwardFragmentInterface {
    const fn option_value(self) -> &'static str {
        match self {
            Self::Position => "position",
            Self::PositionUv0 => "position-uv0",
            Self::PositionUvColor => "position-uv-color",
            Self::Pbr => "pbr",
            Self::PbrUv0 => "pbr-uv0",
            Self::PbrUv0Normal => "pbr-uv0-normal",
            Self::PbrSpecular => "pbr-specular",
        }
    }

}

/// Adds the ordinary renderer-owned fragment entry to a body request.
pub(crate) fn with_ordinary_forward_fragment(
    mut request: ShaderCompositionRequest,
    interface: ForwardFragmentInterface,
) -> ShaderCompositionRequest {
    request
        .static_options
        .insert(INTERFACE_OPTION.into(), interface.option_value().into());
    replace_entry(&mut request, ordinary_source(interface));
    request
        .static_options
        .insert("renderer.forward_fragment_stage".into(), "ordinary".into());
    request
}

fn replace_entry(request: &mut ShaderCompositionRequest, source: String) {
    request.source_units.retain(|unit| unit.id.0 != ENTRY_UNIT);
    request
        .source_units
        .push(SourceUnit::new(ENTRY_UNIT, source));
}

fn ordinary_source(interface: ForwardFragmentInterface) -> String {
    fragment_entry(
        interface,
        "@location(0) vec4<f32>",
        "",
        "return renderer_forward_shade({arguments});",
    )
}

fn fragment_entry(
    interface: ForwardFragmentInterface,
    return_type: &str,
    declarations: &str,
    body: &str,
) -> String {
    let (parameters, arguments, position) = match interface {
        ForwardFragmentInterface::Position => {
            ("@builtin(position) position: vec4<f32>", "", "position")
        }
        ForwardFragmentInterface::PositionUv0 => {
            ("input: ForwardUv0VertexOutput", "input", "input.position")
        }
        ForwardFragmentInterface::PositionUvColor => {
            ("input: ForwardVertexOutput", "input", "input.position")
        }
        ForwardFragmentInterface::Pbr => (
            "input: PbrVertexOutput, @builtin(front_facing) front_facing: bool",
            "input, front_facing",
            "input.position",
        ),
        ForwardFragmentInterface::PbrUv0 => (
            "input: PbrUv0VertexOutput, @builtin(front_facing) front_facing: bool",
            "input, front_facing",
            "input.position",
        ),
        ForwardFragmentInterface::PbrUv0Normal => (
            "input: PbrNormalVertexOutput, @builtin(front_facing) front_facing: bool",
            "input, front_facing",
            "input.position",
        ),
        ForwardFragmentInterface::PbrSpecular => (
            "input: PbrSpecularVertexOutput, @builtin(front_facing) front_facing: bool",
            "input, front_facing",
            "input.position",
        ),
    };
    let body = body
        .replace("{arguments}", arguments)
        .replace("{position}", position);
    format!(
        r#"
{declarations}
@fragment
fn forward_fs({parameters}) -> {return_type} {{
    {body}
}}
"#,
    )
}
