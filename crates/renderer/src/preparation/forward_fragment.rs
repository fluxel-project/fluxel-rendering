//! Renderer-owned Forward fragment entries shared by ordinary and OIT draws.
//!
//! Material contributions only define evaluation helpers.  The selected
//! renderer interface names the input passed to `renderer_forward_shade`, so
//! changing a draw stage never rewrites contributor source text.

use fluxel_shader::{
    BindingDeclaration, BindingKind, BindingShape, ShaderCompositionRequest, ShaderStage,
    SourceUnit,
};

const ENTRY_UNIT: &str = "renderer-forward-fragment-entry";
const INTERFACE_OPTION: &str = "renderer.forward_fragment_interface";

/// The logical OIT material draw stage.  The Graph owns iteration count; this
/// enum deliberately has no Graph or physical-owner dependency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OitDrawStage {
    /// Establish per-pixel nearest/furthest depth extrema.
    InitialDepth,
    /// Render one dual-depth-peeling iteration.
    Peel,
}

/// Actual min/max target representation admitted from device facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OitDepthPrecision {
    /// `Rg32Float` extrema retain the raster depth directly.
    Rg32Float,
    /// `Rg16Float` extrema use Babylon-compatible half precision depth.
    Rg16Float,
}

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

    fn from_request(request: &ShaderCompositionRequest) -> Option<Self> {
        match request.static_options.get(INTERFACE_OPTION)?.as_str() {
            "position" => Some(Self::Position),
            "position-uv0" => Some(Self::PositionUv0),
            "position-uv-color" => Some(Self::PositionUvColor),
            "pbr" => Some(Self::Pbr),
            "pbr-uv0" => Some(Self::PbrUv0),
            "pbr-uv0-normal" => Some(Self::PbrUv0Normal),
            "pbr-specular" => Some(Self::PbrSpecular),
            _ => None,
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

/// Replaces the ordinary entry with a stage-specific OIT entry while retaining
/// all material, camera, environment, transform, and vertex contracts.
///
/// Call this before `prepare_material`: both stage and admitted precision then
/// participate in the composed artifact identity and reflection.
pub(crate) fn with_oit_forward_fragment(
    mut request: ShaderCompositionRequest,
    stage: OitDrawStage,
    precision: OitDepthPrecision,
    two_sided_lighting: bool,
) -> Option<ShaderCompositionRequest> {
    let interface = ForwardFragmentInterface::from_request(&request)?;
    if matches!(
        interface,
        ForwardFragmentInterface::Pbr
            | ForwardFragmentInterface::PbrUv0
            | ForwardFragmentInterface::PbrUv0Normal
            | ForwardFragmentInterface::PbrSpecular
    ) {
        // OIT forces culling off. Babylon rebuilds its two-sided-lighting
        // define after that change, independently of authored culling.
        let normal = request
            .source_units
            .iter_mut()
            .find(|unit| unit.id.0 == "pbr-surface-normal")?;
        *normal = SourceUnit::new(
            "pbr-surface-normal",
            format!(
                "fn pbr_surface_normal(normal: vec3<f32>, front_facing: bool) -> vec3<f32> {{\n    let normalized = pbr_normal(normal);\n    if {two_sided_lighting} && !front_facing {{ return -normalized; }}\n    return normalized;\n}}"
            ),
        );
    }
    request.static_options.insert(
        "renderer.oit_two_sided_lighting".into(),
        two_sided_lighting.to_string(),
    );
    let source = match stage {
        OitDrawStage::InitialDepth => initial_depth_source(interface, precision),
        OitDrawStage::Peel => peel_source(interface, precision),
    };
    replace_entry(&mut request, source);
    request.static_options.insert(
        "renderer.forward_fragment_stage".into(),
        match stage {
            OitDrawStage::InitialDepth => "oit-initial-depth",
            OitDrawStage::Peel => "oit-peel",
        }
        .into(),
    );
    request.static_options.insert(
        "renderer.oit_depth_precision".into(),
        match precision {
            OitDepthPrecision::Rg32Float => "rg32float",
            OitDepthPrecision::Rg16Float => "rg16float",
        }
        .into(),
    );
    if matches!(stage, OitDrawStage::InitialDepth | OitDrawStage::Peel) {
        request = request.with_binding(
            BindingDeclaration::new(3, 0, BindingKind::Texture)
                .with_shape(BindingShape::Texture2dF32)
                .with_visibility([ShaderStage::Fragment]),
        );
    }
    if matches!(stage, OitDrawStage::Peel) {
        request = request.with_binding(
            BindingDeclaration::new(3, 1, BindingKind::Texture)
                .with_shape(BindingShape::Texture2dF32)
                .with_visibility([ShaderStage::Fragment]),
        );
    }
    Some(request)
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

fn initial_depth_source(
    interface: ForwardFragmentInterface,
    precision: OitDepthPrecision,
) -> String {
    let depth_rounding = match precision {
        OitDepthPrecision::Rg32Float => "",
        OitDepthPrecision::Rg16Float => {
            "depth = unpack2x16float(pack2x16float(vec2<f32>(depth))).x;"
        }
    };
    fragment_entry(
        interface,
        "@location(0) vec2<f32>",
        "@group(3) @binding(0) var renderer_oit_min_max: texture_2d<f32>;",
        &format!(
            "let final_color = renderer_forward_shade({{arguments}});\n    var depth = {{position}}.z;\n    {depth_rounding}\n    let coordinate = vec2<i32>({{position}}.xy);\n    let last_depth = textureLoad(renderer_oit_min_max, coordinate, 0);\n    let nearest_depth = -last_depth.x;\n    let furthest_depth = last_depth.y;\n    if depth < nearest_depth || depth > furthest_depth {{ return vec2<f32>(-99999.0); }}\n    return vec2<f32>(-depth, depth);",
        ),
    )
}

fn peel_source(interface: ForwardFragmentInterface, precision: OitDepthPrecision) -> String {
    let depth_rounding = match precision {
        OitDepthPrecision::Rg32Float => "",
        OitDepthPrecision::Rg16Float => {
            "depth = unpack2x16float(pack2x16float(vec2<f32>(depth))).x;"
        }
    };
    fragment_entry(
        interface,
        "RendererOitPeelOutput",
        r#"@group(3) @binding(0) var renderer_oit_min_max: texture_2d<f32>;
@group(3) @binding(1) var renderer_oit_front: texture_2d<f32>;
struct RendererOitPeelOutput {
    @location(0) extrema: vec2<f32>,
    @location(1) front: vec4<f32>,
    @location(2) back: vec4<f32>,
};"#,
        &format!(
            r#"let final_color = renderer_forward_shade({{arguments}});
    var depth = {{position}}.z;
    {depth_rounding}
    let coordinate = vec2<i32>({{position}}.xy);
    let last_depth = textureLoad(renderer_oit_min_max, coordinate, 0);
    let last_front = textureLoad(renderer_oit_front, coordinate, 0);
    let nearest_depth = -last_depth.x;
    let furthest_depth = last_depth.y;
    var output: RendererOitPeelOutput;
    output.extrema = vec2<f32>(-99999.0);
    output.front = last_front;
    output.back = vec4<f32>(0.0);
    if depth < nearest_depth || depth > furthest_depth {{ return output; }}
    if depth > nearest_depth && depth < furthest_depth {{
        output.extrema = vec2<f32>(-depth, depth);
        return output;
    }}
    let alpha_multiplier = 1.0 - last_front.a;
    if depth == nearest_depth {{
        output.front = vec4<f32>(
            last_front.rgb + final_color.rgb * final_color.a * alpha_multiplier,
            1.0 - alpha_multiplier * (1.0 - final_color.a));
    }} else {{ output.back = final_color; }}
    return output;"#,
        ),
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
