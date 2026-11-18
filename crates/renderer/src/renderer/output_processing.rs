//! Frame output processing selected separately from material shading.

use fluxel_rhi::api::{Error, TextureFormat};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderCompositionRequest, ShaderStage, SourceUnit,
};

/// Maps a linear floating-point scene color to a display-referred sRGB output.
///
/// RGB uses the Babylon Standard curve after exposure. Alpha passes through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputProcessing {
    exposure: f32,
}

impl OutputProcessing {
    /// Selects a finite, nonnegative linear exposure multiplier.
    pub fn new(exposure: f32) -> Result<Self, super::RendererError> {
        if !exposure.is_finite() || exposure < 0.0 {
            return Err(Error::InvalidRange.into());
        }
        Ok(Self { exposure })
    }

    /// Returns the linear exposure multiplier.
    pub const fn exposure(self) -> f32 {
        self.exposure
    }

    pub(super) fn shader_source(self, target: TextureFormat) -> Result<String, Error> {
        let transfer = match target {
            TextureFormat::Rgba8UnormSrgb | TextureFormat::Bgra8UnormSrgb => "mapped",
            TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm => "encode_srgb(mapped)",
            _ => return Err(Error::Unsupported("sRGB output target format")),
        };
        // Capture the exact selected value in the frame's terminal program.
        // Bit encoding avoids rounding, locale and decimal-literal issues.
        let exposure = self.exposure().to_bits();
        Ok(format!(
            r#"
fn encode_srgb(linear: vec3<f32>) -> vec3<f32> {{
    let low = 12.92 * linear;
    let high = 1.055 * pow(linear, vec3<f32>(0.41666)) - 0.055;
    return select(high, low, linear <= vec3<f32>(0.0031308));
}}

fn process_scene_output(color: vec4<f32>) -> vec4<f32> {{
    let exposure = bitcast<f32>({exposure}u);
    if exposure == 0.0 {{
        return vec4<f32>(0.0, 0.0, 0.0, color.a);
    }}
    let exposed = max(color.rgb, vec3<f32>(0.0)) * exposure;
    let mapped = vec3<f32>(1.0) - exp2(-1.590579 * exposed);
    return vec4<f32>(clamp({transfer}, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}}
"#
        ))
    }
}

pub(super) fn scene_output_request(
    processing: Option<OutputProcessing>,
    target: TextureFormat,
) -> Result<ShaderCompositionRequest, Error> {
    let output = match processing {
        Some(processing) => processing.shader_source(target)?,
        None => PASSTHROUGH_OUTPUT.into(),
    };
    Ok(ShaderCompositionRequest::new(vec![
        SourceUnit::new("scene-output-processing", output),
        SourceUnit::new("scene-color-composite", SCENE_COLOR_COMPOSITE_TEMPLATE),
    ])
    .with_entry_point(EntryPoint::new(
        "deferred_composite_vs",
        ShaderStage::Vertex,
    ))
    .with_entry_point(EntryPoint::new(
        "deferred_composite_fs",
        ShaderStage::Fragment,
    ))
    .with_binding(
        BindingDeclaration::new(0, 0, BindingKind::Texture)
            .with_visibility([ShaderStage::Fragment]),
    )
    .with_binding(
        BindingDeclaration::new(0, 1, BindingKind::Sampler)
            .with_visibility([ShaderStage::Fragment]),
    ))
}

const PASSTHROUGH_OUTPUT: &str = r#"
fn process_scene_output(color: vec4<f32>) -> vec4<f32> {
    return color;
}
"#;

const SCENE_COLOR_COMPOSITE_TEMPLATE: &str = r#"
@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;

@vertex
fn deferred_composite_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

@fragment
fn deferred_composite_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(scene_color, 0));
    // Fragment position already addresses the pixel centre.
    let color = textureSampleLevel(scene_color, scene_sampler, position.xy / size, 0.0);
    return process_scene_output(color);
}
"#;
