//! Generated WGSL emission for validated material surface outputs.

use super::super::pbr_factors;
use crate::{
    EmissiveConfiguration, MaterialNodeInputSource, MaterialType, PbrSurfaceFactors,
    SurfaceAlphaMode, SurfaceExtensions, SurfaceOutput, SurfacePolicy,
};

pub(in crate::compiler) fn append_evaluators(
    generated: &mut String,
    root_fragment: &str,
    root_pbr_inputs: &[(MaterialNodeInputSource, MaterialType)],
    output: &SurfaceOutput,
    extensions: &SurfaceExtensions,
    emissive_configuration: Option<&EmissiveConfiguration>,
    pbr_surface_factors: Option<&PbrSurfaceFactors>,
    policy: SurfacePolicy,
    vertex_varyings: &[(u32, MaterialType)],
) {
    match output {
        SurfaceOutput::Color => generated.push_str(&format!(
            "fn evaluate_material() -> vec4<f32> {{\n    return {root_fragment}();\n}}\n"
        )),
        SurfaceOutput::ColorUv0 => generated.push_str(&format!(
            "fn evaluate_material_uv0(uv: vec2<f32>) -> vec4<f32> {{\n    return {root_fragment}(uv);\n}}\n"
        )),
        SurfaceOutput::MetallicRoughness { properties } => {
            generated.push_str(&format!(
                "fn evaluate_material() -> vec4<f32> {{\n    return {root_fragment}();\n}}\n"
            ));
            generated.push_str(&format!(
                "fn evaluate_metallic_roughness() -> vec2<f32> {{\n    return {properties}();\n}}\n"
            ));
        }
        SurfaceOutput::MetallicRoughnessUv0 { properties } => pbr_factors::append_uv0_evaluators(
            generated,
            root_fragment,
            root_pbr_inputs,
            Some(properties),
            pbr_surface_factors,
            vertex_varyings,
        ),
        SurfaceOutput::MetallicRoughnessUv0Normal { properties, normal } => {
            pbr_factors::append_uv0_evaluators(
                generated,
                root_fragment,
                root_pbr_inputs,
                Some(properties),
                pbr_surface_factors,
                vertex_varyings,
            );
            append_normal_extensions(
                generated,
                normal,
                extensions,
                emissive_configuration,
                pbr_surface_factors,
            );
        }
        SurfaceOutput::MetallicRoughnessUv0Graph { normal, .. } => {
            pbr_factors::append_uv0_evaluators(
                generated,
                root_fragment,
                root_pbr_inputs,
                None,
                pbr_surface_factors,
                vertex_varyings,
            );
            if let Some(normal) = normal {
                append_normal_extensions(
                    generated,
                    normal,
                    extensions,
                    emissive_configuration,
                    pbr_surface_factors,
                );
            } else {
                append_neutral_effects(generated);
            }
        }
    }
    append_policy_helpers(generated, policy);
}

fn append_neutral_effects(generated: &mut String) {
    generated.push_str(
        "fn evaluate_occlusion_uv0(uv: vec2<f32>) -> f32 { return 1.0; }\n\
fn evaluate_emissive_uv0(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0); }\n",
    );
}

fn append_normal_extensions(
    generated: &mut String,
    normal: &str,
    extensions: &SurfaceExtensions,
    emissive_configuration: Option<&EmissiveConfiguration>,
    pbr_surface_factors: Option<&PbrSurfaceFactors>,
) {
    generated.push_str(&format!(
        "fn evaluate_normal_uv0(uv: vec2<f32>) -> vec3<f32> {{\n    return {normal}(uv);\n}}\n"
    ));
    generated.push_str(&pbr_factors::uv0_occlusion_evaluator(
        extensions.occlusion.as_deref(),
        pbr_surface_factors,
    ));
    match (&extensions.emissive, emissive_configuration) {
        (Some(_), Some(_)) => unreachable!("validated emissive configuration conflict"),
        (Some(emissive), None) => generated.push_str(&format!(
            "fn evaluate_emissive_uv0(uv: vec2<f32>) -> vec3<f32> {{\n    return {emissive}(uv);\n}}\n"
        )),
        (None, Some(configuration)) => append_emissive_configuration(generated, configuration),
        (None, None) => generated.push_str(
            "fn evaluate_emissive_uv0(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0); }\n",
        ),
    }
}

fn append_emissive_configuration(generated: &mut String, configuration: &EmissiveConfiguration) {
    let color = configuration
        .color_parameter
        .as_deref()
        .unwrap_or("vec3<f32>(0.0)");
    let intensity = configuration
        .intensity_parameter
        .as_deref()
        .unwrap_or("1.0");
    let texture = configuration.texture.as_ref();
    let level = texture
        .and_then(|texture| texture.level_parameter.as_deref())
        .unwrap_or("1.0");
    let mut bindings = [color, intensity, level, "", ""];
    if let Some(texture) = texture {
        bindings[3] = &texture.texture;
        bindings[4] = &texture.sampler;
    }
    let uv = unique_emissive_local("emissive_uv", &bindings);
    generated.push_str(&format!(
        "fn evaluate_emissive_uv0({uv}: vec2<f32>) -> vec3<f32> {{\n"
    ));
    if let Some(texture) = texture {
        let sample = format!(
            "textureSample({}, {}, {uv}).rgb",
            texture.texture, texture.sampler
        );
        let linear = match texture.encoding {
            crate::MaterialTextureEncoding::Linear | crate::MaterialTextureEncoding::Srgb => sample,
            crate::MaterialTextureEncoding::SrgbEncodedUnorm => {
                let name = unique_emissive_local("emissive_sample", &bindings);
                generated.push_str(&format!("    let {name} = {sample};\n"));
                match configuration.srgb_encoded_unorm_decode {
                    crate::EmissiveSrgbDecode::BabylonApproximate => {
                        format!("pow({name}, vec3<f32>(2.2))")
                    }
                    crate::EmissiveSrgbDecode::Exact => format!(
                        "select(\n        pow(0.947867299 * ({name} + vec3<f32>(0.055)), vec3<f32>(2.4)),\n        0.0773993808 * {name},\n        {name} <= vec3<f32>(0.04045)\n    )"
                    ),
                }
            }
        };
        generated.push_str(&format!(
            "    return {color} * {linear} * {level} * {intensity};\n"
        ));
    } else {
        generated.push_str(&format!("    return {color} * {intensity};\n"));
    }
    generated.push_str("}\n");
}

fn unique_emissive_local(seed: &str, bindings: &[&str]) -> String {
    let mut name = seed.to_owned();
    while bindings.contains(&name.as_str()) {
        name.push('_');
    }
    name
}

fn append_policy_helpers(generated: &mut String, policy: SurfacePolicy) {
    let rejects_alpha = matches!(
        policy.alpha_mode,
        SurfaceAlphaMode::Mask | SurfaceAlphaMode::MaskBlend
    );
    let preserves_alpha = matches!(
        policy.alpha_mode,
        SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend
    );
    generated.push_str(&format!(
        "fn material_alpha_rejected(alpha: f32) -> bool {{ return {rejects_alpha} && alpha < {}; }}\n",
        policy.alpha_cutoff,
    ));
    generated.push_str(&format!(
        "fn material_output_alpha(alpha: f32) -> f32 {{ if ({preserves_alpha}) {{ return alpha; }} return 1.0; }}\n"
    ));
    let two_sided_lighting = !policy.back_face_culling && policy.two_sided_lighting;
    generated.push_str(&format!(
        "fn material_two_sided_lighting() -> bool {{ return {}; }}\n",
        two_sided_lighting,
    ));
}
