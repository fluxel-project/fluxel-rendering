//! External metallic-roughness Material definition for the S7 native consumer.

use std::collections::BTreeMap;

use fluxel_material::{
    BindingLocation, MaterialCodeNode, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_shader::SourceUnit;

/// The external callable selected as the metallic-roughness property root.
pub const PROPERTIES: &str = "external_pbr_properties";

/// Builds the real Material schema used by the native PBR evidence.
pub fn definition() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "external_pbr_base".into(),
            inputs: vec![],
            output: MaterialType::Vec4,
            calls: vec![],
            wgsl: "fn external_pbr_base() -> vec4<f32> { let texel = textureLoad(material_texture, vec2<i32>(0, 0), 0); return vec4<f32>(texel.rgb * tint.rgb, 1.0); }".into(),
        }],
        "external_pbr_base",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![
            MaterialResource {
                name: "material_tint".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation { group: 0, binding: 0 },
            },
            MaterialResource {
                name: "material_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation { group: 0, binding: 1 },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughness {
            properties: PROPERTIES.into(),
        },
    )
    .with_code_node(MaterialCodeNode {
        name: PROPERTIES.into(),
        inputs: vec![],
        output: MaterialType::Vec2,
        calls: vec![],
        source: SourceUnit::new(
            "external-pbr-properties",
            "fn external_pbr_properties() -> vec2<f32> { let texel = textureLoad(material_texture, vec2<i32>(0, 0), 0); return vec2<f32>(texel.a, tint.a); }",
        ),
        resources: vec![],
    })
}
