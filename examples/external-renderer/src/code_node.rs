//! Typed external Material code nodes for the bounded Deferred consumer.
//!
//! The node owns portable source and no renderer, Graph, or GPU authority.

use std::collections::BTreeMap;

use fluxel_material::{
    BindingLocation, FragmentCall, FragmentInput, MaterialCodeNode, MaterialDefinition,
    MaterialFragment, MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType,
    SurfaceOutput,
};
use fluxel_shader::SourceUnit;

/// Two static source variants whose channel mapping is independently measurable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelMapVariant {
    /// Blue, green, red scaled by `0.5, 1.0, 1.0`.
    First,
    /// Blue, green, red scaled by `1.0, 0.5, 1.0`.
    Second,
}

impl ChannelMapVariant {
    /// Stable evidence label used by the native consumer.
    pub const fn label(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Second => "second",
        }
    }

    fn function(self) -> &'static str {
        match self {
            Self::First => "external_channel_map_first",
            Self::Second => "external_channel_map_second",
        }
    }

    /// Creates the external material node with an explicit vec4 contract.
    pub fn node(self) -> MaterialCodeNode {
        let function = self.function();
        let source = match self {
            Self::First => format!(
                "fn {function}(input: vec4<f32>) -> vec4<f32> {{\n    return vec4<f32>(input.b * 0.5, input.g, input.r, input.a);\n}}\n"
            ),
            Self::Second => format!(
                "fn {function}(input: vec4<f32>) -> vec4<f32> {{\n    return vec4<f32>(input.b, input.g * 0.5, input.r, input.a);\n}}\n"
            ),
        };
        MaterialCodeNode {
            name: function.into(),
            inputs: vec![FragmentInput {
                name: "input".into(),
                ty: MaterialType::Vec4,
            }],
            output: MaterialType::Vec4,
            calls: vec![],
            source: SourceUnit::new(format!("external-channel-map-{}", self.label()), source),
            resources: vec![],
        }
    }

    /// Builds a material whose root samples the usual texture/tint bindings,
    /// then delegates channel mapping to this external code node.
    pub fn material(self) -> MaterialDefinition {
        let function = self.function();
        MaterialDefinition::new(
            vec![MaterialFragment {
                name: "external_channel_root".into(),
                inputs: vec![],
                output: MaterialType::Vec4,
                calls: vec![FragmentCall {
                    fragment: function.into(),
                    arguments: vec![MaterialType::Vec4],
                }],
                wgsl: format!(
                    "fn external_channel_root() -> vec4<f32> {{ return {function}(textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint); }}"
                ),
            }],
            "external_channel_root",
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
            SurfaceOutput::Color,
        )
        .with_code_node(self.node())
    }
}
