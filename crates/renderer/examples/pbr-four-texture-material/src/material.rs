//! Authored normal unpacking, packed ORM and independent color bindings.
use fluxel_material::*;
use fluxel_shader::SourceUnit;
use std::collections::BTreeMap;

pub(super) fn definition() -> MaterialDefinition {
    let resources = [
        ("base_color", MaterialResourceKind::Texture2dF32, 3),
        ("values", MaterialResourceKind::UniformVec4, 5),
        ("emissive", MaterialResourceKind::Texture2dF32, 4),
        (
            "images",
            MaterialResourceKind::Sampler { comparison: false },
            1,
        ),
        ("normal", MaterialResourceKind::Texture2dF32, 0),
        ("orm", MaterialResourceKind::Texture2dF32, 2),
    ]
    .map(|(name, kind, binding)| MaterialResource {
        name: name.into(),
        kind,
        location: BindingLocation { group: 0, binding },
    })
    .to_vec();
    let nodes = [
        (
            "base",
            MaterialType::Vec4,
            "fn base(uv: vec2<f32>) -> vec4<f32> { return textureSample(base_color, images, uv) * tint; }",
        ),
        (
            "properties",
            MaterialType::Vec2,
            "fn properties(uv: vec2<f32>) -> vec2<f32> { let value = textureSample(orm, images, uv); return vec2<f32>(value.b, value.g); }",
        ),
        (
            "direction",
            MaterialType::Vec3,
            "fn direction(uv: vec2<f32>) -> vec3<f32> { return textureSample(normal, images, uv).rgb * 2.0 - vec3<f32>(1.0); }",
        ),
        (
            "occlusion",
            MaterialType::F32,
            "fn occlusion(uv: vec2<f32>) -> f32 { return textureSample(orm, images, uv).r; }",
        ),
        (
            "emission",
            MaterialType::Vec3,
            "fn emission(uv: vec2<f32>) -> vec3<f32> { return textureSample(emissive, images, uv).rgb * 0.2; }",
        ),
    ];
    let mut result = MaterialDefinition::new(
        Vec::new(),
        "unused",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "values".into(),
        }],
        resources,
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughnessUv0Normal {
            properties: "properties".into(),
            normal: "direction".into(),
        },
    )
    .with_texture_encoding_requirements(vec![
        MaterialTextureEncodingRequirement {
            resource: "base_color".into(),
            encoding: MaterialTextureEncoding::Srgb,
        },
        MaterialTextureEncodingRequirement {
            resource: "emissive".into(),
            encoding: MaterialTextureEncoding::Srgb,
        },
        MaterialTextureEncodingRequirement {
            resource: "normal".into(),
            encoding: MaterialTextureEncoding::Linear,
        },
        MaterialTextureEncodingRequirement {
            resource: "orm".into(),
            encoding: MaterialTextureEncoding::Linear,
        },
    ])
    .with_surface_extensions(SurfaceExtensions {
        occlusion: Some("occlusion".into()),
        emissive: Some("emission".into()),
    });
    for (name, output, source) in nodes {
        result = result.with_code_node(MaterialCodeNode {
            name: name.into(),
            inputs: vec![FragmentInput {
                name: "uv".into(),
                ty: MaterialType::Vec2,
            }],
            output,
            calls: Vec::new(),
            source: SourceUnit::new(format!("four-texture:{name}"), source),
            resources: Vec::new(),
        });
    }
    result.with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "base".into(),
            inputs: vec![MaterialNodeInput {
                input: "uv".into(),
                source: MaterialNodeInputSource::Uv0,
            }],
        }],
        output: MaterialNodeId(1),
    })
}
