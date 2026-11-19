//! Mixed asset/capture receiver fixtures for same-Graph Forward submissions.

use std::collections::BTreeMap;

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialCodeNode, MaterialDefinition, MaterialNode,
    MaterialNodeGraph, MaterialNodeId, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialType, SurfaceOutput,
};
use fluxel_shader::{ShaderComposer, SourceUnit};

use crate::{ForwardAssets, RenderTransform, RenderView, SceneAssetRef};

pub(crate) fn receiver(
    assets: &mut ForwardAssets,
    capture_aliases: bool,
) -> crate::PreparedCaptureReceiver {
    build_receiver(assets, capture_aliases, true)
}

pub(crate) fn texture_load_receiver(assets: &mut ForwardAssets) -> crate::PreparedCaptureReceiver {
    build_receiver(assets, false, false)
}

fn build_receiver(
    assets: &mut ForwardAssets,
    capture_aliases: bool,
    sampled: bool,
) -> crate::PreparedCaptureReceiver {
    let geometry = assets.geometry.first().expect("producer geometry").clone();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material = match materials.acquire(&materials.create().unwrap()).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(material(capture_aliases, sampled), ResidentBytes::new(0))
            .unwrap(),
        _ => unreachable!(),
    };
    assets.materials.push(material.clone());
    let texture = SceneAssetRef::from_snapshot(assets.textures.first().expect("producer texture"));
    let mut textures = vec![
        crate::ForwardTextureBinding::asset("base_color", texture),
        crate::ForwardTextureBinding::asset(
            "normal_texture",
            SceneAssetRef::from_snapshot(assets.textures.first().unwrap()),
        ),
        crate::ForwardTextureBinding::asset(
            "orm",
            SceneAssetRef::from_snapshot(assets.textures.first().unwrap()),
        ),
        crate::ForwardTextureBinding::capture("captured_color", "producer_capture"),
    ];
    if capture_aliases {
        textures.push(crate::ForwardTextureBinding::capture(
            "captured_alias",
            "producer_capture",
        ));
    }
    crate::prepare_forward_capture_receiver(
        &RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
        crate::ForwardDrawSource::Feature {
            feature: crate::FeatureIssuer::new(0, 0)
                .issue("mixed-capture")
                .unwrap(),
            local_id: 11,
        },
        0,
        &crate::ForwardCaptureResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            textures,
            samplers: Vec::new(),
            tint: [1.0; 4],
        },
        RenderTransform::identity(),
        assets,
        &mut ShaderComposer::new(),
    )
    .unwrap()
}

fn material(capture_aliases: bool, sampled: bool) -> MaterialDefinition {
    let mut resources = vec![
        MaterialResource {
            name: "tint".into(),
            kind: MaterialResourceKind::UniformVec4,
            location: BindingLocation {
                group: 0,
                binding: 0,
            },
        },
        MaterialResource {
            name: "base_color".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 1,
            },
        },
        MaterialResource {
            name: "normal_texture".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 2,
            },
        },
        MaterialResource {
            name: "orm".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 3,
            },
        },
        MaterialResource {
            name: "captured_color".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 4,
            },
        },
    ];
    if capture_aliases {
        resources.push(MaterialResource {
            name: "captured_alias".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 5,
            },
        });
    }
    if sampled {
        resources.push(MaterialResource {
            name: "sample_sampler".into(),
            kind: MaterialResourceKind::Sampler { comparison: false },
            location: BindingLocation {
                group: 0,
                binding: if capture_aliases { 6 } else { 5 },
            },
        });
    }
    let alias = if capture_aliases && sampled {
        " + textureSample(captured_alias, sample_sampler, vec2<f32>(0.0))"
    } else if capture_aliases {
        " + textureLoad(captured_alias, vec2<i32>(0), 0)"
    } else {
        ""
    };
    let sample = if sampled {
        "textureSample(base_color, sample_sampler, vec2<f32>(0.0)) + textureSample(normal_texture, sample_sampler, vec2<f32>(0.0)) + textureSample(orm, sample_sampler, vec2<f32>(0.0)) + textureSample(captured_color, sample_sampler, vec2<f32>(0.0))"
    } else {
        "textureLoad(base_color, vec2<i32>(0), 0) + textureLoad(normal_texture, vec2<i32>(0), 0) + textureLoad(orm, vec2<i32>(0), 0) + textureLoad(captured_color, vec2<i32>(0), 0)"
    };
    MaterialDefinition::new(
        Vec::new(),
        "mixed",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "tint".into(),
        }],
        resources,
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
    .with_code_node(MaterialCodeNode {
        name: "mixed".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec4,
        calls: Vec::new(),
        resources: Vec::new(),
        source: SourceUnit::new(
            "mixed-capture",
            format!("fn mixed() -> vec4<f32> {{ return ({sample}{alias}) * tint; }}"),
        ),
    })
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "mixed".into(),
            inputs: Vec::new(),
        }],
        output: MaterialNodeId(1),
    })
}
