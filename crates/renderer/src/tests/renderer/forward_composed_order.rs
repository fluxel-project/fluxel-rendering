use super::*;

use std::collections::BTreeMap;

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentCall, MaterialAsset, MaterialDefinition, MaterialFragment,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::ShaderComposer;

use crate::{
    ForwardAssets, ForwardCpuPipeline, ForwardDrawInput, ForwardDrawOrder, ForwardDrawRequest,
    ForwardDrawSource, ForwardResources, FramePipeline, GeometryAsset, GeometryAttribute,
    GeometryData, GeometryLayout, GeometryStream, PipelineError, RenderBounds, RenderChange,
    RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView, SceneAssetRef,
};

#[derive(Clone, Copy)]
enum Ordering {
    Reverse,
    RewriteFirst,
}

impl FramePipeline<ForwardResources> for Ordering {
    fn prepare<'a>(
        &self,
        _: &'a crate::FrameSnapshot<ForwardResources>,
        _: crate::RenderViewId,
        _: RenderBounds,
    ) -> std::result::Result<crate::PreparedFrame<'a, ForwardResources>, PipelineError> {
        unreachable!("order_draws only invokes order_forward")
    }

    fn order_forward(
        &self,
        _: &crate::PreparedFrame<'_, ForwardResources>,
        draws: &mut [ForwardDrawOrder],
    ) -> std::result::Result<(), PipelineError> {
        match self {
            Self::Reverse => draws.reverse(),
            Self::RewriteFirst => {
                if let Some(first) = draws.first_mut() {
                    *first = ForwardDrawOrder::new(first.source(), first.order() + 1);
                }
            }
        }
        Ok(())
    }
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "sample_color".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn sample_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0); }".into(),
            },
            MaterialFragment {
                name: "custom_tint".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: vec![FragmentCall {
                    fragment: "sample_color".into(),
                    arguments: Vec::new(),
                }],
                wgsl: "fn custom_tint() -> vec4<f32> { return sample_color() * tint; }".into(),
            },
        ],
        "custom_tint",
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
}

fn assets_and_resources() -> (ForwardAssets, ForwardResources) {
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometry_store.create().unwrap();
    let geometry = match geometry_store.acquire(&geometry_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                GeometryData {
                    layout: GeometryLayout {
                        streams: vec![GeometryStream {
                            stride: 12,
                            step_mode: VertexStepMode::Vertex,
                            attributes: vec![GeometryAttribute {
                                semantic: "POSITION".into(),
                                offset: 0,
                                format: VertexFormat::Float32(3),
                            }],
                        }],
                    },
                    streams: vec![Vec::new()],
                    vertices: 0..0,
                },
                ResidentBytes::new(0),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material_handle = material_store.create().unwrap();
    let material = match material_store.acquire(&material_handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(material(), ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    };
    let texture_store = AssetStore::<crate::ColorTextureAsset, crate::ColorTextureData, ()>::new();
    let texture_handle = texture_store.create().unwrap();
    let texture = match texture_store.acquire(&texture_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                crate::ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![255; 4],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let resources = ForwardResources {
        geometry: SceneAssetRef::from_snapshot(&geometry),
        material: SceneAssetRef::from_snapshot(&material),
        texture: SceneAssetRef::from_snapshot(&texture),
        tint: [1.0; 4],
    };
    (
        ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: vec![texture],
        },
        resources,
    )
}

fn with_prepared_inputs<T>(
    use_inputs: impl FnOnce(
        &crate::PreparedFrame<'_, ForwardResources>,
        Vec<ForwardDrawInput>,
        Vec<ForwardDrawSource>,
    ) -> T,
) -> T {
    let (assets, resources) = assets_and_resources();
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let issuer = scene.feature_issuer().unwrap();
    let first_feature = issuer.issue("first").unwrap();
    let second_feature = issuer.issue("second").unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).unwrap(),
    ));
    changes.push(RenderChange::Insert(RenderRecord {
        resources: resources.clone(),
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([0.0; 3], [1.0; 3]).unwrap(),
    }));
    let receipt = scene.apply(&changes).unwrap();
    let view = receipt.inserted_views()[0];
    let scene_source = ForwardDrawSource::Scene(receipt.inserted()[0]);
    let snapshot = scene.snapshot();
    let pipeline = ForwardCpuPipeline::<crate::BoundsVisibility>::default();
    let frame = pipeline
        .prepare(
            &snapshot,
            view,
            RenderBounds::new([0.0; 3], [1.0; 3]).unwrap(),
        )
        .unwrap();
    let request = |local_id, order| ForwardDrawRequest {
        local_id,
        order,
        resources: resources.clone(),
        transform: RenderTransform::identity(),
    };
    let mut composer = ShaderComposer::new();
    let mut inputs = crate::prepare_forward(&frame, &assets, &mut composer).unwrap();
    inputs.extend(
        crate::preparation::prepare_feature_draws(
            &frame,
            first_feature,
            &[request(1, 0)],
            &assets,
            &mut composer,
        )
        .unwrap(),
    );
    inputs.extend(
        crate::preparation::prepare_feature_draws(
            &frame,
            second_feature,
            &[request(2, 0)],
            &assets,
            &mut composer,
        )
        .unwrap(),
    );
    let sources = vec![
        scene_source,
        ForwardDrawSource::Feature {
            feature: first_feature,
            local_id: 1,
        },
        ForwardDrawSource::Feature {
            feature: second_feature,
            local_id: 2,
        },
    ];
    use_inputs(&frame, inputs, sources)
}

#[test]
fn custom_pipeline_permutation_reorders_cross_source_draws() {
    with_prepared_inputs(|frame, inputs, sources| {
        let ordered = order_draws(&Ordering::Reverse, frame, inputs).unwrap();
        assert_eq!(
            ordered
                .into_iter()
                .map(|draw| draw.source())
                .collect::<Vec<_>>(),
            vec![sources[2], sources[1], sources[0]]
        );
    });
}

#[test]
fn rewritten_order_metadata_is_rejected() {
    with_prepared_inputs(|frame, inputs, _| {
        assert!(matches!(
            order_draws(&Ordering::RewriteFirst, frame, inputs),
            Err(PipelineError::InvalidDrawOrder)
        ));
    });
}

#[test]
fn duplicate_source_is_rejected_before_pipeline_ordering() {
    with_prepared_inputs(|frame, inputs, _| {
        let duplicate = vec![inputs[0].clone(), inputs[0].clone()];
        assert!(matches!(
            order_draws(&Ordering::Reverse, frame, duplicate),
            Err(PipelineError::InvalidDrawOrder)
        ));
    });
}

#[test]
fn equal_orders_use_scene_then_stable_feature_namespaces() {
    with_prepared_inputs(|frame, inputs, sources| {
        let pipeline = ForwardCpuPipeline::<crate::BoundsVisibility>::default();
        let ordered = order_draws(&pipeline, frame, inputs).unwrap();
        assert_eq!(
            ordered
                .into_iter()
                .map(|draw| draw.source())
                .collect::<Vec<_>>(),
            sources
        );
    });
}

#[test]
fn empty_draw_list_stays_empty() {
    with_prepared_inputs(|frame, _, _| {
        let pipeline = ForwardCpuPipeline::<crate::BoundsVisibility>::default();
        assert!(
            order_draws(&pipeline, frame, Vec::new())
                .unwrap()
                .is_empty()
        );
    });
}
