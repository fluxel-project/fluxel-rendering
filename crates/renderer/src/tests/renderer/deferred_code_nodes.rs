//! Material code-node admission and source-provenance tests for Deferred.

use super::*;

use std::{collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentCall, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialFragment, MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType,
    SurfaceOutput,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{ShaderComposer, ShaderError, SourceUnit};

use crate::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, FramePipeline, GeometryAsset, GeometryAttribute, GeometryData,
    GeometryLayout, GeometryStream, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, RendererError, SceneAssetRef, prepare_forward,
};

const NODE_ID: &str = "c1-node";
const VALID_NODE: &str = "fn node_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }";

fn code_node(source: &str, name: &str) -> MaterialCodeNode {
    MaterialCodeNode {
        name: name.into(),
        inputs: Vec::new(),
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(NODE_ID, source),
        resources: Vec::new(),
    }
}

fn material(node: MaterialCodeNode) -> MaterialDefinition {
    material_with_root(
        node,
        "fn tinted_texture() -> vec4<f32> { return node_color(); }",
    )
}

fn material_with_root(node: MaterialCodeNode, root_source: &str) -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "tinted_texture".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: vec![FragmentCall {
                fragment: "node_color".into(),
                arguments: Vec::new(),
            }],
            wgsl: root_source.into(),
        }],
        "tinted_texture",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![
            MaterialResource {
                name: "material_tint".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation {
                    group: 0,
                    binding: 0,
                },
            },
            MaterialResource {
                name: "material_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation {
                    group: 0,
                    binding: 1,
                },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
    .with_code_node(node)
}

fn scene_and_assets(
    definition: MaterialDefinition,
) -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    let geometries = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometries.create().unwrap();
    let geometry = match geometries.acquire(&geometry_handle).unwrap() {
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
                    streams: vec![vec![0; 36]],
                    vertices: 0..3,
                },
                ResidentBytes::new(36),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material_handle = materials.create().unwrap();
    let material = match materials.acquire(&material_handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(definition, ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    };
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let texture_handle = textures.create().unwrap();
    let texture = match textures.acquire(&texture_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![1, 2, 3, 255],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            texture: SceneAssetRef::from_snapshot(&texture),
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    let receipt = scene.apply(&changes).unwrap();
    (
        scene.snapshot(),
        receipt.inserted_views()[0],
        ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: vec![texture],
        },
    )
}

fn counters(native: &Native) -> (usize, usize, usize) {
    (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    )
}

fn assert_node_shader_error(error: RendererError) {
    let shader = match error {
        RendererError::ForwardPreparation(crate::ForwardPreparationError::Material(
            crate::MaterialPreparationError::Shader(error),
        )) => error,
        other => panic!("expected node shader refusal, got {other:?}"),
    };
    match shader {
        ShaderError::Parse {
            source: Some(fluxel_shader::SourceUnitId(id)),
            line: Some(1),
            ..
        }
        | ShaderError::Validate {
            source: Some(fluxel_shader::SourceUnitId(id)),
            line: Some(1),
            ..
        } if id == NODE_ID => {}
        other => panic!("expected node-local source diagnostic, got {other:?}"),
    }
}

#[test]
fn invalid_code_node_wgsl_keeps_author_source_and_refuses_before_backend_work() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets(material(code_node(
        "fn node_color() -> vec4<f32> { return ; }",
        "node_color",
    )));
    let before = counters(&native);
    let error = block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), &assets))
        .expect_err("invalid author node must be refused");
    assert_node_shader_error(error);
    assert_eq!(counters(&native), before);
}

#[test]
fn malformed_code_node_callable_is_refused_before_backend_work() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    // The complete WGSL is valid, but the node's declared vec4 result disagrees
    // with its actual vec3 result. This reaches callable reflection rather than
    // failing earlier because a function name is absent from the program.
    let (snapshot, view, assets) = scene_and_assets(material_with_root(
        code_node(
            "fn node_color() -> vec3<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0).rgb * tint.rgb; }",
            "node_color",
        ),
        "fn tinted_texture() -> vec4<f32> { return vec4<f32>(node_color(), 1.0); }",
    ));
    let before = counters(&native);
    let error = block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), &assets))
        .expect_err("typed callable mismatch must be refused");
    assert!(matches!(
        error,
        RendererError::ForwardPreparation(crate::ForwardPreparationError::Material(
            crate::MaterialPreparationError::Shader(ShaderError::CallableMismatch { name }),
        )) if name == "node_color"
    ));
    assert_eq!(counters(&native), before);
}

#[test]
fn valid_code_node_is_source_mapped_and_runs_forward_and_deferred() {
    let definition = material(code_node(VALID_NODE, "node_color"));
    let (snapshot, view, assets) = scene_and_assets(definition);
    let selected = ForwardCpuPipeline::new(BoundsVisibility)
        .prepare(&snapshot, view, bounds())
        .unwrap();
    let prepared = prepare_forward(&selected, &assets, &mut ShaderComposer::new()).unwrap();
    let shader = prepared[0].prepared().shader();
    let node_span = shader
        .source_map()
        .iter()
        .find(|span| span.source.0 == NODE_ID)
        .expect("node source span");
    assert_eq!(&shader.source()[node_span.bytes.clone()], VALID_NODE);

    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let deferred =
        block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), &assets)).unwrap();
    assert_eq!(renderer.deferred_probe_facts().unwrap().passes, 4);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 7);
    block_on(renderer.readback(&forward)).unwrap();
    block_on(renderer.readback(&deferred)).unwrap();
}
