//! Material code-node admission and source-provenance tests for Deferred.

use super::*;

use std::{collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentCall, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialError, MaterialFragment, MaterialNode, MaterialNodeGraph, MaterialNodeId,
    MaterialNodeInput, MaterialNodeInputSource, MaterialParameter, MaterialResource,
    MaterialResourceKind, MaterialType, SurfaceOutput,
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

fn node_graph_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "combine_nodes".into(),
            inputs: vec![
                fluxel_material::FragmentInput {
                    name: "left".into(),
                    ty: MaterialType::Vec4,
                },
                fluxel_material::FragmentInput {
                    name: "right".into(),
                    ty: MaterialType::Vec4,
                },
            ],
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn combine_nodes(left: vec4<f32>, right: vec4<f32>) -> vec4<f32> { return (left + right) * 0.5; }".into(),
        }],
        "unused_legacy_root",
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
    .with_code_node(code_node(VALID_NODE, "node_color"))
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![
            MaterialNode {
                id: MaterialNodeId(1),
                callable: "node_color".into(),
                inputs: Vec::new(),
            },
            MaterialNode {
                id: MaterialNodeId(2),
                callable: "combine_nodes".into(),
                inputs: vec![
                    MaterialNodeInput {
                        input: "left".into(),
                        source: MaterialNodeInputSource::Node(MaterialNodeId(1)),
                    },
                    MaterialNodeInput {
                        input: "right".into(),
                        source: MaterialNodeInputSource::Node(MaterialNodeId(1)),
                    },
                ],
            },
        ],
        output: MaterialNodeId(2),
    })
}

fn cyclic_node_graph_material() -> MaterialDefinition {
    node_graph_material().with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "combine_nodes".into(),
            inputs: vec![
                MaterialNodeInput {
                    input: "left".into(),
                    source: MaterialNodeInputSource::Node(MaterialNodeId(1)),
                },
                MaterialNodeInput {
                    input: "right".into(),
                    source: MaterialNodeInputSource::Node(MaterialNodeId(1)),
                },
            ],
        }],
        output: MaterialNodeId(1),
    })
}

pub(super) fn sampled_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "sampled_color".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn sampled_color() -> vec4<f32> { return textureSample(material_texture, material_sampler, vec2<f32>(0.5, 0.5)) * tint; }".into(),
        }],
        "sampled_color",
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
            MaterialResource {
                name: "material_sampler".into(),
                kind: MaterialResourceKind::Sampler { comparison: false },
                location: BindingLocation { group: 0, binding: 2 },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
}

pub(super) fn scene_and_assets(
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
                    encoding: crate::TextureEncoding::Linear,
                    mip_levels: 1,
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
            textures: vec![crate::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&texture),
            )],
            sampler: None,
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
    assert_eq!(renderer.deferred_probe_facts().unwrap().passes, 5);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 8);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    block_on(renderer.readback(&forward)).unwrap();
    block_on(renderer.readback(&deferred)).unwrap();
}

#[test]
fn node_graph_material_runs_forward_and_deferred_through_the_same_definition() {
    let (snapshot, view, assets) = scene_and_assets(node_graph_material());
    let selected = ForwardCpuPipeline::new(BoundsVisibility)
        .prepare(&snapshot, view, bounds())
        .unwrap();
    let prepared = prepare_forward(&selected, &assets, &mut ShaderComposer::new()).unwrap();
    let source = prepared[0].prepared().shader().source();
    assert!(source.contains("fluxel_material_node_1"));
    assert!(source.contains("fluxel_material_node_2"));

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
    block_on(renderer.readback(&forward)).unwrap();
    block_on(renderer.readback(&deferred)).unwrap();
    assert_eq!(renderer.deferred_probe_facts().unwrap().passes, 5);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 8);
}

#[test]
fn cyclic_node_graph_is_refused_before_backend_work() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets(cyclic_node_graph_material());
    let before = counters(&native);
    let error = block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), &assets))
        .expect_err("cyclic author graph must be refused before backend work");
    assert!(matches!(
        error,
        RendererError::ForwardPreparation(crate::ForwardPreparationError::Material(
            crate::MaterialPreparationError::Material(MaterialError::NodeCycle {
                node: MaterialNodeId(1)
            })
        ))
    ));
    assert_eq!(counters(&native), before);
}

#[test]
fn ordinary_sampler_is_bound_and_retained_for_forward_submission() {
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect("ordinary sampled material must submit");
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.readback(&submission)).unwrap();
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 1);
}

#[test]
fn ordinary_sampler_is_bound_and_retained_for_deferred_submission() {
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let submission = block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), &assets))
        .expect("ordinary sampled material must submit through Deferred");
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 2);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.readback(&submission)).unwrap();
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 2);
}

#[test]
fn unfilterable_texture_refuses_sampler_before_backend_allocation() {
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let (device, native) = deferred_device(Script::AcceptedExact);
    native.texture_filterable.store(false, Ordering::SeqCst);
    let mut renderer = Renderer::new(device).unwrap();
    let before = counters(&native);
    let error = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect_err("unfilterable texture must refuse an ordinary sampler");
    assert!(matches!(
        error,
        RendererError::ForwardOperation {
            operation: "sampled material texture admission",
            error: fluxel_rhi::api::Error::Unsupported("filterable material texture"),
        }
    ));
    assert_eq!(counters(&native), before);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 0);
}

#[test]
fn unfilterable_texture_refuses_deferred_sampler_before_backend_allocation() {
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let (device, native) = deferred_device(Script::AcceptedExact);
    native.texture_filterable.store(false, Ordering::SeqCst);
    let mut renderer = Renderer::new(device).unwrap();
    let before = counters(&native);
    let error = block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), &assets))
        .expect_err("unfilterable texture must refuse an ordinary Deferred sampler");
    assert!(matches!(
        error,
        RendererError::Rhi(fluxel_rhi::api::Error::Unsupported(
            "filterable material texture"
        ))
    ));
    assert_eq!(counters(&native), before);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 0);
}

#[test]
fn unfilterable_scene_color_refuses_deferred_composite_before_sampler_allocation() {
    let (snapshot, view, assets) = scene_and_assets(material(code_node(VALID_NODE, "node_color")));
    let (device, native) = deferred_device(Script::AcceptedExact);
    native.texture_filterable.store(false, Ordering::SeqCst);
    let mut renderer = Renderer::new(device).unwrap();
    let error = block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), &assets))
        .expect_err("unfilterable scene-color must refuse composite allocation");
    assert!(matches!(
        error,
        RendererError::Rhi(fluxel_rhi::api::Error::Unsupported(
            "filterable Deferred scene-color"
        ))
    ));
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 0);
}

#[test]
fn sampler_owner_is_released_when_its_bind_group_is_rejected() {
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let (device, native) = deferred_device(Script::SamplerBindGroupRejected);
    let mut renderer = Renderer::new(device).unwrap();
    let error = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect_err("sampler bind group rejection must fail the submission");
    assert!(matches!(
        error,
        RendererError::ForwardOperation {
            operation: "material bind group",
            error: fluxel_rhi::api::Error::InvalidState,
        }
    ));
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 1);
}

#[test]
fn deferred_sampler_owner_is_released_when_its_bind_group_is_rejected() {
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let (device, native) = deferred_device(Script::SamplerBindGroupRejected);
    let mut renderer = Renderer::new(device).unwrap();
    let error = block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), &assets))
        .expect_err("Deferred sampler bind group rejection must fail the submission");
    assert!(matches!(
        error,
        RendererError::Rhi(fluxel_rhi::api::Error::InvalidState)
    ));
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 2);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 2);
}
