//! Multi-record Deferred submission witnesses over the real renderer path.

use super::*;

use std::{cell::Cell, collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialFragment, MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput,
    MaterialNodeInputSource, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialType, SurfaceExtensions, SurfaceOutput,
};
use fluxel_rhi::api::{Device, Error, VertexFormat, VertexStepMode};

use crate::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, DeferredCpuPipeline,
    DeferredPhaseContext, ForwardAssets, ForwardCpuPipeline, ForwardResources, FramePipeline,
    FrameTopology, GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout, GeometryStream,
    PipelineError, PreparedFrame, RenderBounds, RenderChange, RenderChangeSet, RenderRecord,
    RenderScene, RenderTransform, RenderView, RendererError, SceneAssetRef,
};

fn bounds() -> RenderBounds {
    RenderBounds::new([-2.0; 3], [2.0; 3]).unwrap()
}

fn device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    device_with_colors(script, 3)
}

fn device_with_colors(script: Script, colors: u32) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_color_attachments = colors;
        caps.limits.max_bind_groups = 3;
        caps.limits.max_bindings_per_group = 7;
        caps.limits.max_vertex_buffers = 2;
        caps.limits.max_vertex_attributes = 4;
    })
}

fn geometry(offset: f32, tangent: bool) -> GeometryData {
    let mut bytes = Vec::new();
    for position in [
        [offset, 0.0, 0.0],
        [offset + 0.5, 0.0, 0.0],
        [offset, 0.5, 0.0],
    ] {
        for value in position
            .into_iter()
            .chain([0.0, 0.0, 1.0])
            .chain([0.0, 0.0])
        {
            bytes.extend(value.to_le_bytes());
        }
        if tangent {
            for value in [1.0_f32, 0.0, 0.0, 1.0] {
                bytes.extend(value.to_le_bytes());
            }
        }
    }
    let mut attributes = vec![
        GeometryAttribute {
            semantic: "POSITION".into(),
            offset: 0,
            format: VertexFormat::Float32(3),
        },
        GeometryAttribute {
            semantic: "NORMAL".into(),
            offset: 12,
            format: VertexFormat::Float32(3),
        },
        GeometryAttribute {
            semantic: "UV0".into(),
            offset: 24,
            format: VertexFormat::Float32(2),
        },
    ];
    if tangent {
        attributes.push(GeometryAttribute {
            semantic: "TANGENT".into(),
            offset: 32,
            format: VertexFormat::Float32(4),
        });
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: if tangent { 48 } else { 32 },
                step_mode: VertexStepMode::Vertex,
                attributes,
            }],
        },
        streams: vec![bytes],
        vertices: 0..3,
    }
}

fn material_at(effects: bool, locations: [u32; 3]) -> MaterialDefinition {
    let fragments = vec![
        MaterialFragment {
            name: "sampled".into(),
            inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn sampled(uv: vec2<f32>) -> vec4<f32> { let base = textureSample(base_color, material_sampler, uv); let emission = textureSample(emissive, material_sampler, uv); return (base + emission * 0.125) * tint; }".into(),
        },
        MaterialFragment {
            name: "properties".into(),
            inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
            output: MaterialType::Vec2,
            calls: Vec::new(),
            wgsl: "fn properties(uv: vec2<f32>) -> vec2<f32> { let packed = textureSample(orm, material_sampler, uv); return vec2<f32>(packed.g, packed.b); }".into(),
        },
    ];
    let resources = vec![
        MaterialResource {
            name: "material_tint".into(),
            kind: MaterialResourceKind::UniformVec4,
            location: BindingLocation {
                group: 0,
                binding: locations[0],
            },
        },
        MaterialResource {
            name: "base_color".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: locations[1],
            },
        },
        MaterialResource {
            name: "normal_texture".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 3,
            },
        },
        MaterialResource {
            name: "orm".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 4,
            },
        },
        MaterialResource {
            name: "emissive".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 5,
            },
        },
        MaterialResource {
            name: "material_sampler".into(),
            kind: MaterialResourceKind::Sampler { comparison: false },
            location: BindingLocation {
                group: 0,
                binding: locations[2],
            },
        },
    ];
    let definition = MaterialDefinition::new(
        fragments, "unused",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }],
        resources, BTreeMap::new(),
        SurfaceOutput::MetallicRoughnessUv0Normal { properties: "properties".into(), normal: "sample_normal".into() },
    )
    .with_code_node(MaterialCodeNode {
        name: "sample_normal".into(),
        inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
        output: MaterialType::Vec3,
        calls: Vec::new(),
        source: fluxel_shader::SourceUnit::new("multi:normal",
            "fn sample_normal(uv: vec2<f32>) -> vec3<f32> { return textureSample(normal_texture, material_sampler, uv).xyz; }"),
        resources: Vec::new(),
    })
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "sampled".into(),
            inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }],
        output: MaterialNodeId(1),
    });
    if effects {
        definition
            .with_code_node(MaterialCodeNode {
                name: "ao".into(),
                inputs: vec![FragmentInput {
                    name: "uv".into(),
                    ty: MaterialType::Vec2,
                }],
                output: MaterialType::F32,
                calls: vec![],
                source: fluxel_shader::SourceUnit::new(
                    "multi:ao",
                    "fn ao(uv: vec2<f32>) -> f32 { return textureSample(orm, material_sampler, uv).r; }",
                ),
                resources: vec![],
            })
            .with_surface_extensions(SurfaceExtensions {
                occlusion: Some("ao".into()),
                emissive: None,
            })
    } else {
        definition
    }
}

fn multi_scene(
    second_tangent: bool,
) -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    multi_scene_with_effects(second_tangent, true)
}

fn multi_scene_with_effects(
    second_tangent: bool,
    effects: bool,
) -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    multi_scene_with_effects_and_encoding(second_tangent, effects, crate::TextureEncoding::Linear)
}

fn multi_scene_with_effects_and_encoding(
    second_tangent: bool,
    effects: bool,
    encoding: crate::TextureEncoding,
) -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    multi_scene_at(second_tangent, effects, encoding, [0, 1, 2])
}

fn multi_scene_at(
    second_tangent: bool,
    effects: bool,
    encoding: crate::TextureEncoding,
    locations: [u32; 3],
) -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    let geometries = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let mut gs = Vec::new();
    let mut ms = Vec::new();
    let mut ts = Vec::new();
    for (index, (g, m, rgba)) in [
        (
            geometry(0.0, true),
            material_at(effects, locations),
            [240, 80, 40, 255],
        ),
        (
            geometry(0.75, second_tangent),
            material_at(false, locations),
            [40, 120, 240, 255],
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let h = geometries.create().unwrap();
        let bytes = u64::try_from(g.streams.iter().map(Vec::len).sum::<usize>()).unwrap();
        let geometry = match geometries.acquire(&h).unwrap() {
            Acquire::Producer(p) => p.commit(g, ResidentBytes::new(bytes)).unwrap(),
            _ => unreachable!(),
        };
        let h = materials.create().unwrap();
        let material = match materials.acquire(&h).unwrap() {
            Acquire::Producer(p) => p.commit(m, ResidentBytes::new(index as u64)).unwrap(),
            _ => unreachable!(),
        };
        let h = textures.create().unwrap();
        let texture = match textures.acquire(&h).unwrap() {
            Acquire::Producer(p) => p
                .commit(
                    ColorTextureData {
                        extent: [1, 1],
                        encoding,
                        mip_levels: 1,
                        rgba8: rgba.to_vec(),
                    },
                    ResidentBytes::new(4),
                )
                .unwrap(),
            _ => unreachable!(),
        };
        gs.push(geometry);
        ms.push(material);
        ts.push(texture);
    }
    let mut scene = RenderScene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    for index in 0..2 {
        changes.push(RenderChange::Insert(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&gs[index]),
                material: SceneAssetRef::from_snapshot(&ms[index]),
                textures: vec![
                    crate::ForwardTextureBinding::asset(
                        "orm",
                        SceneAssetRef::from_snapshot(&ts[(index + 1) % 2]),
                    ),
                    crate::ForwardTextureBinding::asset(
                        "emissive",
                        SceneAssetRef::from_snapshot(&ts[index]),
                    ),
                    crate::ForwardTextureBinding::asset(
                        "base_color",
                        SceneAssetRef::from_snapshot(&ts[index]),
                    ),
                    crate::ForwardTextureBinding::asset(
                        "normal_texture",
                        SceneAssetRef::from_snapshot(&ts[index]),
                    ),
                ],
                tint: [1.0, 0.75 + index as f32 * 0.25, 1.0, 1.0],
            },
            transform: RenderTransform::identity(),
            bounds: bounds(),
        }));
    }
    let receipt = scene.apply(&changes).unwrap();
    (
        scene.snapshot(),
        receipt.inserted_views()[0],
        ForwardAssets {
            geometry: gs,
            materials: ms,
            textures: ts,
        },
    )
}

#[path = "schema_bindings.rs"]
mod schema_binding_tests;

#[path = "deferred_multi_completion.rs"]
mod deferred_multi_completion_tests;

#[path = "deferred_multi_texture_failures.rs"]
mod deferred_multi_texture_failure_tests;

#[path = "deferred_multi_generation.rs"]
mod deferred_multi_generation_tests;

#[path = "encoding_lifecycle.rs"]
mod encoding_tests;

struct Counting {
    calls: Cell<usize>,
    declare: bool,
}
impl FramePipeline<ForwardResources> for Counting {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }
    fn topology(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
    ) -> std::result::Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::Deferred)
    }
    fn contribute_deferred(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
        context: &mut DeferredPhaseContext<'_, '_>,
    ) -> std::result::Result<(), PipelineError> {
        self.calls.set(self.calls.get() + 1);
        if !self.declare {
            return Ok(());
        }
        context.geometry(|p| {
            p.depends_on_uploads()?;
            p.read_vertex_streams()?;
            p.read_frame_uniforms()?;
            p.read_sampled_textures()?;
            p.write_gbuffer()
        })?;
        context.lighting(|p| {
            p.read_gbuffer()?;
            p.write_main_color()
        })
    }
}

#[test]
fn deferred_multi_record_without_effects_requires_only_two_color_attachments() {
    let (device, native) = device_with_colors(Script::AcceptedExact, 2);
    let mut renderer = Renderer::new(device).unwrap();
    let (scene, view, assets) = multi_scene_with_effects(true, false);
    let submission = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::Deferred { _frame, .. } = &pending.workload else {
        panic!("Deferred work")
    };
    assert_eq!(_frame.draws.len(), 2);
    assert!(_frame.effects.is_none());
    assert!(native.command_calls.load(Ordering::SeqCst) > 0);
    block_on(renderer.readback(&submission)).unwrap();
}

#[test]
fn deferred_multi_record_submission_keeps_distinct_owners_until_completion_and_warms_all_keys() {
    let (device, native) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (snapshot, view, assets) = multi_scene(true);
    let pipeline = Counting {
        calls: Cell::new(0),
        declare: true,
    };
    let cold =
        block_on(renderer.submit_deferred(&pipeline, snapshot.clone(), view, bounds(), &assets))
            .unwrap();
    let pending = renderer.pending_for(&cold).unwrap();
    let crate::renderer::PendingWork::Deferred { _frame, .. } = &pending.workload else {
        panic!("Deferred work")
    };
    assert_eq!(_frame.draws.len(), 2);
    assert_eq!(_frame.resident.streams.len(), 2);
    assert_eq!(_frame.resident.textures.len(), 2);
    assert_eq!(_frame.draws[0].textures.len(), 4);
    assert_eq!(_frame.draws[1].textures.len(), 4);
    assert_eq!(
        _frame.draws[0].textures[0], _frame.draws[0].textures[1],
        "one physical asset aliases the base-color and normal logical bindings"
    );
    assert_eq!(
        _frame.draws[0].textures[0], _frame.draws[1].textures[2],
        "one exact asset generation is shared across records"
    );
    assert_eq!(_frame.publication.textures.len(), 2);
    assert_ne!(
        _frame.draws[0].vertex_streams,
        _frame.draws[1].vertex_streams
    );
    assert_ne!(_frame.draws[0].textures, _frame.draws[1].textures);
    assert_ne!(
        _frame.draws[0].transform_range,
        _frame.draws[1].transform_range
    );
    assert!(
        _frame.effects.is_some(),
        "one effect material forces the shared three-MRT frame"
    );
    assert_eq!(renderer.deferred_cache_stats().geometry_hits, 0);
    assert_eq!(renderer.deferred_cache_stats().texture_hits, 0);
    assert!(native.command_calls.load(Ordering::SeqCst) > 0);
    block_on(renderer.readback(&cold)).unwrap();
    let warm =
        block_on(renderer.submit_deferred(&pipeline, snapshot.clone(), view, bounds(), &assets))
            .unwrap();
    assert_eq!(pipeline.calls.get(), 2);
    let stats = renderer.deferred_cache_stats();
    assert!(stats.geometry_hits >= 2);
    assert!(stats.texture_hits >= 2);
    block_on(renderer.readback(&warm)).unwrap();
    let mut surface = witness_surface(&device);
    block_on(surface.configure(&fluxel_rhi::api::SurfaceConfig {
        extent: [8, 8],
        format: TextureFormat::Rgba8Unorm,
        present_mode: fluxel_rhi::api::PresentMode::Fifo,
        alpha_mode: fluxel_rhi::api::CompositeAlphaMode::Opaque,
        color_space: fluxel_rhi::api::ColorSpace::Srgb,
        usage: fluxel_rhi::api::SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    }))
    .unwrap();
    let shown = block_on(renderer.submit_deferred_surface(
        &pipeline,
        snapshot,
        view,
        bounds(),
        &assets,
        block_on(surface.acquire()).unwrap(),
    ))
    .unwrap();
    let pending = renderer.pending_for(&shown).unwrap();
    let crate::renderer::PendingWork::DeferredSurface { _frame, .. } = &pending.workload else {
        panic!("Deferred Surface work")
    };
    assert_eq!(_frame.draws.len(), 2);
    assert_eq!(pipeline.calls.get(), 3);
    block_on(renderer.complete(&shown)).unwrap();
}

#[test]
fn deferred_multi_record_rejects_missing_inputs_and_mutated_actual_texture_before_recording() {
    let (device, native) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (bad, view, assets) = multi_scene(false);
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.created_textures.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    );
    assert!(
        block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            bad,
            view,
            bounds(),
            &assets
        ))
        .is_err()
    );
    assert_eq!(
        before,
        (
            native.created_buffers.load(Ordering::SeqCst),
            native.created_textures.load(Ordering::SeqCst),
            native.command_calls.load(Ordering::SeqCst)
        )
    );
    let (snapshot, view, assets) = multi_scene(true);
    let pipeline = Counting {
        calls: Cell::new(0),
        declare: true,
    };
    let submission =
        block_on(renderer.submit_deferred(&pipeline, snapshot, view, bounds(), &assets)).unwrap();
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::Deferred { _frame, .. } = &mut pending.workload else {
        panic!()
    };
    let selected = _frame.draws[0].textures[2];
    assert_ne!(selected, _frame.draws[0].textures[0]);
    Arc::get_mut(&mut _frame.resident).unwrap().textures[selected].layout =
        fluxel_rhi::api::ImageLayout::ShaderReadOnly;
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::Deferred { _frame, plan, .. } = &pending.workload else {
        panic!()
    };
    assert!(plan.validate_frame(_frame).is_err());
    let commands = native.command_calls.load(Ordering::SeqCst);
    let mut encoder = device.encoder(device.lanes()[0].id).unwrap();
    assert!(plan.record_geometry(&mut encoder, _frame).is_err());
    assert_eq!(commands, native.command_calls.load(Ordering::SeqCst));
}

#[test]
fn deferred_multi_record_cache_never_accepts_a_missing_current_graph_callback() {
    let (device, native) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = multi_scene(true);
    let good = Counting {
        calls: Cell::new(0),
        declare: true,
    };
    let first =
        block_on(renderer.submit_deferred(&good, snapshot.clone(), view, bounds(), &assets))
            .unwrap();
    block_on(renderer.readback(&first)).unwrap();
    let missing = Counting {
        calls: Cell::new(0),
        declare: false,
    };
    let before = native.command_calls.load(Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.submit_deferred(&missing, snapshot, view, bounds(), &assets)),
        Err(RendererError::DeferredOperation {
            operation: "Graph declaration",
            error: Error::InvalidState
        })
    ));
    assert_eq!(missing.calls.get(), 1);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before);
}
