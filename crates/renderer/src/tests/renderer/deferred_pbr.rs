//! Real PBR Forward/Deferred lifecycle and transform-footprint tests.

use super::*;

use std::{collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialDefinition, MaterialFragment,
    MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{Device, Error, PlanId, VertexFormat, VertexStepMode};

use crate::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, DeferredCpuPipeline,
    EnvironmentIrradianceAsset, EnvironmentIrradianceData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, FramePipeline, GeometryAsset, GeometryAttribute, GeometryData,
    GeometryLayout, GeometryStream, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, RendererError, SceneAssetRef, prepare_forward,
};

fn pbr_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_color_attachments = 2;
        caps.limits.max_bind_groups = 2;
        caps.limits.max_bindings_per_group = 3;
        caps.limits.max_vertex_buffers = 2;
        caps.limits.max_vertex_attributes = 4;
    })
}

pub(super) fn environment_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_color_attachments = 2;
        caps.limits.max_bind_groups = 3;
        caps.limits.max_bindings_per_group = 3;
        caps.limits.max_vertex_buffers = 2;
        caps.limits.max_vertex_attributes = 4;
    })
}

pub(super) fn irradiance_snapshot(
    bytes: [u8; 8],
) -> fluxel_assets::AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData> {
    let store = AssetStore::<EnvironmentIrradianceAsset, EnvironmentIrradianceData, ()>::new();
    let handle = store.create().unwrap();
    match store.acquire(&handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                EnvironmentIrradianceData {
                    extent: [2, 1],
                    rgba8: bytes.to_vec(),
                },
                ResidentBytes::new(8),
            )
            .unwrap(),
        _ => unreachable!(),
    }
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "base_color".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn base_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
            },
            MaterialFragment {
                name: "surface_properties".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec2,
                calls: Vec::new(),
                wgsl: "fn surface_properties() -> vec2<f32> { return vec2<f32>(0.25, 0.6); }".into(),
            },
        ],
        "base_color",
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
            properties: "surface_properties".into(),
        },
    )
}

pub(super) fn geometry(include_normal: bool, normal: [f32; 3]) -> GeometryData {
    let mut bytes = Vec::new();
    for position in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for value in position {
            bytes.extend(value.to_le_bytes());
        }
        if include_normal {
            for value in normal {
                bytes.extend(value.to_le_bytes());
            }
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: if include_normal { 24 } else { 12 },
                step_mode: VertexStepMode::Vertex,
                attributes: if include_normal {
                    vec![
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
                    ]
                } else {
                    vec![GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    }]
                },
            }],
        },
        streams: vec![bytes],
        vertices: 0..3,
    }
}

pub(super) fn scene_and_assets(
    geometry_data: GeometryData,
    transform: RenderTransform,
) -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    scene_and_assets_with_material(geometry_data, transform, material())
}

fn scene_and_assets_with_material(
    geometry_data: GeometryData,
    transform: RenderTransform,
    definition: MaterialDefinition,
) -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    let geometries = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometries.create().unwrap();
    let geometry_bytes =
        u64::try_from(geometry_data.streams.iter().map(Vec::len).sum::<usize>()).unwrap();
    let geometry = match geometries.acquire(&geometry_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(geometry_data, ResidentBytes::new(geometry_bytes))
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
                    rgba8: vec![128, 96, 64, 255],
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
        transform,
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

fn counts(native: &Native) -> (usize, usize, usize) {
    (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    )
}

fn uv0_geometry() -> GeometryData {
    let mut geometry = geometry(true, [0.0, 0.0, 1.0]);
    let mut bytes = Vec::new();
    for vertex in 0..3 {
        bytes.extend_from_slice(&geometry.streams[0][vertex * 24..vertex * 24 + 24]);
        bytes.extend_from_slice(&[0.0f32.to_le_bytes(), 0.0f32.to_le_bytes()].concat());
    }
    geometry.streams[0] = bytes;
    geometry.layout.streams[0].stride = 32;
    geometry.layout.streams[0]
        .attributes
        .push(GeometryAttribute {
            semantic: "UV0".into(),
            offset: 24,
            format: VertexFormat::Float32(2),
        });
    geometry
}

fn uv0_geometry_in_its_own_stream() -> GeometryData {
    let combined = uv0_geometry();
    let mut positions_normals = Vec::new();
    let mut uv0 = Vec::new();
    for vertex in 0..3 {
        let start = vertex * 32;
        positions_normals.extend_from_slice(&combined.streams[0][start..start + 24]);
        uv0.extend_from_slice(&combined.streams[0][start + 24..start + 32]);
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![
                GeometryStream {
                    stride: 24,
                    step_mode: VertexStepMode::Vertex,
                    attributes: combined.layout.streams[0].attributes[..2].to_vec(),
                },
                GeometryStream {
                    stride: 8,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![GeometryAttribute {
                        semantic: "UV0".into(),
                        offset: 0,
                        format: VertexFormat::Float32(2),
                    }],
                },
            ],
        },
        streams: vec![positions_normals, uv0],
        vertices: 0..3,
    }
}

fn uv0_node_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "sampled".into(),
                inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn sampled(uv: vec2<f32>) -> vec4<f32> { return textureSample(material_texture, material_sampler, uv) * tint; }".into(),
            },
            MaterialFragment {
                name: "properties".into(),
                inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
                output: MaterialType::Vec2,
                calls: Vec::new(),
                wgsl: "fn properties(uv: vec2<f32>) -> vec2<f32> { return vec2<f32>(0.25 + uv.x * 0.0, 0.6); }".into(),
            },
        ],
        "unused",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }],
        vec![
            MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "material_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
        ], BTreeMap::new(), SurfaceOutput::MetallicRoughnessUv0 { properties: "properties".into() },
    ).with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "sampled".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(1) })
}

fn uv0_normal_node_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "sampled".into(),
                inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn sampled(uv: vec2<f32>) -> vec4<f32> { return textureSample(material_texture, material_sampler, uv) * tint; }".into(),
            },
            MaterialFragment {
                name: "properties".into(),
                inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
                output: MaterialType::Vec2,
                calls: Vec::new(),
                wgsl: "fn properties(uv: vec2<f32>) -> vec2<f32> { return vec2<f32>(0.25, 0.6); }".into(),
            },
        ],
        "unused",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }],
        vec![
            MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "material_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
        ],
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughnessUv0Normal { properties: "properties".into(), normal: "sample_normal".into() },
    )
    .with_code_node(fluxel_material::MaterialCodeNode {
        name: "sample_normal".into(),
        inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
        output: MaterialType::Vec3,
        calls: Vec::new(),
        source: fluxel_shader::SourceUnit::new(
            "renderer-test:sample-normal",
            "fn sample_normal(uv: vec2<f32>) -> vec3<f32> { return textureSample(material_texture, material_sampler, uv).xyz; }",
        ),
        resources: Vec::new(),
    })
    .with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "sampled".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(1) })
}

fn uv0_tangent_geometry(tangent: [f32; 4]) -> GeometryData {
    let mut bytes = Vec::new();
    for position in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for value in position
            .into_iter()
            .chain([0.0, 0.0, 1.0])
            .chain([0.0, 0.0])
            .chain(tangent)
        {
            bytes.extend(value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 48,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
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
                    GeometryAttribute {
                        semantic: "TANGENT".into(),
                        offset: 32,
                        format: VertexFormat::Float32(4),
                    },
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..3,
    }
}

fn pbr_frame(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: &crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
) -> crate::residency::DeferredFrame<WitnessBackend> {
    let selected = ForwardCpuPipeline::new(BoundsVisibility)
        .prepare(snapshot, view, bounds())
        .unwrap();
    let inputs =
        prepare_forward(&selected, assets, &mut fluxel_shader::ShaderComposer::new()).unwrap();
    let [input] = inputs.as_slice() else {
        panic!("one PBR draw")
    };
    block_on(renderer.prepare_deferred_test_frame(input)).unwrap()
}

#[test]
fn pbr_material_submits_through_forward_and_deferred_with_192_byte_transform() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let deferred = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    assert_eq!(renderer.deferred_probe_facts().unwrap().passes, 5);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 8);
    block_on(renderer.readback(&forward)).unwrap();
    block_on(renderer.readback(&deferred)).unwrap();
}

#[test]
fn uv0_node_material_submits_through_forward_and_deferred() {
    let (device, _) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets_with_material(
        uv0_geometry(),
        RenderTransform::identity(),
        uv0_node_material(),
    );
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .expect("UV0 node Forward submission");
    let deferred = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect("UV0 node Deferred submission");
    block_on(renderer.readback(&forward)).expect("Forward readback");
    block_on(renderer.readback(&deferred)).expect("Deferred readback");
}

#[test]
fn uv0_node_material_with_a_separate_uv_stream_submits_through_both_consumers() {
    let (device, _) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets_with_material(
        uv0_geometry_in_its_own_stream(),
        RenderTransform::identity(),
        uv0_node_material(),
    );
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .expect("separate-UV Forward submission");
    let deferred = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect("separate-UV Deferred submission");
    block_on(renderer.readback(&forward)).expect("Forward readback");
    block_on(renderer.readback(&deferred)).expect("Deferred readback");
}

#[test]
fn uv0_node_material_without_uv0_refuses_before_backend_work() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets_with_material(
        geometry(true, [0.0, 0.0, 1.0]),
        RenderTransform::identity(),
        uv0_node_material(),
    );
    let before = counts(&native);
    for deferred in [false, true] {
        let result = if deferred {
            block_on(renderer.submit_deferred(
                &DeferredCpuPipeline::new(BoundsVisibility),
                snapshot.clone(),
                view,
                bounds(),
                &assets,
            ))
        } else {
            block_on(renderer.submit_forward(
                &ForwardCpuPipeline::new(BoundsVisibility),
                snapshot.clone(),
                view,
                bounds(),
                &assets,
            ))
        };
        assert!(matches!(result, Err(RendererError::ForwardPreparation(_))));
        assert_eq!(counts(&native), before);
    }
}

#[test]
fn pbr_deferred_surface_consumes_actual_lease_format_and_exact_transform_owner() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    for format in [TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm] {
        let mut config = super::surface_tests::config([8, 8]);
        config.format = format;
        block_on(surface.configure(&config)).expect("configured target format");
        let submission = block_on(renderer.submit_deferred_surface(
            &DeferredCpuPipeline::new(BoundsVisibility),
            snapshot.clone(),
            view,
            bounds(),
            &assets,
            block_on(surface.acquire()).expect("image"),
        ))
        .expect("PBR Surface submission");
        let pending = renderer.pending_for(&submission).expect("pending owner");
        let crate::renderer::PendingWork::DeferredSurface { _frame, .. } = &pending.workload else {
            panic!("must retain an actual Deferred Surface owner");
        };
        assert_eq!(_frame.draws[0].transform_bytes, 192);
        assert_eq!(
            _frame.draws[0].transform_range.end - _frame.draws[0].transform_range.start,
            192
        );
        assert_eq!(_frame.color_format, format);
        block_on(renderer.complete(&submission)).expect("Surface completion");
    }
    let formats = native
        .pipeline_color_formats
        .lock()
        .expect("pipeline observations");
    assert!(formats.contains(&vec![TextureFormat::Rgba8Unorm]));
    assert!(
        formats.contains(&vec![TextureFormat::Bgra8Unorm]),
        "a different lease format must realize a matching lighting executable"
    );
    assert_eq!(
        renderer.compilations(),
        2,
        "a cached surface plan must not bind a lease with a different actual format"
    );
}

#[path = "deferred_pbr_failures.rs"]
mod failures;
