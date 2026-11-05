//! Real PBR Forward/Deferred lifecycle and transform-footprint tests.

use super::*;

use std::{collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{Device, Error, PlanId, VertexFormat, VertexStepMode};

use crate::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, DeferredCpuPipeline, ForwardAssets,
    ForwardCpuPipeline, ForwardResources, FramePipeline, GeometryAsset, GeometryAttribute,
    GeometryData, GeometryLayout, GeometryStream, RenderChange, RenderChangeSet, RenderRecord,
    RenderScene, RenderTransform, RenderView, RendererError, SceneAssetRef, prepare_forward,
};

fn pbr_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_color_attachments = 2;
        caps.limits.max_bind_groups = 2;
        caps.limits.max_bindings_per_group = 3;
        caps.limits.max_vertex_buffers = 1;
        caps.limits.max_vertex_attributes = 2;
    })
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

fn geometry(include_normal: bool, normal: [f32; 3]) -> GeometryData {
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

fn scene_and_assets(
    geometry_data: GeometryData,
    transform: RenderTransform,
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
        Acquire::Producer(producer) => producer.commit(material(), ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    };
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let texture_handle = textures.create().unwrap();
    let texture = match textures.acquire(&texture_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
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
            texture: SceneAssetRef::from_snapshot(&texture),
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
fn pbr_material_submits_through_forward_and_deferred_with_128_byte_transform() {
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
    assert_eq!(renderer.deferred_probe_facts().unwrap().passes, 4);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 7);
    block_on(renderer.readback(&forward)).unwrap();
    block_on(renderer.readback(&deferred)).unwrap();
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
        assert_eq!(_frame.draw.transform_bytes, 128);
        assert_eq!(
            _frame.draw.transform_range.end - _frame.draw.transform_range.start,
            128
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
}

#[test]
fn pbr_compiled_plan_refuses_short_or_retyped_transform_footprints_before_work() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let mut frame = pbr_frame(&mut renderer, &snapshot, view, &assets);
    assert_eq!(frame.draw.transform_bytes, 128);
    assert_eq!(
        frame.draw.transform_range.end - frame.draw.transform_range.start,
        128
    );
    let plan = crate::graph::DeferredPlan::compile(
        &device,
        PlanId {
            device: device.id(),
            ordinal: 72_001,
        },
        &frame,
    )
    .unwrap();
    let before = counts(&native);
    frame.draw.transform_range.end = frame.draw.transform_range.start + 64;
    assert!(matches!(
        plan.validate_frame(&frame),
        Err(Error::InvalidRange)
    ));
    assert_eq!(counts(&native), before);

    frame.draw.transform_bytes = 64;
    assert!(matches!(
        plan.validate_frame(&frame),
        Err(Error::InvalidState)
    ));
    assert_eq!(counts(&native), before);
}

#[test]
fn pbr_surface_plan_refuses_short_and_retagged_actual_transform_ranges() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let pipeline = DeferredCpuPipeline::new(BoundsVisibility);
    let selected = pipeline
        .prepare(&snapshot, view, bounds())
        .expect("selected");
    let inputs = prepare_forward(
        &selected,
        &assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .expect("actual PBR input");
    let mut frame = block_on(renderer.prepare_deferred_surface_test_frame(&inputs[0]))
        .expect("actual PBR Surface owners");
    let declaration =
        crate::graph::DeferredPlan::declare_surface_with(&device, &frame, |context| {
            pipeline.contribute_deferred(&selected, context)
        })
        .expect("declared Surface");
    let plan = crate::graph::DeferredPlan::compile_declared(
        &device,
        PlanId {
            device: device.id(),
            ordinal: 81_900,
        },
        declaration,
    )
    .expect("Surface plan");
    plan.validate_surface_frame(&frame)
        .expect("legal frame before mutation");
    let before = counts(&native);
    frame.draw.transform_range.end = frame.draw.transform_range.start + 64;
    assert!(plan.validate_surface_frame(&frame).is_err());
    assert_eq!(counts(&native), before);
    frame.draw.transform_bytes = 64;
    assert!(plan.validate_surface_frame(&frame).is_err());
    assert_eq!(counts(&native), before);
}

#[test]
fn pbr_missing_invalid_or_singular_normals_refuse_before_backend_allocation() {
    let cases = [
        (
            geometry(false, [0.0, 0.0, 1.0]),
            RenderTransform::identity(),
        ),
        (
            geometry(true, [f32::NAN, 0.0, 1.0]),
            RenderTransform::identity(),
        ),
        (geometry(true, [0.0; 3]), RenderTransform::identity()),
        (
            geometry(true, [f32::INFINITY, 0.0, 1.0]),
            RenderTransform::identity(),
        ),
        (
            geometry(true, [0.0, 0.0, 1.0]),
            RenderTransform::from_columns([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            .unwrap(),
        ),
    ];
    for (geometry, transform) in cases {
        let (device, native) = pbr_device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let (snapshot, view, assets) = scene_and_assets(geometry, transform);
        let before = counts(&native);
        assert!(matches!(
            block_on(renderer.submit_forward(
                &ForwardCpuPipeline::new(BoundsVisibility),
                snapshot.clone(),
                view,
                bounds(),
                &assets,
            )),
            Err(RendererError::ForwardPreparation(_))
        ));
        assert_eq!(counts(&native), before);
        assert!(matches!(
            block_on(renderer.submit_deferred(
                &DeferredCpuPipeline::new(BoundsVisibility),
                snapshot,
                view,
                bounds(),
                &assets,
            )),
            Err(RendererError::ForwardPreparation(_))
        ));
        assert_eq!(counts(&native), before);
    }
}
