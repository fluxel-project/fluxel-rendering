//! Typed Deferred deformation admission over the real five-pass Graph.

use super::*;

use std::{collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};

use crate::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, CpuPhase, DeferredCpuPipeline,
    DeferredPositionDeformer, DeformationBinding, FeatureContext, FeatureError, FeatureHandle,
    ForwardAssets, ForwardResources, GeometryAsset, GeometryAttribute, GeometryData,
    GeometryLayout, GeometryStream, PositionDeformation, PreparedFrame, RenderChange,
    RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView, SceneAssetRef,
    TypedFeatureSnapshot,
};

const SOURCE: &str =
    "fn deform_position(p: vec3<f32>, q: vec4<f32>) -> vec3<f32> { return p + q.xyz; }";
const BAD_SOURCE: &str = "fn another_function(p: vec3<f32>) -> vec3<f32> { return p; }";
const UNINITIALIZED_PRODUCT_SOURCE: &str = r#"
fn deform_position(p: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(deformation_product[0u], p.y, p.z);
}
"#;

#[derive(Clone, Copy)]
struct Input([f32; 4]);

struct Deformer {
    handle: FeatureHandle,
    source: &'static str,
    declare: bool,
}

impl DeferredPositionDeformer for Deformer {
    type Input = Input;

    fn handle(&self) -> FeatureHandle {
        self.handle
    }

    fn prepare(
        &self,
        context: &FeatureContext<'_, ForwardResources>,
        input: &TypedFeatureSnapshot<Input>,
        _: &PreparedFrame<'_, ForwardResources>,
    ) -> std::result::Result<PositionDeformation, FeatureError> {
        crate::validate_context(context, input, self.handle)?;
        Ok(PositionDeformation {
            source: fluxel_shader::SourceUnit::new("test-deformation", self.source),
            parameters: input.data().0,
        })
    }

    fn contribute(
        &self,
        pass: &mut crate::DeferredDeformationPass<'_, '_, '_>,
    ) -> std::result::Result<(), crate::PipelineError> {
        pass.read_source()?;
        pass.read_parameters()?;
        if self.declare {
            pass.write_vertices()
        } else {
            Ok(())
        }
    }
}

fn deformation_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.compute = true;
        caps.storage_buffers = true;
        caps.limits.max_color_attachments = 2;
        caps.limits.max_bind_groups = 2;
        caps.limits.max_bindings_per_group = 3;
        caps.limits.max_vertex_buffers = 1;
        caps.limits.max_vertex_attributes = 1;
        caps.limits.max_compute_workgroups = [64, 1, 1];
        caps.limits.max_compute_workgroup_size = [64, 1, 1];
    })
}

fn deformation_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment { name: "tinted_texture".into(), inputs: Vec::new(), output: MaterialType::Vec4, calls: Vec::new(), wgsl: "fn tinted_texture() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into() }],
        "tinted_texture",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }],
        vec![
            MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
        ], BTreeMap::new(), SurfaceOutput::Color,
    )
}

fn deformation_scene() -> (
    RenderScene<ForwardResources>,
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
        Acquire::Producer(producer) => producer
            .commit(deformation_material(), ResidentBytes::new(0))
            .unwrap(),
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
    let mut scene = RenderScene::new().unwrap();
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
    let view = scene.apply(&changes).unwrap().inserted_views()[0];
    (
        scene,
        view,
        ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: vec![texture],
        },
    )
}

fn bind<'a>(
    scene: &'a RenderScene<ForwardResources>,
    snapshot: &'a crate::FrameSnapshot<ForwardResources>,
    source: &'static str,
    values: [f32; 4],
) -> (crate::FeatureIssuer, Deformer, TypedFeatureSnapshot<Input>) {
    let issuer = scene.feature_issuer().unwrap();
    let handle = issuer.issue("deformation-test").unwrap();
    let context = issuer
        .context(snapshot, CpuPhase::Reference, handle)
        .unwrap();
    let mut storage = issuer.storage();
    let input = storage.publish(&context, Input(values)).unwrap();
    (
        issuer,
        Deformer {
            handle,
            source,
            declare: true,
        },
        input,
    )
}

#[allow(clippy::too_many_arguments)]
fn submit(
    renderer: &mut Renderer<WitnessBackend>,
    scene: &RenderScene<ForwardResources>,
    snapshot: crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
    issuer: &crate::FeatureIssuer,
    feature: &Deformer,
    input: &TypedFeatureSnapshot<Input>,
) -> std::result::Result<Submission, RendererError> {
    let _ = scene;
    block_on(renderer.submit_deferred_with_deformation(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        assets,
        DeformationBinding::new(issuer, feature, input),
    ))
}

fn counts(native: &Native) -> (usize, usize, usize) {
    (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    )
}

#[test]
fn deformation_records_real_five_pass_graph_and_retires_two_parameter_owners() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let (issuer, feature, first_input) = bind(&scene, &snapshot, SOURCE, [0.1, 0.0, 0.0, 0.0]);
    let first = submit(
        &mut renderer,
        &scene,
        snapshot.clone(),
        view,
        &assets,
        &issuer,
        &feature,
        &first_input,
    )
    .unwrap();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, feature.handle)
        .unwrap();
    let second_input = TypedFeatureSnapshot::for_context(&context, Input([0.2, 0.0, 0.0, 0.0]));
    let second = submit(
        &mut renderer,
        &scene,
        snapshot,
        view,
        &assets,
        &issuer,
        &feature,
        &second_input,
    )
    .unwrap();
    assert_eq!(renderer.deferred_probe_facts().unwrap().passes, 5);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 10);
    assert_eq!(renderer.pending(), 2);
    block_on(renderer.readback(&first)).unwrap();
    assert_eq!(renderer.pending(), 1);
    block_on(renderer.readback(&second)).unwrap();
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn surface_deformation_records_compute_before_geometry_and_keeps_pending_owners() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    let (scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let (issuer, feature, input) = bind(&scene, &snapshot, SOURCE, [0.1, 0.0, 0.0, 0.0]);
    let pipeline = DeferredCpuPipeline::new(BoundsVisibility);
    let first = block_on(renderer.submit_deferred_surface_with_deformation(
        &pipeline,
        snapshot.clone(),
        view,
        bounds(),
        &assets,
        DeformationBinding::new(&issuer, &feature, &input),
        block_on(surface.acquire()).unwrap(),
    ))
    .expect("first Surface producer/consumer frame");
    let diagnostic = renderer
        .deferred_probe_diagnostic(&first)
        .unwrap()
        .expect("Surface producer owner evidence must be available while pending");
    assert!(diagnostic.contains("deformation_product"));
    assert!(diagnostic.contains("present_step"));
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, feature.handle)
        .unwrap();
    let second_input = TypedFeatureSnapshot::for_context(&context, Input([0.2, 0.0, 0.0, 0.0]));
    let second = block_on(renderer.submit_deferred_surface_with_deformation(
        &pipeline,
        snapshot,
        view,
        bounds(),
        &assets,
        DeformationBinding::new(&issuer, &feature, &second_input),
        block_on(surface.acquire()).unwrap(),
    ))
    .expect("second Surface producer/consumer frame");
    assert_eq!(renderer.deferred_probe_facts().unwrap().passes, 4);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 8);
    assert_eq!(renderer.pending(), 2);
    block_on(renderer.complete(&first)).expect("first present completion");
    assert_eq!(renderer.pending(), 1);
    block_on(renderer.complete(&second)).expect("second present completion");
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn deformation_rejects_foreign_and_stale_typed_inputs_before_device_work() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (mut scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let (issuer, feature, input) = bind(&scene, &snapshot, SOURCE, [0.0; 4]);
    let foreign_scene = RenderScene::<ForwardResources>::new().unwrap();
    let foreign = foreign_scene.feature_issuer().unwrap();
    let before = counts(&native);
    assert!(matches!(
        submit(
            &mut renderer,
            &scene,
            snapshot.clone(),
            view,
            &assets,
            &foreign,
            &feature,
            &input
        ),
        Err(RendererError::DeferredFeature(_))
    ));
    assert_eq!(counts(&native), before);
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).unwrap(),
    ));
    scene.apply(&changes).unwrap();
    assert!(matches!(
        submit(
            &mut renderer,
            &scene,
            scene.snapshot(),
            view,
            &assets,
            &issuer,
            &feature,
            &input
        ),
        Err(RendererError::DeferredFeature(_))
    ));
    assert_eq!(counts(&native), before);
}

#[test]
fn surface_deformation_rejects_foreign_and_stale_inputs_before_gpu_work() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut target = super::surface_tests::surface(&device, [8, 8]);
    let (mut scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let (issuer, feature, input) = bind(&scene, &snapshot, SOURCE, [0.0; 4]);
    let foreign_scene = RenderScene::<ForwardResources>::new().unwrap();
    let foreign = foreign_scene.feature_issuer().unwrap();
    let pipeline = DeferredCpuPipeline::new(BoundsVisibility);
    let before = counts(&native);
    assert!(matches!(
        block_on(renderer.submit_deferred_surface_with_deformation(
            &pipeline,
            snapshot,
            view,
            bounds(),
            &assets,
            DeformationBinding::new(&foreign, &feature, &input),
            block_on(target.acquire()).unwrap(),
        )),
        Err(RendererError::DeferredFeature(_))
    ));
    assert_eq!(counts(&native), before);
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).unwrap(),
    ));
    scene.apply(&changes).unwrap();
    assert!(matches!(
        block_on(renderer.submit_deferred_surface_with_deformation(
            &pipeline,
            scene.snapshot(),
            view,
            bounds(),
            &assets,
            DeformationBinding::new(&issuer, &feature, &input),
            block_on(target.acquire()).unwrap(),
        )),
        Err(RendererError::DeferredFeature(_))
    ));
    assert_eq!(counts(&native), before);
}

#[test]
fn deformation_refuses_capabilities_parameters_and_kernel_before_allocation() {
    for adjust in [0_u8, 1, 2, 3] {
        let (device, native) = fixture_with_caps(Script::AcceptedExact, |caps| {
            caps.compute = false;
            caps.storage_buffers = false;
            caps.limits.max_color_attachments = 2;
            caps.limits.max_bind_groups = 2;
            caps.limits.max_bindings_per_group = 3;
            caps.limits.max_vertex_buffers = 1;
            caps.limits.max_vertex_attributes = 1;
            if adjust == 1 {
                caps.compute = true;
            }
            if adjust == 2 || adjust == 3 {
                caps.compute = true;
                caps.storage_buffers = true;
                caps.limits.max_compute_workgroups = [64, 1, 1];
                caps.limits.max_compute_workgroup_size = [64, 1, 1];
            }
        });
        let mut renderer = Renderer::new(device).unwrap();
        let (scene, view, assets) = deformation_scene();
        let snapshot = scene.snapshot();
        let (issuer, feature, input) = bind(
            &scene,
            &snapshot,
            if adjust == 3 { BAD_SOURCE } else { SOURCE },
            if adjust == 2 {
                [f32::NAN, 0.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            },
        );
        let before = counts(&native);
        assert!(
            submit(
                &mut renderer,
                &scene,
                snapshot,
                view,
                &assets,
                &issuer,
                &feature,
                &input
            )
            .is_err()
        );
        assert_eq!(counts(&native), before);
    }
}

#[test]
fn deformation_missing_feature_product_declaration_refuses_before_mapping_or_recording() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let (issuer, mut feature, input) = bind(&scene, &snapshot, SOURCE, [0.0; 4]);
    feature.declare = false;
    let before = counts(&native);
    assert!(
        submit(
            &mut renderer,
            &scene,
            snapshot,
            view,
            &assets,
            &issuer,
            &feature,
            &input
        )
        .is_err()
    );
    assert_eq!(native.map_calls.load(Ordering::SeqCst), before.1);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before.2);
}

#[test]
fn deformation_kernel_cannot_read_its_uninitialized_output_before_allocation() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let (issuer, feature, input) = bind(&scene, &snapshot, UNINITIALIZED_PRODUCT_SOURCE, [0.0; 4]);
    let before = counts(&native);

    assert!(matches!(
        submit(
            &mut renderer,
            &scene,
            snapshot,
            view,
            &assets,
            &issuer,
            &feature,
            &input
        ),
        Err(RendererError::DeformationShader(_))
    ));
    assert_eq!(counts(&native), before);
}

#[path = "deferred_deformation_owners.rs"]
mod owner_tests;
#[path = "deferred_deformation_pbr.rs"]
mod pbr_tests;
#[path = "deferred_deformation_surface_owners.rs"]
mod surface_owner_tests;
