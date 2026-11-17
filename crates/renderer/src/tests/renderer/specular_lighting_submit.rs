//! Real Forward and Deferred submission witnesses for specular IBL.

use super::*;

use std::{collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{Device, VertexFormat, VertexStepMode};

use crate::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, DeferredCpuPipeline,
    EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout,
    GeometryStream, LinearSpecularLod, RenderBounds, RenderChange, RenderChangeSet, RenderRecord,
    RenderScene, RenderTransform, RenderView, Renderer, SceneAssetRef, ShadingCamera,
    SpecularIblImage, SpecularIblMip,
};

fn bounds() -> RenderBounds {
    RenderBounds::new([-1.0; 3], [1.0; 3]).unwrap()
}
fn specular_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_color_attachments = 2;
        caps.limits.max_bind_groups = 3;
        caps.limits.max_bindings_per_group = 7;
        caps.limits.max_vertex_buffers = 1;
        caps.limits.max_vertex_attributes = 2;
    })
}
fn scene(
    with_camera: bool,
) -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    let gs = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let gh = gs.create().unwrap();
    let mut vertices = vec![0; 72];
    for offset in [20, 44, 68] {
        vertices[offset..offset + 4].copy_from_slice(&1.0_f32.to_le_bytes());
    }
    let g = match gs.acquire(&gh).unwrap() {
        Acquire::Producer(p) => p
            .commit(
                GeometryData {
                    layout: GeometryLayout {
                        streams: vec![GeometryStream {
                            stride: 24,
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
                            ],
                        }],
                    },
                    streams: vec![vertices],
                    vertices: 0..3,
                },
                ResidentBytes::new(72),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let ms = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let mh = ms.create().unwrap();
    let d=MaterialDefinition::new(vec![MaterialFragment{name:"base".into(),inputs:vec![],output:MaterialType::Vec4,calls:vec![],wgsl:"fn base() -> vec4<f32>{return textureLoad(material_texture,vec2<i32>(0),0)*tint;}".into()},MaterialFragment{name:"props".into(),inputs:vec![],output:MaterialType::Vec2,calls:vec![],wgsl:"fn props()->vec2<f32>{return vec2<f32>(0.5);}".into()}],"base",vec![MaterialParameter{name:"tint".into(),ty:MaterialType::Vec4,resource:"tint_u".into()}],vec![MaterialResource{name:"tint_u".into(),kind:MaterialResourceKind::UniformVec4,location:BindingLocation{group:0,binding:0}},MaterialResource{name:"material_texture".into(),kind:MaterialResourceKind::Texture2dF32,location:BindingLocation{group:0,binding:1}}],BTreeMap::new(),SurfaceOutput::MetallicRoughness{properties:"props".into()});
    let m = match ms.acquire(&mh).unwrap() {
        Acquire::Producer(p) => p.commit(d, ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    };
    let ts = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let th = ts.create().unwrap();
    let t = match ts.acquire(&th).unwrap() {
        Acquire::Producer(p) => p
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    encoding: crate::TextureEncoding::Linear,
                    mip_levels: 1,
                    rgba8: vec![255; 4],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let mut s = RenderScene::new().unwrap();
    let mut c = RenderChangeSet::new();
    let view = RenderView::new(RenderTransform::identity(), [8, 8]).unwrap();
    c.push(RenderChange::InsertView(if with_camera {
        view.with_shading_camera(ShadingCamera::perspective([0., 0., 2.]).unwrap())
    } else {
        view
    }));
    c.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&g),
            material: SceneAssetRef::from_snapshot(&m),
            textures: vec![crate::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&t),
            )],
            sampler: None,
            tint: [1.; 4],
        },
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    let r = s.apply(&c).unwrap();
    (
        s.snapshot(),
        r.inserted_views()[0],
        ForwardAssets {
            geometry: vec![g],
            materials: vec![m],
            textures: vec![t],
        },
    )
}

fn image(extent: [u32; 2]) -> SpecularIblImage {
    let texels = usize::try_from(extent[0] * extent[1]).unwrap();
    let pixel: Vec<_> = [0x3c00_u16; 4]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    SpecularIblImage {
        extent,
        rgba16f_le: pixel.repeat(texels),
    }
}

fn snapshot()
-> fluxel_assets::AssetSnapshot<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData> {
    let store = AssetStore::<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, ()>::new();
    let handle = store.create().unwrap();
    match store.acquire(&handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                EnvironmentSpecularIblData {
                    radiance_mips: vec![
                        SpecularIblMip {
                            extent: [2, 2],
                            rgba16f_le: image([2, 2]).rgba16f_le,
                        },
                        SpecularIblMip {
                            extent: [1, 1],
                            rgba16f_le: image([1, 1]).rgba16f_le,
                        },
                    ],
                    brdf_lut: image([2, 2]),
                    lod: LinearSpecularLod {
                        scale: 1.0,
                        offset: 0.0,
                    },
                },
                ResidentBytes::new(40),
            )
            .unwrap(),
        _ => unreachable!(),
    }
}

#[test]
fn specular_ibl_submits_real_pbr_forward_and_deferred_frames() {
    let (device, native) = specular_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    renderer
        .set_environment_specular_ibl(Some(snapshot()))
        .unwrap();
    let (scene, view, assets) = scene(true);
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.clone(),
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let deferred = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    assert_eq!(renderer.pending(), 2);
    assert!(native.command_calls.load(Ordering::SeqCst) > 0);
    block_on(renderer.readback(&forward)).unwrap();
    block_on(renderer.readback(&deferred)).unwrap();
}

#[test]
fn specular_ibl_refuses_missing_shading_camera_before_gpu_allocation() {
    let (device, native) = specular_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    renderer
        .set_environment_specular_ibl(Some(snapshot()))
        .unwrap();
    let (scene, view, assets) = scene(false);
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.created_textures.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    );
    assert!(
        block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            scene,
            view,
            bounds(),
            &assets,
        ))
        .is_err()
    );
    assert_eq!(before.0, native.created_buffers.load(Ordering::SeqCst));
    assert_eq!(before.1, native.created_textures.load(Ordering::SeqCst));
    assert_eq!(before.2, native.command_calls.load(Ordering::SeqCst));
}

#[test]
fn forward_cached_plan_refuses_mutated_actual_specular_radiance_owner() {
    let (device, native) = specular_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    renderer
        .set_environment_specular_ibl(Some(snapshot()))
        .unwrap();
    let (scene, view, assets) = scene(true);
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::Forward { _frame, .. } = &mut pending.workload else {
        panic!("Forward pending")
    };
    _frame.specular.as_mut().unwrap().radiance.desc.usage = TextureUsage::COPY_DST;
    let device = renderer.device.clone();
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::Forward { _frame, plan, .. } = &pending.workload else {
        panic!("Forward pending")
    };
    assert!(plan.validate_frame(_frame).is_err());
    let before = native.command_calls.load(Ordering::SeqCst);
    let mut encoder = device.encoder(device.lanes()[0].id).unwrap();
    assert!(plan.record_raster(&mut encoder, _frame).is_err());
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before);
}

#[test]
fn forward_cached_plan_refuses_mutated_actual_specular_radiance_state() {
    let (device, native) = specular_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    renderer
        .set_environment_specular_ibl(Some(snapshot()))
        .unwrap();
    let (scene, view, assets) = scene(true);
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::Forward { _frame, .. } = &mut pending.workload else {
        panic!()
    };
    _frame.specular.as_mut().unwrap().radiance.layout = ImageLayout::General;
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::Forward { _frame, plan, .. } = &pending.workload else {
        panic!()
    };
    assert!(plan.validate_frame(_frame).is_err());
    let before = native.command_calls.load(Ordering::SeqCst);
    let mut encoder = renderer
        .device
        .encoder(renderer.device.lanes()[0].id)
        .unwrap();
    assert!(plan.record_raster(&mut encoder, _frame).is_err());
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before);
}

#[test]
fn forward_cached_plan_refuses_an_actual_foreign_specular_owner_before_commands() {
    let (device, native) = specular_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    renderer
        .set_environment_specular_ibl(Some(snapshot()))
        .unwrap();
    let (scene, view, assets) = scene(true);
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let (foreign_device, _) = specular_device(Script::AcceptedExact);
    let foreign = crate::residency::SpecularEnvironmentFrameOwner::allocate(
        &foreign_device,
        snapshot().value(),
        &[0; 112],
    )
    .unwrap();
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::Forward { _frame, .. } = &mut pending.workload else {
        panic!("Forward pending")
    };
    _frame.specular = Some(foreign);
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::Forward { _frame, plan, .. } = &pending.workload else {
        panic!("Forward pending")
    };
    assert!(plan.validate_frame(_frame).is_err());
    let before = native.command_calls.load(Ordering::SeqCst);
    let mut encoder = renderer
        .device
        .encoder(renderer.device.lanes()[0].id)
        .unwrap();
    assert!(plan.record_raster(&mut encoder, _frame).is_err());
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before);
}

#[test]
fn deferred_cached_plan_refuses_mutated_actual_specular_radiance_owner() {
    let (device, native) = specular_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    renderer
        .set_environment_specular_ibl(Some(snapshot()))
        .unwrap();
    let (scene, view, assets) = scene(true);
    let submission = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::Deferred { _frame, .. } = &mut pending.workload else {
        panic!()
    };
    _frame.specular.as_mut().unwrap().radiance.desc.usage = TextureUsage::COPY_DST;
    let device = renderer.device.clone();
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::Deferred { _frame, plan, .. } = &pending.workload else {
        panic!()
    };
    assert!(plan.validate_frame(_frame).is_err());
    let before = native.command_calls.load(Ordering::SeqCst);
    let mut encoder = device.encoder(device.lanes()[0].id).unwrap();
    assert!(plan.record_lighting(&mut encoder, _frame).is_err());
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before);
}
