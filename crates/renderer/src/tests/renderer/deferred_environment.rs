//! Environment-irradiance ownership and Deferred-consumer boundaries.

use super::pbr_tests::{environment_device, geometry, irradiance_snapshot, scene_and_assets};
use super::*;

use std::sync::atomic::Ordering;

use crate::{
    BoundsVisibility, DeferredCpuPipeline, EnvironmentIrradianceData, ForwardCpuPipeline,
    RenderTransform,
};
use fluxel_rhi::api::Error;

#[test]
fn forward_environment_snapshot_uploads_and_remains_owned_until_readback() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .unwrap();
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([
            11, 12, 13, 255, 14, 15, 16, 255,
        ])))
        .unwrap();
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    assert!(renderer.pending_for(&submission).is_ok());
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    assert!(
        native.mapped_bytes.lock().unwrap().iter().any(|bytes| {
            bytes.len() == 256 && bytes[..8] == [11, 12, 13, 255, 14, 15, 16, 255]
        })
    );
    assert_eq!(native.texture_uploads.lock().unwrap().len(), 2);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.readback(&submission)).unwrap();
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 1);
}

#[test]
fn environment_owner_rejects_mutated_allocation_or_sampler_facts() {
    let (device, _) = environment_device(Script::AcceptedExact);
    let mut owner = crate::residency::EnvironmentFrameOwner::allocate(
        &device,
        &EnvironmentIrradianceData {
            extent: [2, 1],
            rgba8: vec![1, 2, 3, 255, 4, 5, 6, 255],
        },
    )
    .unwrap();
    owner.filterable = false;
    assert_eq!(
        owner.validate_for_sampling(device.id()),
        Err(Error::InvalidState)
    );
    owner.filterable = true;
    owner.sampler_desc.anisotropy = 2;
    assert_eq!(
        owner.validate_for_sampling(device.id()),
        Err(Error::InvalidState)
    );
}

#[test]
fn forward_environment_rejects_non_filterable_owner_before_upload() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .unwrap();
    native.texture_filterable.store(false, Ordering::SeqCst);
    assert!(
        block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
        ))
        .is_err()
    );
    assert_eq!(native.map_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn deferred_environment_snapshot_uploads_reads_and_retires_with_completion() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([9, 8, 7, 255, 6, 5, 4, 255])))
        .unwrap();
    let submission = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 2);
    assert!(
        native
            .mapped_bytes
            .lock()
            .unwrap()
            .iter()
            .any(|bytes| { bytes.len() == 256 && bytes[..8] == [9, 8, 7, 255, 6, 5, 4, 255] })
    );
    assert_eq!(native.texture_uploads.lock().unwrap().len(), 2);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    assert!(renderer.deferred_probe_facts().unwrap().texture_bindings > 6);
    block_on(renderer.readback(&submission)).unwrap();
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 2);
}

#[test]
fn deferred_none_keeps_legacy_lighting_route_without_environment_owner() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let submission = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    assert_eq!(native.texture_uploads.lock().unwrap().len(), 1);
    block_on(renderer.readback(&submission)).unwrap();
}

#[test]
fn deferred_environment_rejects_non_filterable_actual_owner_before_mapping() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .unwrap();
    native.texture_filterable.store(false, Ordering::SeqCst);
    assert!(
        block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
        ))
        .is_err()
    );
    assert_eq!(native.map_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn deferred_surface_environment_uploads_and_retires_after_present_completion() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .unwrap();
    let image = block_on(surface.acquire()).unwrap();
    let submission = block_on(renderer.submit_deferred_surface(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        image,
    ))
    .unwrap();
    assert_eq!(renderer.pending(), 1);
    assert!(
        native
            .mapped_bytes
            .lock()
            .unwrap()
            .iter()
            .any(|bytes| { bytes.len() == 256 && bytes[..8] == [1, 2, 3, 255, 4, 5, 6, 255] })
    );
    assert_eq!(native.texture_uploads.lock().unwrap().len(), 2);
    block_on(renderer.complete(&submission)).unwrap();
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn deferred_surface_environment_rejects_nonfilterable_owner_before_mapping() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .unwrap();
    native.texture_filterable.store(false, Ordering::SeqCst);
    let image = block_on(surface.acquire()).unwrap();
    assert!(
        block_on(renderer.submit_deferred_surface(
            &DeferredCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
            image,
        ))
        .is_err()
    );
    assert_eq!(native.map_calls.load(Ordering::SeqCst), 0);
}
