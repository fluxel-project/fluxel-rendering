//! Ordinary material sampler ownership on the Deferred surface consumer.

use super::super::code_node_tests::{sampled_material, scene_and_assets};
use super::*;

#[test]
fn deferred_surface_sampler_stays_owned_until_present_completion() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let mut target = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let submission = submit_surface(
        &mut renderer,
        snapshot,
        view,
        &assets,
        block_on(target.acquire()).expect("surface image"),
    )
    .expect("sampled Deferred surface submission");

    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 2);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.complete(&submission)).expect("present completion");
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 2);
}

#[test]
fn deferred_surface_sampler_bind_group_rejection_releases_owner() {
    let (device, native) = deferred_device(Script::SamplerBindGroupRejected);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let mut target = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let error = submit_surface(
        &mut renderer,
        snapshot,
        view,
        &assets,
        block_on(target.acquire()).expect("surface image"),
    )
    .expect_err("sampler bind group rejection must release the unsubmitted surface owner");

    assert!(matches!(error, RendererError::Rhi(Error::InvalidState)));
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 2);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 2);
}

#[test]
fn deferred_surface_refuses_a_mutated_cached_sampler_texture_owner() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let mut target = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets(sampled_material());
    let selected = crate::ForwardCpuPipeline::new(crate::BoundsVisibility)
        .prepare(&snapshot, view, bounds())
        .expect("selected input");
    let inputs = crate::prepare_forward(
        &selected,
        &assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .expect("prepared input");
    let texture_key = crate::residency::ForwardResidentCache::<WitnessBackend>::texture_key(
        &inputs[0],
        device.id(),
    )
    .expect("ordinary Deferred fixture has an asset texture");

    let first = submit_surface(
        &mut renderer,
        snapshot.clone(),
        view,
        &assets,
        block_on(target.acquire()).expect("first surface image"),
    )
    .expect("initial sampled Deferred surface submission");
    block_on(renderer.complete(&first)).expect("initial present completion");
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 2);

    // Mutate the saved allocation fact of the cached owner, not the expected
    // Graph declaration or the capability-admission descriptor.
    let mut cached = renderer
        .deferred_residency
        .texture(texture_key)
        .expect("completed texture cache owner");
    cached.filterable = false;
    renderer.deferred_residency.clear();
    assert!(
        renderer
            .deferred_residency
            .publish_texture(texture_key, &cached)
    );

    let error = submit_surface(
        &mut renderer,
        snapshot,
        view,
        &assets,
        block_on(target.acquire()).expect("cached surface image"),
    )
    .expect_err("mutated cached owner must be refused before sampler binding");
    assert!(matches!(error, RendererError::Rhi(Error::InvalidState)));
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 4);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 4);
}
