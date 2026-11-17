use super::*;

use std::sync::Arc;

use crate::{
    BoundsVisibility, ForwardCpuPipeline,
    renderer::forward::validate_sampled_texture_owner,
    residency::{ForwardResidentBuffers, ForwardTextureInput, plan_forward_resident},
};

#[test]
fn texture_load_paths_accept_nonfilterable_owners_in_both_pipelines() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    native.texture_filterable.store(false, Ordering::SeqCst);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .expect("unfiltered Forward textureLoad");
    block_on(renderer.readback(&forward)).expect("Forward completion");
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 0);
    // Deferred scene-color filtering stays supported; change only the
    // completed material owner's actual filterability fact below.
    native.texture_filterable.store(true, Ordering::SeqCst);
    let pipeline = crate::DeferredCpuPipeline::new(BoundsVisibility);
    let cold =
        block_on(renderer.submit_deferred(&pipeline, snapshot.clone(), view, bounds(), &assets))
            .expect("publish Deferred textureLoad material");
    block_on(renderer.readback(&cold)).expect("cold Deferred completion");
    let key = crate::residency::ForwardTextureCacheKey::from_snapshot(
        &assets.textures[0],
        renderer.device.id(),
    );
    let mut owner = renderer
        .deferred_residency
        .texture(key)
        .expect("cached material owner");
    owner.filterable = false;
    renderer.deferred_residency = Default::default();
    assert!(renderer.deferred_residency.publish_texture(key, &owner));
    let deferred = block_on(renderer.submit_deferred(&pipeline, snapshot, view, bounds(), &assets))
        .expect("unfiltered Deferred textureLoad");
    let pending = renderer.pending_for(&deferred).expect("Deferred pending");
    let crate::renderer::PendingWork::Deferred { _frame, .. } = &pending.workload else {
        panic!("Deferred work")
    };
    assert!(!_frame.resident.textures[0].filterable);
    assert!(_frame.resident.samplers.is_empty());
    block_on(renderer.readback(&deferred)).expect("Deferred completion");
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn sampled_binding_rejects_the_mutated_actual_owner_filterability() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let plan = plan_forward_resident(
        Vec::new(),
        vec![ForwardTextureInput {
            bytes: vec![0; 256],
            extent: [1, 1],
            bytes_per_row: 256,
            encoding: crate::TextureEncoding::Linear,
        }],
    )
    .expect("one real resident texture plan");
    let (resident, _) =
        ForwardResidentBuffers::allocate_planned(&device, plan).expect("real texture owner");
    let mut resident = match Arc::try_unwrap(resident) {
        Ok(resident) => resident,
        Err(_) => panic!("unshared owner"),
    };
    // This changes the allocation fact held by the actual owner. The requested
    // material ABI remains unchanged, so this cannot be a self-confirming
    // expected-descriptor test.
    resident.textures[0].filterable = false;
    let resident = Arc::new(resident);

    assert!(matches!(
        validate_sampled_texture_owner(
            &resident,
            0,
            device.id(),
            crate::TextureEncoding::Linear,
            true,
        ),
        Err(Error::InvalidState)
    ));
}

#[test]
fn sampled_binding_rejects_the_mutated_actual_owner_device() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let plan = plan_forward_resident(
        Vec::new(),
        vec![ForwardTextureInput {
            bytes: vec![0; 256],
            extent: [1, 1],
            bytes_per_row: 256,
            encoding: crate::TextureEncoding::Linear,
        }],
    )
    .expect("one real resident texture plan");
    let (resident, _) =
        ForwardResidentBuffers::allocate_planned(&device, plan).expect("real texture owner");
    let mut resident = match Arc::try_unwrap(resident) {
        Ok(resident) => resident,
        Err(_) => panic!("unshared owner"),
    };
    resident.textures[0].device = DeviceId::allocate().expect("other device id");
    let resident = Arc::new(resident);

    assert!(matches!(
        validate_sampled_texture_owner(
            &resident,
            0,
            device.id(),
            crate::TextureEncoding::Linear,
            true,
        ),
        Err(Error::InvalidState)
    ));
}

#[test]
fn srgb_payload_allocates_an_srgb_actual_owner() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let plan = plan_forward_resident(
        Vec::new(),
        vec![ForwardTextureInput {
            bytes: vec![0; 256],
            extent: [1, 1],
            bytes_per_row: 256,
            encoding: crate::TextureEncoding::Srgb,
        }],
    )
    .expect("sRGB resident plan");
    let (resident, _) =
        ForwardResidentBuffers::allocate_planned(&device, plan).expect("sRGB texture owner");

    assert_eq!(
        resident.textures[0].desc.format,
        fluxel_rhi::api::TextureFormat::Rgba8UnormSrgb
    );
    assert!(
        validate_sampled_texture_owner(
            &resident,
            0,
            device.id(),
            crate::TextureEncoding::Srgb,
            false,
        )
        .is_ok()
    );
}

#[test]
fn sampled_binding_rejects_an_owner_with_the_wrong_encoding_format() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let plan = plan_forward_resident(
        Vec::new(),
        vec![ForwardTextureInput {
            bytes: vec![0; 256],
            extent: [1, 1],
            bytes_per_row: 256,
            encoding: crate::TextureEncoding::Srgb,
        }],
    )
    .expect("sRGB resident plan");
    let (resident, _) =
        ForwardResidentBuffers::allocate_planned(&device, plan).expect("sRGB texture owner");
    let mut resident = match Arc::try_unwrap(resident) {
        Ok(resident) => resident,
        Err(_) => panic!("unshared owner"),
    };
    resident.textures[0].desc.format = fluxel_rhi::api::TextureFormat::Rgba8Unorm;
    let resident = Arc::new(resident);

    assert!(matches!(
        validate_sampled_texture_owner(
            &resident,
            0,
            device.id(),
            crate::TextureEncoding::Srgb,
            false,
        ),
        Err(Error::InvalidState)
    ));
}
