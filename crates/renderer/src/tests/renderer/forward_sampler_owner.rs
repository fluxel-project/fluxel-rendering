use super::*;

use std::sync::Arc;

use crate::{
    renderer::forward::validate_sampled_texture_owner,
    residency::{ForwardResidentBuffers, ForwardTextureInput, plan_forward_resident},
};

#[test]
fn sampled_binding_rejects_the_mutated_actual_owner_filterability() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let plan = plan_forward_resident(
        Vec::new(),
        vec![ForwardTextureInput {
            bytes: vec![0; 256],
            extent: [1, 1],
            bytes_per_row: 256,
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
        validate_sampled_texture_owner(&resident, 0, device.id()),
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
        validate_sampled_texture_owner(&resident, 0, device.id()),
        Err(Error::InvalidState)
    ));
}
