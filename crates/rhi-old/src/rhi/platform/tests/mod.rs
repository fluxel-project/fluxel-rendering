//! Contract tests for the device façade.
//!
//! These tests are about *where* a decision is made, not only about its result.
//! The mock journals every call its backend receives, so a refusal that leaves
//! no journal entry behind is a refusal the device reached before its backend
//! was involved. That distinction is the whole point of the device chapter's
//! rule: a problem portable validation can find may not be handed to a backend
//! to discover, because a backend that forgets to look would send it to a
//! driver — and a driver's answer is not this library's contract.
//!
//! Every test therefore asserts both halves: the error kind the caller sees, and
//! the absence of the backend call that must not have happened. A test that
//! checked only the error would pass just as well against a backend that
//! happened to re-check the same rule.

use crate::rhi::binding::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindingKind, BindingSlot, BindingSlotId,
};
use crate::rhi::mock::Mock;
use crate::rhi::pipeline::PipelineInterfaceDescriptor;
use crate::rhi::platform::RhiErrorKind;
use crate::rhi::shader::ShaderStages;

/// A one-slot group layout on `mock`, which is all an interface needs to name a
/// device. A layout with no entries is itself refused, so the slot is the least
/// a legal layout can declare.
fn layout(mock: &Mock) -> BindGroupLayout {
    mock.device()
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
            BindingSlotId::new(0),
            ShaderStages::VERTEX,
            BindingKind::UniformBuffer { min_size: 64 },
        )]))
        .expect("the mock device creates a one-slot layout")
}

#[test]
fn an_interface_naming_another_devices_layout_is_refused_before_the_backend() {
    let owner = Mock::new();
    let other = Mock::new();
    let foreign = layout(&owner);

    let error = other
        .device()
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![foreign]))
        .expect_err("a layout from another device is not a layout for this one");

    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
    assert_eq!(error.operation(), Some("Device::create_pipeline_interface"));
    assert!(
        !other.saw("device:create_pipeline_interface"),
        "the façade refused it, so the backend was never asked: {:?}",
        other.journal()
    );
}

#[test]
fn an_interface_naming_its_own_devices_layout_reaches_the_backend() {
    let mock = Mock::new();
    let own = layout(&mock);

    // The control for the test above. The two descriptors differ only in which
    // device's layout they name, so without this one the refusal above could be
    // attributed to anything else about the descriptor.
    mock.device()
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![own]))
        .expect("a layout from this device is a layout for this device");

    assert!(
        mock.saw("device:create_pipeline_interface"),
        "an accepted interface is the backend's to create: {:?}",
        mock.journal()
    );
}
