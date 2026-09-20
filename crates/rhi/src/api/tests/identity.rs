//! Identity-token contract tests (specification sections 3, 3.1, and 3.4).

use crate::api::{DeviceIdentity, DeviceInstanceId, Label, ObjectId};

/// A process-local instance value, for tests that need one.
///
/// There is no *public* constructor by design, so this reaches the crate-private
/// one. A test that could not name a token could not test anything about tokens;
/// the `compile_fail` doctests on the types are what pin the caller-facing half
/// of the rule.
fn instance(value: u64) -> DeviceInstanceId {
    DeviceInstanceId::new(value)
}

#[test]
fn every_token_is_comparable_hashable_and_printable() {
    // Section 3: "The caller can compare, hash, and print, but cannot construct
    // any valid token by itself." The *cannot construct* half is pinned by the
    // `compile_fail` doctests on the types; this test pins the three capabilities
    // the same sentence grants, so a derive dropped by accident fails here
    // rather than at a caller.
    fn assert_usable<T: Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug>() {}

    assert_usable::<DeviceInstanceId>();
    assert_usable::<DeviceIdentity>();
    assert_usable::<ObjectId>();

    let identity = DeviceIdentity::new(instance(7));
    assert_eq!(identity, identity);
    assert!(!format!("{identity:?}").is_empty());

    let mut set = std::collections::HashSet::new();
    set.insert(identity);
    assert!(set.contains(&identity));
}

#[test]
fn the_same_instance_composes_the_same_identity() {
    // Section 3.1's first row: `Device::clone()` yields the same
    // `DeviceIdentity`. That is only usable as the `WrongDevice` discriminator if
    // identity is a value rather than a handle, so equal components must compose
    // equal identities.
    let first = DeviceIdentity::new(instance(7));
    let second = DeviceIdentity::new(instance(7));

    assert_eq!(first, second);
    assert_eq!(first.instance(), second.instance());
}

#[test]
fn two_instances_are_different_domains() {
    // A DX12 device and a Vulkan device are distinct execution domains. A
    // recreated device likewise receives a fresh instance identity.
    let dx12 = DeviceIdentity::new(instance(7));
    let vulkan = DeviceIdentity::new(instance(8));

    assert_ne!(dx12, vulkan);
    assert_ne!(dx12.instance(), vulkan.instance());
}

#[test]
fn a_label_defaults_to_none_and_displays_a_placeholder() {
    let label = Label::default();
    assert_eq!(label.as_deref(), None);
    assert_eq!(label.to_string(), "<unlabeled>");
    assert_eq!(Label(Some("gbuffer".into())).to_string(), "gbuffer");
}
