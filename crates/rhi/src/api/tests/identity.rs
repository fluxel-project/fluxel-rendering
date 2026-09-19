//! Identity-token contract tests (specification sections 3, 3.1, and 3.4).

use crate::api::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, Label, ObjectId};

/// A process-local instance value, for tests that need one.
///
/// There is no *public* constructor by design, so this reaches the crate-private
/// one. A test that could not name a token could not test anything about tokens;
/// the `compile_fail` doctests on the types are what pin the caller-facing half
/// of the rule.
fn instance(value: u64) -> DeviceInstanceId {
    DeviceInstanceId::new(value)
}

fn generation(value: u64) -> DeviceGeneration {
    DeviceGeneration::new(value)
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
    assert_usable::<DeviceGeneration>();
    assert_usable::<DeviceIdentity>();
    assert_usable::<ObjectId>();

    let identity = DeviceIdentity::new(instance(7), generation(1));
    assert_eq!(identity, identity);
    assert!(!format!("{identity:?}").is_empty());

    let mut set = std::collections::HashSet::new();
    set.insert(identity);
    assert!(set.contains(&identity));
}

#[test]
fn the_same_instance_and_generation_compose_the_same_identity() {
    // Section 3.1's first row: `Device::clone()` yields the same
    // `DeviceIdentity`. That is only usable as the `WrongDevice` discriminator if
    // identity is a value rather than a handle, so equal components must compose
    // equal identities.
    let first = DeviceIdentity::new(instance(7), generation(1));
    let second = DeviceIdentity::new(instance(7), generation(1));

    assert_eq!(first, second);
    assert_eq!(first.instance(), second.instance());
    assert_eq!(first.generation(), second.generation());
}

#[test]
fn two_generations_of_one_instance_are_different_domains() {
    // A generation is not a recovery counter — section 3.1 puts generation++
    // recovery under "P0 None" — but two identities differing only in generation
    // are still two terminal execution domains, not one, and must compare
    // unequal. This is the assertion that would catch a future "revive by
    // incrementing" shortcut.
    let older = DeviceIdentity::new(instance(7), generation(1));
    let newer = DeviceIdentity::new(instance(7), generation(2));

    assert_eq!(older.instance(), newer.instance());
    assert_ne!(older, newer);
}

#[test]
fn two_instances_are_different_domains_even_at_the_same_generation() {
    // The mirror of the test above, and the one section 3.1's multi-device table
    // is really about: a DX12 device and a Vulkan device are two instances, and
    // resource exchange between them must be `WrongDevice` regardless of
    // generation numbering.
    let dx12 = DeviceIdentity::new(instance(7), generation(1));
    let vulkan = DeviceIdentity::new(instance(8), generation(1));

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
