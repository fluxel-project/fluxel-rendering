//! Section 16: samplers.

use super::*;
use crate::api::error::RhiErrorKind;
use crate::api::identity::Label;
use crate::api::resource::sampler::{
    AddressMode, CompareFunction, FilterMode, Sampler, SamplerDescriptor,
    validate_sampler_anisotropy, validate_sampler_descriptor,
};

#[test]
fn a_sampler_descriptor_starts_at_the_portable_defaults() {
    let descriptor = SamplerDescriptor::new();
    assert_eq!(descriptor.label, Label::default());
    assert_eq!(descriptor.address_u, AddressMode::ClampToEdge);
    assert_eq!(descriptor.address_v, AddressMode::ClampToEdge);
    assert_eq!(descriptor.address_w, AddressMode::ClampToEdge);
    assert_eq!(descriptor.mag_filter, FilterMode::Nearest);
    assert_eq!(descriptor.min_filter, FilterMode::Nearest);
    assert_eq!(descriptor.mip_filter, FilterMode::Nearest);
    assert_eq!(descriptor.lod_min, 0.0);
    assert_eq!(descriptor.lod_max, 32.0);
    assert_eq!(descriptor.compare, None);
    // Section 16.1's default block names this one explicitly.
    assert_eq!(descriptor.max_anisotropy, 1);
}

#[test]
fn a_sampler_descriptor_round_trips_through_its_own_builders() {
    let descriptor = SamplerDescriptor::new()
        .with_label("shadow")
        .with_address_modes(
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
            AddressMode::ClampToEdge,
        )
        .with_filters(FilterMode::Linear, FilterMode::Linear, FilterMode::Nearest)
        .with_lod_clamp(1.0, 8.0)
        .with_compare(CompareFunction::LessEqual)
        .with_max_anisotropy(8);

    assert_eq!(descriptor.label.as_deref(), Some("shadow"));
    assert_eq!(descriptor.address_u, AddressMode::Repeat);
    assert_eq!(descriptor.address_v, AddressMode::MirrorRepeat);
    assert_eq!(descriptor.address_w, AddressMode::ClampToEdge);
    assert_eq!(descriptor.mag_filter, FilterMode::Linear);
    assert_eq!(descriptor.min_filter, FilterMode::Linear);
    assert_eq!(descriptor.mip_filter, FilterMode::Nearest);
    assert_eq!(descriptor.lod_min, 1.0);
    assert_eq!(descriptor.lod_max, 8.0);
    assert_eq!(descriptor.compare, Some(CompareFunction::LessEqual));
    assert_eq!(descriptor.max_anisotropy, 8);
}

#[test]
fn the_lod_clamp_must_be_finite_and_ordered() {
    // Section 16.1's three descriptor-local rules. NaN is the interesting case:
    // a naive `lod_min <= lod_max` check would let it through, because the
    // comparison is false rather than an error.
    assert!(validate_sampler_descriptor(&SamplerDescriptor::new()).is_ok());
    assert!(
        validate_sampler_descriptor(&SamplerDescriptor::new().with_lod_clamp(4.0, 4.0)).is_ok()
    );

    assert_kind(
        validate_sampler_descriptor(&SamplerDescriptor::new().with_lod_clamp(8.0, 1.0)),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        validate_sampler_descriptor(&SamplerDescriptor::new().with_lod_clamp(f32::NAN, 1.0)),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        validate_sampler_descriptor(&SamplerDescriptor::new().with_lod_clamp(0.0, f32::NAN)),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        validate_sampler_descriptor(&SamplerDescriptor::new().with_lod_clamp(0.0, f32::INFINITY)),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn anisotropy_below_one_is_refused() {
    assert_kind(
        validate_sampler_descriptor(&SamplerDescriptor::new().with_max_anisotropy(0)),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn anisotropy_above_one_needs_the_optional_feature_and_the_devices_ceiling() {
    // Section 16.1's capability rule, with the two device facts passed in the
    // shape the capability module will supply them: `None` means the optional
    // feature is not enabled, `Some(limit)` means it is.
    let disabled = SamplerDescriptor::new().with_max_anisotropy(4);
    assert_kind(
        validate_sampler_anisotropy(&disabled, None),
        RhiErrorKind::Unsupported,
    );

    let enabled = SamplerDescriptor::new().with_max_anisotropy(4);
    assert!(validate_sampler_anisotropy(&enabled, Some(16)).is_ok());
    assert!(validate_sampler_anisotropy(&enabled, Some(4)).is_ok());
    assert_kind(
        validate_sampler_anisotropy(&enabled, Some(2)),
        RhiErrorKind::InvalidUsage,
    );

    // Anisotropy 1 is always legal: it means "off", so the feature is not
    // needed at all.
    assert!(validate_sampler_anisotropy(&SamplerDescriptor::new(), None).is_ok());
}

#[test]
fn a_sampler_reports_its_own_id_device_and_descriptor() {
    let descriptor = SamplerDescriptor::new().with_label("linear");
    let sampler = Sampler::new(object(51), identity(5), descriptor);

    assert_eq!(sampler.id(), object(51));
    assert_eq!(sampler.device_identity(), identity(5));
    assert_eq!(sampler.descriptor().label.as_deref(), Some("linear"));
    assert_eq!(sampler.descriptor().max_anisotropy, 1);
    let clone = sampler.clone();
    assert_eq!(clone.id(), sampler.id());
    assert!(std::ptr::eq(clone.native(), sampler.native()));
}
