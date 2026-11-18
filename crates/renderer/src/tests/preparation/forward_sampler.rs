//! Forward ordinary-sampler CPU input admission regressions.

use super::*;
use crate::FeatureIssuer;
use crate::preparation::{MaterialBindingSchema, MaterialTextureBinding};
use fluxel_rhi::api::{AddressMode, FilterMode, SamplerDesc};

fn schema(sampler: Option<&str>) -> MaterialBindingSchema {
    MaterialBindingSchema {
        uniforms: Vec::new(),
        textures: vec![MaterialTextureBinding {
            binding: 1,
            name: "image".into(),
            encoding: None,
        }],
        sampler: sampler.map(|_| 2),
        sampler_name: sampler.map(str::to_owned),
    }
}

fn descriptor() -> SamplerDesc {
    SamplerDesc {
        label: Some("test policy".into()),
        address: [
            AddressMode::MirrorRepeat,
            AddressMode::Repeat,
            AddressMode::ClampToEdge,
        ],
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Linear,
        lod: -1.0..3.0,
        anisotropy: 16,
        compare: None,
        border: None,
    }
}

fn source() -> ForwardDrawSource {
    ForwardDrawSource::Feature {
        feature: FeatureIssuer::new(9, 4).issue("sampler-test").unwrap(),
        local_id: 4,
    }
}

#[test]
fn named_sampler_policy_is_retained_separately_from_image_schema() {
    let binding = ForwardSamplerBinding::new("image_sampler", descriptor());
    let actual = super::super::forward_sampler::resolve_sampler_binding(
        Some(&binding),
        &schema(Some("image_sampler")),
        source(),
    )
    .expect("valid named sampler policy");
    assert_eq!(
        actual.expect("explicit policy").address,
        binding.descriptor.address
    );
}

#[test]
fn omitted_policy_preserves_legacy_renderer_default_selection() {
    assert!(
        super::super::forward_sampler::resolve_sampler_binding(
            None,
            &schema(Some("image_sampler")),
            source(),
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn sampler_name_and_descriptor_fail_before_runtime_admission() {
    let source = source();
    let wrong_name = ForwardSamplerBinding::new("other", descriptor());
    assert!(matches!(
        super::super::forward_sampler::resolve_sampler_binding(
            Some(&wrong_name),
            &schema(Some("image_sampler")),
            source,
        ),
        Err(ForwardPreparationError::SamplerResourceMismatch { .. })
    ));
    let unexpected = ForwardSamplerBinding::new("image_sampler", descriptor());
    assert!(matches!(
        super::super::forward_sampler::resolve_sampler_binding(
            Some(&unexpected),
            &schema(None),
            source,
        ),
        Err(ForwardPreparationError::UnexpectedSamplerResource { .. })
    ));
    let mut invalid = descriptor();
    invalid.anisotropy = 8;
    invalid.mip_filter = FilterMode::Nearest;
    let invalid = ForwardSamplerBinding::new("image_sampler", invalid);
    assert!(matches!(
        super::super::forward_sampler::resolve_sampler_binding(
            Some(&invalid),
            &schema(Some("image_sampler")),
            source,
        ),
        Err(ForwardPreparationError::InvalidSamplerDescriptor { .. })
    ));
}
