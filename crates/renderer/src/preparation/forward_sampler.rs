//! CPU validation for one named ordinary material sampler binding.
//!
//! Sampler policy belongs to the draw/material binding rather than an image
//! asset. This module validates only backend-independent descriptor facts;
//! the selected RHI remains responsible for actual-device admission.

use fluxel_rhi::api::{AddressMode, FilterMode, SamplerDesc};

use super::{
    ForwardDrawSource, ForwardPreparationError, ForwardSamplerBinding, MaterialBindingSchema,
};

/// Resolves the optional explicit policy against the prepared material schema.
pub(super) fn resolve_sampler_binding(
    supplied: Option<&ForwardSamplerBinding>,
    schema: &MaterialBindingSchema,
    source: ForwardDrawSource,
) -> Result<Option<SamplerDesc>, ForwardPreparationError> {
    let Some(binding) = supplied else {
        return Ok(None);
    };
    let Some(expected) = schema.sampler_name.as_deref() else {
        return Err(ForwardPreparationError::UnexpectedSamplerResource {
            source,
            resource: binding.resource.clone(),
        });
    };
    if binding.resource != expected {
        return Err(ForwardPreparationError::SamplerResourceMismatch {
            source,
            expected: expected.into(),
            actual: binding.resource.clone(),
        });
    }
    validate_descriptor(&binding.descriptor, source)?;
    Ok(Some(binding.descriptor.clone()))
}

fn validate_descriptor(
    descriptor: &SamplerDesc,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    let valid_anisotropy = descriptor.anisotropy != 0;
    let valid_lod = descriptor.lod.start.is_finite()
        && descriptor.lod.end.is_finite()
        && descriptor.lod.start <= descriptor.lod.end;
    let aniso_filters = descriptor.anisotropy == 1
        || (descriptor.mag_filter == FilterMode::Linear
            && descriptor.min_filter == FilterMode::Linear
            && descriptor.mip_filter == FilterMode::Linear);
    if descriptor.compare.is_some()
        || descriptor.border.is_some()
        || descriptor.address.contains(&AddressMode::ClampToBorder)
        || !valid_anisotropy
        || !valid_lod
        || !aniso_filters
    {
        return Err(ForwardPreparationError::InvalidSamplerDescriptor { source });
    }
    Ok(())
}
