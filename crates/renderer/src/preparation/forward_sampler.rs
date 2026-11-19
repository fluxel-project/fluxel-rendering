//! CPU validation for named ordinary material sampler bindings.
//!
//! Sampler policy belongs to the draw/material binding rather than an image
//! asset. This module validates only backend-independent descriptor facts;
//! the selected RHI remains responsible for actual-device admission.

use fluxel_rhi::api::{AddressMode, FilterMode, SamplerDesc};

use super::{
    ForwardDrawSource, ForwardPreparationError, ForwardSamplerBinding, MaterialBindingSchema,
};

/// Resolves explicit policies against the prepared material schema by name.
pub(super) fn resolve_sampler_bindings(
    supplied: &[ForwardSamplerBinding],
    schema: &MaterialBindingSchema,
    source: ForwardDrawSource,
) -> Result<Vec<(u32, SamplerDesc)>, ForwardPreparationError> {
    let mut by_name = std::collections::HashMap::with_capacity(supplied.len());
    for binding in supplied {
        if by_name.insert(binding.resource.as_str(), binding).is_some() {
            return Err(ForwardPreparationError::DuplicateSamplerResource {
                source,
                resource: binding.resource.clone(),
            });
        }
        validate_descriptor(&binding.descriptor, source)?;
    }
    let mut resolved = Vec::with_capacity(supplied.len());
    for binding in &schema.samplers {
        if let Some(supplied) = by_name.remove(binding.name.as_str()) {
            resolved.push((binding.binding, supplied.descriptor.clone()));
        }
    }
    if let Some(binding) = supplied
        .iter()
        .find(|binding| by_name.contains_key(binding.resource.as_str()))
    {
        if schema.samplers.is_empty() || schema.samplers.len() > 1 {
            return Err(ForwardPreparationError::UnexpectedSamplerResource {
                source,
                resource: binding.resource.clone(),
            });
        }
        return Err(ForwardPreparationError::SamplerResourceMismatch {
            source,
            expected: schema.samplers[0].name.clone(),
            actual: binding.resource.clone(),
        });
    }
    Ok(resolved)
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
