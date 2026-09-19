//! Sections 19.6-19.10: the portable validation of a shader artifact.
//!
//! Everything `Device::create_shader` must refuse before a backend sees the
//! artifact: the canonical-interface and stage-shape rules, the location-list
//! canonicality, the requirement collections, the two spellings of the workgroup
//! identity, the compiler-option canonicality, and the provenance rules. The
//! device-owned checks (capability, ABI, target support) are deliberately absent
//! — they are the device's, and they arrive as its answers.
//!
//! Not owned here: the shape of the data being checked (the types are in
//! `vocabulary.rs`, `requirements.rs` and `artifact.rs`). No rule here duplicates
//! a rule that belongs to the binding vocabulary; the count invariant is asked
//! through the one function that states it.

use crate::api::binding::vocabulary::{validate_binding_count, validate_binding_kind};
use crate::api::binding::{BindingSupport, BindingSupportQuery};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::requirements::LimitRequirement;

use super::artifact::{ShaderArtifact, ShaderProvenance};
use super::requirements::{ComputeWorkgroupRequirements, ShaderInterface, ShaderRequirements};
use super::vocabulary::{
    InterpolationMode, ShaderLocationInterface, ShaderNumericType, ShaderStage, stage_mask,
};

/// Checks everything about an artifact that does not need a device.
///
/// Section 19.10's `create_shader` validation list, with the three device-owned
/// entries — `ArtifactAcceptance`, `ShaderAbiVersion` acceptance, and binding
/// support — supplied by the caller. The binding-support lookup is a parameter
/// rather than a device read so that this function stays decidable and testable
/// without a backend, which is what section 4 requires of portable validation.
///
/// What is checked here, in the order section 19.10 lists it:
///
/// ```text
/// entry point                 non-empty
/// stage                  ->   interface shape (§19.6) and requirements (§19.7)
/// interface canonical         resources unique and ordered by (group, slot),
///                             inputs and outputs unique and ascending by location
/// interface IO shape          components 1..=4; integer inter-stage IO is Flat
/// resource binding support    BindingSupportQuery == Supported for each resource
/// requirements canonical      features unique and sorted by discriminant,
///                             limits duplicate-free and sorted
/// provenance canonical        compiler option keys unique and sorted
/// ```
///
/// Every refusal is [`RhiErrorKind::InvalidUsage`] except an unsupported binding,
/// which is [`RhiErrorKind::Unsupported`]: a non-canonical artifact is the
/// producer's mistake, while a binding the device cannot express is not.
///
/// It deliberately does **not** normalize. Section 19.6 says a duplicate or
/// misordered interface is a rejection and that the RHI must not silently sort,
/// merge, or choose one, and section 19.8 repeats it for the canonical
/// collections.
pub(crate) fn validate_shader_artifact(
    artifact: &ShaderArtifact,
    binding_support: impl Fn(&BindingSupportQuery) -> BindingSupport,
) -> RhiResult<()> {
    if artifact.entry_point.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a shader artifact must name an entry point",
        ));
    }

    validate_interface(&artifact.interface, artifact.stage)?;
    validate_requirements(&artifact.requirements, artifact.stage)?;
    validate_provenance(&artifact.provenance)?;

    // Section 19.7: binding capability is not repeated in `ShaderRequirements`; it
    // is answered here, by asking about each required resource. `dynamic_offset` is
    // false because it is a layout fact, not a shader semantic (section 19.5).
    let visibility = stage_mask(artifact.stage);
    for requirement in artifact.interface.resources() {
        let query = BindingSupportQuery {
            visibility,
            kind: requirement.kind.clone(),
            count: requirement.count,
            dynamic_offset: false,
        };
        // Also rejects a `min_size` of zero, which is not a size a device can
        // express (section 20.3).
        validate_binding_kind(&requirement.kind)?;
        if binding_support(&query) == BindingSupport::Unsupported {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device cannot express the {:?} binding required at group {} slot {}",
                    requirement.kind,
                    requirement.group.get(),
                    requirement.slot.get()
                ),
            ));
        }
    }

    Ok(())
}

/// Section 19.6's canonical-interface and stage-shape rules.
fn validate_interface(interface: &ShaderInterface, stage: ShaderStage) -> RhiResult<()> {
    // Resources: unique by (group, slot), ordered lexicographically by it. The
    // check is a walk over adjacent pairs rather than a set or a sort: an already
    // canonical list makes duplicates adjacent, so one pass decides both rules
    // without allocating, and the sequence is never rewritten.
    let mut previous: Option<(u32, u32)> = None;
    for requirement in interface.resources() {
        let key = (requirement.group.get(), requirement.slot.get());

        // The count's own invariant travels with the vocabulary. Section 19.5
        // reuses `BindingCount` rather than defining a shader-side copy, so
        // `Fixed(n)` means `n >= 2` here for the same reason it does in a layout —
        // and it is asked through the one function that states it, so the two
        // sides cannot drift. A `Fixed(1)` requirement could never be satisfied by
        // any legal layout; refusing it at the artifact names the producer that
        // has not decided between `Fixed(1)` and `One`.
        validate_binding_count(requirement.count)?;

        if let Some(previous) = previous {
            if key == previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "shader interface declares group {} slot {} twice",
                        key.0, key.1
                    ),
                ));
            }
            if key < previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "shader interface resources are not in canonical order: group {} slot {} \
                         follows group {} slot {}",
                        key.0, key.1, previous.0, previous.1
                    ),
                ));
            }
        }
        previous = Some(key);
    }

    validate_locations(interface.inputs(), "input")?;
    validate_locations(interface.outputs(), "output")?;

    // Section 19.6's stage-specific block. `writes_position` is not required to be
    // false anywhere it is not required to be true: a fragment stage that sets it
    // is not refused by the text, and refusing it here would be inventing a rule.
    match stage {
        ShaderStage::Vertex => {
            if !interface.writes_position() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a vertex entry point must write the position built-in",
                ));
            }
        }
        ShaderStage::Fragment => {}
        ShaderStage::Compute => {
            if !interface.inputs().is_empty() || !interface.outputs().is_empty() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a compute entry point has no stage inputs or outputs",
                ));
            }
            if interface.writes_position()
                || interface.writes_frag_depth()
                || interface.writes_sample_mask()
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a compute entry point writes no raster built-in",
                ));
            }
        }
    }

    // Integer inter-stage IO must be flat. `Flat` is required rather than merely
    // permitted, because there is no interpolation between integers that every
    // backend reproduces, and section 19.6 states this as validation rather than as
    // guidance to the producer.
    let inter_stage: &[ShaderLocationInterface] = match stage {
        ShaderStage::Vertex => interface.outputs(),
        ShaderStage::Fragment => interface.inputs(),
        ShaderStage::Compute => &[],
    };
    for location in inter_stage {
        if !matches!(location.numeric_type, ShaderNumericType::Float32)
            && location
                .interpolation
                .map(|interpolation| interpolation.mode)
                != Some(InterpolationMode::Flat)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "integer inter-stage IO at location {} must be flat",
                    location.location.get()
                ),
            ));
        }
    }

    Ok(())
}

/// Location-list canonicality and width, shared by inputs and outputs.
fn validate_locations(locations: &[ShaderLocationInterface], side: &str) -> RhiResult<()> {
    let mut previous: Option<u32> = None;
    for location in locations {
        if location.components == 0 || location.components > 4 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "shader {side} location {} has {} components; 1..=4 is the portable range",
                    location.location.get(),
                    location.components
                ),
            ));
        }
        let value = location.location.get();
        if let Some(previous) = previous {
            if value == previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!("shader interface declares {side} location {value} twice"),
                ));
            }
            if value < previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "shader interface {side} locations are not ascending: {value} follows \
                         {previous}"
                    ),
                ));
            }
        }
        previous = Some(value);
    }
    Ok(())
}

/// Section 19.7's stage-specific presence rule and the workgroup identity.
fn validate_requirements(requirements: &ShaderRequirements, stage: ShaderStage) -> RhiResult<()> {
    match (stage, requirements.compute_workgroup()) {
        (ShaderStage::Compute, None) => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a compute entry point must declare its workgroup requirements",
            ));
        }
        (ShaderStage::Vertex | ShaderStage::Fragment, Some(_)) => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "only a compute entry point declares workgroup requirements",
            ));
        }
        (ShaderStage::Compute, Some(workgroup)) => {
            validate_compute_workgroup(&workgroup)?;
        }
        (ShaderStage::Vertex | ShaderStage::Fragment, None) => {}
    }

    // Section 19.8's canonical collection rules for the two requirement lists.
    // `OptionalFeature` and `LimitRequirement` are fieldless, so the discriminant
    // is the declaration order of the variant, which is what the specification's
    // "sorted by discriminant" means.
    let mut previous_feature: Option<u16> = None;
    for feature in requirements.required_features() {
        let rank = *feature as u16;
        if let Some(previous) = previous_feature {
            if rank == previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!("shader requirements repeat the feature {feature:?}"),
                ));
            }
            if rank < previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "shader required_features are not sorted by discriminant",
                ));
            }
        }
        previous_feature = Some(rank);
    }

    let mut previous_limit: Option<(u16, u16, u64)> = None;
    for requirement in requirements.limit_requirements() {
        let key = (
            requirement.key() as u16,
            limit_variant_rank(*requirement),
            requirement.value(),
        );
        if let Some(previous) = previous_limit {
            if key == previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "shader limit_requirements repeat one requirement",
                ));
            }
            if key < previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "shader limit_requirements are not sorted by (key, variant, value)",
                ));
            }
        }
        previous_limit = Some(key);
    }

    Ok(())
}

/// The discriminant of a [`LimitRequirement`]'s variant.
///
/// `AtLeast` precedes `AtMost` in the declaration, which is the order section
/// 19.8's canonical encoding sorts by. No wildcard arm, so a third direction
/// fails to compile here.
fn limit_variant_rank(requirement: LimitRequirement) -> u16 {
    match requirement {
        LimitRequirement::AtLeast { .. } => 0,
        LimitRequirement::AtMost { .. } => 1,
    }
}

/// Section 19.7's identity between the two spellings of the workgroup size.
///
/// The non-zero rule belongs to pipeline creation (section 28) as well, and is
/// applied here too because an artifact whose reflection says a workgroup has a
/// zero dimension is internally inconsistent whichever verb reads it first.
pub(crate) fn validate_compute_workgroup(
    workgroup: &ComputeWorkgroupRequirements,
) -> RhiResult<()> {
    if workgroup.x == 0 || workgroup.y == 0 || workgroup.z == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "compute workgroup dimensions must all be non-zero",
        ));
    }
    let product = (workgroup.x as u64)
        .checked_mul(workgroup.y as u64)
        .and_then(|product| product.checked_mul(workgroup.z as u64))
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "compute workgroup size {}x{}x{} overflows",
                    workgroup.x, workgroup.y, workgroup.z
                ),
            )
        })?;
    if product != workgroup.total_invocations as u64 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "compute workgroup declares {} total invocations but {}x{}x{} is {}",
                workgroup.total_invocations, workgroup.x, workgroup.y, workgroup.z, product
            ),
        ));
    }
    Ok(())
}

/// Section 19.8's compiler-option canonicality.
fn validate_provenance(provenance: &ShaderProvenance) -> RhiResult<()> {
    let ShaderProvenance::PortableSource {
        compiler_options, ..
    } = provenance
    else {
        return Ok(());
    };

    let mut previous: Option<&str> = None;
    for (key, _) in compiler_options {
        if key.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "shader compiler option keys must not be empty",
            ));
        }
        if let Some(previous) = previous {
            if key == previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!("shader compiler option {key:?} appears twice"),
                ));
            }
            if key.as_str() < previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "shader compiler options are not sorted lexicographically: {key:?} \
                         follows {previous:?}"
                    ),
                ));
            }
        }
        previous = Some(key);
    }
    Ok(())
}
