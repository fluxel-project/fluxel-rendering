//! Section 23.3: the merge of the participating stages' resource requirements.
//!
//! One requirement per `(group, slot)`, carrying the same kind and count from
//! every stage that declared it, with section 23.3's storage-access lattice
//! applied in the one place that rule is deliberately relaxed. The outcome is a
//! set of requirements *about* an interface rather than a description of one, so
//! nothing here is public.
//!
//! Not owned here: the interface the requirements are read from (section 23.1,
//! `interface.rs`) and the pipelines that merge through this file (`raster.rs`,
//! `compute.rs`). This file decides what the stages require together; those
//! decide whether a layout and a device can satisfy it.

use crate::api::binding::{
    BindGroupIndex, BindingCount, BindingKind, BindingSlotId, BindingSupport, BindingSupportQuery,
    BufferBindingAccess, StorageAccess,
};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::shader::vocabulary::stage_mask;
use crate::api::shader::{ShaderInterface, ShaderStage};

use super::interface::PipelineInterface;

/// One binding requirement after the participating stages have been merged.
///
/// Not a public type: section 23.3's merge is an internal step of pipeline
/// creation, and its result is a set of requirements *about* an interface, not a
/// description of one.
///
/// `stages` is the union of the stages that actually declared this binding, which
/// is what "layout visibility ⊇ actually used stages" is compared against.
#[derive(Clone, Debug)]
pub(crate) struct MergedShaderResource {
    /// The group the binding lives in.
    pub(crate) group: BindGroupIndex,
    /// The slot within that group.
    pub(crate) slot: BindingSlotId,
    /// The merged resource semantics.
    pub(crate) kind: BindingKind,
    /// The merged resource count.
    pub(crate) count: BindingCount,
    /// The stages that declared this binding.
    pub(crate) stages: ShaderStageMask,
    /// Whether `kind`'s storage access is the outcome of section 23.3's lattice
    /// rather than a single stage's declaration.
    ///
    /// Load-bearing: when it is true the access is `ReadWrite` only because two
    /// stages disagreed, and the clause "if BindingSupport does not support
    /// ReadWrite -> pipeline Unsupported" must be asked against the *complete*
    /// stage set — not against the two stages that collided while the merge was
    /// still running.
    pub(crate) storage_access_from_merge: bool,
}

/// The stage set of section 19.1, re-exported here so the merged requirement can
/// state it without importing the shader module's vocabulary twice.
pub(crate) use crate::api::shader::ShaderStages as ShaderStageMask;
/// Merges the resource requirements of the participating stages.
///
/// Section 23.3's first half:
///
/// ```text
/// for the same (group, slot), multiple stages must require
///     the same BindingKind
///     the same BindingCount
/// ```
///
/// and then the merge lattice, which is the one place that rule is relaxed:
///
/// ```text
/// StorageBuffer:  ReadOnly + ReadOnly -> ReadOnly
///                 any ReadWrite       -> ReadWrite
/// StorageTexture: same access         -> that access
///                 mixed different     -> requires ReadWrite
/// ```
///
/// Everything else — dimension, format, sample type, multisampled, sampler kind —
/// must match exactly, and a difference is
/// [`RhiErrorKind::IncompatibleInterface`] rather than a silent choice of one
/// side.
///
/// Buffer `min_size` merges to the *maximum*: section 23.3 requires
/// `layout min_size >= shader required min_size`, so the layout must cover every
/// stage that declared the binding, and the maximum is the only value that
/// satisfies all of them at once.
///
/// The returned vector is sorted by `(group, slot)`. That is a determinism
/// courtesy for diagnostics, not a rule: nothing here compares two orderings for
/// equality, and no caller-visible value depends on the order.
pub(crate) fn merge_shader_resources<'a>(
    stages: impl IntoIterator<Item = (ShaderStage, &'a ShaderInterface)>,
) -> RhiResult<Vec<MergedShaderResource>> {
    let mut merged: Vec<MergedShaderResource> = Vec::new();

    for (stage, interface) in stages {
        let visibility = stage_mask(stage);
        for requirement in interface.resources() {
            let existing = merged
                .iter_mut()
                .find(|entry| entry.group == requirement.group && entry.slot == requirement.slot);
            let Some(existing) = existing else {
                merged.push(MergedShaderResource {
                    group: requirement.group,
                    slot: requirement.slot,
                    kind: requirement.kind.clone(),
                    count: requirement.count,
                    stages: visibility,
                    storage_access_from_merge: false,
                });
                continue;
            };

            if existing.count != requirement.count {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "group {} slot {} is declared with {:?} by one stage and {:?} by another",
                        requirement.group.get(),
                        requirement.slot.get(),
                        existing.count,
                        requirement.count
                    ),
                ));
            }

            let (kind, merged_access) = merge_kinds(&existing.kind, &requirement.kind)?;
            existing.kind = kind;
            existing.storage_access_from_merge |= merged_access;
            existing.stages = existing.stages.union(visibility);
        }
    }

    // A `(group, slot)` that two stages disagree about the *kind* of is refused
    // above; anything that survived is merged, so the order here is only for
    // stable diagnostics.
    merged.sort_by_key(|entry| (entry.group.get(), entry.slot.get()));
    Ok(merged)
}

/// Section 23.3's kind merge, returning the merged kind and whether the storage
/// access came from the lattice rather than from one stage's declaration.
///
/// No wildcard arm on [`BindingKind`]: a sixth variant must be classified here,
/// and the two arms that reject a cross-class pairing name both sides.
fn merge_kinds(left: &BindingKind, right: &BindingKind) -> RhiResult<(BindingKind, bool)> {
    match (left, right) {
        (
            BindingKind::UniformBuffer { min_size: left },
            BindingKind::UniformBuffer { min_size: right },
        ) => Ok((
            BindingKind::UniformBuffer {
                min_size: (*left).max(*right),
            },
            false,
        )),

        (
            BindingKind::StorageBuffer {
                access: left_access,
                min_size: left_size,
            },
            BindingKind::StorageBuffer {
                access: right_access,
                min_size: right_size,
            },
        ) => {
            let merged_access = merge_buffer_access(*left_access, *right_access);
            Ok((
                BindingKind::StorageBuffer {
                    access: merged_access,
                    min_size: (*left_size).max(*right_size),
                },
                merged_access != *left_access || merged_access != *right_access,
            ))
        }

        (
            BindingKind::SampledTexture {
                dimension: left_dimension,
                sample_type: left_sample,
                multisampled: left_multisampled,
            },
            BindingKind::SampledTexture {
                dimension: right_dimension,
                sample_type: right_sample,
                multisampled: right_multisampled,
            },
        ) => {
            if left_dimension != right_dimension
                || left_sample != right_sample
                || left_multisampled != right_multisampled
            {
                return Err(kind_mismatch(left, right));
            }
            Ok((left.clone(), false))
        }

        (
            BindingKind::StorageTexture {
                dimension: left_dimension,
                format: left_format,
                access: left_access,
            },
            BindingKind::StorageTexture {
                dimension: right_dimension,
                format: right_format,
                access: right_access,
            },
        ) => {
            if left_dimension != right_dimension || left_format != right_format {
                return Err(kind_mismatch(left, right));
            }
            // "same access -> that access; mixed different access -> requires
            // ReadWrite". The two `ReadOnly`/`WriteOnly` combination is the only
            // one that reaches `ReadWrite` from a disagreement, and
            // `ReadOnly + WriteOnly` has no narrower common access: a shader that
            // only reads and a shader that only writes still need both, which is
            // exactly `ReadWrite`.
            if left_access == right_access {
                return Ok((left.clone(), false));
            }
            Ok((
                BindingKind::StorageTexture {
                    dimension: *left_dimension,
                    format: *left_format,
                    access: StorageAccess::ReadWrite,
                },
                true,
            ))
        }

        (
            left @ BindingKind::Sampler { kind: left_kind },
            right @ BindingKind::Sampler { kind: right_kind },
        ) => {
            if left_kind != right_kind {
                return Err(kind_mismatch(left, right));
            }
            Ok((left.clone(), false))
        }

        (BindingKind::AccelerationStructure, BindingKind::AccelerationStructure)
        | (BindingKind::ExternalTexture, BindingKind::ExternalTexture) => Ok((left.clone(), false)),

        (
            BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. },
            BindingKind::SampledTexture { .. }
            | BindingKind::StorageTexture { .. }
            | BindingKind::Sampler { .. },
        )
        | (
            BindingKind::SampledTexture { .. }
            | BindingKind::StorageTexture { .. }
            | BindingKind::Sampler { .. },
            BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. },
        ) => Err(kind_mismatch(left, right)),

        // --- Same class, different resource ---------------------------------
        //
        // The pairs below are the ones the class-level arms above do not reach,
        // because both sides sit in the same class: a buffer against a buffer, a
        // texture against a texture, a texture against a sampler. Each is still a
        // mismatch, and each is listed rather than folded into a wildcard: a
        // uniform buffer is not a storage buffer (the access discipline differs),
        // a sampled texture is not a storage texture (the shader may write one and
        // not the other), and neither texture is a sampler (a sampler is not a
        // resource the shader reads texels from). Naming them one by one keeps
        // "the two sides describe different resources" a decision this function
        // makes, so a sixth `BindingKind` variant fails to compile here instead of
        // merging into whatever the wildcard would have returned.
        (BindingKind::UniformBuffer { .. }, BindingKind::StorageBuffer { .. }) => {
            Err(kind_mismatch(left, right))
        }
        (BindingKind::StorageBuffer { .. }, BindingKind::UniformBuffer { .. }) => {
            Err(kind_mismatch(left, right))
        }
        (BindingKind::SampledTexture { .. }, BindingKind::StorageTexture { .. }) => {
            Err(kind_mismatch(left, right))
        }
        (BindingKind::StorageTexture { .. }, BindingKind::SampledTexture { .. }) => {
            Err(kind_mismatch(left, right))
        }
        (BindingKind::SampledTexture { .. }, BindingKind::Sampler { .. }) => {
            Err(kind_mismatch(left, right))
        }
        (BindingKind::Sampler { .. }, BindingKind::SampledTexture { .. }) => {
            Err(kind_mismatch(left, right))
        }
        (BindingKind::StorageTexture { .. }, BindingKind::Sampler { .. }) => {
            Err(kind_mismatch(left, right))
        }
        (BindingKind::Sampler { .. }, BindingKind::StorageTexture { .. }) => {
            Err(kind_mismatch(left, right))
        }
        (BindingKind::AccelerationStructure, _)
        | (_, BindingKind::AccelerationStructure)
        | (BindingKind::ExternalTexture, _)
        | (_, BindingKind::ExternalTexture) => Err(kind_mismatch(left, right)),
    }
}

/// The refusal for a kind that two stages declare differently.
fn kind_mismatch(left: &BindingKind, right: &BindingKind) -> RhiError {
    RhiError::new(
        RhiErrorKind::IncompatibleInterface,
        format!(
            "two stages require incompatible bindings for one (group, slot): {left:?} against {right:?}"
        ),
    )
}

/// Section 23.3's buffer-access lattice: any `ReadWrite` wins, otherwise the only
/// surviving combination is `ReadOnly + ReadOnly`.
fn merge_buffer_access(
    left: BufferBindingAccess,
    right: BufferBindingAccess,
) -> BufferBindingAccess {
    match (left, right) {
        (BufferBindingAccess::ReadOnly, BufferBindingAccess::ReadOnly) => {
            BufferBindingAccess::ReadOnly
        }
        (BufferBindingAccess::ReadWrite, _) | (_, BufferBindingAccess::ReadWrite) => {
            BufferBindingAccess::ReadWrite
        }
    }
}

/// Checks merged requirements against a pipeline interface.
///
/// Section 23.3's second half, for each merged `(group, slot)`:
///
/// ```text
/// layout visibility   superset of the stages that use it
/// layout kind         compatible with the shader requirement
/// layout count        == shader requirement count
/// layout min_size     >= shader required min_size   (buffers)
/// ```
///
/// "Compatible" is a containment for the two orderings section 23.3 defines and
/// equality for everything else: the layout's storage access must *permit* what
/// the shader does (`ReadOnly` is covered by `ReadOnly` or `ReadWrite`, and
/// `WriteOnly` by `WriteOnly` or `ReadWrite`), and every other value must match
/// exactly, as section 23.3's closing sentence requires.
///
/// A binding the interface declares that no shader uses is legal and is not
/// examined: section 23.3 keeps it that way so that a Renderer can share one
/// interface among several pipelines.
///
/// The storage-access support clause is checked here rather than in the merge,
/// because it is the one rule that needs the *complete* stage set — see
/// [`MergedShaderResource::storage_access_from_merge`].
pub(crate) fn validate_shader_resource_requirements(
    merged: &[MergedShaderResource],
    interface: &PipelineInterface,
    binding_support: impl Fn(&BindingSupportQuery) -> BindingSupport,
) -> RhiResult<()> {
    for requirement in merged {
        let Some(layout) = interface.group(requirement.group) else {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "a stage requires a binding in group {}, which the pipeline interface does \
                     not declare",
                    requirement.group.get()
                ),
            ));
        };
        let Some(slot) = layout.slot(requirement.slot) else {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "a stage requires a binding at group {} slot {}, which the layout does not \
                     declare",
                    requirement.group.get(),
                    requirement.slot.get()
                ),
            ));
        };

        if !slot.visibility.contains(requirement.stages) {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "group {} slot {} is visible to {:?} but is used by {:?}",
                    requirement.group.get(),
                    requirement.slot.get(),
                    slot.visibility,
                    requirement.stages
                ),
            ));
        }

        if slot.count != requirement.count {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "group {} slot {} declares {:?} in the layout but {:?} in the shader",
                    requirement.group.get(),
                    requirement.slot.get(),
                    slot.count,
                    requirement.count
                ),
            ));
        }

        validate_kind_against_layout(&slot.kind, &requirement.kind, requirement, &binding_support)?;
    }

    Ok(())
}

/// One merged requirement's kind against the layout slot that must satisfy it.
fn validate_kind_against_layout(
    layout_kind: &BindingKind,
    required: &BindingKind,
    requirement: &MergedShaderResource,
    binding_support: &impl Fn(&BindingSupportQuery) -> BindingSupport,
) -> RhiResult<()> {
    // The support clause comes first, because it is a property of the merged
    // requirement rather than of the layout: two stages that disagree about a
    // storage access need `ReadWrite`, and a device that cannot express
    // `ReadWrite` for the complete stage set cannot run the pipeline at all.
    if requirement.storage_access_from_merge {
        let query = BindingSupportQuery {
            visibility: requirement.stages,
            kind: required.clone(),
            count: requirement.count,
            dynamic_offset: false,
        };
        if binding_support(&query) == BindingSupport::Unsupported {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "group {} slot {} is required with {:?} after merging the stages, and this \
                     device cannot express it",
                    requirement.group.get(),
                    requirement.slot.get(),
                    required
                ),
            ));
        }
    }

    match (layout_kind, required) {
        (BindingKind::AccelerationStructure, BindingKind::AccelerationStructure)
        | (BindingKind::ExternalTexture, BindingKind::ExternalTexture) => Ok(()),
        (
            BindingKind::UniformBuffer {
                min_size: layout_size,
            },
            BindingKind::UniformBuffer {
                min_size: required_size,
            },
        )
        | (
            BindingKind::StorageBuffer {
                min_size: layout_size,
                ..
            },
            BindingKind::StorageBuffer {
                min_size: required_size,
                ..
            },
        ) => {
            if layout_size < required_size {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "group {} slot {} requires {required_size} bytes but the layout \
                         guarantees only {layout_size}",
                        requirement.group.get(),
                        requirement.slot.get()
                    ),
                ));
            }
            if let (
                BindingKind::StorageBuffer {
                    access: layout_access,
                    ..
                },
                BindingKind::StorageBuffer {
                    access: required_access,
                    ..
                },
            ) = (layout_kind, required)
            {
                if !buffer_access_covers(*layout_access, *required_access) {
                    return Err(RhiError::new(
                        RhiErrorKind::IncompatibleInterface,
                        format!(
                            "group {} slot {} needs {required_access:?} access but the layout \
                             declares {layout_access:?}",
                            requirement.group.get(),
                            requirement.slot.get()
                        ),
                    ));
                }
            }
            Ok(())
        }

        (
            BindingKind::StorageTexture {
                dimension: layout_dimension,
                format: layout_format,
                access: layout_access,
            },
            BindingKind::StorageTexture {
                dimension: required_dimension,
                format: required_format,
                access: required_access,
            },
        ) => {
            if layout_dimension != required_dimension || layout_format != required_format {
                return Err(kind_mismatch(layout_kind, required));
            }
            if !storage_access_covers(*layout_access, *required_access) {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "group {} slot {} needs {required_access:?} access but the layout \
                         declares {layout_access:?}",
                        requirement.group.get(),
                        requirement.slot.get()
                    ),
                ));
            }
            Ok(())
        }

        (
            BindingKind::SampledTexture {
                dimension: layout_dimension,
                sample_type: layout_sample,
                multisampled: layout_multisampled,
            },
            BindingKind::SampledTexture {
                dimension: required_dimension,
                sample_type: required_sample,
                multisampled: required_multisampled,
            },
        ) => {
            if layout_dimension != required_dimension
                || layout_sample != required_sample
                || layout_multisampled != required_multisampled
            {
                return Err(kind_mismatch(layout_kind, required));
            }
            Ok(())
        }

        (
            BindingKind::Sampler {
                kind: layout_sampler,
            },
            BindingKind::Sampler {
                kind: required_sampler,
            },
        ) => {
            if layout_sampler != required_sampler {
                return Err(kind_mismatch(layout_kind, required));
            }
            Ok(())
        }

        (
            BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. },
            BindingKind::SampledTexture { .. }
            | BindingKind::StorageTexture { .. }
            | BindingKind::Sampler { .. },
        )
        | (
            BindingKind::SampledTexture { .. }
            | BindingKind::StorageTexture { .. }
            | BindingKind::Sampler { .. },
            BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. },
        ) => Err(kind_mismatch(layout_kind, required)),

        // The same-class pairs `merge_kinds` enumerates, for the same reason: a
        // layout that declares one resource cannot serve a stage that asked for a
        // different one, even when the two are in the same class. Listed one by one
        // so that a sixth `BindingKind` variant is a compile error here.
        (BindingKind::UniformBuffer { .. }, BindingKind::StorageBuffer { .. })
        | (BindingKind::StorageBuffer { .. }, BindingKind::UniformBuffer { .. })
        | (BindingKind::SampledTexture { .. }, BindingKind::StorageTexture { .. })
        | (BindingKind::StorageTexture { .. }, BindingKind::SampledTexture { .. })
        | (BindingKind::SampledTexture { .. }, BindingKind::Sampler { .. })
        | (BindingKind::Sampler { .. }, BindingKind::SampledTexture { .. })
        | (BindingKind::StorageTexture { .. }, BindingKind::Sampler { .. })
        | (BindingKind::Sampler { .. }, BindingKind::StorageTexture { .. }) => {
            Err(kind_mismatch(layout_kind, required))
        }
        (BindingKind::AccelerationStructure, _)
        | (_, BindingKind::AccelerationStructure)
        | (BindingKind::ExternalTexture, _)
        | (_, BindingKind::ExternalTexture) => Err(kind_mismatch(layout_kind, required)),
    }
}

/// Whether a layout's storage-buffer access permits the shader's.
fn buffer_access_covers(layout: BufferBindingAccess, required: BufferBindingAccess) -> bool {
    match (layout, required) {
        (BufferBindingAccess::ReadOnly, BufferBindingAccess::ReadOnly) => true,
        (BufferBindingAccess::ReadOnly, BufferBindingAccess::ReadWrite) => false,
        (BufferBindingAccess::ReadWrite, _) => true,
    }
}

/// Whether a layout's storage-texture access permits the shader's.
fn storage_access_covers(layout: StorageAccess, required: StorageAccess) -> bool {
    match (layout, required) {
        (StorageAccess::ReadWrite, _) => true,
        (StorageAccess::ReadOnly, StorageAccess::ReadOnly) => true,
        (StorageAccess::ReadOnly, StorageAccess::WriteOnly | StorageAccess::ReadWrite) => false,
        (StorageAccess::WriteOnly, StorageAccess::WriteOnly) => true,
        (StorageAccess::WriteOnly, StorageAccess::ReadOnly | StorageAccess::ReadWrite) => false,
    }
}
