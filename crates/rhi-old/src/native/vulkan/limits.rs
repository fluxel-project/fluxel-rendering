//! Step 14's portable floor: the baseline limits an adapter must report.
//!
//! The RHI opens a device to run recipes that are fixed artifacts, so every
//! adapter it is willing to open has to be able to hold them. That requirement is
//! not a capability row: the ledger answers "may this *domain* be entered on this
//! device", which is a different question with a different refusal, and the two
//! are answered at different layers. This module owns only the numeric floor
//! `open` applies before it creates a device.
//!
//! # Where this floor comes from
//!
//! It is the one the borrowed `wgpu-hal` path already applies, restated in this
//! backend's own vocabulary. That path asks the HAL for
//! `wgt::Limits::default()` -- the portable defaults every modern backend and
//! WebGPU are guaranteed to meet -- and refuses an adapter that cannot serve them
//! with [`OpenError::RequiredLimitsUnavailable`](crate::OpenError::RequiredLimitsUnavailable).
//! A fresh implementation that skipped the check would be *more* permissive than
//! the one it replaces, which is why the numbers are ported rather than invented:
//! the plan's preserved-semantics rule is about behavior the rewrite must not
//! quietly drop, and this is one of them.
//!
//! # What is deliberately not checked
//!
//! Every field of [`AdapterLimits`] that is **not** in [`Baseline`] is left out on
//! purpose, and the reason is the same for all of them: nothing in this layer
//! depends on the value yet, so requiring one would be a claim with no consumer.
//! `max_samples` in particular is not checked, because this backend refuses every
//! multisampled image description (`texture::image_create_info` returns `None` for
//! `sample_count != 1`), so a floor derived from it would describe a shape no
//! recorded command can name.
//!
//! The baseline is a *floor*, never a ceiling: an adapter that reports more
//! satisfies it, and reporting the adapter's own larger number onward is what
//! `rhi::capabilities` already does.

use crate::common::caps::AdapterLimits;

/// One field a baseline requirement was not met on.
///
/// Each variant is a different fix -- a driver update, a different adapter, or a
/// different graph -- so they stay separate values rather than one "adapter too
/// weak" sentence. Each carries the numbers it compared, because a diagnostic that
/// says only "too small" has thrown away the one fact a caller needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Limit {
    /// Maximum width and height of a two-dimensional texture.
    MaxTextureDimension2d {
        /// The adapter's report.
        reported: u32,
        /// The floor the report had to reach.
        required: u32,
    },
    /// Maximum colour attachments in one pass.
    MaxColorAttachments {
        /// The adapter's report.
        reported: u32,
        /// The floor the report had to reach.
        required: u32,
    },
    /// Maximum bind groups bound at once.
    MaxBindGroups {
        /// The adapter's report.
        reported: u32,
        /// The floor the report had to reach.
        required: u32,
    },
    /// Required alignment of a dynamic uniform-buffer offset.
    ///
    /// An alignment is a *lower*-is-better limit, so the requirement is that the
    /// report does not exceed the floor. The comparison differs from every other
    /// variant here, which is why the variant carries both numbers rather than a
    /// boolean.
    MinUniformBufferOffsetAlignment {
        /// The adapter's report.
        reported: u32,
        /// The ceiling the report had to stay under.
        required: u32,
    },
    /// Required alignment of a dynamic storage-buffer offset.
    MinStorageBufferOffsetAlignment {
        /// The adapter's report.
        reported: u32,
        /// The ceiling the report had to stay under.
        required: u32,
    },
    /// Maximum byte size of one uniform-buffer binding.
    MaxUniformBufferBindingSize {
        /// The adapter's report.
        reported: u64,
        /// The floor the report had to reach.
        required: u64,
    },
    /// Maximum byte size of one storage-buffer binding.
    MaxStorageBufferBindingSize {
        /// The adapter's report.
        reported: u64,
        /// The floor the report had to reach.
        required: u64,
    },
    /// Maximum workgroup size in each dispatch dimension.
    MaxComputeWorkgroupSize {
        /// The adapter's report.
        reported: [u32; 3],
        /// The floor the report had to reach on every axis.
        required: [u32; 3],
    },
    /// Maximum invocations in one compute workgroup.
    MaxComputeInvocationsPerWorkgroup {
        /// The adapter's report.
        reported: u32,
        /// The floor the report had to reach.
        required: u32,
    },
    /// Maximum workgroup count in each dispatch dimension.
    MaxComputeWorkgroupsPerDimension {
        /// The adapter's report.
        reported: [u32; 3],
        /// The floor the report had to reach on every axis.
        required: [u32; 3],
    },
}

/// Why an adapter is below the portable baseline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BaselineError {
    /// The field that failed, with both numbers it compared.
    pub limit: Limit,
}

impl From<Limit> for BaselineError {
    fn from(limit: Limit) -> Self {
        Self { limit }
    }
}

/// The portable floor every adapter this backend opens must report.
///
/// The values are the WebGPU defaults `wgt::Limits::default()` carries, because
/// those are the numbers the borrowed path refused on and a fresh implementation
/// must not be more permissive than the one it replaces. They are one public
/// constant rather than a literal per field so a test can assert every field
/// against the same value the check reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Baseline {
    /// Maximum width and height of a two-dimensional texture.
    pub max_texture_dimension_2d: u32,
    /// Maximum colour attachments in one pass.
    pub max_color_attachments: u32,
    /// Maximum bind groups bound at once.
    pub max_bind_groups: u32,
    /// The largest alignment a dynamic uniform-buffer offset may require.
    pub min_uniform_buffer_offset_alignment: u32,
    /// The largest alignment a dynamic storage-buffer offset may require.
    pub min_storage_buffer_offset_alignment: u32,
    /// The smallest uniform-buffer binding size that is still usable.
    pub max_uniform_buffer_binding_size: u64,
    /// The smallest storage-buffer binding size that is still usable.
    pub max_storage_buffer_binding_size: u64,
    /// The smallest per-axis compute workgroup size that is still usable.
    pub max_compute_workgroup_size: [u32; 3],
    /// The smallest compute workgroup invocation count that is still usable.
    pub max_compute_invocations_per_workgroup: u32,
    /// The smallest per-axis dispatch count that is still usable.
    pub max_compute_workgroups_per_dimension: [u32; 3],
}

/// The one portable floor of this backend.
pub(crate) const BASELINE: Baseline = Baseline {
    max_texture_dimension_2d: 8_192,
    max_color_attachments: 8,
    max_bind_groups: 4,
    min_uniform_buffer_offset_alignment: 256,
    min_storage_buffer_offset_alignment: 256,
    max_uniform_buffer_binding_size: 64 * 1024,
    max_storage_buffer_binding_size: 128 * 1024 * 1024,
    max_compute_workgroup_size: [256, 256, 64],
    max_compute_invocations_per_workgroup: 256,
    max_compute_workgroups_per_dimension: [65_535, 65_535, 65_535],
};

/// Returns whether an adapter's report satisfies the portable floor.
///
/// This is the boolean half of [`baseline`], kept because a caller that only needs
/// the answer should not have to name the failing field.
pub(crate) fn meets_baseline(adapter: &AdapterLimits) -> bool {
    meets(adapter).is_ok()
}

/// Checks one adapter's report against the portable floor.
///
/// The four groups of fields are checked in the order the structure declares them,
/// and the **first** one that fails is returned, so one adapter always produces one
/// diagnostic and a caller fixing the first does not get a different one on the next
/// attempt. Each group is a small local function because "capacities are floors and
/// alignments are ceilings" is one rule per group rather than one per field; the
/// `?` keeps the groups in order, and each group runs every comparison it owns
/// before returning, so the check is total over the fields it names.
///
/// The two alignment fields compare in the direction their name states: an
/// alignment is a cost, so a *smaller* report is a better adapter and a report
/// above the floor is the refusal. Every other field is a capacity, so the report
/// must reach the floor.
pub(crate) fn meets(adapter: &AdapterLimits) -> Result<(), BaselineError> {
    meets_capacities(adapter)?;
    meets_alignments(adapter)?;
    meets_sizes(adapter)?;
    meets_compute(adapter)
}

/// The capacity fields: a report at or above the floor is accepted.
fn meets_capacities(adapter: &AdapterLimits) -> Result<(), BaselineError> {
    if adapter.max_texture_dimension_2d < BASELINE.max_texture_dimension_2d {
        Err(Limit::MaxTextureDimension2d {
            reported: adapter.max_texture_dimension_2d,
            required: BASELINE.max_texture_dimension_2d,
        }
        .into())
    } else if adapter.max_color_attachments < BASELINE.max_color_attachments {
        Err(Limit::MaxColorAttachments {
            reported: adapter.max_color_attachments,
            required: BASELINE.max_color_attachments,
        }
        .into())
    } else if adapter.max_bind_groups < BASELINE.max_bind_groups {
        Err(Limit::MaxBindGroups {
            reported: adapter.max_bind_groups,
            required: BASELINE.max_bind_groups,
        }
        .into())
    } else {
        Ok(())
    }
}

/// The alignment fields: a report at or below the floor is accepted.
fn meets_alignments(adapter: &AdapterLimits) -> Result<(), BaselineError> {
    if adapter.min_uniform_buffer_offset_alignment > BASELINE.min_uniform_buffer_offset_alignment {
        Err(Limit::MinUniformBufferOffsetAlignment {
            reported: adapter.min_uniform_buffer_offset_alignment,
            required: BASELINE.min_uniform_buffer_offset_alignment,
        }
        .into())
    } else if adapter.min_storage_buffer_offset_alignment
        > BASELINE.min_storage_buffer_offset_alignment
    {
        Err(Limit::MinStorageBufferOffsetAlignment {
            reported: adapter.min_storage_buffer_offset_alignment,
            required: BASELINE.min_storage_buffer_offset_alignment,
        }
        .into())
    } else {
        Ok(())
    }
}

/// The two binding-size fields: a report at or above the floor is accepted.
fn meets_sizes(adapter: &AdapterLimits) -> Result<(), BaselineError> {
    if adapter.max_uniform_buffer_binding_size < BASELINE.max_uniform_buffer_binding_size {
        Err(Limit::MaxUniformBufferBindingSize {
            reported: adapter.max_uniform_buffer_binding_size,
            required: BASELINE.max_uniform_buffer_binding_size,
        }
        .into())
    } else if adapter.max_storage_buffer_binding_size < BASELINE.max_storage_buffer_binding_size {
        Err(Limit::MaxStorageBufferBindingSize {
            reported: adapter.max_storage_buffer_binding_size,
            required: BASELINE.max_storage_buffer_binding_size,
        }
        .into())
    } else {
        Ok(())
    }
}

/// The four compute fields: every axis is compared, and a report at or above the
/// floor is accepted.
fn meets_compute(adapter: &AdapterLimits) -> Result<(), BaselineError> {
    if !at_least_every_axis(
        adapter.max_compute_workgroup_size,
        BASELINE.max_compute_workgroup_size,
    ) {
        Err(Limit::MaxComputeWorkgroupSize {
            reported: adapter.max_compute_workgroup_size,
            required: BASELINE.max_compute_workgroup_size,
        }
        .into())
    } else if adapter.max_compute_invocations_per_workgroup
        < BASELINE.max_compute_invocations_per_workgroup
    {
        Err(Limit::MaxComputeInvocationsPerWorkgroup {
            reported: adapter.max_compute_invocations_per_workgroup,
            required: BASELINE.max_compute_invocations_per_workgroup,
        }
        .into())
    } else if !at_least_every_axis(
        adapter.max_compute_workgroups_per_dimension,
        BASELINE.max_compute_workgroups_per_dimension,
    ) {
        Err(Limit::MaxComputeWorkgroupsPerDimension {
            reported: adapter.max_compute_workgroups_per_dimension,
            required: BASELINE.max_compute_workgroups_per_dimension,
        }
        .into())
    } else {
        Ok(())
    }
}

/// Whether every axis of `reported` reaches the matching axis of `required`.
fn at_least_every_axis(reported: [u32; 3], required: [u32; 3]) -> bool {
    reported
        .into_iter()
        .zip(required)
        .all(|(reported, required)| reported >= required)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An adapter whose every field is exactly the portable floor.
    fn at_baseline() -> AdapterLimits {
        AdapterLimits {
            max_texture_dimension_2d: BASELINE.max_texture_dimension_2d,
            max_texture_dimension_3d: 2_048,
            max_texture_array_layers: 256,
            max_color_attachments: BASELINE.max_color_attachments,
            max_vertex_attributes: 16,
            max_bind_groups: BASELINE.max_bind_groups,
            min_uniform_buffer_offset_alignment: BASELINE.min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment: BASELINE.min_storage_buffer_offset_alignment,
            max_uniform_buffer_binding_size: BASELINE.max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size: BASELINE.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: BASELINE.max_compute_workgroups_per_dimension,
            max_compute_workgroup_size: BASELINE.max_compute_workgroup_size,
            max_compute_invocations_per_workgroup: BASELINE.max_compute_invocations_per_workgroup,
            max_samples: 1,
            max_multiview_view_count: 0,
            max_multi_draw_indirect_count: None,
        }
    }

    #[test]
    fn an_adapter_at_the_floor_meets_the_baseline() {
        // The accepted boundary: "reaches" is not "exceeds", and an adapter that
        // reports exactly the floor is usable. This is the case a strict
        // comparison would wrongly refuse.
        assert_eq!(meets(&at_baseline()), Ok(()));
        assert!(meets_baseline(&at_baseline()));
    }

    #[test]
    fn the_storage_baseline_is_a_hundred_and_twenty_eight_mebibytes() {
        // The constant is written as a shift; this pins the value the diagnostic
        // reports, so the number a caller compares against and the number they are
        // told are one fact.
        assert_eq!(BASELINE.max_storage_buffer_binding_size, 128 * 1024 * 1024);
    }

    #[test]
    fn a_shallow_texture_reports_the_field_both_numbers() {
        let adapter = AdapterLimits {
            max_texture_dimension_2d: BASELINE.max_texture_dimension_2d - 1,
            ..at_baseline()
        };
        assert_eq!(
            meets(&adapter),
            Err(BaselineError {
                limit: Limit::MaxTextureDimension2d {
                    reported: BASELINE.max_texture_dimension_2d - 1,
                    required: BASELINE.max_texture_dimension_2d,
                }
            })
        );
        assert!(!meets_baseline(&adapter));
    }

    /// Lowers one field of an at-baseline adapter, and returns the refusal.
    ///
    /// Every check is exercised through this helper, so a field a later edit stops
    /// reading fails its own case instead of passing silently.
    fn lower_each(case: impl Fn(&mut AdapterLimits)) -> BaselineError {
        let mut adapter = at_baseline();
        case(&mut adapter);
        meets(&adapter).expect_err("the lowered field is below the floor")
    }

    #[test]
    fn the_colour_attachment_count_is_a_floor() {
        // Below the floor:
        let error = lower_each(|adapter| adapter.max_color_attachments = 4);
        assert_eq!(
            error.limit,
            Limit::MaxColorAttachments {
                reported: 4,
                required: BASELINE.max_color_attachments,
            }
        );
        // One above the floor is accepted, because the baseline is a floor rather
        // than an expected value:
        let mut adapter = at_baseline();
        adapter.max_color_attachments = BASELINE.max_color_attachments + 1;
        assert_eq!(meets(&adapter), Ok(()));
    }

    #[test]
    fn the_bind_group_count_is_a_floor() {
        let error = lower_each(|adapter| adapter.max_bind_groups = 1);
        assert_eq!(
            error.limit,
            Limit::MaxBindGroups {
                reported: 1,
                required: BASELINE.max_bind_groups,
            }
        );
    }

    #[test]
    fn both_offset_alignments_are_ceilings_and_compare_the_other_way() {
        // An alignment is a cost: the report must not *exceed* the floor. This is
        // the direction a copied capacity comparison would get backwards, and the
        // case is therefore pinned on both fields.
        let error =
            lower_each(|adapter| adapter.min_uniform_buffer_offset_alignment = 512);
        assert_eq!(
            error.limit,
            Limit::MinUniformBufferOffsetAlignment {
                reported: 512,
                required: BASELINE.min_uniform_buffer_offset_alignment,
            }
        );

        let error = lower_each(|adapter| adapter.min_storage_buffer_offset_alignment = 1024);
        assert_eq!(
            error.limit,
            Limit::MinStorageBufferOffsetAlignment {
                reported: 1024,
                required: BASELINE.min_storage_buffer_offset_alignment,
            }
        );

        // A cheaper alignment than the floor is a better adapter, not a refusal.
        let mut adapter = at_baseline();
        adapter.min_uniform_buffer_offset_alignment = 64;
        adapter.min_storage_buffer_offset_alignment = 32;
        assert_eq!(meets(&adapter), Ok(()));
    }

    #[test]
    fn both_binding_sizes_are_floors() {
        let error = lower_each(|adapter| adapter.max_uniform_buffer_binding_size = 1 << 10);
        assert_eq!(
            error.limit,
            Limit::MaxUniformBufferBindingSize {
                reported: 1 << 10,
                required: BASELINE.max_uniform_buffer_binding_size,
            }
        );

        let error = lower_each(|adapter| adapter.max_storage_buffer_binding_size = 1 << 20);
        assert_eq!(
            error.limit,
            Limit::MaxStorageBufferBindingSize {
                reported: 1 << 20,
                required: BASELINE.max_storage_buffer_binding_size,
            }
        );
    }

    #[test]
    fn a_compute_workgroup_size_that_misses_one_axis_is_refused() {
        // The three axes are one requirement, so a report that reaches the floor on
        // two of them is not a partial pass.
        let error = lower_each(|adapter| adapter.max_compute_workgroup_size = [256, 256, 32]);
        assert_eq!(
            error.limit,
            Limit::MaxComputeWorkgroupSize {
                reported: [256, 256, 32],
                required: BASELINE.max_compute_workgroup_size,
            }
        );
    }

    #[test]
    fn the_workgroup_invocation_count_is_a_floor() {
        let error = lower_each(|adapter| adapter.max_compute_invocations_per_workgroup = 128);
        assert_eq!(
            error.limit,
            Limit::MaxComputeInvocationsPerWorkgroup {
                reported: 128,
                required: BASELINE.max_compute_invocations_per_workgroup,
            }
        );
    }

    #[test]
    fn a_dispatch_count_that_misses_one_axis_is_refused() {
        let error =
            lower_each(|adapter| adapter.max_compute_workgroups_per_dimension = [65_535, 65_535, 65_534]);
        assert_eq!(
            error.limit,
            Limit::MaxComputeWorkgroupsPerDimension {
                reported: [65_535, 65_535, 65_534],
                required: BASELINE.max_compute_workgroups_per_dimension,
            }
        );
    }

    #[test]
    fn an_adapter_that_reports_nothing_fails_the_first_capacity_field() {
        // The fail-closed direction, and the shape every consumer actually meets:
        // `AdapterLimits::unavailable()` is all zeroes, so no floor is satisfied and
        // the refusal names the first field rather than returning a generic answer.
        assert_eq!(
            meets(&AdapterLimits::unavailable()),
            Err(BaselineError {
                limit: Limit::MaxTextureDimension2d {
                    reported: 0,
                    required: BASELINE.max_texture_dimension_2d,
                }
            })
        );
    }

    #[test]
    fn the_first_failing_field_is_the_one_reported() {
        // Two fields below the floor produce the earlier one, so one adapter always
        // yields one diagnostic and a caller fixing it does not meet a new one on
        // the next attempt.
        let adapter = AdapterLimits {
            max_texture_dimension_2d: 1,
            max_bind_groups: 1,
            ..at_baseline()
        };
        assert_eq!(
            meets(&adapter).expect_err("both fields are below the floor").limit,
            Limit::MaxTextureDimension2d {
                reported: 1,
                required: BASELINE.max_texture_dimension_2d,
            }
        );
    }

    #[test]
    fn every_baseline_field_is_read() {
        // The bijection the helper above cannot see: a field that the check stops
        // reading would leave a test above passing while the floor silently widened.
        // Each case lowers exactly one field, so the set of failures names every
        // field the baseline carries.
        let failures = [
            lower_each(|adapter| adapter.max_texture_dimension_2d = 0).limit,
            lower_each(|adapter| adapter.max_color_attachments = 0).limit,
            lower_each(|adapter| adapter.max_bind_groups = 0).limit,
            lower_each(|adapter| adapter.min_uniform_buffer_offset_alignment = u32::MAX).limit,
            lower_each(|adapter| adapter.min_storage_buffer_offset_alignment = u32::MAX).limit,
            lower_each(|adapter| adapter.max_uniform_buffer_binding_size = 0).limit,
            lower_each(|adapter| adapter.max_storage_buffer_binding_size = 0).limit,
            lower_each(|adapter| adapter.max_compute_workgroup_size = [0; 3]).limit,
            lower_each(|adapter| adapter.max_compute_invocations_per_workgroup = 0).limit,
            lower_each(|adapter| adapter.max_compute_workgroups_per_dimension = [0; 3]).limit,
        ];
        let mut names: Vec<&'static str> = failures
            .iter()
            .map(|limit| match limit {
                Limit::MaxTextureDimension2d { .. } => "max_texture_dimension_2d",
                Limit::MaxColorAttachments { .. } => "max_color_attachments",
                Limit::MaxBindGroups { .. } => "max_bind_groups",
                Limit::MinUniformBufferOffsetAlignment { .. } => {
                    "min_uniform_buffer_offset_alignment"
                }
                Limit::MinStorageBufferOffsetAlignment { .. } => {
                    "min_storage_buffer_offset_alignment"
                }
                Limit::MaxUniformBufferBindingSize { .. } => "max_uniform_buffer_binding_size",
                Limit::MaxStorageBufferBindingSize { .. } => "max_storage_buffer_binding_size",
                Limit::MaxComputeWorkgroupSize { .. } => "max_compute_workgroup_size",
                Limit::MaxComputeInvocationsPerWorkgroup { .. } => {
                    "max_compute_invocations_per_workgroup"
                }
                Limit::MaxComputeWorkgroupsPerDimension { .. } => {
                    "max_compute_workgroups_per_dimension"
                }
            })
            .collect();
        names.sort_unstable();
        let expected = [
            "max_bind_groups",
            "max_color_attachments",
            "max_compute_invocations_per_workgroup",
            "max_compute_workgroup_size",
            "max_compute_workgroups_per_dimension",
            "max_storage_buffer_binding_size",
            "max_texture_dimension_2d",
            "max_uniform_buffer_binding_size",
            "min_storage_buffer_offset_alignment",
            "min_uniform_buffer_offset_alignment",
        ];
        assert_eq!(
            names.len(),
            expected.len(),
            "the baseline carries ten fields; one adapter cannot make two of them fail"
        );
        assert_eq!(names, expected);
    }
}
