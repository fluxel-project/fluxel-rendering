//! Adapter facts, and the ledger of what discovery actually proved.
//!
//! Two kinds of value live here, and they are deliberately different things:
//!
//! - [`AdapterLimits`] is a set of numbers copied from the driver. A limit is a
//!   *report*, not permission.
//! - [`CapabilityLedger`] is the outcome of discovery. A row is enabled only when
//!   its evidence was recorded, its numeric floor was satisfied, and its command
//!   probe actually passed.
//!
//! Everything outside a backend acts on the second kind. A driver that reports a
//! compute workgroup limit has not thereby proved that it can dispatch, which is
//! why `supports_compute` on [`AdapterLimits`] is named for the limit it reads
//! and never for the domain it does not establish.
//!
//! # One row, one domain trait
//!
//! [`Capability`] has exactly one variant per optional domain trait in the
//! contract module, and the correspondence is the whole point of the
//! ledger: the graph compiler asks one question -- "does this device support row
//! `X`?" -- and the answer is the availability of one batch of API. A backend
//! that does not implement `X`'s trait cannot be asked at all; a backend that
//! implements it on a context whose row was never proved must still refuse before
//! any side effect.
//!
//! That is why there is no `bool` per feature here and no optional method on one
//! fat trait. An optional method on the floor trait would make every caller
//! handle a runtime branch for a domain the backend type may not have, and would
//! put the refusal in the wrong place. A row plus a domain trait keeps
//! "vocabulary absent" and "vocabulary present, this context unproved" as two
//! facts, exactly as the GL family's compute witness already keeps them.

use std::collections::BTreeMap;

/// Numerical facts a driver reported, with the unavailable value made explicit.
///
/// The default is [`AdapterLimits::unavailable`]: every field is zero, and every
/// predicate below therefore answers `false`. That is the fail-closed direction,
/// so a backend that never queried a limit cannot accidentally satisfy a floor.
///
/// This set covers the rows the native modern backends and the shared lowering
/// need today. GL-family-only rows (texture-unit counts, anisotropy, and the
/// per-stage binding counts a stateful API exposes) join when the GL family
/// converges onto this contract, rather than being guessed here ahead of a
/// consumer -- the same extraction rule the layer's module docs state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdapterLimits {
    /// Maximum width and height of a two-dimensional texture.
    pub max_texture_dimension_2d: u32,
    /// Maximum extent of a three-dimensional texture.
    pub max_texture_dimension_3d: u32,
    /// Maximum array layers in a two-dimensional array texture.
    pub max_texture_array_layers: u32,
    /// Maximum colour attachments in one pass.
    pub max_color_attachments: u32,
    /// Maximum vertex input attributes in one pipeline.
    pub max_vertex_attributes: u32,
    /// Maximum bind groups bound at once.
    pub max_bind_groups: u32,
    /// Required alignment of a dynamic uniform-buffer offset.
    pub min_uniform_buffer_offset_alignment: u32,
    /// Required alignment of a dynamic storage-buffer offset.
    pub min_storage_buffer_offset_alignment: u32,
    /// Maximum byte size of one uniform-buffer binding.
    pub max_uniform_buffer_binding_size: u64,
    /// Maximum byte size of one storage-buffer binding.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum workgroup count in each dispatch dimension.
    pub max_compute_workgroups_per_dimension: [u32; 3],
    /// Maximum workgroup size in each dispatch dimension.
    pub max_compute_workgroup_size: [u32; 3],
    /// Maximum invocations in one compute workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Maximum samples per texel.
    pub max_samples: u32,
    /// Views one attachment may serve in a single pass; 1 is the single-view floor.
    pub max_multiview_view_count: u32,
    /// Maximum draws in one count-bearing multi-draw command.
    ///
    /// `None` means no count-bearing form exists on this backend at all, which is
    /// a different fact from a form that exists with a small limit.
    pub max_multi_draw_indirect_count: Option<u32>,
}

impl AdapterLimits {
    /// Explicit unavailable facts: every field zero, every predicate false.
    pub const fn unavailable() -> Self {
        Self {
            max_texture_dimension_2d: 0,
            max_texture_dimension_3d: 0,
            max_texture_array_layers: 0,
            max_color_attachments: 0,
            max_vertex_attributes: 0,
            max_bind_groups: 0,
            min_uniform_buffer_offset_alignment: 0,
            min_storage_buffer_offset_alignment: 0,
            max_uniform_buffer_binding_size: 0,
            max_storage_buffer_binding_size: 0,
            max_compute_workgroups_per_dimension: [0; 3],
            max_compute_workgroup_size: [0; 3],
            max_compute_invocations_per_workgroup: 0,
            max_samples: 0,
            max_multiview_view_count: 0,
            max_multi_draw_indirect_count: None,
        }
    }

    /// Whether the numeric half of a compute dispatch is satisfiable at all.
    ///
    /// This is the *limit* half only. A true answer here says nothing about
    /// whether a dispatch may be recorded; that is the ledger's answer, and the
    /// two are combined one layer up rather than merged here.
    pub const fn supports_compute(&self) -> bool {
        self.max_compute_workgroups_per_dimension[0] != 0
            && self.max_compute_workgroups_per_dimension[1] != 0
            && self.max_compute_workgroups_per_dimension[2] != 0
            && self.max_compute_workgroup_size[0] != 0
            && self.max_compute_workgroup_size[1] != 0
            && self.max_compute_workgroup_size[2] != 0
            && self.max_compute_invocations_per_workgroup != 0
    }

    /// Whether a storage-buffer binding of `size` bytes fits this adapter.
    pub const fn accepts_storage_binding(&self, size: u64) -> bool {
        self.max_storage_buffer_binding_size != 0 && size <= self.max_storage_buffer_binding_size
    }

    /// Whether the numeric half of a multiview attachment is satisfiable.
    ///
    /// One view is the plain single-view attachment every backend already has, so
    /// it is not a multiview fact; the floor is two, and a context that never
    /// answered the query records a value below it.
    pub const fn supports_multiview(&self) -> bool {
        self.max_multiview_view_count >= 2
    }

    /// Whether a count-bearing multi-draw command exists on this adapter.
    pub const fn supports_multi_draw_indirect(&self) -> bool {
        matches!(self.max_multi_draw_indirect_count, Some(count) if count != 0)
    }
}

/// One optional command domain or facility, as the graph compiler asks about it.
///
/// The variant names are the domain: each one corresponds to one optional trait
/// in the contract module, and that correspondence is what makes a single
/// ledger read answer "which batch of API may this device be asked for".
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum Capability {
    /// Raster passes, raster pipelines and draws.
    ///
    /// A family rather than part of the base, so the base stays minimal; every
    /// backend in the five-platform set serves it, so its requirement is always
    /// met in practice. That is a fact about today's backends, not a rule, and
    /// nothing in the base depends on it.
    Graphics,
    /// Compute passes, compute pipelines and dispatches.
    Compute,
    /// Storage-buffer bindings and the resources they read or write.
    StorageBuffer,
    /// Storage-texture bindings and the resources they read or write.
    StorageImage,
    /// A draw whose parameters are read from a buffer.
    IndirectDraw,
    /// A dispatch whose workgroup count is read from a buffer.
    IndirectDispatch,
    /// One command issuing many draws from per-draw parameter slices in a buffer.
    MultiDrawIndirect,
    /// One command issuing many draws from per-draw parameter slices in memory.
    MultiDraw,
    /// One attachment serving several array layers in a single pass.
    Multiview,
    /// Compute on a queue other than the graphics queue.
    ///
    /// A queue shape is a capability, not a base method: `graphics_queue()` /
    /// `compute_queue()` / `transfer_queue()` on the base would force every
    /// backend to pretend it has all three, and would compress the whole RHI to
    /// the weakest backend's queue model before anyone profiled it.
    AsyncCompute,
    /// Transfers on a queue dedicated to them.
    TransferQueue,
    /// Buffer and texture copies.
    ///
    /// A family rather than floor even though every current backend serves it: the
    /// base holds what Fluxel's semantics require of every backend, and "all five
    /// happen to have it" is a fact about today rather than a requirement. As a row
    /// it is also where the retired `CopyBackend` tier becomes a capability instead
    /// of a parallel type hierarchy.
    Copy,
    /// A query that reports whether any sample passed.
    OcclusionQuery,
    /// A query that reports elapsed time between two points.
    ElapsedQuery,
    /// A query that reports a GPU timestamp.
    TimestampQuery,
    /// A draw that adds a base offset to each index.
    BaseVertex,
    /// A draw that starts at a non-zero instance.
    FirstInstance,
    /// Anisotropic texture sampling.
    AnisotropicFiltering,
}

/// Where a proved capability came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CapabilityEvidence {
    /// The API version itself provides the domain, with no optional facility.
    Core,
    /// A named optional facility provides it.
    ///
    /// The label is provenance for diagnostics only. It is never compared,
    /// parsed, or consulted when deciding anything, so a driver's spelling of an
    /// extension can never become a semantic dependency of this crate.
    Optional(&'static str),
}

/// Result of an actual command-domain probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OperationProbe {
    /// This row is proved without a command, so no probe was required.
    NotRequired,
    /// A real command in this domain was issued and completed.
    Passed,
    /// A real command in this domain was issued and failed.
    Failed,
    /// No probe was run, so nothing is proved either way.
    NotRun,
}

/// Durable evidence for one capability row.
///
/// The three fields are the three independent ways a row can fail to be enabled,
/// and they are kept apart because the sentence a caller gets differs: absent
/// evidence is "this backend never had the domain", a failed probe is "the domain
/// exists and did not work here", and an unsatisfied limit is "the numbers do not
/// allow it".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CapabilityFact {
    /// The route that provides the domain, or `None` when none was found.
    pub evidence: Option<CapabilityEvidence>,
    /// Whether the numeric floor for this row was satisfied.
    pub limits_satisfied: bool,
    /// The command probe outcome.
    pub operation_probe: OperationProbe,
}

impl CapabilityFact {
    /// Whether these three facts together enable `row`.
    ///
    /// A row is enabled when evidence was recorded, its numeric floor was met, and
    /// its probe outcome is one of the two acceptable answers:
    ///
    /// - `Passed`, a real command in the domain ran here and succeeded;
    /// - `NotRequired`, **this backend's** proof for this row is structural and no
    ///   command was needed.
    ///
    /// `Failed` and `NotRun` both leave the row disabled, and they are different
    /// sentences: one says the domain does not work here, the other says nobody has
    /// established that it does.
    ///
    /// Whether a command probe is required is deliberately **not** a property of
    /// the row. It cannot be: the GL family establishes compute by running a
    /// dispatch, while an explicit API establishes it by creating a device on a
    /// queue family that reports compute. A row-level flag would have to be wrong
    /// for one of them, and the first backend to implement this layer found that
    /// out by having its compute row recorded and disabled on a device that
    /// obviously dispatches. The backend therefore states how it proved the row,
    /// and the ledger's job is only to refuse what was not proved.
    pub(crate) const fn is_enabled(self, _row: Capability) -> bool {
        self.evidence.is_some()
            && self.limits_satisfied
            && matches!(
                self.operation_probe,
                OperationProbe::Passed | OperationProbe::NotRequired
            )
    }
}

/// Immutable per-device capability ledger.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CapabilityLedger {
    facts: BTreeMap<Capability, CapabilityFact>,
}

impl CapabilityLedger {
    /// Records the evidence for one row.
    ///
    /// Recording is not enabling: a caller that records a fact still gets `false`
    /// from [`Self::supports`] until all three of the fact's parts agree.
    pub(crate) fn record(&mut self, row: Capability, fact: CapabilityFact) {
        self.facts.insert(row, fact);
    }

    /// Returns the recorded fact for one row, if discovery examined it.
    pub(crate) fn fact(&self, row: Capability) -> Option<CapabilityFact> {
        self.facts.get(&row).copied()
    }

    /// Returns whether this device may be asked for `row`'s domain.
    ///
    /// A row that was never examined answers `false`, which is the same answer as
    /// a row that was examined and refused. That is deliberate: the two are the
    /// same fact to a caller deciding what to record, and the difference is
    /// available from [`Self::fact`] for diagnostics.
    pub(crate) fn supports(&self, row: Capability) -> bool {
        self.fact(row).is_some_and(|fact| fact.is_enabled(row))
    }

    /// Iterates every examined row with its enabled state, in row order.
    ///
    /// The order is the enum's own, so the same discovery always produces the
    /// same sequence -- which matters because the lowered capability value is
    /// compared and cached.
    pub(crate) fn rows(&self) -> impl Iterator<Item = (Capability, bool)> + '_ {
        self.facts
            .iter()
            .map(|(row, fact)| (*row, fact.is_enabled(*row)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proved() -> CapabilityFact {
        CapabilityFact {
            evidence: Some(CapabilityEvidence::Core),
            limits_satisfied: true,
            operation_probe: OperationProbe::Passed,
        }
    }

    #[test]
    fn an_unexamined_row_supports_nothing() {
        let ledger = CapabilityLedger::default();
        assert!(!ledger.supports(Capability::Compute));
        assert_eq!(ledger.fact(Capability::Compute), None);
    }

    #[test]
    fn evidence_alone_does_not_enable_a_probed_row() {
        let mut ledger = CapabilityLedger::default();
        ledger.record(
            Capability::Compute,
            CapabilityFact {
                operation_probe: OperationProbe::NotRun,
                ..proved()
            },
        );
        assert!(!ledger.supports(Capability::Compute));
    }

    #[test]
    fn a_structural_proof_enables_a_row_without_a_command() {
        let mut ledger = CapabilityLedger::default();
        ledger.record(
            Capability::TimestampQuery,
            CapabilityFact {
                operation_probe: OperationProbe::NotRequired,
                ..proved()
            },
        );
        assert!(ledger.supports(Capability::TimestampQuery));
    }

    #[test]
    fn a_probe_that_never_ran_leaves_the_row_disabled() {
        // `NotRun` is the answer for a row whose proof this backend could not make
        // structurally; it is deliberately not the same as `NotRequired`.
        let mut ledger = CapabilityLedger::default();
        ledger.record(
            Capability::TimestampQuery,
            CapabilityFact {
                operation_probe: OperationProbe::NotRun,
                ..proved()
            },
        );
        assert!(!ledger.supports(Capability::TimestampQuery));
    }

    #[test]
    fn an_unsatisfied_limit_keeps_a_probed_row_disabled() {
        let mut ledger = CapabilityLedger::default();
        ledger.record(
            Capability::StorageBuffer,
            CapabilityFact {
                limits_satisfied: false,
                ..proved()
            },
        );
        assert!(!ledger.supports(Capability::StorageBuffer));
    }

    #[test]
    fn unavailable_limits_satisfy_no_floor() {
        let limits = AdapterLimits::unavailable();
        assert!(!limits.supports_compute());
        assert!(!limits.supports_multiview());
        assert!(!limits.supports_multi_draw_indirect());
        assert!(!limits.accepts_storage_binding(0));
    }

    #[test]
    fn one_view_is_not_a_multiview_fact() {
        let limits = AdapterLimits {
            max_multiview_view_count: 1,
            ..AdapterLimits::unavailable()
        };
        assert!(!limits.supports_multiview());
    }

    #[test]
    fn rows_iterate_in_row_order() {
        let mut ledger = CapabilityLedger::default();
        ledger.record(Capability::Multiview, proved());
        ledger.record(Capability::Compute, proved());
        let rows: Vec<Capability> = ledger.rows().map(|(row, _)| row).collect();
        assert_eq!(rows, vec![Capability::Compute, Capability::Multiview]);
    }
}
