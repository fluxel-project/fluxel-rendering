//! Capability families as vocabulary, and requirements as the negotiation input.
//!
//! A [`CapabilityFamily`] is a marker: it names one batch of API and the ledger
//! row that must be proved before that batch may be used. It carries no
//! permission, because a family that hardware might not serve cannot be a
//! permission (see the module docs of [`super`]).
//!
//! [`Requirement`] is the other half. A graph node states what it needs in family
//! terms, and compilation asks the device's ledger once. This is the pipeline the
//! design turns on:
//!
//! ```text
//! Requirement::satisfied_by(&ledger)
//!     Ok     -> the node may be lowered
//!     Err    -> UnsupportedCapability, naming the row and which condition failed
//! ```
//!
//! Nothing here inspects a backend, and nothing here can: the input is a ledger.

use std::collections::BTreeSet;

use crate::common::caps::{Capability, CapabilityLedger, OperationProbe};

/// One batch of API, named as a family.
///
/// A backend that cannot serve the family simply does not implement its trait, so
/// the *vocabulary* question is answered by the type system. The associated row is
/// what a device must then prove.
pub(crate) trait CapabilityFamily {
    /// The ledger row this family requires.
    const ROW: Capability;
}

/// The graphics family: raster passes, pipelines and draws.
pub(crate) struct Graphics;
impl CapabilityFamily for Graphics {
    const ROW: Capability = Capability::Graphics;
}

/// The compute family: compute passes, pipelines and dispatches.
pub(crate) struct Compute;
impl CapabilityFamily for Compute {
    const ROW: Capability = Capability::Compute;
}

/// The storage-buffer family: buffer bindings a shader may read or write.
pub(crate) struct StorageBuffer;
impl CapabilityFamily for StorageBuffer {
    const ROW: Capability = Capability::StorageBuffer;
}

/// The storage-texture family: texture bindings a shader may read or write.
pub(crate) struct StorageTexture;
impl CapabilityFamily for StorageTexture {
    const ROW: Capability = Capability::StorageImage;
}

/// The indirect-draw family: a draw whose parameters come from a buffer.
pub(crate) struct IndirectDraw;
impl CapabilityFamily for IndirectDraw {
    const ROW: Capability = Capability::IndirectDraw;
}

/// The indirect-dispatch family: a dispatch whose counts come from a buffer.
///
/// A separate marker from [`IndirectDraw`] because the rows are separately
/// negotiable. One row per independently negotiable family is what keeps a
/// capability check from answering for a batch the device has not proved.
pub(crate) struct IndirectDispatch;
impl CapabilityFamily for IndirectDispatch {
    const ROW: Capability = Capability::IndirectDispatch;
}

/// The copy family: buffer and texture copies.
pub(crate) struct Copy;
impl CapabilityFamily for Copy {
    const ROW: Capability = Capability::Copy;
}

/// The asynchronous-compute family: compute on a queue other than the graphics one.
pub(crate) struct AsyncCompute;
impl CapabilityFamily for AsyncCompute {
    const ROW: Capability = Capability::AsyncCompute;
}

/// The transfer-queue family: transfers on a dedicated queue.
pub(crate) struct TransferQueue;
impl CapabilityFamily for TransferQueue {
    const ROW: Capability = Capability::TransferQueue;
}

/// The multiview family: one attachment serving several array layers in one pass.
pub(crate) struct Multiview;
impl CapabilityFamily for Multiview {
    const ROW: Capability = Capability::Multiview;
}

/// The base-vertex family: an indexed draw that adds a base offset to each index.
///
/// A marker without a trait, like [`Multiview`]: `GraphicsApi`'s indexed draw
/// deliberately fixes the base vertex at zero (plan section 20.1 -- a base family's
/// parameter space must not be able to name another family's capability), and no
/// retained recipe declares a non-zero one, so there is no verb to write yet and
/// section 20.4 forbids adding vocabulary ahead of its consumer. What the marker
/// buys is what every marker buys: a graph states the requirement in family terms,
/// and a device that never proved the row refuses the graph instead of reaching a
/// draw whose parameter this layer cannot name.
///
/// It is a family of its own rather than a parameter of [`FirstInstance`] because
/// the two are independently negotiable -- one ledger row per family, section
/// 20.1 -- so a device may prove a base offset without proving a non-zero first
/// instance.
pub(crate) struct BaseVertex;
impl CapabilityFamily for BaseVertex {
    const ROW: Capability = Capability::BaseVertex;
}

/// The first-instance family: a draw that starts at a non-zero instance.
///
/// A marker without a trait, for [`BaseVertex`]'s reason, and a separate family
/// from it for the same reason: the two rows are distinct facts rather than one
/// "advanced draw" row.
pub(crate) struct FirstInstance;
impl CapabilityFamily for FirstInstance {
    const ROW: Capability = Capability::FirstInstance;
}

/// Which of the ledger's conditions left a requirement unmet.
///
/// The three are reported separately because they are different sentences to a
/// caller: a family this device never had, a family whose numeric floor the device
/// cannot meet, and a family whose command was tried here and failed or was never
/// tried. Merging them into one "unsupported" would tell a caller nothing about
/// whether another device could run the graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UnmetReason {
    /// Discovery never examined this row.
    NotExamined,
    /// Discovery examined it and found no route that provides the family.
    NoRoute,
    /// A route exists and this device's numbers do not satisfy the floor.
    LimitsUnsatisfied,
    /// The command probe for this row ran and failed.
    ProbeFailed,
    /// The row needs a command probe and none was run.
    ProbeNotRun,
}

/// A requirement this device's ledger does not satisfy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct UnsupportedCapability {
    /// The row that is unmet.
    pub row: Capability,
    /// Which condition left it unmet.
    pub reason: UnmetReason,
}

/// The capability families one graph node requires.
///
/// A requirement is a set of rows rather than a backend name, which is what keeps
/// a compiled graph portable: the same node is executable on every device that
/// satisfies the set, whichever backend serves it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Requirement {
    rows: BTreeSet<Capability>,
}

impl Requirement {
    /// A node that requires no optional family.
    pub(crate) fn none() -> Self {
        Self::default()
    }

    /// Returns this requirement plus one more family.
    pub(crate) fn requiring<F: CapabilityFamily>(mut self) -> Self {
        self.rows.insert(F::ROW);
        self
    }

    /// Returns the rows, in row order, so the same requirement always reports the
    /// same first failure.
    pub(crate) fn rows(&self) -> impl Iterator<Item = Capability> + '_ {
        self.rows.iter().copied()
    }

    /// Whether this device's ledger satisfies every row.
    ///
    /// The first unmet row in row order is reported, and the order is stable, so
    /// the same requirement on the same ledger always produces the same answer.
    pub(crate) fn satisfied_by(&self, ledger: &CapabilityLedger) -> Result<(), UnsupportedCapability> {
        for row in self.rows() {
            if ledger.supports(row) {
                continue;
            }
            return Err(UnsupportedCapability {
                row,
                reason: reason_for(ledger, row),
            });
        }
        Ok(())
    }
}

/// Names which of the ledger's conditions left `row` disabled.
fn reason_for(ledger: &CapabilityLedger, row: Capability) -> UnmetReason {
    let Some(fact) = ledger.fact(row) else {
        return UnmetReason::NotExamined;
    };
    if fact.evidence.is_none() {
        return UnmetReason::NoRoute;
    }
    if !fact.limits_satisfied {
        return UnmetReason::LimitsUnsatisfied;
    }
    match fact.operation_probe {
        OperationProbe::Failed => UnmetReason::ProbeFailed,
        OperationProbe::NotRun => UnmetReason::ProbeNotRun,
        // Reached only when every condition passed, which `satisfied_by` has
        // already excluded; naming it keeps the function total.
        OperationProbe::Passed | OperationProbe::NotRequired => UnmetReason::NoRoute,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::caps::{CapabilityEvidence, CapabilityFact};

    fn fact(
        evidence: Option<CapabilityEvidence>,
        limits_satisfied: bool,
        probe: OperationProbe,
    ) -> CapabilityFact {
        CapabilityFact {
            evidence,
            limits_satisfied,
            operation_probe: probe,
        }
    }

    fn proved_ledger(row: Capability) -> CapabilityLedger {
        let mut ledger = CapabilityLedger::default();
        ledger.record(
            row,
            fact(
                Some(CapabilityEvidence::Core),
                true,
                OperationProbe::Passed,
            ),
        );
        ledger
    }

    #[test]
    fn a_node_requiring_nothing_compiles_on_any_ledger() {
        let requirement = Requirement::none();
        assert_eq!(
            requirement.satisfied_by(&CapabilityLedger::default()),
            Ok(())
        );
    }

    #[test]
    fn a_proved_family_satisfies_its_requirement() {
        let requirement = Requirement::none()
            .requiring::<Graphics>()
            .requiring::<StorageBuffer>();
        let mut ledger = proved_ledger(Capability::Graphics);
        ledger.record(
            Capability::StorageBuffer,
            fact(
                Some(CapabilityEvidence::Core),
                true,
                OperationProbe::Passed,
            ),
        );
        assert_eq!(requirement.satisfied_by(&ledger), Ok(()));
    }

    #[test]
    fn an_unexamined_family_is_reported_as_unexamined() {
        let requirement = Requirement::none().requiring::<Compute>();
        assert_eq!(
            requirement.satisfied_by(&CapabilityLedger::default()),
            Err(UnsupportedCapability {
                row: Capability::Compute,
                reason: UnmetReason::NotExamined,
            })
        );
    }

    #[test]
    fn each_condition_reports_its_own_reason() {
        let requirement = Requirement::none().requiring::<Compute>();

        let mut no_route = CapabilityLedger::default();
        no_route.record(
            Capability::Compute,
            fact(None, true, OperationProbe::Passed),
        );
        assert_eq!(
            requirement.satisfied_by(&no_route),
            Err(UnsupportedCapability {
                row: Capability::Compute,
                reason: UnmetReason::NoRoute,
            })
        );

        let mut limits = CapabilityLedger::default();
        limits.record(
            Capability::Compute,
            fact(Some(CapabilityEvidence::Core), false, OperationProbe::Passed),
        );
        assert_eq!(
            requirement.satisfied_by(&limits),
            Err(UnsupportedCapability {
                row: Capability::Compute,
                reason: UnmetReason::LimitsUnsatisfied,
            })
        );

        let mut failed = CapabilityLedger::default();
        failed.record(
            Capability::Compute,
            fact(Some(CapabilityEvidence::Core), true, OperationProbe::Failed),
        );
        assert_eq!(
            requirement.satisfied_by(&failed),
            Err(UnsupportedCapability {
                row: Capability::Compute,
                reason: UnmetReason::ProbeFailed,
            })
        );

        let mut not_run = CapabilityLedger::default();
        not_run.record(
            Capability::Compute,
            fact(Some(CapabilityEvidence::Core), true, OperationProbe::NotRun),
        );
        assert_eq!(
            requirement.satisfied_by(&not_run),
            Err(UnsupportedCapability {
                row: Capability::Compute,
                reason: UnmetReason::ProbeNotRun,
            })
        );
    }

    #[test]
    fn the_first_unmet_row_in_row_order_is_reported() {
        let requirement = Requirement::none()
            .requiring::<StorageTexture>()
            .requiring::<Compute>();
        let ledger = proved_ledger(Capability::StorageImage);
        // Compute sorts before StorageImage, so it is the row reported.
        assert_eq!(
            requirement.satisfied_by(&ledger),
            Err(UnsupportedCapability {
                row: Capability::Compute,
                reason: UnmetReason::NotExamined,
            })
        );
    }

    #[test]
    fn the_draw_parameter_families_name_distinct_rows() {
        // One row per family (section 20.1): a device that proved a base offset has
        // proved nothing about a non-zero first instance, so neither requirement may
        // be satisfied by the other's row. The two directions are asserted
        // separately because "distinct rows" alone would not catch a `ROW` that
        // pointed at a third row both requirements happened to share.
        assert_ne!(BaseVertex::ROW, FirstInstance::ROW);

        let base_only = proved_ledger(Capability::BaseVertex);
        assert_eq!(
            Requirement::none()
                .requiring::<BaseVertex>()
                .satisfied_by(&base_only),
            Ok(())
        );
        assert_eq!(
            Requirement::none()
                .requiring::<FirstInstance>()
                .satisfied_by(&base_only),
            Err(UnsupportedCapability {
                row: Capability::FirstInstance,
                reason: UnmetReason::NotExamined,
            })
        );

        let instance_only = proved_ledger(Capability::FirstInstance);
        assert_eq!(
            Requirement::none()
                .requiring::<FirstInstance>()
                .satisfied_by(&instance_only),
            Ok(())
        );
        assert_eq!(
            Requirement::none()
                .requiring::<BaseVertex>()
                .satisfied_by(&instance_only),
            Err(UnsupportedCapability {
                row: Capability::BaseVertex,
                reason: UnmetReason::NotExamined,
            })
        );
    }
}
