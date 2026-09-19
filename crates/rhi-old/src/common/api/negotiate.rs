//! Negotiation: turning a family, a backend and a device into a handle.
//!
//! [`require`] is the whole mechanism, and it joins the design's two halves in
//! one place:
//!
//! ```text
//! require::<Compute>(&device)
//!   |
//!   |- D: Provides<Compute>          compile time: the backend has the vocabulary
//!   |- device.ledger() proves ROW    run time: this device proved the capability
//!   `- Ok(handle)                    execution no longer asks again
//! ```
//!
//! # Why the two halves are separate traits
//!
//! [`CapabilityFamily`](super::family::CapabilityFamily) stays a pure marker with
//! one associated row, because a family says *how* a batch of API is asked for and
//! nothing about whether a particular device can serve it. [`Provides`] is the
//! backend's side: it names the handle type that backend hands out and constructs
//! it. Splitting them means a backend declares its vocabulary once per family and
//! never has to invent a handle for a family it cannot serve.
//!
//! # Why the check comes first, always
//!
//! [`require`] consults the ledger *before* it asks the backend for a handle, so a
//! refused negotiation cannot have had a side effect. This is the same rule the
//! rest of the crate already follows for unsupported domains -- reject before any
//! object, extension or command side effect -- and it is the rule that makes a
//! structured refusal trustworthy: a caller that receives
//! [`UnsupportedCapability`] knows nothing was created.
//!
//! # What a handle promises
//!
//! A handle exists only because this device proved the family, so its operations
//! do not re-ask the ledger. That is the point of negotiating once: capability
//! branching stays out of the execution path and out of every call site, which is
//! what an interface built from optional methods on one fat trait cannot achieve.
//!
//! A handle borrows the device, which also makes its lifetime the ledger's
//! lifetime. A replacement device therefore cannot revive an old handle: the
//! ledger a handle was negotiated against is the one that proved it, and a new
//! device generation publishes a new ledger and new handles.

use crate::common::api::family::{CapabilityFamily, Requirement, UnsupportedCapability};
use crate::common::caps::CapabilityLedger;

/// A device that captured its capability ledger.
///
/// The ledger is a *device* fact, captured when the device was opened, not a
/// backend constant read on demand. Capability is per adapter, so a backend that
/// answered from its own type instead of from the opened device would be
/// describing hardware it never looked at.
pub(crate) trait CapabilitySource {
    /// Returns the ledger this device proved.
    fn ledger(&self) -> &CapabilityLedger;
}

/// A backend that holds the vocabulary for one capability family.
///
/// Implementing this is how a backend states that it *can express* the family.
/// It is deliberately not a promise that any particular device supports it: that
/// is the ledger's answer, and [`require`] asks both.
pub(crate) trait Provides<F: CapabilityFamily> {
    /// The handle this backend hands out once the family is proved.
    ///
    /// It borrows the device rather than owning a copy of anything, so a handle
    /// cannot outlive the ledger that justified it.
    type Api<'d>
    where
        Self: 'd;

    /// Builds the handle. Called only after the family was proved on this device.
    fn provide(&self) -> Self::Api<'_>;
}

/// Negotiates one family on one device, or reports why this device cannot serve it.
///
/// A refusal names the family and which of the ledger's conditions failed, so a
/// caller can tell a device that will never have the family from one that has not
/// been shown to have it yet.
pub(crate) fn require<'d, D, F>(
    device: &'d D,
) -> Result<<D as Provides<F>>::Api<'d>, UnsupportedCapability>
where
    D: CapabilitySource + Provides<F>,
    F: CapabilityFamily,
{
    Requirement::none()
        .requiring::<F>()
        .satisfied_by(device.ledger())?;
    Ok(device.provide())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::common::api::family::{Compute, UnmetReason};
    use crate::common::caps::{Capability, CapabilityEvidence, CapabilityFact, OperationProbe};

    /// A device that records whether its provider was asked for a handle.
    #[derive(Debug)]
    struct MockDevice {
        ledger: CapabilityLedger,
        provides_calls: Cell<u32>,
    }

    impl MockDevice {
        fn new() -> Self {
            Self {
                ledger: CapabilityLedger::default(),
                provides_calls: Cell::new(0),
            }
        }

        fn record(&mut self, row: Capability, fact: CapabilityFact) {
            self.ledger.record(row, fact);
        }
    }

    impl CapabilitySource for MockDevice {
        fn ledger(&self) -> &CapabilityLedger {
            &self.ledger
        }
    }

    /// A handle that could only exist on a device that proved the family.
    #[derive(Debug)]
    struct MockComputeApi<'d> {
        device: &'d MockDevice,
    }

    impl MockComputeApi<'_> {
        fn device_address(&self) -> *const MockDevice {
            self.device
        }
    }

    impl Provides<Compute> for MockDevice {
        type Api<'d> = MockComputeApi<'d>;

        fn provide(&self) -> MockComputeApi<'_> {
            self.provides_calls.set(self.provides_calls.get() + 1);
            MockComputeApi { device: self }
        }
    }

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

    fn proved() -> CapabilityFact {
        fact(Some(CapabilityEvidence::Core), true, OperationProbe::Passed)
    }

    #[test]
    fn a_proved_family_yields_a_handle() {
        let mut device = MockDevice::new();
        device.record(Capability::Compute, proved());
        let api = require::<_, Compute>(&device).expect("compute was proved");
        assert_eq!(api.device_address(), &device as *const MockDevice);
        assert_eq!(device.provides_calls.get(), 1);
    }

    #[test]
    fn an_unproved_family_refuses_before_the_backend_is_asked() {
        let device = MockDevice::new();
        let error = require::<_, Compute>(&device).expect_err("compute was never examined");
        assert_eq!(error.row, Capability::Compute);
        assert_eq!(error.reason, UnmetReason::NotExamined);
        assert_eq!(
            device.provides_calls.get(),
            0,
            "a refused negotiation must not reach the backend"
        );
    }

    #[test]
    fn a_family_proved_for_another_row_still_refuses() {
        let mut device = MockDevice::new();
        // Graphics is proved; Compute is not, and they are different families.
        device.record(Capability::Graphics, proved());
        let error = require::<_, Compute>(&device).expect_err("compute was never examined");
        assert_eq!(error.row, Capability::Compute);
        assert_eq!(device.provides_calls.get(), 0);
    }

    #[test]
    fn every_ledger_condition_refuses_without_reaching_the_backend() {
        let cases = [
            (
                fact(None, true, OperationProbe::Passed),
                UnmetReason::NoRoute,
            ),
            (
                fact(Some(CapabilityEvidence::Core), false, OperationProbe::Passed),
                UnmetReason::LimitsUnsatisfied,
            ),
            (
                fact(Some(CapabilityEvidence::Core), true, OperationProbe::Failed),
                UnmetReason::ProbeFailed,
            ),
            (
                fact(Some(CapabilityEvidence::Core), true, OperationProbe::NotRun),
                UnmetReason::ProbeNotRun,
            ),
        ];
        for (recorded, expected) in cases {
            let mut device = MockDevice::new();
            device.record(Capability::Compute, recorded);
            let error = require::<_, Compute>(&device).expect_err("the fact does not enable it");
            assert_eq!(error.reason, expected);
            assert_eq!(device.provides_calls.get(), 0);
        }
    }

    #[test]
    fn a_backend_holds_only_the_families_it_implements() {
        // `MockDevice` implements `Provides<Compute>` and nothing else. That the
        // other families have no call site against it is not a run-time fact to
        // assert: `require::<_, Graphics>(&device)` does not compile, and a
        // non-compiling call is the refusal this design wants for an absent
        // vocabulary. What is assertable here is the positive half -- the family
        // the device does hold negotiates normally.
        let mut device = MockDevice::new();
        device.record(Capability::Compute, proved());
        assert!(require::<_, Compute>(&device).is_ok());
    }
}
