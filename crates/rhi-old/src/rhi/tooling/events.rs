//! Semantic events (rhi-design sections 53.3 and 58.1).
//!
//! # What it is
//!
//! What a device tells its observers, in CPU observation order. Every event
//! carries the [`SemanticEventId`] that was assigned to it before delivery, so
//! an observer can order and deduplicate without trusting thread scheduling.
//!
//! # Ordering, precisely
//!
//! [`SemanticEventId`] is unique and monotonic within one `DeviceIdentity`, and
//! that is all it is: **CPU observation order, not GPU execution order**. Under
//! a multi-threaded recorder `Event 100 < Event 101` does not mean GPU work 100
//! happens before 101. Real GPU order comes only from `PortableCommand` order
//! within a recording, submission-plan lane order, explicit plan dependencies,
//! and the present relation.
//!
//! # Lifetime, precisely
//!
//! Every reference in [`SemanticEvent`] is valid only for the duration of the
//! `on_event` call. An observer that needs to retain anything must clone or copy
//! it into its own storage; RHI never waits for an observer's later processing,
//! and it never keeps an event alive for one.
//!
//! `ObjectCreated` borrows its complete definition rather than making the
//! observer fetch it, because an object may be reclaimed immediately after the
//! observer returns: a lazy fetch at that point would find nothing to describe.

use super::super::diagnostics::DiagnosticEvent;
use super::super::platform::{DeviceLossInfo, ObjectId};
use super::super::presentation::{
    AcquiredFrameId, PresentReceiptId, PresentState, PresentationConfiguration,
};
use super::super::resource::Extent3d;
use super::super::submission::{CompletionPoint, CompletionState};
use super::commands::{CapturedReadbackRequest, CapturedUploadDefinition};
use super::objects::CapturedObjectDefinition;
use super::work::{CapturedRecordedWork, CapturedSubmissionPlan, CapturedSubmissionReceipt};

/// The identity of one semantic event.
///
/// It is assigned by the device before the event is delivered and is unique and
/// monotonic within that `DeviceIdentity`. It orders observation, not
/// execution: see the module documentation.
///
/// The ordering derives exist because the linearization contract is stated in
/// terms of this value: an observer is promised events in increasing
/// `SemanticEventId` order, and a consumer that has to prove that must be able
/// to compare two ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticEventId(u64);

impl SemanticEventId {
    /// Wraps the next value minted by a device's tooling registry.
    pub(crate) const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// The underlying opaque value.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// One observed semantic change.
///
/// The event vocabulary is `#[non_exhaustive]`: a later SPI minor version may
/// add a variant, and an observer is required to tolerate one it does not know.
///
/// It derives `Clone` and `Copy` because an implementation delivers one event to
/// several observers in turn, and every field is a value type or a borrow.
#[non_exhaustive]
#[derive(Clone, Copy)]
pub enum SemanticEvent<'a> {
    /// An object became observable.
    ///
    /// The complete definition is borrowed directly, so a later lazy query is
    /// never needed for an object that may already be gone.
    ObjectCreated {
        /// This event's identity.
        event: SemanticEventId,
        /// The complete definition.
        definition: &'a CapturedObjectDefinition,
    },

    /// GPU-safe backing has actually been reclaimed from RHI inventory.
    ObjectReclaimed {
        /// This event's identity.
        event: SemanticEventId,
        /// The reclaimed object.
        object: ObjectId,
    },

    /// A CPU upload mutation was defined.
    UploadDefined {
        /// This event's identity.
        event: SemanticEventId,
        /// The upload's complete definition, including its source bytes.
        upload: &'a CapturedUploadDefinition,
    },

    /// A readback request was defined.
    ReadbackDefined {
        /// This event's identity.
        event: SemanticEventId,
        /// The request.
        request: &'a CapturedReadbackRequest,
    },

    /// A recording finished and is still retained by plan or submission
    /// ownership.
    WorkFinished {
        /// This event's identity.
        event: SemanticEventId,
        /// The recording's complete portable semantics.
        work: &'a CapturedRecordedWork,
    },

    /// A submission plan was accepted.
    SubmissionAccepted {
        /// This event's identity.
        event: SemanticEventId,
        /// The complete accepted relation.
        plan: &'a CapturedSubmissionPlan,
        /// The receipt that will report what happens to it.
        receipt: &'a CapturedSubmissionReceipt,
    },

    /// A completion point changed state.
    ///
    /// Device loss must drive every pending completion of that identity to a
    /// terminal state here; a capture is never left waiting for an event RHI
    /// already knows will not arrive.
    CompletionChanged {
        /// This event's identity.
        event: SemanticEventId,
        /// The point that changed.
        point: CompletionPoint,
        /// Its new state.
        state: &'a CompletionState,
    },

    /// A frame was acquired.
    ///
    /// `target` names a `CapturedObjectDefinition::PresentationTarget` and
    /// `configured_presentation` names a
    /// `CapturedObjectDefinition::ConfiguredPresentation`, which in turn
    /// references the target through its own `target` field. Every object id
    /// here therefore has a describable logical definition, and no host or
    /// native presentation handle is serialized.
    FrameAcquired {
        /// This event's identity.
        event: SemanticEventId,
        /// The presentation target the frame came from.
        target: ObjectId,
        /// The configured-presentation lease the frame came from.
        configured_presentation: ObjectId,
        /// The acquired frame.
        frame: AcquiredFrameId,
        /// The configuration the lease was accepted with.
        configuration: &'a PresentationConfiguration,
        /// The frame's drawable extent at acquisition.
        extent: Extent3d,
    },

    /// A present changed state.
    PresentChanged {
        /// This event's identity.
        event: SemanticEventId,
        /// The receipt that changed.
        receipt: PresentReceiptId,
        /// Its new state.
        state: &'a PresentState,
    },

    /// The device identity was lost.
    ///
    /// Loss is terminal, and this is the structured terminal observation that
    /// closes every subscription on that identity.
    DeviceLost {
        /// This event's identity.
        event: SemanticEventId,
        /// Why the identity was lost.
        info: &'a DeviceLossInfo,
    },

    /// A diagnostic was produced.
    Diagnostic {
        /// This event's identity.
        event: SemanticEventId,
        /// The diagnostic.
        diagnostic: &'a DiagnosticEvent,
    },
}
