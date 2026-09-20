//! Specification section 58.1 and 58.2: the semantic event and its lifetime.
//!
//! One responsibility: **the message an observer receives, and the rule about how
//! long anything in it is valid.** Section 58.1 defines the enum; section 58.2
//! defines the lifetime discipline, which is the part a caller gets wrong.
//! Both are here because a type whose contract is its borrow discipline cannot be
//! reviewed apart from the callbacks that receive it.
//!
//! Not owned here: the payloads the variants borrow ([`super::definition`],
//! [`super::mutation`], [`super::plan`], [`super::work`]) and the registration
//! that starts and stops the delivery ([`super::ToolingAccess`]).
//!
//! # Section 58.2's discipline, stated once
//!
//! ```text
//! references valid          only during the on_event() call
//! observer retains          clone/copy into its own queue, first
//! RHI waits for             nothing after the callback returns
//! ```
//!
//! So an observer is not a stream reader: it is a synchronous interceptor on the
//! RHI's own thread, and an observer that keeps a `&CapturedObjectDefinition` is
//! holding a dangling reference the moment it returns. Every reference in this
//! enum is `&'a`, which is what makes the compiler say so rather than making the
//! defect reach the capture file.
//!
//! The four prohibitions section 58.2 places on an observer — no re-entering a
//! mutating API on the same device, no waiting for GPU completion or otherwise
//! blocking, no dropping its own subscription, no altering command or submission
//! semantics — are not enforceable here and are not pretended to be. The one that
//! is closest to enforceable is the subscription drop, and it is stated as a
//! deadlock rather than as an error on
//! [`super::ToolingSubscription`]: dropping waits for the callback currently
//! executing, and that callback is the one doing the dropping. What *is* kept is
//! the budget: an observer may add capture and debug CPU overhead, and may not
//! change GPU results.
//!
//! # Why `ObjectCreated` borrows a whole definition
//!
//! Section 58.1 gives the reason, and it is a race rather than a convenience: a
//! lazy `describe_object` afterwards could arrive after the object was reclaimed,
//! and a capture would then hold a reference to something it can no longer
//! describe. Carrying the complete definition in the event closes that window
//! because the definition is copied out while the object is provably alive.
//!
//! The same reasoning is why the events that *do* name an object by identity —
//! `ObjectReclaimed` — carry no definition: there is nothing left to describe.

use crate::api::diagnostics::DiagnosticEvent;
use crate::api::identity::ObjectId;
use crate::api::platform::DeviceLossInfo;
use crate::api::presentation::{
    AcquiredFrameId, PresentReceiptId, PresentState, PresentationConfiguration,
};
use crate::api::resource::texture::Extent3d;
use crate::api::submission::{CompletionPoint, CompletionState};

use super::SemanticEventId;
use super::definition::CapturedObjectDefinition;
use super::mutation::{CapturedReadbackRequest, CapturedUploadDefinition};
use super::plan::{CapturedSubmissionPlan, CapturedSubmissionReceipt};
use super::work::CapturedRecordedWork;

/// One semantic event, delivered to a [`super::SemanticObserver`].
///
/// Not `Clone` (section 58.1), and deliberately: every payload is a borrow, and a
/// type that borrowed and cloned would invite an observer to clone the event
/// instead of copying what it needs, which is the one habit section 58.2 exists
/// to prevent. An observer that wants the event later copies the payload.
///
/// # The identity in every variant
///
/// Each variant's first field is its [`SemanticEventId`]. It is repeated rather
/// than hoisted into a struct wrapper because section 58.1 declares it that way
/// and because a wrapper would add a layer between an observer and its `match` —
/// every observer's first act is to dispatch on the variant, and every capture's
/// first act is to write the id down.
#[non_exhaustive]
pub enum SemanticEvent<'a> {
    /// An object was created, with its complete definition.
    ObjectCreated {
        /// This event's identity.
        event: SemanticEventId,
        /// What was created, borrowed for the duration of the callback.
        definition: &'a CapturedObjectDefinition,
    },

    /// An object's GPU-safe backing was actually reclaimed from RHI inventory.
    ///
    /// "Actually" is the load-bearing word (section 58.1): this is emitted when
    /// the memory is free, not when the caller dropped its last handle. A capture
    /// that treated the drop as reclamation would record a snapshot point at which
    /// the resource was still live.
    ObjectReclaimed {
        /// This event's identity.
        event: SemanticEventId,
        /// The object that was reclaimed.
        object: ObjectId,
    },

    /// An upload was defined, carrying the mutation it will perform.
    UploadDefined {
        /// This event's identity.
        event: SemanticEventId,
        /// The upload, borrowed for the duration of the callback.
        upload: &'a CapturedUploadDefinition,
    },

    /// A readback request was defined.
    ReadbackDefined {
        /// This event's identity.
        event: SemanticEventId,
        /// The request, borrowed for the duration of the callback.
        request: &'a CapturedReadbackRequest,
    },

    /// A recording finished and became describable work.
    WorkFinished {
        /// This event's identity.
        event: SemanticEventId,
        /// The work, borrowed for the duration of the callback.
        work: &'a CapturedRecordedWork,
    },

    /// A plan was accepted, with the relation it established and the receipt it
    /// produced.
    ///
    /// Both, and in one event: the plan and the receipt are the two halves of
    /// "this was submitted", and delivering them apart would let a capture record
    /// a plan whose acceptance it never saw.
    SubmissionAccepted {
        /// This event's identity.
        event: SemanticEventId,
        /// The plan the RHI accepted.
        plan: &'a CapturedSubmissionPlan,
        /// What the acceptance produced.
        receipt: &'a CapturedSubmissionReceipt,
    },

    /// A completion point changed state.
    CompletionChanged {
        /// This event's identity.
        event: SemanticEventId,
        /// Which point changed.
        point: CompletionPoint,
        /// Its state now, borrowed for the duration of the callback.
        state: &'a CompletionState,
    },

    /// A frame was acquired.
    ///
    /// Presentation remains observable by its portable target/configuration and
    /// frame identities; tooling never serializes a host or native surface
    /// handle.
    FrameAcquired {
        /// This event's identity.
        event: SemanticEventId,
        /// The presentation target.
        target: ObjectId,
        /// The configured presentation lease the frame belongs to.
        configured_presentation: ObjectId,
        /// The acquired frame.
        frame: AcquiredFrameId,
        /// The configuration the frame was acquired under.
        configuration: &'a PresentationConfiguration,
        /// The frame's extent.
        extent: Extent3d,
        /// Whether the surface recommends reconfiguration after this acquire.
        suboptimal: bool,
    },

    /// A presentation's state changed.
    PresentChanged {
        /// This event's identity.
        event: SemanticEventId,
        /// Which present's receipt changed. The receipt is a plain
        /// [`PresentReceiptId`] and not a borrow, because it is a value a
        /// capture has to be able to write down and correlate with the plan it
        /// came from.
        receipt: PresentReceiptId,
        /// Its state now, borrowed for the duration of the callback.
        state: &'a PresentState,
    },

    /// The device was lost.
    ///
    /// Section 52.8's terminal observation, and the one event a capture must
    /// receive if it receives nothing else: after it, every outstanding point,
    /// ticket, frame, and receipt has reached its terminal state, and a capture
    /// that has not seen it is entitled to keep waiting for something the RHI
    /// already knows will not complete.
    DeviceLost {
        /// This event's identity.
        event: SemanticEventId,
        /// Why, borrowed for the duration of the callback.
        info: &'a DeviceLossInfo,
    },

    /// A diagnostic was reported.
    ///
    /// The same [`DiagnosticEvent`] a caller drains through
    /// `Device::drain_diagnostics`, delivered by subscription so that a tool does
    /// not have to poll. Draining and subscribing are both offered because they
    /// answer different needs — a caller wants a bounded drain at a frame
    /// boundary, and a tool wants everything — and neither replaces the other.
    Diagnostic {
        /// This event's identity.
        event: SemanticEventId,
        /// The diagnostic, borrowed for the duration of the callback.
        diagnostic: &'a DiagnosticEvent,
    },
}

impl SemanticEvent<'_> {
    /// This event's identity.
    ///
    /// The accessor exists because of section 53.2's ordering contract: a
    /// subscription receives events "exactly once and in increasing
    /// [`SemanticEventId`] order", and a caller cannot check a rule it cannot
    /// read. It is a method rather than a public field because the identity is
    /// already a named field inside every variant, and hoisting it into a struct
    /// wrapper would break the `match` shape section 58.1 declares.
    ///
    /// It borrows nothing: the returned id is `Copy`, so an observer that wants
    /// only the ordering may take it and let the event go.
    pub fn event_id(&self) -> SemanticEventId {
        match self {
            Self::ObjectCreated { event, .. }
            | Self::ObjectReclaimed { event, .. }
            | Self::UploadDefined { event, .. }
            | Self::ReadbackDefined { event, .. }
            | Self::WorkFinished { event, .. }
            | Self::SubmissionAccepted { event, .. }
            | Self::CompletionChanged { event, .. }
            | Self::FrameAcquired { event, .. }
            | Self::PresentChanged { event, .. }
            | Self::DeviceLost { event, .. }
            | Self::Diagnostic { event, .. } => *event,
        }
    }
}
