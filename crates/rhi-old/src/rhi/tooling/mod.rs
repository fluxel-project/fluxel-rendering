//! Tooling SPI: observation, object/work description, and portable IR
//! (rhi-design sections 52 through 58).
//!
//! # What it is
//!
//! The versioned surface a RenderGraph diagnostic, a capture coordinator, or a
//! replay debugger uses to see what this device did — in the device's own
//! portable terms, never in native ones:
//!
//! ```text
//! ToolingAccess::subscribe        an observer of semantic events
//! ToolingAccess::describe_object  lazy definition of a live object
//! ToolingAccess::describe_work    complete portable semantics of live work
//! ```
//!
//! It is deliberately *not* on the ordinary public surface. The stable 0.16
//! public API is the portable execution vocabulary; this is a separately
//! versioned seam (section 64), it is `#[doc(hidden)]`, and it changes on
//! [`ToolingSpiVersion`] rather than on the crate's semver. A release build that
//! enables no observer pays for no observer: a backend checks the registry's
//! observer count before it builds a definition.
//!
//! The design note suggests an independent `rhi-tooling` cargo feature for this
//! surface. The module is exposed `#[doc(hidden)]` without a feature instead,
//! because the cost the feature would avoid is a lazy definition build, and that
//! is already gated on the observer count: no feature combination can make an
//! unobserved device materialize a definition.
//!
//! # The two halves of the seam
//!
//! *Consumers* hold a [`ToolingAccess`] and a [`ToolingSubscription`].
//! *Backends* implement the crate-private `ToolingBackend`, which supplies
//! definitions lazily and owns one `ToolingRegistry` per device identity. The
//! registry — the observer set, the delivery gate, the event-id mint, and the
//! callback-contract enforcement — lives in this module, so every backend gets
//! one implementation of the linearization rules instead of one per backend.
//! Until a backend implements the seam, `Device::tooling` still returns a
//! `ToolingAccess`: it answers `RhiErrorKind::Unsupported` rather than
//! accepting observers it would never call.
//!
//! # Linearization
//!
//! `subscribe` returns only after the observer is in the device's observer set.
//! The subscription then receives exactly once, in increasing
//! [`SemanticEventId`] order, every event assigned after that point and before
//! its `Drop` unregistration point, and none assigned before it. `Drop` waits
//! for every callback that began before unregistration to return, and after it
//! returns no callback can begin or remain active.
//!
//! Three cases are *prohibited* and are refused rather than deadlocked, because
//! a callback runs while its thread holds the delivery gate: an observer
//! dropping its own subscription from `on_event`, a callback subscribing (or
//! emitting) re-entrantly on its own device, and a callback panicking. Each is
//! counted — see [`ToolingAccess::violations`] — because a refusal nobody can
//! observe is indistinguishable from silence. The mechanism is the lock layout
//! documented in the `registry` submodule: a gate held across callbacks, a
//! separately held observer set, and a gate-holder thread id readable without
//! the gate.
//!
//! # Ordering
//!
//! [`SemanticEventId`] is **CPU observation order, not GPU execution order**.
//! Under a multi-threaded recorder, event 100 preceding event 101 does not mean
//! GPU work 100 happens before 101. Real GPU order comes only from
//! [`PortableCommand`] order within a recording, submission-plan lane order,
//! explicit dependencies, and the present relation — all of which are frozen
//! here so that a replay never has to infer them from a schedule.
//!
//! # What this module deliberately does not own
//!
//! RHI makes its portable semantics observable, describable, and
//! reconstructible. It does not build a capture product (sections 58.3 to 58.6):
//!
//! ```text
//! not an artifact layer: no magic, schema, chunk, manifest, compression,
//!                        dedup, blob hashing, signature, encryption, migration
//! not dependency closure: which producers a capture pass also needs, whether a
//!                        persistent texture must be snapshotted, how many
//!                        frames of history to include
//! not snapshot policy:   which point, which subresource, full bytes or hash,
//!                        size budget, redaction, external fixture
//! not ReplayRuntime:     artifact parsing, schema migration, target selection,
//!                        capability negotiation, shader/object/command/
//!                        submission rebuild, diffing, a step debugger
//! ```
//!
//! [`TOOLING_SPI_VERSION`] is not a capture artifact schema version. A consumer
//! records both numbers separately.
//!
//! # No native shapes
//!
//! Nothing in this module may name a native handle, pointer, OS handle, GPU
//! virtual address, descriptor heap index, absolute host-memory address, barrier
//! bit, or credential (section 58.8), and no Rust discriminant or memory layout
//! here is an encoding (section 56). Every identity in a definition, an event,
//! or a command is an [`ObjectId`], an [`AcquiredFrameId`], or a runtime id that
//! the artifact layer remaps to its own capture-local id; every other field is a
//! portable value type. A definition is exactly the descriptor the caller
//! created the object from, with live handles replaced by the ids that name
//! them.
//!
//! [`ObjectId`]: crate::rhi::platform::ObjectId
//! [`AcquiredFrameId`]: crate::rhi::presentation::AcquiredFrameId

mod access;
mod commands;
mod events;
mod objects;
mod registry;
mod spi;
mod work;

pub use access::{SemanticObserver, ToolingAccess, ToolingSubscription, ToolingViolationCounts};
pub use commands::{
    CapturedBlit, CapturedBufferCopy, CapturedBufferTextureCopy, CapturedColorAttachment,
    CapturedColorAttachmentView, CapturedCommand, CapturedDepthStencilAttachment,
    CapturedReadbackRequest, CapturedRasterScope, CapturedResolve, CapturedResourceUse,
    CapturedTextureCopy, CapturedUploadDefinition, PortableCommand,
};
pub use events::{SemanticEvent, SemanticEventId};
pub use objects::{
    CapturedBindGroupDefinition, CapturedBindGroupEntry, CapturedBindingResource,
    CapturedComputePipelineDefinition, CapturedConfiguredPresentationDefinition,
    CapturedObjectDefinition, CapturedPipelineInterfaceDefinition,
    CapturedPresentationTargetDefinition, CapturedPresentationTargetFixture,
    CapturedRasterPipelineDefinition,
};
pub use spi::{TOOLING_SPI_VERSION, ToolingSpiVersion};
pub use work::{
    CapturedDependencySource, CapturedPlanDependency, CapturedPresentPlan, CapturedRecordedWork,
    CapturedSubmissionBatch, CapturedSubmissionPlan, CapturedSubmissionReceipt,
};

pub(crate) use registry::ToolingBackend;

#[cfg(test)]
mod tests;
