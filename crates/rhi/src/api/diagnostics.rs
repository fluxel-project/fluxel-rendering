//! Diagnostic events and label canonicalization (specification section 48).
//!
//! **This file is being written in specification order.** It owns the diagnostic
//! severity vocabulary, the diagnostic event a caller can observe, and the pull
//! verb that drains the device's queue of them.
//!
//! It deliberately does not own logging, sinks, formatting, level filtering, or
//! threading policy. The RHI emits structured events and a host decides what to
//! do with them; a library that formats its own log lines fixes a host's output
//! policy from inside the library, and a library that takes a callback fixes the
//! host's threading policy. Section 48's own comment on `drain_diagnostics` —
//! *"Pull model; avoids imposing a callback threading policy"* — is the whole
//! reason the verb takes an output vector instead of an observer.
//!
//! # Canonicalization (section 48.1)
//!
//! Section 48.1 states a rule that reaches across the whole specification: every
//! descriptor that participates in a compatibility id, a fingerprint, a capture
//! definition, or a statistics identity comparison must be canonicalized first,
//! and it splits descriptors into two classes that must not be treated alike.
//!
//! ```text
//! set-like          stable sort, and reject semantic duplicates or deduplicate
//!                   (as each type defines it)
//!                   TextureDescriptor.view_formats, DeviceRequirements
//!                   required/preferred feature sets, ShaderInterface
//!                   resources/inputs/outputs, BindGroupLayout entries,
//!                   BindGroup entries
//!
//! ordered-semantic  retain semantic order; must NOT be sorted
//!                   PipelineInterface.groups, color target locations,
//!                   Submission batches/work, commands
//! ```
//!
//! The *rule* is stated once, here. The *implementation* is not here, and
//! deliberately so: what counts as a semantic duplicate is decided by the type
//! that owns the vector — two `TextureFormat` entries for the same format are
//! duplicates, while two `PipelineInterface` groups with the same contents are
//! not interchangeable at all — so a canonicalizer that lived in this module
//! would have to know every descriptor in the crate. Each owning module exposes
//! a `pub(crate) fn canonicalize_*` next to the type it canonicalizes, and the
//! hashing code calls it before hashing.
//!
//! The last line of section 48.1 is the boundary that keeps labels from becoming
//! load-bearing: `Label` and diagnostic strings do **not** participate in
//! compatibility or fingerprint correctness. [`DiagnosticEvent::label`] is
//! therefore text for a human, and nothing in this module may be read as a
//! canonicalization of the object's identity.

use crate::api::identity::ObjectId;
use crate::api::platform::device::Device;
use crate::api::platform::requirements::OptionalFeature;
use crate::api::{RhiError, RhiErrorKind, RhiResult};

/// Reliability of allocator information reported by a backend.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocatorReportQuality {
    /// Exact values from the native allocator or memory-budget API.
    Exact,
    /// Values estimated by the backend from allocations it owns.
    Estimated,
    /// The platform exposes no trustworthy value for this field.
    Unknown,
}

/// One portable allocator heap-class observation.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocatorHeapReport {
    /// Backend-neutral heap class, such as `device-local` or `upload`.
    pub class: String,
    /// Bytes currently allocated to live RHI resources, if known.
    pub allocated_bytes: Option<u64>,
    /// Bytes reserved by allocator blocks, if known.
    pub reserved_bytes: Option<u64>,
    /// Bytes backed by native committed memory, if the platform distinguishes
    /// commitment from virtual-address or heap reservation.
    pub committed_bytes: Option<u64>,
    /// Bytes currently resident in device-visible physical memory, if known.
    ///
    /// This is an observation, not a promise that the same bytes remain resident
    /// after the report is returned.
    pub resident_bytes: Option<u64>,
    /// Logical live bytes sharing physical storage through aliasing, if known.
    /// This may exceed physical committed bytes and must not be added to them.
    pub aliased_bytes: Option<u64>,
    /// Bytes retained only because accepted GPU work has not reached its
    /// retirement point, if the backend tracks that quantity.
    pub retired_bytes: Option<u64>,
    /// Number of allocator blocks, if known.
    pub block_count: Option<u64>,
    /// Number of live allocations, if known.
    pub allocation_count: Option<u64>,
    /// Budget visible to the platform, if known.
    pub budget_bytes: Option<u64>,
    /// How the values were obtained.
    pub quality: AllocatorReportQuality,
}

/// Optional backend allocator/memory diagnostic snapshot.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocatorReport {
    /// Heap classes observed by the backend.
    pub heaps: Vec<AllocatorHeapReport>,
}

/// How serious a diagnostic is.
///
/// Three levels, not five. The RHI has no `Fatal` — device loss is a state a
/// caller reads from [`crate::api::platform::device::Device::status`], not a
/// severity a message may declare — and no `Trace`, because per-command tracing
/// is what the tooling surface is for.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// Something the caller may want to know, and does not need to act on.
    Info,
    /// Something that is legal but probably not what was meant.
    Warning,
    /// Something was refused, or the device is no longer able to do it.
    Error,
}

/// One diagnostic message emitted by the RHI.
///
/// A pull-model record rather than a log line: the fields are structured, so a
/// host can filter, count, and display them without parsing text, and the
/// message is the only free-form part.
///
/// Every field except [`Self::severity`] and [`Self::message`] is optional,
/// because a diagnostic is emitted wherever the RHI noticed something and not
/// every site knows which object or which operation it was about. What that
/// means for a reader: an event with no object is a statement about the device
/// or the call, not a statement about an unknown object.
///
/// `#[non_exhaustive]`, per section 47.20.1's rule for externally returned data
/// that may gain fields.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DiagnosticEvent {
    /// How serious this is.
    pub severity: DiagnosticSeverity,
    /// The human-readable message. Not stable text; do not match on it.
    pub message: String,
    /// The object this is about, when it is about exactly one.
    pub object: Option<ObjectId>,
    /// The label the caller attached to that object, when it had one.
    ///
    /// Carried here rather than looked up by the reader so that a diagnostic
    /// remains meaningful after the object is reclaimed and its label is gone.
    pub label: Option<String>,
    /// The RHI operation that emitted this, as a compile-time call-site name.
    ///
    /// `&'static str` for the reason [`crate::api::error::RhiError::operation`]
    /// gives: it names a fixed call site, costs no allocation on a path that may
    /// run per command, and cannot drift into carrying caller data.
    pub operation: Option<&'static str>,
    /// Whatever the backend wanted to add.
    ///
    /// For diagnostics only. Section 48 states that backend detail does not
    /// participate in portable correctness, so this field may contain anything a
    /// driver said and a caller may not branch on it. It is a `String` rather
    /// than a typed value for exactly that reason: giving it structure would
    /// invite a caller to depend on it.
    pub backend_detail: Option<String>,
}

impl Device {
    /// Returns a native allocator report when the selected backend exposes one.
    ///
    /// This is diagnostic data, never an allocation policy input.  In
    /// particular, `Unknown` does not mean zero and callers must not reject a
    /// resource based on it.
    pub fn allocator_report(&self) -> RhiResult<AllocatorReport> {
        self.require_active()?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::AllocatorReport)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "native allocator diagnostics are not enabled on this device",
            )
            .at("Device::allocator_report"));
        }
        self.native().allocator_report()
    }
    /// Moves every diagnostic emitted since the last drain into `out`.
    ///
    /// Appends rather than replaces, so a caller may accumulate across several
    /// devices or drain into a shared buffer. The queue is emptied by the call:
    /// a diagnostic is delivered once, and a caller that wants to keep it must
    /// keep the event.
    ///
    /// Non-blocking and never falls back to stderr. A library that also printed
    /// its own diagnostics would double-report once a host installed its own
    /// sink, and would write to a stream the host may own.
    ///
    pub fn drain_diagnostics(&self, out: &mut Vec<DiagnosticEvent>) {
        let mut queued = self
            .diagnostics_queue()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        out.append(&mut queued);
    }

    /// Records one portable diagnostic on this device.
    ///
    /// Kept crate-private so backend and validation paths can report a fact
    /// without making diagnostics a caller-controlled logging channel.
    pub(crate) fn report_diagnostic(&self, event: DiagnosticEvent) {
        self.diagnostics_queue()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event.clone());
        self.dispatch_diagnostic(&event);
    }
}
