//! Portable diagnostics: the pull-model event log.
//!
//! This module owns rhi-design section 48.
//!
//! # What it is
//!
//! A device-scoped, bounded, pull-model buffer of diagnostic events. Pull
//! rather than callback is deliberate: RHI cannot know whether the caller is a
//! game loop, a browser turn, a benchmark harness, or a debugger, and a
//! callback would impose a threading policy and a reentrancy rule on every one
//! of them. The caller decides when to collect.
//!
//! # What it deliberately does not own
//!
//! A diagnostic is never a correctness channel. Backends report *portable*
//! failures as [`RhiError`] and put whatever native detail they want to keep
//! into [`DiagnosticEvent::backend_detail`], which participates in nothing:
//! not compatibility, not fingerprinting, not capture semantics, not a
//! capability answer. Two devices that differ only in their diagnostic strings
//! are the same device.
//!
//! # Why the log is bounded
//!
//! A per-frame failure inside a render loop can produce diagnostics far faster
//! than a human can read them. An unbounded queue would turn a rendering bug
//! into an out-of-memory kill, which is a strictly worse failure and hides the
//! original one. The log therefore keeps the most recent `capacity` events and
//! *says so* — see [`DiagnosticLog::drain`] — instead of silently discarding
//! the evidence that anything was lost.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::platform::{ObjectId, RhiError};

/// How serious a diagnostic is.
///
/// Severity is a reporting classification, not an error class: an `Error`
/// diagnostic does not imply that the operation that produced it failed
/// portably, and a `Warning` does not imply that it succeeded.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticSeverity {
    /// Context worth keeping: an adaptation, a workaround, a fallback taken.
    Info,
    /// Something legal but suspicious, or a portable guarantee that a backend
    /// could only satisfy approximately.
    Warning,
    /// A backend-level failure or an invalid state the backend detected.
    Error,
}

/// One diagnostic observation.
///
/// The portable fields are the ones a caller can act on. `backend_detail` is
/// free-form text for a human and is explicitly outside the portable contract.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DiagnosticEvent {
    /// How serious this is.
    pub severity: DiagnosticSeverity,

    /// The portable message. Never contains a native handle or pointer.
    pub message: String,

    /// The object this is about, when one is known.
    pub object: Option<ObjectId>,

    /// The human-readable label of that object, when it had one.
    ///
    /// A label is copied here rather than looked up later because the object
    /// may be gone by the time anyone reads the event, and a diagnostic that
    /// outlives its subject is exactly when the label is most useful.
    pub label: Option<String>,

    /// The portable operation that produced this, when one is known.
    pub operation: Option<&'static str>,

    /// Backend-specific detail. Participates in nothing portable.
    pub backend_detail: Option<String>,
}

impl DiagnosticEvent {
    /// A diagnostic with no object, label, or backend detail.
    pub(crate) fn new(severity: DiagnosticSeverity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            object: None,
            label: None,
            operation: None,
            backend_detail: None,
        }
    }

    /// Reports a portable failure as an `Error` diagnostic.
    ///
    /// The classification travels in [`RhiError::kind`] where a caller can
    /// branch on it; the diagnostic carries the human-readable half.
    pub(crate) fn from_error(error: &RhiError) -> Self {
        let mut event = Self::new(DiagnosticSeverity::Error, error.message());
        event.object = error.object();
        event.operation = error.operation();
        event
    }

    /// Attaches the label of the object this is about.
    pub(crate) fn with_label(mut self, label: &super::platform::Label) -> Self {
        self.label = label.as_deref().map(str::to_owned);
        self
    }

    /// Attaches backend-specific detail.
    pub(crate) fn with_backend_detail(mut self, detail: impl Into<String>) -> Self {
        self.backend_detail = Some(detail.into());
        self
    }
}

/// The device-scoped diagnostic buffer.
///
/// One log belongs to one device identity, which is why it is owned by the
/// device backend rather than kept in a process-wide registry: diagnostics from
/// a lost device must not appear in a replacement device's stream.
pub(crate) struct DiagnosticLog {
    state: Mutex<LogState>,
    capacity: usize,
}

/// The bounded state behind the lock.
///
/// `dropped` counts events evicted since the previous drain. It is reported as
/// a real event on the next drain rather than exposed as a separate counter,
/// so a caller that only ever calls `drain` cannot miss it.
struct LogState {
    events: VecDeque<DiagnosticEvent>,
    dropped: u64,
}

impl DiagnosticLog {
    /// A log that keeps at most `capacity` pending events.
    ///
    /// A capacity of zero keeps nothing but still counts what was lost, so a
    /// caller that wants "did anything go wrong?" without the text can ask for
    /// it without paying for storage.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(LogState {
                events: VecDeque::new(),
                dropped: 0,
            }),
            capacity,
        }
    }

    /// Appends one event, evicting the oldest when full.
    pub(crate) fn push(&self, event: DiagnosticEvent) {
        let mut state = self.lock();
        if self.capacity == 0 {
            state.dropped = state.dropped.saturating_add(1);
            return;
        }
        while state.events.len() >= self.capacity {
            state.events.pop_front();
            state.dropped = state.dropped.saturating_add(1);
        }
        state.events.push_back(event);
    }

    /// Moves every pending event into `out`, oldest first.
    ///
    /// When events were evicted since the previous drain, a [`Warning`] saying
    /// how many is appended *after* the surviving events, so the caller learns
    /// both what it has and what it lost. The count is not attached to any
    /// particular event because there is no honest way to say which one it
    /// belonged to.
    ///
    /// [`Warning`]: DiagnosticSeverity::Warning
    pub(crate) fn drain(&self, out: &mut Vec<DiagnosticEvent>) {
        let mut state = self.lock();
        out.extend(state.events.drain(..));
        let dropped = core::mem::take(&mut state.dropped);
        if dropped > 0 {
            out.push(DiagnosticEvent::new(
                DiagnosticSeverity::Warning,
                format!("{dropped} diagnostic events were dropped before this drain"),
            ));
        }
    }

    /// How many events are pending.
    pub(crate) fn pending(&self) -> usize {
        self.lock().events.len()
    }

    /// Locks the state, recovering from a poisoned lock.
    ///
    /// A panic in one writer must not permanently disable diagnostics for the
    /// whole device: the buffer is a reporting channel, not an invariant. The
    /// recovered state is a queue whose worst case is a partially written
    /// event, which is a diagnostic problem rather than a correctness one.
    fn lock(&self) -> std::sync::MutexGuard<'_, LogState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl core::fmt::Debug for DiagnosticLog {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DiagnosticLog")
            .field("capacity", &self.capacity)
            .field("pending", &self.pending())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
