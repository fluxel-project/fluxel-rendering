//! Lowering one Layer 2 failure into the error the common contract carries.
//!
//! Responsibility: be the single place where a [`StateError`] becomes a
//! [`GlError`], so that no verb invents its own mapping.  It is total and it
//! loses nothing that the contract can express: a backend failure is already the
//! backend's own `GlError` and is handed through unchanged, and the two decisions
//! Layer 2 makes before any side effect map one-to-one.
//!
//! Not owned here: the messages themselves (each domain writes its own), and the
//! decision to *stop* at a failure (the domains make it, and report how far they
//! got).
//!
//! # The two facts the common contract has no room for
//!
//! Layer 2's error carries two facts that `GlError` does not: which state domain
//! failed, and how much of that domain's group had already been emitted
//! (`StateError::Backend`'s `applied`).  Both are dropped, and the reason is that
//! neither is a decision the caller of this adapter has left to make.  A failed
//! group is marked entirely unknown and is re-applied in full at the next
//! reconcile, so the count is a diagnostic and not a repair instruction -- and the
//! domain is derivable from the operation name the caller already has, which is
//! why the operation is carried through rather than the domain.
//!
//! # Why the operation name survives
//!
//! Every variant keeps the `operation` Layer 2 recorded, and this layer does not
//! substitute one of its own.  The name a verb would choose is the name of the
//! *contract verb* (`set-raster-pipeline`), while the name Layer 2 recorded is
//! the *domain entry point* it actually failed in, which is the more specific
//! fact and the one a differential trace keys on.  So a verb here reports what
//! failed rather than what it was asked to do, and the verb's own name is still
//! reachable -- it is the call the caller made.

use crate::webgl2::api::GlError;
use crate::webgl2::state::StateError;

/// A capability this family's vocabulary has no case for.
///
/// The reason is a `&'static str` because it is a fact about the backend rather
/// than about the request: the same sentence is true of every caller, and a
/// caller cannot fix it.
pub(super) fn unsupported(operation: &'static str, reason: &'static str) -> GlError {
    GlError::Unsupported { operation, reason }
}

/// A request the frame made that the thing it named cannot be used for.
///
/// Distinct from [`unsupported`] on purpose: nothing about the family is missing
/// here -- the request and the object disagree, and a caller that fixed the
/// request could make the same call succeed.  The message is owned because it
/// names the disagreement.
pub(super) fn malformed(operation: &'static str, message: &str) -> GlError {
    GlError::Validation {
        operation,
        message: message.to_owned(),
    }
}

/// A pass was asked to open while one was already open.
///
/// Stated once because three verbs can report it -- a second `begin_raster`, a
/// second `begin_compute`, and a `finish_encoder` with one still open -- and they
/// have to agree on what the mistake is.
///
/// The sentence is deliberately kind-neutral.  It used to say "a raster pass",
/// which was true while a raster pass was the only kind this adapter could open;
/// now that a compute pass takes the same single slot, naming one kind would make
/// the refusal wrong for exactly the caller that needs it.  What the mistake is is
/// that a pass is open, and *which* pass is a fact the caller already has.
pub(super) fn pass_open(operation: &'static str) -> GlError {
    malformed(
        operation,
        "a pass is already open on this encoder, and this family's context runs one pass at a time",
    )
}

/// A verb that needs an open pass was called without one.
///
/// The sentence is deliberately kind-neutral, on [`pass_open`]'s terms and for the
/// same reason.  It used to say "no raster pass is open ... and a draw has no
/// framebuffer to render into", and both halves of that have since stopped being
/// true of the callers that reach here:
///
/// - The *kind* is wrong for the shared verbs.  `set_bindings` records into either
///   kind of pass and reads this encoder through `GlEncoder::open`, so a compute
///   encoder reaching it was told about a raster pass that was never the subject.
/// - The *second clause* was only ever true of one caller.  `draw` is one of four
///   that land here -- the other three are the raster verbs, `set_bindings`, and
///   `end_raster` -- and a close without a pass has no framebuffer to miss.
///
/// What the mistake is is that no pass is open, and which one the caller wanted is
/// a fact the caller already has: it is the verb named in the error.
pub(super) fn no_pass(operation: &'static str) -> GlError {
    malformed(operation, "no pass is open on this encoder")
}

/// `end_compute` was called with no compute pass open.
///
/// A second function rather than [`no_pass`] with a different message, and the
/// reason is narrower than it first looks: [`no_pass`] may not name a pass *kind*
/// because a shared verb reaches it, while this one is reached by `end_compute`
/// alone, so naming the kind here is a fact about the caller rather than a guess
/// about it.  What it adds over [`no_pass`] is what the caller is missing -- a
/// close does not want a pass to record into, it wants a boundary to close.  The
/// executor brackets every pass it opens, so reaching this means a caller closed
/// twice or never opened.
pub(super) fn no_compute_pass(operation: &'static str) -> GlError {
    malformed(
        operation,
        "no compute pass is open on this encoder, so there is no boundary to close",
    )
}

/// The common contract's error for one Layer 2 failure.
///
/// Exhaustive over `StateError` rather than wildcarded: the type is this crate's
/// own and is not `#[non_exhaustive]`, so a new variant is a change to Layer 2's
/// contract and should stop this lowering rather than fall into a default.
pub(super) fn into_gl_error(error: StateError) -> GlError {
    match error {
        // Layer 2 refused the request before any side effect, which is the
        // contract's `Validation` and not its `Unsupported`: something about the
        // request is wrong, and asking again after fixing it can succeed.
        StateError::Rejected {
            operation, message, ..
        } => GlError::Validation { operation, message },
        // A capability fact is missing, which is the contract's `Unsupported` in
        // its own words and needs no translation.
        StateError::Unsupported {
            operation, reason, ..
        } => GlError::Unsupported { operation, reason },
        // The backend already answered in this vocabulary.  Re-wrapping it in a
        // `Driver` variant would replace a structured failure -- a validation
        // failure, a stale object, an out-of-memory -- with prose about one.
        StateError::Backend { source, .. } => source,
    }
}
