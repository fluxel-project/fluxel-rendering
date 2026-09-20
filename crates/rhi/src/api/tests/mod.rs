//! Contract tests for the portable API surface.
//!
//! Two kinds of test live here, and the distinction matters:
//!
//! * **Behavioural tests** exercise the parts of the surface that are already
//!   real — the identity tokens and the error model. They run.
//! * **Shape tests** are the review instrument for the parts whose bodies are
//!   still `unimplemented!()`. They are ordinary functions that are *compiled*
//!   but never called, and they are written as realistic call sites rather than
//!   as assertions about types. Their job is to answer "is this interface usable
//!   from the caller's side" before a backend exists to answer it with
//!   behaviour: if a call site needs an extra construction step, a lifetime it
//!   should not have to name, or a state precondition it cannot check, the fault
//!   is in the interface and the fix is to change the interface, not to write the
//!   call site differently.
//!
//! A shape test that stops compiling because the interface changed is this
//! module working as intended. A shape test that stops compiling because the
//! *caller* was rewritten is a wasted test — which is why they are transcriptions
//! of the specification's own examples wherever the specification supplies one.

mod binding;
mod capability;
mod command;
mod error;
mod format;
mod identity;
mod pipeline;
mod resource;
mod shader;

// Declared ahead of their contents, alongside the chapter files in `api/` that
// are still module notes. A test module with nothing in it compiles and reports
// nothing; the alternative is that the chapter cannot be compiled at all while it
// is being written, and a chapter written without compiling cannot be reviewed.
mod diagnostics;
mod presentation;
mod statistics;
mod submission;
mod tooling;

/// Fixtures shared by the chapter test sets.
///
/// A small module rather than a `mod.rs` full of helpers: everything here exists
/// because a handle has a native side that a fixture cannot allocate and a test
/// does not want to. It grows only when a second chapter needs the same shape.
pub(crate) mod fixture;
