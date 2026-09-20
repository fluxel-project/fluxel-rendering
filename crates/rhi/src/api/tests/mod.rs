//! Contract tests for the portable API surface.
//!
//! Behavioural tests exercise the portable rules and mock-backed default
//! implementations. Shape tests remain useful as realistic caller-side
//! compilation checks: if a call site needs an extra construction step, a
//! lifetime it should not name, or a state precondition it cannot check, the
//! fault is in the interface rather than in the call site.
//!
//! A shape test that stops compiling because the interface changed is this
//! module working as intended. A shape test that stops compiling because the
//! *caller* was rewritten is a wasted test — which is why they are transcriptions
//! of the specification's own examples wherever the specification supplies one.

mod binding;
mod capability;
mod command;
mod error;
mod external;
mod format;
mod identity;
mod pipeline;
mod query;
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

pub(crate) mod mock;

/// Fixtures shared by the chapter test sets.
///
/// A small module rather than a `mod.rs` full of helpers: everything here exists
/// because a handle has a native side that a fixture cannot allocate and a test
/// does not want to. It grows only when a second chapter needs the same shape.
pub(crate) mod fixture;
