//! Logical shader composition, validation, reflection, and source artifacts.
//!
//! This crate currently establishes the workspace boundary only. Its future
//! consumers include Material, standalone compute, and renderer pass templates.
//! Device routes, backend ABI, and native compilation remain owned by RHI.

#![deny(missing_docs)]
