//! Layer 3: the common RHI compatibility adapter for the GL family.
//!
//! This module is where the GL-family stack meets the contract the rest of the
//! ecosystem compiles against.  It reaches downward only through Layer 2:
//! `scripts/check_gl_architecture.py` forbids `compat` from naming `renderer`,
//! `residency`, `experimental`, or any browser or platform crate, and `api/**`
//! is in turn forbidden from naming `compat`, so the adapter can be reached
//! neither around the state owner nor from beneath it.  Upward it names
//! `fluxel_rendergraph`, which is the crate that declares the contract it fills
//! in -- stated here because "peer backend" reads like an in-crate arrangement
//! and is not one.
//!
//! **It is not selected by a `Backend` variant, and does not need to be.**  A
//! GL-family context is built from a platform source that the common
//! `DeviceOptions` has no field for -- a provider handle on native, a canvas
//! context in the browser -- so the adapter is built from that source and
//! enters the common boundary through its generic.  The series plan records the
//! measurements behind that decision; the short form is that the boundary's
//! consumer is already generic (`FrameExecutor::new(backend)`, no
//! `dyn ExecutionBackend` anywhere) and that Metal and WebGPU have no `Backend`
//! variant either.
//!
//! The three modules here are the two readings every later slice depends on and
//! none of them can supply afterwards, plus the type that carries them:
//!
//! - [`identity`] maps a GL-family context generation onto the common device
//!   identity.
//! - [`capabilities`] lowers a discovery snapshot onto the common capability
//!   contract.
//! - [`device`] is the adapter itself, and owns the contract's associated types.
//!
//! Both readings are per-context facts the graph validates against before any
//! command is lowered, and neither is reachable from the lowering itself -- a
//! lowering cannot report a capability it was already assumed to have.  Which is
//! why the adapter declares what it can do rather than deriving it: the executor
//! compares the graph's fingerprint against `capabilities()` before it records
//! anything, so a device whose description were computed during recording would
//! be describing itself too late.

mod capabilities;
mod device;
mod identity;
