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
//! What exists here so far is the one part every later slice depends on and
//! none of them can supply afterwards: the mapping from a GL-family context
//! generation onto the common device identity.  The command-lowering slices
//! follow, and the contract's associated types are deliberately left to the
//! first of them -- they are fixed by what the lowering needs, so choosing them
//! before that exists would be choosing them from imagination.

mod identity;
