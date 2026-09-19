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
//! The four modules here are the two readings every later slice depends on and
//! none of them can supply afterwards, the type that carries them, and the one
//! lowering that has to exist before the adapter can record anything:
//!
//! - [`identity`] maps a GL-family context generation onto the common device
//!   identity.
//! - [`capabilities`] lowers a discovery snapshot onto the common capability
//!   contract.
//! - [`device`] is the adapter itself, and owns the contract's associated types.
//! - [`shader`] lowers the retained path's fixed raster artifacts onto Layer 1's
//!   shader contract, which is what the device's raster objects are built from.
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
mod shader;

/// The measurement entry, re-exported here because `device` is private.
///
/// A path through `device` is unnameable from outside this module, so the
/// crate's doc-hidden `test_support` surface cannot reach the harness without
/// this hop.  It is the same kind of allowance `api::NativeGlProvider` needed
/// and not a new contract: nothing public is introduced, and the entry stays
/// behind the `test-support` feature it is built for.
#[cfg(all(
    feature = "test-support",
    any(
        all(target_os = "windows", feature = "native-gl-wgl"),
        all(not(target_arch = "wasm32"), feature = "native-gles-egl")
    )
))]
pub use device::harness::{ColourReadback, DomainTally, NativeGlDrawReport};

#[cfg(all(
    target_os = "windows",
    feature = "native-gl-wgl",
    feature = "test-support"
))]
pub use device::harness::drive_desktop_gl4_draws;

/// The measurement entry for the offscreen GLES surface.
///
/// The same hop as above and the same allowance: `device` and `harness` are
/// private, so a path to this entry exists only from inside the crate, and the
/// entry stays behind the `test-support` feature it is built for.  It is gated
/// on `native-gles-egl` and a non-wasm target rather than on Windows, which is
/// the whole reason the vocabulary above moved out of the Windows gate.
#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "native-gles-egl",
    feature = "test-support"
))]
pub use device::harness::drive_gles_pbuffer_draws;

/// The shared measurement workload, for the browser surface.
///
/// The same allowance as above, one surface over.  The browser draw test cannot
/// live inside either layer it joins: `compat` may not name a browser type (this
/// module's own doc says so, and `scripts/check_gl_architecture.py` enforces it),
/// and `api/browser` may not name `compat`.  So the test sits above both, where
/// it may name the browser provider, and reaches the one workload through this
/// hop.  `device` and `workload` are both private, so the path exists only inside
/// the crate and nothing public is introduced.
///
/// `test` is part of the gate because the only consumer is that test: a
/// re-export that exists in a build where nothing can name it is an unused
/// import, and the crate denies warnings.
#[cfg(all(test, target_arch = "wasm32", feature = "webgl2"))]
pub(crate) use device::workload::{DrawCost, drive as drive_workload};
