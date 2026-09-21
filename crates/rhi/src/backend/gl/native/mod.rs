//! Native OpenGL / OpenGL ES context ownership and entry-point evidence.
//!
//! This is deliberately below the platform-neutral [`crate::backend::gl::api`]
//! contract.  WGL and EGL objects never escape this module: a caller hands us a
//! context which it has already made current, and receives a `!Send` executor
//! tied to that thread.  In particular, this is not shared with the WebGL
//! implementation; browser objects have a different lifetime model.
//!
//! The native minimum is desktop GL 4.0 or GLES 3.0.  A newer context is kept
//! at its actual version.  Optional functionality is admitted from *both* the
//! version/extension ledger and the loaded entry-point table, never from a
//! version string alone.

mod context;
mod discovery;
mod driver;
#[cfg(feature = "native-gles-egl")]
pub(crate) mod egl;
mod entry_points;
mod exec_compute;
mod exec_copy;
mod exec_framebuffer;
mod exec_raster;
mod exec_shader;
mod exec_sync;
mod exec_vertex;
mod probe;
mod probes;
mod provider;
mod surface;
mod surface_facts;
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(crate) mod wgl;
mod worker;

pub(crate) use context::{CurrentContextGuard, NativeContext, NativeContextKind};
pub(crate) use driver::{
    NativeGlDriver, NativeGlOwner, NativeOwnedProvider, NativePlatformContext, NativeProviderOwner,
};
pub(crate) use entry_points::{NativeEntryPoint, NativeEntryPoints};
pub(crate) use probe::{NativeProfileError, parse_native_profile};
pub(crate) use provider::NativeGlProvider;
pub(crate) use worker::{NativeOwnerWorker, WorkerStartupError};

// Real GL-family conformance cases live under `crates/rhi/tests` beside the
// other backend fixtures.  The fixture opens its WGL context here, where raw
// window/context ownership is still private, then exercises the public Device
// through the common workload.  It must not turn WGL construction into a
// portable public API merely to make an integration test convenient.
#[cfg(all(test, windows, feature = "native-gl-wgl"))]
#[path = "../../../../tests/gl/wgl_core.rs"]
mod wgl_core_conformance;

/// Adopts an owner-thread route whose context produced `discovery`.
///
/// WGL/EGL adapters call this only after their platform factory has completed
/// discovery on the same context generation that the owner will make current.
/// Keeping the narrowing here prevents either platform from publishing facts
/// assembled from a version string or a different context.
pub(crate) fn adopt_discovered_owner(
    instance: crate::api::identity::DeviceInstanceId,
    name: String,
    discovery: &crate::backend::gl::api::GlDiscoverySnapshot,
    owner: std::sync::Arc<dyn NativeGlOwner>,
) -> crate::api::error::RhiResult<crate::backend::gl::platform::GlProvider> {
    NativeGlDriver::adopt(
        instance,
        crate::api::platform::BackendKind::OpenGl,
        name,
        discovery::v13_capability_snapshot(discovery).into_facts(),
        owner,
    )
}
