//! Crate-private backend contract: the native entry point behind one
//! [`ShaderModule`](crate::api::shader::ShaderModule).
//!
//! A separate module from [`crate::api::resource::backend`] for the reason that one is
//! separate from [`crate::api::platform::backend`]: a shader module is reached from its
//! own handle and grows for its own reason. The resource seam carries objects a
//! command records against; this one carries the compiler's answer about an entry
//! point, which later chapters read when they build a pipeline state.
//!
//! # What this seam deliberately does not do
//!
//! It carries no `create`-shaped method. The creation call lives on
//! [`DeviceBackend`](crate::api::platform::backend::DeviceBackend), next to
//! `create_buffer`, because the device is what performs it and because the portable
//! layer's rules — section 19.10's acceptance verdict, the identity, the retained
//! artifact — all sit *before* it. A trait method here that took an artifact would
//! be a second place a module could be created, and the second place is where the
//! acceptance verdict gets skipped.
//!
//! It also carries no "compile" or "validate" verb. Whether the bytes a producer
//! handed over are a legal program is a question only the native compiler can
//! answer, and the answer arrives where the native API says it does — which for
//! Direct3D 12 is *not* at module creation (see `Dx12ShaderModule`'s note). A verb
//! here named `validate` would promise a check this layer cannot perform.

use std::any::Any;

/// The native entry point behind one
/// [`ShaderModule`](crate::api::shader::ShaderModule).
///
/// Implemented by a backend, held by the portable handle, and never reachable from
/// outside the crate. Like [`crate::api::resource::backend::BufferBackend`] it carries the
/// object and not the operations: a pipeline is built by *the device's* backend,
/// which downcasts both this and every other stage's module in one place, so a
/// method here would put one stage's operations behind an arbitrary receiver.
pub(crate) trait ShaderModuleBackend: Send + Sync + 'static {
    /// This entry point as an opaque native object.
    ///
    /// The downcast's first caller will be the DX12 pipeline lowering, which
    /// reaches a backend's own shader type from each stage's portable handle to
    /// fill a `D3D12_SHADER_BYTECODE`. That lowering is not written, so this method
    /// has no caller in any configuration but a test one — which is why the gate
    /// below is `test` alone rather than the backend feature list
    /// [`crate::api::resource::backend::BufferBackend::as_any`] uses. Claiming a backend
    /// reads it would be a claim that is not yet true of any backend.
    ///
    /// When the lowering lands this `expect` sits unfulfilled in that
    /// configuration and the build fails, which is the intended forcing function:
    /// the attribute then becomes the `not(any(test, feature = "dx12"))` shape its
    /// sibling has, and rule 4.6's "the matrix gets the row" is applied to the
    /// attribute itself.
    #[cfg_attr(
        all(not(test), not(any(feature = "dx12", feature = "vulkan"))),
        expect(dead_code, reason = "read by backend pipeline lowering")
    )]
    fn as_any(&self) -> &dyn Any;
}
