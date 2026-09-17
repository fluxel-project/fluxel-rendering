//! Safe, platform-neutral command vocabulary for GL-family implementations.
//!
//! This module deliberately contains no native or browser handle types. It is
//! the only Layer 1 surface consumed by the future state machine.

#![allow(
    unused_imports,
    reason = "Layer 1 re-exports are the stable private seam for later state and compat layers"
)]

mod binding;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
mod browser;
mod compute;
mod copy;
mod discovery;
#[cfg(all(not(target_arch = "wasm32"), feature = "native-gles-egl"))]
mod egl;
mod error;
mod extensions;
mod formats;
mod framebuffer;
mod indirect;
mod limits;
#[cfg(test)]
mod mock;
mod multi_draw;
#[cfg(all(
    not(target_arch = "wasm32"),
    any(feature = "native-gl-wgl", feature = "native-gles-egl")
))]
mod native;
mod object;
mod presentation;
mod profile;
mod query;
mod raster;
mod resource;
mod sampler;
mod shader;
mod storage;
mod sync;
mod vertex;
#[cfg(all(target_os = "windows", feature = "native-gl-wgl"))]
mod wgl;

/// The context owner, reachable by the one module that opens a context at all.
///
/// It is `pub(crate)` and stops there.  Until `webgl2::conformance` existed,
/// this type and its `open` were named nowhere outside their own module: the
/// layer above owns the decision to open
/// a context, so the layer that knows how belongs to it rather than to the
/// world.  Nothing here exposes a HAL type, a browser object, or a `Backend`
/// variant, and no public path leads to it.
#[cfg(all(target_os = "windows", feature = "native-gl-wgl"))]
pub(crate) use wgl::WglContextSurface;

/// The same three things for the EGL provider, which had none of them.
///
/// `WglContextSurface` above reached `pub(crate)` when the desktop-GL4 entry
/// needed it; `egl` was left behind, and by the time anyone looked,
/// `EglGlesContext::new_pbuffer` had zero callers and no nameable path from
/// outside its own module -- so the GLES half of the GL-family story had no way
/// to open a context at all.  This is the same hop, for the same reason, and it
/// stops in the same place: a context, the version it is asked for, and the
/// offscreen extent it is asked for, all `pub(crate)`, with no public path to
/// any of them.
///
/// The companion type is `EglPbufferSize`, which is how a caller states that
/// extent; nothing here exposes a native display, a window, or an EGL handle.
#[cfg(all(not(target_arch = "wasm32"), feature = "native-gles-egl"))]
pub(crate) use egl::{EglGlesContext, EglGlesVersion, EglPbufferSize};

/// The provider over a current native context, for the same reason as above and
/// one more.
///
/// `conformance` could reach its types without this line because it only opens
/// and observes.  Driving a frame needs the provider itself, and `mod native` is
/// private here, so the type has to be re-exported before any module outside
/// `api` can name it.  It stops at `pub(crate)`: the provider borrows its
/// context, and lending one out for longer than a call is the 0.16 ownership
/// question this crate has deliberately not answered.
#[cfg(all(
    not(target_arch = "wasm32"),
    any(feature = "native-gl-wgl", feature = "native-gles-egl")
))]
pub(crate) use native::NativeGlProvider;

/// The same provider, one platform over, for the browser measurement surface.
///
/// The identical hop and the identical reason: `mod browser` is private, the
/// crate's browser draw test drives a frame rather than observing a context, and
/// a module outside `api` cannot name a type it cannot path to.  It stops at
/// `pub(crate)` for the reason above -- the provider borrows its canvas context
/// for as long as it lives -- and it is reachable only where a browser context
/// exists at all.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(crate) use browser::WebGl2BrowserDiscovery;

pub(crate) use binding::*;
pub(crate) use compute::*;
pub(crate) use copy::*;

pub(crate) use discovery::{
    GlCapability, GlCapabilityFact, GlCapabilitySet, GlContextFlags, GlContextInfo,
    GlDiscoveryBuilder, GlDiscoveryError, GlDiscoverySnapshot, GlOperationProbe, GlSurfaceFacts,
};
pub(crate) use error::{GlContextLifecycle, GlError};
pub(crate) use extensions::{
    CapabilityEvidence, CoreOrExtension, ExtensionProvenance, GlExtensionSet, GlKnownExtension,
};
pub(crate) use formats::{
    GlAstcBlock, GlCompressedColorSpace, GlCompressedFormatInfo, GlCompressedTextureFamily,
    GlFormat, GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable,
    GlFormatTableError,
};
pub(crate) use framebuffer::*;
pub(crate) use indirect::*;
pub(crate) use limits::{GlFiniteF32, GlLimitViolation, GlLimits};
#[cfg(test)]
pub(crate) use mock::{MockCall, MockComputeStorageApi, MockGlFamilyApi};
pub(crate) use multi_draw::*;
#[allow(
    unused_imports,
    reason = "Layer 1 marker exports are consumed by later private layers."
)]
pub(crate) use object::{
    BufferId, BufferObject, ContextEpoch, ContextStamp, DeviceIdentity, FramebufferId,
    FramebufferObject, GlObjectKind, ObjectIdentity, OwnerThreadIdentity, ProgramId, ProgramObject,
    QueryId, QueryObject, RenderbufferId, RenderbufferObject, SamplerId, SamplerObject, ShaderId,
    ShaderObject, SurfaceImageId, SurfaceImageObject, SyncId, SyncObject, TextureId, TextureObject,
    VertexArrayId, VertexArrayObject,
};
pub(crate) use presentation::*;
pub(crate) use profile::{GlFamilyProfile, GlVersion};
pub(crate) use query::*;
pub(crate) use raster::*;
pub(crate) use resource::*;
pub(crate) use sampler::*;
pub(crate) use shader::*;
pub(crate) use storage::*;
pub(crate) use sync::*;
pub(crate) use vertex::*;

/// Minimal owner-context and object lifetime contract shared by all profiles.
pub trait GlFamilyApi {
    /// Returns the normalized profile selected for this context.
    fn profile(&self) -> GlFamilyProfile {
        self.discovery().context().profile()
    }

    /// Returns the current context identity and epoch.
    fn context_stamp(&self) -> ContextStamp {
        self.discovery().context_stamp()
    }

    /// Returns the current context lifecycle.
    fn lifecycle(&self) -> GlContextLifecycle;

    /// Returns the owner-thread identity captured with the context.
    fn owner_thread(&self) -> OwnerThreadIdentity;

    /// Rejects a call made outside the context owner thread.
    ///
    /// Every executable domain entry point must call this before any driver or
    /// browser side effect.
    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError>;

    /// Performs the common owner/lifecycle preflight before any side effect.
    fn assert_ready(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_owner_thread(operation)?;
        match self.lifecycle() {
            GlContextLifecycle::Active => Ok(()),
            GlContextLifecycle::Lost => Err(GlError::ContextLost { operation }),
            GlContextLifecycle::Disposed => Err(GlError::Disposed { operation }),
            GlContextLifecycle::Poisoned => Err(GlError::Poisoned { operation }),
            lifecycle => Err(GlError::InvalidLifecycle {
                operation,
                lifecycle,
            }),
        }
    }

    /// Performs common owner/lifecycle/context-generation object preflight.
    ///
    /// Implementations additionally check their live allocation table and the
    /// object's slot generation before reaching the provider boundary.
    fn validate_object_context(
        &self,
        operation: &'static str,
        object: ContextStamp,
    ) -> Result<(), GlError> {
        self.assert_ready(operation)?;
        let current = self.context_stamp();
        if object.device != current.device {
            Err(GlError::WrongContext {
                operation,
                object,
                current,
            })
        } else if object.epoch != current.epoch {
            Err(GlError::StaleObject {
                operation,
                object,
                current,
            })
        } else {
            Ok(())
        }
    }

    /// Returns the complete immutable evidence bound to the current epoch.
    fn discovery(&self) -> &GlDiscoverySnapshot;

    /// Returns the durable extension ledger from the bound discovery snapshot.
    fn extensions(&self) -> &GlExtensionSet {
        self.discovery().extensions()
    }

    /// Records loss and prevents further object or command operations.
    fn context_lost(&mut self) -> Result<(), GlError>;

    /// Completes recreation and returns the strictly newer context stamp.
    fn context_restored(&mut self) -> Result<ContextStamp, GlError>;
}

#[cfg(test)]
pub(crate) mod tests;
