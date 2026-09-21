//! Backend-private, platform-neutral contracts for the OpenGL family.
//!
//! This layer deliberately has no EGL/WGL/GLX, browser, `glow`, or WebGL handle.
//! Native and browser providers implement these typed contracts above their
//! respective context bindings.  The public RHI never exposes a type from here.
//!
//! The vocabulary is shared by desktop core GL 4.x, GLES 3.x, and WebGL2.  A
//! profile's actual version and extension ledger are part of `GlDiscoverySnapshot`;
//! callers must admit an optional operation only from that evidence, never merely
//! because another GL-family profile has a similarly named entry point.

mod binding;
mod compute;
mod copy;
mod discovery;
mod error;
mod extensions;
mod formats;
mod framebuffer;
mod indirect;
mod limits;
#[cfg(test)]
mod mock;
mod multi_draw;
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

pub(crate) use binding::*;
pub(crate) use compute::*;
pub(crate) use copy::*;
pub(crate) use discovery::*;
pub(crate) use error::{GlContextLifecycle, GlError};
pub(crate) use extensions::*;
pub(crate) use formats::*;
pub(crate) use framebuffer::*;
pub(crate) use indirect::*;
pub(crate) use limits::*;
#[cfg(test)]
pub(crate) use mock::{MockCall, MockComputeStorageApi, MockGlFamilyApi};
pub(crate) use multi_draw::*;
pub(crate) use object::*;
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

/// Minimal context-ownership and object-lifetime contract shared by every GL
/// profile. Providers must run `assert_ready` before each native side effect.
pub(crate) trait GlFamilyApi {
    fn profile(&self) -> GlFamilyProfile {
        self.discovery().context().profile()
    }
    fn context_stamp(&self) -> ContextStamp {
        self.discovery().context_stamp()
    }
    fn lifecycle(&self) -> GlContextLifecycle;
    fn owner_thread(&self) -> OwnerThreadIdentity;
    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError>;
    fn assert_ready(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_owner_thread(operation)?;
        match self.lifecycle() {
            GlContextLifecycle::Active => Ok(()),
            lifecycle => Err(lifecycle.refusal(operation)),
        }
    }
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
    fn discovery(&self) -> &GlDiscoverySnapshot;
    fn extensions(&self) -> &GlExtensionSet {
        self.discovery().extensions()
    }
    fn context_lost(&mut self) -> Result<(), GlError>;
    fn context_restored(&mut self) -> Result<ContextStamp, GlError>;
}
