//! Thread-affine native context carrier.

use crate::backend::gl::api::{GlContextLifecycle, GlError, OwnerThreadIdentity};
use core::marker::PhantomData;
use std::rc::Rc;

/// Which platform family owns the current native context.
///
/// This is diagnostic provenance only.  It deliberately does not carry a WGL
/// or EGL handle: those handles have platform-specific destruction/currentness
/// rules and must remain in the platform adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeContextKind {
    /// A desktop context adopted from WGL/GLX/CGL or an equivalent host.
    Desktop,
    /// An ES context adopted from EGL or an equivalent host.
    Embedded,
}

/// Proof, supplied by the platform adapter, that its native context is current
/// on the calling thread for the duration of the GL executor's use.
///
/// The proof is `!Send` and can only be created through an unsafe constructor:
/// Rust cannot inspect WGL/EGL currentness, and making an ordinary constructor
/// would turn an external platform invariant into a guess.  The host keeps
/// ownership of adopted contexts; an owned context is represented by the same
/// guard after its platform owner has made it current.
#[derive(Debug)]
pub(crate) struct CurrentContextGuard {
    owner: OwnerThreadIdentity,
    kind: NativeContextKind,
    _thread_affine: PhantomData<Rc<()>>,
}

impl CurrentContextGuard {
    /// # Safety
    ///
    /// `kind`'s native context must be current on this thread, exclusively
    /// usable by Fluxel until the guard is dropped, and remain alive longer
    /// than every `NativeContext` created from this guard.
    pub(crate) unsafe fn assume_current(kind: NativeContextKind) -> Self {
        Self {
            owner: OwnerThreadIdentity::current(),
            kind,
            _thread_affine: PhantomData,
        }
    }

    pub(crate) const fn kind(&self) -> NativeContextKind {
        self.kind
    }

    fn assert_owner(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner {
            Ok(())
        } else {
            Err(GlError::WrongThread {
                operation,
                expected: self.owner,
                actual,
            })
        }
    }
}

/// A `glow` dispatch table plus the non-transferable current-context proof.
///
/// This does not call `wglMakeCurrent` or `eglMakeCurrent`.  Platform adapters
/// own that transition and must construct this object only while current.  The
/// explicit guard prevents a native GL object from accidentally gaining the
/// browser/WebGL object's cross-thread shape.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) struct NativeContext<'a> {
    gl: &'a glow::Context,
    guard: CurrentContextGuard,
    lifecycle: GlContextLifecycle,
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl<'a> NativeContext<'a> {
    /// Adopts a host-owned or platform-owned current context.
    ///
    /// The context stays borrowed: Fluxel cannot destroy a WGL/EGL context it
    /// did not create.  A future platform owner may retain its own handle and
    /// use this same adopted carrier after making it current.
    pub(crate) fn adopt_current(gl: &'a glow::Context, guard: CurrentContextGuard) -> Self {
        Self {
            gl,
            guard,
            lifecycle: GlContextLifecycle::Active,
        }
    }

    pub(crate) fn glow(&self, operation: &'static str) -> Result<&glow::Context, GlError> {
        self.assert_ready(operation)?;
        Ok(self.gl)
    }

    pub(crate) const fn kind(&self) -> NativeContextKind {
        self.guard.kind()
    }

    pub(crate) const fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }

    /// Records terminal loss observed by a native call.  It is intentionally
    /// explicit: GL's error flag alone cannot distinguish an ordinary command
    /// error from a platform-declared context reset.
    pub(crate) fn mark_lost(&mut self) {
        self.lifecycle = GlContextLifecycle::Lost;
    }

    pub(crate) fn assert_ready(&self, operation: &'static str) -> Result<(), GlError> {
        self.guard.assert_owner(operation)?;
        if self.lifecycle.accepts_commands() {
            Ok(())
        } else {
            Err(self.lifecycle.refusal(operation))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CurrentContextGuard, NativeContextKind};

    #[test]
    fn currentness_guard_preserves_platform_provenance() {
        // SAFETY: this test does not issue GL; it tests only thread affinity.
        let guard = unsafe { CurrentContextGuard::assume_current(NativeContextKind::Embedded) };
        assert_eq!(guard.kind(), NativeContextKind::Embedded);
        assert!(guard.assert_owner("test").is_ok());
    }
}
