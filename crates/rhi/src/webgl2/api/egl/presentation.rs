//! EGL surface-presentation and family-owner facets.
//!
//! Split from the provider setup code so the EGL file keeps one
//! responsibility per file: `mod.rs` owns context/surface construction and
//! teardown; this file implements the family-owner preflight questions and
//! the presentation domain over the owned surface.

use super::super::{
    ContextStamp, GlContextLifecycle, GlDiscoverySnapshot, GlError, GlFamilyApi, GlSurfaceAcquire,
    GlSurfaceLease, GlSurfacePresentationApi, GlSurfaceSize, OwnerThreadIdentity, SurfaceImageId,
};
use super::{
    EglGlesContext, EglPbufferSize, EglProviderError, EglSurfaceKind, check_drawable_extent,
};

/// Family-owner facet of the EGL surface provider.
///
/// The provider owns the context and surface, so it answers the owner,
/// lifecycle, and discovery questions the presentation domain preflights. GL
/// object domains stay with the borrowed-context executor (`NativeGlProvider`)
/// assembled over `load_glow` while this provider is current.
impl GlFamilyApi for EglGlesContext {
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }

    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner
    }

    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_owner(operation)
    }

    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.snapshot
    }

    /// Records loss durably; every acquire lease is invalidated. The native
    /// context is Host-owned, so nothing beyond the Fluxel-side state changes.
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.assert_owner("context-lost")?;
        self.lifecycle = GlContextLifecycle::Lost;
        let _ = self.surface_leases.invalidate_generation();
        Ok(())
    }

    /// An EGL provider cannot resurrect its own context: the Host recreates it
    /// through the `new_*` constructors, which re-run discovery under a new
    /// stamp instead of pretending the old generation came back.
    fn context_restored(&mut self) -> Result<ContextStamp, GlError> {
        Err(GlError::Unsupported {
            operation: "context-restored",
            reason: "re-create the EGL provider with a newly made context instead",
        })
    }
}

/// Surface presentation over the RHI-owned EGL surface.
///
/// Window surfaces present through `eglSwapBuffers`; pbuffer surfaces are
/// honestly non-presentable. The Host owns window resize events, so a window
/// resize only records the observable surface size; pbuffers recreate their
/// owned EGL surface exactly like `resize` always did.
///
/// # What this domain deliberately does not do (audit P1-11, residue)
///
/// It does not apply the observed [`GlSurfaceSize`] to the GL viewport. That is
/// not an omission a later call here would fix:
///
/// 1. The GL viewport is a single piece of context-global state with no
///    association to a drawable, and this provider does not own the
///    draw-framebuffer binding. A viewport written from an acquisition or a
///    resize would land on whichever framebuffer happened to be bound, and the
///    next pass's own `GlRasterDescriptor` viewport would overwrite it, so the
///    write would be both unordered and ineffective.
/// 2. No Layer-1 pass can target the default framebuffer at all:
///    `GlFramebufferDescriptor::validate` rejects an attachment-less
///    framebuffer, so the window system's own surface has no Layer-1 pass to
///    render through yet. Which viewport a default-framebuffer pass renders
///    with is therefore a Layer 2/3 decision, not a provider decision.
///
/// What this domain does enforce is the fact it owns: the extent EGL reports
/// for the live surface (or the pbuffer extent it was constructed with) is
/// bounded by the context's recorded maximum viewport dimensions, and an
/// extent the context could not map is rejected fail-closed before any lease
/// changes state. Applying a viewport needs a default-framebuffer pass contract
/// that does not exist in this layer; the viewport residue is NOT CLOSED here
/// and is recorded as such.
impl GlSurfacePresentationApi for EglGlesContext {
    fn acquire_surface_image(&mut self) -> Result<GlSurfaceAcquire, GlError> {
        const OP: &str = "acquire-surface-image";
        self.assert_owner(OP)?;
        if self.lifecycle != GlContextLifecycle::Active {
            // A suspended or disposed surface cannot present; report
            // suspension instead of handing out an unbacked lease.
            return Ok(GlSurfaceAcquire::Suspended);
        }
        let size = self
            .query_surface_size()
            .map_err(EglProviderError::into_gl_error)?;
        if size.is_zero() {
            return Ok(GlSurfaceAcquire::Suspended);
        }
        // The observed extent is bounded before a lease is handed out, so a
        // surface this context cannot map never becomes a leased presentation
        // target.
        check_drawable_extent(
            self.snapshot.limits().max_viewport_dimensions,
            [size.width, size.height],
        )
        .map_err(EglProviderError::into_gl_error)?;
        self.present_slot = self.present_slot.wrapping_add(1);
        let image = SurfaceImageId::new(self.stamp, self.present_slot as u32, 0);
        let lease = self.surface_leases.acquire(image, size)?;
        Ok(GlSurfaceAcquire::Lease(lease))
    }

    fn resize_surface(&mut self, size: GlSurfaceSize) -> Result<(), GlError> {
        const OP: &str = "resize-surface";
        self.assert_owner(OP)?;
        // Bounded before the lease invalidation, which is a side effect: an
        // extent this context cannot map must not cost the caller its
        // outstanding leases.
        check_drawable_extent(
            self.snapshot.limits().max_viewport_dimensions,
            [size.width, size.height],
        )
        .map_err(EglProviderError::into_gl_error)?;
        // Resize always invalidates outstanding acquire leases first.
        self.surface_leases.invalidate_generation()?;
        match self.kind {
            EglSurfaceKind::Window { .. } => {
                // The native window resizes with the Host; record nothing but
                // the invalidated generation, like the WGL provider.
                Ok(())
            }
            EglSurfaceKind::Pbuffer(_) => self
                .resize(EglPbufferSize {
                    width: size.width,
                    height: size.height,
                })
                .map_err(EglProviderError::into_gl_error),
        }
    }

    fn suspend_surface(&mut self) -> Result<(), GlError> {
        const OP: &str = "suspend-surface";
        self.assert_owner(OP)?;
        self.surface_leases.invalidate_generation()?;
        self.suspend().map_err(EglProviderError::into_gl_error)
    }

    fn resume_surface(&mut self) -> Result<(), GlError> {
        const OP: &str = "resume-surface";
        self.assert_owner(OP)?;
        self.surface_leases.invalidate_generation()?;
        self.resume().map_err(EglProviderError::into_gl_error)
    }

    fn present_surface(&mut self, lease: GlSurfaceLease) -> Result<(), GlError> {
        const OP: &str = "present-surface";
        self.validate_object_context(OP, lease.image.context)?;
        self.surface_leases.consume(lease)?;
        // The flip happens after the lease is consumed: a rejected swap must
        // not resurrect a consumed lease, and the caller retries with a new
        // acquisition.
        self.present().map_err(EglProviderError::into_gl_error)
    }
}
