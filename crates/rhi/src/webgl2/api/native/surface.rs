//! Surface presentation over the drawable the Host owns.
//!
//! This module owns exactly one thing: the executor's half of the presentation
//! domain. The drawable and the flip belong to the platform layer (the WGL and
//! EGL surface providers, which own the window and call the swap), so the
//! executor records the extent the Host reports, hands out one lease per frame,
//! and flushes the frame's commands so that the platform flip that follows shows
//! a complete frame. It never creates, sizes, or swaps a drawable.

use super::super::{
    GlError, GlFamilyApi as _, GlSurfaceAcquire, GlSurfaceLease, GlSurfacePresentationApi,
    GlSurfaceSize, SurfaceImageId,
};
use super::provider::NativeGlProvider;

/// Surface presentation over the drawable of the Host-owned context.
///
/// Resize, suspend, and resume invalidate every outstanding lease before they
/// return, and the extent only ever comes from the Host: native GL has no core
/// query for the default framebuffer's size, so a size this executor has not
/// been told is unknown rather than defaulted.
impl GlSurfacePresentationApi for NativeGlProvider<'_> {
    fn acquire_surface_image(&mut self) -> Result<GlSurfaceAcquire, GlError> {
        const OP: &str = "acquire-surface-image";
        self.assert_ready(OP)?;
        let SurfaceAcquireAttempt::Lease(size) =
            surface_acquire_attempt(self.surface_suspended, self.surface_extent)
        else {
            // A suspended, unreported, or zero-area drawable cannot present;
            // reporting suspension is what keeps a lease from naming an extent
            // the Host never reported.
            return Ok(GlSurfaceAcquire::Suspended);
        };
        // Surface-image identities come from the provider's slot counter, which
        // never reuses a slot inside one context generation, so a stale lease
        // can never match a later acquisition. Sharing the counter with the
        // object tables costs nothing: the identity types are disjoint.
        let slot = self.slot(OP)?;
        let image = SurfaceImageId::new(self.context_stamp(), slot, 0);
        let lease = self.surface.acquire(image, size)?;
        Ok(GlSurfaceAcquire::Lease(lease))
    }

    fn resize_surface(&mut self, size: GlSurfaceSize) -> Result<(), GlError> {
        const OP: &str = "resize-surface";
        self.assert_ready(OP)?;
        // Resize invalidates outstanding leases first. The Host owns the actual
        // window size and performs the resize; recording the reported extent is
        // the whole of this executor's part, and it must not be reordered after
        // the lease invalidation, or a lease from the old extent would survive
        // into the new one.
        self.surface.invalidate_generation()?;
        self.surface_extent = Some(size);
        self.surface_suspended = size.is_zero();
        Ok(())
    }

    fn suspend_surface(&mut self) -> Result<(), GlError> {
        const OP: &str = "suspend-surface";
        self.assert_ready(OP)?;
        self.surface.invalidate_generation()?;
        self.surface_suspended = true;
        Ok(())
    }

    fn resume_surface(&mut self) -> Result<(), GlError> {
        const OP: &str = "resume-surface";
        self.assert_ready(OP)?;
        self.surface.invalidate_generation()?;
        self.surface_suspended = false;
        Ok(())
    }

    fn present_surface(&mut self, lease: GlSurfaceLease) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "present-surface";
        self.assert_ready(OP)?;
        self.validate_object_context(OP, lease.image.context)?;
        self.surface.consume(lease)?;
        // SAFETY: current-context contract; the lease was consumed above, and
        // flushing is valid on any live context.
        unsafe { self.gl.flush() };
        // The platform swaps the drawable after this returns (the surface
        // provider owns that call); this flush only guarantees the frame's
        // accepted commands reached the driver before the lease is consumed. It
        // never implies the frame was displayed, and it never implies
        // completion.
        self.driver_error(OP)
    }
}

/// What one acquisition attempt may do with the recorded drawable state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SurfaceAcquireAttempt {
    /// No lease can be handed out, so the caller is told the surface is
    /// unavailable instead of being given a lease with an invented extent.
    Suspended,
    /// The Host-reported extent a lease may name.
    Lease(GlSurfaceSize),
}

/// Decides whether an acquisition can name a drawable from recorded facts.
///
/// Native GL exposes no core query for the default framebuffer's extent, so on
/// this family the size is a Host fact that reaches the executor only through
/// `resize_surface`. Three states that look alike are therefore kept apart: a
/// suspended drawable, an extent the Host never reported, and a reported
/// zero-area extent. None of them may be answered with an invented size, because
/// a lease whose extent is wrong is a wrong framebuffer size that no driver
/// reports and no caller can see.
pub(super) fn surface_acquire_attempt(
    suspended: bool,
    extent: Option<GlSurfaceSize>,
) -> SurfaceAcquireAttempt {
    if suspended {
        return SurfaceAcquireAttempt::Suspended;
    }
    match extent {
        // A zero-area drawable is a Host fact rather than an error, and the
        // suspension answer is what keeps a caller from rendering into nothing.
        Some(size) if !size.is_zero() => SurfaceAcquireAttempt::Lease(size),
        _ => SurfaceAcquireAttempt::Suspended,
    }
}
