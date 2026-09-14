//! Browser surface presentation execution.
//!
//! The canvas drawing buffer is the surface: acquisition reports the real
//! `drawingBufferWidth/Height` facts, resize drives the canvas backing store
//! (the only WebGL2 resize mechanism) and invalidates outstanding leases.
//! DOM events, RAF scheduling, and page visibility stay Host-owned; this
//! module never registers listeners or schedules frames.

use super::super::{
    GlError, GlFamilyApi as _, GlSurfaceAcquire, GlSurfaceLease, GlSurfacePresentationApi,
    GlSurfaceSize,
};
use super::discovery::WebGl2BrowserDiscovery;

impl GlSurfacePresentationApi for WebGl2BrowserDiscovery {
    fn acquire_surface_image(&mut self) -> Result<GlSurfaceAcquire, GlError> {
        const OP: &str = "acquire-surface-image";
        self.assert_provider_ready(OP)?;
        if self.surface_suspended {
            return Ok(GlSurfaceAcquire::Suspended);
        }
        // The drawing buffer, not the canvas attributes, is the presentation
        // fact: browsers may deny or round requested sizes.
        let width = self.raw.drawing_buffer_width();
        let height = self.raw.drawing_buffer_height();
        if width <= 0 || height <= 0 {
            // A zero-area drawing buffer cannot present; report suspension
            // instead of handing out an empty lease.
            return Ok(GlSurfaceAcquire::Suspended);
        }
        let size = GlSurfaceSize {
            width: width as u32,
            height: height as u32,
        };
        let slot = Self::allocate_slot(&mut self.next_surface_slot, OP)?;
        let image = super::super::SurfaceImageId::new(self.context_stamp(), slot, 0);
        let lease = self.surface.acquire(image, size)?;
        Ok(GlSurfaceAcquire::Lease(lease))
    }

    fn resize_surface(&mut self, size: GlSurfaceSize) -> Result<(), GlError> {
        const OP: &str = "resize-surface";
        self.assert_provider_ready(OP)?;
        // Resize always invalidates outstanding acquire leases first.
        self.surface.invalidate_generation()?;
        // The canvas backing store is RHI-owned rendering state; Host event
        // and visibility wiring is untouched.
        self.canvas.set_width(size.width);
        self.canvas.set_height(size.height);
        let actual = (
            self.raw.drawing_buffer_width(),
            self.raw.drawing_buffer_height(),
        );
        if actual != (size.width as i32, size.height as i32) {
            return Err(GlError::Driver {
                operation: OP,
                message: format!(
                    "canvas resize did not take effect: requested {size:?}, drawing buffer {actual:?}"
                ),
            });
        }
        self.driver_error(OP)
    }

    fn suspend_surface(&mut self) -> Result<(), GlError> {
        const OP: &str = "suspend-surface";
        self.assert_provider_ready(OP)?;
        self.surface.invalidate_generation()?;
        self.surface_suspended = true;
        Ok(())
    }

    fn resume_surface(&mut self) -> Result<(), GlError> {
        const OP: &str = "resume-surface";
        self.assert_provider_ready(OP)?;
        self.surface.invalidate_generation()?;
        self.surface_suspended = false;
        Ok(())
    }

    fn present_surface(&mut self, lease: GlSurfaceLease) -> Result<(), GlError> {
        const OP: &str = "present-surface";
        self.assert_provider_ready(OP)?;
        self.validate_object_context(OP, lease.image.context)?;
        self.surface.consume(lease)?;
        // The browser compositor presents the canvas after the frame; this
        // flush only guarantees accepted work reached the submission queue
        // before the lease is consumed. It never implies completion.
        self.raw.flush();
        self.driver_error(OP)
    }
}
