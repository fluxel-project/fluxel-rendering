//! Host-sized surface lifecycle with generation-bound acquire leases.

use super::{GlError, GlFamilyApi, GlTextureDesc, GlTextureDimension, SurfaceImageId, TextureId};
use std::collections::BTreeSet;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlSurfaceSize {
    pub width: u32,
    pub height: u32,
}
impl GlSurfaceSize {
    pub const fn is_zero(self) -> bool {
        self.width == 0 || self.height == 0
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlSurfaceGeneration(NonZeroU64);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlSurfaceLease {
    pub image: SurfaceImageId,
    pub generation: GlSurfaceGeneration,
    serial: NonZeroU64,
    pub size: GlSurfaceSize,
}
/// Tracks the leases owned by the current surface generation.
#[derive(Debug)]
pub(crate) struct GlSurfaceLeaseBook {
    generation: GlSurfaceGeneration,
    next_serial: u64,
    live: BTreeSet<u64>,
}
impl GlSurfaceLeaseBook {
    pub(crate) fn new() -> Self {
        Self {
            generation: GlSurfaceGeneration(NonZeroU64::MIN),
            next_serial: 0,
            live: BTreeSet::new(),
        }
    }
    pub(crate) fn acquire(
        &mut self,
        image: SurfaceImageId,
        size: GlSurfaceSize,
    ) -> Result<GlSurfaceLease, GlError> {
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or_else(|| GlError::Validation {
                operation: "acquire_surface_image",
                message: "surface lease serial exhausted".into(),
            })?;
        let serial = NonZeroU64::new(self.next_serial).unwrap();
        self.live.insert(serial.get());
        Ok(GlSurfaceLease {
            image,
            generation: self.generation,
            serial,
            size,
        })
    }
    /// Checks a lease against the current generation and the live set.
    ///
    /// `operation` is the caller's, not this book's. The two ways a lease is
    /// refused -- superseded by a generation change, or consumed already --
    /// are answered from one place and share a message because a caller cannot
    /// act differently on them, but the operation that failed is not shared:
    /// the book refuses a present and a publish from the same lines, so it
    /// names whichever of them the caller was performing rather than a fixed
    /// verb.
    pub(crate) fn validate(
        &self,
        operation: &'static str,
        lease: GlSurfaceLease,
    ) -> Result<(), GlError> {
        (lease.generation == self.generation && self.live.contains(&lease.serial.get()))
            .then_some(())
            .ok_or_else(|| GlError::Validation {
                operation,
                message: "surface acquire lease is stale or already consumed".into(),
            })
    }
    pub(crate) fn consume(
        &mut self,
        operation: &'static str,
        lease: GlSurfaceLease,
    ) -> Result<(), GlError> {
        self.validate(operation, lease)?;
        self.live.remove(&lease.serial.get());
        Ok(())
    }
    /// Resize, suspension, and resume all require this before another acquire.
    pub(crate) fn invalidate_generation(&mut self) -> Result<(), GlError> {
        let next = self
            .generation
            .0
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .ok_or_else(|| GlError::Validation {
                operation: "surface_lifecycle",
                message: "surface generation exhausted".into(),
            })?;
        self.generation = GlSurfaceGeneration(next);
        self.live.clear();
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlSurfaceAcquire {
    Lease(GlSurfaceLease),
    Suspended,
}

/// RHI owns the surface executor; Host owns the native window. Resize,
/// suspend, and resume invalidate all old leases before they return.
///
/// # Two families, and what the drawable is to each
///
/// The window system's drawable is reached in one of two ways, and the trait
/// says which way each implementation takes rather than papering over the
/// difference. A provider that owns the drawable *and* the object tables its
/// images are created in can publish: it takes the frame the acquired image
/// holds and puts it on the drawable. A provider that owns only the drawable
/// -- the WGL and EGL surface types, which borrow no object tables and mint no
/// identities -- cannot, because it has no texture to publish and no table to
/// find one in; those implementations refuse and keep `present_surface` as the
/// flip it already is.
///
/// The two are mutually exclusive per acquisition: both consume the lease, and
/// a lease is consumed once, so a frame is either already in its acquired image
/// or published into it from another texture, never both.
pub(crate) trait GlSurfacePresentationApi: GlFamilyApi {
    fn acquire_surface_image(&mut self) -> Result<GlSurfaceAcquire, GlError>;
    fn resize_surface(&mut self, size: GlSurfaceSize) -> Result<(), GlError>;
    fn suspend_surface(&mut self) -> Result<(), GlError>;
    fn resume_surface(&mut self) -> Result<(), GlError>;
    fn present_surface(&mut self, lease: GlSurfaceLease) -> Result<(), GlError>;
    /// Puts the frame that `source` holds on the drawable `lease` acquired.
    ///
    /// `source` is the texture the caller allocated for `lease`'s image, which
    /// is why it is an argument rather than something this layer looks up: the
    /// surface image is an acquisition identity, and the storage behind it
    /// belongs to whoever created it. The lease is consumed on success, so the
    /// acquisition ends here exactly as it does through `present_surface`, and
    /// the extent check
    /// [`validate_publish_source`] performs is what keeps a source that does
    /// not belong to this acquisition from being scaled onto the drawable by
    /// the driver's own choice of what to keep.
    fn publish_surface_image(
        &mut self,
        lease: GlSurfaceLease,
        source: TextureId,
    ) -> Result<(), GlError>;
}

/// Checks that a texture can serve as one acquired image's presentation source.
///
/// The rule is shared by every provider that publishes because it is a property
/// of the contract rather than of a driver: the blit that follows copies the
/// whole source into a drawable that has exactly the acquired extent and no
/// sample count of its own to reconcile with. A multisampled, layered, or
/// differently-sized source is therefore refused before any driver call instead
/// of being resolved by whatever the driver decides to keep.
pub(crate) fn validate_publish_source(
    operation: &'static str,
    descriptor: GlTextureDesc,
    lease: GlSurfaceLease,
) -> Result<(), GlError> {
    if descriptor.dimension != GlTextureDimension::D2
        || descriptor.sample_count != 1
        || descriptor.extent.depth_or_layers != 1
    {
        return Err(GlError::Validation {
            operation,
            message: "presentation source must be a single-sample 2D texture".into(),
        });
    }
    if descriptor.extent.width != lease.size.width || descriptor.extent.height != lease.size.height
    {
        return Err(GlError::Validation {
            operation,
            message: "presentation source extent does not match the acquired image".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        GlSurfaceLeaseBook, GlSurfaceSize, GlTextureDesc, GlTextureDimension,
        validate_publish_source,
    };
    use crate::webgl2::api::{
        ContextEpoch, ContextStamp, DeviceIdentity, GlError, GlExtent3d, GlFormat, GlTextureUsage,
        SurfaceImageId,
    };

    fn stamp() -> ContextStamp {
        ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
    }

    /// One acquisition at `width` x `height`, which is the extent a publish
    /// source has to match.
    fn lease(width: u32, height: u32) -> super::GlSurfaceLease {
        let mut book = GlSurfaceLeaseBook::new();
        book.acquire(
            SurfaceImageId::new(stamp(), 1, 1),
            GlSurfaceSize { width, height },
        )
        .expect("lease")
    }

    fn source(
        dimension: GlTextureDimension,
        extent: GlExtent3d,
        sample_count: u32,
    ) -> GlTextureDesc {
        GlTextureDesc {
            dimension,
            extent,
            mip_level_count: 1,
            sample_count,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::RENDER_ATTACHMENT,
        }
    }

    const OP: &str = "publish-surface-image";

    #[test]
    fn a_refused_lease_names_the_operation_that_was_attempted() {
        let mut book = GlSurfaceLeaseBook::new();
        let lease = book
            .acquire(
                SurfaceImageId::new(stamp(), 1, 1),
                GlSurfaceSize {
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        book.consume("present-surface", lease)
            .expect("the first consume owns the lease");
        // A present and a publish are refused from the same lines, so the
        // second attempt is what the diagnostic has to name: reporting the
        // book's own fixed verb would send a caller looking at the wrong call.
        assert_eq!(
            book.consume("publish-surface-image", lease),
            Err(GlError::Validation {
                operation: "publish-surface-image",
                message: "surface acquire lease is stale or already consumed".into(),
            })
        );
    }

    #[test]
    fn resize_invalidates_an_old_acquire_lease() {
        let s = stamp();
        let mut b = GlSurfaceLeaseBook::new();
        let lease = b
            .acquire(
                SurfaceImageId::new(s, 1, 1),
                GlSurfaceSize {
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        b.invalidate_generation().unwrap();
        assert!(b.validate("present-surface", lease).is_err());
    }

    #[test]
    fn a_presentation_source_must_be_a_single_sample_2d_texture_of_the_acquired_extent() {
        let plain = GlExtent3d {
            width: 1,
            height: 1,
            depth_or_layers: 1,
        };
        assert!(
            validate_publish_source(OP, source(GlTextureDimension::D2, plain, 1), lease(1, 1))
                .is_ok()
        );

        // Three refusals, each of which a driver would otherwise resolve by its
        // own choice of what to keep: a source that is the wrong size, one with
        // samples to reconcile, and one with layers to choose between.
        let wrong_size = GlExtent3d {
            width: 2,
            height: 2,
            depth_or_layers: 1,
        };
        let layered = GlExtent3d {
            width: 1,
            height: 1,
            depth_or_layers: 2,
        };
        for (descriptor, against) in [
            (source(GlTextureDimension::D2, wrong_size, 1), lease(1, 1)),
            (source(GlTextureDimension::D2, plain, 4), lease(1, 1)),
            (source(GlTextureDimension::D3, layered, 1), lease(1, 1)),
        ] {
            let refused = validate_publish_source(OP, descriptor, against)
                .expect_err("refused before any driver call");
            assert!(
                matches!(refused, GlError::Validation { operation, .. } if operation == OP),
                "expected a validation refusal naming the verb, got {refused:?}"
            );
        }
    }
}
