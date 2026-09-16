//! The acquired presentable image, and the token that ends its acquisition.
//!
//! Responsibility: one verb that turns an acquisition of the drawable into the
//! two halves a frame needs -- a texture the graph renders into, held in the
//! provider's own object table like every other texture, and an opaque token
//! that carries the acquisition forward until something presents it.
//!
//! Not owned here: *when* a frame presents (that is [`super::submission`]) or
//! *how* a drawable is reached (that is Layer 1's `GlSurfacePresentationApi`,
//! whose two families differ in whether they own object tables at all -- see
//! `webgl2/api/presentation.rs`).  This module is the one place those two meet,
//! and it meets them by holding the acquisition rather than by acting on it.
//!
//! # Why the descriptor is the caller's
//!
//! The graph declares the surface resource's `TextureDesc` and the executor
//! rejects a binding whose descriptor differs from its compiled contract
//! (`execution/run/resolution.rs:477`), so the extent and the format a
//! presentable image must have are compiled facts before any acquisition
//! happens.  This verb therefore takes them instead of asking the context: a
//! texture created at an extent the adapter guessed would be one the frame's own
//! binding check rejects a moment later, and a format read back from a discovery
//! marker would be a second opinion about a fact the graph has already fixed.
//!
//! What the descriptor cannot carry is checked here, and it is exactly one
//! thing: its extent against the extent the acquisition reported.  That check is
//! the same rule `validate_publish_source` applies at the other end of the
//! frame, for the same reason -- a disagreement about the size of the drawable
//! is one a driver would otherwise resolve by its own choice of what to keep.
//!
//! # Why the drawable can be unavailable without being an error
//!
//! [`acquire_surface_texture`] answers `Ok(None)` where Layer 1 answers
//! `GlSurfaceAcquire::Suspended`.  A suspended, never-reported, or zero-area
//! drawable is a Host fact rather than a failure: there is no image to render
//! into and nothing an operator could fix, so a frame with nothing to present
//! reports that instead of an error it would have to invent a cause for.
//!
//! [`acquire_surface_texture`]: GlCompatibilityDevice::acquire_surface_texture

use std::rc::Rc;

use fluxel_rendergraph::{
    BoundSurfaceTexture, BoundTexture, ResourceAccessState, TextureDesc, TextureUsage,
};

use crate::webgl2::api::{GlError, GlSurfaceAcquire, GlSurfaceLease, GlSurfaceSize, TextureId};
use crate::webgl2::state::GlStateBackend;

use super::GlCompatibilityDevice;
use super::compute::ComputeDomain;
use super::pass;
use super::retention::{GlRetentionLease, RetainedObject};
use super::transient;

/// Checks that the declared surface texture describes the acquired image.
///
/// The rule lives on its own so that it can be exercised without an advertised
/// surface, on the terms this suite already uses for
/// [`transient`](super::transient)'s lowerings: the shape a verb would accept is
/// not observable through the adapter while the capability that reaches the verb
/// is withheld, and a rule that is only checkable through a path nothing can
/// take is a rule with no test.  That is also why it takes the acquired *size*
/// rather than the lease it came from: a lease can only be minted by Layer 1's
/// own lease book, and widening that to reach a fixture would trade an ownership
/// boundary for a test.
///
/// It is one comparison and not a wider descriptor check, because every other
/// field is either fixed by the contract (a presentable image is two-dimensional
/// and single-sampled, which `validate_publish_source` refuses at the other end
/// of the frame) or already validated by the creation lowering. What is left is
/// the one thing only the acquisition knows.
pub(super) fn validate_surface_extent(
    operation: &'static str,
    descriptor: TextureDesc,
    acquired: GlSurfaceSize,
) -> Result<(), GlError> {
    if descriptor.extent.width != acquired.width || descriptor.extent.height != acquired.height {
        return Err(GlError::Validation {
            operation,
            message: "declared surface texture extent does not match the acquired image".into(),
        });
    }
    Ok(())
}

/// What one acquisition of the drawable is presented through.
///
/// This is the contract's `PresentationToken`: it is produced by acquiring an
/// image and consumed by [`submit`](fluxel_rendergraph::ExecutionBackend::submit),
/// which is the only boundary allowed to present.  It is one-shot by
/// construction rather than by convention -- it holds the only copy of the
/// acquisition's lease, `GlSurfaceLease` is `Copy` but the copy a caller could
/// make is not a token, and every verb that ends an acquisition consumes the
/// lease through Layer 1's lease book, so a second attempt is refused there.
/// Dropping it without submitting therefore abandons the acquisition, and the
/// texture it holds is released the way every other retained object is: at the
/// adapter's next entry point, from the release queue.
///
/// The texture is retained here as well as by the `BoundTexture` the frame
/// binds, so the object survives a frame that drops its bindings before it
/// submits -- and dies when the last of the two goes, which is the handle-count
/// lifetime [`super::retention`] exists to provide.
pub(crate) struct GlSurfaceToken {
    /// The acquisition this token ends.
    ///
    /// Read by the verb that presents, which does not exist yet: this landing is
    /// the path an acquisition takes, and the capability that would let a graph
    /// compile a present root is deliberately still absent.  See the module
    /// documentation of [`super`] for why the two land in that order.
    #[allow(
        dead_code,
        reason = "the consuming verb is the next step; the path lands before the capability that reaches it"
    )]
    lease: GlSurfaceLease,
    /// The texture created for that acquisition's image.
    #[allow(
        dead_code,
        reason = "the consuming verb is the next step; the path lands before the capability that reaches it"
    )]
    texture: TextureId,
    /// This token's share of the texture's lifetime.
    ///
    /// Never read, and that is the whole of what it does: releasing the object
    /// is the work its `Drop` performs. The leading underscore is how a field
    /// that exists for its destructor is spelled, `dead_code` included.
    _retention: GlRetentionLease,
}

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// Acquires the drawable's next image as a texture this device owns.
    ///
    /// The texture is created through the provider's own creation verb, so it
    /// lives in the provider's object table and is described by this adapter's
    /// attachment record like every other texture a frame renders into -- there
    /// is no second kind of texture here, which is what makes the acquired image
    /// usable as a pass's colour attachment without any special case.
    ///
    /// `Ok(None)` is a drawable that cannot be acquired; see the module
    /// documentation.  Every refusal that can be made before the driver is
    /// reached is made before it, so a request this adapter cannot honour never
    /// costs an acquisition.
    pub(crate) fn acquire_surface_texture(
        &mut self,
        descriptor: TextureDesc,
        usage: TextureUsage,
    ) -> Result<Option<BoundSurfaceTexture<TextureId, GlRetentionLease, GlSurfaceToken>>, GlError>
    {
        const OP: &str = "acquire-surface-texture";
        // Fail-closed on the capability rather than on a state flag: the
        // advertisement is what a graph compiler consults, so a caller that
        // reached this verb without one could not have compiled a present root
        // in the first place.  Keeping the check here is what lets this landing
        // add the path without also widening what a graph may ask for, and the
        // next step removes it in the same change that reports a surface.
        if self.capabilities.surface.is_none() {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this adapter advertises no surface, so an acquired image cannot be \
                         described to a graph",
            });
        }
        self.refresh();
        self.release_pending()?;
        // Lowered before the acquisition, and in this order for one reason: this
        // is the only step that can refuse the request on its own terms, and a
        // refusal after the acquisition would leave a live lease behind for a
        // request that never reached the driver.
        let lowered = transient::texture_descriptor(descriptor, usage)?;
        let lease = match self.machine.backend().acquire_surface_image()? {
            GlSurfaceAcquire::Lease(lease) => lease,
            GlSurfaceAcquire::Suspended => return Ok(None),
        };
        validate_surface_extent(OP, descriptor, lease.size)?;
        let attachment = pass::Attachment::of(&lowered);
        // A creation that fails here leaves the lease live in Layer 1's lease
        // book, which is a bookkeeping residue rather than a resource one: a
        // later acquisition takes a fresh serial, and the generation change a
        // resize or a context loss performs clears the set.  Consuming it
        // instead is not available -- the only verb that consumes a lease
        // presents the frame -- and inventing a third one here would be this
        // layer deciding what an abandoned acquisition means.
        let physical = self.machine.backend().create_texture_resource(lowered)?;
        self.attachments.insert(physical, attachment);
        let retention = GlRetentionLease::new(
            [RetainedObject::Texture(physical)],
            Rc::clone(&self.releases),
        );
        Ok(Some(BoundSurfaceTexture {
            texture: BoundTexture {
                device: self.identity.identity(),
                identity: transient::resource_identity(physical.slot, physical.generation),
                physical,
                descriptor,
                // Derived from the creation facts rather than echoed from the
                // request, on `create_transient_texture`'s terms: one GL usage
                // bit covers more than one common operation and the physical
                // object really does permit all of them.
                usage: transient::texture_usage(lowered.usage, lowered.format),
                // An acquired drawable's contents are not defined by anything
                // this frame has done, which is what `Undefined` says.
                initial_state: ResourceAccessState::Undefined,
                lease: retention.clone(),
            },
            presentation: GlSurfaceToken {
                lease,
                texture: physical,
                _retention: retention,
            },
        }))
    }
}
