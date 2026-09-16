//! Writing host bytes into a buffer this device created.
//!
//! Responsibility: turn a caller's `BufferId`, an offset and a byte slice into
//! the one Layer 1 verb that writes them, and refuse the case Layer 1 cannot
//! recognise.
//!
//! Not owned here: how a buffer comes to exist ([`super::backend`]'s two
//! transient verbs), and when it is destroyed ([`super::retention`]).
//!
//! # Why this is not a contract verb
//!
//! [`ExecutionBackend`](fluxel_rendergraph::ExecutionBackend) has no verb that
//! writes host bytes, and the omission is not an oversight to close here.  Every
//! resource verb in that contract is about *existence* -- create, transition,
//! bind, copy device-to-device -- because a graph-created resource's contents are
//! produced by the passes that write it, and the executor has no bytes of its
//! own to put anywhere.  Upload is what a *caller* does to a resource it owns,
//! which is exactly the population an import is drawn from: the contract's
//! import half says a caller-owned object arrives already backed, and says
//! nothing about how it got that way.
//!
//! So this is an inherent verb on the adapter, and the composability is the
//! point: a caller calls `create_transient_buffer` for the object, calls this to
//! fill it, and hands the `BufferId` back through a
//! [`FrameResourceProvider`](fluxel_rendergraph::FrameResourceProvider).  None of
//! the three steps needs a frame, an encoder or an executor, which is what makes
//! the imported half of the frame contract reachable at all.
//!
//! # The size is the slice's, not a parameter
//!
//! Layer 1 uploads exactly `range.size` bytes and refuses a slice that disagrees
//! (`api/native/exec_copy.rs` checks the two against each other).  This verb
//! therefore *derives* the range's size from `bytes.len()` instead of accepting
//! one, and a length mismatch is not a case it detects but a state it cannot
//! construct.  A caller with a size in hand and a slice that disagrees with it
//! has a bug this signature leaves no room to express.
//!
//! # What the check here is, and what is left to Layer 1
//!
//! The adapter checks one thing: that the buffer is one *this device* created,
//! which is also what gives it the allocation size.  That record is adapter
//! knowledge -- Layer 1 exposes no verb that reads a descriptor back, and a
//! `BufferId` names an object rather than describing one -- and it is the same
//! record [`Self::storage_range`] reads for the identical reason, kept and
//! dropped beside the creation that wrote it so an identity reused after a
//! deletion cannot resolve to its predecessor's size.
//!
//! The range's own legality is **not** re-checked here.  Whether the range is
//! non-empty and inside the allocation is a rule about a `GlBufferDesc`, and the
//! layer that holds the descriptor states it once (`GlBufferRange::validate_for`);
//! a second spelling of it here would be a second thing to keep true and would
//! drift.  That refusal reaches the caller under the same operation string this
//! verb uses, so nothing about it misnames the call, and by then no driver
//! command has been issued.
//!
//! Nothing is batched and nothing is staged: one call is one upload.  §1 of the
//! plan forbids building the framework ahead of the consumer, and the consumer
//! this exists for is one imported vertex stream.

use crate::webgl2::api::{BufferId, GlError};

use super::GlCompatibilityDevice;
use super::compute::ComputeDomain;
use super::failure::malformed;
use super::region;
use crate::webgl2::state::GlStateBackend;

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// Writes `bytes` at `offset` in one buffer this device created.
    ///
    /// The lifecycle is adopted and pending retirements are dispatched first,
    /// like every other verb on this adapter and for the reason stated there: a
    /// context generation change must be observed before an identity is resolved
    /// against a record, since [`Self::refresh`] is what drops records belonging
    /// to the superseded generation.
    ///
    /// A buffer this device did not create is refused rather than attempted.
    /// There are two ways to hold such an identity -- one minted by another
    /// device, or one of this device's that has already been destroyed -- and the
    /// sentence covers both because the caller's mistake is the same one: the
    /// upload and the allocation did not come from the same place.
    pub(super) fn upload_buffer(
        &mut self,
        buffer: BufferId,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        const OP: &str = "upload-buffer";
        self.refresh();
        self.release_pending()?;
        if !self.buffers.contains_key(&buffer) {
            return Err(malformed(
                OP,
                "an upload names a buffer this device did not create",
            ));
        }
        let range = region::buffer_range(buffer, offset, bytes.len() as u64);
        self.machine.backend().upload_buffer(range, bytes)
    }
}
