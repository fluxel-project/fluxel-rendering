//! Specification section 55: the captured mutation and command values.
//!
//! One responsibility: **the values a recorded command carries that are not
//! object definitions.** Section 54 describes things that exist; this file
//! describes things that *happened* — an attachment set, a copy, a resolve, a
//! blit, an upload, a readback request. Together with section 56's command IR
//! they are what a Replay feeds back into the normal RHI API.
//!
//! Not owned here: the commands that hold these values ([`super::work`]), the
//! object definitions they refer to ([`super::definition`]), and the live
//! recorder that produced them (`api::command`, module 04). Every type below is
//! built from that module's vocabulary rather than from a second copy of it: an
//! attachment *is* a [`LoadOp`] and a [`StoreOp`], a copy is a pair of
//! [`ObjectId`]s and a byte count, and none of those concepts needs a tooling
//! spelling.
//!
//! # The rule that decides every difference from the live types
//!
//! A captured value **preserves everything that determines the GPU result, and
//! may drop what is diagnostic only.** Two places in this file apply that rule
//! and they pull in opposite directions, so it is worth stating before either:
//!
//! * [`CapturedResolve`] **gains** `src_origin` and `dst_origin` relative to
//!   section 55.2's field list. The live
//!   [`crate::api::command::TextureResolve`] carries them, root specification
//!   section 4 corrected the chapter to require explicit origins, and a resolve
//!   of a sub-region that the record could not name is a resolve that replays to
//!   a different image. That is not a diagnostic loss, it is a wrong result.
//! * [`CapturedReadbackRequest`] **keeps** section 55.3's list exactly, which
//!   means it drops the `Label` the live
//!   [`crate::api::resource::transfer::ReadbackRequest`] carries on both variants.
//!   A label is excluded from every canonical hash and decides nothing; a record
//!   that omits it is a record whose debug name for its own checkpoint is gone,
//!   which is a real but acceptable loss under the rule above.
//!
//! # Two more deliberate shape differences
//!
//! * [`CapturedColorAttachment::load`] is a `LoadOp<ColorClearValue>` and not
//!   section 55.1's `LoadOp<Color>`. `Color` is this crate's *blend constant*
//!   type (four floats) and `ColorClearValue` is the attachment clear value,
//!   which also covers the signed- and unsigned-integer attachment formats. The
//!   live [`crate::api::command::ColorAttachment`] uses `ColorClearValue`, so a
//!   captured value typed `Color` could neither be produced from the live one nor
//!   replay an integer clear.
//! * [`CapturedRasterScope`] keeps the live scope's shape — a
//!   `Vec<Option<...>>` indexed by attachment location with `None` for a hole —
//!   rather than a dense list, because section 26's canonicalization removes
//!   trailing holes only. A dense list would silently renumber a deferred
//!   renderer's locations 0 and 3 into 0 and 1.

use std::sync::Arc;

use crate::api::command::{
    BlitFilter, ColorClearValue, DepthAttachmentMode, LoadOp, StencilAttachmentMode, StoreOp,
};
use crate::api::identity::{Label, ObjectId};
use crate::api::presentation::AcquiredFrameId;
use crate::api::resource::buffer::BufferRange;
use crate::api::resource::subresource::{HostTexelLayout, Origin3d, TextureSubresourceLayers};
use crate::api::resource::texture::Extent3d;

/// What a captured colour attachment is attached to.
///
/// Two members, matching the live
/// [`crate::api::command::ColorAttachmentView`]: a texture view, or an acquired
/// frame. A frame is named by [`AcquiredFrameId`] and not by the
/// [`crate::api::presentation::FrameAttachment`] handle, for the file's
/// standing reason — the frame identity is a value, and the handle is not.
#[derive(Clone)]
pub enum CapturedColorAttachmentView {
    /// A texture view, named by identity.
    TextureView(ObjectId),

    /// An acquired frame, named by identity.
    Frame(AcquiredFrameId),
}

/// One captured colour attachment.
#[derive(Clone)]
pub struct CapturedColorAttachment {
    /// What is attached.
    pub view: CapturedColorAttachmentView,

    /// What happens to the attachment when the scope begins.
    pub load: LoadOp<ColorClearValue>,

    /// What happens to it when the scope ends.
    pub store: StoreOp,

    /// The view this attachment resolves into, when it is multisampled.
    pub resolve: Option<CapturedColorAttachmentView>,
}

/// One captured depth/stencil attachment.
///
/// `view` is a plain [`ObjectId`] rather than a
/// [`CapturedColorAttachmentView`] because the live
/// [`crate::api::command::DepthStencilAttachment`] holds a `TextureView` and has
/// no frame variant: a swapchain image is a colour target in this API, and giving
/// the captured type a frame arm the live type cannot produce would describe a
/// state that never exists.
#[derive(Clone)]
pub struct CapturedDepthStencilAttachment {
    /// The view that is attached.
    pub view: ObjectId,

    /// How the depth plane is treated, or `None` if there is no depth plane.
    pub depth: Option<DepthAttachmentMode>,

    /// How the stencil plane is treated, or `None` if there is no stencil plane.
    pub stencil: Option<StencilAttachmentMode>,
}

/// One captured raster scope: the attachment set a scope opened with.
///
/// The scope's fixed state is *not* here, and that is not an omission. A draw
/// records the state that was current when it was issued (section 32), so a
/// scope-level copy would be a second, weaker answer to the same question — and
/// [`super::work::PortableCommand::Draw`] carries the authoritative one.
#[derive(Clone)]
pub struct CapturedRasterScope {
    /// The scope's diagnostic label.
    pub label: Label,

    /// One attachment per colour location, with `None` for a hole.
    pub colors: Vec<Option<CapturedColorAttachment>>,

    /// The depth and/or stencil attachment, if any.
    pub depth_stencil: Option<CapturedDepthStencilAttachment>,
}

/// A captured buffer-to-buffer copy.
#[derive(Clone)]
pub struct CapturedBufferCopy {
    /// The source buffer.
    pub src: ObjectId,

    /// The byte offset into it.
    pub src_offset: u64,

    /// The destination buffer.
    pub dst: ObjectId,

    /// The byte offset into it.
    pub dst_offset: u64,

    /// How many bytes move.
    pub size: u64,
}

/// A captured copy between a buffer and a texture, in either direction.
///
/// One type for both directions, as in the live
/// [`crate::api::command::BufferTextureCopy`] and as in section 55.2: the
/// direction is which command carries it
/// ([`super::work::PortableCommand::CopyBufferToTexture`] or
/// `CopyTextureToBuffer`), and a second type would be a second place for the
/// layout fields to be spelled slightly differently.
#[derive(Clone)]
pub struct CapturedBufferTextureCopy {
    /// The buffer side.
    pub buffer: ObjectId,

    /// The byte offset into the buffer side.
    pub buffer_offset: u64,

    /// How the buffer's bytes are laid out as rows.
    pub bytes_per_row: u32,

    /// How many rows make one image.
    pub rows_per_image: u32,

    /// The texture side.
    pub texture: ObjectId,

    /// Which subresource of it.
    pub texture_subresource: TextureSubresourceLayers,

    /// Where in that subresource the region starts.
    pub texture_origin: Origin3d,

    /// How large the region is.
    pub extent: Extent3d,
}

/// A captured texture-to-texture copy.
#[derive(Clone)]
pub struct CapturedTextureCopy {
    /// The source texture.
    pub src: ObjectId,

    /// Which subresource of it.
    pub src_subresource: TextureSubresourceLayers,

    /// Where in that subresource the region starts.
    pub src_origin: Origin3d,

    /// The destination texture.
    pub dst: ObjectId,

    /// Which subresource of it.
    pub dst_subresource: TextureSubresourceLayers,

    /// Where in that subresource the region starts.
    pub dst_origin: Origin3d,

    /// How large the region is.
    pub extent: Extent3d,
}

/// A captured multisampled-to-single-sampled resolve.
///
/// Carries explicit source and destination origins. Section 55.2's field list
/// omits them and the live [`crate::api::command::TextureResolve`] has them; the
/// live side wins, for the reason this file's note gives — a resolve that cannot
/// name the sub-region it covered replays to a different image, and the omission
/// is a wrong GPU result rather than a lost diagnostic name.
#[derive(Clone)]
pub struct CapturedResolve {
    /// The source texture.
    pub src: ObjectId,

    /// Which subresource of it.
    pub src_subresource: TextureSubresourceLayers,

    /// Where in that subresource the region starts.
    pub src_origin: Origin3d,

    /// The destination texture.
    pub dst: ObjectId,

    /// Which subresource of it.
    pub dst_subresource: TextureSubresourceLayers,

    /// Where in that subresource the region starts.
    pub dst_origin: Origin3d,

    /// How large the region is.
    pub extent: Extent3d,
}

/// A captured filtered blit.
///
/// Separate source and destination extents, because scaling is what makes this a
/// blit rather than a copy — the same distinction the live
/// [`crate::api::command::TextureBlit`] draws.
#[derive(Clone)]
pub struct CapturedBlit {
    /// The source texture.
    pub src: ObjectId,

    /// Which subresource of it.
    pub src_subresource: TextureSubresourceLayers,

    /// Where the source region starts.
    pub src_origin: Origin3d,

    /// How large the source region is.
    pub src_extent: Extent3d,

    /// The destination texture.
    pub dst: ObjectId,

    /// Which subresource of it.
    pub dst_subresource: TextureSubresourceLayers,

    /// Where the destination region starts.
    pub dst_origin: Origin3d,

    /// How large the destination region is.
    pub dst_extent: Extent3d,

    /// How the source is sampled into the destination.
    pub filter: BlitFilter,
}

/// A captured CPU upload: an observable mutation of resource contents.
///
/// The bytes are here. Section 52.4 makes that the point of the whole type —
/// `UploadJob` retains its source, so capture never has to read back the data it
/// just wrote — and section 55.3 adds the warning that goes with it: upload bytes
/// are content, and whether they are redacted, omitted, or encrypted is the
/// Artifact layer's policy, not RHI's. RHI's job is to make them *available* and
/// to keep them free of native identifiers.
///
/// The `id` is the upload job's own [`ObjectId`] and `dst` is the destination's,
/// so the record says both which mutation this was and what it mutated. Both are
/// needed: two uploads into the same texture are two mutations, and a record that
/// kept only the destination could not tell them apart.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedUploadDefinition {
    /// An upload into a buffer.
    Buffer {
        /// The upload job's identity.
        id: ObjectId,
        /// The destination buffer.
        dst: ObjectId,
        /// The byte offset into it.
        dst_offset: u64,
        /// The source bytes.
        bytes: Arc<[u8]>,
    },

    /// An upload into a texture.
    Texture {
        /// The upload job's identity.
        id: ObjectId,
        /// The destination texture.
        dst: ObjectId,
        /// Which subresource of it.
        subresource: TextureSubresourceLayers,
        /// Where in that subresource the region starts.
        origin: Origin3d,
        /// How large the region is.
        extent: Extent3d,
        /// How the source bytes are laid out as rows.
        source_layout: HostTexelLayout,
        /// The source bytes.
        bytes: Arc<[u8]>,
    },
}

/// A captured readback request.
///
/// The other half of section 52.5: RHI offers the primitive, and *this* is the
/// record of one having been made, so that a capture can say where it took a
/// snapshot without RHI ever having decided that a snapshot was wanted. The
/// request is recorded, not its result — a readback's bytes arrive later, through
/// the ticket, and a record that embedded them would be claiming a completion
/// that may not have happened when the record was written.
///
/// `ticket` names the [`crate::api::resource::transfer::ReadbackTicket`] the
/// request was encoded as, which is the identity the receipt path and the
/// terminal `ReadbackStatus` both answer about.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedReadbackRequest {
    /// A readback out of a buffer.
    Buffer {
        /// The ticket this request was encoded as.
        ticket: ObjectId,
        /// The source buffer.
        src: ObjectId,
        /// The byte range read.
        range: BufferRange,
    },

    /// A readback out of a texture.
    Texture {
        /// The ticket this request was encoded as.
        ticket: ObjectId,
        /// The source texture.
        src: ObjectId,
        /// Which subresource of it.
        subresource: TextureSubresourceLayers,
        /// Where in that subresource the region starts.
        origin: Origin3d,
        /// How large the region is.
        extent: Extent3d,
    },
}
