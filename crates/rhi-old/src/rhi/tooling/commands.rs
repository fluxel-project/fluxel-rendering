//! Captured command values and the portable command IR (rhi-design sections 55
//! and 56).
//!
//! # What it is
//!
//! The in-memory semantic IR of a recording. `PortableCommand` is one recorded
//! command with every live handle replaced by the `ObjectId` that names it, and
//! the value types around it describe attachments, copies, uploads, and
//! readbacks the same way. This is not a file format: nothing here defines a
//! magic number, a chunk, an encoding, or a byte order (section 58.3).
//!
//! A use summary travels next to the command it belongs to, because the command
//! says what was *asked for* and the use says what the recording *actually
//! touched*; a replay or a diff needs both, and only RHI knows the second.
//!
//! # What it deliberately does not own
//!
//! No native command buffer, encoder, descriptor index, barrier bit, or
//! pipeline-stage flag is representable here. A Rust enum discriminant or
//! memory layout is also never an opcode: the artifact layer must supply its own
//! tagged, versioned, bounds-checked, canonical encoding (section 56, tail).

use super::super::binding::BindGroupIndex;
use super::super::command::{
    BlitFilter, Color, ColorClearValue, DepthAttachmentMode, IndexFormat, LoadOp, Rect, StoreOp,
    StencilAttachmentMode, Viewport,
};
use super::super::graph_bridge::{AccessMask, PipelineScope, TextureUseIntent};
use super::super::platform::{Label, ObjectId};
use super::super::presentation::AcquiredFrameId;
use super::super::resource::{
    BufferRange, Extent3d, HostTexelLayout, Origin3d, TextureSubresourceLayers,
    TextureSubresourceRange,
};

/// Where a captured color attachment's texels come from.
#[derive(Clone)]
pub enum CapturedColorAttachmentView {
    /// An ordinary texture view, named by object id.
    TextureView(ObjectId),
    /// A presentation frame's drawable.
    Frame(AcquiredFrameId),
}

/// One captured color attachment location.
///
/// The clear value is the live `ColorClearValue`, not a float quadruple, because
/// a sint or uint attachment has a clear that a float form cannot express and a
/// capture that lost that distinction could not be replayed.
#[derive(Clone)]
pub struct CapturedColorAttachment {
    /// The attachment's texels.
    pub view: CapturedColorAttachmentView,
    /// What happens to the contents when the scope begins.
    pub load: LoadOp<ColorClearValue>,
    /// What happens to the contents when the scope ends.
    pub store: StoreOp,
    /// The single-sample resolve destination.
    pub resolve: Option<CapturedColorAttachmentView>,
}

/// A captured depth and/or stencil attachment.
#[derive(Clone)]
pub struct CapturedDepthStencilAttachment {
    /// The attachment's texels.
    pub view: ObjectId,
    /// The depth aspect's mode. `None` leaves the aspect untouched.
    pub depth: Option<DepthAttachmentMode>,
    /// The stencil aspect's mode. `None` leaves the aspect untouched.
    pub stencil: Option<StencilAttachmentMode>,
}

/// The attachment set of one captured raster scope.
#[derive(Clone)]
pub struct CapturedRasterScope {
    /// A diagnostic label.
    pub label: Label,
    /// The color attachments, indexed by location.
    pub colors: Vec<Option<CapturedColorAttachment>>,
    /// The depth/stencil attachment.
    pub depth_stencil: Option<CapturedDepthStencilAttachment>,
}

/// A captured buffer-to-buffer copy.
#[derive(Clone)]
pub struct CapturedBufferCopy {
    /// The source buffer.
    pub src: ObjectId,
    /// The source byte offset.
    pub src_offset: u64,
    /// The destination buffer.
    pub dst: ObjectId,
    /// The destination byte offset.
    pub dst_offset: u64,
    /// The number of bytes to copy.
    pub size: u64,
}

/// A captured buffer-to-texture or texture-to-buffer copy.
///
/// One shape serves both directions, as in the live value: the direction is the
/// command that carried it.
#[derive(Clone)]
pub struct CapturedBufferTextureCopy {
    /// The buffer side of the copy.
    pub buffer: ObjectId,
    /// The buffer byte offset.
    pub buffer_offset: u64,
    /// The byte distance between rows in the buffer.
    pub bytes_per_row: u32,
    /// The number of rows between depth slices in the buffer.
    pub rows_per_image: u32,

    /// The texture side of the copy.
    pub texture: ObjectId,
    /// The texture subresource on the texture side.
    pub texture_subresource: TextureSubresourceLayers,
    /// The texel origin on the texture side.
    pub texture_origin: Origin3d,
    /// The texel extent of the copy.
    pub extent: Extent3d,
}

/// A captured texture-to-texture copy.
#[derive(Clone)]
pub struct CapturedTextureCopy {
    /// The source texture.
    pub src: ObjectId,
    /// The source subresource.
    pub src_subresource: TextureSubresourceLayers,
    /// The source texel origin.
    pub src_origin: Origin3d,

    /// The destination texture.
    pub dst: ObjectId,
    /// The destination subresource.
    pub dst_subresource: TextureSubresourceLayers,
    /// The destination texel origin.
    pub dst_origin: Origin3d,

    /// The texel extent of the copy.
    pub extent: Extent3d,
}

/// A captured multisample-to-single-sample resolve.
#[derive(Clone)]
pub struct CapturedResolve {
    /// The multisampled source texture.
    pub src: ObjectId,
    /// The source subresource.
    pub src_subresource: TextureSubresourceLayers,

    /// The single-sample destination texture.
    pub dst: ObjectId,
    /// The destination subresource.
    pub dst_subresource: TextureSubresourceLayers,

    /// The texel extent of the resolve.
    pub extent: Extent3d,
}

/// A captured filtered or unfiltered texture blit.
#[derive(Clone)]
pub struct CapturedBlit {
    /// The source texture.
    pub src: ObjectId,
    /// The source subresource.
    pub src_subresource: TextureSubresourceLayers,
    /// The source texel origin.
    pub src_origin: Origin3d,
    /// The source texel extent.
    pub src_extent: Extent3d,

    /// The destination texture.
    pub dst: ObjectId,
    /// The destination subresource.
    pub dst_subresource: TextureSubresourceLayers,
    /// The destination texel origin.
    pub dst_origin: Origin3d,
    /// The destination texel extent.
    pub dst_extent: Extent3d,

    /// The sampling filter.
    pub filter: BlitFilter,
}

/// The captured definition of a CPU upload mutation.
///
/// The source bytes are carried, not re-read: an upload is an observable
/// mutation of GPU state, and a capture that recorded only "some upload
/// happened" could not rebuild the resource contents.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedUploadDefinition {
    /// An upload into a buffer range.
    Buffer {
        /// The upload job's object id.
        id: ObjectId,
        /// The destination buffer.
        dst: ObjectId,
        /// The destination byte offset.
        dst_offset: u64,
        /// The uploaded bytes.
        bytes: std::sync::Arc<[u8]>,
    },

    /// An upload into a texture region.
    Texture {
        /// The upload job's object id.
        id: ObjectId,
        /// The destination texture.
        dst: ObjectId,
        /// The destination subresource.
        subresource: TextureSubresourceLayers,
        /// The destination texel origin.
        origin: Origin3d,
        /// The destination texel extent.
        extent: Extent3d,
        /// The CPU byte layout of the supplied bytes.
        source_layout: HostTexelLayout,
        /// The uploaded bytes.
        bytes: std::sync::Arc<[u8]>,
    },
}

/// The captured definition of a readback request.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedReadbackRequest {
    /// A buffer readback.
    Buffer {
        /// The ticket that will carry the result.
        ticket: ObjectId,
        /// The source buffer.
        src: ObjectId,
        /// The read byte range.
        range: BufferRange,
    },

    /// A texture readback.
    Texture {
        /// The ticket that will carry the result.
        ticket: ObjectId,
        /// The source texture.
        src: ObjectId,
        /// The read subresource.
        subresource: TextureSubresourceLayers,
        /// The read texel origin.
        origin: Origin3d,
        /// The read texel extent.
        extent: Extent3d,
    },
}

/// One captured recorded command.
///
/// `PortableCommand` is an **in-memory semantic IR**, not a file format. Its
/// order is the recording order, which is the only order that is real: the
/// semantic-event order is CPU observation order and says nothing about GPU
/// execution order (section 53.3).
#[non_exhaustive]
#[derive(Clone)]
pub enum PortableCommand {
    /// Opens a raster scope with the given attachments.
    BeginRaster(CapturedRasterScope),
    /// Closes the open raster scope.
    EndRaster,

    /// Binds a raster pipeline.
    SetRasterPipeline(ObjectId),
    /// Binds a compute pipeline.
    SetComputePipeline(ObjectId),

    /// Binds a bind group with its dynamic offsets.
    SetBindGroup {
        /// The bind group index.
        index: BindGroupIndex,
        /// The bind group.
        group: ObjectId,
        /// The dynamic offsets, in the frozen order.
        dynamic_offsets: Vec<u32>,
    },

    /// Binds a vertex buffer.
    SetVertexBuffer {
        /// The vertex buffer slot.
        slot: u32,
        /// The buffer.
        buffer: ObjectId,
        /// The bound range.
        range: BufferRange,
    },

    /// Binds an index buffer.
    SetIndexBuffer {
        /// The buffer.
        buffer: ObjectId,
        /// The bound range.
        range: BufferRange,
        /// The index width.
        format: IndexFormat,
    },

    /// Sets the viewport.
    SetViewport(Viewport),
    /// Sets the scissor rectangle.
    SetScissor(Rect),
    /// Sets the blend constant.
    SetBlendConstant(Color),
    /// Sets the stencil reference.
    SetStencilReference(u32),

    /// A non-indexed draw.
    Draw {
        /// The vertex range.
        vertices: std::ops::Range<u32>,
        /// The instance range.
        instances: std::ops::Range<u32>,
    },

    /// An indexed draw.
    DrawIndexed {
        /// The index range.
        indices: std::ops::Range<u32>,
        /// The vertex index bias.
        base_vertex: i32,
        /// The instance range.
        instances: std::ops::Range<u32>,
    },

    /// Opens a compute scope.
    BeginCompute {
        /// A diagnostic label.
        label: Label,
    },
    /// Closes the open compute scope.
    EndCompute,

    /// Dispatches a compute grid.
    Dispatch {
        /// The X workgroup count.
        x: u32,
        /// The Y workgroup count.
        y: u32,
        /// The Z workgroup count.
        z: u32,
    },

    /// Encodes an upload.
    Upload {
        /// The upload job.
        upload: ObjectId,
    },

    /// Encodes a readback.
    Readback {
        /// The readback ticket.
        ticket: ObjectId,
    },

    /// Copies buffer to buffer.
    CopyBuffer(CapturedBufferCopy),
    /// Copies buffer to texture.
    CopyBufferToTexture(CapturedBufferTextureCopy),
    /// Copies texture to buffer.
    CopyTextureToBuffer(CapturedBufferTextureCopy),
    /// Copies texture to texture.
    CopyTexture(CapturedTextureCopy),
    /// Resolves multisample to single sample.
    Resolve(CapturedResolve),
    /// Blits a texture region.
    Blit(CapturedBlit),

    /// Opens a debug group.
    PushDebugGroup(String),
    /// Closes the innermost debug group.
    PopDebugGroup,
    /// Emits a debug marker.
    DebugMarker(String),
}

/// One captured command with the uses it actually performed.
#[derive(Clone)]
pub struct CapturedCommand {
    /// The command itself.
    pub command: PortableCommand,
    /// The resource uses this command actually performed, in command order.
    pub actual_uses: Vec<CapturedResourceUse>,
}

/// One resource a captured command actually touched.
///
/// It is the tooling-owned counterpart of the recorder's use value: the same
/// stages, access mask, and intent, with every live handle replaced by the
/// `ObjectId` that names it.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedResourceUse {
    /// A buffer use.
    Buffer {
        /// The buffer.
        buffer: ObjectId,
        /// The byte range touched.
        range: BufferRange,
        /// The stages that touch it.
        stages: PipelineScope,
        /// The accesses performed.
        access: AccessMask,
    },

    /// A texture use.
    Texture {
        /// The texture.
        texture: ObjectId,
        /// The mip/layer/aspect set touched.
        subresources: TextureSubresourceRange,
        /// The stages that touch it.
        stages: PipelineScope,
        /// The accesses performed.
        access: AccessMask,
        /// What the use is for.
        intent: TextureUseIntent,
    },

    /// A presentation frame use.
    Frame {
        /// The acquired frame.
        frame: AcquiredFrameId,
        /// The stages that touch it.
        stages: PipelineScope,
        /// The accesses performed.
        access: AccessMask,
    },
}
