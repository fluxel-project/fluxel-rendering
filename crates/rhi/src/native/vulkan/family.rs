//! W2's family wiring: `VulkanDevice` as the graphics, copy and compute families'
//! provider.
//!
//! The common layer's contract says a device *negotiates* a family and receives a
//! handle that already borrows the device and the ledger that proved it (plan
//! section 11). This module is the first real backend to do that: `VulkanDevice`
//! implements [`Provides<Graphics>`], [`Provides<Copy>`] and [`Provides<Compute>`]
//! and hands out [`GraphicsRecording`], [`CopyRecording`] and [`ComputeRecording`],
//! which implement [`FamilyApi`] and their own family's trait over the device's own
//! resource table and one recording from the device's own command pool.
//!
//! # One recording owner, one handle type per family
//!
//! [`Recording`] is the machinery the handles share: the recording begun by the
//! first command that needs one, and the three states that recording can be in. The
//! handles are distinct types on purpose. A single type implementing several family
//! traits would let a caller that negotiated only `Copy` reach a draw, because a
//! family verb does not re-ask the ledger once a handle exists -- so the type, not a
//! run-time check, is what keeps one family's vocabulary out of another's call site.
//!
//! Each handle is therefore its own recording context, which the contract permits
//! (plan section 21: negotiation and recording context need not remain the same
//! object). A graph execution that names two families negotiates two handles and so
//! produces two recordings; composing them into one submission is the
//! execution-layer migration's decision and is deliberately not invented here.
//!
//! # The encoder is private, and the recording is begun lazily
//!
//! Every family verb takes `&mut self`, so a handle can own the [`Encoder`] without a
//! backend encoder type appearing in the common vocabulary. The encoder is created
//! lazily, on the first command that needs one rather than by `provide`:
//! `Provides::provide` cannot report a failure, and allocating a command buffer can
//! fail, so the allocation happens at the verb that can return the family's
//! `Recording` sentence. A handle that was negotiated and never used therefore
//! allocates nothing.
//!
//! # Attachment ids resolve through the device's table, not through the caller
//!
//! [`GraphicsApi::begin_raster`] takes `RasterPassDescriptor<'_, TextureId>` -- the
//! portable descriptor, already generic over the resource type -- and a
//! `TextureId` is opaque. The table the device owns is what turns it into the image
//! view, format and sample count a `VkFramebuffer` needs, which is exactly why the
//! table moved onto the device: `provide(&self)` reaches the device and nothing
//! else. A [`BufferId`] resolves the same way for the two buffer verbs.
//!
//! The table is keyed by the whole id -- device stamp, generation and physical
//! identity -- so an id from another device or from a replaced generation is not a
//! special case the handle has to check for: it simply has no record, and the
//! refusal is `UnknownTexture` / `UnknownBuffer` rather than a driver call. That is
//! the same shape [`crate::common::api::handle::verify_texture`] states, reached
//! through the map key rather than through a second comparison.
//!
//! # What is deliberately not here
//!
//! No transitions and no submission verbs. A pipeline barrier is a backend mechanism
//! and the common contract has none (plan section 1), so the graph's own barrier is a
//! crate-private method on the handle rather than a family verb; copies are the
//! `Copy` family and reach the driver through [`CopyRecording`]; and the contract has
//! no submission verb yet, so each handle's `finish` hands its ended recording to
//! step 9's submission path without pretending to be family vocabulary.

use std::ops::Range;

use fluxel_rendergraph::{
    BufferCopyRegion, BufferRange, IndexFormat, RasterPassDescriptor, ResourceAccessState,
    ScissorRect, TextureCopyRegion, TextureRange, Viewport,
};

use crate::common::api::families::{ComputeApi, CopyApi};
use crate::common::api::family::{Compute, Copy, Graphics};
use crate::common::api::graphics::GraphicsApi;
use crate::common::api::handle::FamilyApi;
use crate::common::api::negotiate::Provides;
use crate::common::base::resource::{BufferId, TextureId};
use crate::common::base::stamp::DeviceStamp;

use super::bind_group::BindGroup;
use super::command::{Encoder, Finished, RecordError};
use super::device::VulkanDevice;
use super::format;
use super::framebuffer::{Framebuffer, FramebufferError};
use super::pipeline::{ComputePipeline, RasterPipeline};
use super::render_pass::{self, PassError};

/// Why a graphics command was refused.
///
/// Three layers can refuse and each keeps its own sentence rather than being
/// flattened into one:
///
/// - the pass's attachment set is not one this backend runs ([`Self::Pass`]) -- the
///   preserved one-colour-at-index-zero / no-depth-stencil rule;
/// - the pass target could not be created ([`Self::Target`], which carries
///   [`FramebufferError`]'s own distinctions: an unsupported shape, an unmapped
///   format and a driver refusal are different fixes);
/// - the recording refused the command ([`Self::Recording`], which carries
///   [`RecordError`]'s: not recording, no pass open, or a value the driver would
///   reject).
///
/// The two "no such resource" variants are the handle's own, because the table
/// lookup happens here rather than in any of those three.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphicsError {
    /// The recording itself refused the command.
    Recording(RecordError),
    /// The pass's attachment set is not one this backend runs.
    Pass(PassError),
    /// The render pass and framebuffer the pass needs could not be created.
    Target(FramebufferError),
    /// The id names no live buffer of this device generation.
    UnknownBuffer,
    /// The id names no live texture of this device generation.
    UnknownTexture,
    /// The texture's format is one this backend has not been taught, so no image
    /// handle it owns can be addressed.
    UnsupportedFormat,
}

/// Why a copy command was refused.
///
/// Two layers can refuse and each keeps its own sentence rather than being flattened
/// into one: the id lookup is the handle's own, and the recording's refusals -- not
/// recording, a raster pass still open, or a copy region the driver would reject --
/// are carried from [`RecordError`] rather than restated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CopyError {
    /// The recording itself refused the copy.
    Recording(RecordError),
    /// The id names no live buffer of this device generation.
    UnknownBuffer,
    /// The id names no live texture of this device generation.
    UnknownTexture,
}

/// Why a compute command was refused.
///
/// One layer can refuse: the recording, whose refusals -- not recording, no compute
/// pass open, a second begin, a dispatch with a zero group dimension -- are carried
/// from [`RecordError`] rather than restated. There is no id lookup here, because
/// every `ComputeApi` verb either takes a value the caller already holds ([`BindGroup`],
/// [`ComputePipeline`]) or takes none at all. The family keeps its own error type
/// anyway, per the convention section 11.9 fixes: a backend that later needs a
/// compute-specific refusal has a place to put it without widening every other
/// family's sentence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ComputeError {
    /// The recording itself refused the command.
    Recording(RecordError),
}

/// One handle's recording, as the three states it can be in.
///
/// The state is a value rather than a bool pair because the three are genuinely
/// different sentences: `Fresh` begins a recording on demand, `Recording` is the
/// live one, and `Finished` refuses every further command -- a handle whose
/// recording was already handed to submission must not silently begin a second one,
/// which is what an `Option` alone would allow.
///
/// The recording is boxed because an [`Encoder`] carries a loaded device function
/// table and is far larger than the other two variants; the indirection is one
/// allocation per recording, which is nothing beside the driver calls it enables.
enum Stage {
    /// Negotiated but nothing recorded yet; the first command begins the recording.
    Fresh,
    /// The live recording.
    Recording(Box<Encoder>),
    /// The handle's `finish` took the recording.
    Finished,
}

/// The recording one family handle owns, in whichever state it is in.
///
/// This is the machinery the graphics and copy handles share; it is deliberately not
/// itself a family handle. Only a concrete `Provides<F>::Api` type implements a
/// family trait, and keeping this type out of that position is what makes "a handle
/// negotiated for `Copy` cannot reach a draw" a fact of the type system.
struct Recording<'d> {
    device: &'d VulkanDevice,
    stage: Stage,
}

impl<'d> Recording<'d> {
    /// Wraps one device generation without touching the driver.
    fn new(device: &'d VulkanDevice) -> Self {
        Self {
            device,
            stage: Stage::Fresh,
        }
    }

    /// Returns the device this recording belongs to.
    ///
    /// The returned reference carries the handle's own `'d` rather than a borrow of
    /// `self`, so a family verb can resolve its ids through the device's table and
    /// then take the recording mutably without the two borrows being entangled.
    fn device(&self) -> &'d VulkanDevice {
        self.device
    }

    /// Returns the live recording, beginning it if this is the first command.
    ///
    /// A finished handle refuses even though it could allocate another command
    /// buffer: recording past the handoff would be a second recording the caller
    /// never asked for and would never submit.
    fn encoder(&mut self) -> Result<&mut Encoder, RecordError> {
        if let Stage::Finished = self.stage {
            return Err(RecordError::NotRecording);
        }
        if matches!(self.stage, Stage::Fresh) {
            let encoder = self.device.pool().begin()?;
            self.stage = Stage::Recording(Box::new(encoder));
        }
        match &mut self.stage {
            Stage::Recording(encoder) => Ok(encoder),
            // `Fresh` was replaced just above and `Finished` returned above, so this
            // arm is not reachable; it is a value rather than a panic for the same
            // reason every other impossible shape in this backend is.
            Stage::Fresh | Stage::Finished => Err(RecordError::NotRecording),
        }
    }

    /// Ends the recording and hands it to submission.
    ///
    /// A handle that never recorded has nothing to hand over, and one that already
    /// finished has nothing left; both answer the same sentence, and neither is a
    /// reason to begin a recording.
    fn finish(&mut self) -> Result<Finished, RecordError> {
        match core::mem::replace(&mut self.stage, Stage::Finished) {
            Stage::Recording(encoder) => (*encoder).finish(),
            Stage::Fresh | Stage::Finished => Err(RecordError::NotRecording),
        }
    }
}

impl core::fmt::Debug for Recording<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Recording")
            .field("recording", &matches!(self.stage, Stage::Recording(_)))
            .field("finished", &matches!(self.stage, Stage::Finished))
            .finish_non_exhaustive()
    }
}

/// The graphics family's handle on one `Vulkan` device.
///
/// It borrows the device -- so it cannot outlive the ledger that proved the family
/// -- and owns the one recording it records into and every pass target that
/// recording names. The targets are kept for the handle's whole life rather than
/// released at `end_raster`, because a recorded `vkCmdBeginRenderPass` refers to its
/// framebuffer until the commands that name it complete; the caller keeps the handle
/// alive until submission reports terminal, exactly as the borrowed path's recording
/// does.
pub(crate) struct GraphicsRecording<'d> {
    /// The recording this handle records into, begun by the first command that
    /// needs one.
    ///
    /// It is begun lazily rather than by `provide` because `Provides::provide`
    /// cannot report a failure and allocating a command buffer can fail; it is begun
    /// by *any* command rather than only by `begin_raster` because the graph records
    /// its transitions before it opens the pass.
    recording: Recording<'d>,
    /// Every pass target this recording names, kept alive for its lifetime.
    targets: Vec<Framebuffer>,
}

impl<'d> GraphicsRecording<'d> {
    /// Wraps one device generation without touching the driver.
    ///
    /// Infallible, which is what `Provides::provide` requires: the recording itself
    /// is begun by the first verb that needs it.
    fn new(device: &'d VulkanDevice) -> Self {
        Self {
            recording: Recording::new(device),
            targets: Vec::new(),
        }
    }

    /// Returns the live recording, in this family's sentence.
    fn encoder(&mut self) -> Result<&mut Encoder, GraphicsError> {
        self.recording.encoder().map_err(GraphicsError::Recording)
    }

    /// Records the barrier one portable buffer transition requires.
    ///
    /// Deliberately not a `GraphicsApi` verb: a pipeline barrier is a backend
    /// mechanism, and the common contract carries none (plan section 1). It is the
    /// entry point the graph's own lowering uses, and it resolves the id to a handle
    /// the same way the family verbs do.
    pub(crate) fn transition_buffer(
        &mut self,
        buffer: BufferId,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), GraphicsError> {
        let handle = self
            .recording
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.encoder()?
            .transition_buffer(handle, range, before, after)
            .map_err(GraphicsError::Recording)
    }

    /// Records the barrier one portable texture transition requires.
    ///
    /// The mapped `Vulkan` format is read from the texture's own description, so the
    /// one layout whose answer depends on the format -- a sampled read, and whether
    /// it is a depth layout -- keeps the single source of truth
    /// [`super::barrier::image_state`] already uses.
    pub(crate) fn transition_texture(
        &mut self,
        texture: TextureId,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), GraphicsError> {
        let desc = self
            .recording
            .device()
            .table()
            .texture_desc(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let image = self
            .recording
            .device()
            .table()
            .texture_image(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let mapped = format::image_format(desc.format).ok_or(GraphicsError::UnsupportedFormat)?;
        self.encoder()?
            .transition_image(image, mapped, range, before, after)
            .map_err(GraphicsError::Recording)
    }

    /// Ends the recording and hands it to submission.
    ///
    /// Not family vocabulary: the common contract has no submission verb yet, so
    /// this is how the piece that owns submission takes the ended command buffer. It
    /// takes `&mut self` rather than `self` on purpose -- the pass targets stay in
    /// the handle, and the caller must keep the handle alive until the submission
    /// that names them reports terminal.
    ///
    /// A handle that never recorded has nothing to hand over, and one that already
    /// finished has nothing left; both answer the same sentence, and neither is a
    /// reason to begin a recording.
    pub(crate) fn finish(&mut self) -> Result<Finished, GraphicsError> {
        self.recording.finish().map_err(GraphicsError::Recording)
    }
}

impl FamilyApi for GraphicsRecording<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.recording.device().stamp()
    }
}

impl GraphicsApi for GraphicsRecording<'_> {
    type Error = GraphicsError;
    type Pipeline = RasterPipeline;
    type Bindings = BindGroup;

    fn begin_raster(
        &mut self,
        descriptor: &RasterPassDescriptor<'_, TextureId>,
    ) -> Result<(), Self::Error> {
        // The recording is begun here, not in `provide`, because allocating a
        // command buffer can fail and this is the first verb that can report it.
        self.encoder()?;

        // The preserved attachment rule runs first, before any id is resolved: a
        // pass with no pipeline that could run in it is refused for that reason
        // rather than for whatever its first attachment happens to name.
        let admitted = render_pass::admit(descriptor.colors, descriptor.depth_stencil.as_ref())
            .map_err(GraphicsError::Pass)?;
        let texture = *admitted.texture;
        let desc = self
            .recording
            .device()
            .table()
            .texture_desc(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let view = self
            .recording
            .device()
            .table()
            .texture_view(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let framebuffer = Framebuffer::create(self.recording.device().device(), admitted, &desc, view)
            .map_err(GraphicsError::Target)?;
        // Only a target the driver accepted is kept: a refused begin leaves the
        // framebuffer to its own drop, so the recording names nothing that was
        // released.
        self.encoder()?
            .begin_raster(&framebuffer)
            .map_err(GraphicsError::Recording)?;
        self.targets.push(framebuffer);
        Ok(())
    }

    fn end_raster(&mut self) -> Result<(), Self::Error> {
        self.encoder()?.end_raster().map_err(GraphicsError::Recording)
    }

    fn set_raster_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<(), Self::Error> {
        self.encoder()?
            .set_raster_pipeline(pipeline)
            .map_err(GraphicsError::Recording)
    }

    fn set_bindings(&mut self, bindings: &Self::Bindings) -> Result<(), Self::Error> {
        self.encoder()?
            .set_bindings(bindings)
            .map_err(GraphicsError::Recording)
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: BufferId,
        offset: u64,
    ) -> Result<(), Self::Error> {
        let handle = self
            .recording
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.encoder()?
            .set_vertex_buffer(slot, handle, offset)
            .map_err(GraphicsError::Recording)
    }

    fn set_index_buffer(
        &mut self,
        buffer: BufferId,
        offset: u64,
        index_format: IndexFormat,
    ) -> Result<(), Self::Error> {
        let handle = self
            .recording
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.encoder()?
            .set_index_buffer(handle, offset, index_format)
            .map_err(GraphicsError::Recording)
    }

    fn set_viewport(&mut self, viewport: Viewport) -> Result<(), Self::Error> {
        self.encoder()?
            .set_viewport(viewport)
            .map_err(GraphicsError::Recording)
    }

    fn set_scissor(&mut self, scissor: ScissorRect) -> Result<(), Self::Error> {
        self.encoder()?
            .set_scissor(scissor)
            .map_err(GraphicsError::Recording)
    }

    fn draw(&mut self, vertices: Range<u32>, instance_count: u32) -> Result<(), Self::Error> {
        self.encoder()?
            .draw(vertices, instance_count)
            .map_err(GraphicsError::Recording)
    }

    fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        instance_count: u32,
    ) -> Result<(), Self::Error> {
        self.encoder()?
            .draw_indexed(indices, instance_count)
            .map_err(GraphicsError::Recording)
    }
}

impl core::fmt::Debug for GraphicsRecording<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GraphicsRecording")
            .field("recording", &self.recording)
            .field("targets", &self.targets.len())
            .finish_non_exhaustive()
    }
}

/// The copy family's handle on one `Vulkan` device.
///
/// It is a distinct type from [`GraphicsRecording`] so that a handle negotiated for
/// `Copy` cannot reach a draw: a family verb does not re-ask the ledger once a handle
/// exists, so only the type keeps one family's vocabulary out of another's call site.
/// It owns its own recording for the reason the module docs state, and it borrows the
/// device for the reason every handle does -- so it cannot outlive the ledger that
/// justified it.
pub(crate) struct CopyRecording<'d> {
    /// The recording this handle records into, begun by the first copy.
    recording: Recording<'d>,
}

impl<'d> CopyRecording<'d> {
    /// Wraps one device generation without touching the driver.
    ///
    /// Infallible, which is what `Provides::provide` requires: the recording itself
    /// is begun by the first copy that needs it.
    fn new(device: &'d VulkanDevice) -> Self {
        Self {
            recording: Recording::new(device),
        }
    }

    /// Ends the recording and hands it to submission.
    ///
    /// Not family vocabulary, for the reason [`GraphicsRecording::finish`] states:
    /// the common contract has no submission verb yet, so this is how the piece that
    /// owns submission takes the ended command buffer.
    pub(crate) fn finish(&mut self) -> Result<Finished, CopyError> {
        self.recording.finish().map_err(CopyError::Recording)
    }

    /// Records the barrier one portable buffer transition requires.
    ///
    /// Deliberately not a `CopyApi` verb: a pipeline barrier is a backend mechanism
    /// and the common contract carries none (plan section 1). It is here because a
    /// copy-only recording still needs it -- a buffer is ordered for the transfer
    /// stage before a copy reads or writes it -- and it resolves the id exactly as
    /// [`GraphicsRecording::transition_buffer`] does.
    pub(crate) fn transition_buffer(
        &mut self,
        buffer: BufferId,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), CopyError> {
        let handle = self
            .recording
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(CopyError::UnknownBuffer)?;
        self.recording
            .encoder()
            .map_err(CopyError::Recording)?
            .transition_buffer(handle, range, before, after)
            .map_err(CopyError::Recording)
    }

    /// Records the barrier one portable texture transition requires.
    ///
    /// A copy-only recording needs it more than the graphics one does: `Vulkan`
    /// requires an image to be in `TRANSFER_SRC_OPTIMAL` / `TRANSFER_DST_OPTIMAL`
    /// before [`CopyApi::copy_texture`] addresses it, and those are the two layouts
    /// the copy command names. The mapped `Vulkan` format is read from the texture's
    /// own description, so the depth fact keeps the single source of truth
    /// [`super::barrier::image_state`] already uses.
    pub(crate) fn transition_texture(
        &mut self,
        texture: TextureId,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), CopyError> {
        let desc = self
            .recording
            .device()
            .table()
            .texture_desc(texture)
            .ok_or(CopyError::UnknownTexture)?;
        let image = self
            .recording
            .device()
            .table()
            .texture_image(texture)
            .ok_or(CopyError::UnknownTexture)?;
        let mapped = format::image_format(desc.format).ok_or(CopyError::Recording(
            RecordError::UnsupportedFormat,
        ))?;
        self.recording
            .encoder()
            .map_err(CopyError::Recording)?
            .transition_image(image, mapped, range, before, after)
            .map_err(CopyError::Recording)
    }
}

impl FamilyApi for CopyRecording<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.recording.device().stamp()
    }
}

impl CopyApi for CopyRecording<'_> {
    type Error = CopyError;

    fn copy_buffer(
        &mut self,
        source: BufferId,
        destination: BufferId,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error> {
        // Both handles and both declared sizes come from the device's table. The
        // sizes are the ones the buffers were created with -- the same values the
        // graph's own check used -- rather than sizes recovered from the driver.
        let table = self.recording.device().table();
        let source_handle = table.buffer_handle(source).ok_or(CopyError::UnknownBuffer)?;
        let source_size = table.buffer_size(source).ok_or(CopyError::UnknownBuffer)?;
        let destination_handle = table
            .buffer_handle(destination)
            .ok_or(CopyError::UnknownBuffer)?;
        let destination_size = table
            .buffer_size(destination)
            .ok_or(CopyError::UnknownBuffer)?;
        self.recording
            .encoder()
            .map_err(CopyError::Recording)?
            .copy_buffer(
                source_handle,
                source_size,
                destination_handle,
                destination_size,
                region,
            )
            .map_err(CopyError::Recording)
    }

    fn copy_texture(
        &mut self,
        source: TextureId,
        destination: TextureId,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error> {
        // The image and the description it was created from, for both sides: the
        // aspect, the mip bounds, the layer rule and the format check are all derived
        // from those descriptions in `copy`, so this call spells none of them.
        let table = self.recording.device().table();
        let source_image = table.texture_image(source).ok_or(CopyError::UnknownTexture)?;
        let source_desc = table.texture_desc(source).ok_or(CopyError::UnknownTexture)?;
        let destination_image = table
            .texture_image(destination)
            .ok_or(CopyError::UnknownTexture)?;
        let destination_desc = table
            .texture_desc(destination)
            .ok_or(CopyError::UnknownTexture)?;
        self.recording
            .encoder()
            .map_err(CopyError::Recording)?
            .copy_texture(
                source_image,
                &source_desc,
                destination_image,
                &destination_desc,
                region,
            )
            .map_err(CopyError::Recording)
    }
}

impl core::fmt::Debug for CopyRecording<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CopyRecording")
            .field("recording", &self.recording)
            .finish_non_exhaustive()
    }
}

/// The compute family's handle on one `Vulkan` device.
///
/// A third distinct type, for the reason [`CopyRecording`] states: a family verb does
/// not re-ask the ledger once a handle exists, so a handle must not be able to reach a
/// verb whose row was never proved. `ComputeApi`'s pipe is the same shape as
/// `GraphicsApi`'s -- a bracket, a pipeline, a binding set and the command that does
/// the work -- but its bracket is [`super::command::Encoder::begin_compute`], which
/// `Vulkan` has no driver command for, and its binding is recorded through the
/// `COMPUTE` bind point.
///
/// # What it deliberately does not carry
///
/// No transitions: a dispatch that reads a storage buffer needs its barrier, but the
/// bindings a compute recipe declares are the storage-role families
/// (`StorageBufferApi` / `StorageTextureApi`), and those are not wired yet. The
/// transitions arrive with them, exactly as they arrived with the copy family when
/// its own verbs first named a transfer layout. What is here is the negotiation and
/// the recording bracket the compute row already proves.
pub(crate) struct ComputeRecording<'d> {
    /// The recording this handle records into, begun by the first compute command.
    recording: Recording<'d>,
}

impl<'d> ComputeRecording<'d> {
    /// Wraps one device generation without touching the driver.
    ///
    /// Infallible, which is what `Provides::provide` requires: the recording itself
    /// is begun by the first verb that needs it.
    fn new(device: &'d VulkanDevice) -> Self {
        Self {
            recording: Recording::new(device),
        }
    }

    /// Returns the live recording, in this family's sentence.
    fn encoder(&mut self) -> Result<&mut Encoder, ComputeError> {
        self.recording.encoder().map_err(ComputeError::Recording)
    }

    /// Ends the recording and hands it to submission.
    ///
    /// Not family vocabulary, for the reason [`GraphicsRecording::finish`] states:
    /// the common contract has no submission verb yet, so this is how the piece that
    /// owns submission takes the ended command buffer.
    pub(crate) fn finish(&mut self) -> Result<Finished, ComputeError> {
        self.recording.finish().map_err(ComputeError::Recording)
    }
}

impl FamilyApi for ComputeRecording<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.recording.device().stamp()
    }
}

impl ComputeApi for ComputeRecording<'_> {
    type Error = ComputeError;
    type Pipeline = ComputePipeline;
    type Bindings = BindGroup;

    fn begin_compute(&mut self) -> Result<(), Self::Error> {
        // The recording is begun here, not in `provide`, because allocating a command
        // buffer can fail and this is the first verb that can report it.
        self.encoder()?
            .begin_compute()
            .map_err(ComputeError::Recording)
    }

    fn end_compute(&mut self) -> Result<(), Self::Error> {
        self.encoder()?
            .end_compute()
            .map_err(ComputeError::Recording)
    }

    fn set_compute_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<(), Self::Error> {
        self.encoder()?
            .set_compute_pipeline(pipeline)
            .map_err(ComputeError::Recording)
    }

    fn set_bindings(&mut self, bindings: &Self::Bindings) -> Result<(), Self::Error> {
        self.encoder()?
            .set_compute_bindings(bindings)
            .map_err(ComputeError::Recording)
    }

    fn dispatch(&mut self, groups: [u32; 3]) -> Result<(), Self::Error> {
        self.encoder()?
            .dispatch(groups)
            .map_err(ComputeError::Recording)
    }
}

impl core::fmt::Debug for ComputeRecording<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ComputeRecording")
            .field("recording", &self.recording)
            .finish_non_exhaustive()
    }
}

/// The graphics family's provider.
///
/// Implementing this is what makes the family *expressible* here; whether a device
/// can serve it is the ledger's answer, and
/// [`require`](crate::common::api::negotiate::require) reads that before it calls
/// this. A device whose `Graphics` row is unproved therefore still has the
/// vocabulary and still refuses the negotiation -- the two refusals the design keeps
/// apart.
impl Provides<Graphics> for VulkanDevice {
    type Api<'d> = GraphicsRecording<'d>;

    fn provide(&self) -> GraphicsRecording<'_> {
        GraphicsRecording::new(self)
    }
}

/// The copy family's provider, beside the graphics one.
///
/// A separate impl rather than a second family on the graphics handle, for the reason
/// [`CopyRecording`] states: the handle type is what bounds a caller's vocabulary.
/// The row this negotiation proves is `Capability::Copy`, which `device::ledger`
/// records structurally for every device this backend opens -- and which is still
/// checked, so a device that had not proved it would refuse before `provide` ran.
impl Provides<Copy> for VulkanDevice {
    type Api<'d> = CopyRecording<'d>;

    fn provide(&self) -> CopyRecording<'_> {
        CopyRecording::new(self)
    }
}

/// The compute family's provider, beside the graphics and copy ones.
///
/// A separate impl, for the reason [`Provides<Copy>`] states: the handle type bounds a
/// caller's vocabulary. The row this negotiation proves is `Capability::Compute`,
/// which `device::ledger` records only where the selected queue family reports compute
/// -- so a device created on a graphics family without it has the vocabulary and still
/// refuses the negotiation, the two refusals the design keeps apart.
impl Provides<Compute> for VulkanDevice {
    type Api<'d> = ComputeRecording<'d>;

    fn provide(&self) -> ComputeRecording<'_> {
        ComputeRecording::new(self)
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use fluxel_rendergraph::{
        AttachmentOps, BufferUsage, BufferUsageKind, CompletionStatus, Extent3d, LoadOp,
        PhysicalResourceIdentity, RasterColorAttachment, StoreOp, TextureDesc, TextureDimension,
        TextureFormat, TextureUsage, TextureUsageKind, WriteCoverage,
    };

    use super::*;
    use crate::Validation;
    use crate::common::api::negotiate::require;
    use crate::native::vulkan::compute::DispatchError;
    use crate::native::vulkan::pipeline::{create_compute, create_layout, create_raster};
    use crate::native::vulkan::shader::MINIMAL_COMPUTE_SPIRV;
    use crate::native::vulkan::test_support::{colour_only_state, position_stream, raster_shaders};
    use crate::native::vulkan::{memory, open, submission};
    use ash::vk;

    /// Opens a headless device, or returns `None` where no adapter exists: having no
    /// GPU is not what these tests are about.
    fn device() -> Option<open::OpenedVulkan> {
        open::open(Validation::Disabled, 0).ok()
    }

    fn colour_desc() -> TextureDesc {
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 16,
                height: 8,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        }
    }

    fn clear_ops() -> AttachmentOps<[f32; 4]> {
        AttachmentOps {
            load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Full,
        }
    }

    fn declared_buffer(kinds: &[BufferUsageKind]) -> BufferUsage {
        BufferUsage::from_kinds(kinds.iter().copied())
    }

    /// The one colour target and the two draw buffers the device's own table owns.
    fn resources(
        opened: &mut open::OpenedVulkan,
    ) -> (TextureId, BufferId, BufferId, Vec<vk::MemoryType>) {
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let target = opened
            .device
            .table_mut()
            .create_texture(
                colour_desc(),
                TextureUsage::from_kinds([TextureUsageKind::ColorAttachment]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local colour target");
        let vertices = opened
            .device
            .table_mut()
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::Vertex]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");
        let indices = opened
            .device
            .table_mut()
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::Index]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local index buffer");
        (target, vertices, indices, memory_types)
    }

    fn viewport() -> Viewport {
        Viewport {
            x: 0.0,
            y: 0.0,
            width: 16.0,
            height: 8.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    fn scissor() -> ScissorRect {
        ScissorRect {
            x: 0,
            y: 0,
            width: 16,
            height: 8,
        }
    }

    #[test]
    fn a_negotiated_handle_records_a_whole_raster_pass_the_driver_completes() {
        // W2's first family wiring against the real driver: `require` yields the
        // handle, the handle's own table supplies the attachment and the draw
        // buffers, the whole bracket and both draws record through the family
        // vocabulary, and the recording is submitted to completion. Skips where no
        // adapter exists.
        let Some(mut opened) = device() else {
            return;
        };
        let (target, vertices, indices, _) = resources(&mut opened);
        let pipeline = create_raster(
            opened.device.device(),
            create_layout(opened.device.device(), Vec::new()).expect("an empty layout"),
            &raster_shaders(),
            &position_stream(),
            &colour_only_state(),
        )
        .expect("a raster pipeline over the retained recipe");

        let mut api = require::<_, Graphics>(&opened.device)
            .expect("a created device proves the graphics row");
        assert_eq!(
            api.stamp(),
            opened.device.stamp(),
            "the handle reports the generation it was negotiated on"
        );

        // The graph's own transition, which the family trait deliberately does not
        // carry: the pass's initial layout is the one this barrier leaves.
        api.transition_texture(
            target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentWrite,
        )
        .expect("the attachment enters the layout the pass begins at");

        let colors = [RasterColorAttachment {
            index: 0,
            texture: &target,
            range: TextureRange::Whole,
            operations: clear_ops(),
        }];
        let descriptor = RasterPassDescriptor {
            label: "graphics family",
            colors: &colors,
            depth_stencil: None,
        };
        api.begin_raster(&descriptor).expect("the pass opens");
        api.set_raster_pipeline(&pipeline)
            .expect("the pipeline binds");
        api.set_viewport(viewport()).expect("the viewport sets");
        api.set_scissor(scissor()).expect("the scissor sets");
        api.set_vertex_buffer(0, vertices, 0)
            .expect("the vertex buffer binds");
        api.set_index_buffer(indices, 0, IndexFormat::Uint16)
            .expect("the index buffer binds");
        api.draw(0..3, 1).expect("a non-indexed draw records");
        api.draw_indexed(0..3, 1).expect("an indexed draw records");
        api.end_raster().expect("the pass closes");

        let finished = api.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "a pass recorded through the family handle runs to completion"
        );
        assert!(submission.is_terminal());
        // The handle is still alive here, which is the point: the framebuffer the
        // recorded pass names outlived the submission that referenced it. It is
        // released only now, when nothing is executing.
        drop(api);
    }

    #[test]
    fn a_refused_graphics_command_is_a_value_and_leaves_the_recording_usable() {
        // The contract says the executor still ends a recording after a callback
        // error, so every refusal here is a value and the pass that follows it still
        // records. The three layers keep their own sentences.
        let Some(mut opened) = device() else {
            return;
        };
        let (target, _, _, _) = resources(&mut opened);
        let mut api = require::<_, Graphics>(&opened.device).expect("graphics is proved");

        // The first command begins the recording, so a raster verb with no pass open
        // answers the encoder's own sentence rather than "no recording".
        assert_eq!(
            api.set_viewport(viewport()),
            Err(GraphicsError::Recording(RecordError::NoPass))
        );
        assert_eq!(
            api.end_raster(),
            Err(GraphicsError::Recording(RecordError::NoPass))
        );

        // The pass admission runs before any id is resolved, so a depth-stencil
        // attachment is refused for what it is even though its texture id is
        // fabricated.
        let foreign = TextureId::new(
            opened.device.stamp().next_generation(),
            PhysicalResourceIdentity::new(1),
        );
        let colors = [RasterColorAttachment {
            index: 0,
            texture: &foreign,
            range: TextureRange::Whole,
            operations: clear_ops(),
        }];
        let depth = fluxel_rendergraph::RasterDepthStencilAttachment {
            texture: &foreign,
            range: TextureRange::Whole,
            depth: Some(AttachmentOps {
                load: LoadOp::DontCare,
                store: StoreOp::Discard,
                write_coverage: WriteCoverage::Full,
            }),
            stencil: None,
        };
        assert_eq!(
            api.begin_raster(&RasterPassDescriptor {
                label: "depth",
                colors: &colors,
                depth_stencil: Some(depth),
            }),
            Err(GraphicsError::Pass(PassError::DepthStencil))
        );

        // An id from a replaced generation has no record in the table, so it is
        // refused before the driver rather than reaching it.
        assert_eq!(
            api.begin_raster(&RasterPassDescriptor {
                label: "foreign",
                colors: &colors,
                depth_stencil: None,
            }),
            Err(GraphicsError::UnknownTexture)
        );
        assert_eq!(
            api.set_vertex_buffer(0, BufferId::new(foreign.stamp(), foreign.identity()), 0),
            Err(GraphicsError::UnknownBuffer)
        );

        // A real pass then opens, and the value refusals inside it are the
        // lowering's own sentences rather than a poisoned recording.
        let colors = [RasterColorAttachment {
            index: 0,
            texture: &target,
            range: TextureRange::Whole,
            operations: clear_ops(),
        }];
        api.transition_texture(
            target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentWrite,
        )
        .expect("the attachment enters the pass's layout");
        api.begin_raster(&RasterPassDescriptor {
            label: "recovery",
            colors: &colors,
            depth_stencil: None,
        })
        .expect("the pass opens");
        assert_eq!(
            api.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 0.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }),
            Err(GraphicsError::Recording(RecordError::Draw(
                super::super::draw::DrawError::Viewport
            )))
        );
        // The refused value did not poison the pass: a good one still records and
        // the recording still ends.
        api.set_viewport(viewport())
            .expect("the recording is usable");
        api.end_raster().expect("the pass closes");
        api.finish().expect("the recording ends");
        // A finished handle must not begin a second recording the caller never asked
        // for and would never submit, so the sentence is "not recording" rather than
        // a fresh command buffer.
        assert_eq!(
            api.set_viewport(viewport()),
            Err(GraphicsError::Recording(RecordError::NotRecording))
        );
    }

    /// The two copy buffers and the two copy textures the device's own table owns.
    fn copy_resources(
        opened: &mut open::OpenedVulkan,
    ) -> (BufferId, BufferId, TextureId, TextureId) {
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let source = opened
            .device
            .table_mut()
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::CopySource]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy source");
        let destination = opened
            .device
            .table_mut()
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::CopyDestination]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy destination");
        let source_texture = opened
            .device
            .table_mut()
            .create_texture(
                colour_desc(),
                TextureUsage::from_kinds([TextureUsageKind::CopySource]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy source texture");
        let destination_texture = opened
            .device
            .table_mut()
            .create_texture(
                colour_desc(),
                TextureUsage::from_kinds([TextureUsageKind::CopyDestination]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy destination texture");
        (source, destination, source_texture, destination_texture)
    }

    /// A whole-layer box, which is the one texture region this backend records.
    fn texture_region() -> TextureCopyRegion {
        TextureCopyRegion {
            source_origin: [0, 0, 0],
            destination_origin: [4, 2, 0],
            extent: [8, 4, 1],
            source_mip_level: 0,
            destination_mip_level: 0,
        }
    }

    #[test]
    fn a_negotiated_copy_handle_records_both_routes_the_driver_completes() {
        // W2's second family wiring against the real driver: `require` yields the
        // copy handle, the handle's own table supplies the two buffers and the two
        // images, the graph's transitions and both copy routes record through the
        // family vocabulary, and the recording is submitted to completion. Skips
        // where no adapter exists.
        let Some(mut opened) = device() else {
            return;
        };
        let (source, destination, source_texture, destination_texture) =
            copy_resources(&mut opened);

        let mut api = require::<_, crate::common::api::family::Copy>(&opened.device)
            .expect("a created device proves the copy row");
        assert_eq!(
            api.stamp(),
            opened.device.stamp(),
            "the handle reports the generation it was negotiated on"
        );

        // The graph's own transitions, which the family trait deliberately does not
        // carry: both sources are readable and both destinations writable through the
        // transfer stage before either copy records.
        api.transition_buffer(
            source,
            BufferRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::CopySource,
        )
        .expect("the source buffer is readable");
        api.transition_buffer(
            destination,
            BufferRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::CopyDestination,
        )
        .expect("the destination buffer is writable");
        api.transition_texture(
            source_texture,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::CopySource,
        )
        .expect("the source image is readable");
        api.transition_texture(
            destination_texture,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::CopyDestination,
        )
        .expect("the destination image is writable");

        api.copy_buffer(
            source,
            destination,
            BufferCopyRegion {
                source_offset: 0,
                destination_offset: 64,
                size: 128,
            },
        )
        .expect("a real buffer copy records");
        api.copy_texture(source_texture, destination_texture, texture_region())
            .expect("a real image copy records");

        let finished = api.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "copies recorded through the family handle run to completion"
        );
        assert!(submission.is_terminal());
        // The handle outlives the submission that named its resources, exactly as the
        // graphics handle does; it is released only now, when nothing is executing.
        drop(api);
    }

    #[test]
    fn a_refused_copy_is_a_value_and_leaves_the_recording_usable() {
        // Every copy refusal is a value: the two id sentences are the handle's own,
        // the region refusals are the recorder's, and the recording stays endable
        // after either. The contract says the executor still ends a recording after a
        // callback error, so a poisoned recording here would violate it.
        let Some(mut opened) = device() else {
            return;
        };
        let (source, destination, source_texture, _) = copy_resources(&mut opened);
        let mut api = require::<_, crate::common::api::family::Copy>(&opened.device)
            .expect("copy is proved");

        // An id from a replaced generation has no record in the table, so it is
        // refused before the driver is reached.
        let foreign_texture = TextureId::new(
            opened.device.stamp().next_generation(),
            PhysicalResourceIdentity::new(1),
        );
        let foreign_buffer = BufferId::new(foreign_texture.stamp(), foreign_texture.identity());
        assert_eq!(
            api.copy_buffer(
                foreign_buffer,
                destination,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 4,
                },
            ),
            Err(CopyError::UnknownBuffer)
        );
        assert_eq!(
            api.copy_texture(source_texture, foreign_texture, texture_region()),
            Err(CopyError::UnknownTexture)
        );

        // A region the driver would reject keeps its own sentence rather than
        // becoming a driver error or being folded into the id refusal.
        assert_eq!(
            api.copy_buffer(
                source,
                destination,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 2,
                    size: 4,
                },
            ),
            Err(CopyError::Recording(RecordError::Region(
                super::super::copy::CopyRegionError::Misaligned
            )))
        );

        // The refused values did not poison the recording: a real copy still records
        // and the recording still ends.
        api.copy_buffer(
            source,
            destination,
            BufferCopyRegion {
                source_offset: 0,
                destination_offset: 0,
                size: 64,
            },
        )
        .expect("a refused copy does not poison the recording");
        api.finish().expect("the recording ends");
        // A finished handle must not begin a second recording the caller never asked
        // for and would never submit.
        assert_eq!(
            api.copy_buffer(
                source,
                destination,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 64,
                },
            ),
            Err(CopyError::Recording(RecordError::NotRecording))
        );
    }

    /// A compute pipeline over an empty pipeline layout, which is the shape the
    /// retained kernel declares: it reads no bindings, so the layout names no set
    /// layouts and the dispatch is the whole recording.
    fn compute_pipeline(opened: &open::OpenedVulkan) -> ComputePipeline {
        create_compute(
            opened.device.device(),
            create_layout(opened.device.device(), Vec::new()).expect("an empty layout"),
            &MINIMAL_COMPUTE_SPIRV,
            c"main",
        )
        .expect("a compute pipeline over the retained kernel")
    }

    #[test]
    fn a_negotiated_compute_handle_records_a_dispatch_the_driver_completes() {
        // W2's third family wiring against the real driver: `require` yields the
        // compute handle, the handle records the bracket, a real pipeline and a real
        // dispatch, and the recording is submitted to completion. Skips where no
        // adapter exists.
        let Some(opened) = device() else {
            return;
        };
        let pipeline = compute_pipeline(&opened);

        let mut api = require::<_, Compute>(&opened.device)
            .expect("a device whose selected family reports compute proves the row");
        assert_eq!(
            api.stamp(),
            opened.device.stamp(),
            "the handle reports the generation it was negotiated on"
        );

        api.begin_compute().expect("the compute pass opens");
        api.set_compute_pipeline(&pipeline)
            .expect("the pipeline binds");
        api.dispatch([1, 1, 1]).expect("a non-zero dispatch records");
        api.end_compute().expect("the compute pass closes");

        let finished = api.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "a dispatch recorded through the family handle runs to completion"
        );
        assert!(submission.is_terminal());
        // The handle is still alive here, so the pipeline and layout outlived the
        // submission that named them. It is released only now, when nothing executes.
        drop(api);
    }

    #[test]
    fn a_refused_compute_command_is_a_value_and_leaves_the_recording_usable() {
        // Every compute refusal is the recording's own sentence, carried rather than
        // restated, and the contract says the executor still ends a recording after a
        // callback error -- so a refused dispatch must not poison the pass.
        let Some(opened) = device() else {
            return;
        };
        let pipeline = compute_pipeline(&opened);
        let mut api = require::<_, Compute>(&opened.device).expect("compute is proved");

        // With no compute pass open every verb that needs one answers `NoPass`, and
        // the close answers its own sentence rather than borrowing `NoPass`.
        assert_eq!(
            api.set_compute_pipeline(&pipeline),
            Err(ComputeError::Recording(RecordError::NoPass))
        );
        assert_eq!(
            api.dispatch([1, 1, 1]),
            Err(ComputeError::Recording(RecordError::NoPass))
        );
        assert_eq!(
            api.end_compute(),
            Err(ComputeError::Recording(RecordError::NoComputePass))
        );

        api.begin_compute().expect("the compute pass opens");
        // A second begin is a caller mistake rather than a nested pass.
        assert_eq!(
            api.begin_compute(),
            Err(ComputeError::Recording(RecordError::PassAlreadyOpen))
        );
        // A zero dimension is the dispatch's own sentence, not a driver no-op.
        assert_eq!(
            api.dispatch([1, 0, 1]),
            Err(ComputeError::Recording(RecordError::Dispatch(
                DispatchError::ZeroGroups([1, 0, 1])
            )))
        );

        // The refused values did not poison the pass: a real dispatch still records
        // and the recording still ends.
        api.set_compute_pipeline(&pipeline)
            .expect("the recording is usable");
        api.dispatch([1, 1, 1]).expect("a real dispatch records");
        api.end_compute().expect("the pass closes");
        api.finish().expect("the recording ends");
        // A finished handle must not begin a second recording the caller never asked
        // for and would never submit.
        assert_eq!(
            api.dispatch([1, 1, 1]),
            Err(ComputeError::Recording(RecordError::NotRecording))
        );
    }
}
