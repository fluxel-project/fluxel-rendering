//! W2's family wiring: `VulkanDevice` as the graphics, copy, compute,
//! indirect-dispatch and storage-role families' provider.
//!
//! The common layer's contract says a device *negotiates* a family and receives a
//! handle that already borrows the device and the ledger that proved it (plan
//! section 11). This module is the first real backend to do that: `VulkanDevice`
//! implements [`Provides<Graphics>`], [`Provides<Copy>`], [`Provides<Compute>`],
//! [`Provides<IndirectDispatch>`], [`Provides<StorageBuffer>`] and
//! [`Provides<StorageTexture>`] and hands out [`GraphicsRecording`],
//! [`CopyRecording`], [`ComputeRecording`], [`IndirectDispatchRecording`],
//! [`StorageBufferBindings`] and [`StorageTextureBindings`], which implement
//! [`FamilyApi`] and their own family's trait over the device's own resource table --
//! and, for the command families, over one recording from the device's own command
//! pool.
//!
//! # One recording engine, one handle type per family
//!
//! [`Recorder`] is the machinery the command handles share: the recording begun by the
//! first command that needs one, the three states that recording can be in, and the
//! pass targets a raster bracket names. It moved into its own module when the
//! execution layer became the consumer that must record several families into **one**
//! submission -- see [`super::recording`] -- and the handles below are now thin
//! wrappers over it.
//!
//! The handles stay distinct types on purpose. A single type implementing several
//! family traits would let a caller that negotiated only `Copy` reach a draw, because
//! a family verb does not re-ask the ledger once a handle exists -- so the type, not a
//! run-time check, is what keeps one family's vocabulary out of another's call site.
//! What each handle keeps is exactly that: the type that decides which verbs a caller
//! can name, plus the id resolution and the family-shaped refusals that go with it.
//!
//! [`StorageBufferBindings`] and [`StorageTextureBindings`] are *resource roles* rather
//! than command domains, so they own no recording at all: each resolves a base
//! resource id through the table and builds a value the graph's own bind-group step
//! places at a binding number. That is one place the wired families differ
//! structurally rather than only in their verbs, and it is stated by those types
//! having no `Recorder` field.
//!
//! [`IndirectDispatchRecording`] is the other. An indirect dispatch is a compute
//! command, so its handle wraps [`ComputeRecording`] rather than owning a second
//! recording of its own -- the bracket, the pipeline and the bindings are the compute
//! family's body -- and implements [`ComputeApi`] beside [`IndirectDispatchApi`]. That
//! is sound in one direction only, and the asymmetry is the point: the ledger records
//! `IndirectDispatch` only beside a proved `Compute` row, while [`ComputeRecording`]
//! does not implement [`IndirectDispatchApi`], so a caller that negotiated only
//! `Compute` still cannot name the indirect form.
//!
//! The graphics, copy and compute handles are each their own recording context, which
//! the contract permits (plan section 21: negotiation and recording context need not
//! remain the same object); the indirect handle shares the compute one because its
//! command is a compute command. A capability-oriented caller therefore negotiates one
//! handle per family, while a *submission* that names several families records them
//! into one [`Recorder`] -- the composition [`super::recording`] lands.
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
    BufferCopyRegion, BufferRange, BufferUsageKind, IndexFormat, RasterPassDescriptor,
    ResourceAccessState, ScissorRect, TextureCopyRegion, TextureFormat, TextureRange, Viewport,
};

use crate::common::api::families::{
    ComputeApi, CopyApi, IndirectDispatchApi, StorageBufferApi, StorageTextureApi,
};
use crate::common::api::family::{
    Compute, Copy, Graphics, IndirectDispatch, StorageBuffer, StorageTexture,
};
use crate::common::api::graphics::GraphicsApi;
use crate::common::api::handle::FamilyApi;
use crate::common::api::negotiate::Provides;
use crate::common::base::resource::{BufferId, TextureId};
use crate::common::base::stamp::DeviceStamp;

use super::bind_group::BindGroup;
use super::command::{Finished, RecordError};
use super::device::VulkanDevice;
use super::format;
use super::framebuffer::{Framebuffer, FramebufferError};
use super::pipeline::{ComputePipeline, RasterPipeline};
use super::recording::Recorder;
use super::render_pass::{self, PassError};
use super::storage::{self, StorageBufferBinding, StorageTextureBinding};

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
/// The recording's refusals -- not recording, no compute pass open, a second begin, a
/// dispatch with a zero group dimension -- are carried from [`RecordError`] rather
/// than restated, and the two id sentences are the handle's own, because a transition
/// resolves its resource through the table before the recorder is reached. The family
/// keeps its own error type, per the convention section 11.9 fixes: a backend that
/// later needs a compute-specific refusal has a place to put it without widening every
/// other family's sentence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ComputeError {
    /// The recording itself refused the command.
    Recording(RecordError),
    /// The id names no live buffer of this device generation.
    UnknownBuffer,
    /// The id names no live texture of this device generation.
    UnknownTexture,
}

/// The graphics family's handle on one `Vulkan` device.
///
/// It borrows the device -- so it cannot outlive the ledger that proved the family
/// -- and owns the one [`Recorder`] it records into. The recorder retains every pass
/// target the recording names rather than releasing them at `end_raster`, because a
/// recorded `vkCmdBeginRenderPass` refers to its framebuffer until the commands that
/// name it complete; the caller keeps the handle alive until submission reports
/// terminal, exactly as the borrowed path's recording does.
pub(crate) struct GraphicsRecording<'d> {
    /// The recording this handle records into, begun by the first command that
    /// needs one.
    ///
    /// It is begun lazily rather than by `provide` because `Provides::provide`
    /// cannot report a failure and allocating a command buffer can fail; it is begun
    /// by *any* command rather than only by `begin_raster` because the graph records
    /// its transitions before it opens the pass.
    recorder: Recorder<'d>,
}

impl<'d> GraphicsRecording<'d> {
    /// Wraps one device generation without touching the driver.
    ///
    /// Infallible, which is what `Provides::provide` requires: the recording itself
    /// is begun by the first verb that needs it.
    fn new(device: &'d VulkanDevice) -> Self {
        Self {
            recorder: Recorder::new(device),
        }
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
            .recorder
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.recorder
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
            .recorder
            .device()
            .table()
            .texture_desc(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let image = self
            .recorder
            .device()
            .table()
            .texture_image(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let mapped = format::image_format(desc.format).ok_or(GraphicsError::UnsupportedFormat)?;
        self.recorder
            .transition_image(image, mapped, range, before, after)
            .map_err(GraphicsError::Recording)
    }

    /// Ends the recording and hands it to submission.
    ///
    /// Not family vocabulary: the common contract has no submission verb yet, so
    /// this is how the piece that owns submission takes the ended command buffer. It
    /// takes `&mut self` rather than `self` on purpose -- the pass targets stay in
    /// the recorder, and the caller must keep the handle alive until the submission
    /// that names them reports terminal.
    ///
    /// A handle that never recorded has nothing to hand over, and one that already
    /// finished has nothing left; both answer the same sentence, and neither is a
    /// reason to begin a recording.
    pub(crate) fn finish(&mut self) -> Result<Finished, GraphicsError> {
        self.recorder.finish().map_err(GraphicsError::Recording)
    }
}

impl FamilyApi for GraphicsRecording<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.recorder.device().stamp()
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
        // command buffer can fail and this is the first verb that can report it --
        // and it is begun *before* the attachment set is admitted, so a pool failure
        // is named before a target is built.
        self.recorder
            .ensure_recording()
            .map_err(GraphicsError::Recording)?;

        // The preserved attachment rule runs first, before any id is resolved: a
        // pass with no pipeline that could run in it is refused for that reason
        // rather than for whatever its first attachment happens to name.
        let admitted = render_pass::admit(descriptor.colors, descriptor.depth_stencil.as_ref())
            .map_err(GraphicsError::Pass)?;
        let texture = *admitted.texture;
        let desc = self
            .recorder
            .device()
            .table()
            .texture_desc(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let view = self
            .recorder
            .device()
            .table()
            .texture_view(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let framebuffer = Framebuffer::create(self.recorder.device().device(), admitted, &desc, view)
            .map_err(GraphicsError::Target)?;
        // Only a target the driver accepted is kept: a refused begin leaves the
        // framebuffer to its own drop, so the recording names nothing that was
        // released. The recorder is what retains it, because the target outlives the
        // bracket and composed recordings must keep exactly what they name.
        self.recorder
            .begin_raster(framebuffer)
            .map_err(GraphicsError::Recording)
    }

    fn end_raster(&mut self) -> Result<(), Self::Error> {
        self.recorder.end_raster().map_err(GraphicsError::Recording)
    }

    fn set_raster_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<(), Self::Error> {
        self.recorder
            .set_raster_pipeline(pipeline)
            .map_err(GraphicsError::Recording)
    }

    fn set_bindings(&mut self, bindings: &Self::Bindings) -> Result<(), Self::Error> {
        self.recorder
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
            .recorder
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.recorder
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
            .recorder
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.recorder
            .set_index_buffer(handle, offset, index_format)
            .map_err(GraphicsError::Recording)
    }

    fn set_viewport(&mut self, viewport: Viewport) -> Result<(), Self::Error> {
        self.recorder
            .set_viewport(viewport)
            .map_err(GraphicsError::Recording)
    }

    fn set_scissor(&mut self, scissor: ScissorRect) -> Result<(), Self::Error> {
        self.recorder
            .set_scissor(scissor)
            .map_err(GraphicsError::Recording)
    }

    fn draw(&mut self, vertices: Range<u32>, instance_count: u32) -> Result<(), Self::Error> {
        self.recorder
            .draw(vertices, instance_count)
            .map_err(GraphicsError::Recording)
    }

    fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        instance_count: u32,
    ) -> Result<(), Self::Error> {
        self.recorder
            .draw_indexed(indices, instance_count)
            .map_err(GraphicsError::Recording)
    }
}

impl core::fmt::Debug for GraphicsRecording<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GraphicsRecording")
            .field("recorder", &self.recorder)
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
    recorder: Recorder<'d>,
}

impl<'d> CopyRecording<'d> {
    /// Wraps one device generation without touching the driver.
    ///
    /// Infallible, which is what `Provides::provide` requires: the recording itself
    /// is begun by the first copy that needs it.
    fn new(device: &'d VulkanDevice) -> Self {
        Self {
            recorder: Recorder::new(device),
        }
    }

    /// Ends the recording and hands it to submission.
    ///
    /// Not family vocabulary, for the reason [`GraphicsRecording::finish`] states:
    /// the common contract has no submission verb yet, so this is how the piece that
    /// owns submission takes the ended command buffer.
    pub(crate) fn finish(&mut self) -> Result<Finished, CopyError> {
        self.recorder.finish().map_err(CopyError::Recording)
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
            .recorder
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(CopyError::UnknownBuffer)?;
        self.recorder
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
            .recorder
            .device()
            .table()
            .texture_desc(texture)
            .ok_or(CopyError::UnknownTexture)?;
        let image = self
            .recorder
            .device()
            .table()
            .texture_image(texture)
            .ok_or(CopyError::UnknownTexture)?;
        let mapped = format::image_format(desc.format).ok_or(CopyError::Recording(
            RecordError::UnsupportedFormat,
        ))?;
        self.recorder
            .transition_image(image, mapped, range, before, after)
            .map_err(CopyError::Recording)
    }
}

impl FamilyApi for CopyRecording<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.recorder.device().stamp()
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
        let table = self.recorder.device().table();
        let source_handle = table.buffer_handle(source).ok_or(CopyError::UnknownBuffer)?;
        let source_size = table.buffer_size(source).ok_or(CopyError::UnknownBuffer)?;
        let destination_handle = table
            .buffer_handle(destination)
            .ok_or(CopyError::UnknownBuffer)?;
        let destination_size = table
            .buffer_size(destination)
            .ok_or(CopyError::UnknownBuffer)?;
        self.recorder
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
        let table = self.recorder.device().table();
        let source_image = table.texture_image(source).ok_or(CopyError::UnknownTexture)?;
        let source_desc = table.texture_desc(source).ok_or(CopyError::UnknownTexture)?;
        let destination_image = table
            .texture_image(destination)
            .ok_or(CopyError::UnknownTexture)?;
        let destination_desc = table
            .texture_desc(destination)
            .ok_or(CopyError::UnknownTexture)?;
        self.recorder
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
            .field("recorder", &self.recorder)
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
/// # What it carries beyond the family trait, and why
///
/// The graph's own transitions, as crate-private methods. A dispatch that reads or
/// writes a storage binding needs that buffer or image ordered into the shader-storage
/// state first, and the barriers belong in the recording the dispatch is in -- so they
/// arrived with the storage-buffer family rather than with this one, exactly as
/// [`CopyRecording`]'s arrived with the copy verbs that first named a transfer layout.
/// They are not `ComputeApi` verbs, because a pipeline barrier is a backend mechanism
/// and the common contract carries none (plan section 1).
pub(crate) struct ComputeRecording<'d> {
    /// The recording this handle records into, begun by the first compute command.
    recorder: Recorder<'d>,
}

impl<'d> ComputeRecording<'d> {
    /// Wraps one device generation without touching the driver.
    ///
    /// Infallible, which is what `Provides::provide` requires: the recording itself
    /// is begun by the first verb that needs it.
    fn new(device: &'d VulkanDevice) -> Self {
        Self {
            recorder: Recorder::new(device),
        }
    }

    /// Records the barrier one portable buffer transition requires.
    ///
    /// Deliberately not a `ComputeApi` verb: a pipeline barrier is a backend
    /// mechanism and the common contract carries none (plan section 1). It arrived
    /// with the storage-buffer family, because a dispatch that reads or writes a
    /// storage binding is the first compute command whose resources need ordering --
    /// the same reason [`CopyRecording::transition_buffer`] exists, and the reason
    /// this family had no transitions until now. It resolves the id exactly as the
    /// other handles do.
    pub(crate) fn transition_buffer(
        &mut self,
        buffer: BufferId,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), ComputeError> {
        let handle = self
            .recorder
            .device()
            .table()
            .buffer_handle(buffer)
            .ok_or(ComputeError::UnknownBuffer)?;
        self.recorder
            .transition_buffer(handle, range, before, after)
            .map_err(ComputeError::Recording)
    }

    /// Records the barrier one portable texture transition requires.
    ///
    /// The mapped `Vulkan` format is read from the texture's own description, so the
    /// one layout whose answer depends on the format -- a sampled read, and whether it
    /// is a depth layout -- keeps the single source of truth
    /// [`super::barrier::image_state`] already uses.
    pub(crate) fn transition_texture(
        &mut self,
        texture: TextureId,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), ComputeError> {
        let desc = self
            .recorder
            .device()
            .table()
            .texture_desc(texture)
            .ok_or(ComputeError::UnknownTexture)?;
        let image = self
            .recorder
            .device()
            .table()
            .texture_image(texture)
            .ok_or(ComputeError::UnknownTexture)?;
        let mapped = format::image_format(desc.format)
            .ok_or(ComputeError::Recording(RecordError::UnsupportedFormat))?;
        self.recorder
            .transition_image(image, mapped, range, before, after)
            .map_err(ComputeError::Recording)
    }

    /// Ends the recording and hands it to submission.
    ///
    /// Not family vocabulary, for the reason [`GraphicsRecording::finish`] states:
    /// the common contract has no submission verb yet, so this is how the piece that
    /// owns submission takes the ended command buffer.
    pub(crate) fn finish(&mut self) -> Result<Finished, ComputeError> {
        self.recorder.finish().map_err(ComputeError::Recording)
    }

    /// The recording context, for the sibling handle that shares it.
    ///
    /// [`IndirectDispatchRecording`] wraps this type rather than owning a second
    /// [`Recorder`], so a compute bracket and an indirect dispatch recorded through
    /// one handle share one command buffer. The accessor exists so the field stays
    /// private to this type while the sibling can still reach the table and the
    /// recorder through it.
    fn recorder_mut(&mut self) -> &mut Recorder<'d> {
        &mut self.recorder
    }
}

impl FamilyApi for ComputeRecording<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.recorder.device().stamp()
    }
}

impl ComputeApi for ComputeRecording<'_> {
    type Error = ComputeError;
    type Pipeline = ComputePipeline;
    type Bindings = BindGroup;

    fn begin_compute(&mut self) -> Result<(), Self::Error> {
        // The recording is begun here, not in `provide`, because allocating a command
        // buffer can fail and this is the first verb that can report it.
        self.recorder
            .begin_compute()
            .map_err(ComputeError::Recording)
    }

    fn end_compute(&mut self) -> Result<(), Self::Error> {
        self.recorder.end_compute().map_err(ComputeError::Recording)
    }

    fn set_compute_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<(), Self::Error> {
        self.recorder
            .set_compute_pipeline(pipeline)
            .map_err(ComputeError::Recording)
    }

    fn set_bindings(&mut self, bindings: &Self::Bindings) -> Result<(), Self::Error> {
        self.recorder
            .set_compute_bindings(bindings)
            .map_err(ComputeError::Recording)
    }

    fn dispatch(&mut self, groups: [u32; 3]) -> Result<(), Self::Error> {
        self.recorder
            .dispatch(groups)
            .map_err(ComputeError::Recording)
    }
}

impl core::fmt::Debug for ComputeRecording<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ComputeRecording")
            .field("recorder", &self.recorder)
            .finish_non_exhaustive()
    }
}

/// Why an indirect dispatch was refused.
///
/// The recorder's refusals -- not recording, no compute pass open, a second begin, a
/// command-buffer read the driver would reject -- are carried from [`RecordError`]
/// rather than restated, and the three id/usage sentences are the handle's own,
/// because the table lookup and the declared-usage check happen before the recorder is
/// reached. [`Self::UsageNotDeclared`] is the buffer step's own rule at a new boundary:
/// the portable usage mapping never widens (section 18 of the lead 3F plan), so a
/// buffer the graph created for vertices is not one a dispatch may read counts from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IndirectDispatchError {
    /// The recording itself refused the command.
    Recording(RecordError),
    /// The id names no live buffer of this device generation.
    UnknownBuffer,
    /// The id names no live texture of this device generation.
    ///
    /// The family's own verbs never take a texture, but the recording context it wraps
    /// does, so the sentence is carried rather than folded into [`Self::UnknownBuffer`].
    UnknownTexture,
    /// The buffer was created without the indirect usage, so no dispatch may read its
    /// counts.
    UsageNotDeclared,
}

/// Lowers the wrapped compute recording's refusal onto this family's sentence.
///
/// Total, and every arm is the same fact: this handle *is* a compute recording, so the
/// recorder's sentence and its two id sentences are this handle's too. There is no
/// impossible arm and therefore no invented value.
fn from_compute(error: ComputeError) -> IndirectDispatchError {
    match error {
        ComputeError::Recording(inner) => IndirectDispatchError::Recording(inner),
        ComputeError::UnknownBuffer => IndirectDispatchError::UnknownBuffer,
        ComputeError::UnknownTexture => IndirectDispatchError::UnknownTexture,
    }
}

/// The indirect-dispatch family's handle on one `Vulkan` device.
///
/// # Why this handle also implements `ComputeApi`
///
/// An indirect dispatch is a dispatch: `vkCmdDispatchIndirect` reads its counts from a
/// buffer instead of taking them as arguments, but the command still needs a compute
/// pipeline bound in the recording, so the handle that records it must own a compute
/// recording. It therefore implements [`ComputeApi`] beside [`IndirectDispatchApi`],
/// and that is sound in exactly one direction -- which is the asymmetry the design
/// turns on:
///
/// - `device::ledger` records `Capability::IndirectDispatch` **only** beside a proved
///   `Capability::Compute` row, and keeps that row's own numeric floor, so a caller
///   that negotiated the indirect family cannot reach an unproved compute capability;
/// - [`ComputeRecording`] does **not** implement [`IndirectDispatchApi`], so a caller
///   that negotiated only `Compute` still cannot name the indirect form.
///
/// It is a wrapper rather than a second recording implementation for the reason
/// [`Recorder`] exists: the bracket, the pipeline, the bindings and the transitions
/// are the compute family's own body, and a second copy would be the "second spelling
/// of one shape" the plan keeps refusing.
pub(crate) struct IndirectDispatchRecording<'d> {
    /// The compute recording this handle records into, begun by the first command.
    compute: ComputeRecording<'d>,
}

impl<'d> IndirectDispatchRecording<'d> {
    /// Wraps one device generation without touching the driver.
    ///
    /// Infallible, which is what `Provides::provide` requires: the recording itself is
    /// begun by the first verb that needs it.
    fn new(device: &'d VulkanDevice) -> Self {
        Self {
            compute: ComputeRecording::new(device),
        }
    }

    /// Records the barrier one portable buffer transition requires.
    ///
    /// Deliberately not an `IndirectDispatchApi` verb: a pipeline barrier is a backend
    /// mechanism and the common contract carries none (plan section 1). It is here
    /// because the command buffer a dispatch reads counts from is ordered by the graph
    /// before the dispatch, and it resolves the id exactly as the compute handle does.
    pub(crate) fn transition_buffer(
        &mut self,
        buffer: BufferId,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), IndirectDispatchError> {
        self.compute
            .transition_buffer(buffer, range, before, after)
            .map_err(from_compute)
    }

    /// Records the barrier one portable texture transition requires.
    pub(crate) fn transition_texture(
        &mut self,
        texture: TextureId,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), IndirectDispatchError> {
        self.compute
            .transition_texture(texture, range, before, after)
            .map_err(from_compute)
    }

    /// Ends the recording and hands it to submission.
    ///
    /// Not family vocabulary, for the reason [`GraphicsRecording::finish`] states: the
    /// common contract has no submission verb yet, so this is how the piece that owns
    /// submission takes the ended command buffer.
    pub(crate) fn finish(&mut self) -> Result<Finished, IndirectDispatchError> {
        self.compute.finish().map_err(from_compute)
    }
}

impl FamilyApi for IndirectDispatchRecording<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.compute.stamp()
    }
}

/// The compute recording the indirect handle wraps, exposed as this family's own.
///
/// The delegation is deliberate and not a second implementation: an indirect dispatch
/// is a compute command, so every compute verb records into the same buffer and answers
/// through one error type. The two families stay separable where it matters --
/// [`ComputeRecording`] cannot dispatch indirectly, and this handle exists only for a
/// device that proved the indirect row.
impl ComputeApi for IndirectDispatchRecording<'_> {
    type Error = IndirectDispatchError;
    type Pipeline = ComputePipeline;
    type Bindings = BindGroup;

    fn begin_compute(&mut self) -> Result<(), Self::Error> {
        self.compute.begin_compute().map_err(from_compute)
    }

    fn end_compute(&mut self) -> Result<(), Self::Error> {
        self.compute.end_compute().map_err(from_compute)
    }

    fn set_compute_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<(), Self::Error> {
        self.compute
            .set_compute_pipeline(pipeline)
            .map_err(from_compute)
    }

    fn set_bindings(&mut self, bindings: &Self::Bindings) -> Result<(), Self::Error> {
        self.compute.set_bindings(bindings).map_err(from_compute)
    }

    fn dispatch(&mut self, groups: [u32; 3]) -> Result<(), Self::Error> {
        self.compute.dispatch(groups).map_err(from_compute)
    }
}

impl IndirectDispatchApi for IndirectDispatchRecording<'_> {
    type Error = IndirectDispatchError;

    fn dispatch_indirect(
        &mut self,
        commands: BufferId,
        offset: u64,
    ) -> Result<(), Self::Error> {
        // Every fact comes from the device's table and none from the driver: the
        // created size and the declared usage are the values the graph's own checks
        // saw, so the read cannot be admitted against a weaker fact than the buffer
        // was.
        let recorder = self.compute.recorder_mut();
        let table = recorder.device().table();
        let handle = table
            .buffer_handle(commands)
            .ok_or(IndirectDispatchError::UnknownBuffer)?;
        let size = table
            .buffer_size(commands)
            .ok_or(IndirectDispatchError::UnknownBuffer)?;
        let usage = table
            .buffer_usage(commands)
            .ok_or(IndirectDispatchError::UnknownBuffer)?;
        if !usage.contains(BufferUsageKind::Indirect) {
            return Err(IndirectDispatchError::UsageNotDeclared);
        }
        recorder
            .dispatch_indirect(handle, size, offset)
            .map_err(IndirectDispatchError::Recording)
    }
}

impl core::fmt::Debug for IndirectDispatchRecording<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("IndirectDispatchRecording")
            .field("recording", &self.compute)
            .finish_non_exhaustive()
    }
}

/// Why a storage-role binding was refused.
///
/// Every variant is a value returned before the driver is reached. Two are the
/// handle's own -- the id lookup and the usage check -- and the other two are
/// [`storage::RangeError`]'s sentences carried rather than restated, because "this
/// range is empty" and "this range does not fit" are different fixes.
///
/// [`Self::UsageNotDeclared`] is the check this family adds over the bind-group step,
/// and it is the buffer step's own rule applied at the point a binding is built: the
/// portable usage mapping never widens (section 18 of the lead 3F plan), so a storage
/// binding over a buffer the graph created for vertices would be an operation no
/// retained access declared.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageError {
    /// The id names no live buffer of this device generation.
    UnknownBuffer,
    /// The buffer was created without a storage usage, so no storage binding may name
    /// it.
    UsageNotDeclared,
    /// The binding's range is zero bytes.
    ZeroRange,
    /// The binding's range does not fit inside the buffer it names.
    RangeOutOfBounds,
}

/// The storage-buffer family's handle on one `Vulkan` device.
///
/// A *resource role* rather than a command domain: it gates how a binding is built and
/// records nothing, so unlike the three command families' handles it owns no encoder
/// and its `finish` does not exist. What it borrows is the device whose table answers
/// whether a buffer exists, how large it is and what it was declared for.
///
/// It is a distinct type from every other family's handle for the reason
/// [`CopyRecording`] states: a family verb does not re-ask the ledger once a handle
/// exists, so only the type keeps one family's vocabulary out of another's call site.
/// A caller that negotiated `StorageBuffer` therefore cannot reach a draw, and a
/// caller that negotiated nothing cannot build a storage binding at all.
pub(crate) struct StorageBufferBindings<'d> {
    /// The device whose table every fact this family checks comes from.
    device: &'d VulkanDevice,
}

impl<'d> StorageBufferBindings<'d> {
    /// Wraps one device generation without touching the driver.
    fn new(device: &'d VulkanDevice) -> Self {
        Self { device }
    }
}

impl FamilyApi for StorageBufferBindings<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.device.stamp()
    }
}

impl StorageBufferApi for StorageBufferBindings<'_> {
    type Error = StorageError;
    type Binding = StorageBufferBinding;

    fn create_storage_binding(
        &mut self,
        buffer: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Self::Binding, Self::Error> {
        // Every fact comes from the device's table and none from the driver: the
        // created size and the declared usage are the values the graph's own checks
        // saw, so a binding cannot be admitted against a weaker fact than the buffer
        // was.
        let table = self.device.table();
        let buffer_size = table.buffer_size(buffer).ok_or(StorageError::UnknownBuffer)?;
        let usage = table.buffer_usage(buffer).ok_or(StorageError::UnknownBuffer)?;
        let storage = usage.contains(BufferUsageKind::StorageRead)
            || usage.contains(BufferUsageKind::StorageWrite);
        if !storage {
            return Err(StorageError::UsageNotDeclared);
        }
        storage::check_range(offset, size, buffer_size).map_err(|error| match error {
            storage::RangeError::Zero => StorageError::ZeroRange,
            storage::RangeError::OutOfBounds => StorageError::RangeOutOfBounds,
        })?;
        Ok(StorageBufferBinding::new(buffer, offset, size))
    }
}

impl core::fmt::Debug for StorageBufferBindings<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("StorageBufferBindings")
            .field("device", &self.device.stamp())
            .finish_non_exhaustive()
    }
}

/// Why a storage-texture binding was refused.
///
/// Each variant is a value returned before the driver is reached, and the family keeps
/// its own sentences rather than reusing [`StorageError`]'s: the two roles are separate
/// families, so a texture-shaped refusal has no buffer-shaped sentence to borrow.
/// [`Self::UnknownTexture`] is the handle's own -- the id lookup -- and the other four
/// are [`storage::StorageTextureRuleError`]'s carried rather than restated, because
/// "the description names more levels than this vocabulary addresses", "the texture was
/// not declared for storage", "nothing examined its format" and "its format missed the
/// direction it declared" are four different fixes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageTextureError {
    /// The id names no live texture of this device generation.
    UnknownTexture,
    /// The description declares more than one mip level.
    MultiLevel {
        /// The level count the description declared.
        levels: u32,
    },
    /// The texture was created without a storage usage.
    UsageNotDeclared,
    /// Nothing examined the texture's `(format, sample count)` pair.
    FormatUnproved {
        /// The format the texture was created with.
        format: TextureFormat,
        /// The sample count the texture was created with.
        sample_count: u32,
    },
    /// The pair was examined and the declared direction is not supported.
    DirectionUnproved {
        /// The direction the declared usage named.
        direction: storage::StorageDirection,
        /// The format that failed to prove it.
        format: TextureFormat,
    },
}

/// The storage-texture family's handle on one `Vulkan` device.
///
/// The texture-shaped sibling of [`StorageBufferBindings`], and the same kind of value:
/// a resource role records nothing, so this handle owns no encoder and its whole
/// contract is a table lookup and a validated binding. What it borrows is the device
/// whose table answers whether a texture exists, what it was declared for, and -- from
/// the device's own [`FormatTable`](crate::common::formats::FormatTable) -- which
/// storage directions its format proved.
///
/// It is a distinct type from every other family's handle for the reason the buffer
/// role states: a family verb does not re-ask the ledger once a handle exists, so only
/// the type keeps one family's vocabulary out of another's call site. A caller that
/// negotiated `StorageTexture` therefore cannot build a storage *buffer* binding, and a
/// caller that negotiated nothing cannot build either.
pub(crate) struct StorageTextureBindings<'d> {
    /// The device whose table and format facts every fact this family checks come from.
    device: &'d VulkanDevice,
}

impl<'d> StorageTextureBindings<'d> {
    /// Wraps one device generation without touching the driver.
    fn new(device: &'d VulkanDevice) -> Self {
        Self { device }
    }
}

impl FamilyApi for StorageTextureBindings<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.device.stamp()
    }
}

impl StorageTextureApi for StorageTextureBindings<'_> {
    type Error = StorageTextureError;
    type Binding = StorageTextureBinding;

    fn create_storage_binding(
        &mut self,
        texture: TextureId,
    ) -> Result<Self::Binding, Self::Error> {
        // Every fact comes from the device and none from the driver: the description
        // and the declared usage are the values the graph's own checks saw, and the
        // format facts are what discovery recorded for exactly this pair.
        let table = self.device.table();
        let desc = table
            .texture_desc(texture)
            .ok_or(StorageTextureError::UnknownTexture)?;
        let usage = table
            .texture_usage(texture)
            .ok_or(StorageTextureError::UnknownTexture)?;
        let facts = self.device.formats().get(desc.format, desc.sample_count);
        storage::check_storage_texture(&desc, usage, facts).map_err(|error| match error {
            storage::StorageTextureRuleError::MultiLevel { levels } => {
                StorageTextureError::MultiLevel { levels }
            }
            storage::StorageTextureRuleError::UsageNotDeclared => {
                StorageTextureError::UsageNotDeclared
            }
            storage::StorageTextureRuleError::FormatUnproved {
                format,
                sample_count,
            } => StorageTextureError::FormatUnproved {
                format,
                sample_count,
            },
            storage::StorageTextureRuleError::DirectionUnproved { direction, format } => {
                StorageTextureError::DirectionUnproved { direction, format }
            }
        })?;
        Ok(StorageTextureBinding::new(texture))
    }
}

impl core::fmt::Debug for StorageTextureBindings<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("StorageTextureBindings")
            .field("device", &self.device.stamp())
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

/// The indirect-dispatch family's provider, beside the compute one.
///
/// A separate impl, for the reason [`Provides<Copy>`] states: the handle type bounds a
/// caller's vocabulary. The row this negotiation proves is `Capability::IndirectDispatch`,
/// which `device::ledger` records only beside a proved compute row and with that row's
/// own numeric floor -- so a device whose selected family does not report compute has
/// the vocabulary and still refuses the negotiation, the two refusals the design keeps
/// apart. The handle is [`IndirectDispatchRecording`], which is a compute recording
/// plus the one indirect verb; the module docs state why that one-directional
/// composition is sound.
impl Provides<IndirectDispatch> for VulkanDevice {
    type Api<'d> = IndirectDispatchRecording<'d>;

    fn provide(&self) -> IndirectDispatchRecording<'_> {
        IndirectDispatchRecording::new(self)
    }
}

/// The storage-buffer family's provider, beside the three command families'.
///
/// A separate impl, for the reason [`Provides<Copy>`] states: the handle type bounds a
/// caller's vocabulary, and a resource role is no exception -- a caller that
/// negotiated no storage family has no way to build a storage binding. The row this
/// negotiation proves is `Capability::StorageBuffer`, which `device::ledger` records
/// only where the device was created with the shader-store pair every declared stage's
/// write half needs, so a device without it has the vocabulary and still refuses the
/// negotiation -- the two refusals the design keeps apart.
impl Provides<StorageBuffer> for VulkanDevice {
    type Api<'d> = StorageBufferBindings<'d>;

    fn provide(&self) -> StorageBufferBindings<'_> {
        StorageBufferBindings::new(self)
    }
}

/// The storage-texture family's provider, beside the storage-buffer one.
///
/// A separate impl, for the reason [`Provides<StorageBuffer>`] states: the handle type
/// bounds a caller's vocabulary, and a resource role is no exception -- a caller that
/// negotiated no storage family has no way to build a storage binding. The row this
/// negotiation proves is `Capability::StorageImage`, which `device::ledger` records
/// only where the device was created with the shader-store pair **and** the format
/// table proved a format with both storage directions, so a device without it has the
/// vocabulary and still refuses the negotiation -- the two refusals the design keeps
/// apart.
impl Provides<StorageTexture> for VulkanDevice {
    type Api<'d> = StorageTextureBindings<'d>;

    fn provide(&self) -> StorageTextureBindings<'_> {
        StorageTextureBindings::new(self)
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
    use crate::common::api::negotiate::{CapabilitySource, require};
    use crate::common::binding::BindingResource;
    use crate::common::caps::Capability;
    use crate::native::vulkan::command::CommandPool;
    use crate::native::vulkan::compute::DispatchError;
    use crate::native::vulkan::indirect;
    use crate::native::vulkan::pipeline::{create_compute, create_layout, create_raster};
    use crate::native::vulkan::shader::MINIMAL_COMPUTE_SPIRV;
    use crate::native::vulkan::test_support::{
        colour_only_state, position_stream, raster_shaders, write_indirect_counts,
    };
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

    /// A device-local buffer created for shader storage.
    fn storage_buffer(opened: &mut open::OpenedVulkan, size: u64) -> BufferId {
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        opened
            .device
            .table_mut()
            .create_buffer(
                size,
                declared_buffer(&[
                    BufferUsageKind::StorageRead,
                    BufferUsageKind::StorageWrite,
                ]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local storage buffer")
    }

    /// A device-local texture created for shader storage, or `None` where no format
    /// this backend maps proved both storage directions on this board.
    fn storage_texture(opened: &mut open::OpenedVulkan, mip_levels: u32) -> Option<TextureId> {
        let format = opened
            .device
            .formats()
            .iter()
            .find(|facts| facts.storage_read && facts.storage_write)
            .map(|facts| facts.format)?;
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        Some(
            opened
                .device
                .table_mut()
                .create_texture(
                    TextureDesc {
                        dimension: TextureDimension::D2,
                        extent: Extent3d {
                            width: 8,
                            height: 8,
                            depth: 1,
                        },
                        mip_levels,
                        array_layers: 1,
                        sample_count: 1,
                        format,
                    },
                    TextureUsage::from_kinds([
                        TextureUsageKind::StorageRead,
                        TextureUsageKind::StorageWrite,
                    ]),
                    &memory_types,
                    memory::MemoryPurpose::DeviceLocal,
                )
                .expect("a device-local storage texture"),
        )
    }

    /// A device-local texture created for sampling, which is what a storage binding
    /// must refuse by name.
    fn sampled_texture(opened: &mut open::OpenedVulkan) -> TextureId {
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        opened
            .device
            .table_mut()
            .create_texture(
                TextureDesc {
                    dimension: TextureDimension::D2,
                    extent: Extent3d {
                        width: 8,
                        height: 8,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    sample_count: 1,
                    format: TextureFormat::Rgba8Unorm,
                },
                TextureUsage::from_kinds([TextureUsageKind::Sampled]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local sampled texture")
    }

    #[test]
    fn a_negotiated_storage_binding_is_validated_against_the_device_table() {
        // Step 13's storage-buffer family against the real driver: `require` yields the
        // handle, the handle reads the buffer's created size and declared usage from
        // the device's own table, and every refusal is a value returned before a
        // descriptor exists. Skips where no adapter exists, and where the adapter
        // cannot serve the family -- a device created without the shader-store pair is
        // not this test's subject.
        let Some(mut opened) = device() else {
            return;
        };
        if !opened.device.ledger().supports(Capability::StorageBuffer) {
            return;
        }
        let storage = storage_buffer(&mut opened, 256);
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
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

        let mut api = require::<_, StorageBuffer>(&opened.device)
            .expect("a device created with the shader-store pair proves the row");
        assert_eq!(
            api.stamp(),
            opened.device.stamp(),
            "the handle reports the generation it was negotiated on"
        );

        let binding = api
            .create_storage_binding(storage, 0, 256)
            .expect("a range that fits the buffer");
        assert_eq!(binding.buffer(), storage);
        assert_eq!(binding.offset(), 0);
        assert_eq!(binding.size(), 256);

        // The value carries base identity, never a driver handle, and reaches a bind
        // group at whatever number the layout declares -- which the family never saw.
        let entry = binding.at(1);
        assert_eq!(entry.binding, 1);
        assert_eq!(
            entry.resource,
            BindingResource::Buffer {
                buffer: storage,
                offset: 0,
                size: 256,
            }
        );

        // Every refusal is its own sentence, and none of them reached the driver.
        assert_eq!(
            api.create_storage_binding(storage, 0, 0),
            Err(StorageError::ZeroRange)
        );
        assert_eq!(
            api.create_storage_binding(storage, 128, 256),
            Err(StorageError::RangeOutOfBounds)
        );
        assert_eq!(
            api.create_storage_binding(vertices, 0, 4),
            Err(StorageError::UsageNotDeclared),
            "the buffer step's mapping never widens, so a vertex buffer is not a storage one"
        );
        let foreign = BufferId::new(
            opened.device.stamp().next_generation(),
            PhysicalResourceIdentity::new(1),
        );
        assert_eq!(
            api.create_storage_binding(foreign, 0, 4),
            Err(StorageError::UnknownBuffer)
        );
    }

    #[test]
    fn a_negotiated_storage_texture_binding_reads_the_table_and_the_format_facts() {
        // Step 13's storage-texture role against the real driver: `require` yields the
        // handle, the handle reads the texture's description and declared usage from the
        // device's own table and the `(format, sample count)` facts from the device's own
        // format table, and every refusal is a value returned before a view or a
        // descriptor is touched. Skips where no adapter exists, where the adapter cannot
        // serve the family, and where no mapped format proved both storage directions.
        let Some(mut opened) = device() else {
            return;
        };
        if !opened.device.ledger().supports(Capability::StorageImage) {
            return;
        }
        let Some(storage) = storage_texture(&mut opened, 1) else {
            return;
        };
        let Some(multi_level) = storage_texture(&mut opened, 2) else {
            return;
        };
        let sampled = sampled_texture(&mut opened);

        let mut api = require::<_, StorageTexture>(&opened.device)
            .expect("a device whose format table proved a storage format proves the row");
        assert_eq!(
            api.stamp(),
            opened.device.stamp(),
            "the handle reports the generation it was negotiated on"
        );

        let binding = api
            .create_storage_binding(storage)
            .expect("a texture whose usage and format facts admit a storage binding");
        assert_eq!(binding.texture(), storage);
        // The value carries base identity, never the driver's view, and reaches a bind
        // group at whatever number the layout declares -- which the family never saw.
        let entry = binding.at(2);
        assert_eq!(entry.binding, 2);
        assert_eq!(entry.resource, BindingResource::Texture(storage));

        // Every refusal is its own sentence, and none of them reached the driver.
        assert_eq!(
            api.create_storage_binding(sampled),
            Err(StorageTextureError::UsageNotDeclared),
            "the texture step's mapping never widens, so a sampled texture is not a storage one"
        );
        assert_eq!(
            api.create_storage_binding(multi_level),
            Err(StorageTextureError::MultiLevel { levels: 2 }),
            "the table's view spans every declared level, which this vocabulary cannot name"
        );
        let foreign = TextureId::new(
            opened.device.stamp().next_generation(),
            PhysicalResourceIdentity::new(1),
        );
        assert_eq!(
            api.create_storage_binding(foreign),
            Err(StorageTextureError::UnknownTexture)
        );
    }

    #[test]
    fn a_compute_handle_orders_a_storage_binding_before_its_dispatch() {
        // The transitions the compute family gained with the storage-role family: the
        // graph's own barrier orders the buffer into the state its dispatch reads and
        // writes, and the whole recording is work the driver executes.
        let Some(mut opened) = device() else {
            return;
        };
        if !opened.device.ledger().supports(Capability::StorageBuffer) {
            return;
        }
        let storage = storage_buffer(&mut opened, 256);
        let pipeline = compute_pipeline(&opened);

        let mut api = require::<_, Compute>(&opened.device).expect("compute is proved");
        api.transition_buffer(
            storage,
            BufferRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ShaderStorageReadWrite,
        )
        .expect("the storage buffer enters the state the dispatch reads and writes");
        api.begin_compute().expect("the compute pass opens");
        api.set_compute_pipeline(&pipeline).expect("the pipeline binds");
        api.dispatch([1, 1, 1]).expect("a non-zero dispatch records");
        api.end_compute().expect("the compute pass closes");

        let finished = api.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "a dispatch ordered behind a storage transition runs to completion"
        );
        assert!(submission.is_terminal());
    }

    #[test]
    fn a_refused_compute_transition_is_a_value_and_leaves_the_recording_usable() {
        // The two id sentences are the handle's own; the contract says the executor
        // still ends a recording after a callback error, so neither may poison it.
        let Some(opened) = device() else {
            return;
        };
        let mut api = require::<_, Compute>(&opened.device).expect("compute is proved");
        let foreign = TextureId::new(
            opened.device.stamp().next_generation(),
            PhysicalResourceIdentity::new(1),
        );
        assert_eq!(
            api.transition_texture(
                foreign,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ShaderStorageRead,
            ),
            Err(ComputeError::UnknownTexture)
        );
        assert_eq!(
            api.transition_buffer(
                BufferId::new(foreign.stamp(), foreign.identity()),
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ShaderStorageRead,
            ),
            Err(ComputeError::UnknownBuffer)
        );

        // The refused transitions did not poison the recording: a real pass still
        // records and still ends.
        let pipeline = compute_pipeline(&opened);
        api.begin_compute().expect("the compute pass opens");
        api.set_compute_pipeline(&pipeline)
            .expect("the recording is usable");
        api.dispatch([1, 1, 1]).expect("a real dispatch records");
        api.end_compute().expect("the pass closes");
        api.finish().expect("the recording ends");
    }

    /// Writes `counts` into an indirect command buffer and leaves it readable.
    ///
    /// Recorded in its own submission because the family handle begins its own
    /// recording lazily: the counts have to exist before the dispatch that reads them
    /// is recorded, and this backend owns no indirect-command writer -- producing the
    /// counts is the graph's own business. The buffer is ordered into `CopyDestination`
    /// for the write, and the family's own barrier moves it to `IndirectRead` where the
    /// dispatch reads it.
    fn initialise_indirect_counts(
        opened: &open::OpenedVulkan,
        handle: vk::Buffer,
        counts: [u32; 3],
    ) {
        let pool = CommandPool::new(opened.device.device(), opened.device.selected_queue().family)
            .expect("a command pool on an opened device");
        let mut encoder = pool.begin().expect("a recording encoder");
        encoder
            .transition_buffer(
                handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopyDestination,
            )
            .expect("the command buffer is writable");
        write_indirect_counts(
            opened.device.device(),
            encoder.command_buffer(),
            handle,
            counts,
        );
        let finished = encoder.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "the command buffer's counts are written"
        );
    }

    #[test]
    fn a_negotiated_indirect_dispatch_handle_records_the_indirect_form() {
        // Step 13's indirect-dispatch family against the real driver: `require` yields
        // the handle, the compute bracket and pipeline come from the recording it
        // wraps, the command buffer is read at an offset the pure rule checked, and the
        // submission reports complete. Skips where no adapter exists, and where the
        // adapter cannot serve the family -- a device whose selected family reports no
        // compute is not this test's subject.
        let Some(mut opened) = device() else {
            return;
        };
        if !opened
            .device
            .ledger()
            .supports(Capability::IndirectDispatch)
        {
            return;
        }
        let pipeline = compute_pipeline(&opened);
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let commands = opened
            .device
            .table_mut()
            .create_buffer(
                12,
                declared_buffer(&[
                    BufferUsageKind::Indirect,
                    BufferUsageKind::CopyDestination,
                ]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local indirect command buffer");
        let handle = opened
            .device
            .table()
            .buffer_handle(commands)
            .expect("a live buffer");
        initialise_indirect_counts(&opened, handle, [1, 1, 1]);

        let mut api = require::<_, IndirectDispatch>(&opened.device)
            .expect("a device whose selected family reports compute proves the row");
        assert_eq!(
            api.stamp(),
            opened.device.stamp(),
            "the handle reports the generation it was negotiated on"
        );

        // The graph's own barrier orders the command buffer into the state the
        // dispatch reads; the bracket, the pipeline and the command are the family's.
        api.transition_buffer(
            commands,
            BufferRange::Whole,
            ResourceAccessState::CopyDestination,
            ResourceAccessState::IndirectRead,
        )
        .expect("the command buffer enters the state the dispatch reads");
        api.begin_compute().expect("the compute pass opens");
        api.set_compute_pipeline(&pipeline)
            .expect("the pipeline binds");
        api.dispatch_indirect(commands, 0)
            .expect("an indirect dispatch records");
        api.end_compute().expect("the compute pass closes");

        let finished = api.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "an indirect dispatch recorded through the family handle runs to completion"
        );
        assert!(submission.is_terminal());
        // The handle is still alive here, so the pipeline outlived the submission that
        // named it; it is released only now, when nothing executes.
        drop(api);
    }

    #[test]
    fn a_refused_indirect_dispatch_is_a_value_and_leaves_the_recording_usable() {
        // The table's two sentences, the usage rule and the recorder's range sentences
        // are all values returned before the driver, and the contract says the executor
        // still ends a recording after a callback error -- so none of them may poison
        // the pass.
        let Some(mut opened) = device() else {
            return;
        };
        if !opened
            .device
            .ledger()
            .supports(Capability::IndirectDispatch)
        {
            return;
        }
        let pipeline = compute_pipeline(&opened);
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let commands = opened
            .device
            .table_mut()
            .create_buffer(
                64,
                declared_buffer(&[
                    BufferUsageKind::Indirect,
                    BufferUsageKind::CopyDestination,
                ]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local indirect command buffer");
        let vertices = opened
            .device
            .table_mut()
            .create_buffer(
                64,
                declared_buffer(&[BufferUsageKind::Vertex]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");

        let mut api =
            require::<_, IndirectDispatch>(&opened.device).expect("the indirect row is proved");

        // Without a compute pass every verb that needs one answers with the same
        // sentence, and the commands id is valid so the guard is what refuses it.
        assert_eq!(
            api.dispatch_indirect(commands, 0),
            Err(IndirectDispatchError::Recording(RecordError::NoPass))
        );

        // A foreign id has no record in the table, and a buffer the graph created for
        // vertices is not one a dispatch may read counts from.
        let foreign = BufferId::new(
            opened.device.stamp().next_generation(),
            PhysicalResourceIdentity::new(1),
        );
        assert_eq!(
            api.dispatch_indirect(foreign, 0),
            Err(IndirectDispatchError::UnknownBuffer)
        );
        assert_eq!(
            api.dispatch_indirect(vertices, 0),
            Err(IndirectDispatchError::UsageNotDeclared),
            "the buffer step's mapping never widens, so a vertex buffer is not an indirect one"
        );

        api.begin_compute().expect("the compute pass opens");
        // The range rules are the recorder's own sentences, decided before the driver.
        assert_eq!(
            api.dispatch_indirect(commands, 2),
            Err(IndirectDispatchError::Recording(RecordError::Indirect(
                indirect::IndirectRangeError::Misaligned
            )))
        );
        assert_eq!(
            api.dispatch_indirect(commands, 60),
            Err(IndirectDispatchError::Recording(RecordError::Indirect(
                indirect::IndirectRangeError::OutOfBounds
            )))
        );

        // The refused values did not poison the pass: the recording still ends.
        api.set_compute_pipeline(&pipeline)
            .expect("the recording is usable after the refusals");
        api.end_compute().expect("the pass closes");
        api.finish().expect("the recording ends");
        // A finished handle must not begin a second recording the caller never asked
        // for and would never submit.
        assert_eq!(
            api.dispatch_indirect(commands, 0),
            Err(IndirectDispatchError::Recording(RecordError::NotRecording))
        );
    }
}
