//! Step 14's first piece: the composition seat the execution layer records into.
//!
//! The family handles ([`super::family`]) each own *their own* recording, and that is
//! what plan section 21 permits: negotiation and recording context need not be the same
//! object, and a capability-oriented handle proves a family rather than a submission.
//! The RHI's execution layer, though, submits exactly one command buffer per graph
//! execution, and a graph may name several families -- a raster pass, a compute
//! dispatch and a copy in one recording, which is the X01 recipe. Composing several
//! families' recordings into one submission is therefore the *execution-layer
//! migration's* decision rather than a backend preference, and it is the decision this
//! module lands: [`Recorder`] is the one recording context a single submission is
//! recorded through.
//!
//! # One engine, no second spelling
//!
//! The three states a recording can be in, the lazy begin and the ended handoff were
//! already written once, for the graphics handle, and moved into a private
//! `Recording` type when the copy family became the second consumer (section 53.2 of
//! the lead 3F plan extracted it when two concrete consumers existed). This module is
//! the third consumer arriving, and it is the one that makes the engine *composable*
//! rather than per-handle: [`Recorder`] owns the same state machine plus the pass
//! targets a raster bracket names, and exposes every command this backend records.
//!
//! The family handles now delegate to it instead of owning a private copy, so there is
//! one place where "a recorded command belongs to an open bracket" is implemented and
//! one place where a finished recording refuses further work. What each handle keeps is
//! exactly what the one-handle-type-per-family rule requires: the type that decides
//! which verbs a caller can name.
//!
//! # Raw handles in, one [`RecordError`] out
//!
//! Every verb here takes a driver handle rather than a base id. Id resolution is the
//! *caller's* sentence -- a stale or foreign generation has no record in the device's
//! table, and the family handle maps that to its own `UnknownBuffer` / `UnknownTexture`
//! -- so this type never looks an id up and never returns a family-shaped refusal. Its
//! whole error surface is [`RecordError`], the recorder's own taxonomy, which the
//! handles carry into their family's error rather than restating.
//!
//! # Pass targets are retained here, and that is the composition
//!
//! A recorded `vkCmdBeginRenderPass` names its framebuffer until the commands that
//! reference it complete, so the target must outlive the recording it was recorded
//! into. Holding them beside the recording -- rather than in the graphics handle that
//! happens to call `begin_raster` -- is what lets a composed recording of several
//! families keep exactly the objects its own command buffer names.
//!
//! Deliberately absent: a device-owned recorder and any pooling of recordings. The
//! execution layer owns the one [`Recorder`] for one submission and drops it when the
//! submission reports terminal, which is the ownership the migration's adapter states.

use std::ops::Range;

use ash::vk;
use fluxel_rendergraph::{
    BufferCopyRegion, BufferRange, IndexFormat, ResourceAccessState, ScissorRect,
    TextureCopyRegion, TextureDesc, TextureRange, Viewport,
};

use super::bind_group::BindGroup;
use super::command::{Encoder, Finished, RecordError};
use super::device::VulkanDevice;
use super::framebuffer::Framebuffer;
use super::pipeline::{ComputePipeline, RasterPipeline};

/// The recording one [`Recorder`] owns, as the three states it can be in.
///
/// The state is a value rather than a bool pair because the three are genuinely
/// different sentences: `Fresh` begins a recording on demand, `Recording` is the live
/// one, and `Finished` refuses every further command -- a recording already handed to
/// submission must not silently begin a second one, which is what an `Option` alone
/// would allow.
///
/// The recording is boxed because an [`Encoder`] carries a loaded device function
/// table and is far larger than the other two variants; the indirection is one
/// allocation per recording, which is nothing beside the driver calls it enables.
enum Stage {
    /// Negotiated but nothing recorded yet; the first command begins the recording.
    Fresh,
    /// The live recording.
    Recording(Box<Encoder>),
    /// `finish` took the recording.
    Finished,
}

/// One recording context, composed across the families a single submission names.
///
/// It borrows the device -- so it cannot outlive the ledger that justified it -- and
/// owns the recording plus every pass target that recording names. See the module docs
/// for why this type exists beside the family handles rather than instead of them.
pub(crate) struct Recorder<'d> {
    /// The device whose command pool begins the recording.
    device: &'d VulkanDevice,
    /// The one command buffer, in whichever state it is in.
    stage: Stage,
    /// Every pass target this recording names, kept alive for its lifetime.
    targets: Vec<Framebuffer>,
}

impl<'d> Recorder<'d> {
    /// Wraps one device generation without touching the driver.
    ///
    /// Infallible: the recording itself is begun by the first verb that needs one, so
    /// allocating a command buffer can still be reported by that verb.
    pub(crate) fn new(device: &'d VulkanDevice) -> Self {
        Self {
            device,
            stage: Stage::Fresh,
            targets: Vec::new(),
        }
    }

    /// Returns the device this recording belongs to.
    ///
    /// The returned reference carries the caller's own `'d` rather than a borrow of
    /// `self`, so a caller can resolve a resource through the device's table and then
    /// take the recorder mutably without the two borrows being entangled.
    pub(crate) fn device(&self) -> &'d VulkanDevice {
        self.device
    }

    /// Begins the recording without recording anything.
    ///
    /// The raster bracket needs this: its attachment set is admitted and its
    /// framebuffer created *before* the first command is recorded, and a pool that
    /// cannot allocate must be reported as the recording's own refusal rather than
    /// surfacing after a target was built. Every command begins the recording anyway,
    /// so this is only the ordering hook.
    pub(crate) fn ensure_recording(&mut self) -> Result<(), RecordError> {
        self.encoder().map(|_| ())
    }

    /// Returns the live recording, beginning it if this is the first command.
    ///
    /// A finished recording refuses even though another command buffer could be
    /// allocated: recording past the handoff would be a second recording the caller
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
    /// A recording that never recorded has nothing to hand over, and one that already
    /// finished has nothing left; both answer the same sentence, and neither is a
    /// reason to begin a recording.
    pub(crate) fn finish(&mut self) -> Result<Finished, RecordError> {
        match core::mem::replace(&mut self.stage, Stage::Finished) {
            Stage::Recording(encoder) => (*encoder).finish(),
            Stage::Fresh | Stage::Finished => Err(RecordError::NotRecording),
        }
    }

    /// Records the barrier one semantic buffer transition requires.
    pub(crate) fn transition_buffer(
        &mut self,
        buffer: vk::Buffer,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), RecordError> {
        self.encoder()?
            .transition_buffer(buffer, range, before, after)
    }

    /// Records the barrier one semantic texture transition requires.
    ///
    /// `image_format` is the *mapped* `Vulkan` format the image was created with, so
    /// the one layout whose answer depends on the format keeps the single source of
    /// truth [`super::barrier`] already uses.
    pub(crate) fn transition_image(
        &mut self,
        image: vk::Image,
        image_format: vk::Format,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), RecordError> {
        self.encoder()?
            .transition_image(image, image_format, range, before, after)
    }

    /// Records the beginning of the raster pass `framebuffer` describes, and retains
    /// the target for as long as this recording names it.
    pub(crate) fn begin_raster(&mut self, framebuffer: Framebuffer) -> Result<(), RecordError> {
        self.encoder()?.begin_raster(&framebuffer)?;
        // Only a target the driver accepted is kept: a refused begin leaves the
        // framebuffer to its own drop, so the recording names nothing released.
        self.targets.push(framebuffer);
        Ok(())
    }

    /// Records the end of the raster pass this recording began.
    pub(crate) fn end_raster(&mut self) -> Result<(), RecordError> {
        self.encoder()?.end_raster()
    }

    /// Binds a raster pipeline for the draws recorded in the open pass.
    pub(crate) fn set_raster_pipeline(
        &mut self,
        pipeline: &RasterPipeline,
    ) -> Result<(), RecordError> {
        self.encoder()?.set_raster_pipeline(pipeline)
    }

    /// Binds a descriptor set for the draws recorded in the open raster pass.
    pub(crate) fn set_bindings(&mut self, group: &BindGroup) -> Result<(), RecordError> {
        self.encoder()?.set_bindings(group)
    }

    /// Selects one vertex buffer for the currently open raster pass.
    pub(crate) fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: vk::Buffer,
        offset: u64,
    ) -> Result<(), RecordError> {
        self.encoder()?.set_vertex_buffer(slot, buffer, offset)
    }

    /// Selects the index buffer for the currently open raster pass.
    pub(crate) fn set_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: u64,
        index_format: IndexFormat,
    ) -> Result<(), RecordError> {
        self.encoder()?
            .set_index_buffer(buffer, offset, index_format)
    }

    /// Sets the dynamic viewport the open pass draws through.
    pub(crate) fn set_viewport(&mut self, viewport: Viewport) -> Result<(), RecordError> {
        self.encoder()?.set_viewport(viewport)
    }

    /// Sets the scissor rectangle for the currently open raster pass.
    pub(crate) fn set_scissor(&mut self, scissor: ScissorRect) -> Result<(), RecordError> {
        self.encoder()?.set_scissor(scissor)
    }

    /// Records one non-indexed draw in the currently open raster pass.
    pub(crate) fn draw(
        &mut self,
        vertices: Range<u32>,
        instance_count: u32,
    ) -> Result<(), RecordError> {
        self.encoder()?.draw(vertices, instance_count)
    }

    /// Records one indexed draw in the currently open raster pass.
    pub(crate) fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        instance_count: u32,
    ) -> Result<(), RecordError> {
        self.encoder()?.draw_indexed(indices, instance_count)
    }

    /// Opens the compute pass the compute verbs record into.
    pub(crate) fn begin_compute(&mut self) -> Result<(), RecordError> {
        self.encoder()?.begin_compute()
    }

    /// Closes the compute pass this recording began.
    pub(crate) fn end_compute(&mut self) -> Result<(), RecordError> {
        self.encoder()?.end_compute()
    }

    /// Binds a compute pipeline for the dispatches recorded in the open compute pass.
    pub(crate) fn set_compute_pipeline(
        &mut self,
        pipeline: &ComputePipeline,
    ) -> Result<(), RecordError> {
        self.encoder()?.set_compute_pipeline(pipeline)
    }

    /// Binds a descriptor set for the dispatches recorded in the open compute pass.
    pub(crate) fn set_compute_bindings(&mut self, group: &BindGroup) -> Result<(), RecordError> {
        self.encoder()?.set_compute_bindings(group)
    }

    /// Records one `vkCmdDispatch` over `groups` workgroups per axis.
    pub(crate) fn dispatch(&mut self, groups: [u32; 3]) -> Result<(), RecordError> {
        self.encoder()?.dispatch(groups)
    }

    /// Records one `vkCmdDispatchIndirect` whose counts are read from `buffer`.
    ///
    /// `buffer_size` is the size the buffer was *created* with, supplied by the caller
    /// from the table, so the bound is the fact the graph's own check saw rather than a
    /// size recovered from the driver.
    pub(crate) fn dispatch_indirect(
        &mut self,
        buffer: vk::Buffer,
        buffer_size: u64,
        offset: u64,
    ) -> Result<(), RecordError> {
        self.encoder()?
            .dispatch_indirect(buffer, buffer_size, offset)
    }

    /// Records one buffer-to-buffer copy.
    pub(crate) fn copy_buffer(
        &mut self,
        source: vk::Buffer,
        source_size: u64,
        destination: vk::Buffer,
        destination_size: u64,
        region: BufferCopyRegion,
    ) -> Result<(), RecordError> {
        self.encoder()?
            .copy_buffer(source, source_size, destination, destination_size, region)
    }

    /// Records one image-to-image copy.
    pub(crate) fn copy_texture(
        &mut self,
        source: vk::Image,
        source_desc: &TextureDesc,
        destination: vk::Image,
        destination_desc: &TextureDesc,
        region: TextureCopyRegion,
    ) -> Result<(), RecordError> {
        self.encoder()?
            .copy_texture(source, source_desc, destination, destination_desc, region)
    }

    /// How many pass targets this recording still retains.
    ///
    /// A lifecycle fact rather than a tuning knob: a caller keeping the recorder alive
    /// until submission completes is keeping exactly these targets alive too.
    pub(crate) fn retained_targets(&self) -> usize {
        self.targets.len()
    }
}

impl core::fmt::Debug for Recorder<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Recorder")
            .field("recording", &matches!(self.stage, Stage::Recording(_)))
            .field("finished", &matches!(self.stage, Stage::Finished))
            .field("targets", &self.targets.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use fluxel_rendergraph::{BufferUsage, BufferUsageKind, CompletionStatus};

    use crate::Validation;
    use crate::native::vulkan::allocator::GpuAllocator;
    use crate::native::vulkan::open;
    use crate::native::vulkan::pipeline::{create_compute, create_layout, create_raster};
    use crate::native::vulkan::resource::ResourceTable;
    use crate::native::vulkan::shader::MINIMAL_COMPUTE_SPIRV;
    use crate::native::vulkan::test_support::{
        colour_only_state, colour_target_pass, position_stream, raster_shaders,
    };
    use crate::native::vulkan::{memory, submission};

    use super::*;

    fn declared_buffer(kinds: &[BufferUsageKind]) -> BufferUsage {
        BufferUsage::from_kinds(kinds.iter().copied())
    }

    /// Opens a headless device, or returns `None` where no adapter exists: having no
    /// GPU is not what these tests are about.
    fn device() -> Option<open::OpenedVulkan> {
        open::open(Validation::Disabled, 0).ok()
    }

    /// A composed recording: buffer barriers and a copy, a raster pass, a compute
    /// dispatch and back to a copy, all in **one** command buffer and one submit.
    ///
    /// This is the shape the execution layer needs and the reason [`Recorder`] exists:
    /// a graph may name several families in one execution, and `Vulkan` submits one
    /// command buffer per execution. The recording is composed across every family this
    /// backend serves, and the driver executes the result to completion.
    #[test]
    fn one_recording_composes_every_family_and_the_driver_completes_it() {
        let Some(opened) = device() else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        // The resources live in their own table, as the recorder tests already do: the
        // composed recording owns the recording, not the resources.
        let mut table =
            ResourceTable::new(opened.device.device(), opened.device.stamp(), allocator);

        let (framebuffer, target_image, mapped_target) =
            colour_target_pass(&opened, &mut table, &memory_types);
        let source = table
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::CopySource]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy source");
        let destination = table
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::CopyDestination]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy destination");
        let source_handle = table.buffer_handle(source).expect("a live source buffer");
        let destination_handle = table
            .buffer_handle(destination)
            .expect("a live destination buffer");

        let raster = create_raster(
            opened.device.device(),
            create_layout(opened.device.device(), Vec::new()).expect("an empty layout"),
            &raster_shaders(),
            &position_stream(),
            &colour_only_state(),
        )
        .expect("a raster pipeline over the retained recipe");
        let compute = create_compute(
            opened.device.device(),
            create_layout(opened.device.device(), Vec::new()).expect("an empty layout"),
            &MINIMAL_COMPUTE_SPIRV,
            c"main",
        )
        .expect("a compute pipeline over the retained artifact");

        let mut recorder = Recorder::new(&opened.device);
        assert_eq!(recorder.retained_targets(), 0);

        // The graph's own order: the copy's two barriers, then the copy.
        recorder
            .transition_buffer(
                source_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopySource,
            )
            .expect("undefined to a copy source");
        recorder
            .transition_buffer(
                destination_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopyDestination,
            )
            .expect("undefined to a copy destination");
        recorder
            .copy_buffer(
                source_handle,
                256,
                destination_handle,
                256,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 256,
                },
            )
            .expect("a whole-buffer copy records");

        // A raster pass in the same recording, over a real framebuffer the recorder
        // now retains.
        recorder
            .transition_image(
                target_image,
                mapped_target,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .expect("undefined to a colour attachment");
        recorder.begin_raster(framebuffer).expect("the pass opens");
        assert_eq!(recorder.retained_targets(), 1);
        recorder
            .set_raster_pipeline(&raster)
            .expect("the pipeline binds");
        recorder
            .set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 8.0,
                min_depth: 0.0,
                max_depth: 1.0,
            })
            .expect("the viewport sets");
        recorder
            .set_scissor(ScissorRect {
                x: 0,
                y: 0,
                width: 16,
                height: 8,
            })
            .expect("the scissor sets");
        recorder.draw(0..3, 1).expect("a draw records");
        recorder.end_raster().expect("the pass closes");

        // A compute dispatch in the same recording.
        recorder.begin_compute().expect("the compute bracket opens");
        recorder
            .set_compute_pipeline(&compute)
            .expect("the compute pipeline binds");
        recorder.dispatch([1, 1, 1]).expect("a dispatch records");
        recorder.end_compute().expect("the compute bracket closes");

        let finished = recorder.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "one recording composed across the families runs to completion"
        );
        assert!(submission.is_terminal());
        // The recorder still names the raster target, which is why the caller keeps it
        // alive until the submission above reported terminal.
        assert_eq!(recorder.retained_targets(), 1);
    }

    /// The two states a recording can be in that are not `Recording`, asserted without
    /// a driver: a fresh recorder has nothing to end, and a finished one records
    /// nothing further. Both are values rather than poison.
    #[test]
    fn a_finished_or_fresh_recording_refuses_rather_than_poisoning() {
        let Some(opened) = device() else {
            return;
        };
        let mut recorder = Recorder::new(&opened.device);

        // Nothing was recorded, so there is nothing to end -- and the handoff is a
        // state transition even then, because the sentence is the same as an
        // already-finished one: no later verb may silently begin a second recording
        // the caller never asked for and would never submit.
        assert!(matches!(recorder.finish(), Err(RecordError::NotRecording)));
        assert_eq!(recorder.begin_compute(), Err(RecordError::NotRecording));
        assert_eq!(recorder.end_raster(), Err(RecordError::NotRecording));
        assert!(matches!(recorder.finish(), Err(RecordError::NotRecording)));
        assert_eq!(recorder.retained_targets(), 0);
    }

    /// A used recording ends once, and the ended handoff refuses every further verb
    /// rather than being treated as idempotent.
    #[test]
    fn a_used_recording_ends_once_and_refuses_afterwards() {
        let Some(opened) = device() else {
            return;
        };
        let mut recorder = Recorder::new(&opened.device);
        recorder.begin_compute().expect("the compute bracket opens");
        recorder.end_compute().expect("the bracket closes");
        recorder.finish().expect("the recording ends");

        assert_eq!(recorder.begin_compute(), Err(RecordError::NotRecording));
        assert_eq!(recorder.end_compute(), Err(RecordError::NotRecording));
        assert!(matches!(recorder.finish(), Err(RecordError::NotRecording)));
    }

    /// A raster verb with no pass open, and a compute verb with the wrong bracket
    /// open, are the recorder's own sentences rather than a family's.
    #[test]
    fn the_composed_brackets_admit_only_their_own_verbs() {
        let Some(opened) = device() else {
            return;
        };
        let mut recorder = Recorder::new(&opened.device);
        assert_eq!(recorder.draw(0..3, 1), Err(RecordError::NoPass));
        assert_eq!(recorder.end_raster(), Err(RecordError::NoPass));
        assert_eq!(recorder.end_compute(), Err(RecordError::NoComputePass));

        // The two brackets share one slot, so a second begin of either kind -- here the
        // same kind, which needs no framebuffer -- answers the one refusal.
        recorder.begin_compute().expect("the compute bracket opens");
        assert_eq!(recorder.begin_compute(), Err(RecordError::PassAlreadyOpen));
    }
}
