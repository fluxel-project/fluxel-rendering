//! Step 7's owning half: the command pool and the one recorder.
//!
//! The retained execution model submits one ordered command buffer per graph
//! execution on logical queue 0, so this module owns exactly the objects that
//! sentence needs: one [`CommandPool`] on the device's selected queue family, and
//! one [`Encoder`] holding one primary command buffer allocated from it.
//!
//! # The barrier is built by the pure half, never spelled here
//!
//! Every mask and layout in a recorded barrier comes from [`super::barrier`].
//! [`Encoder::transition_buffer`] and [`Encoder::transition_image`] lower the two
//! access states, refuse anything the lowering does not know, and only then call
//! `vkCmdPipelineBarrier`. That split is what keeps the preserved semantic -- a
//! `before == after` transition is still a barrier -- in one place: the recorder has
//! no `before == after` branch, so it cannot have gained one.
//!
//! # The raster bracket is explicit state, and commands illegal inside a pass refuse
//!
//! [`Encoder::begin_raster`] records `vkCmdBeginRenderPass` for a
//! [`Framebuffer`](super::framebuffer::Framebuffer), which owns the render pass and
//! the clear value its attachment states, and [`Encoder::end_raster`] records
//! `vkCmdEndRenderPass`. The encoder tracks whether a pass is open, and the state is
//! what refuses the commands `Vulkan` does not allow inside one: a barrier and a copy
//! answer [`RecordError::PassOpen`], a second `begin_raster` answers
//! [`RecordError::PassAlreadyOpen`], and an `end_raster` with no pass open answers
//! [`RecordError::NoPass`]. Ending the recording with a pass open is the same
//! refusal, because `vkEndCommandBuffer` requires no active render pass.
//!
//! # Refusals happen before the driver, and the encoder stays usable
//!
//! A state this backend has not been taught, a state of the wrong resource kind, a
//! zero-count subresource range and a transition on an encoder that is not recording
//! are all values returned without recording anything. The
//! [`ExecutionBackend`](fluxel_rendergraph::ExecutionBackend) contract says the
//! executor still calls the matching `end_*` after a callback error, so an encoder
//! that refused a transition is not poisoned: it can be ended normally, and its
//! already-recorded commands are unaffected.
//!
//! # Teardown, and what step 9 still owes
//!
//! A command buffer that is still recording when the encoder is dropped is freed,
//! which discards the recording without executing it -- exactly what the contract
//! asks of a dropped unfinished encoder. Submission does not exist yet (it is step
//! 9), so freeing on drop is safe here; that step owns the handoff that keeps a
//! finished buffer alive until its fence signals, and this module's note is the
//! reminder that the ownership changes there.

use ash::vk;
use fluxel_rendergraph::{
    BufferCopyRegion, BufferRange, ResourceAccessState, TextureCopyRegion, TextureRange,
};

use super::{barrier, copy, format, framebuffer::Framebuffer};

/// Why a command pool or an encoder could not be created or used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecordError {
    /// The driver refused to create the command pool.
    Pool(vk::Result),
    /// The driver refused to allocate a command buffer.
    Allocate(vk::Result),
    /// The driver reported a successful allocation but returned no command buffer.
    ///
    /// `ash` grows the output vector only on success, so this cannot happen with a
    /// conformant loader; it is a value rather than a panic so the impossible shape
    /// still leaves nothing allocated behind.
    NoCommandBuffer,
    /// The driver refused to begin recording.
    Begin(vk::Result),
    /// The driver refused to end recording.
    End(vk::Result),
    /// The encoder is not recording, so there is nothing to record into or to end.
    NotRecording,
    /// A raster pass is already open, so a second begin is a caller mistake rather
    /// than a nested pass.
    PassAlreadyOpen,
    /// No raster pass is open, so there is none to end.
    ///
    /// Refused rather than accepted idempotently for the same reason a second `end`
    /// is: silently accepting it would hide the caller's mistake until submission.
    NoPass,
    /// A raster pass is still open, so this command may not be recorded here.
    ///
    /// A barrier, a copy and the end of a recording are all illegal inside a render
    /// pass, and the encoder refuses them while one is open rather than recording a
    /// command the driver would reject.
    PassOpen,
    /// The state names an access this backend has not been taught how to order, or
    /// one that belongs to the other resource kind.
    UnsupportedState(ResourceAccessState),
    /// The subresource range is not one a barrier may name.
    UnsupportedRange,
    /// The copy region is not one the driver would accept, and the reason is
    /// carried rather than flattened: a misaligned buffer range and an
    /// out-of-bounds texture box are different mistakes to fix.
    Region(copy::CopyRegionError),
    /// The id names no live buffer of this device generation.
    UnknownBuffer,
    /// The id names no live texture of this device generation.
    UnknownTexture,
    /// The texture's format is one this backend has not been taught, so the copy
    /// record cannot name the image's aspect or check the two formats agree.
    UnsupportedFormat,
}

/// The one command pool of one device generation.
///
/// The pool is created for the device's selected queue family, because a command
/// buffer is only submittable to a queue of the family its pool was created for.
/// Keeping the family here states that fact next to the object it constrains rather
/// than at each `begin`.
pub(crate) struct CommandPool {
    device: ash::Device,
    pool: vk::CommandPool,
    queue_family: u32,
}

impl CommandPool {
    /// Creates a command pool for `queue_family` on `device`.
    ///
    /// No flags are passed, which claims nothing: the pool is neither transient nor
    /// individually resettable, and this backend allocates a fresh command buffer
    /// per recording rather than recycling one.
    pub(crate) fn new(device: &ash::Device, queue_family: u32) -> Result<Self, RecordError> {
        let create_info = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family);
        // SAFETY: `create_info` is a local that outlives the call, no allocation
        // callbacks are supplied, and the device is live for this pool's lifetime
        // because the caller owns it.
        let pool = unsafe { device.create_command_pool(&create_info, None) }
            .map_err(RecordError::Pool)?;
        Ok(Self {
            device: device.clone(),
            pool,
            queue_family,
        })
    }

    /// Returns the queue family this pool's command buffers are submittable to.
    pub(crate) const fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// Returns the raw pool handle.
    pub(crate) const fn handle(&self) -> vk::CommandPool {
        self.pool
    }

    /// Allocates one primary command buffer and begins recording it.
    ///
    /// The begin flags are left empty, which claims nothing: `ONE_TIME_SUBMIT` would
    /// promise the driver the buffer is submitted once, and the submission path that
    /// could keep that promise is step 9's.
    pub(crate) fn begin(&self) -> Result<Encoder, RecordError> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool belongs to the live device, the allocate info is a local
        // that outlives the call, and no allocation callbacks are supplied.
        let buffers = unsafe { self.device.allocate_command_buffers(&allocate_info) }
            .map_err(RecordError::Allocate)?;
        // Exactly one buffer was asked for. Anything else is a loader that broke its
        // own contract, so everything it did return is released before the refusal.
        let command_buffer = match buffers.as_slice() {
            [only] => *only,
            _ => {
                // SAFETY: every handle in `buffers` came from `self.pool`.
                unsafe { self.device.free_command_buffers(self.pool, &buffers) };
                return Err(RecordError::NoCommandBuffer);
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default();
        // SAFETY: the command buffer was just allocated from this pool and is in the
        // initial state, and `begin_info` is a local that outlives the call.
        if let Err(result) = unsafe { self.device.begin_command_buffer(command_buffer, &begin_info) }
        {
            // The buffer exists even though recording did not start, so it is freed
            // before the failure is returned: a refused encoder leaves nothing.
            // SAFETY: the handle was allocated from this pool.
            unsafe { self.device.free_command_buffers(self.pool, &[command_buffer]) };
            return Err(RecordError::Begin(result));
        }

        Ok(Encoder {
            device: self.device.clone(),
            pool: self.pool,
            command_buffer,
            recording: true,
            pass_open: false,
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // Destroying the pool implicitly frees every command buffer still allocated
        // from it, which is why an `Encoder` may hold only a handle and not an
        // ownership claim on the pool.
        // SAFETY: this is the only owner of the pool; no allocation callbacks were
        // supplied at creation, and the device outlives it by the caller's field
        // order.
        unsafe { self.device.destroy_command_pool(self.pool, None) };
    }
}

impl core::fmt::Debug for CommandPool {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CommandPool")
            .field("queue_family", &self.queue_family)
            .finish_non_exhaustive()
    }
}

/// One recording: a primary command buffer and the pool it came from.
///
/// The encoder owns the command buffer for its lifetime and frees it on drop. It
/// does not own the pool: a command buffer is freed *through* its pool, and the
/// device's pool must outlive every encoder created from it. That ordering is the
/// caller's to guarantee by field order, exactly as
/// [`ResourceTable`](super::resource::ResourceTable) requires its device to outlive
/// it; there is one such owner in the finished backend, so the invariant is stated
/// once rather than encoded as a lifetime every later integration step would have to
/// thread.
pub(crate) struct Encoder {
    device: ash::Device,
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    recording: bool,
    /// Whether a raster pass is open on this recording.
    ///
    /// It is state rather than a comment because the commands `Vulkan` forbids inside
    /// a render pass are refused from it, and because the matching `end_raster` is
    /// what clears it.
    pass_open: bool,
}

impl Encoder {
    /// Returns the raw command buffer handle.
    ///
    /// Exposed for the submission step, which is not this increment: nothing here
    /// submits, so a caller cannot yet turn this handle into work on a queue.
    pub(crate) const fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Returns whether this encoder is still recording.
    pub(crate) const fn is_recording(&self) -> bool {
        self.recording
    }

    /// Records the barrier one semantic buffer transition requires.
    ///
    /// Both the source and destination sides are lowered; nothing here compares
    /// `before` with `after`, so a same-state transition records the same barrier
    /// shape a state change does (with equal masks and the same layout).
    pub(crate) fn transition_buffer(
        &mut self,
        buffer: vk::Buffer,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass_open {
            return Err(RecordError::PassOpen);
        }
        let before_scope =
            barrier::buffer_scope(before).ok_or(RecordError::UnsupportedState(before))?;
        let after_scope =
            barrier::buffer_scope(after).ok_or(RecordError::UnsupportedState(after))?;
        let memory_barrier = barrier::buffer_barrier(buffer, range, before_scope, after_scope);

        // SAFETY: the command buffer is recording, the barrier was built from the
        // same two scopes that supply the stage masks, and both empty slices are
        // valid for the barrier kinds this call passes.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                before_scope.stage,
                after_scope.stage,
                vk::DependencyFlags::empty(),
                &[],
                &[memory_barrier],
                &[],
            );
        }
        Ok(())
    }

    /// Records the barrier one semantic texture transition requires.
    ///
    /// `image_format` is the *mapped* `Vulkan` format the image was created with,
    /// because the one layout whose answer depends on the format -- a sampled read --
    /// asks [`format::is_depth`] rather than second-guessing it.
    pub(crate) fn transition_image(
        &mut self,
        image: vk::Image,
        image_format: vk::Format,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass_open {
            return Err(RecordError::PassOpen);
        }
        let is_depth = format::is_depth(image_format);
        let before_state =
            barrier::image_state(before, is_depth).ok_or(RecordError::UnsupportedState(before))?;
        let after_state =
            barrier::image_state(after, is_depth).ok_or(RecordError::UnsupportedState(after))?;
        let subresource =
            barrier::image_subresource_range(range, is_depth).ok_or(RecordError::UnsupportedRange)?;
        let memory_barrier =
            barrier::image_barrier(image, subresource, before_state, after_state);

        // SAFETY: the command buffer is recording, the barrier was built from the
        // same two states that supply the stage masks, and both empty slices are
        // valid for the barrier kinds this call passes.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                before_state.scope.stage,
                after_state.scope.stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[memory_barrier],
            );
        }
        Ok(())
    }

    /// Records one buffer-to-buffer copy.
    ///
    /// The region is lowered and refused by value before the driver is reached, and
    /// the two sizes it is checked against are the sizes the buffers were *created*
    /// with -- read from the table by the caller -- rather than sizes recovered
    /// from the driver, because the creation size is what the graph's own check
    /// used.
    pub(crate) fn copy_buffer(
        &mut self,
        source: vk::Buffer,
        source_size: u64,
        destination: vk::Buffer,
        destination_size: u64,
        region: BufferCopyRegion,
    ) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass_open {
            return Err(RecordError::PassOpen);
        }
        let lowered = copy::buffer_copy(source_size, destination_size, region)
            .map_err(RecordError::Region)?;
        // SAFETY: the command buffer is recording, both handles were created by the
        // device this encoder belongs to and are alive for the recording's lifetime
        // because the table owns them, and `lowered` is a validated region.
        unsafe {
            self.device
                .cmd_copy_buffer(self.command_buffer, source, destination, &[lowered]);
        }
        Ok(())
    }

    /// Records one image-to-image copy.
    ///
    /// Both descriptions are the ones the images were created from and `format` is
    /// the mapped `Vulkan` format they both carry; the aspect and the layer rule are
    /// derived from those in [`copy`], so this method spells no aspect itself.
    /// Differing formats are refused by name rather than reaching a driver that
    /// would report a malformed copy.
    pub(crate) fn copy_texture(
        &mut self,
        source: vk::Image,
        source_desc: &fluxel_rendergraph::TextureDesc,
        destination: vk::Image,
        destination_desc: &fluxel_rendergraph::TextureDesc,
        region: TextureCopyRegion,
    ) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass_open {
            return Err(RecordError::PassOpen);
        }
        // The mapped format is what the copy record needs, and it is also the
        // cheapest honest answer to "is this a format this backend was taught": a
        // portable format with no equivalent cannot be copied through.
        let mapped = format::image_format(source_desc.format).ok_or(RecordError::UnsupportedFormat)?;
        let lowered = copy::image_copy(source_desc, destination_desc, region, mapped)
            .map_err(RecordError::Region)?;
        // SAFETY: the command buffer is recording, both handles were created by the
        // device this encoder belongs to and are alive for the recording's lifetime
        // because the table owns them, and `lowered` is a validated box whose
        // subresources exist on the images those descriptions created.
        unsafe {
            self.device.cmd_copy_image(
                self.command_buffer,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[lowered],
            );
        }
        Ok(())
    }

    /// Records the beginning of the raster pass `framebuffer` describes.
    ///
    /// The framebuffer owns both the render pass and the clear value its own
    /// attachment operations state, so this call cannot name a render pass the
    /// framebuffer was not built for, and the render area is the framebuffer's own
    /// extent -- `Vulkan` requires the two to agree. Subpass contents are `INLINE`,
    /// which is what a single-subpass pass with no secondary command buffers means.
    ///
    /// A second begin while a pass is open is refused rather than nested: `Vulkan`
    /// has no nested render passes, and a caller that reached one has a bug the type
    /// of a single bracket should have caught.
    pub(crate) fn begin_raster(&mut self, framebuffer: &Framebuffer) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass_open {
            return Err(RecordError::PassAlreadyOpen);
        }
        let clear_values = [framebuffer.clear_value()];
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(framebuffer.render_pass())
            .framebuffer(framebuffer.handle())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: framebuffer.extent(),
            })
            .clear_values(&clear_values);
        // SAFETY: the command buffer is recording and belongs to this encoder; the
        // render pass and the framebuffer were created together by this device and
        // the caller keeps both alive for the recording; `info` borrows locals that
        // outlive the call.
        unsafe {
            self.device
                .cmd_begin_render_pass(self.command_buffer, &info, vk::SubpassContents::INLINE);
        }
        self.pass_open = true;
        Ok(())
    }

    /// Records the end of the raster pass this encoder began.
    ///
    /// An `end_raster` with no pass open is refused rather than treated as
    /// idempotent, for the same reason a second `end` is: the caller's mistake would
    /// otherwise be hidden until the recording is submitted.
    pub(crate) fn end_raster(&mut self) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if !self.pass_open {
            return Err(RecordError::NoPass);
        }
        // SAFETY: the command buffer is recording with a pass open, which is the only
        // state in which `vkCmdEndRenderPass` is legal.
        unsafe { self.device.cmd_end_render_pass(self.command_buffer) };
        self.pass_open = false;
        Ok(())
    }

    /// Ends recording.
    ///
    /// A second call is refused rather than treated as idempotent: "end an encoder
    /// that is not recording" is a caller mistake, and silently accepting it would
    /// hide the mistake until submission. A recording with a raster pass still open
    /// is refused for the same reason: `vkEndCommandBuffer` requires no active render
    /// pass, and the caller's matching `end_raster` is what closes it.
    pub(crate) fn end(&mut self) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass_open {
            return Err(RecordError::PassOpen);
        }
        // SAFETY: the command buffer is recording and belongs to this encoder.
        unsafe { self.device.end_command_buffer(self.command_buffer) }.map_err(RecordError::End)?;
        self.recording = false;
        Ok(())
    }

    /// Ends recording if it is still open and hands the recording to step 9.
    ///
    /// This is the only way to obtain a [`Finished`], and that is the point: a
    /// command buffer is submittable only in the executable state, so the type
    /// makes "submit a recording that has not ended" unrepresentable rather than a
    /// run-time refusal. A recording that already ended passes through unchanged,
    /// and one that is still open is ended here.
    pub(crate) fn finish(mut self) -> Result<Finished, RecordError> {
        if self.recording {
            self.end()?;
        }
        Ok(Finished { encoder: self })
    }
}

/// A recording that has ended, so it is the only value submission accepts.
///
/// It owns the [`Encoder`] -- and therefore the command buffer -- for exactly as
/// long as the submission that holds it needs that buffer to stay alive, which is
/// the handoff the module docs above record as step 9's.
pub(crate) struct Finished {
    encoder: Encoder,
}

impl Finished {
    /// Returns the command buffer the submission must name.
    pub(crate) const fn command_buffer(&self) -> vk::CommandBuffer {
        self.encoder.command_buffer()
    }
}

impl core::fmt::Debug for Finished {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Finished")
            .field("command_buffer", &self.command_buffer())
            .finish_non_exhaustive()
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // Freeing a command buffer that is still recording discards the recording
        // without executing it, which is what a dropped unfinished encoder must do.
        // The buffer has never been submitted, because no submission path exists
        // yet; step 9 owns keeping a finished buffer alive until its fence signals.
        // SAFETY: the handle was allocated from `self.pool`, which outlives this
        // encoder, and it is not pending on any queue.
        unsafe {
            self.device
                .free_command_buffers(self.pool, &[self.command_buffer])
        };
    }
}

impl core::fmt::Debug for Encoder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Encoder")
            .field("recording", &self.recording)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::native::vulkan::resource::ResourceTable;
    use crate::native::vulkan::{allocator::GpuAllocator, memory, open};
    use fluxel_rendergraph::{
        BufferUsage, BufferUsageKind, Extent3d, TextureDesc, TextureDimension, TextureFormat,
        TextureUsage, TextureUsageKind,
    };

    /// Opens a headless device and a real command pool, or returns `None` where no
    /// adapter exists. Having no GPU is not what these tests are about.
    fn pool() -> Option<(open::OpenedVulkan, CommandPool)> {
        let opened = open::open(Validation::Disabled, 0).ok()?;
        let pool = CommandPool::new(opened.device.device(), opened.device.selected_queue().family)
            .expect("a command pool on an opened device");
        Some((opened, pool))
    }

    fn declared_buffer(kinds: &[BufferUsageKind]) -> BufferUsage {
        BufferUsage::from_kinds(kinds.iter().copied())
    }

    fn declared_texture(kinds: &[TextureUsageKind]) -> TextureUsage {
        TextureUsage::from_kinds(kinds.iter().copied())
    }

    #[test]
    fn a_real_command_pool_records_a_buffer_and_an_image_barrier() {
        // Step 7 against the real driver: one pool, one recording, one buffer
        // transition and one image transition, each driven by portable access states.
        // Skips where no adapter exists.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let pool_family = pool.queue_family();
        assert_eq!(pool_family, opened.device.selected_queue().family);

        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );

        let buffer = table
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::Vertex, BufferUsageKind::CopyDestination]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");
        let texture = table
            .create_texture(
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
                },
                declared_texture(&[
                    TextureUsageKind::ColorAttachment,
                    TextureUsageKind::CopySource,
                ]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local colour target");

        let buffer_handle = table.buffer_handle(buffer).expect("a live buffer");
        let image_handle = table.texture_image(texture).expect("a live texture");
        let image_format = format::image_format(TextureFormat::Rgba8Unorm).expect("a mapped format");

        let mut encoder = pool.begin().expect("a recording encoder");
        assert!(encoder.is_recording());
        assert_ne!(encoder.command_buffer(), vk::CommandBuffer::null());

        // A transient's first use: undefined to a vertex read, then to a copy
        // destination. The whole-buffer form and the explicit byte range are both
        // exercised.
        encoder
            .transition_buffer(
                buffer_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::VertexRead,
            )
            .expect("undefined to a vertex read");
        encoder
            .transition_buffer(
                buffer_handle,
                BufferRange::Bytes {
                    offset: 0,
                    size: 256,
                },
                ResourceAccessState::VertexRead,
                ResourceAccessState::CopyDestination,
            )
            .expect("a vertex read to a copy destination");
        // The same state on both sides is still a barrier, and the recorder has no
        // way to treat it as anything else.
        encoder
            .transition_buffer(
                buffer_handle,
                BufferRange::Whole,
                ResourceAccessState::CopyDestination,
                ResourceAccessState::CopyDestination,
            )
            .expect("a same-state transition records");

        // A colour target's first use and its export to a copy source, over the
        // whole image and then over one explicit subresource.
        encoder
            .transition_image(
                image_handle,
                image_format,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .expect("undefined to a colour attachment");
        encoder
            .transition_image(
                image_handle,
                image_format,
                TextureRange::Subresources {
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    aspect: fluxel_rendergraph::TextureAspect::Color,
                },
                ResourceAccessState::ColorAttachmentWrite,
                ResourceAccessState::CopySource,
            )
            .expect("a colour attachment to a copy source");

        encoder.end().expect("recording ends");
        assert!(!encoder.is_recording());
    }

    #[test]
    fn an_unsupported_state_is_refused_without_poisoning_the_encoder() {
        // The contract says the executor still ends an encoder after a recording
        // error, so a refused transition must leave the recording usable. The
        // refusal is a value and the driver is never reached.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );
        let buffer = table
            .create_buffer(
                64,
                declared_buffer(&[BufferUsageKind::Vertex]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local buffer");
        let buffer_handle = table.buffer_handle(buffer).expect("a live buffer");

        let mut encoder = pool.begin().expect("a recording encoder");
        // A colour attachment is a texture state, so it is not a buffer access.
        assert_eq!(
            encoder.transition_buffer(
                buffer_handle,
                BufferRange::Whole,
                ResourceAccessState::ColorAttachmentWrite,
                ResourceAccessState::CopyDestination,
            ),
            Err(RecordError::UnsupportedState(
                ResourceAccessState::ColorAttachmentWrite
            ))
        );
        // A vertex read is a buffer state, so it is not a texture access.
        assert_eq!(
            encoder.transition_image(
                vk::Image::null(),
                vk::Format::R8G8B8A8_UNORM,
                TextureRange::Whole,
                ResourceAccessState::VertexRead,
                ResourceAccessState::CopySource,
            ),
            Err(RecordError::UnsupportedState(ResourceAccessState::VertexRead))
        );
        // A barrier may not name a zero-count subresource range.
        assert_eq!(
            encoder.transition_image(
                vk::Image::null(),
                vk::Format::R8G8B8A8_UNORM,
                TextureRange::Subresources {
                    base_mip_level: 0,
                    mip_level_count: 0,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    aspect: fluxel_rendergraph::TextureAspect::Color,
                },
                ResourceAccessState::Undefined,
                ResourceAccessState::CopySource,
            ),
            Err(RecordError::UnsupportedRange)
        );

        // The already-recorded barrier and the still-open recording are unaffected.
        encoder
            .transition_buffer(
                buffer_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::VertexRead,
            )
            .expect("a refused transition does not poison the encoder");
        encoder.end().expect("the encoder still ends");
    }

    #[test]
    fn an_ended_encoder_refuses_further_recording() {        let Some((opened, pool)) = pool() else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );
        let buffer = table
            .create_buffer(
                64,
                declared_buffer(&[BufferUsageKind::Vertex]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local buffer");
        let buffer_handle = table.buffer_handle(buffer).expect("a live buffer");

        let mut encoder = pool.begin().expect("a recording encoder");
        encoder.end().expect("recording ends");
        assert_eq!(
            encoder.end(),
            Err(RecordError::NotRecording),
            "ending twice is a caller mistake, not an idempotent request"
        );
        assert_eq!(
            encoder.transition_buffer(
                buffer_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::VertexRead,
            ),
            Err(RecordError::NotRecording)
        );
    }

    #[test]
    fn the_copy_alignment_is_the_portable_alignment() {
        // The safe layer enforces `COPY_BUFFER_ALIGNMENT` before the native
        // boundary; this backend repeats the number at the boundary itself. The
        // two live in different modules, so a change to either one would silently
        // move the other unless something states that they are the same fact.
        assert_eq!(
            copy::COPY_BUFFER_ALIGNMENT,
            wgpu_types::COPY_BUFFER_ALIGNMENT
        );
    }

    #[test]
    fn a_real_encoder_records_a_real_buffer_and_image_copy() {
        // Step 8 against the real driver: one recording that copies a buffer to a
        // buffer and one image of a texture to another image, each through the
        // table's handles and the lowered region. Skips where no adapter exists.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );

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

        let described = TextureDesc {
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
        };
        let source_texture = table
            .create_texture(
                described,
                declared_texture(&[TextureUsageKind::CopySource]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy source texture");
        let destination_texture = table
            .create_texture(
                described,
                declared_texture(&[TextureUsageKind::CopyDestination]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy destination texture");

        let (source_handle, source_size) = (
            table.buffer_handle(source).expect("a live buffer"),
            table.buffer_size(source).expect("a created size"),
        );
        let (destination_handle, destination_size) = (
            table.buffer_handle(destination).expect("a live buffer"),
            table.buffer_size(destination).expect("a created size"),
        );
        let source_image = table.texture_image(source_texture).expect("a live image");
        let destination_image = table
            .texture_image(destination_texture)
            .expect("a live image");

        let mut encoder = pool.begin().expect("a recording encoder");
        // The transactions the copy commands require: a source is read through the
        // transfer stage and a destination is written through it, and an image is
        // in a transfer layout while that happens.
        encoder
            .transition_buffer(
                source_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopySource,
            )
            .expect("the source is readable");
        encoder
            .transition_buffer(
                destination_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopyDestination,
            )
            .expect("the destination is writable");
        encoder
            .transition_image(
                source_image,
                vk::Format::R8G8B8A8_UNORM,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopySource,
            )
            .expect("the source image is readable");
        encoder
            .transition_image(
                destination_image,
                vk::Format::R8G8B8A8_UNORM,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopyDestination,
            )
            .expect("the destination image is writable");

        encoder
            .copy_buffer(
                source_handle,
                source_size,
                destination_handle,
                destination_size,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 64,
                    size: 128,
                },
            )
            .expect("a real buffer copy records");
        encoder
            .copy_texture(
                source_image,
                &described,
                destination_image,
                &described,
                TextureCopyRegion {
                    source_origin: [0, 0, 0],
                    destination_origin: [4, 2, 0],
                    extent: [8, 4, 1],
                    source_mip_level: 0,
                    destination_mip_level: 0,
                },
            )
            .expect("a real image copy records");

        encoder.end().expect("recording ends");
        assert!(!encoder.is_recording());
    }

    #[test]
    fn a_refused_copy_is_a_value_and_the_encoder_stays_usable() {
        // Every copy refusal happens before the driver is reached, and the contract
        // says the executor still ends an encoder after a recording error. The
        // table's handles are what make the stale-id cases answerable at all.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );
        let buffer = table
            .create_buffer(
                64,
                declared_buffer(&[BufferUsageKind::CopySource]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local buffer");
        let handle = table.buffer_handle(buffer).expect("a live buffer");
        let size = table.buffer_size(buffer).expect("a created size");

        let described = TextureDesc {
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
        };
        let texture = table
            .create_texture(
                described,
                declared_texture(&[TextureUsageKind::CopySource]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local texture");
        let image = table.texture_image(texture).expect("a live image");

        let mut encoder = pool.begin().expect("a recording encoder");

        // A misaligned buffer region, an out-of-bounds texture box and a differing
        // texture format are three different mistakes, and each keeps its own
        // sentence.
        assert_eq!(
            encoder.copy_buffer(
                handle,
                size,
                handle,
                size,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 2,
                    size: 4,
                },
            ),
            Err(RecordError::Region(copy::CopyRegionError::Misaligned))
        );
        assert_eq!(
            encoder.copy_texture(
                image,
                &described,
                image,
                &described,
                TextureCopyRegion {
                    source_origin: [7, 0, 0],
                    destination_origin: [0, 0, 0],
                    extent: [2, 1, 1],
                    source_mip_level: 0,
                    destination_mip_level: 0,
                },
            ),
            Err(RecordError::Region(copy::CopyRegionError::OutOfBounds))
        );
        let mut other = described;
        other.format = TextureFormat::Bgra8Unorm;
        assert_eq!(
            encoder.copy_texture(
                image,
                &described,
                image,
                &other,
                TextureCopyRegion {
                    source_origin: [0, 0, 0],
                    destination_origin: [0, 0, 0],
                    extent: [1, 1, 1],
                    source_mip_level: 0,
                    destination_mip_level: 0,
                },
            ),
            Err(RecordError::Region(copy::CopyRegionError::FormatMismatch))
        );

        // The already-recorded state and the still-open recording are unaffected.
        encoder
            .copy_buffer(
                handle,
                size,
                handle,
                size,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 64,
                },
            )
            .expect("a refused copy does not poison the encoder");
        encoder.end().expect("the encoder still ends");
    }

    #[test]
    fn a_stale_id_is_refused_by_the_table_rather_than_reaching_the_driver() {
        // The caller resolves an id to a handle before recording, so a destroyed
        // resource answers `None` and a copy call with it is never reached. This is
        // the shape that keeps a stale generation from becoming a driver call, and
        // it is asserted here rather than only in the table's own tests because the
        // copy path is a caller that depends on it.
        let Some((opened, _pool)) = pool() else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );
        let buffer = table
            .create_buffer(
                64,
                declared_buffer(&[BufferUsageKind::CopySource]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local buffer");
        table.destroy_buffer(buffer).expect("the buffer is released");
        assert_eq!(
            table.buffer_handle(buffer),
            None,
            "a destroyed id resolves to no handle"
        );
        assert_eq!(table.buffer_size(buffer), None);
    }
}
