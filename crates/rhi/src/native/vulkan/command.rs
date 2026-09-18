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
//! # One pass slot, two brackets
//!
//! [`Encoder::begin_raster`] records `vkCmdBeginRenderPass` for a
//! [`Framebuffer`](super::framebuffer::Framebuffer), which owns the render pass and
//! the clear value its attachment states, and [`Encoder::end_raster`] records
//! `vkCmdEndRenderPass`. [`Encoder::begin_compute`] and [`Encoder::end_compute`]
//! bracket the same slot for a compute recording; `Vulkan` has no compute-pass
//! command, so that bracket is this layer's own and its whole effect is which verbs
//! the encoder admits -- and, for `set_bindings`, which bind point it names. The two
//! brackets share one slot, which is the GL family's model: a pass is open or it is
//! not, and a second begin of either kind answers [`RecordError::PassAlreadyOpen`].
//!
//! The state is what refuses the commands `Vulkan` does not allow inside a pass: a
//! barrier and a copy answer [`RecordError::PassOpen`], and ending the recording with
//! either bracket open is the same refusal, because `vkEndCommandBuffer` requires no
//! active render pass and the graph's own order records transitions, then the pass,
//! then transitions. A bracket of the wrong kind answers [`RecordError::NoPass`], and
//! `end_compute` with no compute pass open answers [`RecordError::NoComputePass`] --
//! the one close that names its kind, because a close wants a boundary rather than a
//! pass to record into.
//!
//! # The raster and compute verbs belong to their own open pass
//!
//! The pipeline, the bindings, the vertex and index buffers, the dynamic viewport and
//! scissor and the two draws are recorded only while a raster pass is open, and
//! answer [`RecordError::NoPass`] otherwise; the compute pipeline, the compute
//! bindings and the dispatch are recorded only while a compute pass is open, for the
//! same reason. `Vulkan` permits most of them outside a render pass, but it treats
//! them as command-buffer state the *next* pass inherits, and this backend's
//! execution model has no such state to inherit: a pass begins with exactly the state
//! its own commands set. Refusing is the fail-closed direction, and it is the same
//! sentence the bracket already uses for an `end_raster` with nothing open.
//!
//! The dispatch's workgroup counts are lowered by [`compute`] rather than spelled
//! here, so the one rule that command owns -- no zero dimension -- stays provable
//! without a device.
//!
//! Every raster value is lowered by [`draw`] rather than spelled here: the viewport's
//! Y flip, the scissor's signed offset, the index type and the count computed from a
//! half-open range all come from that module, so the checks that keep the driver from
//! seeing an invalid value live in one place. A refused value is a
//! [`RecordError::Draw`] and, like every other refusal here, leaves the recording
//! usable.
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

use std::ops::Range;

use ash::vk;
use fluxel_rendergraph::{
    BufferCopyRegion, BufferRange, IndexFormat, ResourceAccessState, ScissorRect, TextureCopyRegion,
    TextureRange, Viewport,
};

use super::pipeline::{ComputePipeline, RasterPipeline};
use super::{
    barrier, bind_group::BindGroup, compute, copy, draw, format, framebuffer::Framebuffer,
};

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
    /// A pass is already open, so a second begin is a caller mistake rather than a
    /// nested pass.
    ///
    /// Kind-neutral on purpose: a raster and a compute bracket share one slot (see
    /// the module docs), so naming one kind would make the refusal wrong for the
    /// caller that needed the other.
    PassAlreadyOpen,
    /// No pass of the kind this verb needs is open, so there is none to end and no
    /// pass for the verb to belong to.
    ///
    /// Refused rather than accepted idempotently for the same reason a second `end`
    /// is: silently accepting it would hide the caller's mistake until submission.
    NoPass,
    /// `end_compute` was called with no compute pass open.
    ///
    /// A sentence of its own rather than [`Self::NoPass`], and the reason is narrow:
    /// [`Self::NoPass`] must stay kind-neutral because shared verbs reach it, while
    /// this one is reached by the compute close alone. What it adds is what the
    /// caller is missing -- a close wants a boundary to close, not a pass to record
    /// into -- so a caller that closed twice or never opened can tell which.
    NoComputePass,
    /// A pass is still open, so this command may not be recorded here.
    ///
    /// A barrier, a copy and the end of a recording are all illegal inside a render
    /// pass, and the graph's own order records transitions outside every bracket, so
    /// the encoder refuses them while either pass is open rather than recording a
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
    /// The raster state or range is not one the driver would accept, and the reason
    /// is carried rather than flattened: a viewport whose numbers are wrong, a
    /// scissor whose shape is wrong and an inverted range are different mistakes.
    Draw(draw::DrawError),
    /// The dispatch's workgroup counts are not ones this backend records, and the
    /// reason carries the whole triple rather than only the offending axis.
    Dispatch(compute::DispatchError),
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
            pass: None,
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
    /// The pass bracket open on this recording, if any.
    ///
    /// One slot holds either kind rather than one flag each, because `Vulkan` has no
    /// nested passes and the GL family's adapter already models the slot this way.
    /// It is state rather than a comment because the commands `Vulkan` forbids inside
    /// a render pass are refused from it, because it decides which bind point
    /// `set_bindings` names, and because the matching `end_*` is what clears it.
    pass: Option<PassKind>,
}

/// Which pass bracket a recording has open.
///
/// Private, and deliberately not shared with `common`: which bracket is open is a
/// recording-context fact, not capability vocabulary, and section 21 of the plan
/// keeps the two from being frozen into one object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PassKind {
    /// A raster pass begun by [`Encoder::begin_raster`].
    Raster,
    /// A compute pass begun by [`Encoder::begin_compute`].
    Compute,
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

    /// The guard every pass verb shares: the encoder is recording `kind`'s pass.
    ///
    /// The two refusals stay distinct because they are different sentences -- an
    /// ended encoder and an encoder whose open pass is the wrong kind or absent --
    /// and the module docs state why a raster or compute verb belongs to an open pass
    /// at all. It takes the kind rather than reading one field, so a verb cannot be
    /// admitted into the other family's bracket by accident.
    fn check_pass(&self, kind: PassKind) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass != Some(kind) {
            return Err(RecordError::NoPass);
        }
        Ok(())
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
        if self.pass.is_some() {
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
        if self.pass.is_some() {
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
        if self.pass.is_some() {
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
        if self.pass.is_some() {
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
    /// of a single bracket should have caught. The refusal is
    /// [`RecordError::PassAlreadyOpen`], which is kind-neutral because a compute
    /// bracket's begin answers it too.
    pub(crate) fn begin_raster(&mut self, framebuffer: &Framebuffer) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass.is_some() {
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
        self.pass = Some(PassKind::Raster);
        Ok(())
    }

    /// Records the end of the raster pass this encoder began.
    ///
    /// An `end_raster` with no raster pass open is refused rather than treated as
    /// idempotent, for the same reason a second `end` is: the caller's mistake would
    /// otherwise be hidden until the recording is submitted. A compute bracket open
    /// instead answers the same sentence, because there is no raster pass to close.
    pub(crate) fn end_raster(&mut self) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass != Some(PassKind::Raster) {
            return Err(RecordError::NoPass);
        }
        // SAFETY: the command buffer is recording with a raster pass open, which is
        // the only state in which `vkCmdEndRenderPass` is legal.
        unsafe { self.device.cmd_end_render_pass(self.command_buffer) };
        self.pass = None;
        Ok(())
    }

    /// Opens the compute pass the compute verbs record into.
    ///
    /// `Vulkan` has no compute-pass command: a dispatch is legal outside every render
    /// pass, so this bracket creates no driver state. It is still state here, because
    /// the compute verbs belong to *their* open pass exactly as the raster verbs do,
    /// and because the two brackets sharing one slot is what keeps a raster pass from
    /// being open at the same time. The alternative -- admitting a dispatch with no
    /// bracket -- would make the family's `begin_compute`/`end_compute` vocabulary a
    /// pair of no-ops that promise a boundary nothing enforces.
    ///
    /// A second begin of either kind answers [`RecordError::PassAlreadyOpen`].
    pub(crate) fn begin_compute(&mut self) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass.is_some() {
            return Err(RecordError::PassAlreadyOpen);
        }
        self.pass = Some(PassKind::Compute);
        Ok(())
    }

    /// Closes the compute pass this encoder began.
    ///
    /// A close with no compute pass open answers [`RecordError::NoComputePass`], the
    /// one bracket close that names its kind, because this verb alone reaches it.
    pub(crate) fn end_compute(&mut self) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass != Some(PassKind::Compute) {
            return Err(RecordError::NoComputePass);
        }
        self.pass = None;
        Ok(())
    }

    /// Binds a raster pipeline for the draws recorded in the open pass.
    ///
    /// The pipeline owns the layout it was created over, so the layout stays alive
    /// for as long as this binding can be used; the caller keeps the pipeline alive
    /// for the recording, exactly as it keeps a framebuffer's view alive.
    pub(crate) fn set_raster_pipeline(
        &mut self,
        pipeline: &RasterPipeline,
    ) -> Result<(), RecordError> {
        self.check_pass(PassKind::Raster)?;
        // SAFETY: the command buffer is recording inside a render pass, and the
        // pipeline is a live handle this device created, kept alive by the caller.
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle(),
            );
        }
        Ok(())
    }

    /// Binds a descriptor set for the draws recorded in the open raster pass.
    ///
    /// The group carries the `VkPipelineLayout` its set was allocated against, so the
    /// binding and the layout cannot be told different things and `Vulkan`'s
    /// compatibility rule is satisfied by construction. The caller keeps the group and
    /// the layout's owner alive for the recording, exactly as it keeps a pipeline and
    /// a vertex buffer alive.
    ///
    /// The compute family's binding is [`Self::set_compute_bindings`]: the bind point
    /// is a fact about the open bracket, so the two families keep one method each
    /// rather than a parameter a caller could set against its own pass.
    pub(crate) fn set_bindings(&mut self, group: &BindGroup) -> Result<(), RecordError> {
        self.check_pass(PassKind::Raster)?;
        self.bind_descriptor_sets(group, vk::PipelineBindPoint::GRAPHICS);
        Ok(())
    }

    /// Binds a descriptor set for the dispatches recorded in the open compute pass.
    ///
    /// The same shared body as [`Self::set_bindings`]; only the bind point differs,
    /// and it comes from the bracket the guard just proved rather than from the
    /// caller. `Vulkan` requires a compute pipeline's set to be bound through
    /// `COMPUTE`, so recording a compute binding through the graphics point would
    /// produce a dispatch that reads nothing.
    pub(crate) fn set_compute_bindings(&mut self, group: &BindGroup) -> Result<(), RecordError> {
        self.check_pass(PassKind::Compute)?;
        self.bind_descriptor_sets(group, vk::PipelineBindPoint::COMPUTE);
        Ok(())
    }

    /// Records one `vkCmdBindDescriptorSets` at `bind_point`.
    ///
    /// Private because the bind point is what distinguishes the two families' verbs:
    /// a public parameter would let a caller name the other family's point, which is
    /// exactly what the one-handle-type-per-family rule exists to make impossible.
    fn bind_descriptor_sets(&mut self, group: &BindGroup, bind_point: vk::PipelineBindPoint) {
        let sets = [group.set()];
        // SAFETY: the caller's guard proved a pass of the bind point's own kind is
        // open; both handles are live objects this device created and the caller keeps
        // them alive; and no bind-time dynamic offsets exist, because a bind group
        // cannot be created for a layout that declares one (`bind_group::validate`
        // refuses it).
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                bind_point,
                group.pipeline_layout(),
                group.set_index(),
                &sets,
                &[],
            );
        }
    }

    /// Binds a compute pipeline for the dispatches recorded in the open compute pass.
    ///
    /// The pipeline owns the layout it was created over, so the layout stays alive
    /// for as long as this binding can be used; the caller keeps the pipeline alive
    /// for the recording, exactly as it keeps a raster pipeline alive.
    pub(crate) fn set_compute_pipeline(
        &mut self,
        pipeline: &ComputePipeline,
    ) -> Result<(), RecordError> {
        self.check_pass(PassKind::Compute)?;
        // SAFETY: the command buffer is recording inside a compute pass, and the
        // pipeline is a live handle this device created, kept alive by the caller.
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle(),
            );
        }
        Ok(())
    }

    /// Records one `vkCmdDispatch` over `groups` workgroups per axis.
    ///
    /// The counts are lowered by [`compute::dispatch_groups`], which refuses a zero
    /// dimension before the driver is reached; `Vulkan` would accept it as a legal
    /// no-op, and a caller's mistake would then look like a dispatch that did nothing.
    pub(crate) fn dispatch(&mut self, groups: [u32; 3]) -> Result<(), RecordError> {
        self.check_pass(PassKind::Compute)?;
        let lowered = compute::dispatch_groups(groups).map_err(RecordError::Dispatch)?;
        // SAFETY: the command buffer is recording inside a compute pass with a
        // pipeline bound by the caller's recorded order, and the three counts are the
        // driver's own arguments.
        unsafe {
            self.device.cmd_dispatch(
                self.command_buffer,
                lowered[0],
                lowered[1],
                lowered[2],
            );
        }
        Ok(())
    }

    /// Records the dynamic viewport the open pass draws through.
    ///
    /// The value is lowered by [`draw::viewport`], which keeps the borrowed path's
    /// Y flip and refuses a viewport the driver would reject.
    pub(crate) fn set_viewport(&mut self, viewport: Viewport) -> Result<(), RecordError> {
        self.check_pass(PassKind::Raster)?;
        let lowered = draw::viewport(viewport).map_err(RecordError::Draw)?;
        // SAFETY: the command buffer is recording inside a render pass, and the
        // slice is a local that outlives the call.
        unsafe {
            self.device
                .cmd_set_viewport(self.command_buffer, 0, &[lowered]);
        }
        Ok(())
    }

    /// Records the dynamic scissor rectangle the open pass draws through.
    pub(crate) fn set_scissor(&mut self, scissor: ScissorRect) -> Result<(), RecordError> {
        self.check_pass(PassKind::Raster)?;
        let lowered = draw::scissor(scissor).map_err(RecordError::Draw)?;
        // SAFETY: the command buffer is recording inside a render pass, and the
        // slice is a local that outlives the call.
        unsafe {
            self.device
                .cmd_set_scissor(self.command_buffer, 0, &[lowered]);
        }
        Ok(())
    }

    /// Binds the vertex buffer that occupies `slot`.
    ///
    /// `buffer` is a handle the caller resolved from the resource table, and the
    /// table keeps it alive for the recording -- the same ownership rule the copy
    /// verbs state. The stride and the attribute formats are the pipeline's, so
    /// nothing about the stream's shape is repeated here.
    pub(crate) fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: vk::Buffer,
        offset: u64,
    ) -> Result<(), RecordError> {
        self.check_pass(PassKind::Raster)?;
        // SAFETY: the command buffer is recording inside a render pass, the buffer
        // is a live handle of this device, and both slices are locals that outlive
        // the call and have the same length.
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.command_buffer, slot, &[buffer], &[offset]);
        }
        Ok(())
    }

    /// Binds the index buffer subsequent indexed draws read.
    ///
    /// The portable format is lowered by [`draw::index_type`], which is exhaustive
    /// over this workspace's closed format enum.
    pub(crate) fn set_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: u64,
        format: IndexFormat,
    ) -> Result<(), RecordError> {
        self.check_pass(PassKind::Raster)?;
        // SAFETY: the command buffer is recording inside a render pass, and the
        // buffer is a live handle this device created, kept alive by the caller.
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.command_buffer, buffer, offset, draw::index_type(format));
        }
        Ok(())
    }

    /// Records a non-indexed draw over `vertices`, `instance_count` times.
    ///
    /// The first instance is fixed at zero, because the family's verbs take an
    /// instance *count* rather than a range (plan section 20.1): a non-zero first
    /// instance is the `FirstInstance` capability, which this backend does not
    /// record.
    pub(crate) fn draw(
        &mut self,
        vertices: Range<u32>,
        instance_count: u32,
    ) -> Result<(), RecordError> {
        self.check_pass(PassKind::Raster)?;
        let (first_vertex, vertex_count) = draw::range(vertices).map_err(RecordError::Draw)?;
        // SAFETY: the command buffer is recording inside a render pass with a
        // pipeline bound by the caller's recorded order.
        unsafe {
            self.device.cmd_draw(
                self.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                0,
            );
        }
        Ok(())
    }

    /// Records an indexed draw over `indices`, `instance_count` times.
    ///
    /// The base vertex is fixed at zero for [`draw`](Self::draw)'s reason: it is the
    /// `BaseVertex` capability, which this backend does not record.
    pub(crate) fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        instance_count: u32,
    ) -> Result<(), RecordError> {
        self.check_pass(PassKind::Raster)?;
        let (first_index, index_count) = draw::range(indices).map_err(RecordError::Draw)?;
        // SAFETY: the command buffer is recording inside a render pass with a
        // pipeline and an index buffer bound by the caller's recorded order.
        unsafe {
            self.device.cmd_draw_indexed(
                self.command_buffer,
                index_count,
                instance_count,
                first_index,
                0,
                0,
            );
        }
        Ok(())
    }

    /// Ends recording.
    ///
    /// A second call is refused rather than treated as idempotent: "end an encoder
    /// that is not recording" is a caller mistake, and silently accepting it would
    /// hide the mistake until submission. A recording with either bracket still open
    /// is refused for the same reason: `vkEndCommandBuffer` requires no active render
    /// pass, the graph's own order closes its bracket before the recording ends, and
    /// the caller's matching `end_raster` / `end_compute` is what closes it.
    pub(crate) fn end(&mut self) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        if self.pass.is_some() {
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
    use core::time::Duration;

    use super::*;
    use crate::Validation;
    use crate::native::vulkan::pipeline::{create_compute, create_layout, create_raster};
    use crate::native::vulkan::resource::ResourceTable;
    use crate::native::vulkan::shader::MINIMAL_COMPUTE_SPIRV;
    use crate::native::vulkan::test_support::{
        colour_only_state, colour_target_pass, position_stream, raster_shaders,
    };
    use crate::native::vulkan::{allocator::GpuAllocator, memory, open, submission};
    use fluxel_rendergraph::{
        BufferUsage, BufferUsageKind, CompletionStatus, Extent3d, TextureDesc, TextureDimension,
        TextureFormat, TextureUsage, TextureUsageKind,
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

    #[test]
    fn a_real_encoder_records_a_real_raster_pass_with_both_draws() {        // Step 12's state and draw verbs against the real driver: a real pipeline, the
        // dynamic viewport and scissor, a vertex buffer and an index buffer, and both
        // draws, recorded inside a real pass and submitted to completion. Skips where
        // no adapter exists.
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

        let (framebuffer, image, mapped) =
            colour_target_pass(&opened, &mut table, &memory_types);

        // The pipeline the draws record through, over the shared minimal recipe whose
        // vertex stream is exactly `position_stream`.
        let layout = create_layout(opened.device.device(), Vec::new()).expect("an empty layout");
        let pipeline = create_raster(
            opened.device.device(),
            layout,
            &raster_shaders(),
            &position_stream(),
            &colour_only_state(),
        )
        .expect("a raster pipeline over the retained recipe");

        // The two buffers the draws read. Their contents are never written, which is
        // deliberate: what this test records is the draw's shape and its execution,
        // not its pixels, so a device-local buffer the driver has not been handed any
        // data for is the honest fixture.
        let vertices = table
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::Vertex]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");
        let indices = table
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::Index]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local index buffer");
        let vertex_handle = table.buffer_handle(vertices).expect("a live vertex buffer");
        let index_handle = table.buffer_handle(indices).expect("a live index buffer");

        let mut encoder = pool.begin().expect("a recording encoder");
        // The graph's own order: the transition the pass's initial layout needs, then
        // the bracket, then the state and the draws, then the close.
        encoder
            .transition_image(
                image,
                mapped,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .expect("undefined to a colour attachment");
        encoder.begin_raster(&framebuffer).expect("the pass opens");
        encoder
            .set_raster_pipeline(&pipeline)
            .expect("the pipeline binds");
        encoder
            .set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 8.0,
                min_depth: 0.0,
                max_depth: 1.0,
            })
            .expect("the viewport sets");
        encoder
            .set_scissor(ScissorRect {
                x: 0,
                y: 0,
                width: 16,
                height: 8,
            })
            .expect("the scissor sets");
        encoder
            .set_vertex_buffer(0, vertex_handle, 0)
            .expect("the vertex buffer binds");
        encoder
            .set_index_buffer(index_handle, 0, IndexFormat::Uint16)
            .expect("the index buffer binds");
        encoder.draw(0..3, 1).expect("a non-indexed draw records");
        encoder
            .draw_indexed(0..3, 1)
            .expect("an indexed draw records");
        encoder.end_raster().expect("the pass closes");

        let finished = encoder.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "a recorded raster pass with both draws runs to completion"
        );
        // The framebuffer, the pipeline and both buffers are still alive here, so the
        // terminal submission above is the one that released the recording.
        assert!(submission.is_terminal());
    }

    #[test]
    fn the_raster_verbs_refuse_outside_a_pass_and_a_bad_value_without_poisoning_it() {
        // Two different sentences: a verb with no pass open is `NoPass`, and a value
        // the lowering refuses is `Draw`. Both are values returned before the driver,
        // and a refused call leaves the recording usable, which the contract requires
        // because the executor still ends an encoder after a recording error.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let mut encoder = pool.begin().expect("a recording encoder");

        // No pass is open yet, so every raster verb answers with the same sentence.
        assert_eq!(
            encoder.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 8.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }),
            Err(RecordError::NoPass)
        );
        assert_eq!(
            encoder.set_scissor(ScissorRect {
                x: 0,
                y: 0,
                width: 16,
                height: 8,
            }),
            Err(RecordError::NoPass)
        );
        assert_eq!(encoder.draw(0..3, 1), Err(RecordError::NoPass));
        assert_eq!(encoder.draw_indexed(0..3, 1), Err(RecordError::NoPass));

        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );
        let (framebuffer, image, mapped) =
            colour_target_pass(&opened, &mut table, &memory_types);
        encoder
            .transition_image(
                image,
                mapped,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .expect("undefined to a colour attachment");
        encoder.begin_raster(&framebuffer).expect("the pass opens");

        // With a pass open the guard lets the call through and the lowering refuses
        // the value, each as its own sentence.
        assert_eq!(
            encoder.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 0.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }),
            Err(RecordError::Draw(draw::DrawError::Viewport))
        );
        assert_eq!(
            encoder.set_scissor(ScissorRect {
                x: 0,
                y: 0,
                width: 0,
                height: 8,
            }),
            Err(RecordError::Draw(draw::DrawError::Scissor))
        );
        // The ends are locals rather than literals because a literal reversed range is
        // what `clippy::reversed_empty_ranges` forbids; the draw only lowers it.
        let (start, end) = (3u32, 2u32);
        assert_eq!(
            encoder.draw(start..end, 1),
            Err(RecordError::Draw(draw::DrawError::Range))
        );

        // A refused value does not poison the pass or the recording.
        encoder
            .set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 8.0,
                min_depth: 0.0,
                max_depth: 1.0,
            })
            .expect("the encoder is usable after the refusals");
        encoder.end_raster().expect("the pass closes");
        encoder.end().expect("the recording ends");
    }

    #[test]
    fn a_real_encoder_records_a_real_compute_dispatch() {
        // W2's compute family against the real driver: one bracket, one real compute
        // pipeline over an empty layout and one real `vkCmdDispatch`, submitted and
        // reported complete. Skips where no adapter exists.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let pipeline = create_compute(
            opened.device.device(),
            create_layout(opened.device.device(), Vec::new()).expect("an empty layout"),
            &MINIMAL_COMPUTE_SPIRV,
            c"main",
        )
        .expect("a compute pipeline over the retained kernel");

        let mut encoder = pool.begin().expect("a recording encoder");
        encoder.begin_compute().expect("the compute pass opens");
        encoder
            .set_compute_pipeline(&pipeline)
            .expect("the pipeline binds");
        encoder.dispatch([1, 1, 1]).expect("a dispatch records");
        encoder.end_compute().expect("the compute pass closes");

        let finished = encoder.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "a recorded compute dispatch runs to completion"
        );
        assert!(submission.is_terminal());
    }

    #[test]
    fn the_compute_bracket_refuses_the_wrong_state_without_poisoning_the_recording() {
        // The compute verbs belong to the open compute pass and nowhere else, the two
        // closes name their own kind, and every refusal is a value. The contract says
        // the executor still ends a recording after a callback error, so a refused
        // dispatch must leave the pass usable.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let pipeline = create_compute(
            opened.device.device(),
            create_layout(opened.device.device(), Vec::new()).expect("an empty layout"),
            &MINIMAL_COMPUTE_SPIRV,
            c"main",
        )
        .expect("a compute pipeline over the retained kernel");
        let mut encoder = pool.begin().expect("a recording encoder");

        // No pass is open yet: the compute verbs answer `NoPass`, and the closes name
        // their own kind rather than borrowing it.
        assert_eq!(
            encoder.set_compute_pipeline(&pipeline),
            Err(RecordError::NoPass)
        );
        assert_eq!(encoder.dispatch([1, 1, 1]), Err(RecordError::NoPass));
        assert_eq!(encoder.end_compute(), Err(RecordError::NoComputePass));
        assert_eq!(encoder.end_raster(), Err(RecordError::NoPass));

        encoder.begin_compute().expect("the compute pass opens");
        // A second begin of either kind is the one kind-neutral sentence.
        assert_eq!(encoder.begin_compute(), Err(RecordError::PassAlreadyOpen));
        // A raster verb belongs to a raster pass, so it is refused inside this one.
        assert_eq!(encoder.draw(0..3, 1), Err(RecordError::NoPass));
        // A zero group dimension is the dispatch's own sentence rather than the legal
        // driver no-op it would otherwise become.
        assert_eq!(
            encoder.dispatch([1, 0, 1]),
            Err(RecordError::Dispatch(compute::DispatchError::ZeroGroups([
                1, 0, 1
            ])))
        );
        // A barrier and the end of the recording are outside every bracket, so the
        // open compute pass refuses them too; the handle is never read, which is why
        // a null one is the honest fixture here.
        assert_eq!(
            encoder.transition_buffer(
                vk::Buffer::null(),
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::VertexRead,
            ),
            Err(RecordError::PassOpen)
        );
        assert_eq!(encoder.end(), Err(RecordError::PassOpen));

        // A refused value does not poison the pass or the recording.
        encoder
            .set_compute_pipeline(&pipeline)
            .expect("the encoder is usable after the refusals");
        encoder.dispatch([64, 1, 1]).expect("a real dispatch records");
        encoder.end_compute().expect("the pass closes");
        encoder.end().expect("the recording ends");
        assert_eq!(encoder.end_compute(), Err(RecordError::NotRecording));
    }
}
