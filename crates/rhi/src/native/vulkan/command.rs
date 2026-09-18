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
use fluxel_rendergraph::{BufferRange, ResourceAccessState, TextureRange};

use super::{barrier, format};

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
    /// The state names an access this backend has not been taught how to order, or
    /// one that belongs to the other resource kind.
    UnsupportedState(ResourceAccessState),
    /// The subresource range is not one a barrier may name.
    UnsupportedRange,
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

    /// Ends recording.
    ///
    /// A second call is refused rather than treated as idempotent: "end an encoder
    /// that is not recording" is a caller mistake, and silently accepting it would
    /// hide the mistake until submission.
    pub(crate) fn end(&mut self) -> Result<(), RecordError> {
        if !self.recording {
            return Err(RecordError::NotRecording);
        }
        // SAFETY: the command buffer is recording and belongs to this encoder.
        unsafe { self.device.end_command_buffer(self.command_buffer) }.map_err(RecordError::End)?;
        self.recording = false;
        Ok(())
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
    fn an_ended_encoder_refuses_further_recording() {
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
}
