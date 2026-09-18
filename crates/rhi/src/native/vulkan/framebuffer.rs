//! Step 12's pass target: the `VkRenderPass` and `VkFramebuffer` a recorded raster
//! pass begins.
//!
//! # Why one type owns both
//!
//! `vkCmdBeginRenderPass` names a `VkRenderPass` and a `VkFramebuffer`, and `Vulkan`
//! fixes their dependency: the framebuffer refers to the render pass, and the render
//! pass refers to nothing but its attachments. One type therefore owns both, and the
//! order is field order rather than a comment -- the same rule
//! [`PipelineLayout`](super::pipeline::PipelineLayout) states for the descriptor set
//! layouts it names, and [`ResourceTable`](super::resource::ResourceTable) for its
//! buffer and image views.
//!
//! # The pass is described once, and the shared lowering is what keeps the two passes
//! # comparable
//!
//! `Vulkan` compares the render pass a raster pipeline was created against with the
//! one a command buffer begins, by the attachment formats, sample counts and
//! reference layouts. [`create`] therefore builds its render pass from
//! [`render_pass::PassAttachment::description`] and
//! [`render_pass::PassAttachment::color_reference`], which
//! [`pipeline::create_raster`](super::pipeline::create_raster) already uses for the
//! creation pass: only the contents operations differ, because those are what the
//! graph compiled. Neither pass restates a format or a layout.
//!
//! # What [`create`] refuses before the driver is reached
//!
//! - a **subresource range** -- the framebuffer is built from the texture's whole
//!   view, and a range the view does not address is a different resource than the
//!   graph named. The borrowed native raster path makes the same refusal;
//! - an **unsupported attachment shape** -- [`render_pass::framebuffer_extent`]
//!   refuses a layered, multi-mip, volumetric or zero-sized description, which is
//!   the shape a framebuffer attachment may not have;
//! - a portable **format** or **sample count** this backend has not been taught.
//!
//! # What it deliberately does not own
//!
//! The image view. It is a handle the [`ResourceTable`](super::resource::ResourceTable)
//! owns, and `Vulkan` requires it to outlive the framebuffer that names it, so the
//! caller keeps the table alive for the framebuffer's lifetime -- the same invariant
//! [`Encoder`](super::command::Encoder) states for its command pool.

use ash::vk;
use fluxel_rendergraph::{
    RasterColorAttachment, TextureDesc, TextureFormat, TextureRange,
};

use crate::common::base::resource::TextureId;

use super::render_pass::{self, PassAttachment};
use super::{format, pipeline};

/// Why a pass target could not be created.
///
/// The two "not one this backend builds" variants stay separate from the two driver
/// refusals, and from each other, because each names a different fix: a pass that
/// asked for a subresource range needs a view built for that range, while a layered
/// attachment needs a resource and a pass model this backend does not have.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FramebufferError {
    /// The pass selects a subresource range rather than the whole texture, and the
    /// framebuffer is built from the texture's whole view.
    UnsupportedRange,
    /// The attachment's texture is not one this backend builds a framebuffer from:
    /// layered, multi-mip, volumetric, of another dimension, or zero-sized.
    UnsupportedAttachment,
    /// The portable format has no `Vulkan` equivalent.
    UnsupportedFormat(TextureFormat),
    /// The sample count is not one of the counts `Vulkan` names.
    UnsupportedSampleCount(u32),
    /// The driver refused to create the render pass, so no framebuffer was asked for.
    RenderPass(vk::Result),
    /// The driver refused the framebuffer; the render pass it was created against was
    /// destroyed before the refusal.
    Framebuffer(vk::Result),
}

/// A raster pass target: the render pass one admitted colour attachment describes and
/// the framebuffer that binds the attachment's view to it.
pub(crate) struct Framebuffer {
    device: ash::Device,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    /// The attachment's size, which is also the render area a begin uses: `Vulkan`
    /// requires the framebuffer to be exactly as large as its attachments, so the
    /// two are one fact rather than two that could disagree.
    extent: vk::Extent2D,
    /// The lowered attachment, kept for the clear value a begin supplies.
    attachment: PassAttachment,
}

impl Framebuffer {
    /// Creates the render pass and framebuffer one admitted attachment describes.
    ///
    /// `desc` and `view` are that attachment's own texture facts, as the resource
    /// table reports them: the description names the format, sample count and extent
    /// the image was created with, and the view is the image view the framebuffer
    /// attaches. They are parameters rather than lookups because the table that owns
    /// them is the caller's, exactly as the copy recording takes its handles.
    ///
    /// Every refusal is a value returned before a driver handle exists, and the one
    /// failure that can happen *after* the render pass exists destroys it, so a
    /// refused target leaves nothing behind.
    pub(crate) fn create(
        device: &ash::Device,
        attachment: &RasterColorAttachment<'_, TextureId>,
        desc: &TextureDesc,
        view: vk::ImageView,
    ) -> Result<Self, FramebufferError> {
        if attachment.range != TextureRange::Whole {
            return Err(FramebufferError::UnsupportedRange);
        }
        let extent = render_pass::framebuffer_extent(desc)
            .ok_or(FramebufferError::UnsupportedAttachment)?;
        let format = format::image_format(desc.format)
            .ok_or(FramebufferError::UnsupportedFormat(desc.format))?;
        // Lowered through the one mapping that names `Vulkan`'s counts rather than
        // written as a constant, so the render pass states the count the image was
        // created with. The image lowering refuses every count but one, so this is
        // one sample in practice; the `None` branch is still a value because the
        // mapping is total over what the driver names.
        let samples = pipeline::sample_count(desc.sample_count)
            .ok_or(FramebufferError::UnsupportedSampleCount(desc.sample_count))?;
        let described = render_pass::describe(attachment, format, samples);

        let descriptions = [described.description()];
        let references = [described.color_reference()];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&references);
        let subpasses = [subpass];
        // No explicit dependencies: the implicit external-to-first-subpass dependency
        // of a single-subpass pass is the one the specification adds, and an explicit
        // one would state a synchronization the graph's own barriers own.
        let pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&descriptions)
            .subpasses(&subpasses);
        // SAFETY: the device is live and owned above this call; every slice the
        // create-info points at is a local that outlives the call, and no allocation
        // callbacks are supplied.
        let render_pass = match unsafe { device.create_render_pass(&pass_info, None) } {
            Ok(handle) => handle,
            Err(error) => return Err(FramebufferError::RenderPass(error)),
        };

        let views = [view];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            // One attachment, because the pass admits exactly one colour target at
            // index zero. `attachments` derives the count from the slice, so the two
            // cannot disagree.
            .attachments(&views)
            .width(extent.width)
            .height(extent.height)
            // One layer: a render pass without multiview attaches one layer of each
            // view, and `framebuffer_extent` refused a layered description already.
            .layers(1);
        // SAFETY: the render pass is a live handle this call just created; the view
        // was created by this device and the caller keeps it alive for this
        // framebuffer's lifetime; the create-info points at locals that outlive the
        // call; and no allocation callbacks are supplied.
        match unsafe { device.create_framebuffer(&framebuffer_info, None) } {
            Ok(framebuffer) => Ok(Self {
                device: device.clone(),
                render_pass,
                framebuffer,
                extent,
                attachment: described,
            }),
            Err(error) => {
                // Nothing refers to the render pass yet, so it is destroyed before
                // the refusal: a refused framebuffer leaves neither handle behind.
                // SAFETY: the handle was created just above and is destroyed once.
                unsafe { device.destroy_render_pass(render_pass, None) };
                Err(FramebufferError::Framebuffer(error))
            }
        }
    }

    /// Returns the render pass a recorded begin names.
    pub(crate) const fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    /// Returns the framebuffer a recorded begin binds.
    pub(crate) const fn handle(&self) -> vk::Framebuffer {
        self.framebuffer
    }

    /// Returns the size the framebuffer was created at, which is its render area.
    pub(crate) const fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Returns the clear value the attachment's own load operation states.
    ///
    /// One entry is always supplied, even where the load operation does not clear,
    /// because `VkRenderPassBeginInfo` indexes one entry per attachment and a value
    /// the driver ignores must not be an uninitialized one.
    pub(crate) fn clear_value(&self) -> vk::ClearValue {
        self.attachment.clear_value()
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of both handles, and the framebuffer refers
        // to the render pass, so the framebuffer is destroyed first. Every command
        // buffer that recorded a pass with this framebuffer has completed execution
        // or been freed -- a recorded pass is submittable exactly once and the
        // submission owns that recording -- and the view the framebuffer names is
        // released by the caller's table after this type.
        unsafe { self.device.destroy_framebuffer(self.framebuffer, None) };
        // SAFETY: the handle was created by this device and is destroyed once here,
        // after the framebuffer that referred to it.
        unsafe { self.device.destroy_render_pass(self.render_pass, None) };
    }
}

impl core::fmt::Debug for Framebuffer {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Framebuffer")
            .field("extent", &self.extent)
            .field("format", &self.attachment.format)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use fluxel_rendergraph::{
        AttachmentOps, CompletionStatus, Extent3d, LoadOp, ResourceAccessState, StoreOp, TextureDesc,
        TextureDimension, TextureFormat, TextureUsage, TextureUsageKind, WriteCoverage,
    };

    use super::*;
    use crate::Validation;
    use crate::native::vulkan::command::{CommandPool, RecordError};
    use crate::native::vulkan::open::OpenedVulkan;
    use crate::native::vulkan::resource::ResourceTable;
    use crate::native::vulkan::{allocator::GpuAllocator, memory, open, submission};

    /// Opens a headless device, its memory types and a real resource table, or
    /// returns `None` where no adapter exists. Having no GPU is not what these tests
    /// are about.
    fn fixture() -> Option<(OpenedVulkan, Vec<vk::MemoryType>, ResourceTable)> {
        let opened = open::open(Validation::Disabled, 0).ok()?;
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let table = ResourceTable::new(opened.device.device(), opened.device.stamp(), allocator);
        Some((opened, memory_types, table))
    }

    fn texture_desc(array_layers: u32) -> TextureDesc {
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 16,
                height: 8,
                depth: 1,
            },
            mip_levels: 1,
            array_layers,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        }
    }

    fn clear_ops() -> AttachmentOps<[f32; 4]> {
        AttachmentOps {
            load: LoadOp::Clear([0.25, 0.5, 0.75, 1.0]),
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Full,
        }
    }

    /// A real colour target the table owns, created with the one usage an attachment
    /// is legal with.
    fn colour_target(
        table: &mut ResourceTable,
        memory_types: &[vk::MemoryType],
        desc: TextureDesc,
    ) -> TextureId {
        table
            .create_texture(
                desc,
                TextureUsage::from_kinds([TextureUsageKind::ColorAttachment]),
                memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local colour target")
    }

    #[test]
    fn a_real_framebuffer_records_a_pass_the_driver_accepts() {
        // Step 12's owning half against the real driver: the render pass and
        // framebuffer one admitted attachment describes, a transition into the layout
        // the pass begins at, the bracket recorded on the real encoder, and one
        // submission the driver completes. Skips where no adapter exists.
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let id = colour_target(&mut table, &memory_types, texture_desc(1));
        let colors = [RasterColorAttachment {
            index: 0,
            texture: &id,
            range: TextureRange::Whole,
            operations: clear_ops(),
        }];
        let admitted = render_pass::admit(&colors, None).expect("one colour at index zero");
        let resolved = table.texture_desc(*admitted.texture).expect("a live texture");
        let view = table.texture_view(*admitted.texture).expect("a live view");
        let image = table.texture_image(*admitted.texture).expect("a live image");
        let mapped = format::image_format(resolved.format).expect("a mapped format");

        let framebuffer = Framebuffer::create(opened.device.device(), admitted, &resolved, view)
            .expect("a render pass and framebuffer for a real colour target");
        assert_ne!(framebuffer.render_pass(), vk::RenderPass::null());
        assert_ne!(framebuffer.handle(), vk::Framebuffer::null());
        assert_eq!(
            framebuffer.extent(),
            vk::Extent2D {
                width: 16,
                height: 8
            }
        );
        // SAFETY: `clear_value` writes the `color` member, so reading the same member
        // is the initialized variant of the union.
        let clear = unsafe { framebuffer.clear_value().color.float32 };
        assert_eq!(clear, [0.25, 0.5, 0.75, 1.0]);

        let pool = CommandPool::new(
            opened.device.device(),
            opened.device.selected_queue().family,
        )
        .expect("a command pool on an opened device");
        let mut encoder = pool.begin().expect("a recording encoder");
        // The graph's own order: the attachment is transitioned to the layout the
        // render pass begins at, which is what makes the recorded pass valid.
        encoder
            .transition_image(
                image,
                mapped,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .expect("undefined to a colour attachment");
        encoder
            .begin_raster(&framebuffer)
            .expect("the pass bracket opens");
        encoder
            .end_raster()
            .expect("the pass bracket closes");

        let finished = encoder.finish().expect("the recording ends");
        let mut submission =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "a recorded pass the driver accepts runs to completion"
        );
        // The framebuffer and the view it names are still alive here, so the
        // submission that referenced them is the terminal one above.
        assert!(submission.is_terminal());
    }

    #[test]
    fn a_layered_attachment_is_refused_before_the_driver_is_reached() {
        // The table can create a layered colour target, and the framebuffer refuses
        // it by name: a render pass without multiview attaches one layer, so a view
        // spanning four would name layers the pass never covers.
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let id = colour_target(&mut table, &memory_types, texture_desc(4));
        let colors = [RasterColorAttachment {
            index: 0,
            texture: &id,
            range: TextureRange::Whole,
            operations: clear_ops(),
        }];
        let admitted = render_pass::admit(&colors, None).expect("one colour at index zero");
        let resolved = table.texture_desc(*admitted.texture).expect("a live texture");
        let view = table.texture_view(*admitted.texture).expect("a live view");
        assert_eq!(
            Framebuffer::create(opened.device.device(), admitted, &resolved, view).err(),
            Some(FramebufferError::UnsupportedAttachment)
        );
    }

    #[test]
    fn a_subresource_range_is_refused_before_the_driver_is_reached() {
        // The framebuffer is built from the texture's whole view, so a pass that
        // selects a range is refused rather than attached to a view that does not
        // address it -- the same refusal the borrowed native raster path makes.
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let id = colour_target(&mut table, &memory_types, texture_desc(1));
        let colors = [RasterColorAttachment {
            index: 0,
            texture: &id,
            range: TextureRange::Subresources {
                base_mip_level: 0,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: fluxel_rendergraph::TextureAspect::Color,
            },
            operations: clear_ops(),
        }];
        let admitted = render_pass::admit(&colors, None).expect("one colour at index zero");
        let resolved = table.texture_desc(*admitted.texture).expect("a live texture");
        let view = table.texture_view(*admitted.texture).expect("a live view");
        assert_eq!(
            Framebuffer::create(opened.device.device(), admitted, &resolved, view).err(),
            Some(FramebufferError::UnsupportedRange)
        );
    }

    #[test]
    fn the_bracket_refuses_a_second_begin_an_unopened_end_and_commands_inside_a_pass() {
        // The pass bracket is explicit state, and each wrong transition is its own
        // sentence. A barrier or a copy inside a render pass is illegal in Vulkan, so
        // those verbs refuse while the pass is open rather than recording a command
        // the driver would reject.
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let id = colour_target(&mut table, &memory_types, texture_desc(1));
        let colors = [RasterColorAttachment {
            index: 0,
            texture: &id,
            range: TextureRange::Whole,
            operations: clear_ops(),
        }];
        let admitted = render_pass::admit(&colors, None).expect("one colour at index zero");
        let resolved = table.texture_desc(*admitted.texture).expect("a live texture");
        let view = table.texture_view(*admitted.texture).expect("a live view");
        let image = table.texture_image(*admitted.texture).expect("a live image");
        let mapped = format::image_format(resolved.format).expect("a mapped format");
        let framebuffer = Framebuffer::create(opened.device.device(), admitted, &resolved, view)
            .expect("a render pass and framebuffer for a real colour target");

        let pool = CommandPool::new(
            opened.device.device(),
            opened.device.selected_queue().family,
        )
        .expect("a command pool on an opened device");
        let mut encoder = pool.begin().expect("a recording encoder");

        assert_eq!(
            encoder.end_raster(),
            Err(RecordError::NoPass),
            "there is no pass to end yet"
        );

        encoder
            .begin_raster(&framebuffer)
            .expect("the pass bracket opens");
        assert_eq!(
            encoder.begin_raster(&framebuffer),
            Err(RecordError::PassAlreadyOpen)
        );
        assert_eq!(
            encoder.transition_image(
                image,
                mapped,
                TextureRange::Whole,
                ResourceAccessState::ColorAttachmentWrite,
                ResourceAccessState::CopySource,
            ),
            Err(RecordError::PassOpen),
            "a barrier is not legal inside a render pass"
        );
        assert_eq!(
            encoder.end(),
            Err(RecordError::PassOpen),
            "a recording may not end with a pass open"
        );

        encoder.end_raster().expect("the pass bracket closes");
        assert_eq!(encoder.end_raster(), Err(RecordError::NoPass));
        // The refused calls did not poison the recording: it still ends, and the
        // transition before them is intact.
        encoder
            .transition_image(
                image,
                mapped,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .expect("the encoder is usable after the refusals");
        encoder.end().expect("the recording ends");
    }
}
