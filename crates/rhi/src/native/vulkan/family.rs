//! W2's first family wiring: `VulkanDevice` as the graphics family's provider.
//!
//! The common layer's contract says a device *negotiates* a family and receives a
//! handle that already borrows the device and the ledger that proved it (plan
//! section 11). This module is the first real backend to do that: `VulkanDevice`
//! implements [`Provides<Graphics>`] and hands out [`GraphicsRecording`], which
//! implements [`FamilyApi`] and [`GraphicsApi`] over the device's own resource table
//! and one recording from the device's own command pool.
//!
//! # The handle is the recording context, and the encoder is private
//!
//! Every `GraphicsApi` verb takes `&mut self`, so the handle can own the
//! [`Encoder`] without a backend encoder type appearing in the common vocabulary.
//! The encoder is created lazily, on the first [`GraphicsRecording::begin_raster`],
//! rather than by `provide`: `Provides::provide` cannot report a failure, and
//! allocating a command buffer can fail, so the allocation happens at the verb that
//! can return [`GraphicsError::Recording`]. A handle that was negotiated and never
//! used therefore allocates nothing.
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
//! No transitions, no copies and no submission verbs. A pipeline barrier is a
//! backend mechanism and the common contract has none (plan section 1), so the
//! graph's own barrier is a crate-private method on the handle rather than a
//! `GraphicsApi` verb; copies are the `Copy` family; and the contract has no
//! submission verb yet, so [`GraphicsRecording::finish`] hands the ended recording
//! to step 9's submission path without pretending to be family vocabulary.

use std::ops::Range;

use fluxel_rendergraph::{
    BufferRange, IndexFormat, RasterPassDescriptor, ResourceAccessState, ScissorRect, TextureRange,
    Viewport,
};

use crate::common::api::family::Graphics;
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
use super::pipeline::RasterPipeline;
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
    /// [`GraphicsRecording::finish`] took the recording.
    Finished,
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
    device: &'d VulkanDevice,
    /// The recording, begun by the first command that needs one.
    ///
    /// It is begun lazily rather than by `provide` because `Provides::provide`
    /// cannot report a failure and allocating a command buffer can fail; it is begun
    /// by *any* command rather than only by `begin_raster` because the graph records
    /// its transitions before it opens the pass.
    stage: Stage,
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
            device,
            stage: Stage::Fresh,
            targets: Vec::new(),
        }
    }

    /// Returns the live recording, beginning it if this is the first command.
    ///
    /// A finished handle refuses even though it could allocate another command
    /// buffer: recording past the handoff would be a second recording the caller
    /// never asked for and would never submit.
    fn recording(&mut self) -> Result<&mut Encoder, GraphicsError> {
        if let Stage::Finished = self.stage {
            return Err(GraphicsError::Recording(RecordError::NotRecording));
        }
        if matches!(self.stage, Stage::Fresh) {
            let encoder = self
                .device
                .pool()
                .begin()
                .map_err(GraphicsError::Recording)?;
            self.stage = Stage::Recording(Box::new(encoder));
        }
        match &mut self.stage {
            Stage::Recording(encoder) => Ok(encoder),
            // `Fresh` was replaced just above and `Finished` returned above, so this
            // arm is not reachable; it is a value rather than a panic for the same
            // reason every other impossible shape in this backend is.
            Stage::Fresh | Stage::Finished => {
                Err(GraphicsError::Recording(RecordError::NotRecording))
            }
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
            .device
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.recording()?
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
            .device
            .table()
            .texture_desc(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let image = self
            .device
            .table()
            .texture_image(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let mapped = format::image_format(desc.format).ok_or(GraphicsError::UnsupportedFormat)?;
        self.recording()?
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
        match core::mem::replace(&mut self.stage, Stage::Finished) {
            Stage::Recording(encoder) => (*encoder).finish().map_err(GraphicsError::Recording),
            Stage::Fresh | Stage::Finished => {
                Err(GraphicsError::Recording(RecordError::NotRecording))
            }
        }
    }
}

impl FamilyApi for GraphicsRecording<'_> {
    fn stamp(&self) -> DeviceStamp {
        self.device.stamp()
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
        self.recording()?;

        // The preserved attachment rule runs first, before any id is resolved: a
        // pass with no pipeline that could run in it is refused for that reason
        // rather than for whatever its first attachment happens to name.
        let admitted = render_pass::admit(descriptor.colors, descriptor.depth_stencil.as_ref())
            .map_err(GraphicsError::Pass)?;
        let texture = *admitted.texture;
        let desc = self
            .device
            .table()
            .texture_desc(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let view = self
            .device
            .table()
            .texture_view(texture)
            .ok_or(GraphicsError::UnknownTexture)?;
        let framebuffer = Framebuffer::create(self.device.device(), admitted, &desc, view)
            .map_err(GraphicsError::Target)?;
        // Only a target the driver accepted is kept: a refused begin leaves the
        // framebuffer to its own drop, so the recording names nothing that was
        // released.
        self.recording()?
            .begin_raster(&framebuffer)
            .map_err(GraphicsError::Recording)?;
        self.targets.push(framebuffer);
        Ok(())
    }

    fn end_raster(&mut self) -> Result<(), Self::Error> {
        self.recording()?
            .end_raster()
            .map_err(GraphicsError::Recording)
    }

    fn set_raster_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<(), Self::Error> {
        self.recording()?
            .set_raster_pipeline(pipeline)
            .map_err(GraphicsError::Recording)
    }

    fn set_bindings(&mut self, bindings: &Self::Bindings) -> Result<(), Self::Error> {
        self.recording()?
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
            .device
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.recording()?
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
            .device
            .table()
            .buffer_handle(buffer)
            .ok_or(GraphicsError::UnknownBuffer)?;
        self.recording()?
            .set_index_buffer(handle, offset, index_format)
            .map_err(GraphicsError::Recording)
    }

    fn set_viewport(&mut self, viewport: Viewport) -> Result<(), Self::Error> {
        self.recording()?
            .set_viewport(viewport)
            .map_err(GraphicsError::Recording)
    }

    fn set_scissor(&mut self, scissor: ScissorRect) -> Result<(), Self::Error> {
        self.recording()?
            .set_scissor(scissor)
            .map_err(GraphicsError::Recording)
    }

    fn draw(&mut self, vertices: Range<u32>, instance_count: u32) -> Result<(), Self::Error> {
        self.recording()?
            .draw(vertices, instance_count)
            .map_err(GraphicsError::Recording)
    }

    fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        instance_count: u32,
    ) -> Result<(), Self::Error> {
        self.recording()?
            .draw_indexed(indices, instance_count)
            .map_err(GraphicsError::Recording)
    }
}

impl core::fmt::Debug for GraphicsRecording<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GraphicsRecording")
            .field("recording", &matches!(self.stage, Stage::Recording(_)))
            .field("finished", &matches!(self.stage, Stage::Finished))
            .field("targets", &self.targets.len())
            .finish_non_exhaustive()
    }
}

/// The first real backend to hold a family's vocabulary.
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
    use crate::native::vulkan::pipeline::{create_layout, create_raster};
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
}
