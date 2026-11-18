//! Forward material work on attachments already owned by an enclosing Graph.

use super::capture_declaration::{CaptureFrameSlots, FrameHandles, SharedResidentImports};
use super::shared_record::{
    self, RasterColorSelection, RasterDepthSelection, RasterSelection, RecordSlots,
};
use super::*;
use crate::graph::definition::{GraphError, PassBuilder, TextureHandle};

pub(in crate::graph) struct OverlayPrefix {
    pub(in crate::graph) slots: OverlaySlots,
    pub(in crate::graph) upload: GraphPass,
    pub(in crate::graph) mip_generation: Vec<super::super::mip_generation::MipGenerationPass>,
    handles: FrameHandles,
    spec: ForwardGraphSpec,
}

pub(in crate::graph) struct OverlaySlots {
    slots: CaptureFrameSlots,
    shape: ForwardShape,
    alignment: u64,
    writes_depth: bool,
}

pub(in crate::graph) fn declare_overlay_prefix<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    frame: &ForwardFrame<B>,
    color: TextureHandle,
    depth: TextureHandle,
    alignment: u64,
) -> Result<OverlayPrefix> {
    let mut shared = SharedResidentImports::default();
    declare_overlay_prefix_with_shared(graph, frame, color, depth, alignment, &mut shared)
}

/// Declares an overlay namespace using caller-owned completed resident imports.
pub(in crate::graph) fn declare_overlay_prefix_with_shared<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    frame: &ForwardFrame<B>,
    color: TextureHandle,
    depth: TextureHandle,
    alignment: u64,
    shared: &mut SharedResidentImports,
) -> Result<OverlayPrefix> {
    let spec = ForwardGraphSpec::from_frame(frame)?;
    if frame.draws.is_empty() || alignment == 0 {
        return Err(Error::InvalidRange);
    }
    let (slots, handles) =
        capture_declaration::import_frame_inner(graph, &spec, None, shared, Some((color, depth)))
            .map_err(plan_error)?;
    let (upload, handles) =
        capture_declaration::declare_upload(graph, &spec, handles).map_err(plan_error)?;
    let (handles, mip_generation) =
        capture_declaration::declare_mip_generations(graph, &spec, handles, shared)
            .map_err(plan_error)?;
    Ok(OverlayPrefix {
        slots: OverlaySlots {
            slots,
            shape: spec.shape.clone(),
            alignment,
            writes_depth: frame
                .draws
                .iter()
                .any(|draw| draw.compatibility.writes_depth()),
        },
        upload,
        mip_generation,
        handles,
        spec,
    })
}

impl OverlayPrefix {
    pub(in crate::graph) fn declare_reads(
        &self,
        pass: &mut PassBuilder<'_>,
    ) -> std::result::Result<(), GraphError> {
        self.declare_draw_reads(pass, 0..self.spec.draws.len())
    }

    pub(in crate::graph) fn declare_draw_reads(
        &self,
        pass: &mut PassBuilder<'_>,
        draws: Range<usize>,
    ) -> std::result::Result<(), GraphError> {
        let reads =
            super::oit_draw_reads::OitDrawReads::from_capture_handles(&self.handles, &self.spec);
        let mut failure = None;
        reads.declare_with(
            pass,
            draws,
            &mut |context| context.all_reads(),
            &mut failure,
        )
    }
}

impl OverlaySlots {
    pub(in crate::graph) fn validate_frame<B: ResourceBackend>(
        &self,
        device: DeviceId,
        frame: &ForwardFrame<B>,
    ) -> Result<()> {
        validate_resident_texture_devices(&frame.resident, device)?;
        if frame.resident.device != device || forward_shape(frame) != self.shape {
            return Err(Error::InvalidState);
        }
        let depth = frame.frame.depth.as_ref().ok_or(Error::InvalidState)?;
        if frame.frame.target.device() != device
            || frame.frame.target_view.device() != device
            || depth.device != device
            || depth.texture.device() != device
            || depth.view.device() != device
        {
            return Err(Error::WrongDevice);
        }
        validate_uploads(frame)?;
        validate_batches(frame)?;
        if frame
            .draws
            .iter()
            .any(|draw| draw.compatibility.writes_depth())
            != self.writes_depth
        {
            return Err(Error::InvalidState);
        }
        for draw in &frame.draws {
            validate_draw(draw, frame.resident.streams.len())?;
            validate_offsets(draw, self.alignment)?;
            draw_accesses(frame, draw)?;
            if draw.pipeline.device() != device
                || draw
                    .bind_groups
                    .iter()
                    .any(|group| group.group.device() != device)
            {
                return Err(Error::WrongDevice);
            }
        }
        if let Some(owner) = frame.environment.as_ref() {
            owner.validate_for_sampling(device)?;
        }
        if let Some(owner) = frame.specular.as_ref() {
            owner.validate_for_sampling(device)?;
        }
        Ok(())
    }

    /// Whether this overlay has an opaque or masked Forward draw which writes
    /// the enclosing owner's actual depth attachment.
    pub(in crate::graph) const fn writes_depth(&self) -> bool {
        self.writes_depth
    }

    /// The enclosing owner binds color and depth once. This namespace only
    /// contributes the concrete material/geometry/uniform owners it introduces.
    pub(in crate::graph) fn append_bindings<'a, B: ResourceBackend>(
        &self,
        execution: &CompiledExecution,
        frame: &'a ForwardFrame<B>,
        buffers: &mut Vec<BoundBuffer<'a, B>>,
        textures: &mut Vec<BoundTexture<'a, B>>,
    ) -> Result<()> {
        let slots = &self.slots;
        let uses = |slot| {
            execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|r| r.slot == slot)
        };
        let mut buffer = |slot, instance, owner, desc, incoming| {
            if uses(slot) {
                buffers.push(BoundBuffer::new(slot, instance, owner, desc, incoming));
            }
        };
        buffer(
            slots.resident_upload,
            frame.resident.upload_instance,
            &frame.resident.upload,
            &frame.resident.upload_desc,
            frame.resident.upload_incoming,
        );
        buffer(
            slots.frame_upload,
            frame.frame.upload_instance,
            &frame.frame.upload,
            &frame.frame.upload_desc,
            frame.frame.upload_incoming,
        );
        buffer(
            slots.uniform,
            frame.frame.uniform_instance,
            &frame.frame.uniform,
            &frame.frame.uniform_desc,
            frame.frame.uniform_incoming,
        );
        for (&slot, owner) in slots.streams.iter().zip(&frame.resident.streams) {
            buffer(
                slot,
                owner.instance,
                &owner.buffer,
                &owner.desc,
                owner.incoming,
            );
        }
        if let (Some(slot), Some(owner)) = (slots.environment_upload, frame.environment.as_ref()) {
            buffer(
                slot,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming,
            );
        }
        for (&slot, owner) in slots.textures.iter().zip(&frame.resident.textures) {
            textures.push(BoundTexture::new(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout,
            ));
        }
        if let (Some(slot), Some(owner)) = (slots.environment, frame.environment.as_ref()) {
            textures.push(BoundTexture::new(
                slot,
                owner.texture_instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout,
            ));
        }
        super::capture_specular_bindings::append(
            slots,
            frame.specular.as_ref(),
            frame.resident.device,
            buffers,
            textures,
        )
    }

    pub(in crate::graph) fn draw_slots(&self) -> crate::graph::oit_record::OitDrawSlots<'_> {
        let s = &self.slots;
        crate::graph::oit_record::OitDrawSlots {
            streams: &s.streams,
            uniform: s.uniform,
            textures: &s.textures,
            environment: s.environment,
            specular_radiance: s.specular_radiance,
            specular_brdf_lut: s.specular_brdf_lut,
            specular_view_uniform: s.specular_view_uniform,
        }
    }

    fn record_slots(&self, device: DeviceId) -> RecordSlots<'_> {
        let s = &self.slots;
        RecordSlots {
            device,
            topology: FrameTopology::SortedForward,
            resident_upload_slot: s.resident_upload,
            frame_upload_slot: s.frame_upload,
            uniform_slot: s.uniform,
            environment_upload_slot: s.environment_upload,
            specular_upload_slot: s.specular_upload,
            specular_view_uniform_slot: s.specular_view_uniform,
            stream_slots: &s.streams,
            texture_slots: &s.textures,
            environment_slot: s.environment,
            specular_radiance_slot: s.specular_radiance,
            specular_brdf_lut_slot: s.specular_brdf_lut,
            depth_slot: s.depth,
            target_slot: s.target,
            readback_slot: s.readback,
        }
    }

    pub(in crate::graph) fn record_upload<B: ResourceBackend + CommandBackend>(
        &self,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        frame: &ForwardFrame<B>,
        pass: GraphPass,
    ) -> Result<()> {
        shared_record::record_upload(
            &self.record_slots(frame.resident.device),
            pass,
            bindings,
            encoder,
            frame,
        )
    }

    pub(in crate::graph) fn record_mip<B: ResourceBackend + CommandBackend>(
        &self,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        frame: &ForwardFrame<B>,
        generation: &super::super::mip_generation::MipGenerationPass,
    ) -> Result<()> {
        let slot = *self
            .slots
            .textures
            .get(generation.texture)
            .ok_or(Error::InvalidRange)?;
        let owner = frame
            .resident
            .textures
            .get(generation.texture)
            .ok_or(Error::InvalidRange)?;
        super::super::mip_generation::record(generation, bindings, slot, owner, encoder)
    }

    pub(in crate::graph) fn record_raster<B: ResourceBackend + RasterBackend>(
        &self,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        frame: &ForwardFrame<B>,
        color: &crate::residency::ForwardTexture<B>,
        depth: &crate::residency::ForwardTexture<B>,
        pass: GraphPass,
    ) -> Result<()> {
        self.record_raster_range(
            encoder,
            bindings,
            frame,
            crate::graph::oit_record::OitColorTargetRef {
                texture: &color.texture,
                view: &color.view,
                desc: &color.desc,
                instance: color.instance,
                device: color.device,
            },
            self.slots.target,
            depth,
            0..frame.draws.len(),
            pass,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::graph) fn record_raster_range<B: ResourceBackend + RasterBackend>(
        &self,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        frame: &ForwardFrame<B>,
        color: crate::graph::oit_record::OitColorTargetRef<'_, B>,
        color_slot: TextureSlot,
        depth: &crate::residency::ForwardTexture<B>,
        draws: Range<usize>,
        pass: GraphPass,
    ) -> Result<()> {
        shared_record::record_raster_selection(
            &self.record_slots(frame.resident.device),
            pass,
            bindings,
            encoder,
            frame,
            RasterSelection {
                draws,
                color: RasterColorSelection {
                    slot: color_slot,
                    texture: color.texture,
                    view: color.view,
                    instance: color.instance,
                    load: LoadOp::Load,
                    read_write: true,
                },
                depth: Some(RasterDepthSelection {
                    slot: self.slots.depth.ok_or(Error::InvalidState)?,
                    texture: &depth.texture,
                    view: &depth.view,
                    instance: depth.instance,
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                    read_only: !self.writes_depth,
                    read_write: self.writes_depth,
                }),
            },
        )
    }
}

fn plan_error(error: ForwardPlanBuildError) -> Error {
    match error {
        ForwardPlanBuildError::Rhi(error) => error,
        ForwardPlanBuildError::Pipeline(_) => Error::InvalidState,
    }
}
