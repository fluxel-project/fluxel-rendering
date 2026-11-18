//! Recording for the capture-to-surface Graph route.
use super::super::mip_generation::{MipGenerationPass, record as record_mip_generation};
use super::shared_record::{self, RecordSlots};
use super::*;
use crate::renderer::SurfaceCompositeOwner;

impl<B: ResourceBackend> CaptureForwardSurfacePlan<B> {
    fn record_slots<'a>(
        &self,
        slots: &'a CaptureFrameSlots,
        topology: FrameTopology,
    ) -> RecordSlots<'a> {
        RecordSlots {
            device: self.id.device,
            topology,
            resident_upload_slot: slots.resident_upload,
            frame_upload_slot: slots.frame_upload,
            uniform_slot: slots.uniform,
            environment_upload_slot: slots.environment_upload,
            specular_upload_slot: slots.specular_upload,
            specular_view_uniform_slot: slots.specular_view_uniform,
            stream_slots: &slots.streams,
            texture_slots: &slots.textures,
            environment_slot: slots.environment,
            specular_radiance_slot: slots.specular_radiance,
            specular_brdf_lut_slot: slots.specular_brdf_lut,
            depth_slot: slots.depth,
            target_slot: slots.target,
            readback_slot: slots.readback,
        }
    }
    pub(crate) fn record_capture_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        shared_record::record_upload(
            &self.record_slots(&self.capture, self.capture_topology),
            self.capture_upload,
            &self.bindings(capture, receiver, image)?,
            encoder,
            capture,
        )
    }
    pub(crate) fn record_capture_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        shared_record::record_raster(
            &self.record_slots(&self.capture, self.capture_topology),
            self.capture_raster,
            &self.bindings(capture, receiver, image)?,
            encoder,
            capture,
        )
    }
    pub(crate) fn record_capture_mip_generation(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
        generation: &MipGenerationPass,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        record_mip_generation(
            generation,
            &self.bindings(capture, receiver, image)?,
            *self
                .capture
                .textures
                .get(generation.texture)
                .ok_or(Error::InvalidRange)?,
            capture
                .resident
                .textures
                .get(generation.texture)
                .ok_or(Error::InvalidRange)?,
            encoder,
        )
    }
    pub(crate) fn record_receiver_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        shared_record::record_upload(
            &self.record_slots(&self.receiver, self.receiver_topology),
            self.receiver_upload,
            &self.bindings(capture, receiver, image)?,
            encoder,
            receiver,
        )
    }
    pub(crate) fn record_receiver_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let slots = self.record_slots(&self.receiver, self.receiver_topology);
        if self.receiver_background_copy.is_some() {
            return shared_record::record_raster_with_color_load(
                &slots,
                self.receiver_raster,
                &self.bindings(capture, receiver, image)?,
                encoder,
                receiver,
                LoadOp::Load,
            );
        }
        shared_record::record_raster(
            &slots,
            self.receiver_raster,
            &self.bindings(capture, receiver, image)?,
            encoder,
            receiver,
        )
    }

    pub(crate) fn record_receiver_background_copy(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let pass_id = self.receiver_background_copy.ok_or(Error::InvalidState)?;
        let bindings = self.bindings(capture, receiver, image)?;
        super::capture_terminal::record_background_copy(
            &bindings,
            pass_id,
            encoder,
            self.capture.target,
            self.receiver.target,
            capture.extent,
        )
    }
    pub(crate) fn record_receiver_mip_generation(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
        generation: &MipGenerationPass,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        record_mip_generation(
            generation,
            &self.bindings(capture, receiver, image)?,
            *self
                .receiver
                .textures
                .get(generation.texture)
                .ok_or(Error::InvalidRange)?,
            receiver
                .resident
                .textures
                .get(generation.texture)
                .ok_or(Error::InvalidRange)?,
            encoder,
        )
    }
    pub(crate) fn record_surface_composite(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        composite: &SurfaceCompositeOwner<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if image.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != self.extent
            || image.format() != self.color_format
            || composite.extent() != self.extent
            || composite.target_instance() != receiver.frame.target_instance
        {
            return Err(Error::InvalidRange);
        }
        let bindings = self.bindings(capture, receiver, image)?;
        let pass = bindings.pass(self.composite)?;
        pass.texture(
            self.receiver.target,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
        pass.surface_texture(
            self.surface_target,
            whole(),
            AccessKind::Write,
            color_write_uses(),
        )?;
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel forward capture surface composite"),
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: self.extent[0],
                height: self.extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        raster.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent[0] as f32,
            height: self.extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        raster.scissor(Rect {
            x: 0,
            y: 0,
            width: self.extent[0],
            height: self.extent[1],
        })?;
        raster.set_pipeline(composite.pipeline())?;
        raster.set_bind_group(0, composite.bind_group(), &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }
}
