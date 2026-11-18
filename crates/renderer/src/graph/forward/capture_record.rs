//! Recording against the shared, checked capture/receiver Graph bindings.
use super::super::mip_generation::{MipGenerationPass, record as record_mip_generation};
use super::shared_record::{self, RecordSlots};
use super::*;

impl<B: ResourceBackend> CaptureForwardPlan<B> {
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
            multisample_color_slot: None,
            readback_slot: slots.readback,
        }
    }

    pub(crate) fn record_capture_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        shared_record::record_upload(
            &self.record_slots(&self.capture, self.capture_topology),
            self.capture_upload,
            &self.bindings(capture, receiver)?,
            encoder,
            capture,
        )
    }

    pub(crate) fn record_capture_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if self.receiver_depth_alias {
            return shared_record::record_raster_with_loads(
                &self.record_slots(&self.capture, self.capture_topology),
                self.capture_raster,
                &self.bindings(capture, receiver)?,
                encoder,
                capture,
                capture.frame.depth.as_ref(),
                LoadOp::Clear(ClearColor::Float([0.0; 4])),
                LoadOp::Clear(1.0),
                StoreOp::Store,
                false,
            );
        }
        shared_record::record_raster(
            &self.record_slots(&self.capture, self.capture_topology),
            self.capture_raster,
            &self.bindings(capture, receiver)?,
            encoder,
            capture,
        )
    }

    pub(crate) fn record_capture_mip_generation(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        generation: &MipGenerationPass,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        record_mip_generation(
            generation,
            &self.bindings(capture, receiver)?,
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
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        shared_record::record_upload(
            &self.record_slots(&self.receiver, self.receiver_topology),
            self.receiver_upload,
            &self.bindings(capture, receiver)?,
            encoder,
            receiver,
        )
    }

    pub(crate) fn record_receiver_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let slots = self.record_slots(&self.receiver, self.receiver_topology);
        if self.receiver_depth_alias {
            return shared_record::record_raster_with_loads(
                &slots,
                self.receiver_raster,
                &self.bindings(capture, receiver)?,
                encoder,
                receiver,
                capture.frame.depth.as_ref(),
                LoadOp::Load,
                LoadOp::Load,
                StoreOp::Store,
                !self
                    .receiver_shape
                    .draws
                    .iter()
                    .any(|draw| draw.compatibility.writes_depth()),
            );
        }
        shared_record::record_raster(
            &slots,
            self.receiver_raster,
            &self.bindings(capture, receiver)?,
            encoder,
            receiver,
        )
    }

    pub(crate) fn record_receiver_background_copy(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let pass_id = self.receiver_background_copy.ok_or(Error::InvalidState)?;
        let bindings = self.bindings(capture, receiver)?;
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
        generation: &MipGenerationPass,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        record_mip_generation(
            generation,
            &self.bindings(capture, receiver)?,
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

    pub(crate) fn record_receiver_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        if let Some(terminal) = self.hdr_terminal() {
            let mut slots = self.record_slots(&self.receiver, self.receiver_topology);
            slots.target_slot = terminal.output_slot;
            return shared_record::record_readback(
                &slots,
                terminal.readback_pass,
                &self.bindings(capture, receiver)?,
                encoder,
                receiver,
            );
        }
        shared_record::record_readback(
            &self.record_slots(&self.receiver, self.receiver_topology),
            self.receiver_readback,
            &self.bindings(capture, receiver)?,
            encoder,
            receiver,
        )
    }

    pub(crate) fn record_hdr_terminal(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.hdr_terminal().ok_or(Error::InvalidState)?.record(
            device,
            &self.bindings(capture, receiver)?,
            encoder,
            self.receiver.target,
            receiver.frame.target_instance,
            receiver,
            composite,
        )
    }
}
