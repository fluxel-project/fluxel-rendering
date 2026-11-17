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
        shared_record::record_raster(
            &self.record_slots(&self.receiver, self.receiver_topology),
            self.receiver_raster,
            &self.bindings(capture, receiver)?,
            encoder,
            receiver,
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
        shared_record::record_readback(
            &self.record_slots(&self.receiver, self.receiver_topology),
            self.receiver_readback,
            &self.bindings(capture, receiver)?,
            encoder,
            receiver,
        )
    }
}
