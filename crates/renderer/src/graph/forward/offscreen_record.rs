//! Offscreen recording against the ordinary Forward graph namespace.
use super::shared_record::{self, RecordSlots};
use super::*;

impl<B: ResourceBackend> ForwardPlan<B> {
    pub(super) fn record_slots(&self) -> RecordSlots<'_> {
        RecordSlots {
            device: self.id.device,
            topology: self.topology,
            resident_upload_slot: self.resident_upload_slot,
            frame_upload_slot: self.frame_upload_slot,
            uniform_slot: self.uniform_slot,
            environment_upload_slot: self.environment_upload_slot,
            specular_upload_slot: self.specular_upload_slot,
            specular_view_uniform_slot: self.specular_view_uniform_slot,
            stream_slots: &self.stream_slots,
            texture_slots: &self.texture_slots,
            environment_slot: self.environment_slot,
            specular_radiance_slot: self.specular_radiance_slot,
            specular_brdf_lut_slot: self.specular_brdf_lut_slot,
            depth_slot: self.depth_slot,
            target_slot: self.target_slot,
            multisample_color_slot: self.multisample_color_slot,
            readback_slot: self.readback_slot,
        }
    }
    pub(crate) fn record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        shared_record::record_upload(
            &self.record_slots(),
            self.upload_pass,
            &self.bindings(frame)?,
            encoder,
            frame,
        )
    }
    pub(crate) fn record_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        shared_record::record_raster(
            &self.record_slots(),
            self.raster_pass,
            &self.bindings(frame)?,
            encoder,
            frame,
        )
    }
    pub(crate) fn record_mip(
        &self,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let generation = self.mip_passes.get(index).ok_or(Error::InvalidRange)?;
        let slot = *self
            .texture_slots
            .get(generation.texture)
            .ok_or(Error::InvalidRange)?;
        let texture = frame
            .resident
            .textures
            .get(generation.texture)
            .ok_or(Error::InvalidRange)?;
        super::super::mip_generation::record(
            generation,
            &self.bindings(frame)?,
            slot,
            texture,
            encoder,
        )
    }
    pub(crate) fn record_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        shared_record::record_readback(
            &self.record_slots(),
            self.readback_pass,
            &self.bindings(frame)?,
            encoder,
            frame,
        )
    }
}
