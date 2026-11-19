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
    /// Records an ordinary Forward raster pass that consumes the frame's
    /// declared indirect draw records.
    pub(crate) fn record_raster_indirect(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let bindings = self.bindings(frame)?;
        for (declared, (_, product)) in self.indirect.iter().zip(&frame.indirect) {
            declared
                .graph
                .validate_raster_read(self.raster_pass, &bindings, product)?;
        }
        for (declared, (_, step)) in self.particles.iter().zip(&frame.particles) {
            declared
                .graph
                .validate_vertex_read(self.raster_pass, &bindings, step)?;
        }
        shared_record::record_raster_with_loads_indirect(
            &self.record_slots(),
            self.raster_pass,
            &bindings,
            encoder,
            frame,
            frame.frame.depth.as_ref(),
            LoadOp::Clear(ClearColor::Float([0.0; 4])),
            LoadOp::Clear(1.0),
            StoreOp::Discard,
            false,
            device,
        )
    }
    /// Records one frame-local indirect parameter copy.
    pub(crate) fn record_indirect_upload(
        &self,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let declared = self.indirect.get(index).ok_or(Error::InvalidRange)?;
        if frame.indirect.get(index).is_none() {
            return Err(Error::InvalidRange);
        }
        declared
            .graph
            .record_upload(encoder, &self.bindings(frame)?)
    }
    /// Records one frame-local indirect argument producer.
    pub(crate) fn record_indirect_compute(
        &self,
        device: &Device<B>,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let declared = self.indirect.get(index).ok_or(Error::InvalidRange)?;
        let (_, product) = frame.indirect.get(index).ok_or(Error::InvalidRange)?;
        declared
            .graph
            .record_compute(device, encoder, &self.bindings(frame)?, product)
    }
    pub(crate) fn record_particle_upload(
        &self,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let declared = self.particles.get(index).ok_or(Error::InvalidRange)?;
        if frame.particles.get(index).is_none() {
            return Err(Error::InvalidRange);
        }
        declared
            .graph
            .record_upload(encoder, &self.bindings(frame)?)
    }
    pub(crate) fn record_particle_update(
        &self,
        device: &Device<B>,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let declared = self.particles.get(index).ok_or(Error::InvalidRange)?;
        let (_, step) = frame.particles.get(index).ok_or(Error::InvalidRange)?;
        declared
            .graph
            .record_update(device, encoder, &self.bindings(frame)?, step)
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
