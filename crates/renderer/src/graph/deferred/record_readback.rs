//! Deferred offscreen readback recording.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn record_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        let bindings = self.bindings(frame)?;
        self.record_readback_with_bindings(encoder, frame, &bindings)
    }

    pub(super) fn record_readback_with_bindings(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        let target_slot = self.target_slot.ok_or(Error::InvalidState)?;
        let readback_slot = self.readback_slot.ok_or(Error::InvalidState)?;
        let length = u64::from(frame.readback_row_pitch)
            .checked_mul(u64::from(frame.extent[1]))
            .ok_or(Error::InvalidRange)?;
        let readback_pass = match self.output {
            DeferredOutputPlan::Offscreen { readback_pass } => readback_pass,
            DeferredOutputPlan::Capture | DeferredOutputPlan::Surface { .. } => {
                return Err(Error::InvalidState);
            }
        };
        let pass = bindings.pass(readback_pass)?;
        let target = pass.texture(
            target_slot,
            whole_color(),
            AccessKind::Read,
            texture_copy_read(),
        )?;
        let readback = pass.buffer(readback_slot, 0..length, AccessKind::Write, copy_write())?;
        pass.transition(encoder)?;
        encoder.texture_to_buffer(
            TextureRegion {
                texture: target,
                mip: 0,
                origin: [0, 0, 0],
                extent: [frame.extent[0], frame.extent[1], 1],
                aspect: TextureAspects::COLOR,
            },
            readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: frame.readback_row_pitch,
                rows_per_image: frame.extent[1],
            },
        )?;
        encoder.readback_buffer(readback, 0..length)
    }
}
