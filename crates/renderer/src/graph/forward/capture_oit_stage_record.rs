//! Opaque, peeling, and excluded-pass recording for capture OIT plans.

use super::shared_record::{self, RasterColorSelection, RasterDepthSelection, RasterSelection};
use super::*;
use crate::graph::oit_record::{OitDepthTargetRef, OitDrawSlots, OitRecordCommands};

impl<B: ResourceBackend> CaptureOitPlan<B> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_oit_opaque(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        slots: &CaptureFrameSlots,
        oit: &super::oit_plan::ForwardOitPlan,
        depth: &crate::residency::ForwardDepthAttachment<B>,
        load_existing: bool,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        shared_record::record_raster_selection(
            &self.slots(slots),
            pass,
            bindings,
            encoder,
            frame,
            RasterSelection {
                draws: 0..oit.opaque_count,
                color: RasterColorSelection {
                    slot: slots.target,
                    texture: &frame.frame.target,
                    view: &frame.frame.target_view,
                    instance: frame.frame.target_instance,
                    load: if load_existing {
                        LoadOp::Load
                    } else {
                        LoadOp::Clear(ClearColor::Float([0.0; 4]))
                    },
                    read_write: load_existing,
                },
                depth: Some(RasterDepthSelection {
                    slot: slots.depth.ok_or(Error::InvalidState)?,
                    texture: &depth.texture,
                    view: &depth.view,
                    instance: depth.instance,
                    load: if load_existing {
                        LoadOp::Load
                    } else {
                        LoadOp::Clear(1.0)
                    },
                    store: StoreOp::Store,
                    read_only: false,
                    read_write: load_existing,
                }),
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_oit_stage(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
        slots: &CaptureFrameSlots,
        oit: &super::oit_plan::ForwardOitPlan,
        opaque_depth: &crate::residency::ForwardDepthAttachment<B>,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let depth = OitDepthTargetRef::from(opaque_depth);
        crate::graph::oit_record::record_pass(
            self.id.device,
            encoder,
            bindings,
            &oit.product,
            pass,
            &oit.slots,
            input.owners,
            &depth,
            frame,
            &OitDrawSlots {
                streams: &slots.streams,
                uniform: slots.uniform,
                textures: &slots.textures,
                environment: slots.environment,
                specular_radiance: slots.specular_radiance,
                specular_brdf_lut: slots.specular_brdf_lut,
                specular_view_uniform: slots.specular_view_uniform,
            },
            commands,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_oit_fallback(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        slots: &CaptureFrameSlots,
        oit: &super::oit_plan::ForwardOitPlan,
        opaque_depth: &crate::residency::ForwardDepthAttachment<B>,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        shared_record::record_raster_selection(
            &self.slots(slots),
            pass,
            bindings,
            encoder,
            frame,
            RasterSelection {
                draws: (oit.opaque_count + oit.peel_count)..frame.draws.len(),
                color: RasterColorSelection {
                    slot: oit.slots.output,
                    texture: &input.output.texture,
                    view: &input.output.view,
                    instance: input.output.instance,
                    load: LoadOp::Load,
                    read_write: true,
                },
                depth: Some(RasterDepthSelection {
                    slot: slots.depth.ok_or(Error::InvalidState)?,
                    texture: &opaque_depth.texture,
                    view: &opaque_depth.view,
                    instance: opaque_depth.instance,
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                    read_only: true,
                    read_write: false,
                }),
            },
        )
    }
}
