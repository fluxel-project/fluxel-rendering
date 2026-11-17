//! Compiled same-submission Deferred dual-depth-peeling plan.

use super::*;

pub(crate) struct DeferredOitPlan<B: ResourceBackend> {
    pub(super) base: DeferredPlan<B>,
    pub(super) overlay: crate::graph::forward::OverlaySlots,
    pub(super) overlay_upload: GraphPass,
    pub(super) overlay_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) product: crate::graph::oit::DualDepthPeelGraph,
    pub(super) slots: crate::graph::oit_record::OitRecordSlots,
    pub(super) fallback: Option<GraphPass>,
    pub(super) final_slot: TextureSlot,
    pub(super) surface: Option<([u32; 2], TextureFormat)>,
    pub(super) peel_count: usize,
    pub(super) rounds: u32,
}

impl<B: ResourceBackend> DeferredOitPlan<B> {
    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: DeferredOitDeclaration,
    ) -> Result<Self> {
        let mut composition = vec![declaration.overlay_upload];
        composition.extend(declaration.overlay_mips.iter().map(|item| item.pass));
        let p = &declaration.product.passes;
        composition.extend([
            p.initial_clear[0].pass,
            p.initial_clear[1].pass,
            p.initialize_back_accumulator,
            p.initial_depth,
        ]);
        for iteration in &p.iterations {
            composition.extend([
                iteration.clear.pass,
                iteration.peel,
                iteration.back_accumulate,
            ]);
        }
        composition.push(p.final_composite);
        if let Some(pass) = declaration.fallback {
            composition.push(pass);
        }
        let base =
            DeferredPlan::compile_with_composition(device, id, declaration.base, &composition)?;
        Ok(Self {
            base,
            overlay: declaration.overlay,
            overlay_upload: declaration.overlay_upload,
            overlay_mips: declaration.overlay_mips,
            product: declaration.product,
            slots: declaration.slots,
            fallback: declaration.fallback,
            final_slot: declaration.final_slot,
            surface: declaration.surface,
            peel_count: declaration.peel_count,
            rounds: declaration.rounds,
        })
    }

    pub(crate) fn validate_frames(
        &self,
        device: &Device<B>,
        opaque: &DeferredFrame<B>,
        overlay: &crate::residency::ForwardFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()> {
        if self.surface.is_some() {
            return Err(Error::InvalidState);
        }
        self.validate_common(device, opaque, overlay, input)?;
        if composite.target_instance() != input.output.instance {
            return Err(Error::InvalidState);
        }
        composite.validate_offscreen(device, opaque.extent, opaque.frame.target_desc.format)?;
        self.bindings(opaque, overlay, input).map(|_| ())
    }

    pub(crate) fn validate_frames_with_image(
        &self,
        device: &Device<B>,
        opaque: &DeferredFrame<B>,
        overlay: &crate::residency::ForwardFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()> {
        let (extent, format) = self.surface.ok_or(Error::InvalidState)?;
        if image.device() != device.id()
            || !image.is_current()
            || image.extent() != extent
            || image.format() != format
        {
            return Err(Error::InvalidState);
        }
        self.validate_common(device, opaque, overlay, input)?;
        if composite.target_instance() != input.output.instance {
            return Err(Error::InvalidState);
        }
        composite.validate_for(device, image)?;
        self.surface_bindings(opaque, overlay, input, image)
            .map(|_| ())
    }

    fn validate_common(
        &self,
        device: &Device<B>,
        opaque: &DeferredFrame<B>,
        overlay: &crate::residency::ForwardFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
    ) -> Result<()> {
        super::validation::validate_frame_inputs(device, opaque)?;
        input.owners.validate(device.id())?;
        if device.id() != self.base.id.device
            || opaque.deformation.is_some()
            || opaque.extent != overlay.extent
            || opaque.extent != input.owners.extent
            || input.opaque_count != 0
            || input.peel_count != self.peel_count
            || input.peel_count == 0
            || input.peel_count > overlay.draws.len()
            || input.rounds.get() != self.rounds
            || self.product.passes.iterations.len() != self.rounds as usize
            || input.output.device != device.id()
            || input.output.texture.device() != device.id()
            || input.output.view.device() != device.id()
            || input.output.desc.dimension != fluxel_rhi::api::TextureDimension::D2
            || input.output.desc.extent != [opaque.extent[0], opaque.extent[1], 1]
            || input.output.desc.array_layers != 1
            || input.output.desc.mip_levels != 1
            || input.output.desc.samples != 1
            || input.output.desc.format != TextureFormat::Rgba8Unorm
            || !input
                .output
                .desc
                .usage
                .contains(TextureUsage::RENDER_ATTACHMENT)
            || !input
                .output
                .desc
                .usage
                .contains(TextureUsage::TEXTURE_BINDING)
            || input.output.view_desc.format != input.output.desc.format
            || input.output.view_desc.dimension != fluxel_rhi::api::TextureViewDimension::D2
            || input.output.view_desc.aspects != TextureAspects::COLOR
            || input.output.view_desc.mips != (0..1)
            || input.output.view_desc.layers != (0..1)
        {
            return Err(Error::InvalidState);
        }
        let depth = overlay.frame.depth.as_ref().ok_or(Error::InvalidState)?;
        if overlay.frame.target_instance != opaque.scene_color.instance
            || !crate::graph::forward::same_texture_owner_desc(
                &overlay.frame.target_desc,
                &opaque.scene_color.desc,
            )
            || overlay.frame.target_incoming != opaque.scene_color.incoming
            || overlay.frame.target_layout != opaque.scene_color.layout
            || depth.instance != opaque.gbuffer[2].instance
            || !crate::graph::forward::same_texture_owner_desc(&depth.desc, &opaque.gbuffer[2].desc)
            || depth.incoming != opaque.gbuffer[2].incoming
            || depth.layout != opaque.gbuffer[2].layout
        {
            return Err(Error::InvalidState);
        }
        if overlay.resident.textures.iter().any(|x| {
            x.instance == opaque.scene_color.instance
                || x.instance == opaque.gbuffer[2].instance
                || x.instance == input.output.instance
        }) {
            return Err(Error::InvalidState);
        }
        let mut ids = vec![
            opaque.scene_color.instance,
            opaque.gbuffer[2].instance,
            input.output.instance,
        ];
        for side in &input.owners.ping_pong {
            ids.extend([
                side.min_max.instance,
                side.front.instance,
                side.back.instance,
            ]);
        }
        ids.push(input.owners.back_accumulator.instance);
        ids.extend(overlay.resident.textures.iter().map(|owner| owner.instance));
        if let Some(owner) = overlay.environment.as_ref() {
            ids.push(owner.texture_instance);
        }
        if let Some(owner) = overlay.specular.as_ref() {
            ids.extend([owner.radiance.instance, owner.brdf_lut.instance]);
        }
        if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
            return Err(Error::InvalidState);
        }
        Ok(())
    }

    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        self.base.submission()
    }
    pub(crate) fn steps(&self) -> impl Iterator<Item = (StepId, GraphPass)> + '_ {
        let id = self.base.id;
        self.base
            .execution
            .logical()
            .passes()
            .iter()
            .copied()
            .enumerate()
            .map(move |(i, p)| {
                (
                    StepId {
                        plan: id,
                        index: u32::try_from(i).expect("graph pass index fits StepId"),
                    },
                    p,
                )
            })
    }
    pub(crate) fn present_step(&self) -> Result<StepId> {
        self.surface.ok_or(Error::InvalidState)?;
        self.base.present_step()
    }
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        self.base.completion(outcome)
    }
    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        self.surface.ok_or(Error::InvalidState)?;
        self.base.surface_completion(outcome)
    }
}
