//! Per-submission plan for the Deferred ordinary-transparent composition.

use super::*;

pub(crate) struct DeferredTransparentPlan<B: ResourceBackend> {
    pub(super) base: DeferredPlan<B>,
    pub(super) overlay: crate::graph::forward::OverlaySlots,
    pub(super) overlay_upload: GraphPass,
    pub(super) overlay_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) overlay_raster: GraphPass,
    pub(super) surface: Option<(TextureSlot, [u32; 2], TextureFormat)>,
}

impl<B: ResourceBackend> DeferredTransparentPlan<B> {
    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: DeferredTransparentDeclaration,
    ) -> Result<Self> {
        let mut composition = Vec::with_capacity(declaration.overlay_mips.len() + 2);
        composition.push(declaration.overlay_upload);
        composition.extend(
            declaration
                .overlay_mips
                .iter()
                .map(|generation| generation.pass),
        );
        composition.push(declaration.overlay_raster);
        let base =
            DeferredPlan::compile_with_composition(device, id, declaration.base, &composition)?;
        let surface = declaration
            .surface
            .map(|(extent, format)| {
                Ok((
                    base.surface_target_slot.ok_or(Error::InvalidState)?,
                    extent,
                    format,
                ))
            })
            .transpose()?;
        Ok(Self {
            base,
            overlay: declaration.overlay,
            overlay_upload: declaration.overlay_upload,
            overlay_mips: declaration.overlay_mips,
            overlay_raster: declaration.overlay_raster,
            surface,
        })
    }

    pub(crate) fn validate_frames(
        &self,
        opaque: &DeferredFrame<B>,
        transparent: &crate::residency::ForwardFrame<B>,
    ) -> Result<()> {
        if self.surface.is_some() {
            return Err(Error::InvalidState);
        }
        super::validation::validate_frame_inputs(&self.base.device, opaque)?;
        if opaque.deformation.is_some()
            || opaque.extent != transparent.extent
            || transparent.frame.target_instance != opaque.scene_color.instance
            || !crate::graph::forward::same_texture_owner_desc(
                &transparent.frame.target_desc,
                &opaque.scene_color.desc,
            )
            || transparent.frame.target_incoming != opaque.scene_color.incoming
            || transparent.frame.target_layout != opaque.scene_color.layout
            || transparent.frame.target.device() != self.base.id.device
            || transparent.frame.target_view.device() != self.base.id.device
            || opaque.scene_color.device != self.base.id.device
        {
            return Err(Error::InvalidState);
        }
        let depth = transparent
            .frame
            .depth
            .as_ref()
            .ok_or(Error::InvalidState)?;
        if depth.instance != opaque.gbuffer[2].instance
            || !crate::graph::forward::same_texture_owner_desc(&depth.desc, &opaque.gbuffer[2].desc)
            || depth.incoming != opaque.gbuffer[2].incoming
            || depth.layout != opaque.gbuffer[2].layout
            || depth.device != opaque.gbuffer[2].device
            || depth.texture.device() != self.base.id.device
            || depth.view.device() != self.base.id.device
        {
            return Err(Error::InvalidState);
        }
        if transparent.resident.textures.iter().any(|texture| {
            texture.instance == opaque.scene_color.instance
                || texture.instance == opaque.gbuffer[2].instance
        }) {
            return Err(Error::InvalidState);
        }
        self.bindings(opaque, transparent).map(|_| ())
    }

    pub(crate) fn validate_frames_with_image(
        &self,
        device: &Device<B>,
        opaque: &DeferredFrame<B>,
        transparent: &crate::residency::ForwardFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()> {
        let (_, extent, format) = self.surface.ok_or(Error::InvalidState)?;
        super::validation::validate_frame_inputs(device, opaque)?;
        let depth = transparent
            .frame
            .depth
            .as_ref()
            .ok_or(Error::InvalidState)?;
        if device.id() != self.base.id.device
            || opaque.deformation.is_some()
            || opaque.extent != transparent.extent
            || opaque.extent != extent
            || transparent.frame.target_instance != opaque.scene_color.instance
            || !crate::graph::forward::same_texture_owner_desc(
                &transparent.frame.target_desc,
                &opaque.scene_color.desc,
            )
            || transparent.frame.target_incoming != opaque.scene_color.incoming
            || transparent.frame.target_layout != opaque.scene_color.layout
            || depth.instance != opaque.gbuffer[2].instance
            || !crate::graph::forward::same_texture_owner_desc(&depth.desc, &opaque.gbuffer[2].desc)
            || depth.incoming != opaque.gbuffer[2].incoming
            || depth.layout != opaque.gbuffer[2].layout
            || image.device() != self.base.id.device
            || !image.is_current()
            || image.extent() != extent
            || image.format() != format
            || composite.target_instance() != opaque.scene_color.instance
        {
            return Err(Error::InvalidState);
        }
        composite.validate_for(device, image)?;
        if transparent.resident.textures.iter().any(|texture| {
            texture.instance == opaque.scene_color.instance
                || texture.instance == opaque.gbuffer[2].instance
        }) {
            return Err(Error::InvalidState);
        }
        self.surface_bindings(opaque, transparent, image)
            .map(|_| ())
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
            .map(move |(index, pass)| {
                (
                    StepId {
                        plan: id,
                        index: u32::try_from(index).expect("graph pass index fits StepId"),
                    },
                    pass,
                )
            })
    }

    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        self.base.completion(outcome)
    }

    pub(crate) fn present_step(&self) -> Result<StepId> {
        self.surface.ok_or(Error::InvalidState)?;
        self.base.present_step()
    }

    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        self.surface.ok_or(Error::InvalidState)?;
        self.base.surface_completion(outcome)
    }
}
