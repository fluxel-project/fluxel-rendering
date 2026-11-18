//! Deferred lighting followed by one real dual-depth-peeling product.

use super::*;
use crate::{
    graph::{
        forward,
        oit::{declare_dual_depth_peeling_rounds, DualDepthPeelInputs, OitPeelSide},
        oit_record::OitRecordSlots,
    },
    residency::{ForwardFrame, OitTextureOwner},
};

pub(crate) struct DeferredOitDeclaration {
    pub(super) base: DeferredDeclaration,
    pub(super) overlay: forward::OverlaySlots,
    pub(super) overlay_upload: GraphPass,
    pub(super) overlay_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) product: crate::graph::oit::DualDepthPeelGraph,
    pub(super) slots: OitRecordSlots,
    pub(super) fallback: Option<GraphPass>,
    pub(super) final_slot: TextureSlot,
    pub(super) surface: Option<([u32; 2], TextureFormat)>,
    pub(super) peel_count: usize,
    pub(super) rounds: u32,
}

/// Reusable dual-depth-peeling contribution following a Deferred lighting prefix.
pub(super) struct DeferredOitComposition {
    pub(super) overlay: forward::OverlaySlots,
    pub(super) overlay_upload: GraphPass,
    pub(super) overlay_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) product: crate::graph::oit::DualDepthPeelGraph,
    pub(super) slots: OitRecordSlots,
    pub(super) fallback: Option<GraphPass>,
    pub(super) final_source: crate::graph::definition::TextureHandle,
    pub(super) final_slot: TextureSlot,
    pub(super) peel_count: usize,
    pub(super) rounds: u32,
}

impl<B: ResourceBackend> DeferredOitPlan<B> {
    pub(crate) fn declare<F>(
        device: &Device<B>,
        opaque: &DeferredFrame<B>,
        overlay: &ForwardFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
        contribute: F,
    ) -> Result<DeferredOitDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        let prefix = DeferredPlan::declare_prefix(
            device,
            super::record_view::DeferredRecordView::offscreen(opaque),
            contribute,
        )?;
        Self::declare_from_prefix(device, prefix, opaque, overlay, input, None)
    }

    pub(crate) fn declare_surface<F>(
        device: &Device<B>,
        opaque: &DeferredFrame<B>,
        overlay: &ForwardFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        contribute: F,
    ) -> Result<DeferredOitDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        super::validation::validate_frame_inputs(device, opaque)?;
        if image.device() != device.id() {
            return Err(Error::WrongDevice);
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != opaque.extent {
            return Err(Error::InvalidRange);
        }
        let mut prefix = DeferredPlan::declare_prefix(
            device,
            super::record_view::DeferredRecordView::offscreen(opaque),
            contribute,
        )?;
        let desc = fluxel_rhi::api::TextureDesc {
            label: Some("fluxel deferred oit surface lease".into()),
            dimension: fluxel_rhi::api::TextureDimension::D2,
            extent: [image.extent()[0], image.extent()[1], 1],
            array_layers: 1,
            format: image.format(),
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: Vec::new(),
        };
        prefix.target = None;
        prefix.readback = None;
        prefix.row_pitch = None;
        prefix.surface_target = Some(
            prefix
                .graph
                .import_surface_texture(desc)
                .map_err(|_| Error::InvalidState)?,
        );
        Self::declare_from_prefix(
            device,
            prefix,
            opaque,
            overlay,
            input,
            Some((image.extent(), image.format())),
        )
    }

    fn declare_from_prefix(
        device: &Device<B>,
        mut prefix: super::prefix::DeferredGraphPrefix,
        opaque: &DeferredFrame<B>,
        overlay: &ForwardFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
        surface: Option<([u32; 2], TextureFormat)>,
    ) -> Result<DeferredOitDeclaration> {
        let composition = declare_oit_composition(device, &mut prefix, opaque, overlay, input)?;
        let base = super::terminal::declare_terminal(
            prefix,
            Some((
                composition.final_source,
                composition
                    .fallback
                    .unwrap_or(composition.product.passes.final_composite),
            )),
        )?;
        Ok(DeferredOitDeclaration {
            base,
            overlay: composition.overlay,
            overlay_upload: composition.overlay_upload,
            overlay_mips: composition.overlay_mips,
            product: composition.product,
            slots: composition.slots,
            fallback: composition.fallback,
            final_slot: composition.final_slot,
            surface,
            peel_count: composition.peel_count,
            rounds: composition.rounds,
        })
    }
}

pub(super) fn declare_oit_composition<B: ResourceBackend>(
    device: &Device<B>,
    prefix: &mut super::prefix::DeferredGraphPrefix,
    opaque: &DeferredFrame<B>,
    overlay: &ForwardFrame<B>,
    input: &crate::graph::ForwardOitInput<'_, B>,
) -> Result<DeferredOitComposition> {
    declare_oit_composition_with_shared(device, prefix, opaque, overlay, input, None)
}

pub(super) fn declare_oit_composition_with_shared<B: ResourceBackend>(
    device: &Device<B>,
    prefix: &mut super::prefix::DeferredGraphPrefix,
    opaque: &DeferredFrame<B>,
    overlay: &ForwardFrame<B>,
    input: &crate::graph::ForwardOitInput<'_, B>,
    shared: Option<&mut crate::graph::forward::SharedResidentImports>,
) -> Result<DeferredOitComposition> {
    let mut contribute =
        |context: &mut crate::graph::ForwardOitReadContext<'_, '_>| context.all_reads();
    let mut contribution_error = None;
    declare_oit_composition_with_shared_and_reads(
        device,
        prefix,
        opaque,
        overlay,
        input,
        shared,
        &mut contribute,
        &mut contribution_error,
    )
}

/// Variant whose material reads are declared inside each real peel or sorted
/// fallback pass.  The context receives the pass's actual ordered draw range.
pub(super) fn declare_oit_composition_with_shared_and_reads<B, C>(
    device: &Device<B>,
    prefix: &mut super::prefix::DeferredGraphPrefix,
    opaque: &DeferredFrame<B>,
    overlay: &ForwardFrame<B>,
    input: &crate::graph::ForwardOitInput<'_, B>,
    shared: Option<&mut crate::graph::forward::SharedResidentImports>,
    contribute: &mut C,
    contribution_error: &mut Option<crate::PipelineError>,
) -> Result<DeferredOitComposition>
where
    B: ResourceBackend,
    C: FnMut(
        &mut crate::graph::ForwardOitReadContext<'_, '_>,
    ) -> std::result::Result<(), crate::PipelineError>,
{
    if opaque.deformation.is_some() {
        return Err(Error::Unsupported("deferred OIT deformation"));
    }
    if input.opaque_count != 0 || input.peel_count == 0 || input.peel_count > overlay.draws.len() {
        return Err(Error::InvalidState);
    }
    input.owners.validate(device.id())?;
    if input.owners.extent != opaque.extent || input.output.device != device.id() {
        return Err(Error::InvalidState);
    }
    let overlay_prefix = match shared {
        Some(shared) => forward::declare_overlay_prefix_with_shared(
            &mut prefix.graph,
            overlay,
            prefix.scene_color,
            prefix.gbuffer[2],
            prefix.uniform_alignment,
            shared,
        )?,
        None => forward::declare_overlay_prefix(
            &mut prefix.graph,
            overlay,
            prefix.scene_color,
            prefix.gbuffer[2],
            prefix.uniform_alignment,
        )?,
    };
    let dependency = overlay_prefix
        .mip_generation
        .last()
        .map(|item| item.pass)
        .unwrap_or(overlay_prefix.upload);
    let import = |graph: &mut GraphBuilder, owner: &OitTextureOwner<B>| {
        graph
            .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
            .map_err(|_| Error::InvalidState)
    };
    let inputs = DualDepthPeelInputs {
        ping_pong: [
            OitPeelSide {
                min_max: import(&mut prefix.graph, &input.owners.ping_pong[0].min_max)?,
                front: import(&mut prefix.graph, &input.owners.ping_pong[0].front)?,
                back: import(&mut prefix.graph, &input.owners.ping_pong[0].back)?,
            },
            OitPeelSide {
                min_max: import(&mut prefix.graph, &input.owners.ping_pong[1].min_max)?,
                front: import(&mut prefix.graph, &input.owners.ping_pong[1].front)?,
                back: import(&mut prefix.graph, &input.owners.ping_pong[1].back)?,
            },
        ],
        back_accumulator: import(&mut prefix.graph, &input.owners.back_accumulator)?,
        opaque_color: prefix.scene_color,
        opaque_depth: prefix.gbuffer[2],
        output: import(&mut prefix.graph, input.output)?,
    };
    let slots = OitRecordSlots::from(inputs);
    let mut first = true;
    let product =
        declare_dual_depth_peeling_rounds(&mut prefix.graph, inputs, input.rounds, |_, pass| {
            if first {
                pass.depends_on(prefix.lighting_pass)?;
                pass.depends_on(dependency)?;
                first = false;
            }
            overlay_prefix.declare_draw_reads_with(
                pass,
                0..input.peel_count,
                contribute,
                contribution_error,
            )
        })
        .map_err(|_| Error::InvalidState)?;
    if contribution_error.is_some() {
        return Err(Error::InvalidState);
    }
    let mut final_source = product.output;
    let mut fallback = None;
    if input.peel_count < overlay.draws.len() {
        let (pass, output) = prefix
            .graph
            .add_pass("deferred-oit-sorted-fallback", |pass| {
                pass.depends_on(product.passes.final_composite)?;
                overlay_prefix.declare_draw_reads_with(
                    pass,
                    input.peel_count..overlay.draws.len(),
                    contribute,
                    contribution_error,
                )?;
                pass.read_texture_with_use(
                    prefix.gbuffer[2],
                    whole_depth(),
                    texture_contract(
                        TextureUses {
                            stages: StageMask::FRAGMENT,
                            accesses: AccessMask::DEPTH_READ,
                        },
                        ImageLayout::DepthStencilAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .uses,
                    ImageLayout::DepthStencilAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )?;
                pass.write_texture_with_use(
                    product.output,
                    whole_color(),
                    texture_contract(
                        TextureUses {
                            stages: StageMask::COLOR_OUTPUT,
                            accesses: AccessMask::COLOR_READ | AccessMask::COLOR_WRITE,
                        },
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .uses,
                    ImageLayout::ColorAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )
            })
            .map_err(|_| Error::InvalidState)?;
        if contribution_error.is_some() {
            return Err(Error::InvalidState);
        }
        fallback = Some(pass);
        final_source = output;
    }
    let final_slot = TextureSlot::from(final_source);
    Ok(DeferredOitComposition {
        overlay: overlay_prefix.slots,
        overlay_upload: overlay_prefix.upload,
        overlay_mips: overlay_prefix.mip_generation,
        product,
        slots,
        fallback,
        final_slot,
        final_source,
        peel_count: input.peel_count,
        rounds: input.rounds.get(),
    })
}
