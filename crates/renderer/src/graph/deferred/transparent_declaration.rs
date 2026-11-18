//! One Deferred opaque prefix followed by an ordinary Forward transparent overlay.

use super::*;
use crate::{
    graph::forward::{self, OverlaySlots},
    residency::ForwardFrame,
};

pub(crate) struct DeferredTransparentDeclaration {
    pub(super) base: DeferredDeclaration,
    pub(super) overlay: OverlaySlots,
    pub(super) overlay_upload: GraphPass,
    pub(super) overlay_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) overlay_raster: GraphPass,
    pub(super) surface: Option<([u32; 2], TextureFormat)>,
}

/// The reusable Graph portion of an ordinary Forward overlay.
///
/// It deliberately retains the version written by the overlay: capture
/// consumers must alias that version, rather than the Deferred lighting input.
pub(super) struct DeferredTransparentComposition {
    pub(super) overlay: OverlaySlots,
    pub(super) overlay_upload: GraphPass,
    pub(super) overlay_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) overlay_raster: GraphPass,
    pub(super) final_source: crate::graph::definition::TextureHandle,
}

impl<B: ResourceBackend> DeferredTransparentPlan<B> {
    pub(crate) fn declare<F>(
        device: &Device<B>,
        opaque: &DeferredFrame<B>,
        transparent: &ForwardFrame<B>,
        contribute: F,
    ) -> Result<DeferredTransparentDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        if opaque.deformation.is_some() {
            return Err(Error::Unsupported("deferred transparent deformation"));
        }
        let prefix = DeferredPlan::declare_prefix(
            device,
            super::record_view::DeferredRecordView::offscreen(opaque),
            contribute,
        )?;
        Self::declare_from_prefix(prefix, transparent, None)
    }

    pub(crate) fn declare_surface<F>(
        device: &Device<B>,
        opaque: &DeferredFrame<B>,
        transparent: &ForwardFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        contribute: F,
    ) -> Result<DeferredTransparentDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        if opaque.deformation.is_some() {
            return Err(Error::Unsupported("deferred transparent deformation"));
        }
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
        let surface_desc = fluxel_rhi::api::TextureDesc {
            label: Some("fluxel deferred surface lease".into()),
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
                .import_surface_texture(surface_desc)
                .map_err(|_| Error::InvalidState)?,
        );
        Self::declare_from_prefix(prefix, transparent, Some((image.extent(), image.format())))
    }

    fn declare_from_prefix(
        mut prefix: super::prefix::DeferredGraphPrefix,
        transparent: &ForwardFrame<B>,
        surface: Option<([u32; 2], TextureFormat)>,
    ) -> Result<DeferredTransparentDeclaration> {
        let composition = declare_transparent_composition(&mut prefix, transparent)?;
        let base = super::terminal::declare_terminal(
            prefix,
            Some((composition.final_source, composition.overlay_raster)),
        )?;
        Ok(DeferredTransparentDeclaration {
            base,
            overlay: composition.overlay,
            overlay_upload: composition.overlay_upload,
            overlay_mips: composition.overlay_mips,
            overlay_raster: composition.overlay_raster,
            surface,
        })
    }
}

pub(super) fn declare_transparent_composition<B: ResourceBackend>(
    prefix: &mut super::prefix::DeferredGraphPrefix,
    transparent: &ForwardFrame<B>,
) -> Result<DeferredTransparentComposition> {
    declare_transparent_composition_with_shared(prefix, transparent, None)
}

/// Variant for a larger single Graph which shares completed Forward residents
/// with a later namespace.
pub(super) fn declare_transparent_composition_with_shared<B: ResourceBackend>(
    prefix: &mut super::prefix::DeferredGraphPrefix,
    transparent: &ForwardFrame<B>,
    shared: Option<&mut crate::graph::forward::SharedResidentImports>,
) -> Result<DeferredTransparentComposition> {
    if transparent.extent != prefix.extent {
        return Err(Error::InvalidState);
    }
    let overlay_prefix = match shared {
        Some(shared) => forward::declare_overlay_prefix_with_shared(
            &mut prefix.graph,
            transparent,
            prefix.scene_color,
            prefix.gbuffer[2],
            prefix.uniform_alignment,
            shared,
        )?,
        None => forward::declare_overlay_prefix(
            &mut prefix.graph,
            transparent,
            prefix.scene_color,
            prefix.gbuffer[2],
            prefix.uniform_alignment,
        )?,
    };
    let overlay_dependency = overlay_prefix
        .mip_generation
        .last()
        .map(|generation| generation.pass)
        .unwrap_or(overlay_prefix.upload);
    let overlay_writes_depth = overlay_prefix.slots.writes_depth();
    let (overlay_raster, (scene_color, depth)) = prefix
        .graph
        .add_pass("deferred-transparent-overlay", |pass| {
            pass.depends_on(prefix.lighting_pass)?;
            pass.depends_on(overlay_dependency)?;
            overlay_prefix.declare_reads(pass)?;
            let depth = if overlay_writes_depth {
                pass.write_texture_with_use(
                    prefix.gbuffer[2],
                    whole_depth(),
                    texture_contract(
                        TextureUses {
                            stages: StageMask::FRAGMENT,
                            accesses: AccessMask::DEPTH_READ | AccessMask::DEPTH_WRITE,
                        },
                        ImageLayout::DepthStencilAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .uses,
                    ImageLayout::DepthStencilAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )?
            } else {
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
                prefix.gbuffer[2]
            };
            let scene_color = pass.write_texture_with_use(
                prefix.scene_color,
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
            )?;
            Ok((scene_color, depth))
        })
        .map_err(|_| Error::InvalidState)?;
    prefix.gbuffer[2] = depth;
    Ok(DeferredTransparentComposition {
        overlay: overlay_prefix.slots,
        overlay_upload: overlay_prefix.upload,
        overlay_mips: overlay_prefix.mip_generation,
        overlay_raster,
        final_source: scene_color,
    })
}
