//! Forward attachment admission and allocation before frame assembly.

use super::*;
use crate::residency::{ForwardDepthAttachment, ForwardTexture};

pub(super) struct PreparedAttachments<B: ResourceBackend> {
    pub(super) depth: Option<ForwardDepthAttachment<B>>,
    pub(super) multisample_color: Option<ForwardTexture<B>>,
    pub(super) direct_surface: bool,
    pub(super) composite_surface: bool,
    pub(super) sampled_target: bool,
}

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_forward_attachments(
        &mut self,
        inputs: &[crate::ForwardDrawInput],
        extent: [u32; 2],
        topology: crate::FrameTopology,
        depth_mode: crate::ForwardDepthMode,
        samples: u32,
        color_format: TextureFormat,
        surface: bool,
        sampled_target: bool,
        scene_refraction: bool,
        capture_texture: Option<&super::captured::CaptureTextureOwner<B>>,
        attachments: Option<&BorrowedForwardAttachments<B>>,
    ) -> std::result::Result<PreparedAttachments<B>, RendererError> {
        let process_output = self.output_processing().is_some()
            && color_format == TextureFormat::Rgba16Float
            && attachments.is_none();
        let composite_surface = surface && (process_output || samples > 1);
        let direct_surface = surface && !composite_surface;
        let sampled_target = sampled_target || process_output || composite_surface;
        if samples > 1 && (attachments.is_some() || capture_texture.is_some() || scene_refraction) {
            return Err(Error::InvalidState.into());
        }
        let multisample_desc = if samples > 1 {
            Some(
                admit_multisample(
                    &self.device,
                    extent,
                    color_format,
                    depth_mode,
                    topology,
                    samples,
                    sampled_target,
                )
                .map_err(|error| {
                    self.observe_forward_error("Forward multisample admission", error)
                })?,
            )
        } else {
            None
        };
        if let Some(attachments) = attachments {
            if direct_surface
                || depth_mode != crate::ForwardDepthMode::Depth32LessEqualWrite
                || !matches!(topology, crate::FrameTopology::SortedForward)
                || color_format != attachments.color.desc.format
            {
                return Err(Error::InvalidState.into());
            }
            attachments
                .validate(self.device.id(), extent)
                .map_err(|error| {
                    self.observe_forward_error("borrowed Forward attachments", error)
                })?;
            crate::renderer::require_supported_surface_modes(inputs, topology)
                .map_err(|error| self.observe_forward_error("borrowed surface policy", error))?;
            crate::renderer::require_blendable_target(
                &self.device,
                &attachments.color.desc,
                topology,
            )
            .map_err(|error| self.observe_forward_error("borrowed target admission", error))?;
        } else if multisample_desc.is_none() {
            super::target_admission::admit_assembly(
                &self.device,
                inputs,
                topology,
                extent,
                color_format,
                direct_surface,
                sampled_target,
            )
            .map_err(|error| self.observe_forward_error("surface target admission", error))?;
        } else {
            crate::renderer::require_supported_surface_modes(inputs, topology)
                .map_err(|error| self.observe_forward_error("multisample surface policy", error))?;
        }

        let depth = if attachments.is_some() {
            None
        } else if scene_refraction {
            Some(
                capture_texture
                    .and_then(|capture| capture.refraction_depth.as_ref())
                    .ok_or(Error::InvalidState)?
                    .retain(),
            )
        } else {
            super::depth::allocate_depth(&self.device, depth_mode, topology, extent, samples)?
        };
        let multisample_color = multisample_desc
            .map(|desc| allocate_multisample(&self.device, desc))
            .transpose()
            .map_err(|error| self.observe_forward_error("Forward multisample allocation", error))?;
        Ok(PreparedAttachments {
            depth,
            multisample_color,
            direct_surface,
            composite_surface,
            sampled_target,
        })
    }
}

fn admit_multisample<B: ResourceBackend>(
    device: &Device<B>,
    extent: [u32; 2],
    format: TextureFormat,
    depth_mode: crate::ForwardDepthMode,
    topology: crate::FrameTopology,
    samples: u32,
    sampled_target: bool,
) -> Result<TextureDesc> {
    let mut color = TextureDesc {
        label: Some("fluxel forward multisample color".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: Vec::new(),
    };
    match device.texture_support(&color) {
        Support::Supported(facts)
            if facts.sample_counts.contains(&samples)
                && (topology != crate::FrameTopology::SortedForward || facts.blendable) => {}
        Support::Supported(_) => {
            return Err(Error::Unsupported("Forward multisample color attachment"));
        }
        Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
    }
    if depth_mode != crate::ForwardDepthMode::Disabled {
        let depth = ForwardDepthAttachment::<B>::descriptor(extent, samples);
        match device.texture_support(&depth) {
            Support::Supported(facts) if facts.sample_counts.contains(&samples) => {}
            Support::Supported(_) => {
                return Err(Error::Unsupported("Forward multisample depth attachment"));
            }
            Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
        }
    }

    let raster = color.clone();
    color.samples = 1;
    color.usage |= TextureUsage::COPY_SRC;
    if sampled_target {
        color.usage |= TextureUsage::TEXTURE_BINDING;
    }
    match device.texture_support(&color) {
        Support::Supported(facts) if !sampled_target || facts.filterable => Ok(raster),
        Support::Supported(_) => Err(Error::Unsupported("sampleable Forward resolve target")),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

fn allocate_multisample<B: ResourceBackend>(
    device: &Device<B>,
    desc: TextureDesc,
) -> Result<ForwardTexture<B>> {
    let texture = device.create_texture(&desc)?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format: desc.format,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok(ForwardTexture {
        texture,
        view,
        view_dimension: TextureViewDimension::D2,
        desc,
        instance: crate::graph::ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
        incoming: TextureUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        },
        layout: ImageLayout::Undefined,
        device: device.id(),
        filterable: false,
    })
}
