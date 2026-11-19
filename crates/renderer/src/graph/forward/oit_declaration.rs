//! Opaque, peeling, excluded transparency and readback in one Forward Graph.

use super::*;
use super::{oit_draw_reads::OitDrawReads, oit_plan::ForwardOitPlan};

pub(crate) struct OffscreenForwardOitDeclaration {
    pub(super) base: OffscreenForwardDeclaration,
    pub(super) oit: ForwardOitPlan,
}

pub(super) fn declare_with_contribution<B, C>(
    device: &Device<B>,
    frame: &ForwardFrame<B>,
    input: &ForwardOitInput<'_, B>,
    contribute: C,
) -> std::result::Result<OffscreenForwardOitDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    declare_inner(device, frame, input, None, false, contribute)
}

pub(super) fn declare_surface_with_contribution<B, C>(
    device: &Device<B>,
    frame: &ForwardFrame<B>,
    input: &ForwardOitInput<'_, B>,
    image: &SurfaceImage<B>,
    contribute: C,
) -> std::result::Result<OffscreenForwardOitDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    if image.device() != device.id() {
        return Err(Error::WrongDevice.into());
    }
    if !image.is_current() {
        return Err(Error::SurfaceOutdated.into());
    }
    if image.extent() != frame.extent {
        return Err(Error::InvalidRange.into());
    }
    declare_inner(device, frame, input, Some(image), false, contribute)
}

pub(super) fn declare_hdr_with_contribution<B, C>(
    device: &Device<B>,
    frame: &ForwardFrame<B>,
    input: &ForwardOitInput<'_, B>,
    contribute: C,
) -> std::result::Result<OffscreenForwardOitDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    if frame.frame.target_desc.format != TextureFormat::Rgba16Float {
        return Err(Error::InvalidState.into());
    }
    declare_inner(device, frame, input, None, true, contribute)
}

fn declare_inner<B, C>(
    device: &Device<B>,
    frame: &ForwardFrame<B>,
    input: &ForwardOitInput<'_, B>,
    image: Option<&SurfaceImage<B>>,
    hdr_readback: bool,
    mut contribute: C,
) -> std::result::Result<OffscreenForwardOitDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    validate_resident_texture_devices(&frame.resident, device.id())?;
    if frame.frame.multisample_color.is_some() {
        // OIT has its own accumulation attachments.  It cannot silently bind
        // the ordinary Forward MSAA raster owner as though it were that
        // product's resolve attachment.
        return Err(Error::InvalidState.into());
    }
    input.owners.validate(device.id())?;
    let peel_end = input
        .opaque_count
        .checked_add(input.peel_count)
        .ok_or(Error::InvalidRange)?;
    if input.peel_count == 0
        || peel_end > frame.draws.len()
        || input.owners.extent != frame.extent
        || input.output.device != device.id()
        || input.output.desc.format != frame.frame.target_desc.format
    {
        return Err(Error::InvalidState.into());
    }
    let spec = ForwardGraphSpec::from_frame(frame)?;
    let mut prefix =
        super::offscreen_prefix::declare_prefix(device, &spec, FrameTopology::SortedForward)?;
    let reads = OitDrawReads::new(&prefix, &spec);
    let dependency = prefix
        .mip_passes
        .last()
        .map_or(prefix.upload_pass, |pass| pass.pass);
    let stages = super::oit_stages::declare_stages(
        &mut prefix.graph,
        &spec,
        &reads,
        input,
        dependency,
        prefix.target,
        prefix.depth.ok_or(Error::InvalidState)?,
        &mut contribute,
    )?;
    let super::oit_stages::OitStageDeclaration {
        raster_pass,
        opaque_color,
        opaque_depth,
        output,
        mut oit,
    } = stages;
    let (readback_pass, surface, hdr_terminal) = if let Some(image) = image {
        let surface_target = prefix
            .graph
            .import_surface_texture(TextureDesc {
                label: Some("fluxel forward OIT surface lease".into()),
                dimension: fluxel_rhi::api::TextureDimension::D2,
                extent: [image.extent()[0], image.extent()[1], 1],
                array_layers: 1,
                format: image.format(),
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::RENDER_ATTACHMENT,
                view_formats: Vec::new(),
            })
            .map_err(|_| Error::InvalidState)?;
        let (terminal, ()) = prefix
            .graph
            .add_pass("forward-oit-surface-composite", |pass| {
                pass.read_texture_with_use(
                    output,
                    whole(),
                    texture_read_uses().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
                pass.write_texture_with_use(
                    surface_target,
                    whole(),
                    color_write_uses().uses,
                    ImageLayout::ColorAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )?;
                pass.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        (
            terminal,
            Some(super::oit_plan::OitSurfaceTarget {
                slot: surface_target.into(),
                extent: frame.extent,
                format: image.format(),
            }),
            None,
        )
    } else if hdr_readback {
        let display = frame.frame.output.as_ref().ok_or(Error::InvalidState)?;
        let terminal = super::hdr_output::declare_hdr_readback(
            &mut prefix.graph,
            output,
            display,
            prefix.readback,
            spec.extent,
            spec.readback_row_pitch,
        )?;
        (terminal.readback_pass, None, Some(terminal))
    } else {
        let readback_len = u64::from(spec.readback_row_pitch)
            .checked_mul(u64::from(spec.extent[1]))
            .ok_or(Error::InvalidRange)?;
        let (readback, ()) = prefix
            .graph
            .add_pass("forward-readback", |pass| {
                pass.read_texture_with_use(
                    output,
                    whole(),
                    texture_copy_read_uses().uses,
                    ImageLayout::CopySrc,
                    TextureUsage::COPY_SRC,
                )?;
                pass.write_with_use(
                    prefix.readback,
                    0..readback_len,
                    copy_write_uses().uses,
                    BufferUsage::COPY_DST,
                )?;
                pass.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        (readback, None, None)
    };
    oit.surface = surface;
    oit.hdr_readback = hdr_terminal;
    Ok(OffscreenForwardOitDeclaration {
        oit,
        base: OffscreenForwardDeclaration {
            graph: prefix.graph,
            lane: prefix.lane,
            upload_pass: prefix.upload_pass,
            mip_passes: prefix.mip_passes,
            indirect: Vec::new(),
            raster_pass,
            readback_pass,
            resident_upload_slot: prefix.resident_upload.into(),
            frame_upload_slot: prefix.frame_upload.into(),
            uniform_slot: prefix.uniform.into(),
            environment_upload_slot: prefix.environment_upload.map(Into::into),
            specular_upload_slot: prefix.specular_upload.map(Into::into),
            specular_view_uniform_slot: prefix.specular_view_uniform.map(Into::into),
            stream_slots: prefix.streams.into_iter().map(Into::into).collect(),
            texture_slots: prefix.textures.into_iter().map(Into::into).collect(),
            environment_slot: prefix.environment.map(Into::into),
            specular_radiance_slot: prefix.specular_radiance.map(Into::into),
            specular_brdf_lut_slot: prefix.specular_brdf_lut.map(Into::into),
            depth_slot: Some(opaque_depth.into()),
            target_slot: opaque_color.into(),
            multisample_color_slot: None,
            readback_slot: prefix.readback.into(),
            draw_count: spec.draws.len(),
            stream_count: spec.streams.len(),
            texture_count: spec.textures.len(),
            extent: spec.extent,
            shape: spec.shape.clone(),
            uniform_alignment: prefix.uniform_alignment,
            topology: FrameTopology::SortedForward,
            hdr_terminal: None,
        },
    })
}
