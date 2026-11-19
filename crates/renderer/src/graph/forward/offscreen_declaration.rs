//! Complete private offscreen Forward graph declarations.

use fluxel_rhi::{
    api::{
        BufferUsage, Device, Error, ImageLayout, LaneId, SurfaceImage, TextureDesc,
        TextureDimension, TextureFormat, TextureUsage, TextureUses,
    },
    backend::ResourceBackend,
};

use crate::{
    FrameTopology,
    graph::definition::{BufferHandle, TextureHandle},
    pipeline::{ForwardPhaseContext, PipelineError},
    residency::ForwardFrame,
};

use super::{
    BufferSlot, ForwardGraphSpec, ForwardPhaseDriver, ForwardPlanBuildError, ForwardShape,
    GraphBuilder, GraphPass, TextureSlot, copy_write_uses, texture_copy_read_uses,
    validate_resident_texture_devices, whole,
};

/// A complete, uncompiled offscreen Forward declaration.
pub(crate) struct OffscreenForwardDeclaration {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) upload_pass: GraphPass,
    pub(super) mip_passes: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) indirect: Vec<super::ForwardIndirectGraph>,
    pub(super) particles: Vec<super::ForwardParticleGraph>,
    pub(super) raster_pass: GraphPass,
    pub(super) readback_pass: GraphPass,
    pub(super) resident_upload_slot: BufferSlot,
    pub(super) frame_upload_slot: BufferSlot,
    pub(super) uniform_slot: BufferSlot,
    pub(super) environment_upload_slot: Option<BufferSlot>,
    pub(super) specular_upload_slot: Option<BufferSlot>,
    pub(super) specular_view_uniform_slot: Option<BufferSlot>,
    pub(super) stream_slots: Vec<BufferSlot>,
    pub(super) texture_slots: Vec<TextureSlot>,
    pub(super) environment_slot: Option<TextureSlot>,
    pub(super) specular_radiance_slot: Option<TextureSlot>,
    pub(super) specular_brdf_lut_slot: Option<TextureSlot>,
    pub(super) depth_slot: Option<TextureSlot>,
    pub(super) target_slot: TextureSlot,
    pub(super) multisample_color_slot: Option<TextureSlot>,
    pub(super) readback_slot: BufferSlot,
    pub(super) draw_count: usize,
    pub(super) stream_count: usize,
    pub(super) texture_count: usize,
    pub(super) extent: [u32; 2],
    pub(super) shape: ForwardShape,
    pub(super) uniform_alignment: u64,
    pub(super) topology: FrameTopology,
    pub(super) hdr_terminal: Option<HdrForwardTerminal>,
}

/// The last two Forward HDR operations are declared with the real scene and
/// display owners.  Native recording consumes the selected terminal only after
/// ordinary Forward raster work has completed.
pub(super) enum HdrForwardTerminal {
    Offscreen {
        composite_pass: GraphPass,
        output_slot: TextureSlot,
        readback_pass: GraphPass,
    },
    Surface {
        composite_pass: GraphPass,
        surface_slot: TextureSlot,
        extent: [u32; 2],
        format: TextureFormat,
    },
}

enum ForwardOutput<'a> {
    Readback,
    ProcessedOffscreen {
        desc: &'a TextureDesc,
        incoming: TextureUses,
        layout: ImageLayout,
    },
    ProcessedSurface {
        format: TextureFormat,
    },
}

pub(super) fn declare_offscreen_with<B, F>(
    device: &Device<B>,
    frame: &ForwardFrame<B>,
    expected: FrameTopology,
    contribute: F,
) -> std::result::Result<OffscreenForwardDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    F: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    validate_resident_texture_devices(&frame.resident, device.id())?;
    crate::residency::validate_particle_draws(&frame.draws, &frame.particles, &frame.indirect)?;
    let spec = ForwardGraphSpec::from_frame(frame)?;
    declare_offscreen_with_output(
        device,
        &spec,
        expected,
        ForwardOutput::Readback,
        &frame.indirect,
        &frame.particles,
        contribute,
    )
}

fn declare_offscreen_with_output<B, F>(
    device: &Device<B>,
    spec: &ForwardGraphSpec,
    expected: FrameTopology,
    output: ForwardOutput<'_>,
    indirect_products: &[(
        usize,
        crate::renderer::indirect_producer::IndirectProduct<B>,
    )],
    particles: &[(usize, crate::renderer::particle_update::ParticleStep<B>)],
    contribute: F,
) -> std::result::Result<OffscreenForwardDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    F: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    let super::offscreen_prefix::ForwardDeclarationPrefix {
        mut graph,
        lane,
        upload_pass,
        mip_passes,
        uniform_alignment,
        resident_upload,
        frame_upload,
        uniform,
        readback,
        environment_upload,
        specular_upload,
        specular_view_uniform,
        streams,
        textures,
        environment,
        specular_radiance,
        specular_brdf_lut,
        depth,
        target,
        multisample_color,
    } = super::offscreen_prefix::declare_prefix(device, spec, expected)?;
    let raster_dependency = mip_passes.last().map_or(upload_pass, |pass| pass.pass);
    let indirect = super::declare_indirect(&mut graph, indirect_products)?;
    let particles = super::declare_particles(&mut graph, particles)?;
    let (raster_pass, target, _) = {
        let mut driver = ForwardPhaseDriver {
            graph: Some(&mut graph),
            marker: std::marker::PhantomData,
            expected,
            upload: Some(raster_dependency),
            streams: streams.clone(),
            uniform: Some(uniform),
            textures: textures.clone(),
            environment,
            specular_radiance,
            specular_brdf_lut,
            specular_view_uniform,
            specular_mip_levels: spec
                .specular_radiance
                .as_ref()
                .map(|value| value.desc.mip_levels),
            target: multisample_color.unwrap_or(target),
            resolve_target: multisample_color.map(|_| target),
            depth,
            draws: spec.draws.clone(),
            indirect: indirect.clone(),
            particles: particles.clone(),
            declared: false,
            output: None,
        };
        let mut context = ForwardPhaseContext::new(&mut driver);
        contribute(&mut context).map_err(ForwardPlanBuildError::Pipeline)?;
        driver
            .finish()
            .map_err(|_| ForwardPlanBuildError::Pipeline(PipelineError::InvalidPhaseContribution))?
    };
    let (readback_pass, hdr_terminal) = match output {
        ForwardOutput::Readback => (declare_readback(&mut graph, spec, target, readback)?, None),
        ForwardOutput::ProcessedOffscreen {
            desc,
            incoming,
            layout,
        } => {
            let output = graph
                .import_texture_with_state(desc.clone(), incoming, layout)
                .map_err(|_| Error::InvalidState)?;
            let (composite_pass, output) =
                declare_output_composite(&mut graph, target, output, false)?;
            let readback_pass = declare_readback(&mut graph, spec, output, readback)?;
            (
                readback_pass,
                Some(HdrForwardTerminal::Offscreen {
                    composite_pass,
                    output_slot: TextureSlot::from(output),
                    readback_pass,
                }),
            )
        }
        ForwardOutput::ProcessedSurface { format } => {
            let extent = spec.extent;
            let output = graph
                .import_surface_texture(TextureDesc {
                    label: Some("fluxel forward HDR surface lease".into()),
                    dimension: TextureDimension::D2,
                    extent: [extent[0], extent[1], 1],
                    array_layers: 1,
                    format,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsage::RENDER_ATTACHMENT,
                    view_formats: Vec::new(),
                })
                .map_err(|_| Error::InvalidState)?;
            let (composite_pass, _) = declare_output_composite(&mut graph, target, output, true)?;
            (
                composite_pass,
                Some(HdrForwardTerminal::Surface {
                    composite_pass,
                    surface_slot: TextureSlot::from(output),
                    extent,
                    format,
                }),
            )
        }
    };
    Ok(OffscreenForwardDeclaration {
        graph,
        lane,
        upload_pass,
        mip_passes,
        indirect,
        particles,
        raster_pass,
        readback_pass,
        resident_upload_slot: BufferSlot::from(resident_upload),
        frame_upload_slot: BufferSlot::from(frame_upload),
        uniform_slot: BufferSlot::from(uniform),
        environment_upload_slot: environment_upload.map(BufferSlot::from),
        specular_upload_slot: specular_upload.map(BufferSlot::from),
        specular_view_uniform_slot: specular_view_uniform.map(BufferSlot::from),
        stream_slots: streams.into_iter().map(BufferSlot::from).collect(),
        texture_slots: textures.into_iter().map(TextureSlot::from).collect(),
        environment_slot: environment.map(TextureSlot::from),
        specular_radiance_slot: specular_radiance.map(TextureSlot::from),
        specular_brdf_lut_slot: specular_brdf_lut.map(TextureSlot::from),
        depth_slot: depth.map(TextureSlot::from),
        target_slot: TextureSlot::from(target),
        multisample_color_slot: multisample_color.map(TextureSlot::from),
        readback_slot: BufferSlot::from(readback),
        draw_count: spec.draws.len(),
        stream_count: spec.streams.len(),
        texture_count: spec.textures.len(),
        extent: spec.extent,
        shape: spec.shape.clone(),
        uniform_alignment,
        topology: expected,
        hdr_terminal,
    })
}

/// Declares a linear HDR Forward scene followed by a fullscreen terminal.
pub(super) fn declare_offscreen_hdr_with<B, F>(
    device: &Device<B>,
    frame: &ForwardFrame<B>,
    expected: FrameTopology,
    image: Option<&SurfaceImage<B>>,
    contribute: F,
) -> std::result::Result<OffscreenForwardDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    F: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    validate_resident_texture_devices(&frame.resident, device.id())?;
    crate::residency::validate_particle_draws(&frame.draws, &frame.particles, &frame.indirect)?;
    let standard_msaa_surface = frame.frame.multisample_color.is_some()
        && image.is_some()
        && frame.frame.output.is_none()
        && frame.frame.target_desc.format == TextureFormat::Rgba8Unorm;
    if frame.frame.target_desc.format != TextureFormat::Rgba16Float && !standard_msaa_surface {
        return Err(Error::InvalidState.into());
    }
    if image.is_some_and(|image| {
        image.device() != device.id() || !image.is_current() || image.extent() != frame.extent
    }) {
        return Err(Error::InvalidState.into());
    }
    let spec = ForwardGraphSpec::from_frame(frame)?;
    let output = match image {
        Some(image) => ForwardOutput::ProcessedSurface {
            format: image.format(),
        },
        None => {
            if frame.frame.target_desc.format != TextureFormat::Rgba16Float {
                return Err(Error::InvalidState.into());
            }
            let output = frame.frame.output.as_ref().ok_or(Error::InvalidState)?;
            if output.desc.format != TextureFormat::Rgba8Unorm
                || output.desc.extent != [frame.extent[0], frame.extent[1], 1]
                || output.desc.dimension != TextureDimension::D2
                || output.desc.array_layers != 1
                || output.desc.mip_levels != 1
                || output.desc.samples != 1
                || !output
                    .desc
                    .usage
                    .contains(TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC)
                || output.instance == frame.frame.target_instance
            {
                return Err(Error::InvalidState.into());
            }
            ForwardOutput::ProcessedOffscreen {
                desc: &output.desc,
                incoming: output.incoming,
                layout: output.layout,
            }
        }
    };
    declare_offscreen_with_output(
        device,
        &spec,
        expected,
        output,
        &frame.indirect,
        &frame.particles,
        contribute,
    )
}

fn declare_readback(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    color: TextureHandle,
    readback: BufferHandle,
) -> Result<GraphPass, Error> {
    let readback_len = u64::from(spec.readback_row_pitch)
        .checked_mul(u64::from(spec.extent[1]))
        .ok_or(Error::InvalidRange)?;
    let (pass, ()) = graph
        .add_pass("forward-readback", |pass| {
            pass.read_texture_with_use(
                color,
                whole(),
                texture_copy_read_uses().uses,
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            )?;
            pass.write_with_use(
                readback,
                0..readback_len,
                copy_write_uses().uses,
                BufferUsage::COPY_DST,
            )?;
            pass.side_effect();
            Ok(())
        })
        .map_err(|_| Error::InvalidState)?;
    Ok(pass)
}

fn declare_output_composite(
    graph: &mut GraphBuilder,
    scene_color: TextureHandle,
    output: TextureHandle,
    present: bool,
) -> Result<(GraphPass, TextureHandle), Error> {
    graph
        .add_pass("forward-hdr-output-composite", |pass| {
            pass.read_texture_with_use(
                scene_color,
                whole(),
                super::texture_read_uses().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            let output = pass.write_texture_with_use(
                output,
                whole(),
                super::color_write_uses().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            if present {
                pass.side_effect();
            }
            Ok(output)
        })
        .map_err(|_| Error::InvalidState)
}
