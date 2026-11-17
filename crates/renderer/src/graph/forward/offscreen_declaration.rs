//! Complete private offscreen Forward graph declarations.

use fluxel_rhi::{
    api::{BufferUsage, Device, Error, ImageLayout, LaneId, TextureUsage},
    backend::ResourceBackend,
};

use crate::{
    FrameTopology,
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
    pub(super) readback_slot: BufferSlot,
    pub(super) draw_count: usize,
    pub(super) stream_count: usize,
    pub(super) texture_count: usize,
    pub(super) extent: [u32; 2],
    pub(super) shape: ForwardShape,
    pub(super) uniform_alignment: u64,
    pub(super) topology: FrameTopology,
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
    let spec = ForwardGraphSpec::from_frame(frame)?;
    declare_offscreen_spec_with(device, &spec, expected, contribute)
}

pub(super) fn declare_offscreen_spec_with<B, F>(
    device: &Device<B>,
    spec: &ForwardGraphSpec,
    expected: FrameTopology,
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
    } = super::offscreen_prefix::declare_prefix(device, spec, expected)?;
    let raster_dependency = mip_passes.last().map_or(upload_pass, |pass| pass.pass);
    let (raster_pass, target) = {
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
            target,
            depth,
            draws: spec.draws.clone(),
            declared: false,
            output: None,
        };
        let mut context = ForwardPhaseContext::new(&mut driver);
        contribute(&mut context).map_err(ForwardPlanBuildError::Pipeline)?;
        driver
            .finish()
            .map_err(|_| ForwardPlanBuildError::Pipeline(PipelineError::InvalidPhaseContribution))?
    };
    let readback_len = u64::from(spec.readback_row_pitch)
        .checked_mul(u64::from(spec.extent[1]))
        .ok_or(Error::InvalidRange)?;
    let (readback_pass, ()) = graph
        .add_pass("forward-readback", |pass| {
            pass.read_texture_with_use(
                target,
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
    Ok(OffscreenForwardDeclaration {
        graph,
        lane,
        upload_pass,
        mip_passes,
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
        readback_slot: BufferSlot::from(readback),
        draw_count: spec.draws.len(),
        stream_count: spec.streams.len(),
        texture_count: spec.textures.len(),
        extent: spec.extent,
        shape: spec.shape.clone(),
        uniform_alignment,
        topology: expected,
    })
}
