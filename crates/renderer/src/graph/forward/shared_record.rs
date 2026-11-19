//! Checked recording shared by ordinary and captured Forward frames.
use super::*;
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;
#[path = "shared_raster.rs"]
mod shared_raster;
pub(super) use shared_raster::{
    RasterColorSelection, RasterDepthSelection, RasterSelection, bind_and_draw,
    bind_and_draw_indirect, record_raster_selection, record_raster_selection_with_indirect,
};

pub(super) struct RecordSlots<'a> {
    pub(super) device: DeviceId,
    pub(super) topology: FrameTopology,
    pub(super) resident_upload_slot: BufferSlot,
    pub(super) frame_upload_slot: BufferSlot,
    pub(super) uniform_slot: BufferSlot,
    pub(super) environment_upload_slot: Option<BufferSlot>,
    pub(super) specular_upload_slot: Option<BufferSlot>,
    pub(super) specular_view_uniform_slot: Option<BufferSlot>,
    pub(super) stream_slots: &'a [BufferSlot],
    pub(super) texture_slots: &'a [TextureSlot],
    pub(super) environment_slot: Option<TextureSlot>,
    pub(super) specular_radiance_slot: Option<TextureSlot>,
    pub(super) specular_brdf_lut_slot: Option<TextureSlot>,
    pub(super) depth_slot: Option<TextureSlot>,
    pub(super) target_slot: TextureSlot,
    /// Multisampled raster attachment; `target_slot` remains its single-sample
    /// in-pass resolve destination.
    pub(super) multisample_color_slot: Option<TextureSlot>,
    pub(super) readback_slot: BufferSlot,
}

pub(super) fn record_upload<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
) -> Result<()>
where
    B: CommandBackend,
{
    let pass = bindings.pass(pass_id)?;
    for upload in &frame.resident_upload.vertices {
        let slot = *slots
            .stream_slots
            .get(upload.stream)
            .ok_or(Error::InvalidRange)?;
        pass.buffer(
            slots.resident_upload_slot,
            upload.source.clone(),
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.buffer(
            slot,
            upload.destination.clone(),
            AccessKind::Write,
            copy_write_uses(),
        )?;
        if upload.source.end - upload.source.start
            != upload.destination.end - upload.destination.start
        {
            return Err(Error::InvalidRange);
        }
    }
    pass.buffer(
        slots.frame_upload_slot,
        frame.uniform_upload.source.clone(),
        AccessKind::Read,
        copy_read_uses(),
    )?;
    pass.buffer(
        slots.uniform_slot,
        frame.uniform_upload.destination.clone(),
        AccessKind::Write,
        copy_write_uses(),
    )?;
    if frame.uniform_upload.source.end - frame.uniform_upload.source.start
        != frame.uniform_upload.destination.end - frame.uniform_upload.destination.start
    {
        return Err(Error::InvalidRange);
    }
    for copy in &frame.resident_upload.textures {
        let owner = frame
            .resident
            .textures
            .get(copy.texture)
            .ok_or(Error::InvalidRange)?;
        let slot = *slots
            .texture_slots
            .get(copy.texture)
            .ok_or(Error::InvalidRange)?;
        pass.buffer(
            slots.resident_upload_slot,
            copy.source_offset..texture_copy_end(copy)?,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.texture(
            slot,
            material_mip_layer(&owner.desc, copy.mip_level, copy.array_layer),
            AccessKind::Write,
            texture_copy_write_uses(),
        )?;
    }
    if let (Some(slot), Some(owner)) = (slots.environment_upload_slot, frame.environment.as_ref()) {
        pass.buffer(
            slot,
            0..owner.upload_desc.size,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.texture(
            slots.environment_slot.ok_or(Error::InvalidState)?,
            whole(),
            AccessKind::Write,
            texture_copy_write_uses(),
        )?;
    }
    if let (Some(upload), Some(uniform), Some(radiance), Some(lut), Some(owner)) = (
        slots.specular_upload_slot,
        slots.specular_view_uniform_slot,
        slots.specular_radiance_slot,
        slots.specular_brdf_lut_slot,
        frame.specular.as_ref(),
    ) {
        owner.validate_for_sampling(slots.device)?;
        for copy in &owner.radiance_copies {
            let end = copy
                .source_offset
                .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                .ok_or(Error::InvalidRange)?;
            pass.buffer(
                upload,
                copy.source_offset..end,
                AccessKind::Read,
                copy_read_uses(),
            )?;
            pass.texture(
                radiance,
                one_color_mip(copy.mip),
                AccessKind::Write,
                texture_copy_write_uses(),
            )?;
        }
        let copy = &owner.brdf_lut_copy;
        let end = copy
            .source_offset
            .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
            .ok_or(Error::InvalidRange)?;
        pass.buffer(
            upload,
            copy.source_offset..end,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.texture(lut, whole(), AccessKind::Write, texture_copy_write_uses())?;
        pass.buffer(
            upload,
            owner.view_uniform_copy.clone(),
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.buffer(
            uniform,
            0..SPECULAR_VIEW_BYTES,
            AccessKind::Write,
            copy_write_uses(),
        )?;
    }
    pass.transition(encoder)?;
    for upload in &frame.resident_upload.vertices {
        let slot = *slots
            .stream_slots
            .get(upload.stream)
            .ok_or(Error::InvalidRange)?;
        let src = pass.buffer(
            slots.resident_upload_slot,
            upload.source.clone(),
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let dst = pass.buffer(
            slot,
            upload.destination.clone(),
            AccessKind::Write,
            copy_write_uses(),
        )?;
        encoder.copy_buffer(
            src,
            upload.source.start,
            dst,
            upload.destination.start,
            upload.destination.end - upload.destination.start,
        )?;
    }
    let src = pass.buffer(
        slots.frame_upload_slot,
        frame.uniform_upload.source.clone(),
        AccessKind::Read,
        copy_read_uses(),
    )?;
    let dst = pass.buffer(
        slots.uniform_slot,
        frame.uniform_upload.destination.clone(),
        AccessKind::Write,
        copy_write_uses(),
    )?;
    encoder.copy_buffer(
        src,
        frame.uniform_upload.source.start,
        dst,
        frame.uniform_upload.destination.start,
        frame.uniform_upload.destination.end - frame.uniform_upload.destination.start,
    )?;
    for copy in &frame.resident_upload.textures {
        let owner = frame
            .resident
            .textures
            .get(copy.texture)
            .ok_or(Error::InvalidRange)?;
        let src = pass.buffer(
            slots.resident_upload_slot,
            copy.source_offset..texture_copy_end(copy)?,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let texture = pass.texture(
            *slots
                .texture_slots
                .get(copy.texture)
                .ok_or(Error::InvalidRange)?,
            material_mip_layer(&owner.desc, copy.mip_level, copy.array_layer),
            AccessKind::Write,
            texture_copy_write_uses(),
        )?;
        encoder.buffer_to_texture(
            src,
            TexelCopyBufferLayout {
                offset: copy.source_offset,
                bytes_per_row: copy.bytes_per_row,
                rows_per_image: copy.rows_per_image,
            },
            TextureRegion {
                texture,
                mip: copy.mip_level,
                origin: [0, 0, copy.array_layer],
                extent: copy.extent,
                aspect: material_aspect(&owner.desc),
            },
        )?;
    }
    if let (Some(slot), Some(owner)) = (slots.environment_upload_slot, frame.environment.as_ref()) {
        let src = pass.buffer(
            slot,
            0..owner.upload_desc.size,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let texture = pass.texture(
            slots.environment_slot.ok_or(Error::InvalidState)?,
            whole(),
            AccessKind::Write,
            texture_copy_write_uses(),
        )?;
        encoder.buffer_to_texture(
            src,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: owner.bytes_per_row,
                rows_per_image: owner.desc.extent[1],
            },
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: owner.desc.extent,
                aspect: TextureAspects::COLOR,
            },
        )?;
    }
    if let (Some(upload), Some(uniform), Some(radiance_slot), Some(lut_slot), Some(owner)) = (
        slots.specular_upload_slot,
        slots.specular_view_uniform_slot,
        slots.specular_radiance_slot,
        slots.specular_brdf_lut_slot,
        frame.specular.as_ref(),
    ) {
        for copy in &owner.radiance_copies {
            let src = pass.buffer(
                upload,
                copy.source_offset
                    ..copy
                        .source_offset
                        .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                        .ok_or(Error::InvalidRange)?,
                AccessKind::Read,
                copy_read_uses(),
            )?;
            let texture = pass.texture(
                radiance_slot,
                one_color_mip(copy.mip),
                AccessKind::Write,
                texture_copy_write_uses(),
            )?;
            encoder.buffer_to_texture(
                src,
                TexelCopyBufferLayout {
                    offset: copy.source_offset,
                    bytes_per_row: copy.bytes_per_row,
                    rows_per_image: copy.rows_per_image,
                },
                TextureRegion {
                    texture,
                    mip: copy.mip,
                    origin: [0, 0, 0],
                    extent: copy.extent,
                    aspect: TextureAspects::COLOR,
                },
            )?;
        }
        let copy = &owner.brdf_lut_copy;
        let src = pass.buffer(
            upload,
            copy.source_offset
                ..copy
                    .source_offset
                    .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                    .ok_or(Error::InvalidRange)?,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let texture = pass.texture(
            lut_slot,
            whole(),
            AccessKind::Write,
            texture_copy_write_uses(),
        )?;
        encoder.buffer_to_texture(
            src,
            TexelCopyBufferLayout {
                offset: copy.source_offset,
                bytes_per_row: copy.bytes_per_row,
                rows_per_image: copy.rows_per_image,
            },
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: copy.extent,
                aspect: TextureAspects::COLOR,
            },
        )?;
        let src = pass.buffer(
            upload,
            owner.view_uniform_copy.clone(),
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let dst = pass.buffer(
            uniform,
            0..SPECULAR_VIEW_BYTES,
            AccessKind::Write,
            copy_write_uses(),
        )?;
        encoder.copy_buffer(
            src,
            owner.view_uniform_copy.start,
            dst,
            0,
            SPECULAR_VIEW_BYTES,
        )?;
    }
    Ok(())
}

pub(super) fn record_raster<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
) -> Result<()>
where
    B: RasterBackend,
{
    record_raster_with_color_load(
        slots,
        pass_id,
        bindings,
        encoder,
        frame,
        LoadOp::Clear(ClearColor::Float([0.0; 4])),
    )
}

pub(super) fn record_raster_with_color_load<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
    color_load: LoadOp<ClearColor>,
) -> Result<()>
where
    B: RasterBackend,
{
    record_raster_with_loads(
        slots,
        pass_id,
        bindings,
        encoder,
        frame,
        frame.frame.depth.as_ref(),
        color_load,
        LoadOp::Clear(1.0),
        StoreOp::Discard,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_raster_with_loads<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
    depth_owner: Option<&crate::residency::ForwardDepthAttachment<B>>,
    color_load: LoadOp<ClearColor>,
    depth_load: LoadOp<f32>,
    depth_store: StoreOp,
    depth_read_only: bool,
) -> Result<()>
where
    B: RasterBackend,
{
    record_raster_with_loads_inner(
        slots,
        pass_id,
        bindings,
        encoder,
        frame,
        depth_owner,
        color_load,
        depth_load,
        depth_store,
        depth_read_only,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_raster_with_loads_indirect<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
    depth_owner: Option<&crate::residency::ForwardDepthAttachment<B>>,
    color_load: LoadOp<ClearColor>,
    depth_load: LoadOp<f32>,
    depth_store: StoreOp,
    depth_read_only: bool,
    device: &fluxel_rhi::api::Device<B>,
) -> Result<()>
where
    B: RasterBackend,
{
    record_raster_with_loads_inner(
        slots,
        pass_id,
        bindings,
        encoder,
        frame,
        depth_owner,
        color_load,
        depth_load,
        depth_store,
        depth_read_only,
        Some(device),
    )
}

#[allow(clippy::too_many_arguments)]
fn record_raster_with_loads_inner<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
    depth_owner: Option<&crate::residency::ForwardDepthAttachment<B>>,
    color_load: LoadOp<ClearColor>,
    depth_load: LoadOp<f32>,
    depth_store: StoreOp,
    depth_read_only: bool,
    device: Option<&fluxel_rhi::api::Device<B>>,
) -> Result<()>
where
    B: RasterBackend,
{
    validate_forward_topology(slots.topology, frame.draws.len(), frame.draw_batches.len())?;
    validate_batches(frame)?;
    if depth_owner.is_some_and(|owner| owner.device != slots.device) {
        return Err(Error::WrongDevice);
    }
    let depth = match (slots.depth_slot, depth_owner) {
        (None, None) => None,
        (Some(slot), Some(owner)) => Some(RasterDepthSelection {
            slot,
            texture: &owner.texture,
            view: &owner.view,
            instance: owner.instance,
            load: depth_load,
            store: depth_store,
            read_only: depth_read_only,
            read_write: !depth_read_only && slots.topology == FrameTopology::SortedForward,
        }),
        _ => return Err(Error::InvalidState),
    };
    record_raster_selection_with_indirect(
        slots,
        pass_id,
        bindings,
        encoder,
        frame,
        RasterSelection {
            draws: 0..frame.draws.len(),
            color: RasterColorSelection {
                slot: slots.target_slot,
                texture: &frame.frame.target,
                view: &frame.frame.target_view,
                instance: frame.frame.target_instance,
                load: color_load,
                read_write: slots.topology == FrameTopology::SortedForward,
            },
            depth,
        },
        device,
    )
}

pub(super) fn record_readback<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
where
    B: ReadbackBackend,
{
    let length = u64::from(frame.readback_row_pitch)
        .checked_mul(u64::from(frame.extent[1]))
        .ok_or(Error::InvalidRange)?;
    let pass = bindings.pass(pass_id)?;
    let target = pass.texture(
        slots.target_slot,
        whole(),
        AccessKind::Read,
        texture_copy_read_uses(),
    )?;
    let readback = pass.buffer(
        slots.readback_slot,
        0..length,
        AccessKind::Write,
        copy_write_uses(),
    )?;
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
