//! Range-scoped Forward raster recording shared by ordinary and OIT paths.

use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;
use super::*;

pub(in crate::graph::forward) struct RasterColorSelection<'a, B: ResourceBackend> {
    pub(in crate::graph::forward) slot: TextureSlot,
    pub(in crate::graph::forward) texture: &'a fluxel_rhi::api::Texture<B>,
    pub(in crate::graph::forward) view: &'a fluxel_rhi::api::TextureView<B>,
    pub(in crate::graph::forward) instance: crate::graph::ResourceInstanceId,
    pub(in crate::graph::forward) load: LoadOp<ClearColor>,
    pub(in crate::graph::forward) read_write: bool,
}

pub(in crate::graph::forward) struct RasterDepthSelection<'a, B: ResourceBackend> {
    pub(in crate::graph::forward) slot: TextureSlot,
    pub(in crate::graph::forward) texture: &'a fluxel_rhi::api::Texture<B>,
    pub(in crate::graph::forward) view: &'a fluxel_rhi::api::TextureView<B>,
    pub(in crate::graph::forward) instance: crate::graph::ResourceInstanceId,
    pub(in crate::graph::forward) load: LoadOp<f32>,
    pub(in crate::graph::forward) store: StoreOp,
    pub(in crate::graph::forward) read_only: bool,
    pub(in crate::graph::forward) read_write: bool,
}

pub(in crate::graph::forward) struct RasterSelection<'a, B: ResourceBackend> {
    pub(in crate::graph::forward) draws: Range<usize>,
    pub(in crate::graph::forward) color: RasterColorSelection<'a, B>,
    pub(in crate::graph::forward) depth: Option<RasterDepthSelection<'a, B>>,
}

pub(in crate::graph::forward) fn record_raster_selection<B: RasterBackend + ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
    selection: RasterSelection<'_, B>,
) -> Result<()> {
    if selection.draws.start > selection.draws.end || selection.draws.end > frame.draws.len() {
        return Err(Error::InvalidRange);
    }
    let pass = bindings.pass(pass_id)?;
    if selection.color.texture.device() != slots.device
        || selection.color.view.device() != slots.device
    {
        return Err(Error::WrongDevice);
    }
    bindings.require_texture_owner(
        selection.color.slot,
        selection.color.instance,
        selection.color.texture,
    )?;
    let multisample_color = slots
        .multisample_color_slot
        .map(|slot| {
            let owner = frame
                .frame
                .multisample_color
                .as_ref()
                .ok_or(Error::InvalidState)?;
            if selection.color.slot != slots.target_slot {
                return Err(Error::InvalidState);
            }
            if owner.device != slots.device
                || owner.texture.device() != slots.device
                || owner.view.device() != slots.device
            {
                return Err(Error::WrongDevice);
            }
            bindings.require_texture_owner(slot, owner.instance, &owner.texture)?;
            Ok((slot, owner))
        })
        .transpose()?;
    for draw in &frame.draws[selection.draws.clone()] {
        validate_draw(draw, slots.stream_slots.len())?;
        if draw.pipeline.device() != slots.device
            || draw
                .bind_groups
                .iter()
                .any(|g| g.group.device() != slots.device)
        {
            return Err(Error::WrongDevice);
        }
        for (stream, range) in draw.vertex_streams.iter().zip(draw_accesses(frame, draw)?) {
            if let Some(range) = range {
                pass.buffer(
                    *slots.stream_slots.get(*stream).ok_or(Error::InvalidRange)?,
                    range,
                    AccessKind::Read,
                    vertex_read_uses(),
                )?;
            }
        }
        for group in &draw.bind_groups {
            for range in &group.uniform_ranges {
                pass.buffer(
                    slots.uniform_slot,
                    range.clone(),
                    AccessKind::Read,
                    shader_read_uses(),
                )?;
            }
        }
        for texture in &draw.textures {
            let owner = frame
                .resident
                .textures
                .get(*texture)
                .ok_or(Error::InvalidRange)?;
            pass.texture(
                *slots
                    .texture_slots
                    .get(*texture)
                    .ok_or(Error::InvalidRange)?,
                all_color_mips(owner.desc.mip_levels),
                AccessKind::Read,
                texture_read_uses(),
            )?;
        }
    }
    if !selection.draws.is_empty() {
        if frame.environment.is_some() {
            pass.texture(
                slots.environment_slot.ok_or(Error::InvalidState)?,
                whole(),
                AccessKind::Read,
                texture_read_uses(),
            )?;
        }
        if let Some(owner) = frame.specular.as_ref() {
            owner.validate_for_sampling(slots.device)?;
            pass.texture(
                slots.specular_radiance_slot.ok_or(Error::InvalidState)?,
                all_color_mips(owner.radiance.desc.mip_levels),
                AccessKind::Read,
                texture_read_uses(),
            )?;
            pass.texture(
                slots.specular_brdf_lut_slot.ok_or(Error::InvalidState)?,
                whole(),
                AccessKind::Read,
                texture_read_uses(),
            )?;
            pass.buffer(
                slots
                    .specular_view_uniform_slot
                    .ok_or(Error::InvalidState)?,
                0..SPECULAR_VIEW_BYTES,
                AccessKind::Read,
                shader_read_uses(),
            )?;
        }
    }
    let raster_color_slot = multisample_color.map_or(selection.color.slot, |(slot, _)| slot);
    pass.texture(
        raster_color_slot,
        whole(),
        AccessKind::Write,
        if selection.color.read_write {
            color_read_write_uses()
        } else {
            color_write_uses()
        },
    )?;
    if multisample_color.is_some() {
        // This is the product of the in-pass color resolve, never an input
        // attachment.  Keep its Graph contract separate from the raster
        // attachment, whose descriptor has the multisample count.
        pass.texture(
            selection.color.slot,
            whole(),
            AccessKind::Write,
            color_write_uses(),
        )?;
    }
    let depth_stencil = selection
        .depth
        .as_ref()
        .map(|depth| {
            bindings.require_texture_owner(depth.slot, depth.instance, depth.texture)?;
            if depth.texture.device() != slots.device || depth.view.device() != slots.device {
                return Err(Error::WrongDevice);
            }
            pass.texture(
                depth.slot,
                whole_depth(),
                if depth.read_only {
                    AccessKind::Read
                } else {
                    AccessKind::Write
                },
                if depth.read_only {
                    depth_read_uses()
                } else if depth.read_write {
                    depth_read_write_uses()
                } else {
                    depth_write_uses()
                },
            )?;
            Ok(DepthStencilAttachment {
                view: AttachmentView::Texture(depth.view),
                depth: Some((depth.load, depth.store)),
                stencil: None,
                depth_read_only: depth.read_only,
                stencil_read_only: true,
            })
        })
        .transpose()?;
    pass.transition(encoder)?;
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(
            multisample_color.map_or(selection.color.view, |(_, owner)| &owner.view),
        ),
        resolve_target: multisample_color.map(|_| AttachmentView::Texture(selection.color.view)),
        load: selection.color.load,
        store: if multisample_color.is_some() {
            StoreOp::Discard
        } else {
            StoreOp::Store
        },
    })];
    let mut raster = encoder.begin_raster(&RasterPassDesc {
        label: Some("fluxel forward selection"),
        colors: &colors,
        depth_stencil,
        render_area: Rect {
            x: 0,
            y: 0,
            width: frame.extent[0],
            height: frame.extent[1],
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })?;
    raster.viewport(Viewport {
        x: 0.0,
        y: 0.0,
        width: frame.extent[0] as f32,
        height: frame.extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    raster.scissor(Rect {
        x: 0,
        y: 0,
        width: frame.extent[0],
        height: frame.extent[1],
    })?;
    let mut bound_batch = None;
    for draw in &frame.draws[selection.draws] {
        if bound_batch != Some(draw.batch) {
            raster.set_pipeline(&draw.pipeline)?;
            bound_batch = Some(draw.batch);
        }
        for group in &draw.bind_groups {
            raster.set_bind_group(group.slot, &group.group, &group.dynamic_offsets)?;
        }
        for (slot, (stream, range)) in draw
            .vertex_streams
            .iter()
            .zip(&draw.vertex_ranges)
            .enumerate()
        {
            raster.set_vertex_buffer(
                u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                &frame.resident.streams[*stream].buffer,
                range.clone(),
            )?;
        }
        raster.draw(draw.vertices.clone(), draw.instances.clone())?;
    }
    raster.finish()
}

fn depth_read_uses() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::DEPTH_READ,
        },
        layout: ImageLayout::DepthStencilAttachment,
        descriptor_usage: TextureUsage::RENDER_ATTACHMENT,
    }
}
