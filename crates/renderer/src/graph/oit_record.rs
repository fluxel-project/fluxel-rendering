//! Checked command recording for the declared OIT product.
//!
//! Program and bind-group references here are command facts, like ForwardDraw:
//! they never authorize a resource access. Every texture and buffer is first
//! resolved through the Graph pass authority before any transition or raster use.

use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;
use fluxel_rhi::{
    api::{
        AccessMask, AttachmentView, ClearColor, ColorAttachment, CommandEncoder,
        DepthStencilAttachment, Error, ImageLayout, LoadOp, RasterPassDesc, Rect, Result,
        StageMask, StoreOp, TextureUsage, TextureUses, TextureView,
    },
    backend::{RasterBackend, ResourceBackend},
};

use super::forward::forward_draw_accesses;
use crate::{
    graph::{
        binding::{ResourceInstanceId, TextureSlot},
        definition::{AccessKind, BufferAccessContract, TextureAccessContract, TextureRange},
        execution::FrameBindings,
        oit::DualDepthPeelGraph,
    },
    residency::{ForwardFrame, OitOwners},
};
#[path = "oit_record_facts.rs"]
mod oit_record_facts;
#[path = "oit_record_final.rs"]
mod oit_record_final;
#[path = "oit_record_owners.rs"]
mod oit_record_owners;
pub(crate) use oit_record_facts::*;
use oit_record_final::{clear, final_composite};

/// Records exactly one declared OIT Graph pass. The execution owner calls this
/// in compiled-Graph order; this function never replays the product itself.
pub(crate) fn record_pass<B: RasterBackend + ResourceBackend>(
    device: fluxel_rhi::api::DeviceId,
    encoder: &mut CommandEncoder<B>,
    bindings: &FrameBindings<'_, B>,
    product: &DualDepthPeelGraph,
    pass_id: crate::graph::definition::GraphPass,
    slots: &OitRecordSlots,
    owners: &OitOwners<B>,
    opaque_depth: &OitDepthTargetRef<'_, B>,
    frame: &ForwardFrame<B>,
    draw_slots: &OitDrawSlots<'_>,
    commands: &OitRecordCommands<'_, B>,
) -> Result<()> {
    owners.validate(device)?;
    validate_external(device, frame.extent, &commands.opaque_color)?;
    validate_external(device, frame.extent, &commands.output)?;
    validate_depth(device, frame.extent, opaque_depth)?;
    validate_instances(
        owners,
        opaque_depth,
        &commands.opaque_color,
        &commands.output,
    )?;
    oit_record_owners::validate(
        bindings,
        slots,
        owners,
        opaque_depth,
        frame,
        draw_slots,
        commands,
    )?;
    if commands
        .draws
        .iter()
        .any(|entry| entry.draw >= frame.draws.len())
    {
        return Err(Error::InvalidRange);
    }
    if pass_id == product.passes.initial_clear[0].pass {
        return clear(
            encoder,
            bindings,
            pass_id,
            &slots.ping_pong[0],
            &owners.ping_pong[0],
            frame.extent,
            [-99999.0, -99999.0, 0.0, 0.0],
        );
    }
    if pass_id == product.passes.initial_clear[1].pass {
        return clear(
            encoder,
            bindings,
            pass_id,
            &slots.ping_pong[1],
            &owners.ping_pong[1],
            frame.extent,
            [0.0, 1.0, 0.0, 0.0],
        );
    }
    if pass_id == product.passes.initialize_back_accumulator {
        validate_group(
            &commands.init_back.bind_group,
            &[commands.opaque_color.instance],
        )?;
        return fullscreen(
            encoder,
            bindings,
            pass_id,
            slots.opaque_color,
            slots.back_accumulator,
            &commands.opaque_color.view,
            &owners.back_accumulator.view,
            &commands.init_back,
            frame.extent,
            false,
        );
    }
    if pass_id == product.passes.initial_depth {
        for draw in commands.draws {
            validate_group(
                &draw.initial_group3,
                &[owners.ping_pong[1].min_max.instance],
            )?;
        }
        return material(
            encoder,
            bindings,
            pass_id,
            slots,
            slots.ping_pong[1].min_max,
            slots.ping_pong[0].min_max,
            None,
            &owners.ping_pong[1].min_max.view,
            &owners.ping_pong[0].min_max.view,
            None,
            Some((slots.opaque_depth, &opaque_depth.view)),
            frame,
            draw_slots,
            commands.draws,
            true,
        );
    }
    let mut read = 0;
    for iteration in &product.passes.iterations {
        let write = 1 - read;
        if pass_id == iteration.clear.pass {
            return clear(
                encoder,
                bindings,
                pass_id,
                &slots.ping_pong[write],
                &owners.ping_pong[write],
                frame.extent,
                [-99999.0, -99999.0, 0.0, 0.0],
            );
        }
        if pass_id == iteration.peel {
            for draw in commands.draws {
                validate_group(
                    &draw.peel_group3[read],
                    &[
                        owners.ping_pong[read].min_max.instance,
                        owners.ping_pong[read].front.instance,
                    ],
                )?;
            }
            return material(
                encoder,
                bindings,
                pass_id,
                slots,
                slots.ping_pong[read].min_max,
                slots.ping_pong[write].min_max,
                Some(slots.ping_pong[read].front),
                &owners.ping_pong[read].min_max.view,
                &owners.ping_pong[write].min_max.view,
                Some((
                    &slots.ping_pong[write],
                    &owners.ping_pong[write].front.view,
                    &owners.ping_pong[write].back.view,
                )),
                None,
                frame,
                draw_slots,
                commands.draws,
                false,
            );
        }
        if pass_id == iteration.back_accumulate {
            validate_group(
                &commands.back_accumulate[write].bind_group,
                &[owners.ping_pong[write].back.instance],
            )?;
            return fullscreen(
                encoder,
                bindings,
                pass_id,
                slots.ping_pong[write].back,
                slots.back_accumulator,
                &owners.ping_pong[write].back.view,
                &owners.back_accumulator.view,
                &commands.back_accumulate[write],
                frame.extent,
                true,
            );
        }
        read = write;
    }
    if pass_id == product.passes.final_composite {
        let read = slots
            .ping_pong
            .iter()
            .position(|side| side.front == TextureSlot::from(product.final_front))
            .ok_or(Error::InvalidState)?;
        if slots.back_accumulator != TextureSlot::from(product.final_back_accumulator) {
            return Err(Error::InvalidState);
        }
        validate_group(
            &commands.final_composite[read].bind_group,
            &[
                owners.ping_pong[read].front.instance,
                owners.back_accumulator.instance,
            ],
        )?;
        return final_composite(
            encoder,
            bindings,
            pass_id,
            slots.ping_pong[read].front,
            slots.back_accumulator,
            slots.output,
            &owners.ping_pong[read].front.view,
            &owners.ping_pong[read].front.texture,
            &owners.back_accumulator.texture,
            &commands.output.view,
            commands.output.texture,
            &commands.final_composite[read],
            frame.extent,
        );
    }
    Err(Error::InvalidState)
}

fn fullscreen<B: RasterBackend + ResourceBackend>(
    encoder: &mut CommandEncoder<B>,
    bindings: &FrameBindings<'_, B>,
    id: crate::graph::definition::GraphPass,
    read: TextureSlot,
    write: TextureSlot,
    source: &TextureView<B>,
    target: &TextureView<B>,
    command: &OitFullscreenCommand<'_, B>,
    extent: [u32; 2],
    read_write: bool,
) -> Result<()> {
    let pass = bindings.pass(id)?;
    checked(&pass, read, AccessKind::Read, shader_read())?;
    checked(
        &pass,
        write,
        AccessKind::Write,
        if read_write {
            color_rw()
        } else {
            color_write()
        },
    )?;
    pass.transition(encoder)?;
    let colors = [Some(color(
        target,
        if read_write {
            LoadOp::Load
        } else {
            LoadOp::Clear(ClearColor::Float([0.0; 4]))
        },
    ))];
    let mut r = encoder.begin_raster(&desc("oit-fullscreen", &colors, None, extent))?;
    r.set_pipeline(command.pipeline)?;
    r.set_bind_group(0, command.bind_group.group, &[])?;
    let _ = source;
    r.draw(0..3, 0..1)?;
    r.finish()
}

fn material<B: RasterBackend + ResourceBackend>(
    encoder: &mut CommandEncoder<B>,
    bindings: &FrameBindings<'_, B>,
    id: crate::graph::definition::GraphPass,
    slots: &OitRecordSlots,
    read_slot: TextureSlot,
    write_slot: TextureSlot,
    read_front: Option<TextureSlot>,
    read_minmax: &TextureView<B>,
    write_minmax: &TextureView<B>,
    peel_targets: Option<(&OitSideSlots, &TextureView<B>, &TextureView<B>)>,
    depth: Option<(TextureSlot, &TextureView<B>)>,
    frame: &ForwardFrame<B>,
    draw_slots: &OitDrawSlots<'_>,
    programs: &[OitDrawPrograms<'_, B>],
    initial: bool,
) -> Result<()> {
    let pass = bindings.pass(id)?;
    checked(&pass, read_slot, AccessKind::Read, shader_read())?;
    if let Some(slot) = read_front {
        checked(&pass, slot, AccessKind::Read, shader_read())?;
    }
    checked(&pass, write_slot, AccessKind::Write, color_rw())?;
    if let Some((side, _, _)) = peel_targets {
        checked(&pass, side.front, AccessKind::Write, color_rw())?;
        checked(&pass, side.back, AccessKind::Write, color_rw())?;
    }
    if let Some((slot, _)) = depth {
        checked_depth(&pass, slot)?;
    }
    for item in programs {
        let draw = &frame.draws[item.draw];
        for (stream, range) in draw
            .vertex_streams
            .iter()
            .zip(forward_draw_accesses(frame, draw)?)
        {
            let Some(range) = range else {
                continue;
            };
            pass.buffer(
                *draw_slots.streams.get(*stream).ok_or(Error::InvalidRange)?,
                range,
                AccessKind::Read,
                vertex_read(),
            )?;
        }
        for group in &draw.bind_groups {
            for range in &group.uniform_ranges {
                pass.buffer(
                    draw_slots.uniform,
                    range.clone(),
                    AccessKind::Read,
                    uniform_read(),
                )?;
            }
        }
        for texture in &draw.textures {
            let owner = frame
                .resident
                .textures
                .get(*texture)
                .ok_or(Error::InvalidRange)?;
            checked_mips(
                &pass,
                *draw_slots
                    .textures
                    .get(*texture)
                    .ok_or(Error::InvalidRange)?,
                owner.desc.mip_levels,
            )?;
        }
    }
    if let Some(environment) = frame.environment.as_ref() {
        checked(
            &pass,
            draw_slots.environment.ok_or(Error::InvalidState)?,
            AccessKind::Read,
            shader_read(),
        )?;
        let _ = environment;
    }
    if let Some(specular) = frame.specular.as_ref() {
        checked_mips(
            &pass,
            draw_slots.specular_radiance.ok_or(Error::InvalidState)?,
            specular.radiance.desc.mip_levels,
        )?;
        checked(
            &pass,
            draw_slots.specular_brdf_lut.ok_or(Error::InvalidState)?,
            AccessKind::Read,
            shader_read(),
        )?;
        pass.buffer(
            draw_slots
                .specular_view_uniform
                .ok_or(Error::InvalidState)?,
            0..SPECULAR_VIEW_BYTES,
            AccessKind::Read,
            uniform_read(),
        )?;
    }
    pass.transition(encoder)?;
    let depth_stencil = depth.map(|(_, view)| DepthStencilAttachment {
        view: AttachmentView::Texture(view),
        depth: Some((LoadOp::Load, StoreOp::Store)),
        stencil: None,
        depth_read_only: true,
        stencil_read_only: true,
    });
    let first = [Some(color(write_minmax, LoadOp::Load))];
    let peel = peel_targets.map(|(_, front, back)| {
        [
            Some(color(write_minmax, LoadOp::Load)),
            Some(color(front, LoadOp::Load)),
            Some(color(back, LoadOp::Load)),
        ]
    });
    let colors: &[Option<ColorAttachment<'_, B>>] = match peel.as_ref() {
        Some(colors) => colors,
        None => &first,
    };
    let mut r = encoder.begin_raster(&desc("oit-material", colors, depth_stencil, frame.extent))?;
    for item in programs {
        let draw = &frame.draws[item.draw];
        r.set_pipeline(if initial { item.initial } else { item.peel })?;
        for group in &draw.bind_groups {
            r.set_bind_group(group.slot, &group.group, &group.dynamic_offsets)?;
        }
        if let Some(empty) = item.empty_group2 {
            r.set_bind_group(2, empty, &[])?;
        }
        r.set_bind_group(
            3,
            if initial {
                item.initial_group3.group
            } else {
                item.peel_group3[read_side_index(slots, read_slot)?].group
            },
            &[],
        )?;
        for (index, (stream, range)) in draw
            .vertex_streams
            .iter()
            .zip(&draw.vertex_ranges)
            .enumerate()
        {
            r.set_vertex_buffer(
                u32::try_from(index).map_err(|_| Error::InvalidRange)?,
                bindings.canonical_buffer(
                    *draw_slots.streams.get(*stream).ok_or(Error::InvalidRange)?,
                    frame.resident.streams[*stream].instance,
                    &frame.resident.streams[*stream].buffer,
                    &frame.resident.streams[*stream].desc,
                    frame.resident.streams[*stream].incoming,
                )?,
                range.clone(),
            )?;
        }
        r.draw(draw.vertices.clone(), draw.instances.clone())?;
    }
    let _ = read_minmax;
    r.finish()
}

fn checked<B: ResourceBackend>(
    pass: &crate::graph::execution::CheckedPass<'_, '_, B>,
    slot: TextureSlot,
    kind: AccessKind,
    contract: TextureAccessContract,
) -> Result<()> {
    pass.texture(
        slot,
        TextureRange::new(fluxel_rhi::api::TextureAspects::COLOR, 0..1, 0..1),
        kind,
        contract,
    )
    .map(|_| ())
}
fn checked_depth<B: ResourceBackend>(
    pass: &crate::graph::execution::CheckedPass<'_, '_, B>,
    slot: TextureSlot,
) -> Result<()> {
    pass.texture(
        slot,
        TextureRange::new(fluxel_rhi::api::TextureAspects::DEPTH, 0..1, 0..1),
        AccessKind::Read,
        depth_read(),
    )
    .map(|_| ())
}
fn checked_mips<B: ResourceBackend>(
    pass: &crate::graph::execution::CheckedPass<'_, '_, B>,
    slot: TextureSlot,
    mips: u32,
) -> Result<()> {
    pass.texture(
        slot,
        TextureRange::new(fluxel_rhi::api::TextureAspects::COLOR, 0..mips, 0..1),
        AccessKind::Read,
        shader_read(),
    )
    .map(|_| ())
}
fn read_side_index(slots: &OitRecordSlots, slot: TextureSlot) -> Result<usize> {
    slots
        .ping_pong
        .iter()
        .position(|side| side.min_max == slot)
        .ok_or(Error::InvalidState)
}
fn validate_group<B: ResourceBackend>(
    group: &OitSampledGroupRef<'_, B>,
    expected: &[ResourceInstanceId],
) -> Result<()> {
    if group.textures == expected {
        Ok(())
    } else {
        Err(Error::InvalidState)
    }
}
fn color<'a, B: ResourceBackend>(
    view: &'a TextureView<B>,
    load: LoadOp<ClearColor>,
) -> ColorAttachment<'a, B> {
    ColorAttachment {
        view: AttachmentView::Texture(view),
        resolve_target: None,
        load,
        store: StoreOp::Store,
    }
}
fn desc<'a, B: ResourceBackend>(
    label: &'a str,
    colors: &'a [Option<ColorAttachment<'a, B>>],
    depth: Option<DepthStencilAttachment<'a, B>>,
    extent: [u32; 2],
) -> RasterPassDesc<'a, B> {
    RasterPassDesc {
        label: Some(label),
        colors,
        depth_stencil: depth,
        render_area: Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    }
}
fn shader_read() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        layout: ImageLayout::ShaderReadOnly,
        descriptor_usage: TextureUsage::TEXTURE_BINDING,
    }
}
fn color_write() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::COLOR_OUTPUT,
            accesses: AccessMask::COLOR_WRITE,
        },
        layout: ImageLayout::ColorAttachment,
        descriptor_usage: TextureUsage::RENDER_ATTACHMENT,
    }
}
fn color_rw() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::COLOR_OUTPUT,
            accesses: AccessMask::COLOR_READ | AccessMask::COLOR_WRITE,
        },
        layout: ImageLayout::ColorAttachment,
        descriptor_usage: TextureUsage::RENDER_ATTACHMENT,
    }
}
fn depth_read() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::DEPTH_READ,
        },
        layout: ImageLayout::DepthStencilAttachment,
        descriptor_usage: TextureUsage::RENDER_ATTACHMENT,
    }
}
fn vertex_read() -> BufferAccessContract {
    BufferAccessContract {
        uses: fluxel_rhi::api::BufferUses {
            stages: StageMask::VERTEX_INPUT,
            accesses: AccessMask::VERTEX_READ,
        },
        descriptor_usage: fluxel_rhi::api::BufferUsage::VERTEX,
    }
}
fn uniform_read() -> BufferAccessContract {
    BufferAccessContract {
        uses: fluxel_rhi::api::BufferUses {
            stages: StageMask::VERTEX | StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        descriptor_usage: fluxel_rhi::api::BufferUsage::UNIFORM,
    }
}
