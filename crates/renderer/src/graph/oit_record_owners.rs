//! Bind command-owner identities to the frame's actual validated resources.

use super::*;

pub(super) fn validate<B: ResourceBackend>(
    bindings: &FrameBindings<'_, B>,
    slots: &OitRecordSlots,
    owners: &OitOwners<B>,
    depth: &OitDepthTargetRef<'_, B>,
    frame: &ForwardFrame<B>,
    draw_slots: &OitDrawSlots<'_>,
    commands: &OitRecordCommands<'_, B>,
) -> Result<()> {
    for (slot, owner) in slots.ping_pong.iter().zip(&owners.ping_pong) {
        for (slot, owner) in [
            (slot.min_max, &owner.min_max),
            (slot.front, &owner.front),
            (slot.back, &owner.back),
        ] {
            bindings.require_texture_owner(slot, owner.instance, &owner.texture)?;
        }
    }
    bindings.require_texture_owner(
        slots.back_accumulator,
        owners.back_accumulator.instance,
        &owners.back_accumulator.texture,
    )?;
    bindings.require_texture_owner(slots.opaque_depth, depth.instance, &depth.texture)?;
    bindings.require_texture_owner(
        slots.opaque_color,
        commands.opaque_color.instance,
        commands.opaque_color.texture,
    )?;
    bindings.require_texture_owner(
        slots.output,
        commands.output.instance,
        commands.output.texture,
    )?;
    // OIT uses these existing draw bind groups. Match their retained frame
    // owners independently of the per-pass access authorization in material().
    for item in commands.draws {
        let draw = frame.draws.get(item.draw).ok_or(Error::InvalidRange)?;
        for (&index, range) in draw
            .vertex_streams
            .iter()
            .zip(crate::graph::forward::forward_draw_accesses(frame, draw)?)
        {
            if range.is_none() {
                continue;
            }
            let slot = *draw_slots.streams.get(index).ok_or(Error::InvalidRange)?;
            let owner = frame
                .resident
                .streams
                .get(index)
                .ok_or(Error::InvalidRange)?;
            bindings.canonical_buffer(
                slot,
                owner.instance,
                &owner.buffer,
                &owner.desc,
                owner.incoming,
            )?;
        }
        for &index in &draw.textures {
            let slot = *draw_slots.textures.get(index).ok_or(Error::InvalidRange)?;
            let owner = frame
                .resident
                .textures
                .get(index)
                .ok_or(Error::InvalidRange)?;
            bindings.canonical_texture(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout,
            )?;
        }
    }
    bindings.require_buffer_owner(
        draw_slots.uniform,
        frame.frame.uniform_instance,
        &frame.frame.uniform,
    )?;
    if let Some(owner) = frame.environment.as_ref() {
        bindings.require_texture_owner(
            draw_slots.environment.ok_or(Error::InvalidState)?,
            owner.texture_instance,
            &owner.texture,
        )?;
    }
    if let Some(owner) = frame.specular.as_ref() {
        bindings.require_texture_owner(
            draw_slots.specular_radiance.ok_or(Error::InvalidState)?,
            owner.radiance.instance,
            &owner.radiance.texture,
        )?;
        bindings.require_texture_owner(
            draw_slots.specular_brdf_lut.ok_or(Error::InvalidState)?,
            owner.brdf_lut.instance,
            &owner.brdf_lut.texture,
        )?;
        bindings.require_buffer_owner(
            draw_slots
                .specular_view_uniform
                .ok_or(Error::InvalidState)?,
            owner.view_uniform_instance,
            &owner.view_uniform,
        )?;
    }
    Ok(())
}
