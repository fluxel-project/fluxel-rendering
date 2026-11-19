//! Deferred output selection after the shared Graph prefix.

use super::{prefix::DeferredGraphPrefix, *};
use crate::graph::definition::TextureHandle;

/// Appends the existing Deferred output terminal.
///
/// `final_source` is a version already declared in `prefix.graph`, paired with
/// the pass that produces it.  The ordinary path defaults to the lighting
/// scene color; a future OIT contribution can select its own final version
/// without a second import or scheduler.
pub(super) fn declare_terminal(
    mut prefix: DeferredGraphPrefix,
    final_source: Option<(TextureHandle, GraphPass)>,
) -> Result<DeferredDeclaration> {
    let (final_source, final_dependency) =
        final_source.unwrap_or((prefix.scene_color, prefix.lighting_pass));
    let (composite_pass, output) = if let Some(readback) = prefix.readback {
        let target = prefix.target.ok_or(Error::InvalidState)?;
        let (composite_pass, target) = prefix
            .graph
            .add_pass("deferred-scene-color-composite", |pass| {
                pass.depends_on(final_dependency)?;
                pass.read_texture_with_use(
                    final_source,
                    whole_color(),
                    sampled_read().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
                let target = pass.write_texture_with_use(
                    target,
                    whole_color(),
                    color_write().uses,
                    ImageLayout::ColorAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )?;
                Ok(target)
            })
            .map_err(|_| Error::InvalidState)?;
        let (readback_pass, ()) = prefix
            .graph
            .add_pass("deferred-readback", |pass| {
                let length = u64::from(prefix.row_pitch.ok_or(GraphError::InvalidRange)?)
                    .checked_mul(u64::from(prefix.extent[1]))
                    .ok_or(GraphError::InvalidRange)?;
                pass.read_texture_with_use(
                    target,
                    whole_color(),
                    texture_copy_read().uses,
                    ImageLayout::CopySrc,
                    TextureUsage::COPY_SRC,
                )?;
                pass.write_with_use(
                    readback,
                    0..length,
                    copy_write().uses,
                    BufferUsage::COPY_DST,
                )?;
                pass.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        (
            Some(composite_pass),
            DeferredOutputDeclaration::Offscreen { readback_pass },
        )
    } else {
        let surface_target = prefix.surface_target.ok_or(Error::InvalidState)?;
        let (composite_pass, ()) = prefix
            .graph
            .add_pass("deferred-surface-scene-color-composite", |pass| {
                pass.depends_on(final_dependency)?;
                pass.read_texture_with_use(
                    final_source,
                    whole_color(),
                    sampled_read().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
                pass.write_texture_with_use(
                    surface_target,
                    whole_color(),
                    color_write().uses,
                    ImageLayout::ColorAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )?;
                // Presentation consumes this lease after the Graph's command
                // steps; retain the terminal write in the live Graph.
                pass.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        (Some(composite_pass), DeferredOutputDeclaration::Surface)
    };

    Ok(DeferredDeclaration {
        graph: prefix.graph,
        lane: prefix.lane,
        upload_pass: prefix.upload_pass,
        mip_generation: prefix.mip_generation,
        indirect: prefix.indirect,
        deformation_pass: prefix.deformation_pass,
        geometry_pass: prefix.geometry_pass,
        resolve_pass: prefix.resolve_pass,
        lighting_pass: prefix.lighting_pass,
        composite_pass,
        output,
        resident_upload_slot: BufferSlot::from(prefix.resident_upload),
        frame_upload_slot: BufferSlot::from(prefix.frame_upload),
        uniform_slot: BufferSlot::from(prefix.uniform),
        stream_slots: prefix.streams.into_iter().map(BufferSlot::from).collect(),
        texture_slots: prefix
            .materials
            .into_iter()
            .map(TextureSlot::from)
            .collect(),
        environment_upload_slot: prefix.environment_upload.map(BufferSlot::from),
        environment_slot: prefix.environment.map(TextureSlot::from),
        specular_upload_slot: prefix.specular_upload.map(BufferSlot::from),
        specular_view_uniform_slot: prefix.specular_view_uniform.map(BufferSlot::from),
        specular_radiance_slot: prefix.specular_radiance.map(TextureSlot::from),
        specular_brdf_lut_slot: prefix.specular_brdf_lut.map(TextureSlot::from),
        gbuffer_slots: std::array::from_fn(|index| TextureSlot::from(prefix.gbuffer[index])),
        effects_slot: prefix.effects.map(TextureSlot::from),
        diffuse_roughness_slot: prefix.diffuse_roughness.map(TextureSlot::from),
        ambient_occlusion_slot: prefix.ambient_occlusion.map(TextureSlot::from),
        msaa_gbuffer_slots: prefix
            .msaa_gbuffer
            .map(|items| items.map(TextureSlot::from)),
        msaa_effects_slot: prefix.msaa_effects.map(TextureSlot::from),
        msaa_diffuse_roughness_slot: prefix.msaa_diffuse_roughness.map(TextureSlot::from),
        msaa_ambient_occlusion_slot: prefix.msaa_ambient_occlusion.map(TextureSlot::from),
        // This remains the lighting owner/version.  A final OIT source needs
        // its own actual owner, view, and composite binding before it can be
        // recorded; terminal selection alone must not retarget this slot.
        scene_color_slot: Some(TextureSlot::from(prefix.scene_color)),
        surface_target_slot: prefix.surface_target.map(TextureSlot::from),
        target_slot: prefix.target.map(TextureSlot::from),
        readback_slot: prefix.readback.map(BufferSlot::from),
        deformation_slots: prefix.deformation_handles.map(
            |(upload, source, product, parameters)| {
                [
                    BufferSlot::from(upload),
                    BufferSlot::from(source),
                    BufferSlot::from(product),
                    BufferSlot::from(parameters),
                ]
            },
        ),
        extent: prefix.extent,
        uniform_alignment: prefix.uniform_alignment,
        transform_bytes: prefix.transform_bytes,
    })
}
