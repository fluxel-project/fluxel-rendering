//! Raster recording for the presented Forward product.
use super::*;

pub(super) fn record<B: ResourceBackend + RasterBackend>(
    plan: &SurfaceForwardPlan<B>,
    device: Option<&Device<B>>,
    encoder: &mut CommandEncoder<B>,
    frame: &SurfaceForwardFrame<B>,
    image: &SurfaceImage<B>,
) -> Result<()> {
    if image.device() != plan.id.device {
        return Err(Error::WrongDevice);
    }
    if !image.is_current() {
        return Err(Error::SurfaceOutdated);
    }
    if image.extent() != frame.extent || image.format() != frame.color_format {
        return Err(Error::InvalidRange);
    }
    validate_surface_topology(plan.topology, frame.draws.len(), frame.draw_batches.len())?;
    validate_surface_depth(frame)?;
    validate_depth_topology(plan.topology, frame.frame.depth.is_some())?;
    validate_surface_batches(frame)?;
    let bindings = plan.bindings(frame, image)?;
    for (declared, (_, product)) in plan.indirect.iter().zip(&frame.indirect) {
        declared
            .graph
            .validate_raster_read(plan.raster_pass, &bindings, product)?;
    }
    for (declared, (_, step)) in plan.particles.iter().zip(&frame.particles) {
        declared
            .graph
            .validate_vertex_read(plan.raster_pass, &bindings, step)?;
    }
    let pass = bindings.pass(plan.raster_pass)?;
    for draw in &frame.draws {
        validate_draw(draw, plan.stream_slots.len())?;
        if !draw.pipelines_on(plan.id.device)
            || draw
                .bind_groups
                .iter()
                .any(|group| group.group.device() != plan.id.device)
        {
            return Err(Error::WrongDevice);
        }
        for (stream_index, range) in draw
            .vertex_streams
            .iter()
            .zip(surface_draw_accesses(frame, draw)?)
        {
            if let Some(range) = range {
                pass.buffer(
                    *plan
                        .stream_slots
                        .get(*stream_index)
                        .ok_or(Error::InvalidRange)?,
                    range,
                    AccessKind::Read,
                    vertex_read_uses(),
                )?;
            }
        }
        for group in &draw.bind_groups {
            for range in &group.uniform_ranges {
                pass.buffer(
                    plan.uniform_slot,
                    range.clone(),
                    AccessKind::Read,
                    shader_read_uses(),
                )?;
            }
        }
        for texture in &draw.textures {
            pass.texture(
                *plan
                    .texture_slots
                    .get(*texture)
                    .ok_or(Error::InvalidRange)?,
                material_mips_layers(&frame.resident.textures[*texture].desc),
                AccessKind::Read,
                material_read_contract(&frame.resident.textures[*texture].desc)
                    .ok_or(Error::InvalidState)?,
            )?;
        }
    }
    if frame.environment.is_some() {
        pass.texture(
            plan.environment_slot.ok_or(Error::InvalidState)?,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
    }
    if let Some(owner) = frame.specular.as_ref() {
        owner.validate_for_sampling(plan.id.device)?;
        pass.texture(
            plan.specular_radiance_slot.ok_or(Error::InvalidState)?,
            all_color_mips(owner.radiance.desc.mip_levels),
            AccessKind::Read,
            texture_read_uses(),
        )?;
        pass.texture(
            plan.specular_brdf_lut_slot.ok_or(Error::InvalidState)?,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
        pass.buffer(
            plan.specular_view_uniform_slot.ok_or(Error::InvalidState)?,
            0..crate::frame_data::specular_view::SPECULAR_VIEW_BYTES,
            AccessKind::Read,
            shader_read_uses(),
        )?;
    }
    pass.surface_texture(
        plan.surface_target_slot,
        whole(),
        AccessKind::Write,
        if plan.topology == FrameTopology::SortedForward {
            color_read_write_uses()
        } else {
            color_write_uses()
        },
    )?;
    let depth_stencil = match (plan.depth_slot, frame.frame.depth.as_ref()) {
        (None, None) => None,
        (Some(slot), Some(owner)) => {
            if owner.device != plan.id.device
                || owner.texture.device() != plan.id.device
                || owner.view.device() != plan.id.device
            {
                return Err(Error::WrongDevice);
            }
            pass.texture(
                slot,
                whole_depth(),
                AccessKind::Write,
                if plan.topology == FrameTopology::SortedForward {
                    depth_read_write_uses()
                } else {
                    depth_write_uses()
                },
            )?;
            Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&owner.view),
                depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: true,
            })
        }
        _ => return Err(Error::InvalidState),
    };
    pass.transition(encoder)?;
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Surface(image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut raster = encoder.begin_raster(&RasterPassDesc {
        label: Some(if plan.topology == FrameTopology::SortedForward {
            "fluxel forward sorted surface"
        } else {
            "fluxel forward surface"
        }),
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
    for (batch, range) in frame.draw_batches.iter().enumerate() {
        let draws = frame.draws.get(range.clone()).ok_or(Error::InvalidRange)?;
        raster.set_pipeline(&draws.first().ok_or(Error::InvalidRange)?.pipeline)?;
        for (offset, draw) in draws.iter().enumerate() {
            if draw.batch != batch {
                return Err(Error::InvalidState);
            }
            let draw_index = range.start.checked_add(offset).ok_or(Error::InvalidRange)?;
            let particle = frame
                .particles
                .iter()
                .find(|(index, _)| *index == draw_index);
            let indirect = crate::residency::indirect_for_draw(&frame.indirect, draw_index);
            if let Some(preliminary) = draw.preliminary_pipeline.as_ref() {
                raster.set_pipeline(preliminary)?;
                draw_one(
                    device,
                    &mut raster,
                    &frame.resident,
                    draw,
                    particle.map(|(_, step)| step),
                    indirect,
                )?;
                raster.set_pipeline(&draw.pipeline)?;
            }
            draw_one(
                device,
                &mut raster,
                &frame.resident,
                draw,
                particle.map(|(_, step)| step),
                indirect,
            )?;
        }
    }
    raster.finish()
}

fn draw_one<B: ResourceBackend + RasterBackend>(
    device: Option<&Device<B>>,
    raster: &mut fluxel_rhi::api::RasterPass<'_, B>,
    resident: &crate::residency::ForwardResidentBuffers<B>,
    draw: &crate::residency::ForwardDraw<B>,
    particle: Option<&crate::renderer::particle_update::ParticleStep<B>>,
    indirect: Option<&crate::renderer::indirect_producer::IndirectProduct<B>>,
) -> Result<()> {
    if let Some(step) = particle {
        return super::shared_record::bind_and_draw_particle(raster, resident, draw, step);
    }
    match indirect {
        Some(product) => super::shared_record::bind_and_draw_indirect(
            device.ok_or(Error::InvalidState)?,
            raster,
            resident,
            draw,
            product,
        ),
        None => super::shared_record::bind_and_draw(raster, resident, draw),
    }
}
