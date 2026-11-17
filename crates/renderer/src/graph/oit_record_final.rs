use super::*;
pub(super) fn clear<B: RasterBackend + ResourceBackend>(
    encoder: &mut CommandEncoder<B>,
    bindings: &FrameBindings<'_, B>,
    id: crate::graph::definition::GraphPass,
    slots: &OitSideSlots,
    side: &crate::residency::OitPingPongOwners<B>,
    extent: [u32; 2],
    minmax: [f32; 4],
) -> Result<()> {
    let pass = bindings.pass(id)?;
    checked_owner(
        &pass,
        slots.min_max,
        AccessKind::Write,
        color_write(),
        &side.min_max.texture,
    )?;
    checked_owner(
        &pass,
        slots.front,
        AccessKind::Write,
        color_write(),
        &side.front.texture,
    )?;
    checked_owner(
        &pass,
        slots.back,
        AccessKind::Write,
        color_write(),
        &side.back.texture,
    )?;
    pass.transition(encoder)?;
    let colors = [
        Some(color(
            &side.min_max.view,
            LoadOp::Clear(ClearColor::Float(minmax)),
        )),
        Some(color(
            &side.front.view,
            LoadOp::Clear(ClearColor::Float([0.0; 4])),
        )),
        Some(color(
            &side.back.view,
            LoadOp::Clear(ClearColor::Float([0.0; 4])),
        )),
    ];
    encoder
        .begin_raster(&desc("oit-clear", &colors, None, extent))?
        .finish()
}
pub(super) fn checked_owner<B: ResourceBackend>(
    pass: &crate::graph::execution::CheckedPass<'_, '_, B>,
    slot: TextureSlot,
    kind: AccessKind,
    contract: TextureAccessContract,
    owner: &fluxel_rhi::api::Texture<B>,
) -> Result<()> {
    let bound = pass.texture(
        slot,
        TextureRange::new(fluxel_rhi::api::TextureAspects::COLOR, 0..1, 0..1),
        kind,
        contract,
    )?;
    if std::ptr::eq(bound, owner) {
        Ok(())
    } else {
        Err(Error::InvalidState)
    }
}
pub(super) fn final_composite<B: RasterBackend + ResourceBackend>(
    encoder: &mut CommandEncoder<B>,
    bindings: &FrameBindings<'_, B>,
    id: crate::graph::definition::GraphPass,
    front: TextureSlot,
    back: TextureSlot,
    output: TextureSlot,
    front_view: &TextureView<B>,
    front_texture: &fluxel_rhi::api::Texture<B>,
    back_texture: &fluxel_rhi::api::Texture<B>,
    output_view: &TextureView<B>,
    output_texture: &fluxel_rhi::api::Texture<B>,
    command: &OitFullscreenCommand<'_, B>,
    extent: [u32; 2],
) -> Result<()> {
    let pass = bindings.pass(id)?;
    checked_owner(&pass, front, AccessKind::Read, shader_read(), front_texture)?;
    checked_owner(&pass, back, AccessKind::Read, shader_read(), back_texture)?;
    checked_owner(
        &pass,
        output,
        AccessKind::Write,
        color_write(),
        output_texture,
    )?;
    pass.transition(encoder)?;
    let colors = [Some(color(
        output_view,
        LoadOp::Clear(ClearColor::Float([0.0; 4])),
    ))];
    let mut r = encoder.begin_raster(&desc("oit-final-composite", &colors, None, extent))?;
    r.set_pipeline(command.pipeline)?;
    r.set_bind_group(0, command.bind_group.group, &[])?;
    let _ = front_view;
    r.draw(0..3, 0..1)?;
    r.finish()
}
