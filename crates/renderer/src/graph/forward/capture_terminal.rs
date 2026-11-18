//! Capture background initialization and offscreen readback declaration.

use super::*;

pub(super) fn declare_capture_background_copy(
    graph: &mut GraphBuilder,
    source: crate::graph::definition::TextureHandle,
    destination: crate::graph::definition::TextureHandle,
) -> std::result::Result<(GraphPass, crate::graph::definition::TextureHandle), ForwardPlanBuildError>
{
    graph
        .add_pass("forward-capture-background-copy", |pass| {
            pass.read_texture_with_use(
                source,
                whole(),
                texture_copy_read_uses().uses,
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            )?;
            pass.write_texture_with_use(
                destination,
                whole(),
                texture_copy_write_uses().uses,
                ImageLayout::CopyDst,
                TextureUsage::COPY_DST,
            )
        })
        .map_err(|_| Error::InvalidState.into())
}

pub(super) fn record_background_copy<B: CommandBackend>(
    bindings: &FrameBindings<'_, B>,
    pass_id: GraphPass,
    encoder: &mut CommandEncoder<B>,
    source: TextureSlot,
    destination: TextureSlot,
    extent: [u32; 2],
) -> Result<()> {
    let pass = bindings.pass(pass_id)?;
    let source = pass.texture(source, whole(), AccessKind::Read, texture_copy_read_uses())?;
    let destination = pass.texture(
        destination,
        whole(),
        AccessKind::Write,
        texture_copy_write_uses(),
    )?;
    pass.transition(encoder)?;
    let extent = [extent[0], extent[1], 1];
    encoder.copy_texture(
        TextureRegion {
            texture: source,
            mip: 0,
            origin: [0, 0, 0],
            extent,
            aspect: TextureAspects::COLOR,
        },
        TextureRegion {
            texture: destination,
            mip: 0,
            origin: [0, 0, 0],
            extent,
            aspect: TextureAspects::COLOR,
        },
    )
}

pub(super) fn declare_readback(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    target: crate::graph::definition::TextureHandle,
    readback: crate::graph::definition::BufferHandle,
) -> std::result::Result<GraphPass, ForwardPlanBuildError> {
    let length = u64::from(spec.readback_row_pitch)
        .checked_mul(u64::from(spec.extent[1]))
        .ok_or(Error::InvalidRange)?;
    graph
        .add_pass("forward-capture-readback", |pass| {
            pass.read_texture_with_use(
                target,
                whole(),
                texture_copy_read_uses().uses,
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            )?;
            pass.write_with_use(
                readback,
                0..length,
                copy_write_uses().uses,
                BufferUsage::COPY_DST,
            )?;
            pass.side_effect();
            Ok(())
        })
        .map(|(pass, ())| pass)
        .map_err(|_| Error::InvalidState.into())
}
