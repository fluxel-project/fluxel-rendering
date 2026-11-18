//! Terminal encoding of an already-composed linear scene product for readback.

use super::*;
use crate::{
    graph::{
        ResourceInstanceId,
        definition::{BufferHandle, TextureHandle},
    },
    residency::ForwardTexture,
};
use fluxel_rhi::api::TextureDimension;

#[derive(Clone, Copy)]
pub(in crate::graph) struct HdrReadbackTerminal {
    pub(in crate::graph) composite_pass: GraphPass,
    pub(in crate::graph) output_slot: TextureSlot,
    pub(in crate::graph) readback_pass: GraphPass,
}

pub(in crate::graph) fn declare_hdr_readback<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    scene: TextureHandle,
    output: &ForwardTexture<B>,
    buffer: BufferHandle,
    extent: [u32; 2],
    row_pitch: u32,
) -> Result<HdrReadbackTerminal> {
    if output.desc.format != TextureFormat::Rgba8Unorm
        || output.desc.dimension != TextureDimension::D2
        || output.desc.extent != [extent[0], extent[1], 1]
        || output.desc.array_layers != 1
        || output.desc.mip_levels != 1
        || output.desc.samples != 1
        || !output
            .desc
            .usage
            .contains(TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC)
        || u64::from(row_pitch) < u64::from(extent[0]) * 4
    {
        return Err(Error::InvalidState);
    }
    let length = u64::from(row_pitch)
        .checked_mul(u64::from(extent[1]))
        .ok_or(Error::InvalidRange)?;
    let target = graph
        .import_texture_with_state(output.desc.clone(), output.incoming, output.layout)
        .map_err(|_| Error::InvalidState)?;
    let (composite_pass, target) = graph
        .add_pass("hdr-output-composite", |pass| {
            pass.read_texture_with_use(
                scene,
                whole(),
                texture_read_uses().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.write_texture_with_use(
                target,
                whole(),
                color_write_uses().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )
        })
        .map_err(|_| Error::InvalidState)?;
    let (readback_pass, ()) = graph
        .add_pass("hdr-output-readback", |pass| {
            pass.read_texture_with_use(
                target,
                whole(),
                texture_copy_read_uses().uses,
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            )?;
            pass.write_with_use(
                buffer,
                0..length,
                copy_write_uses().uses,
                BufferUsage::COPY_DST,
            )?;
            pass.side_effect();
            Ok(())
        })
        .map_err(|_| Error::InvalidState)?;
    Ok(HdrReadbackTerminal {
        composite_pass,
        output_slot: target.into(),
        readback_pass,
    })
}

impl HdrReadbackTerminal {
    pub(in crate::graph) fn append_binding<'a, B: ResourceBackend>(
        self,
        output: &'a ForwardTexture<B>,
        textures: &mut Vec<BoundTexture<'a, B>>,
    ) {
        textures.push(BoundTexture::new(
            self.output_slot,
            output.instance,
            &output.texture,
            &output.desc,
            output.incoming,
            output.layout,
        ));
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::graph) fn record<B: ResourceBackend + RasterBackend>(
        self,
        device: &Device<B>,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        source_slot: TextureSlot,
        source_instance: ResourceInstanceId,
        frame: &ForwardFrame<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()> {
        let output = frame.frame.output.as_ref().ok_or(Error::InvalidState)?;
        if output.instance == source_instance || composite.target_instance() != source_instance {
            return Err(Error::InvalidState);
        }
        composite.validate_offscreen(device, frame.extent, output.desc.format)?;
        bindings.require_texture_owner(self.output_slot, output.instance, &output.texture)?;
        let pass = bindings.pass(self.composite_pass)?;
        pass.texture(source_slot, whole(), AccessKind::Read, texture_read_uses())?;
        pass.texture(
            self.output_slot,
            whole(),
            AccessKind::Write,
            color_write_uses(),
        )?;
        pass.transition(encoder)?;
        super::hdr_record::record_composite_raster(
            encoder,
            AttachmentView::Texture(&output.view),
            frame.extent,
            composite,
            "fluxel HDR output composite",
        )
    }
}
