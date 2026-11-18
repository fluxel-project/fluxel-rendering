//! Graph declaration for the shader-based Deferred MSAA resolve.

use super::*;
use crate::graph::definition::TextureHandle;

pub(super) fn declare(
    graph: &mut GraphBuilder,
    geometry: GraphPass,
    source: [TextureHandle; 3],
    source_effects: Option<TextureHandle>,
    destination: [TextureHandle; 3],
    destination_effects: Option<TextureHandle>,
) -> std::result::Result<(GraphPass, ([TextureHandle; 3], Option<TextureHandle>)), ()> {
    graph
        .add_pass("deferred-msaa-resolve", |pass| {
            pass.depends_on(geometry)?;
            for (index, texture) in source.iter().enumerate() {
                pass.read_texture_with_use(
                    *texture,
                    if index == 2 {
                        whole_depth()
                    } else {
                        whole_color()
                    },
                    sampled_read().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
            }
            if let Some(effects) = source_effects {
                pass.read_texture_with_use(
                    effects,
                    whole_color(),
                    sampled_read().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
            }
            let mut gbuffer = destination;
            for (index, texture) in gbuffer.iter_mut().enumerate() {
                *texture = pass.write_texture_with_use(
                    *texture,
                    if index == 2 {
                        whole_depth()
                    } else {
                        whole_color()
                    },
                    if index == 2 {
                        depth_write().uses
                    } else {
                        color_write().uses
                    },
                    if index == 2 {
                        ImageLayout::DepthStencilAttachment
                    } else {
                        ImageLayout::ColorAttachment
                    },
                    TextureUsage::RENDER_ATTACHMENT,
                )?;
            }
            let effects = destination_effects
                .map(|effects| {
                    pass.write_texture_with_use(
                        effects,
                        whole_color(),
                        color_write().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                })
                .transpose()?;
            Ok((gbuffer, effects))
        })
        .map_err(|_| ())
}
