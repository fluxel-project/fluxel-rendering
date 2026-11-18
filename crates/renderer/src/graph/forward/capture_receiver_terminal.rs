//! Output terminal appended to a combined capture Graph.

use super::*;

/// Terminal selected by the enclosing combined Graph, never a second plan.
pub(in crate::graph) enum ForwardReceiverTerminal {
    Offscreen {
        pass: GraphPass,
    },
    HdrReadback(super::hdr_output::HdrReadbackTerminal),
    Surface {
        pass: GraphPass,
        slot: TextureSlot,
        extent: [u32; 2],
        format: TextureFormat,
    },
}

pub(in crate::graph) fn declare_receiver_terminal<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    receiver: &ForwardReceiverComposition,
    output: Option<&crate::residency::ForwardTexture<B>>,
    image: Option<&SurfaceImage<B>>,
) -> std::result::Result<ForwardReceiverTerminal, ForwardPlanBuildError> {
    match image {
        None if output.is_some() => super::hdr_output::declare_hdr_readback(
            graph,
            receiver.final_handle,
            output.ok_or(Error::InvalidState)?,
            receiver.handles.readback,
            receiver.spec.extent,
            receiver.spec.readback_row_pitch,
        )
        .map(ForwardReceiverTerminal::HdrReadback)
        .map_err(Into::into),
        None => {
            let length = u64::from(receiver.spec.readback_row_pitch)
                .checked_mul(u64::from(receiver.spec.extent[1]))
                .ok_or(Error::InvalidRange)?;
            let (pass, ()) = graph
                .add_pass("deferred-forward-capture-readback", |pass| {
                    pass.read_texture_with_use(
                        receiver.final_handle,
                        whole(),
                        texture_copy_read_uses().uses,
                        ImageLayout::CopySrc,
                        TextureUsage::COPY_SRC,
                    )?;
                    pass.write_with_use(
                        receiver.handles.readback,
                        0..length,
                        copy_write_uses().uses,
                        BufferUsage::COPY_DST,
                    )?;
                    pass.side_effect();
                    Ok(())
                })
                .map_err(|_| Error::InvalidState)?;
            Ok(ForwardReceiverTerminal::Offscreen { pass })
        }
        Some(image) => {
            if image.extent() != receiver.spec.extent {
                return Err(Error::InvalidRange.into());
            }
            let surface = graph
                .import_surface_texture(TextureDesc {
                    label: Some("fluxel deferred forward capture surface lease".into()),
                    dimension: fluxel_rhi::api::TextureDimension::D2,
                    extent: [receiver.spec.extent[0], receiver.spec.extent[1], 1],
                    array_layers: 1,
                    format: image.format(),
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsage::RENDER_ATTACHMENT,
                    view_formats: Vec::new(),
                })
                .map_err(|_| Error::InvalidState)?;
            let (pass, ()) = graph
                .add_pass("deferred-forward-capture-surface-composite", |pass| {
                    pass.read_texture_with_use(
                        receiver.final_handle,
                        whole(),
                        texture_read_uses().uses,
                        ImageLayout::ShaderReadOnly,
                        TextureUsage::TEXTURE_BINDING,
                    )?;
                    pass.write_texture_with_use(
                        surface,
                        whole(),
                        color_write_uses().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )?;
                    pass.side_effect();
                    Ok(())
                })
                .map_err(|_| Error::InvalidState)?;
            Ok(ForwardReceiverTerminal::Surface {
                pass,
                slot: surface.into(),
                extent: receiver.spec.extent,
                format: image.format(),
            })
        }
    }
}
