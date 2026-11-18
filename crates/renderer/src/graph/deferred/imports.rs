//! Concrete-owner imports for one Deferred Graph declaration.

pub(super) use super::shared_imports::SharedDeferredImports;
use super::{record_view::DeferredRecordView, *};
use crate::graph::definition::{BufferHandle, TextureHandle};

pub(super) struct DeferredImports {
    pub(super) resident_upload: BufferHandle,
    pub(super) frame_upload: BufferHandle,
    pub(super) uniform: BufferHandle,
    pub(super) streams: Vec<BufferHandle>,
    pub(super) materials: Vec<TextureHandle>,
    pub(super) environment_upload: Option<BufferHandle>,
    pub(super) environment: Option<TextureHandle>,
    pub(super) specular_upload: Option<BufferHandle>,
    pub(super) specular_view_uniform: Option<BufferHandle>,
    pub(super) specular_radiance: Option<TextureHandle>,
    pub(super) specular_brdf_lut: Option<TextureHandle>,
    pub(super) gbuffer: Vec<TextureHandle>,
    pub(super) effects: Option<TextureHandle>,
    pub(super) msaa_gbuffer: Option<[TextureHandle; 3]>,
    pub(super) msaa_effects: Option<TextureHandle>,
    pub(super) target: Option<TextureHandle>,
    pub(super) scene_color: Option<TextureHandle>,
    pub(super) surface_target: Option<TextureHandle>,
    pub(super) readback: Option<BufferHandle>,
    pub(super) row_pitch: Option<u32>,
    pub(super) deformation: Option<(BufferHandle, BufferHandle, BufferHandle, BufferHandle)>,
}

pub(super) fn import<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    frame: &DeferredRecordView<'_, B>,
    fresh_mip_chains: &[bool],
    aliases: Option<&[Option<TextureHandle>]>,
    mut shared: Option<&mut SharedDeferredImports>,
) -> Result<DeferredImports> {
    super::capture_aliases::validate(&frame, aliases)?;
    let resident_upload = graph
        .import_buffer_desc(
            frame.resident.upload_desc.clone(),
            frame.resident.upload_incoming,
        )
        .map_err(|_| Error::InvalidState)?;
    let frame_upload = graph
        .import_buffer_desc(frame.frame.upload_desc.clone(), frame.frame.upload_incoming)
        .map_err(|_| Error::InvalidState)?;
    let uniform = graph
        .import_buffer_desc(
            frame.frame.uniform_desc.clone(),
            frame.frame.uniform_incoming,
        )
        .map_err(|_| Error::InvalidState)?;
    let streams = frame
        .resident
        .streams
        .iter()
        .enumerate()
        .map(|(index, stream)| match shared.as_deref_mut() {
            Some(shared) => shared.deferred_stream(
                graph,
                frame.resident.device,
                stream,
                frame
                    .resident_upload
                    .vertices
                    .iter()
                    .any(|copy| copy.stream == index),
            ),
            None => graph
                .import_buffer_desc(stream.desc.clone(), stream.incoming)
                .map_err(|_| Error::InvalidState),
        })
        .collect::<Result<Vec<_>>>()?;
    if fresh_mip_chains.len() != frame.resident.textures.len() {
        return Err(Error::InvalidState);
    }
    let materials = frame
        .resident
        .textures
        .iter()
        .zip(fresh_mip_chains)
        .enumerate()
        .map(|(index, (texture, fresh_mip_chain))| {
            if let Some(alias) = aliases.and_then(|items| items[index]) {
                return Ok(alias);
            }
            if let Some(shared) = shared.as_deref_mut() {
                return shared.deferred_texture(
                    graph,
                    frame.resident.device,
                    texture,
                    frame
                        .resident_upload
                        .textures
                        .iter()
                        .any(|copy| copy.texture == index),
                    *fresh_mip_chain,
                );
            }
            (if *fresh_mip_chain {
                graph.create_texture(texture.desc.clone())
            } else {
                graph.import_texture_with_state(
                    texture.desc.clone(),
                    texture.incoming,
                    texture.layout,
                )
            })
            .map_err(|_| Error::InvalidState)
        })
        .collect::<Result<Vec<_>>>()?;
    let environment_upload = frame
        .environment
        .map(|owner| graph.import_buffer_desc(owner.upload_desc.clone(), owner.upload_incoming))
        .transpose()
        .map_err(|_| Error::InvalidState)?;
    let environment = frame
        .environment
        .map(|owner| {
            graph.import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
        })
        .transpose()
        .map_err(|_| Error::InvalidState)?;
    let specular_upload = frame
        .specular
        .map(|owner| graph.import_buffer_desc(owner.upload_desc.clone(), owner.upload_incoming))
        .transpose()
        .map_err(|_| Error::InvalidState)?;
    let specular_view_uniform = frame
        .specular
        .map(|owner| {
            graph.import_buffer_desc(owner.view_uniform_desc.clone(), owner.view_uniform_incoming)
        })
        .transpose()
        .map_err(|_| Error::InvalidState)?;
    let specular_radiance = frame
        .specular
        .map(|owner| {
            graph.import_texture_with_state(
                owner.radiance.desc.clone(),
                owner.radiance.incoming,
                owner.radiance.layout,
            )
        })
        .transpose()
        .map_err(|_| Error::InvalidState)?;
    let specular_brdf_lut = frame
        .specular
        .map(|owner| {
            graph.import_texture_with_state(
                owner.brdf_lut.desc.clone(),
                owner.brdf_lut.incoming,
                owner.brdf_lut.layout,
            )
        })
        .transpose()
        .map_err(|_| Error::InvalidState)?;
    let gbuffer = frame
        .gbuffer
        .iter()
        .map(|texture| {
            graph.import_texture_with_state(texture.desc.clone(), texture.incoming, texture.layout)
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| Error::InvalidState)?;
    let effects = frame
        .effects
        .map(|texture| {
            graph.import_texture_with_state(texture.desc.clone(), texture.incoming, texture.layout)
        })
        .transpose()
        .map_err(|_| Error::InvalidState)?;
    let msaa_gbuffer = frame
        .msaa
        .map(|msaa| {
            let mut import = |owner: &crate::residency::ForwardTexture<B>| {
                graph.import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
            };
            let [albedo, normal, depth] = &msaa.gbuffer;
            Ok([import(albedo)?, import(normal)?, import(depth)?])
        })
        .transpose()
        .map_err(|_: GraphError| Error::InvalidState)?;
    let msaa_effects = frame
        .msaa
        .and_then(|msaa| msaa.effects.as_ref())
        .map(|texture| {
            graph.import_texture_with_state(texture.desc.clone(), texture.incoming, texture.layout)
        })
        .transpose()
        .map_err(|_| Error::InvalidState)?;
    let (target, scene_color, surface_target, readback, row_pitch) = match &frame.source {
        super::record_view::RecordSource::Offscreen(owner) => (
            Some(
                graph
                    .import_texture_with_state(
                        owner.frame.target_desc.clone(),
                        owner.frame.target_incoming,
                        owner.frame.target_layout,
                    )
                    .map_err(|_| Error::InvalidState)?,
            ),
            Some(
                graph
                    .import_texture_with_state(
                        owner.scene_color.desc.clone(),
                        owner.scene_color.incoming,
                        owner.scene_color.layout,
                    )
                    .map_err(|_| Error::InvalidState)?,
            ),
            None,
            Some(
                graph
                    .import_buffer_desc(
                        owner.frame.readback_desc.clone(),
                        owner.frame.readback_incoming,
                    )
                    .map_err(|_| Error::InvalidState)?,
            ),
            Some(owner.readback_row_pitch),
        ),
        super::record_view::RecordSource::Surface { frame: owner, .. } => {
            let desc = fluxel_rhi::api::TextureDesc {
                label: Some("fluxel deferred surface lease".into()),
                dimension: fluxel_rhi::api::TextureDimension::D2,
                extent: [owner.extent[0], owner.extent[1], 1],
                array_layers: 1,
                format: owner.color_format,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::RENDER_ATTACHMENT,
                view_formats: Vec::new(),
            };
            (
                None,
                Some(
                    graph
                        .import_texture_with_state(
                            owner.scene_color.desc.clone(),
                            owner.scene_color.incoming,
                            owner.scene_color.layout,
                        )
                        .map_err(|_| Error::InvalidState)?,
                ),
                Some(
                    graph
                        .import_surface_texture(desc)
                        .map_err(|_| Error::InvalidState)?,
                ),
                None,
                None,
            )
        }
    };
    let deformation = frame
        .deformation
        .as_ref()
        .map(|deformation| {
            Ok::<_, Error>((
                graph
                    .import_buffer_desc(
                        deformation.upload.desc.clone(),
                        deformation.upload.incoming,
                    )
                    .map_err(|_| Error::InvalidState)?,
                graph
                    .import_buffer_desc(
                        deformation.source.desc.clone(),
                        deformation.source.incoming,
                    )
                    .map_err(|_| Error::InvalidState)?,
                graph
                    .import_buffer_desc(
                        deformation.product.desc.clone(),
                        deformation.product.incoming,
                    )
                    .map_err(|_| Error::InvalidState)?,
                graph
                    .import_buffer_desc(
                        deformation.parameters.desc.clone(),
                        deformation.parameters.incoming,
                    )
                    .map_err(|_| Error::InvalidState)?,
            ))
        })
        .transpose()?;
    Ok(DeferredImports {
        resident_upload,
        frame_upload,
        uniform,
        streams,
        materials,
        environment_upload,
        environment,
        specular_upload,
        specular_view_uniform,
        specular_radiance,
        specular_brdf_lut,
        gbuffer,
        effects,
        msaa_gbuffer,
        msaa_effects,
        target,
        scene_color,
        surface_target,
        readback,
        row_pitch,
        deformation,
    })
}
