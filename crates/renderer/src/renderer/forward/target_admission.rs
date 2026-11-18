//! Color-attachment support admission before Forward owner allocation.

use fluxel_rhi::{
    api::{Device, Error, TextureDesc, TextureDimension, TextureFormat, TextureUsage},
    backend::ResourceBackend,
};

/// Admits the selected material modes and the exact target selected for their
/// Forward phase before any depth, buffer, or target owner is allocated.
pub(super) fn admit_assembly<B: ResourceBackend>(
    device: &Device<B>,
    inputs: &[crate::ForwardDrawInput],
    topology: crate::FrameTopology,
    extent: [u32; 2],
    format: TextureFormat,
    imported_surface: bool,
    sampled_target: bool,
) -> Result<(), Error> {
    crate::renderer::require_supported_surface_modes(inputs, topology)?;
    admit(
        device,
        topology,
        extent,
        format,
        imported_surface,
        sampled_target,
        inputs
            .iter()
            .any(|input| input.prepared().uses_scene_refraction()),
    )
}

/// Checks the exact target descriptor selected for this frame's topology.
pub(in crate::renderer) fn admit<B: ResourceBackend>(
    device: &Device<B>,
    topology: crate::FrameTopology,
    extent: [u32; 2],
    format: TextureFormat,
    imported_surface: bool,
    sampled_target: bool,
    copy_destination: bool,
) -> Result<(), Error> {
    let usage = if imported_surface {
        TextureUsage::RENDER_ATTACHMENT
    } else {
        let mut usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC;
        if sampled_target {
            usage |= TextureUsage::TEXTURE_BINDING;
        }
        if copy_destination {
            usage |= TextureUsage::COPY_DST;
        }
        usage
    };
    crate::renderer::require_blendable_target(
        device,
        &TextureDesc {
            label: Some("fluxel forward selected color target".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: Vec::new(),
        },
        topology,
    )
}
