//! Forward material sampler admission and owner checks.
use super::*;
pub(crate) fn material_requires_ordinary_sampler(prepared: &PreparedMaterial) -> bool {
    prepared.material().contribution().bindings.iter().any(|b| {
        b.location.group == 0
            && b.location.binding == 2
            && b.kind == fluxel_material::MaterialResourceKind::Sampler { comparison: false }
    })
}
pub(crate) fn default_material_sampler() -> SamplerDesc {
    SamplerDesc {
        label: Some("Fluxel material sampler".into()),
        address: [AddressMode::ClampToEdge; 3],
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Nearest,
        lod: 0.0..0.0,
        anisotropy: 1,
        compare: None,
        border: None,
    }
}
pub(crate) fn validate_ordinary_sampler(desc: &SamplerDesc) -> std::result::Result<(), Error> {
    if desc.compare.is_some()
        || desc.mag_filter != FilterMode::Linear
        || desc.min_filter != FilterMode::Linear
        || !desc.lod.start.is_finite()
        || !desc.lod.end.is_finite()
        || desc.lod.start > desc.lod.end
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}
pub(crate) fn require_filterable_material_texture<B: ResourceBackend>(
    device: &Device<B>,
    extent: [u32; 2],
) -> std::result::Result<(), Error> {
    let desc = TextureDesc {
        label: Some("Fluxel Forward sampled material texture".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    };
    match device.texture_support(&desc) {
        Support::Supported(s) if s.filterable => Ok(()),
        Support::Supported(_) => Err(Error::Unsupported("filterable material texture")),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}
pub(crate) fn validate_sampled_texture_owner<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    texture: usize,
    selected_device: DeviceId,
) -> std::result::Result<(), Error> {
    let owner = resident.textures.get(texture).ok_or(Error::InvalidRange)?;
    if resident.device != selected_device
        || owner.device != selected_device
        || owner.desc.dimension != TextureDimension::D2
        || owner.desc.format != TextureFormat::Rgba8Unorm
        || owner.desc.extent.contains(&0)
        || owner.desc.array_layers != 1
        || owner.desc.mip_levels != 1
        || owner.desc.samples != 1
        || !owner.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        || !owner.filterable
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}
