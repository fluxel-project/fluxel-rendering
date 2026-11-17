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
    encoding: crate::TextureEncoding,
) -> std::result::Result<(), Error> {
    let desc = TextureDesc {
        label: Some("Fluxel Forward sampled material texture".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: encoding.rgba8_format(),
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
    expected_encoding: crate::TextureEncoding,
    require_filterable: bool,
) -> std::result::Result<(), Error> {
    let owner = resident.textures.get(texture).ok_or(Error::InvalidRange)?;
    if resident.device != selected_device
        || owner.device != selected_device
        || owner.texture.device() != selected_device
        || owner.view.device() != selected_device
        || owner.desc.dimension != TextureDimension::D2
        || owner.desc.format != expected_encoding.rgba8_format()
        || owner.desc.extent[0] == 0
        || owner.desc.extent[1] == 0
        || owner.desc.extent[2] != 1
        || owner.desc.array_layers != 1
        || owner.desc.mip_levels != 1
        || owner.desc.samples != 1
        || !owner.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        || (require_filterable && !owner.filterable)
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

/// Revalidates the concrete resident owner against the source selected for a
/// draw. Capture products are renderer-produced linear textures; CPU assets
/// retain their explicit transfer encoding.
pub(crate) fn validate_material_texture_owner<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    texture: usize,
    selected_device: DeviceId,
    input: &crate::ForwardDrawInput,
    requires_filtering: bool,
) -> std::result::Result<(), Error> {
    if let Some(source) = input.asset_texture() {
        let extent = source.value().extent;
        let owner = resident.textures.get(texture).ok_or(Error::InvalidRange)?;
        if owner.desc.extent != [extent[0], extent[1], 1] {
            return Err(Error::InvalidState);
        }
    }
    let expected_encoding = input
        .asset_texture()
        .map(|texture| texture.value().encoding)
        .unwrap_or(crate::TextureEncoding::Linear);
    validate_sampled_texture_owner(
        resident,
        texture,
        selected_device,
        expected_encoding,
        requires_filtering,
    )
}

/// Admits the renderer-owned capture texture selected by a receiver before a
/// sampler bind group can consume it.
pub(crate) fn validate_capture_material_texture<B: ResourceBackend>(
    capture: Option<&super::captured::CaptureTextureOwner<B>>,
    input: &crate::ForwardDrawInput,
    device: DeviceId,
) -> std::result::Result<(), crate::RendererError> {
    let capture = capture.ok_or_else(|| {
        crate::RendererError::ForwardPreparation(
            crate::ForwardPreparationError::CaptureReceiverRequiresSlot {
                source: input.source(),
            },
        )
    })?;
    let slot = input.capture_texture_slot().ok_or(Error::InvalidState)?;
    capture.validate_for_receiver(device, slot)?;
    Ok(())
}
