//! Forward material sampler admission and owner checks.
use super::*;
pub(crate) fn material_requires_ordinary_sampler(prepared: &PreparedMaterial) -> bool {
    prepared
        .binding_schema()
        .is_some_and(|schema| schema.sampler.is_some())
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

/// Builds the sampler descriptor for an immutable material texture with an
/// exact, already-validated mip count. The single-level descriptor remains
/// byte-for-byte the legacy material sampler.
pub(crate) fn material_sampler(mip_levels: u32) -> std::result::Result<SamplerDesc, Error> {
    if mip_levels == 0 {
        return Err(Error::InvalidRange);
    }
    if mip_levels == 1 {
        return Ok(default_material_sampler());
    }
    let mut desc = default_material_sampler();
    desc.mip_filter = FilterMode::Linear;
    desc.lod = 0.0..mip_levels as f32;
    Ok(desc)
}

/// Chooses one ordinary material sampler for all immutable asset textures in a
/// prepared frame. Capture slots are renderer-owned single-level products;
/// the maximum asset count keeps LOD enabled when any bound asset has mips.
pub(crate) fn material_sampler_for_inputs(
    inputs: &[crate::ForwardDrawInput],
) -> std::result::Result<SamplerDesc, Error> {
    let mip_levels = inputs
        .iter()
        .flat_map(|input| {
            (0..input.resolved_textures().len()).filter_map(|index| {
                input
                    .asset_texture(index)
                    .map(|texture| texture.value().mip_levels)
            })
        })
        .max()
        .unwrap_or(1);
    material_sampler(mip_levels)
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
/// Admits the exact texture descriptor that an immutable material asset will
/// allocate, including its requested mip chain.
pub(crate) fn require_filterable_material_texture_with_mip_levels<B: ResourceBackend>(
    device: &Device<B>,
    extent: [u32; 2],
    encoding: crate::TextureEncoding,
    mip_levels: u32,
) -> std::result::Result<(), Error> {
    if extent.contains(&0) || mip_levels == 0 {
        return Err(Error::InvalidRange);
    }
    let max_mip_levels = 1 + extent[0].max(extent[1]).ilog2();
    if mip_levels > max_mip_levels {
        return Err(Error::InvalidRange);
    }
    if encoding == crate::TextureEncoding::Srgb && mip_levels > 1 {
        return Err(Error::Unsupported("linear RGBA8 material mip generation"));
    }
    let mut usage = TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING;
    if mip_levels > 1 {
        usage |= TextureUsage::COPY_SRC;
    }
    let desc = TextureDesc {
        label: Some("Fluxel Forward sampled material texture".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: encoding.rgba8_format(),
        mip_levels,
        samples: 1,
        usage,
        view_formats: Vec::new(),
    };
    match device.texture_support(&desc) {
        Support::Supported(s) if mip_levels > s.max_mips => {
            Err(Error::Unsupported("material texture mip levels"))
        }
        Support::Supported(s) if s.filterable => Ok(()),
        Support::Supported(_) => Err(Error::Unsupported("filterable material texture")),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}
#[cfg(test)]
pub(crate) fn validate_sampled_texture_owner<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    texture: usize,
    selected_device: DeviceId,
    expected_encoding: crate::TextureEncoding,
    require_filterable: bool,
) -> std::result::Result<(), Error> {
    validate_sampled_texture_owner_with_mips(
        resident,
        texture,
        selected_device,
        expected_encoding,
        1,
        require_filterable,
    )
}

/// Revalidates a sampled owner against the exact immutable descriptor facts.
pub(crate) fn validate_sampled_texture_owner_with_mips<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    texture: usize,
    selected_device: DeviceId,
    expected_encoding: crate::TextureEncoding,
    expected_mip_levels: u32,
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
        || owner.desc.mip_levels != expected_mip_levels
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
    texture_index: usize,
    requires_filtering: bool,
) -> std::result::Result<(), Error> {
    if let Some(source) = input.asset_texture(texture_index) {
        let extent = source.value().extent;
        let owner = resident.textures.get(texture).ok_or(Error::InvalidRange)?;
        if owner.desc.extent != [extent[0], extent[1], 1] {
            return Err(Error::InvalidState);
        }
    }
    let expected_encoding = input
        .asset_texture(texture_index)
        .map(|texture| texture.value().encoding)
        .unwrap_or(crate::TextureEncoding::Linear);
    let expected_mip_levels = input
        .asset_texture(texture_index)
        .map(|texture| texture.value().mip_levels)
        .unwrap_or(1);
    validate_sampled_texture_owner_with_mips(
        resident,
        texture,
        selected_device,
        expected_encoding,
        expected_mip_levels,
        requires_filtering,
    )
}

/// Admits the renderer-owned capture texture selected by a receiver before a
/// sampler bind group can consume it.
pub(crate) fn validate_capture_material_texture<B: ResourceBackend>(
    capture: Option<&super::captured::CaptureTextureOwner<B>>,
    input: &crate::ForwardDrawInput,
    texture_index: usize,
    device: DeviceId,
) -> std::result::Result<(), crate::RendererError> {
    let capture = capture.ok_or_else(|| {
        crate::RendererError::ForwardPreparation(
            crate::ForwardPreparationError::CaptureReceiverRequiresSlot {
                source: input.source(),
            },
        )
    })?;
    let slot = input
        .capture_texture_slot(texture_index)
        .ok_or(Error::InvalidState)?;
    capture.validate_for_receiver(device, slot, true)?;
    Ok(())
}
