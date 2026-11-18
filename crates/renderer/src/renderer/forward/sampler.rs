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

/// Builds the exact sampler for one material draw. Immutable image ownership
/// stays separate: an explicit sampling policy is a draw binding, while the
/// default derives only the LOD range needed by that draw's image chain.
pub(crate) fn material_sampler_for_input(
    input: &crate::ForwardDrawInput,
) -> std::result::Result<SamplerDesc, Error> {
    material_sampler_for_input_with_capture(input, None)
}

/// Builds the exact sampler for one material draw, accounting for the actual
/// capture chain selected by an opt-in projected-refraction receiver.
pub(crate) fn material_sampler_for_input_with_capture(
    input: &crate::ForwardDrawInput,
    capture_mip_levels: Option<u32>,
) -> std::result::Result<SamplerDesc, Error> {
    if let Some(desc) = input.sampler_descriptor() {
        validate_material_sampler(desc)?;
        return Ok(desc.clone());
    }
    let mip_levels = (0..input.resolved_textures().len())
        .filter_map(|index| {
            input
                .asset_texture(index)
                .map(|texture| texture.value().mip_levels)
        })
        .chain(
            input
                .prepared()
                .uses_scene_refraction()
                .then_some(capture_mip_levels)
                .flatten(),
        )
        .max()
        .unwrap_or(1);
    material_sampler(mip_levels)
}

/// Deduplicates the concrete sampler policies needed by one resident frame.
/// Texture allocation and caching intentionally do not participate in this
/// calculation: image identity is independent of how a draw samples it.
pub(crate) fn material_sampler_descriptors(
    inputs: &[crate::ForwardDrawInput],
    capture_mip_levels: Option<u32>,
) -> std::result::Result<Vec<SamplerDesc>, Error> {
    let mut descriptors = Vec::new();
    for input in inputs
        .iter()
        .filter(|input| material_requires_ordinary_sampler(input.prepared()))
    {
        let descriptor = material_sampler_for_input_with_capture(input, capture_mip_levels)?;
        if !descriptors
            .iter()
            .any(|existing| same_material_sampler(existing, &descriptor))
        {
            descriptors.push(descriptor);
        }
    }
    Ok(descriptors)
}

/// Compares all semantic native sampler fields while deliberately ignoring a
/// debug label. Reusing a native sampler is valid only for an exact policy.
pub(crate) fn same_material_sampler(left: &SamplerDesc, right: &SamplerDesc) -> bool {
    left.address == right.address
        && left.mag_filter == right.mag_filter
        && left.min_filter == right.min_filter
        && left.mip_filter == right.mip_filter
        && left.lod.start.to_bits() == right.lod.start.to_bits()
        && left.lod.end.to_bits() == right.lod.end.to_bits()
        && left.anisotropy == right.anisotropy
        && left.compare == right.compare
        && left.border == right.border
}

/// Validates the bounded ordinary float-sampler contract consumed by material
/// bind groups. It intentionally permits nearest filtering and non-clamp
/// addressing; scene-composite users retain `validate_ordinary_sampler`.
pub(crate) fn validate_material_sampler(desc: &SamplerDesc) -> std::result::Result<(), Error> {
    let ordinary_filter = |filter| matches!(filter, FilterMode::Nearest | FilterMode::Linear);
    if desc.compare.is_some()
        || desc.border.is_some()
        || desc.address.contains(&AddressMode::ClampToBorder)
        || !ordinary_filter(desc.mag_filter)
        || !ordinary_filter(desc.min_filter)
        || !ordinary_filter(desc.mip_filter)
        || desc.anisotropy == 0
        || !desc.lod.start.is_finite()
        || !desc.lod.end.is_finite()
        || desc.lod.start > desc.lod.end
    {
        return Err(Error::InvalidState);
    }
    if desc.anisotropy > 1
        && (desc.mag_filter != FilterMode::Linear
            || desc.min_filter != FilterMode::Linear
            || desc.mip_filter != FilterMode::Linear)
    {
        return Err(Error::InvalidState);
    }
    Ok(())
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
