//! Forward material sampler admission and owner checks.
use super::*;
use crate::{
    preparation::{ColorTextureArrayData, ResolvedForwardTexture},
    residency::LayeredTextureView,
};
pub(crate) fn material_requires_ordinary_sampler(prepared: &PreparedMaterial) -> bool {
    prepared
        .binding_schema()
        .is_some_and(|schema| schema.samplers.iter().any(|sampler| !sampler.comparison))
}

pub(crate) fn validate_material_binding_limit(
    prepared: &PreparedMaterial,
    limit: u32,
) -> std::result::Result<(), Error> {
    let schema = prepared.binding_schema().ok_or(Error::InvalidState)?;
    let locations = schema
        .uniforms
        .iter()
        .map(|binding| binding.binding)
        .chain(schema.textures.iter().map(|binding| binding.binding))
        .chain(schema.samplers.iter().map(|binding| binding.binding));
    if locations.into_iter().any(|location| location >= limit) {
        return Err(Error::Unsupported("material binding locations"));
    }
    Ok(())
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

/// Resolves an explicit policy by binding. For omitted policies the default
/// LOD range still covers the draw's image chains, including scene capture.
pub(crate) fn material_sampler_for_input(
    input: &crate::ForwardDrawInput,
    binding: u32,
    comparison: bool,
    capture_mip_levels: Option<u32>,
) -> std::result::Result<SamplerDesc, Error> {
    if let Some(desc) = input.sampler_descriptor(binding) {
        validate_material_sampler(desc, comparison)?;
        return Ok(desc.clone());
    }
    if comparison {
        return Err(Error::InvalidState);
    }
    let mip_levels = input
        .resolved_textures()
        .iter()
        .filter_map(|texture| {
            material_texture_properties(texture).map(|properties| properties.mip_levels)
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
    for input in inputs {
        let Some(schema) = input.prepared().binding_schema() else {
            continue;
        };
        for sampler in &schema.samplers {
            let desc = material_sampler_for_input(
                input,
                sampler.binding,
                sampler.comparison,
                capture_mip_levels,
            )?;
            if !descriptors
                .iter()
                .any(|existing| same_material_sampler(existing, &desc))
            {
                descriptors.push(desc);
            }
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

/// Validates the concrete descriptor against the reflected sampler mode.
pub(crate) fn validate_material_sampler(
    desc: &SamplerDesc,
    comparison: bool,
) -> std::result::Result<(), Error> {
    let supported_filter = |filter| matches!(filter, FilterMode::Nearest | FilterMode::Linear);
    if desc.compare.is_some() != comparison
        || desc.border.is_some()
        || desc.address.contains(&AddressMode::ClampToBorder)
        || !supported_filter(desc.mag_filter)
        || !supported_filter(desc.min_filter)
        || !supported_filter(desc.mip_filter)
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
// Image facts come from the immutable CPU payload admitted during preparation.
struct MaterialTextureProperties {
    extent: Option<[u32; 3]>,
    format: TextureFormat,
    layers: u32,
    mip_levels: u32,
    generate_mips: bool,
    view_dimension: TextureViewDimension,
    binding_usage: TextureUsage,
}

impl MaterialTextureProperties {
    fn layered(
        data: &ColorTextureArrayData,
        generated_mips: Option<u32>,
        view: LayeredTextureView,
    ) -> Self {
        Self {
            extent: Some([data.extent[0], data.extent[1], 1]),
            format: data.encoding.rgba8_format(),
            layers: data.layers,
            mip_levels: generated_mips.unwrap_or(data.mip_levels),
            generate_mips: generated_mips.is_some(),
            view_dimension: view.dimension(),
            binding_usage: TextureUsage::TEXTURE_BINDING,
        }
    }

    fn dimension(&self) -> TextureDimension {
        if self.view_dimension == TextureViewDimension::D3 {
            TextureDimension::D3
        } else {
            TextureDimension::D2
        }
    }

    fn uses_float_filtering(&self) -> bool {
        self.binding_usage == TextureUsage::TEXTURE_BINDING
            && self.format != TextureFormat::Depth32Float
    }

    fn d2(format: TextureFormat) -> Self {
        Self {
            extent: None,
            format,
            layers: 1,
            mip_levels: 1,
            generate_mips: false,
            view_dimension: TextureViewDimension::D2,
            binding_usage: TextureUsage::TEXTURE_BINDING,
        }
    }
}

fn material_texture_properties(
    texture: &ResolvedForwardTexture,
) -> Option<MaterialTextureProperties> {
    Some(match texture {
        ResolvedForwardTexture::Asset(texture) => {
            let data = texture.value();
            MaterialTextureProperties {
                extent: Some([data.extent[0], data.extent[1], 1]),
                mip_levels: data.mip_levels,
                generate_mips: data.mip_levels > 1,
                ..MaterialTextureProperties::d2(data.encoding.rgba8_format())
            }
        }
        ResolvedForwardTexture::StorageAsset(texture) => {
            let [width, height] = texture.value().extent;
            MaterialTextureProperties {
                extent: Some([width, height, 1]),
                binding_usage: TextureUsage::STORAGE_BINDING,
                ..MaterialTextureProperties::d2(TextureFormat::Rgba8Unorm)
            }
        }
        ResolvedForwardTexture::ArrayAsset(texture) => {
            MaterialTextureProperties::layered(texture.value(), None, LayeredTextureView::Array)
        }
        ResolvedForwardTexture::GeneratedArray { asset, mip_levels } => {
            MaterialTextureProperties::layered(
                asset.value(),
                Some(*mip_levels),
                LayeredTextureView::Array,
            )
        }
        ResolvedForwardTexture::CubeAsset {
            asset,
            generated_mips,
        } => MaterialTextureProperties::layered(
            asset.value(),
            *generated_mips,
            LayeredTextureView::Cube,
        ),
        ResolvedForwardTexture::VolumeAsset {
            asset,
            generated_mips,
        } => {
            let data = asset.value();
            MaterialTextureProperties {
                extent: Some(data.extent),
                mip_levels: generated_mips.unwrap_or(data.mip_levels),
                generate_mips: generated_mips.is_some(),
                view_dimension: TextureViewDimension::D3,
                ..MaterialTextureProperties::d2(data.encoding.rgba8_format())
            }
        }
        ResolvedForwardTexture::DepthAsset(texture) => {
            let [width, height] = texture.value().extent;
            MaterialTextureProperties {
                extent: Some([width, height, 1]),
                ..MaterialTextureProperties::d2(TextureFormat::Depth32Float)
            }
        }
        ResolvedForwardTexture::RefractionRadiance(texture) => {
            let mips = texture.value().mips();
            MaterialTextureProperties {
                extent: Some([mips[0].extent[0], mips[0].extent[1], 1]),
                mip_levels: mips.len() as u32,
                ..MaterialTextureProperties::d2(TextureFormat::Rgba16Float)
            }
        }
        ResolvedForwardTexture::CaptureSlot(_) => return None,
    })
}

/// Admits the descriptor required to bind an immutable material image.
pub(crate) fn require_filterable_material_texture<B: ResourceBackend>(
    device: &Device<B>,
    texture: &ResolvedForwardTexture,
) -> std::result::Result<(), Error> {
    let properties = material_texture_properties(texture).ok_or(Error::InvalidState)?;
    let extent = properties.extent.ok_or(Error::InvalidState)?;
    let mip_levels = properties.mip_levels;
    if extent.contains(&0) || mip_levels == 0 || properties.layers == 0 {
        return Err(Error::InvalidRange);
    }
    let max_mip_levels = 1 + extent[0].max(extent[1]).max(extent[2]).ilog2();
    if mip_levels > max_mip_levels {
        return Err(Error::InvalidRange);
    }
    let mut usage = TextureUsage::COPY_DST | properties.binding_usage;
    if properties.generate_mips {
        usage |= TextureUsage::COPY_SRC;
    }
    let desc = TextureDesc {
        label: Some("Fluxel Forward sampled material texture".into()),
        dimension: properties.dimension(),
        extent,
        array_layers: properties.layers,
        format: properties.format,
        mip_levels,
        samples: 1,
        usage,
        view_formats: Vec::new(),
    };
    match device.texture_support(&desc) {
        Support::Supported(s) if mip_levels > s.max_mips => {
            Err(Error::Unsupported("material texture mip levels"))
        }
        Support::Supported(s) if !properties.uses_float_filtering() || s.filterable => Ok(()),
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
    validate_texture_owner_with_properties(
        resident,
        texture,
        selected_device,
        MaterialTextureProperties::d2(expected_encoding.rgba8_format()),
        require_filterable,
    )
}

/// Revalidates an image owner against the exact immutable descriptor facts.
fn validate_texture_owner_with_properties<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    texture: usize,
    selected_device: DeviceId,
    expected: MaterialTextureProperties,
    require_filterable: bool,
) -> std::result::Result<(), Error> {
    let owner = resident.textures.get(texture).ok_or(Error::InvalidRange)?;
    if resident.device != selected_device
        || owner.device != selected_device
        || owner.texture.device() != selected_device
        || owner.view.device() != selected_device
        || owner.desc.dimension != expected.dimension()
        || owner.desc.format != expected.format
        || owner.desc.extent.contains(&0)
        || (owner.desc.dimension == TextureDimension::D2 && owner.desc.extent[2] != 1)
        || expected
            .extent
            .is_some_and(|extent| owner.desc.extent != extent)
        || owner.desc.array_layers != expected.layers
        || owner.desc.mip_levels != expected.mip_levels
        || owner.view_dimension != expected.view_dimension
        || owner.desc.samples != 1
        || !owner.desc.usage.contains(expected.binding_usage)
        || (expected.binding_usage == TextureUsage::STORAGE_BINDING
            && owner.desc.usage != (TextureUsage::COPY_DST | TextureUsage::STORAGE_BINDING))
        || (expected.generate_mips && !owner.desc.usage.contains(TextureUsage::COPY_SRC))
        || (require_filterable && expected.uses_float_filtering() && !owner.filterable)
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
    expected_dimension: TextureViewDimension,
) -> std::result::Result<(), Error> {
    let source = input
        .resolved_textures()
        .get(texture_index)
        .ok_or(Error::InvalidRange)?;
    let expected = material_texture_properties(source)
        .unwrap_or_else(|| MaterialTextureProperties::d2(TextureFormat::Rgba8Unorm));
    if expected.view_dimension != expected_dimension {
        return Err(Error::InvalidState);
    }
    validate_texture_owner_with_properties(
        resident,
        texture,
        selected_device,
        expected,
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
