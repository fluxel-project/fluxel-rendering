//! Definedness and descriptor range checks for graph authoring.
use super::*;

pub(super) fn range_is_defined(defined: &[Range<u64>], wanted: &ResourceRange) -> bool {
    defined
        .iter()
        .any(|range| range.start <= wanted.0.start && wanted.0.end <= range.end)
}

pub(super) fn empty_buffer_uses() -> BufferUses {
    BufferUses {
        stages: fluxel_rhi::api::StageMask::EMPTY,
        accesses: fluxel_rhi::api::AccessMask::EMPTY,
    }
}

pub(super) fn empty_texture_uses() -> TextureUses {
    TextureUses {
        stages: fluxel_rhi::api::StageMask::EMPTY,
        accesses: fluxel_rhi::api::AccessMask::EMPTY,
    }
}

pub(super) fn normalize_ranges(ranges: &mut Vec<Range<u64>>) {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged = Vec::<Range<u64>>::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if let Some(last) = merged.last_mut().filter(|last| range.start <= last.end) {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    *ranges = merged;
}

#[allow(dead_code)]
pub(super) fn validate_texture_desc(desc: &TextureDesc) -> Result<(), GraphError> {
    if desc.extent.contains(&0)
        || desc.array_layers == 0
        || desc.mip_levels == 0
        || desc.samples == 0
    {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

#[allow(dead_code)]
pub(super) fn full_texture_range(desc: &TextureDesc) -> TextureRange {
    TextureRange::new(
        format_aspects(desc.format),
        0..desc.mip_levels,
        0..desc.array_layers,
    )
}

#[allow(dead_code)]
pub(super) fn format_aspects(format: TextureFormat) -> TextureAspects {
    match format {
        TextureFormat::Depth16Unorm | TextureFormat::Depth24Plus | TextureFormat::Depth32Float => {
            TextureAspects::DEPTH
        }
        TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8 => {
            TextureAspects::DEPTH | TextureAspects::STENCIL
        }
        TextureFormat::Stencil8 => TextureAspects::STENCIL,
        _ => TextureAspects::COLOR,
    }
}

#[allow(dead_code)]
pub(super) fn validate_texture_range(
    desc: &TextureDesc,
    range: &TextureRange,
) -> Result<(), GraphError> {
    let valid_aspects = format_aspects(desc.format);
    if range.aspects.0 == 0
        || range.aspects.0 & !valid_aspects.0 != 0
        || range.mips.start >= range.mips.end
        || range.mips.end > desc.mip_levels
        || range.layers.start >= range.layers.end
        || range.layers.end > desc.array_layers
    {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

#[allow(dead_code)]
pub(super) fn texture_range_defined(defined: &[TextureRange], wanted: &TextureRange) -> bool {
    // Texture versions retain individual written subresources. A requested
    // range can consequently be fully defined by several adjacent writes; it
    // need not be enclosed by one original declaration.
    selected_texture_aspects(wanted.aspects).all(|aspect| {
        wanted.mips.clone().all(|mip| {
            wanted.layers.clone().all(|layer| {
                defined.iter().any(|range| {
                    range.aspects.0 & aspect != 0
                        && range.mips.start <= mip
                        && mip < range.mips.end
                        && range.layers.start <= layer
                        && layer < range.layers.end
                })
            })
        })
    })
}

pub(super) fn selected_texture_aspects(aspects: TextureAspects) -> impl Iterator<Item = u8> {
    [
        TextureAspects::COLOR.0,
        TextureAspects::DEPTH.0,
        TextureAspects::STENCIL.0,
    ]
    .into_iter()
    .filter(move |aspect| aspects.0 & aspect != 0)
}

#[allow(dead_code)]
pub(super) fn merge_texture_range(defined: &mut Vec<TextureRange>, range: TextureRange) {
    // A new logical version preserves all prior defined pieces. Coalescing is
    // intentionally conservative: only identical aspect/layer spans with
    // adjacent mip spans merge. It cannot invent defined subresources.
    if let Some(existing) = defined.iter_mut().find(|existing| {
        existing.aspects == range.aspects
            && existing.layers == range.layers
            && range.mips.start <= existing.mips.end
            && existing.mips.start <= range.mips.end
    }) {
        existing.mips.start = existing.mips.start.min(range.mips.start);
        existing.mips.end = existing.mips.end.max(range.mips.end);
    } else {
        defined.push(range);
    }
}
