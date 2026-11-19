//! Bind-group layout entry builders for Deferred programs.

use fluxel_rhi::api::{
    BindGroupLayoutEntry, BindingCount, BindingType, ShaderStage, TextureSampleType,
    TextureViewDimension,
};

pub(super) fn material_uniform(
    binding: u32,
    min_size: u64,
    visibility: Vec<ShaderStage>,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::UniformBuffer {
            dynamic_offset: true,
            min_size,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}

pub(super) fn sampled_texture(binding: u32, sample: TextureSampleType) -> BindGroupLayoutEntry {
    sampled_texture_with_dimension(binding, sample, TextureViewDimension::D2)
}

pub(super) fn sampled_texture_with_dimension(
    binding: u32,
    sample: TextureSampleType,
    dimension: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::SampledTexture {
            sample,
            dimension,
            multisampled: false,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}

pub(super) fn ordinary_sampler(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::Sampler {
            filtering: true,
            comparison: false,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
