use super::*;

use fluxel_rhi::api::{TextureDesc, TextureDimension, TextureFormat, TextureUsage};

fn descriptor(format: TextureFormat) -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [320, 180, 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

#[test]
fn gbuffer_descriptor_compatibility_requires_the_actual_deferred_shapes() {
    assert!(descriptor_compatible(
        &descriptor(TextureFormat::Rgba8Unorm),
        [320, 180],
        TextureFormat::Rgba8Unorm,
    ));
    assert!(descriptor_compatible(
        &descriptor(TextureFormat::Depth32Float),
        [320, 180],
        TextureFormat::Depth32Float,
    ));
}

#[test]
fn gbuffer_descriptor_compatibility_refuses_extent_format_and_usage_changes() {
    let color = descriptor(TextureFormat::Rgba8Unorm);
    assert!(!descriptor_compatible(
        &color,
        [256, 144],
        TextureFormat::Rgba8Unorm,
    ));
    assert!(!descriptor_compatible(
        &color,
        [320, 180],
        TextureFormat::Depth32Float,
    ));

    let mut missing_sampled_use = descriptor(TextureFormat::Depth32Float);
    missing_sampled_use.usage = TextureUsage::RENDER_ATTACHMENT;
    assert!(!descriptor_compatible(
        &missing_sampled_use,
        [320, 180],
        TextureFormat::Depth32Float,
    ));
}

#[test]
fn recycled_gbuffer_state_is_fragment_sampled_and_shader_read_only() {
    let uses = sampled_fragment_uses();
    assert_eq!(uses.stages, StageMask::FRAGMENT);
    assert_eq!(uses.accesses, AccessMask::SHADER_READ);
    assert_eq!(pooled_layout(), ImageLayout::ShaderReadOnly);
}
