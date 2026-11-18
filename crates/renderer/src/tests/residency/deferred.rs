use super::*;

use fluxel_rhi::api::{TextureDimension, TextureFormat, TextureUsage};

#[test]
fn gbuffer_descriptors_require_two_sampled_attachments_and_depth() {
    let descs = gbuffer_descs([320, 180], true, 1);

    assert_eq!(descs.len(), 3);
    for desc in &descs {
        assert_eq!(desc.dimension, TextureDimension::D2);
        assert_eq!(desc.extent, [320, 180, 1]);
        assert_eq!(desc.array_layers, 1);
        assert_eq!(desc.mip_levels, 1);
        assert_eq!(desc.samples, 1);
        assert_eq!(
            desc.usage,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING
        );
        assert!(desc.view_formats.is_empty());
    }
    assert_eq!(descs[0].format, TextureFormat::Rgba16Float);
    assert_eq!(descs[1].format, TextureFormat::Rgba8Unorm);
    assert_eq!(descs[2].format, TextureFormat::Depth32Float);
}

#[test]
fn request_rejects_zero_extent_and_insufficient_color_attachments() {
    assert!(matches!(
        validate_request([0, 1], 2, false),
        Err(fluxel_rhi::api::Error::InvalidRange)
    ));
    assert!(matches!(
        validate_request([1, 0], 2, false),
        Err(fluxel_rhi::api::Error::InvalidRange)
    ));
    assert!(matches!(
        validate_request([1, 1], 1, false),
        Err(fluxel_rhi::api::Error::Unsupported(_))
    ));
}

#[test]
fn effects_require_a_third_color_attachment_without_penalizing_legacy_gbuffer() {
    assert!(validate_request([1, 1], 2, false).is_ok());
    assert!(matches!(
        validate_request([1, 1], 2, true),
        Err(fluxel_rhi::api::Error::Unsupported(_))
    ));
    assert!(validate_request([1, 1], 3, true).is_ok());
    let effects = gbuffer_desc("effects", [1, 1], TextureFormat::Rgba16Float, 1);
    assert_eq!(effects.format, TextureFormat::Rgba16Float);
    assert_eq!(
        effects.usage,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING
    );
}
