//! Immutable color-texture mip metadata validation.

use super::*;
use crate::FeatureIssuer;

fn source() -> ForwardDrawSource {
    ForwardDrawSource::Feature {
        feature: FeatureIssuer::new(0, 0).issue("mip-validation").unwrap(),
        local_id: 0,
    }
}

fn texture(mip_levels: u32, encoding: TextureEncoding) -> ColorTextureData {
    ColorTextureData {
        extent: [4, 2],
        encoding,
        mip_levels,
        rgba8: vec![0; 4 * 2 * 4],
    }
}

#[test]
fn color_texture_mip_count_is_a_bounded_asset_fact() {
    assert!(
        forward_validation::validate_texture(&texture(1, TextureEncoding::Linear), source())
            .is_ok()
    );
    assert!(
        forward_validation::validate_texture(&texture(3, TextureEncoding::Linear), source())
            .is_ok()
    );
    assert!(matches!(
        forward_validation::validate_texture(&texture(0, TextureEncoding::Linear), source()),
        Err(ForwardPreparationError::InvalidTexture { .. })
    ));
    assert!(matches!(
        forward_validation::validate_texture(&texture(4, TextureEncoding::Linear), source()),
        Err(ForwardPreparationError::InvalidTexture { .. })
    ));
}

#[test]
fn srgb_mip_generation_is_rejected_before_gpu_allocation() {
    assert!(matches!(
        forward_validation::validate_texture(&texture(2, TextureEncoding::Srgb), source()),
        Err(ForwardPreparationError::UnsupportedTextureMipGeneration {
            encoding: TextureEncoding::Srgb,
            mip_levels: 2,
            ..
        })
    ));
}
