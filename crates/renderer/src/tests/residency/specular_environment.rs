//! Specular IBL residency tests run beside the shared renderer witness.

use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;
use super::*;

use std::sync::atomic::Ordering;

use crate::{
    EnvironmentSpecularIblData, LinearSpecularLod, SpecularIblImage, SpecularIblMip,
    residency::SpecularEnvironmentFrameOwner,
};

fn image(extent: [u32; 2]) -> SpecularIblImage {
    let texels = usize::try_from(extent[0] * extent[1]).expect("small fixture");
    let pixel: Vec<_> = [0x3c00_u16; 4]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    SpecularIblImage {
        extent,
        rgba16f_le: pixel.repeat(texels),
    }
}

fn data() -> EnvironmentSpecularIblData {
    EnvironmentSpecularIblData {
        radiance_mips: vec![
            SpecularIblMip {
                extent: [2, 2],
                rgba16f_le: image([2, 2]).rgba16f_le,
            },
            SpecularIblMip {
                extent: [1, 1],
                rgba16f_le: image([1, 1]).rgba16f_le,
            },
        ],
        brdf_lut: image([2, 2]),
        lod: LinearSpecularLod {
            scale: 1.0,
            offset: 0.0,
        },
    }
}

#[test]
fn specular_owner_allocates_exact_two_textures_two_samplers_and_aligned_common_staging() {
    let (device, native) = fixture(Script::AcceptedExact);
    let owner = SpecularEnvironmentFrameOwner::allocate(&device, &data(), &[3; SPECULAR_VIEW_BYTES as usize])
        .expect("specular owner");
    assert_eq!(native.created_textures.load(Ordering::SeqCst), 2);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 2);
    assert_eq!(owner.radiance.desc.format, TextureFormat::Rgba16Float);
    assert_eq!(owner.radiance.desc.mip_levels, 2);
    assert_eq!(owner.brdf_lut.desc.mip_levels, 1);
    assert_eq!(owner.radiance_copies.len(), 2);
    assert!(
        owner
            .radiance_copies
            .iter()
            .all(|copy| copy.source_offset % 512 == 0 && copy.bytes_per_row % 256 == 0)
    );
    assert_eq!(owner.brdf_lut_copy.source_offset % 512, 0);
    assert_eq!(
        owner.view_uniform_copy.end - owner.view_uniform_copy.start,
        SPECULAR_VIEW_BYTES
    );
    assert_eq!(owner.view_uniform_copy.start % 512, 0);
    owner
        .validate_for_sampling(device.id())
        .expect("actual facts");
}

#[test]
fn capability_failure_is_refused_before_any_specular_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    native.texture_filterable.store(false, Ordering::SeqCst);
    assert!(matches!(
        SpecularEnvironmentFrameOwner::allocate(&device, &data(), &[0; SPECULAR_VIEW_BYTES as usize]),
        Err(Error::Unsupported("filterable specular environment"))
    ));
    assert_eq!(native.created_textures.load(Ordering::SeqCst), 0);
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 0);
}

#[test]
fn actual_texture_shape_and_complete_mip_chain_are_rechecked_before_sampling() {
    let (device, _) = fixture(Script::AcceptedExact);
    let mut owner = SpecularEnvironmentFrameOwner::allocate(&device, &data(), &[0; SPECULAR_VIEW_BYTES as usize])
        .expect("specular owner");
    owner.radiance.desc.extent[2] = 2;
    assert!(owner.validate_for_sampling(device.id()).is_err());
    owner.radiance.desc.extent[2] = 1;
    // A self-consistent truncated view/copy list must not hide missing mips.
    owner.radiance.desc.mip_levels = 1;
    owner.radiance.view_desc.mips = 0..1;
    owner.radiance_copies.pop();
    owner.radiance_sampler_desc.lod = 0.0..0.0;
    assert!(owner.validate_for_sampling(device.id()).is_err());
}
