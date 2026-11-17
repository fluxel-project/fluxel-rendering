use crate::{EnvironmentSpecularIblData, LinearSpecularLod, SpecularIblImage, SpecularIblMip};

fn pixel(r: u16, g: u16, b: u16, a: u16) -> Vec<u8> {
    [r, g, b, a]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect()
}

fn image(extent: [u32; 2]) -> SpecularIblImage {
    let texels = usize::try_from(extent[0] * extent[1]).unwrap();
    SpecularIblImage {
        extent,
        rgba16f_le: pixel(0x3c00, 0x3c00, 0x3c00, 0x3c00).repeat(texels),
    }
}

fn valid_data() -> EnvironmentSpecularIblData {
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
fn complete_half_float_mip_chain_and_babylon_linear_mapping_are_admitted() {
    let data = valid_data();
    assert!(data.is_valid());
    assert_eq!(data.lod.map(0.5, 2), Some(0.5));
    assert_eq!(data.lod.map(0.5, 5), Some(2.0));
    assert_eq!(data.lod.map(4.0, 2), Some(1.0));
}

#[test]
fn missing_mip_wrong_extent_bad_range_and_nonfinite_or_negative_rgb_are_refused() {
    let mut missing = valid_data();
    missing.radiance_mips.pop();
    assert!(!missing.is_valid());

    let mut extent = valid_data();
    extent.radiance_mips[1].extent = [2, 1];
    assert!(!extent.is_valid());

    let mut length = valid_data();
    length.brdf_lut.rgba16f_le.pop();
    assert!(!length.is_valid());

    let mut infinity = valid_data();
    infinity.radiance_mips[0].rgba16f_le[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(!infinity.is_valid());

    let mut negative = valid_data();
    negative.radiance_mips[0].rgba16f_le[..2].copy_from_slice(&0xbc00_u16.to_le_bytes());
    assert!(!negative.is_valid());
}

#[test]
fn nonfinite_lod_metadata_is_refused() {
    let mut data = valid_data();
    data.lod.scale = f32::NAN;
    assert!(!data.is_valid());
    assert_eq!(data.lod.map(0.5, 2), None);
}
