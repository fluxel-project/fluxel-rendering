//! Stable shape of Forward specular environment resources and copies.

use super::{ResourceBackend, TextureShape, texture_shape};

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct SpecularIblShape {
    pub(crate) upload_size: u64,
    pub(crate) radiance: TextureShape,
    pub(crate) brdf_lut: TextureShape,
    pub(crate) radiance_copies: Vec<SpecularIblCopyShape>,
    pub(crate) brdf_lut_copy: SpecularIblCopyShape,
    pub(crate) view_uniform: (u64, std::ops::Range<u64>),
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct SpecularIblCopyShape {
    pub(crate) mip: u32,
    pub(crate) source_offset: u64,
    pub(crate) bytes_per_row: u32,
    pub(crate) rows_per_image: u32,
    pub(crate) extent: [u32; 3],
}

pub(super) fn specular_copy_shape(
    copy: &crate::residency::SpecularIblCopyFootprint,
) -> SpecularIblCopyShape {
    SpecularIblCopyShape {
        mip: copy.mip,
        source_offset: copy.source_offset,
        bytes_per_row: copy.bytes_per_row,
        rows_per_image: copy.rows_per_image,
        extent: copy.extent,
    }
}

pub(super) fn specular_shape<B: ResourceBackend>(
    owner: &crate::residency::SpecularEnvironmentFrameOwner<B>,
) -> SpecularIblShape {
    SpecularIblShape {
        upload_size: owner.upload_desc.size,
        radiance: texture_shape(
            &owner.radiance.desc,
            owner.radiance.incoming,
            owner.radiance.layout,
        ),
        brdf_lut: texture_shape(
            &owner.brdf_lut.desc,
            owner.brdf_lut.incoming,
            owner.brdf_lut.layout,
        ),
        radiance_copies: owner
            .radiance_copies
            .iter()
            .map(specular_copy_shape)
            .collect(),
        brdf_lut_copy: specular_copy_shape(&owner.brdf_lut_copy),
        view_uniform: (
            owner.view_uniform_desc.size,
            owner.view_uniform_copy.clone(),
        ),
    }
}
