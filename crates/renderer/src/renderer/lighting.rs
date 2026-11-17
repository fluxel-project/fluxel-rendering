//! Shared concrete bind layout for separately owned diffuse and specular inputs.

use crate::residency::{EnvironmentFrameOwner, SpecularEnvironmentFrameOwner};
use fluxel_rhi::{api::*, backend::ResourceBackend};

pub(super) fn layout_entries(diffuse: bool, specular: bool) -> Vec<BindGroupLayoutEntry> {
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let texture = || BindingType::SampledTexture {
        sample: TextureSampleType::Float,
        dimension: TextureViewDimension::D2,
        multisampled: false,
    };
    let sampler = || BindingType::Sampler {
        filtering: true,
        comparison: false,
    };
    let mut entries = Vec::new();
    if diffuse {
        entries.extend([entry(0, texture()), entry(1, sampler())]);
    }
    if specular {
        entries.extend([
            entry(2, texture()),
            entry(3, sampler()),
            entry(4, texture()),
            entry(5, sampler()),
            entry(
                6,
                BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 112,
                },
            ),
        ]);
    }
    entries
}

pub(super) fn bind_group<B: ResourceBackend>(
    device: &Device<B>,
    layout: &BindGroupLayout<B>,
    diffuse: Option<&EnvironmentFrameOwner<B>>,
    specular: Option<&SpecularEnvironmentFrameOwner<B>>,
    uses_diffuse: bool,
    uses_specular: bool,
) -> Result<BindGroup<B>> {
    let mut entries = Vec::new();
    if uses_diffuse {
        let owner = diffuse.ok_or(Error::InvalidState)?;
        owner.validate_for_sampling(device.id())?;
        entries.extend([
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&owner.view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&owner.sampler),
            },
        ]);
    }
    if uses_specular {
        let owner = specular.ok_or(Error::InvalidState)?;
        owner.validate_for_sampling(device.id())?;
        entries.extend([
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&owner.radiance.view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Sampler(&owner.radiance_sampler),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&owner.brdf_lut.view),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::Sampler(&owner.brdf_lut_sampler),
            },
            BindGroupEntry {
                binding: 6,
                resource: BindingResource::Buffer {
                    buffer: &owner.view_uniform,
                    range: 0..112,
                },
            },
        ]);
    }
    device.create_bind_group(&BindGroupDesc {
        label: Some("PBR environment inputs".into()),
        layout,
        entries: &entries,
    })
}

pub(super) fn allocate_specular<B: ResourceBackend>(
    device: &Device<B>,
    snapshot: &fluxel_assets::AssetSnapshot<
        crate::EnvironmentSpecularIblAsset,
        crate::EnvironmentSpecularIblData,
    >,
    view: &crate::RenderView,
) -> Result<SpecularEnvironmentFrameOwner<B>> {
    let data = snapshot.value();
    if !data.is_valid() {
        return Err(Error::InvalidRange);
    }
    let max_lod = (data.radiance_mips.len() - 1) as f32;
    let bytes = crate::frame_data::specular_view::pack_specular_view(
        view,
        max_lod,
        data.lod.scale,
        data.lod.offset,
    )?;
    SpecularEnvironmentFrameOwner::allocate(device, data, &bytes)
}
