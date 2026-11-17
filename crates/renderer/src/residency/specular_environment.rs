//! Concrete per-frame ownership for split-sum specular IBL.

use fluxel_rhi::{
    api::{
        AccessMask, AddressMode, Buffer, BufferDesc, BufferUsage, BufferUses, Device, DeviceId,
        Error, FilterMode, ImageLayout, MemoryClass, Sampler, SamplerDesc, StageMask, Support,
        Texture, TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureUses,
        TextureView, TextureViewDesc, TextureViewDimension,
    },
    backend::ResourceBackend,
};

use crate::{EnvironmentSpecularIblData, graph::ResourceInstanceId};

/// One actual physical copy from the common staging buffer into one texture mip.
#[derive(Clone, Debug)]
pub(crate) struct SpecularIblCopyFootprint {
    pub(crate) mip: u32,
    pub(crate) source_offset: u64,
    pub(crate) bytes_per_row: u32,
    pub(crate) rows_per_image: u32,
    pub(crate) extent: [u32; 3],
}

/// Actual texture ownership facts for one specular IBL input.
pub(crate) struct SpecularIblTextureOwner<B: ResourceBackend> {
    pub(crate) texture: Texture<B>,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) view: TextureView<B>,
    pub(crate) view_desc: TextureViewDesc,
    pub(crate) desc: TextureDesc,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
    pub(crate) device: DeviceId,
}

/// Concrete per-frame specular environment retained until completion.
pub(crate) struct SpecularEnvironmentFrameOwner<B: ResourceBackend> {
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) upload_incoming: BufferUses,
    pub(crate) upload_bytes: Vec<u8>,
    pub(crate) radiance: SpecularIblTextureOwner<B>,
    pub(crate) brdf_lut: SpecularIblTextureOwner<B>,
    pub(crate) radiance_sampler: Sampler<B>,
    pub(crate) radiance_sampler_desc: SamplerDesc,
    pub(crate) brdf_lut_sampler: Sampler<B>,
    pub(crate) brdf_lut_sampler_desc: SamplerDesc,
    pub(crate) radiance_copies: Vec<SpecularIblCopyFootprint>,
    pub(crate) brdf_lut_copy: SpecularIblCopyFootprint,
    /// Source range for the exact 112-byte view uniform copied into `view_uniform`.
    pub(crate) view_uniform_copy: std::ops::Range<u64>,
    pub(crate) view_uniform: Buffer<B>,
    pub(crate) view_uniform_desc: BufferDesc,
    pub(crate) view_uniform_instance: ResourceInstanceId,
    pub(crate) view_uniform_incoming: BufferUses,
    pub(crate) device: DeviceId,
}

impl<B: ResourceBackend> SpecularEnvironmentFrameOwner<B> {
    /// Admits static IBL payload plus frame-owned camera bytes before allocating.
    pub(crate) fn allocate(
        device: &Device<B>,
        data: &EnvironmentSpecularIblData,
        view_uniform_bytes: &[u8; 112],
    ) -> std::result::Result<Self, Error> {
        if !data.is_valid() {
            return Err(Error::InvalidRange);
        }
        let mip_levels =
            u32::try_from(data.radiance_mips.len()).map_err(|_| Error::InvalidRange)?;
        let radiance_desc = TextureDesc {
            label: Some("fluxel environment specular radiance".into()),
            dimension: TextureDimension::D2,
            extent: [
                data.radiance_mips[0].extent[0],
                data.radiance_mips[0].extent[1],
                1,
            ],
            array_layers: 1,
            format: TextureFormat::Rgba16Float,
            mip_levels,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: Vec::new(),
        };
        let brdf_lut_desc = TextureDesc {
            label: Some("fluxel environment specular BRDF LUT".into()),
            dimension: TextureDimension::D2,
            extent: [data.brdf_lut.extent[0], data.brdf_lut.extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Rgba16Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: Vec::new(),
        };
        require_filterable(device, &radiance_desc)?;
        require_filterable(device, &brdf_lut_desc)?;
        let (upload_bytes, radiance_copies, brdf_lut_copy, view_uniform_copy) =
            staging_payload(data, view_uniform_bytes)?;
        let upload_desc = BufferDesc {
            label: Some("fluxel environment specular upload".into()),
            size: u64::try_from(upload_bytes.len()).map_err(|_| Error::InvalidRange)?,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        };
        let view_uniform_desc = BufferDesc {
            label: Some("fluxel environment specular view uniform".into()),
            size: 112,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        };
        let radiance = device.create_texture(&radiance_desc)?;
        let radiance_view_desc = color_view(&radiance_desc, 0..mip_levels);
        let radiance_view = device.create_view(&radiance, &radiance_view_desc)?;
        let brdf_lut = device.create_texture(&brdf_lut_desc)?;
        let brdf_lut_view_desc = color_view(&brdf_lut_desc, 0..1);
        let brdf_lut_view = device.create_view(&brdf_lut, &brdf_lut_view_desc)?;
        let upload = device.create_buffer(&upload_desc)?;
        let view_uniform = device.create_buffer(&view_uniform_desc)?;
        let radiance_sampler_desc = SamplerDesc {
            label: Some("fluxel environment specular radiance sampler".into()),
            address: [
                AddressMode::Repeat,
                AddressMode::ClampToEdge,
                AddressMode::ClampToEdge,
            ],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..(mip_levels.saturating_sub(1) as f32),
            anisotropy: 1,
            compare: None,
            border: None,
        };
        let radiance_sampler = device.create_sampler(&radiance_sampler_desc)?;
        let brdf_lut_sampler_desc = SamplerDesc {
            label: Some("fluxel environment specular BRDF LUT sampler".into()),
            address: [
                AddressMode::ClampToEdge,
                AddressMode::ClampToEdge,
                AddressMode::ClampToEdge,
            ],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..0.0,
            anisotropy: 1,
            compare: None,
            border: None,
        };
        let brdf_lut_sampler = device.create_sampler(&brdf_lut_sampler_desc)?;
        let device_id = device.id();
        Ok(Self {
            upload,
            upload_desc,
            upload_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            upload_incoming: empty_buffer_uses(),
            upload_bytes,
            radiance: texture_owner(
                radiance,
                radiance_view,
                radiance_view_desc,
                radiance_desc,
                device_id,
            )?,
            brdf_lut: texture_owner(
                brdf_lut,
                brdf_lut_view,
                brdf_lut_view_desc,
                brdf_lut_desc,
                device_id,
            )?,
            radiance_sampler,
            radiance_sampler_desc,
            brdf_lut_sampler,
            brdf_lut_sampler_desc,
            radiance_copies,
            brdf_lut_copy,
            view_uniform_copy,
            view_uniform,
            view_uniform_desc,
            view_uniform_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            view_uniform_incoming: empty_buffer_uses(),
            device: device_id,
        })
    }

    /// Rechecks actual allocated facts before a shader binds this owner.
    pub(crate) fn validate_for_sampling(&self, device: DeviceId) -> std::result::Result<(), Error> {
        if self.device != device
            || self.upload.device() != device
            || self.view_uniform.device() != device
            || self.radiance_sampler.device() != device
            || self.brdf_lut_sampler.device() != device
            || !valid_texture_owner(&self.radiance, device, true)
            || !valid_texture_owner(&self.brdf_lut, device, false)
            || self.upload_desc.size
                != u64::try_from(self.upload_bytes.len()).map_err(|_| Error::InvalidRange)?
            || self.upload_desc.usage != (BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE)
            || self.upload_desc.memory != MemoryClass::Upload
            || self.upload_incoming != empty_buffer_uses()
            || self.view_uniform_desc.size != 112
            || self.view_uniform_desc.usage != (BufferUsage::UNIFORM | BufferUsage::COPY_DST)
            || self.view_uniform_desc.memory != MemoryClass::Device
            || self.view_uniform_incoming != empty_buffer_uses()
            || self
                .view_uniform_copy
                .end
                .checked_sub(self.view_uniform_copy.start)
                != Some(112)
            || self.view_uniform_copy.end > self.upload_desc.size
            || self.view_uniform_copy.start % 512 != 0
            || !valid_radiance_sampler(&self.radiance_sampler_desc, self.radiance.desc.mip_levels)
            || !valid_brdf_lut_sampler(&self.brdf_lut_sampler_desc)
            || !valid_footprints(self)
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
}

fn require_filterable<B: ResourceBackend>(
    device: &Device<B>,
    desc: &TextureDesc,
) -> std::result::Result<(), Error> {
    match device.texture_support(desc) {
        Support::Supported(support) if support.filterable => Ok(()),
        Support::Supported(_) => Err(Error::Unsupported("filterable specular environment")),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

fn texture_owner<B: ResourceBackend>(
    texture: Texture<B>,
    view: TextureView<B>,
    view_desc: TextureViewDesc,
    desc: TextureDesc,
    device: DeviceId,
) -> std::result::Result<SpecularIblTextureOwner<B>, Error> {
    Ok(SpecularIblTextureOwner {
        texture,
        instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
        view,
        view_desc,
        desc,
        incoming: TextureUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        },
        layout: ImageLayout::Undefined,
        device,
    })
}

fn color_view(desc: &TextureDesc, mips: std::ops::Range<u32>) -> TextureViewDesc {
    TextureViewDesc {
        format: desc.format,
        dimension: TextureViewDimension::D2,
        aspects: fluxel_rhi::api::TextureAspects::COLOR,
        mips,
        layers: 0..1,
    }
}

fn empty_buffer_uses() -> BufferUses {
    BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}

fn staging_payload(
    data: &EnvironmentSpecularIblData,
    view_uniform_bytes: &[u8; 112],
) -> std::result::Result<
    (
        Vec<u8>,
        Vec<SpecularIblCopyFootprint>,
        SpecularIblCopyFootprint,
        std::ops::Range<u64>,
    ),
    Error,
> {
    let mut bytes = Vec::new();
    let mut copies = Vec::with_capacity(data.radiance_mips.len());
    for (mip, image) in data.radiance_mips.iter().enumerate() {
        let copy = append_image(
            &mut bytes,
            u32::try_from(mip).map_err(|_| Error::InvalidRange)?,
            image.extent,
            &image.rgba16f_le,
        )?;
        copies.push(copy);
    }
    let brdf_lut_copy = append_image(
        &mut bytes,
        0,
        data.brdf_lut.extent,
        &data.brdf_lut.rgba16f_le,
    )?;
    let uniform_start = align_up(
        u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?,
        512,
    )?;
    bytes.resize(
        usize::try_from(uniform_start).map_err(|_| Error::InvalidRange)?,
        0,
    );
    bytes.extend_from_slice(view_uniform_bytes);
    Ok((
        bytes,
        copies,
        brdf_lut_copy,
        uniform_start..uniform_start + 112,
    ))
}

fn append_image(
    bytes: &mut Vec<u8>,
    mip: u32,
    extent: [u32; 2],
    source: &[u8],
) -> std::result::Result<SpecularIblCopyFootprint, Error> {
    let offset = align_up(
        u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?,
        512,
    )?;
    bytes.resize(usize::try_from(offset).map_err(|_| Error::InvalidRange)?, 0);
    let tight = extent[0].checked_mul(8).ok_or(Error::InvalidRange)?;
    let bytes_per_row =
        u32::try_from(align_up(u64::from(tight), 256)?).map_err(|_| Error::InvalidRange)?;
    let size = u64::from(bytes_per_row)
        .checked_mul(u64::from(extent[1]))
        .ok_or(Error::InvalidRange)?;
    let finish = offset.checked_add(size).ok_or(Error::InvalidRange)?;
    bytes.resize(usize::try_from(finish).map_err(|_| Error::InvalidRange)?, 0);
    let source_row = usize::try_from(tight).map_err(|_| Error::InvalidRange)?;
    let destination_row = usize::try_from(bytes_per_row).map_err(|_| Error::InvalidRange)?;
    let base = usize::try_from(offset).map_err(|_| Error::InvalidRange)?;
    for (row, input) in source.chunks_exact(source_row).enumerate() {
        let start = base
            .checked_add(
                row.checked_mul(destination_row)
                    .ok_or(Error::InvalidRange)?,
            )
            .ok_or(Error::InvalidRange)?;
        bytes[start..start + source_row].copy_from_slice(input);
    }
    Ok(SpecularIblCopyFootprint {
        mip,
        source_offset: offset,
        bytes_per_row,
        rows_per_image: extent[1],
        extent: [extent[0], extent[1], 1],
    })
}

fn align_up(value: u64, alignment: u64) -> std::result::Result<u64, Error> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(Error::InvalidRange)
}

fn valid_texture_owner<B: ResourceBackend>(
    owner: &SpecularIblTextureOwner<B>,
    device: DeviceId,
    radiance: bool,
) -> bool {
    owner.device == device
        && owner.texture.device() == device
        && owner.view.device() == device
        && owner.desc.dimension == TextureDimension::D2
        && !owner.desc.extent.contains(&0)
        && owner.desc.extent[2] == 1
        && owner.desc.array_layers == 1
        && owner.desc.format == TextureFormat::Rgba16Float
        && owner.desc.mip_levels
            == if radiance {
                u32::BITS
                    - owner.desc.extent[0]
                        .max(owner.desc.extent[1])
                        .leading_zeros()
            } else {
                1
            }
        && owner.desc.samples == 1
        && owner.desc.usage == (TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING)
        && owner.view_desc.format == owner.desc.format
        && owner.view_desc.dimension == TextureViewDimension::D2
        && owner.view_desc.aspects == fluxel_rhi::api::TextureAspects::COLOR
        && owner.view_desc.mips == (0..owner.desc.mip_levels)
        && owner.view_desc.layers == (0..1)
        && owner.incoming
            == TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            }
        && owner.layout == ImageLayout::Undefined
}

fn valid_radiance_sampler(desc: &SamplerDesc, mip_levels: u32) -> bool {
    desc.address
        == [
            AddressMode::Repeat,
            AddressMode::ClampToEdge,
            AddressMode::ClampToEdge,
        ]
        && desc.mag_filter == FilterMode::Linear
        && desc.min_filter == FilterMode::Linear
        && desc.mip_filter == FilterMode::Linear
        && desc.lod == (0.0..mip_levels.saturating_sub(1) as f32)
        && desc.anisotropy == 1
        && desc.compare.is_none()
        && desc.border.is_none()
}

fn valid_brdf_lut_sampler(desc: &SamplerDesc) -> bool {
    desc.address
        == [
            AddressMode::ClampToEdge,
            AddressMode::ClampToEdge,
            AddressMode::ClampToEdge,
        ]
        && desc.mag_filter == FilterMode::Linear
        && desc.min_filter == FilterMode::Linear
        && desc.mip_filter == FilterMode::Nearest
        && desc.lod == (0.0..0.0)
        && desc.anisotropy == 1
        && desc.compare.is_none()
        && desc.border.is_none()
}

fn valid_footprints<B: ResourceBackend>(owner: &SpecularEnvironmentFrameOwner<B>) -> bool {
    Some(owner.radiance_copies.len()) == usize::try_from(owner.radiance.desc.mip_levels).ok()
        && owner
            .radiance_copies
            .iter()
            .enumerate()
            .all(|(index, copy)| {
                valid_copy(copy, index as u32, &owner.radiance, owner.upload_desc.size)
            })
        && valid_copy(
            &owner.brdf_lut_copy,
            0,
            &owner.brdf_lut,
            owner.upload_desc.size,
        )
}

fn valid_copy<B: ResourceBackend>(
    copy: &SpecularIblCopyFootprint,
    mip: u32,
    texture: &SpecularIblTextureOwner<B>,
    upload_size: u64,
) -> bool {
    let extent = [
        (texture.desc.extent[0] >> mip).max(1),
        (texture.desc.extent[1] >> mip).max(1),
        1,
    ];
    copy.mip == mip
        && copy.extent == extent
        && copy.rows_per_image == extent[1]
        && copy.source_offset % 512 == 0
        && copy.bytes_per_row % 256 == 0
        && copy.bytes_per_row >= extent[0].saturating_mul(8)
        && copy
            .source_offset
            .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
            .is_some_and(|end| end <= upload_size)
}
