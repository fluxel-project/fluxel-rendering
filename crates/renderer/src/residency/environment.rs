//! Concrete per-frame ownership for a selected diffuse-irradiance map.

use fluxel_rhi::{
    api::{
        AccessMask, AddressMode, Buffer, BufferDesc, BufferUsage, BufferUses, Device, DeviceId,
        Error, FilterMode, ImageLayout, MemoryClass, Sampler, SamplerDesc, StageMask, Support,
        Texture, TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureUses,
        TextureView, TextureViewDesc, TextureViewDimension,
    },
    backend::ResourceBackend,
};

use crate::EnvironmentIrradianceData;
use crate::graph::ResourceInstanceId;

/// Concrete environment owner retained by a submitted frame.
pub(crate) struct EnvironmentFrameOwner<B: ResourceBackend> {
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) upload_incoming: BufferUses,
    pub(crate) texture: Texture<B>,
    pub(crate) texture_instance: ResourceInstanceId,
    pub(crate) view: TextureView<B>,
    pub(crate) view_desc: TextureViewDesc,
    pub(crate) desc: TextureDesc,
    pub(crate) sampler: Sampler<B>,
    pub(crate) sampler_desc: SamplerDesc,
    pub(crate) device: DeviceId,
    pub(crate) filterable: bool,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
    pub(crate) upload_bytes: Vec<u8>,
    pub(crate) bytes_per_row: u32,
}

impl<B: ResourceBackend> EnvironmentFrameOwner<B> {
    /// Allocates from the exact selected snapshot payload before Graph work.
    pub(crate) fn allocate(
        device: &Device<B>,
        data: &EnvironmentIrradianceData,
    ) -> Result<Self, Error> {
        if !data.is_valid() {
            return Err(Error::InvalidRange);
        }
        let bytes_per_row = data.extent[0].checked_mul(4).ok_or(Error::InvalidRange)?;
        let aligned = bytes_per_row.checked_add(255).ok_or(Error::InvalidRange)? / 256 * 256;
        let size = usize::try_from(
            u64::from(aligned)
                .checked_mul(u64::from(data.extent[1]))
                .ok_or(Error::InvalidRange)?,
        )
        .map_err(|_| Error::InvalidRange)?;
        let mut upload_bytes = vec![0; size];
        let tight = usize::try_from(bytes_per_row).map_err(|_| Error::InvalidRange)?;
        for (row, source) in data.rgba8.chunks_exact(tight).enumerate() {
            let start = row
                .checked_mul(usize::try_from(aligned).map_err(|_| Error::InvalidRange)?)
                .ok_or(Error::InvalidRange)?;
            upload_bytes[start..start + tight].copy_from_slice(source);
        }
        let desc = TextureDesc {
            label: Some("fluxel environment diffuse irradiance".into()),
            dimension: TextureDimension::D2,
            extent: [data.extent[0], data.extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: Vec::new(),
        };
        let filterable = match device.texture_support(&desc) {
            Support::Supported(s) if s.filterable => true,
            Support::Supported(_) => {
                return Err(Error::Unsupported("filterable environment irradiance"));
            }
            Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
        };
        let texture = device.create_texture(&desc)?;
        let view_desc = TextureViewDesc {
            format: desc.format,
            dimension: TextureViewDimension::D2,
            aspects: fluxel_rhi::api::TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        };
        let view = device.create_view(&texture, &view_desc)?;
        let upload_desc = BufferDesc {
            label: Some("fluxel environment diffuse irradiance upload".into()),
            size: u64::try_from(size).map_err(|_| Error::InvalidRange)?,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        };
        let upload = device.create_buffer(&upload_desc)?;
        let sampler_desc = SamplerDesc {
            label: Some("fluxel environment irradiance sampler".into()),
            address: [
                fluxel_rhi::api::AddressMode::Repeat,
                fluxel_rhi::api::AddressMode::ClampToEdge,
                fluxel_rhi::api::AddressMode::ClampToEdge,
            ],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..0.0,
            anisotropy: 1,
            compare: None,
            border: None,
        };
        let sampler = device.create_sampler(&sampler_desc)?;
        Ok(Self {
            upload,
            upload_desc,
            upload_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            upload_incoming: BufferUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            texture,
            texture_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            view,
            view_desc,
            desc,
            sampler,
            sampler_desc,
            device: device.id(),
            filterable,
            incoming: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            layout: ImageLayout::Undefined,
            upload_bytes,
            bytes_per_row: aligned,
        })
    }

    pub(crate) fn validate_for_sampling(&self, device: DeviceId) -> Result<(), Error> {
        if self.device != device
            || self.upload.device() != device
            || self.texture.device() != device
            || self.view.device() != device
            || self.sampler.device() != device
            || !self.filterable
            || self.upload_desc.size
                != u64::from(self.bytes_per_row) * u64::from(self.desc.extent[1])
            || self.upload_desc.size
                != u64::try_from(self.upload_bytes.len()).map_err(|_| Error::InvalidRange)?
            || !self
                .upload_desc
                .usage
                .contains(BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE)
            || self.upload_desc.memory != MemoryClass::Upload
            || self.upload_incoming.stages != StageMask::EMPTY
            || self.upload_incoming.accesses != AccessMask::EMPTY
            || self.bytes_per_row % 256 != 0
            || self.bytes_per_row
                < self.desc.extent[0]
                    .checked_mul(4)
                    .ok_or(Error::InvalidRange)?
            || self.desc.dimension != TextureDimension::D2
            || self.desc.extent.contains(&0)
            || self.desc.array_layers != 1
            || self.desc.format != TextureFormat::Rgba8Unorm
            || self.desc.mip_levels != 1
            || self.desc.samples != 1
            || !self
                .desc
                .usage
                .contains(TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING)
            || self.view_desc.format != self.desc.format
            || self.view_desc.dimension != TextureViewDimension::D2
            || self.view_desc.aspects != fluxel_rhi::api::TextureAspects::COLOR
            || self.view_desc.mips != (0..1)
            || self.view_desc.layers != (0..1)
            || self.sampler_desc.address
                != [
                    AddressMode::Repeat,
                    AddressMode::ClampToEdge,
                    AddressMode::ClampToEdge,
                ]
            || self.sampler_desc.mag_filter != FilterMode::Linear
            || self.sampler_desc.min_filter != FilterMode::Linear
            || self.sampler_desc.mip_filter != FilterMode::Nearest
            || self.sampler_desc.lod != (0.0..0.0)
            || self.sampler_desc.anisotropy != 1
            || self.sampler_desc.compare.is_some()
            || self.sampler_desc.border.is_some()
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
}
