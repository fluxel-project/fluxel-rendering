//! Fresh buffer allocation for one byte-copy submission.

use fluxel_rhi::{
    api::{
        Buffer, BufferDesc, BufferUsage, Device, Error, MemoryClass, Result, Support, Texture,
        TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureView, TextureViewDesc,
        TextureViewDimension,
    },
    backend::ResourceBackend,
};

use crate::graph::ResourceInstanceId;

/// The source and target buffers owned by exactly one submission.
pub(crate) struct UploadBuffers<B: ResourceBackend> {
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) target: Buffer<B>,
    pub(crate) target_desc: BufferDesc,
    pub(crate) target_instance: ResourceInstanceId,
}

/// Per-submission resources for the fixed offscreen raster fixture.
///
/// The target is deliberately non-transient because its pixels are copied to a
/// readback buffer. All resources are dedicated and remain owned by the
/// submission until the exact final completion point.
pub(crate) struct RasterBuffers<B: ResourceBackend> {
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) vertex: Buffer<B>,
    pub(crate) vertex_desc: BufferDesc,
    pub(crate) vertex_instance: ResourceInstanceId,
    pub(crate) target: Texture<B>,
    pub(crate) target_desc: TextureDesc,
    pub(crate) target_instance: ResourceInstanceId,
    pub(crate) view: TextureView<B>,
    pub(crate) readback: Buffer<B>,
    pub(crate) readback_desc: BufferDesc,
    pub(crate) readback_instance: ResourceInstanceId,
}

/// Per-submission resources for the fixed raster fixture rendered to a surface.
///
/// The acquired surface image is intentionally not stored here. Its lease is
/// transferred to the RHI presentation protocol and remains owned by the
/// execution outcome until acceptance is known.
pub(crate) struct SurfaceRasterBuffers<B: ResourceBackend> {
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) vertex: Buffer<B>,
    pub(crate) vertex_desc: BufferDesc,
    pub(crate) vertex_instance: ResourceInstanceId,
}

/// Per-submission resources for the fixed in-place `u32` compute fixture.
pub(crate) struct ComputeBuffers<B: ResourceBackend> {
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) storage: Buffer<B>,
    pub(crate) storage_desc: BufferDesc,
    pub(crate) storage_instance: ResourceInstanceId,
}

/// Dedicated owners for one fixed read-set raster submission.
pub(crate) struct ReadSetRasterBuffers<B: ResourceBackend> {
    pub(crate) source_initial: fluxel_rhi::api::TextureUses,
    pub(crate) source_layout: fluxel_rhi::api::ImageLayout,
    pub(crate) target_initial: fluxel_rhi::api::TextureUses,
    pub(crate) target_layout: fluxel_rhi::api::ImageLayout,
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) uniform: Buffer<B>,
    pub(crate) uniform_desc: BufferDesc,
    pub(crate) uniform_instance: ResourceInstanceId,
    pub(crate) source: Texture<B>,
    pub(crate) source_desc: TextureDesc,
    pub(crate) source_instance: ResourceInstanceId,
    pub(crate) source_view: TextureView<B>,
    pub(crate) target: Texture<B>,
    pub(crate) target_desc: TextureDesc,
    pub(crate) target_instance: ResourceInstanceId,
    pub(crate) target_view: TextureView<B>,
    pub(crate) readback: Buffer<B>,
    pub(crate) readback_desc: BufferDesc,
    pub(crate) readback_instance: ResourceInstanceId,
}

impl<B: ResourceBackend> ReadSetRasterBuffers<B> {
    pub(crate) fn allocate(device: &Device<B>, uniform_end: u64, row_pitch: u32) -> Result<Self> {
        if uniform_end < 16 {
            return Err(Error::InvalidRange);
        }
        let upload_desc = BufferDesc {
            label: Some("fluxel renderer read-set upload".into()),
            size: crate::graph::read_set_raster::UPLOAD_BYTES,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        };
        let uniform_desc = BufferDesc {
            label: Some("fluxel renderer read-set uniform".into()),
            size: uniform_end,
            usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
            memory: MemoryClass::Device,
        };
        let source_desc = fixture_texture(
            "fluxel renderer read-set source",
            [1, 1],
            TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        );
        let target_desc = fixture_texture(
            "fluxel renderer read-set target",
            [8, 8],
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        );
        let readback_desc = BufferDesc {
            label: Some("fluxel renderer read-set readback".into()),
            size: u64::from(row_pitch) * 8,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        };
        for desc in [&upload_desc, &uniform_desc, &readback_desc] {
            require_support(device, desc)?;
        }
        for desc in [&source_desc, &target_desc] {
            match device.texture_support(desc) {
                Support::Supported(_) => (),
                Support::Unsupported(r) => return Err(Error::Unsupported(r)),
            }
        }
        let upload = device.create_buffer(&upload_desc)?;
        let uniform = device.create_buffer(&uniform_desc)?;
        let source = device.create_texture(&source_desc)?;
        let target = device.create_texture(&target_desc)?;
        let view = |texture: &Texture<B>| {
            device.create_view(
                texture,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: fluxel_rhi::api::TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
            )
        };
        let source_view = view(&source)?;
        let target_view = view(&target)?;
        let readback = device.create_buffer(&readback_desc)?;
        Ok(Self {
            source_initial: fluxel_rhi::api::TextureUses {
                stages: fluxel_rhi::api::StageMask::EMPTY,
                accesses: fluxel_rhi::api::AccessMask::EMPTY,
            },
            source_layout: fluxel_rhi::api::ImageLayout::Undefined,
            target_initial: fluxel_rhi::api::TextureUses {
                stages: fluxel_rhi::api::StageMask::EMPTY,
                accesses: fluxel_rhi::api::AccessMask::EMPTY,
            },
            target_layout: fluxel_rhi::api::ImageLayout::Undefined,
            upload,
            upload_desc,
            upload_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            uniform,
            uniform_desc,
            uniform_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            source,
            source_desc,
            source_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            source_view,
            target,
            target_desc,
            target_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            target_view,
            readback,
            readback_desc,
            readback_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
        })
    }
}

fn fixture_texture(label: &str, extent: [u32; 2], usage: TextureUsage) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage,
        view_formats: Vec::new(),
    }
}

impl<B: ResourceBackend> ComputeBuffers<B> {
    /// Allocates a fresh upload source and a device-local storage/readback buffer.
    pub(crate) fn allocate(device: &Device<B>, len: u64) -> Result<Self> {
        let upload = BufferDesc {
            label: Some("fluxel renderer compute upload".to_owned()),
            size: len,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        };
        let storage = BufferDesc {
            label: Some("fluxel renderer compute storage".to_owned()),
            size: len,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC | BufferUsage::STORAGE,
            memory: MemoryClass::Device,
        };
        require_support(device, &upload)?;
        require_support(device, &storage)?;
        let upload_buffer = device.create_buffer(&upload)?;
        let storage_buffer = device.create_buffer(&storage)?;
        let upload_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        let storage_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        Ok(Self {
            upload: upload_buffer,
            upload_desc: upload,
            upload_instance,
            storage: storage_buffer,
            storage_desc: storage,
            storage_instance,
        })
    }
}

impl<B: ResourceBackend> RasterBuffers<B> {
    /// Allocates resources for one fixed-format RGBA8 triangle fixture.
    pub(crate) fn allocate(
        device: &Device<B>,
        vertex_bytes: u64,
        extent: [u32; 2],
        row_pitch: u32,
    ) -> Result<Self> {
        let upload_desc = BufferDesc {
            label: Some("fluxel renderer raster upload".to_owned()),
            size: vertex_bytes,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        };
        let vertex_desc = BufferDesc {
            label: Some("fluxel renderer raster vertices".to_owned()),
            size: vertex_bytes,
            usage: BufferUsage::COPY_DST | BufferUsage::VERTEX,
            memory: MemoryClass::Device,
        };
        let target_desc = TextureDesc {
            label: Some("fluxel renderer raster target".to_owned()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: Vec::new(),
        };
        let readback_size = u64::from(row_pitch)
            .checked_mul(u64::from(extent[1]))
            .ok_or(Error::InvalidRange)?;
        let readback_desc = BufferDesc {
            label: Some("fluxel renderer raster readback".to_owned()),
            size: readback_size,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        };
        require_support(device, &upload_desc)?;
        require_support(device, &vertex_desc)?;
        match device.texture_support(&target_desc) {
            Support::Supported(_) => {}
            Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
        }
        require_support(device, &readback_desc)?;
        let target = device.create_texture(&target_desc)?;
        let target_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        let view = device.create_view(
            &target,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: fluxel_rhi::api::TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let upload = device.create_buffer(&upload_desc)?;
        let vertex = device.create_buffer(&vertex_desc)?;
        let upload_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        let vertex_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        let readback_buffer = device.create_buffer(&readback_desc)?;
        let readback_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        Ok(Self {
            upload,
            upload_desc,
            upload_instance,
            vertex,
            vertex_desc,
            vertex_instance,
            target,
            target_desc,
            target_instance,
            view,
            readback: readback_buffer,
            readback_desc,
            readback_instance,
        })
    }
}

impl<B: ResourceBackend> SurfaceRasterBuffers<B> {
    /// Allocates the upload and vertex buffers for one surface raster fixture.
    pub(crate) fn allocate(device: &Device<B>, vertex_bytes: u64) -> Result<Self> {
        let upload = BufferDesc {
            label: Some("fluxel renderer surface raster upload".to_owned()),
            size: vertex_bytes,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        };
        let vertex = BufferDesc {
            label: Some("fluxel renderer surface raster vertices".to_owned()),
            size: vertex_bytes,
            usage: BufferUsage::COPY_DST | BufferUsage::VERTEX,
            memory: MemoryClass::Device,
        };
        require_support(device, &upload)?;
        require_support(device, &vertex)?;
        let upload_buffer = device.create_buffer(&upload)?;
        let vertex_buffer = device.create_buffer(&vertex)?;
        let upload_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        let vertex_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        Ok(Self {
            upload: upload_buffer,
            upload_desc: upload,
            upload_instance,
            vertex: vertex_buffer,
            vertex_desc: vertex,
            vertex_instance,
        })
    }
}

impl<B: ResourceBackend> UploadBuffers<B> {
    /// Allocates an upload-mappable copy source and a device-local copy target.
    pub(crate) fn allocate(
        device: &Device<B>,
        source_bytes: u64,
        target_bytes: u64,
    ) -> Result<Self> {
        let upload = BufferDesc {
            label: Some("fluxel renderer upload".to_owned()),
            size: source_bytes,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        };
        let target = BufferDesc {
            label: Some("fluxel renderer target".to_owned()),
            size: target_bytes,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        };
        require_support(device, &upload)?;
        require_support(device, &target)?;
        let upload_buffer = device.create_buffer(&upload)?;
        let target_buffer = device.create_buffer(&target)?;
        let upload_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        let target_instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
        Ok(Self {
            upload: upload_buffer,
            upload_desc: upload,
            upload_instance,
            target: target_buffer,
            target_desc: target,
            target_instance,
        })
    }
}

fn require_support<B: ResourceBackend>(device: &Device<B>, desc: &BufferDesc) -> Result<()> {
    match device.buffer_support(desc) {
        Support::Supported(_) => Ok(()),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}
