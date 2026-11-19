//! Persistent and staging resource facts for the Forward consumer.

use super::*;

/// Source bytes and declared layout for one complete vertex stream.
pub(crate) struct ForwardStreamInput {
    pub(crate) bytes: Vec<u8>,
    pub(crate) layout: VertexBufferLayout,
}

/// Source bytes and copied extent for one RGBA8 material texture base level.
pub(crate) struct ForwardTextureInput {
    pub(crate) bytes: Vec<u8>,
    pub(crate) extent: [u32; 2],
    pub(crate) bytes_per_row: u32,
    pub(crate) encoding: crate::preparation::TextureEncoding,
    /// Exact total mip count requested by the immutable asset.
    pub(crate) mip_levels: u32,
}

/// Copy-aligned bytes for one layered image or one complete volume mip.
pub(crate) struct ForwardTextureCopyInput {
    pub(crate) bytes: Vec<u8>,
    pub(crate) extent: [u32; 3],
    pub(crate) bytes_per_row: u32,
}

/// An immutable linear `RGBA16Float` texture whose complete mip chain was
/// authored by the asset producer.
pub(crate) struct ForwardAuthoredRgba16FloatTextureInput {
    pub(crate) mips: Vec<ForwardTextureCopyInput>,
}

/// View of a layered material image.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum LayeredTextureView {
    Array,
    Cube,
}

impl LayeredTextureView {
    pub(crate) const fn dimension(self) -> TextureViewDimension {
        match self {
            Self::Array => TextureViewDimension::D2Array,
            Self::Cube => TextureViewDimension::Cube,
        }
    }
}

/// Immutable RGBA8 images ordered by mip, then by layer.
/// A generated chain carries the base image for every layer.
pub(crate) struct ForwardArrayTextureInput {
    pub(crate) layers: u32,
    pub(crate) mip_levels: u32,
    pub(crate) copies: Vec<ForwardTextureCopyInput>,
    pub(crate) generate_mips: bool,
    pub(crate) view: LayeredTextureView,
    pub(crate) encoding: crate::preparation::TextureEncoding,
}

/// Concrete texture and view facts consumed during residency allocation.
pub(crate) struct ForwardTextureAllocation {
    pub(crate) desc: TextureDesc,
    pub(crate) view_dimension: TextureViewDimension,
}

/// One resident material texture source.
pub(crate) enum ForwardResidentTextureInput {
    Generated(ForwardTextureInput),
    Array(ForwardArrayTextureInput),
    Volume {
        mips: Vec<ForwardTextureCopyInput>,
        encoding: crate::preparation::TextureEncoding,
        generated_mips: Option<u32>,
    },
    AuthoredRgba16Float(ForwardAuthoredRgba16FloatTextureInput),
}

impl From<ForwardTextureInput> for ForwardResidentTextureInput {
    fn from(value: ForwardTextureInput) -> Self {
        Self::Generated(value)
    }
}

/// One device-local vertex stream and its allocation facts.
pub(crate) struct ForwardVertexStream<B: ResourceBackend> {
    pub(crate) buffer: Buffer<B>,
    pub(crate) desc: BufferDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) incoming: BufferUses,
}

/// Device-local texture and its allocation facts.
pub(crate) struct ForwardTexture<B: ResourceBackend> {
    pub(crate) texture: Texture<B>,
    pub(crate) view: TextureView<B>,
    pub(crate) view_dimension: TextureViewDimension,
    pub(crate) desc: TextureDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
    /// Device that allocated this owner, retained for binding provenance checks.
    pub(crate) device: DeviceId,
    /// Filterability reported for this exact allocation descriptor.
    pub(crate) filterable: bool,
}

/// A concrete sampler retained with its compatible texture owners.
pub(crate) struct ForwardSampler<B: ResourceBackend> {
    pub(crate) sampler: Sampler<B>,
    pub(crate) desc: SamplerDesc,
}

/// Persistent logical geometry/material realization for one device.
pub(crate) struct ForwardResidentBuffers<B: ResourceBackend> {
    pub(crate) device: DeviceId,
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) upload_incoming: BufferUses,
    pub(crate) streams: Vec<ForwardVertexStream<B>>,
    pub(crate) textures: Vec<ForwardTexture<B>>,
    pub(crate) samplers: Vec<ForwardSampler<B>>,
}

/// One source-to-device vertex copy, with physical byte offsets.
#[derive(Clone, Debug)]
pub(crate) struct ForwardVertexUpload {
    pub(crate) stream: usize,
    pub(crate) source: Range<u64>,
    pub(crate) destination: Range<u64>,
}

/// One source-to-device texture copy, with physical copy layout.
#[derive(Clone, Debug)]
pub(crate) struct ForwardTextureUpload {
    /// Physical resident slot written by this copy.
    pub(crate) texture: usize,
    /// Exact destination mip level selected by this copy.
    pub(crate) mip_level: u32,
    /// Exact destination array layer selected by this copy.
    pub(crate) array_layer: u32,
    pub(crate) source_offset: u64,
    pub(crate) bytes_per_row: u32,
    pub(crate) rows_per_image: u32,
    pub(crate) extent: [u32; 3],
}

/// Exact upload map contents and declared copy ranges for a resident product.
pub(crate) struct ForwardResidentUpload {
    pub(crate) bytes: Vec<u8>,
    pub(crate) vertices: Vec<ForwardVertexUpload>,
    pub(crate) textures: Vec<ForwardTextureUpload>,
}
