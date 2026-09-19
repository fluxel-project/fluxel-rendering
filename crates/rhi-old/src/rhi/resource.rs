//! Resources, transfer, and readback.
//!
//! This module owns rhi-design sections 11 to 18.
//!
//! # What it is
//!
//! It freezes only the resource semantics a caller can actually observe: size,
//! extent, mip and layer counts, format, usage, view compatibility, device
//! identity, logical lifetime, and upload/readback behaviour. Everything a
//! caller cannot observe or control stays out: native heap types, memory type
//! indices, storage and tiling modes, allocation offsets, resource states, and
//! CPU mapping modes.
//!
//! Usage flags are a creation-time correctness contract. A backend may not
//! bypass portable usage validation because a platform "happens to allow it":
//! a buffer created without `COPY_DST` cannot be an upload destination, a
//! texture without `COPY_SRC` cannot be read back, and a resource without
//! `STORAGE` cannot become a storage binding.
//!
//! # What it deliberately does not own
//!
//! There is no public map, so `HostAccessIntent` and `HostPreferred` do not
//! exist: under this API they have no user semantics to honour. Upload does not
//! require the caller to meet native staging alignment, because the caller's CPU
//! layout is not the GPU copy-buffer layout; RHI repacks into private staging
//! instead. Readback likewise returns an explicit layout rather than promising
//! tightly packed bytes, so RHI never pays a repack the caller did not ask for.
//!
//! Retirement is logical, not reference-counted by the caller: a dropped public
//! handle is not "the object is unreferenced", and is not "native backing may be
//! released now". Backing survives until the last CPU logical owner is gone
//! *and* every accepted GPU work item referencing the object is terminal.

use std::sync::Arc;

use super::format::{
    format_facts, BufferCopyLayoutLimits, TextureFormat, TextureViewCompatibility,
};
use super::platform::{DeviceIdentity, Label, ObjectId, RhiError, RhiErrorKind, RhiResult};
use super::submission::CompletionPoint;

/// A set of buffer usages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferUsage(u32);

impl BufferUsage {
    /// Usable as a copy source.
    pub const COPY_SRC: Self = Self(1 << 0);
    /// Usable as a copy destination.
    pub const COPY_DST: Self = Self(1 << 1);
    /// Usable as vertex input.
    pub const VERTEX: Self = Self(1 << 2);
    /// Usable as an index buffer.
    pub const INDEX: Self = Self(1 << 3);
    /// Usable as a uniform buffer binding.
    pub const UNIFORM: Self = Self(1 << 4);
    /// Usable as a storage buffer binding.
    pub const STORAGE: Self = Self(1 << 5);

    /// Whether every bit of `other` is present.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two usage sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no usage is set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw bits, for canonical hashing and diagnostics.
    pub fn bits(self) -> u32 {
        self.0
    }
}

/// A set of texture usages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureUsage(u32);

impl TextureUsage {
    /// Usable as a copy source.
    pub const COPY_SRC: Self = Self(1 << 0);
    /// Usable as a copy destination.
    pub const COPY_DST: Self = Self(1 << 1);
    /// Usable as a sampled texture binding.
    pub const SAMPLED: Self = Self(1 << 2);
    /// Usable as a storage texture binding.
    pub const STORAGE: Self = Self(1 << 3);
    /// Usable as a color attachment.
    pub const COLOR_ATTACHMENT: Self = Self(1 << 4);
    /// Usable as a depth and/or stencil attachment.
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(1 << 5);

    /// Whether every bit of `other` is present.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two usage sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no usage is set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw bits, for canonical hashing and diagnostics.
    pub fn bits(self) -> u32 {
        self.0
    }
}

/// A placement preference for a resource's memory.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceMemoryPreference {
    /// Chosen by the backend.
    Automatic,

    /// Prefer GPU/device-local placement where possible.
    ///
    /// This is a preference, not a correctness guarantee. UMA, WebGPU, and GL
    /// backends may treat it as equivalent to [`Self::Automatic`] or ignore it.
    DeviceLocalPreferred,
}

/// The logical dimension of a texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureDimension {
    /// One-dimensional.
    D1,
    /// Two-dimensional, possibly an array.
    D2,
    /// Three-dimensional.
    D3,
}

/// A three-component extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extent3d {
    /// The width in texels or blocks.
    pub width: u32,
    /// The height in texels or blocks.
    pub height: u32,
    /// The depth in texels or blocks.
    pub depth: u32,
}

impl Extent3d {
    /// A 1D extent; height and depth are one.
    pub fn d1(width: u32) -> Self {
        Self {
            width,
            height: 1,
            depth: 1,
        }
    }

    /// A 2D extent; depth is one.
    pub fn d2(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }

    /// A 3D extent.
    pub fn d3(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// Whether every component is non-zero.
    pub fn is_non_zero(self) -> bool {
        self.width > 0 && self.height > 0 && self.depth > 0
    }

    /// The largest component, used for the mip-level upper bound.
    pub fn max_dimension(self) -> u32 {
        self.width.max(self.height).max(self.depth)
    }

    /// The extent of `level`, each component halving and never reaching zero.
    ///
    /// One definition is shared by the view, the copy, the upload, and the
    /// readback, because a second spelling of "how big is mip 3" is a second
    /// answer to it, and the two would disagree first at the sizes where the
    /// question is hard.
    pub fn mip_extent(self, level: u32) -> Self {
        let shift = level.min(31);
        Self {
            width: (self.width >> shift).max(1),
            height: (self.height >> shift).max(1),
            depth: (self.depth >> shift).max(1),
        }
    }
}

/// One aspect of a texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureAspect {
    /// The color aspect.
    Color,
    /// The depth aspect.
    Depth,
    /// The stencil aspect.
    Stencil,
}

/// A set of texture aspects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureAspects(u8);

impl TextureAspects {
    pub(crate) const COLOR_BITS: u8 = 1 << 0;
    pub(crate) const DEPTH_BITS: u8 = 1 << 1;
    pub(crate) const STENCIL_BITS: u8 = 1 << 2;

    /// The color aspect bit.
    pub const COLOR: Self = Self(Self::COLOR_BITS);
    /// The depth aspect bit.
    pub const DEPTH: Self = Self(Self::DEPTH_BITS);
    /// The stencil aspect bit.
    pub const STENCIL: Self = Self(Self::STENCIL_BITS);

    pub(crate) const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Whether every bit of `other` is present.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two aspect sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no aspect is set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether exactly one aspect is set.
    pub fn is_single(self) -> bool {
        self.0.count_ones() == 1
    }

    /// The raw bits, for canonical hashing and diagnostics.
    pub fn bits(self) -> u8 {
        self.0
    }
}

/// A mip and array-layer range used for view and hazard tracking.
///
/// For [`TextureDimension::D3`] the array fields are fixed at `base_layer = 0`
/// and `layer_count = 1`: a 3D texture's Z slice is not an independent array
/// subresource, and its Z range is expressed only by a copy origin and extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureSubresourceRange {
    /// The aspects this range covers.
    pub aspects: TextureAspects,
    /// The first mip level.
    pub base_mip: u32,
    /// How many mip levels.
    pub mip_count: u32,
    /// The first array layer.
    pub base_layer: u32,
    /// How many array layers.
    pub layer_count: u32,
}

/// One mip level and a range of array layers, used by copy, upload, and
/// readback.
///
/// Semantically analogous to a Vulkan `ImageSubresourceLayers`, but not a native
/// struct. Each value may select exactly one aspect, so a depth/stencil copy
/// queries and encodes its depth and stencil routes separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureSubresourceLayers {
    /// The single aspect this layer range addresses.
    pub aspect: TextureAspect,
    /// The mip level.
    pub mip_level: u32,
    /// The first array layer.
    pub base_layer: u32,
    /// How many array layers.
    pub layer_count: u32,
}

impl TextureSubresourceLayers {
    /// Checks this value against a texture's dimension and shape.
    pub(crate) fn validate_for(
        &self,
        dimension: TextureDimension,
        mip_levels: u32,
        array_layers: u32,
        aspects: TextureAspects,
    ) -> RhiResult<()> {
        if self.mip_level >= mip_levels {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture subresource mip level is beyond the texture",
            ));
        }
        if self.layer_count == 0 || self.base_layer.checked_add(self.layer_count).is_none() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture subresource layer count is invalid",
            ));
        }
        if self.base_layer + self.layer_count > array_layers {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture subresource layer range is beyond the texture",
            ));
        }
        let selected = match self.aspect {
            TextureAspect::Color => TextureAspects::COLOR,
            TextureAspect::Depth => TextureAspects::DEPTH,
            TextureAspect::Stencil => TextureAspects::STENCIL,
        };
        if !aspects.contains(selected) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture subresource aspect is not present in the texture format",
            ));
        }
        match dimension {
            TextureDimension::D1 | TextureDimension::D3 => {
                if self.base_layer != 0 || self.layer_count != 1 {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "only a 2D texture addresses array layers in a subresource",
                    ));
                }
            }
            TextureDimension::D2 => {}
        }
        Ok(())
    }
}

/// A texel or block origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Origin3d {
    /// The X origin.
    pub x: u32,
    /// The Y origin.
    pub y: u32,
    /// The Z origin.
    pub z: u32,
}

impl Origin3d {
    /// The origin.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };
}

/// The CPU byte layout of texel data supplied by a caller.
///
/// This describes the caller's bytes only. It is not the GPU-side copy-buffer
/// layout requirement of [`super::RouteCapabilities`], and it does not have to
/// satisfy a 256-byte row pitch: RHI repacks into private staging when the
/// backend needs that alignment, because imposing a native copy footprint on an
/// asset loader's bytes would leak a backend detail into the portable API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HostTexelLayout {
    /// The byte distance between the starts of adjacent rows.
    pub bytes_per_row: u32,
    /// The number of rows between the starts of adjacent images, layers, or
    /// depth slices.
    pub rows_per_image: u32,
}

/// The dimension a texture view presents.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureViewDimension {
    /// One-dimensional.
    D1,
    /// Two-dimensional.
    D2,
    /// Two-dimensional with array layers.
    D2Array,
    /// A cube.
    Cube,
    /// A cube array.
    CubeArray,
    /// Three-dimensional.
    D3,
}

/// A buffer description.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferDescriptor {
    /// A diagnostic label.
    pub label: Label,
    /// The size in bytes.
    pub size: u64,
    /// The usage set.
    pub usage: BufferUsage,
    /// The placement preference.
    pub memory: ResourceMemoryPreference,
}

impl BufferDescriptor {
    /// A description of a `size`-byte buffer with `usage`.
    pub fn new(size: u64, usage: BufferUsage) -> Self {
        Self {
            label: Label::none(),
            size,
            usage,
            memory: ResourceMemoryPreference::Automatic,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }

    /// Sets the placement preference.
    pub fn with_memory_preference(mut self, preference: ResourceMemoryPreference) -> Self {
        self.memory = preference;
        self
    }

    /// Normalizes and validates the portable creation invariants.
    ///
    /// This is the single lowering shared by the capability query, creation
    /// validation, and backend creation, so those three never use three
    /// different sets of conditions. A zero-size buffer is refused rather than
    /// rounded up: the caller asked for an object with no bytes, and silently
    /// giving it a different one would hide the mistake until a later copy.
    pub(crate) fn normalize_and_validate(&self) -> RhiResult<()> {
        if self.size == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "buffer size must be non-zero",
            ));
        }
        if self.usage.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "buffer usage must be non-empty",
            ));
        }
        Ok(())
    }
}

/// A GPU buffer.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<dyn BufferBackend>,
}

impl core::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer")
            .field("id", &self.id())
            .field("size", &self.descriptor().size)
            .finish_non_exhaustive()
    }
}

impl Buffer {
    pub(crate) fn new(inner: Arc<dyn BufferBackend>) -> Self {
        Self { inner }
    }

    /// The backing this handle retains, for the device's retirement registry.
    pub(crate) fn backing(&self) -> Arc<dyn BufferBackend> {
        Arc::clone(&self.inner)
    }

    /// This buffer's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this buffer belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The descriptor this buffer was created from.
    pub fn descriptor(&self) -> &BufferDescriptor {
        self.inner.descriptor()
    }
}

/// The backend half of a [`Buffer`].
pub(crate) trait BufferBackend: Send + Sync + 'static {
    /// This buffer's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this buffer belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The descriptor this buffer was created from.
    fn descriptor(&self) -> &BufferDescriptor;
}

/// A byte range within a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferRange {
    /// The first byte.
    pub offset: u64,
    /// The number of bytes.
    pub size: u64,
}

impl BufferRange {
    /// A range of `size` bytes starting at `offset`.
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// The exclusive end offset, absent on integer overflow.
    ///
    /// P0 does not use an `offset` plus `WHOLE_BUFFER` sentinel; a range is
    /// always an explicit offset and size.
    pub fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.size)
    }

    /// Whether this range covers all of `buffer`.
    pub fn covers(&self, buffer: &Buffer) -> bool {
        self.size > 0 && self.end().is_some_and(|end| end <= buffer.descriptor().size)
    }
}

/// A buffer together with the range a binding sees.
#[derive(Clone, Debug)]
pub struct BufferBinding {
    /// The buffer.
    pub buffer: Buffer,
    /// The visible range.
    pub range: BufferRange,
}

impl BufferBinding {
    /// Binds `range` of `buffer`.
    pub fn new(buffer: Buffer, range: BufferRange) -> Self {
        Self { buffer, range }
    }

    /// Validates the range against the buffer and the device it belongs to.
    pub(crate) fn validate(&self, operation: &'static str) -> RhiResult<()> {
        if self.range.size == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "buffer binding range must be non-empty",
            )
            .at(operation));
        }
        if !self.range.covers(&self.buffer) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "buffer binding range is out of bounds",
            )
            .at(operation)
            .on(self.buffer.id()));
        }
        Ok(())
    }
}

/// A texture description.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureDescriptor {
    /// A diagnostic label.
    pub label: Label,

    /// The logical dimension.
    pub dimension: TextureDimension,
    /// The base-level extent.
    pub extent: Extent3d,

    /// The mip level count.
    pub mip_levels: u32,
    /// The array layer count.
    pub array_layers: u32,
    /// The sample count.
    pub sample_count: u32,

    /// The base format.
    pub format: TextureFormat,
    /// The usage set.
    pub usage: TextureUsage,

    /// The set of formats allowed for alternate-format views.
    ///
    /// The stored order is the order the caller declared. The *canonical* form
    /// is [`Self::canonicalized`], which sorts and deduplicates this vector;
    /// that form is what participates in capability queries, fingerprints, and
    /// capture. The raw vector is never an identity for this texture.
    pub view_formats: Vec<TextureFormat>,

    /// View intent that must be fixed at creation.
    pub view_compatibility: TextureViewCompatibility,

    /// The placement preference.
    pub memory: ResourceMemoryPreference,
}

impl TextureDescriptor {
    /// A 1D texture description.
    pub fn new_1d(width: u32, format: TextureFormat, usage: TextureUsage) -> Self {
        Self::base(TextureDimension::D1, Extent3d::d1(width), format, usage)
    }

    /// A 2D texture description.
    pub fn new_2d(width: u32, height: u32, format: TextureFormat, usage: TextureUsage) -> Self {
        Self::base(
            TextureDimension::D2,
            Extent3d::d2(width, height),
            format,
            usage,
        )
    }

    /// A 3D texture description.
    pub fn new_3d(
        width: u32,
        height: u32,
        depth: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self {
        Self::base(
            TextureDimension::D3,
            Extent3d::d3(width, height, depth),
            format,
            usage,
        )
    }

    fn base(
        dimension: TextureDimension,
        extent: Extent3d,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self {
        Self {
            label: Label::none(),
            dimension,
            extent,
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format,
            usage,
            view_formats: Vec::new(),
            view_compatibility: TextureViewCompatibility::NONE,
            memory: ResourceMemoryPreference::Automatic,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }

    /// Sets the mip level count.
    pub fn with_mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    /// Sets the array layer count.
    pub fn with_array_layers(mut self, layers: u32) -> Self {
        self.array_layers = layers;
        self
    }

    /// Sets the sample count.
    pub fn with_sample_count(mut self, samples: u32) -> Self {
        self.sample_count = samples;
        self
    }

    /// Adds one permitted alternate view format.
    pub fn with_view_format(mut self, format: TextureFormat) -> Self {
        self.view_formats.push(format);
        self
    }

    /// Declares the view intent that must be fixed at creation.
    pub fn with_view_compatibility(mut self, compatibility: TextureViewCompatibility) -> Self {
        self.view_compatibility = compatibility;
        self
    }

    /// Sets the placement preference.
    pub fn with_memory_preference(mut self, preference: ResourceMemoryPreference) -> Self {
        self.memory = preference;
        self
    }

    /// The canonical form of this descriptor.
    ///
    /// `view_formats` is a set, so the order a caller declared it in carries no
    /// meaning and duplicates carry no extra meaning. Without this step two
    /// spellings of one texture would produce two capability queries and two
    /// fingerprints for one object, which is the descriptor-canonicalization
    /// requirement of rhi-design section 48.1.
    ///
    /// A view format equal to the base format is *not* dropped here: it is a
    /// contradiction rather than a duplicate, so it stays for
    /// [`Self::normalize_and_validate`] to reject.
    pub(crate) fn canonicalized(&self) -> Self {
        let mut canonical = self.clone();
        canonical.view_formats.sort_unstable();
        canonical.view_formats.dedup();
        canonical
    }

    /// Normalizes and validates the portable creation invariants.
    ///
    /// This is the single lowering shared by the capability query, creation
    /// validation, and backend creation, so those three never use three
    /// different sets of conditions.
    ///
    /// It requires the canonical form: an un-canonicalized `view_formats` list
    /// is not an identity for this texture, so a backend must be handed
    /// [`Self::canonicalized`] rather than the descriptor as the caller built
    /// it. Every other field here is already order-free.
    pub(crate) fn normalize_and_validate(&self) -> RhiResult<()> {
        if self.usage.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture usage must be non-empty",
            ));
        }
        if !self.extent.is_non_zero() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture extent components must be non-zero",
            ));
        }
        if self.mip_levels == 0 || self.array_layers == 0 || self.sample_count == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture mip levels, array layers, and sample count must be non-zero",
            ));
        }
        match self.dimension {
            TextureDimension::D1 => {
                if self.extent.height != 1
                    || self.extent.depth != 1
                    || self.array_layers != 1
                    || self.sample_count != 1
                {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a 1D texture must have height 1, depth 1, one layer, and one sample",
                    ));
                }
            }
            TextureDimension::D2 => {
                if self.extent.depth != 1 {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a 2D texture must have depth 1",
                    ));
                }
            }
            TextureDimension::D3 => {
                if self.array_layers != 1 || self.sample_count != 1 {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a 3D texture must have one layer and one sample",
                    ));
                }
            }
        }
        if self.sample_count > 1
            && (self.dimension != TextureDimension::D2 || self.mip_levels != 1)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a multisampled texture must be 2D with exactly one mip level",
            ));
        }
        if !self.sample_count.is_power_of_two() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture sample count must be a power of two",
            ));
        }
        let max_mips = 32 - self.extent.max_dimension().leading_zeros() + 1;
        if self.mip_levels > max_mips {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture mip level count exceeds the base extent",
            ));
        }
        if self.view_formats.contains(&self.format) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view formats must not repeat the base format",
            ));
        }
        if self.view_compatibility.contains(TextureViewCompatibility::CUBE)
            && (self.dimension != TextureDimension::D2
                || self.extent.width != self.extent.height
                || self.array_layers < 6
                || self.sample_count != 1)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cube-compatible textures must be square 2D with at least 6 layers and one sample",
            ));
        }
        let aspects = format_facts(self.format).aspects();
        let attachment_bits = TextureUsage::COLOR_ATTACHMENT.union(
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        );
        if self.usage.contains(attachment_bits) && self.sample_count == 1 {
            // Legal: single-sample attachments are the common case.
        }
        if self
            .usage
            .contains(TextureUsage::COLOR_ATTACHMENT)
            && !aspects.contains(TextureAspects::COLOR)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a depth/stencil format cannot be a color attachment",
            ));
        }
        if self
            .usage
            .contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
            && aspects.contains(TextureAspects::COLOR)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a color format cannot be a depth/stencil attachment",
            ));
        }
        Ok(())
    }
}

/// A GPU texture.
#[derive(Clone)]
pub struct Texture {
    inner: Arc<dyn TextureBackend>,
}

impl core::fmt::Debug for Texture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Texture")
            .field("id", &self.id())
            .field("format", &self.descriptor().format)
            .finish_non_exhaustive()
    }
}

impl Texture {
    pub(crate) fn new(inner: Arc<dyn TextureBackend>) -> Self {
        Self { inner }
    }

    /// The backing this handle retains, for the device's retirement registry.
    pub(crate) fn backing(&self) -> Arc<dyn TextureBackend> {
        Arc::clone(&self.inner)
    }

    /// This texture's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this texture belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The descriptor this texture was created from.
    pub fn descriptor(&self) -> &TextureDescriptor {
        self.inner.descriptor()
    }
}

/// The backend half of a [`Texture`].
pub(crate) trait TextureBackend: Send + Sync + 'static {
    /// This texture's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this texture belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The descriptor this texture was created from.
    fn descriptor(&self) -> &TextureDescriptor;
}

/// A texture view description.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureViewDescriptor {
    /// A diagnostic label.
    pub label: Label,

    /// The presented dimension.
    pub dimension: TextureViewDimension,

    /// The view format, or `None` to use the base texture format.
    pub format: Option<TextureFormat>,

    /// The aspects the view exposes.
    pub aspects: TextureAspects,

    /// The first mip level.
    pub base_mip: u32,
    /// How many mip levels.
    pub mip_count: u32,

    /// The first array layer.
    pub base_layer: u32,
    /// How many array layers.
    pub layer_count: u32,
}

impl TextureViewDescriptor {
    /// A view over an explicit subresource range.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dimension: TextureViewDimension,
        aspects: TextureAspects,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
        layer_count: u32,
    ) -> Self {
        Self {
            label: Label::none(),
            dimension,
            format: None,
            aspects,
            base_mip,
            mip_count,
            base_layer,
            layer_count,
        }
    }

    /// A view covering a texture's complete logical subresource range.
    ///
    /// Cube and cube-array compatibility is still validated; this is a
    /// convenience over the general constructor, not a security bypass.
    pub fn whole(texture: &Texture, dimension: TextureViewDimension) -> RhiResult<Self> {
        let descriptor = texture.descriptor();
        let aspects = format_facts(descriptor.format).aspects();
        if aspects.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture format declares no aspect",
            ));
        }
        Ok(Self {
            label: Label::none(),
            dimension,
            format: None,
            aspects,
            base_mip: 0,
            mip_count: descriptor.mip_levels,
            base_layer: 0,
            layer_count: descriptor.array_layers,
        })
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }

    /// Sets the view format.
    pub fn with_format(mut self, format: TextureFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// The subresource range this view covers.
    pub fn subresource(&self) -> TextureSubresourceRange {
        TextureSubresourceRange {
            aspects: self.aspects,
            base_mip: self.base_mip,
            mip_count: self.mip_count,
            base_layer: self.base_layer,
            layer_count: self.layer_count,
        }
    }

    /// The actual view format, resolving `None` to the texture's base format.
    ///
    /// A view's identity is its resolved format, not the absence of one: two
    /// views that both mean `Rgba8Unorm` are the same view whether or not the
    /// caller spelled the format out.
    pub fn resolved_format(&self, texture: &Texture) -> TextureFormat {
        self.format.unwrap_or(texture.descriptor().format)
    }

    /// Validates this view against the texture it views.
    ///
    /// This is the single lowering shared by the capability query, creation
    /// validation, and backend creation. Every rejection here is portable: a
    /// view that fails is invalid on every backend, so a caller is never told
    /// its request was fine and then handed a driver error.
    pub(crate) fn validate_for(&self, texture: &Texture) -> RhiResult<()> {
        let descriptor = texture.descriptor();
        if self.mip_count == 0 || self.layer_count == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view mip and layer counts must be non-zero",
            ));
        }
        if self.aspects.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view must select at least one aspect",
            ));
        }
        let base_aspects = format_facts(descriptor.format).aspects();
        if !base_aspects.contains(self.aspects) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view selects an aspect the base format does not have",
            ));
        }
        let mip_end = self.base_mip.checked_add(self.mip_count);
        if mip_end.is_none_or(|end| end > descriptor.mip_levels) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view mip range is beyond the texture",
            ));
        }
        // A 3D texture's array fields are fixed by the descriptor, so a view is
        // never allowed to reinterpret its Z slices as array layers.
        let layer_end = self.base_layer.checked_add(self.layer_count);
        if layer_end.is_none_or(|end| end > descriptor.array_layers) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view layer range is beyond the texture",
            ));
        }

        let format = self.resolved_format(texture);
        if format != descriptor.format && !descriptor.view_formats.contains(&format) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view format was not declared in the texture's view formats",
            ));
        }
        if !format_facts(format).aspects().contains(self.aspects) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view format does not provide the selected aspect",
            ));
        }

        self.validate_dimension(descriptor, format)
    }

    /// Checks the declared dimension against the texture's shape.
    ///
    /// Cube and cube-array views are the reason this is not a plain dimension
    /// comparison: they are a D2 texture plus a view intent, and the layer count
    /// is what makes the intent honest.
    fn validate_dimension(
        &self,
        descriptor: &TextureDescriptor,
        format: TextureFormat,
    ) -> RhiResult<()> {
        let cube_compatible = descriptor
            .view_compatibility
            .contains(TextureViewCompatibility::CUBE);
        let requested = match self.dimension {
            TextureViewDimension::D1 => TextureViewDimension::D1,
            TextureViewDimension::D2 | TextureViewDimension::D2Array => {
                TextureViewDimension::D2
            }
            TextureViewDimension::D3 => TextureViewDimension::D3,
            TextureViewDimension::Cube | TextureViewDimension::CubeArray => {
                TextureViewDimension::D2
            }
        };
        if requested != dimension_of(descriptor.dimension) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view dimension is incompatible with the texture's dimension",
            ));
        }
        // An array view needs more than one layer to have anything to index.
        // A single-layer texture viewed as an array is not a degenerate case
        // any backend agrees on, so it is refused rather than reinterpreted.
        if self.dimension == TextureViewDimension::D2Array && self.layer_count < 2 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "an array view requires at least two array layers",
            ));
        }
        match self.dimension {
            TextureViewDimension::Cube => {
                if !cube_compatible || self.layer_count != 6 {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a cube view requires a cube-compatible texture and exactly 6 layers",
                    ));
                }
            }
            TextureViewDimension::CubeArray => {
                if !cube_compatible || self.layer_count % 6 != 0 {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a cube-array view requires a cube-compatible texture and a layer count that is a multiple of 6",
                    ));
                }
            }
            TextureViewDimension::D3 => {}
            _ => {}
        }
        // A multisampled texture has exactly one mip level and one layer, so a
        // format reinterpretation is the only thing left that could differ.
        if descriptor.sample_count > 1 && format != descriptor.format {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a multisampled texture view cannot reinterpret the format",
            ));
        }
        Ok(())
    }
}

/// The view dimension a texture dimension implies.
///
/// The match is deliberately exhaustive over the known variants: a new texture
/// dimension must be given a view dimension here rather than silently falling
/// into a wildcard that answers for it.
fn dimension_of(dimension: TextureDimension) -> TextureViewDimension {
    match dimension {
        TextureDimension::D1 => TextureViewDimension::D1,
        TextureDimension::D2 => TextureViewDimension::D2,
        TextureDimension::D3 => TextureViewDimension::D3,
    }
}

/// A view onto a texture's subresource range.
#[derive(Clone)]
pub struct TextureView {
    inner: Arc<dyn TextureViewBackend>,
}

impl core::fmt::Debug for TextureView {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TextureView")
            .field("id", &self.id())
            .field("format", &self.format())
            .finish_non_exhaustive()
    }
}

impl TextureView {
    pub(crate) fn new(inner: Arc<dyn TextureViewBackend>) -> Self {
        Self { inner }
    }

    /// The backing this handle retains, for the device's retirement registry.
    pub(crate) fn backing(&self) -> Arc<dyn TextureViewBackend> {
        Arc::clone(&self.inner)
    }

    /// This view's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this view belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The texture this view is on.
    pub fn texture(&self) -> &Texture {
        self.inner.texture()
    }

    /// The descriptor this view was created from.
    pub fn descriptor(&self) -> &TextureViewDescriptor {
        self.inner.descriptor()
    }

    /// The actual view format, resolved from the base texture when the
    /// descriptor did not name one.
    pub fn format(&self) -> TextureFormat {
        self.descriptor()
            .format
            .unwrap_or(self.texture().descriptor().format)
    }

    /// The aspects this view exposes.
    pub fn aspects(&self) -> TextureAspects {
        self.descriptor().aspects
    }

    /// The logical texel extent of the base mip. Array layers do not contribute
    /// to depth.
    pub fn extent(&self) -> Extent3d {
        let descriptor = self.texture().descriptor();
        descriptor
            .extent
            .mip_extent(self.descriptor().base_mip)
    }


    /// The sample count of the underlying texture.
    pub fn sample_count(&self) -> u32 {
        self.texture().descriptor().sample_count
    }

    /// How many array layers this view exposes.
    pub fn layer_count(&self) -> u32 {
        self.descriptor().layer_count
    }
}

/// The backend half of a [`TextureView`].
pub(crate) trait TextureViewBackend: Send + Sync + 'static {
    /// This view's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this view belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The texture this view is on.
    fn texture(&self) -> &Texture;

    /// The descriptor this view was created from.
    fn descriptor(&self) -> &TextureViewDescriptor;
}

/// How out-of-range sampling coordinates are resolved.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AddressMode {
    /// Clamp to the edge texel.
    ClampToEdge,
    /// Repeat the coordinate.
    Repeat,
    /// Mirror and repeat the coordinate.
    MirrorRepeat,
}

/// How texels are selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FilterMode {
    /// Nearest texel selection.
    Nearest,
    /// Linear interpolation.
    Linear,
}

/// A comparison function for a comparison sampler.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompareFunction {
    /// Never passes.
    Never,
    /// Passes when the new value is less.
    Less,
    /// Passes when the values are equal.
    Equal,
    /// Passes when the new value is less or equal.
    LessEqual,
    /// Passes when the new value is greater.
    Greater,
    /// Passes when the values differ.
    NotEqual,
    /// Passes when the new value is greater or equal.
    GreaterEqual,
    /// Always passes.
    Always,
}

/// A sampler description.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerDescriptor {
    /// A diagnostic label.
    pub label: Label,

    /// The U axis address mode.
    pub address_u: AddressMode,
    /// The V axis address mode.
    pub address_v: AddressMode,
    /// The W axis address mode.
    pub address_w: AddressMode,

    /// The magnification filter.
    pub mag_filter: FilterMode,
    /// The minification filter.
    pub min_filter: FilterMode,
    /// The mip filter.
    pub mip_filter: FilterMode,

    /// The minimum level of detail.
    pub lod_min: f32,
    /// The maximum level of detail.
    pub lod_max: f32,

    /// The comparison function, for a comparison sampler.
    pub compare: Option<CompareFunction>,

    /// The maximum anisotropy. One disables anisotropy.
    pub max_anisotropy: u16,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplerDescriptor {
    /// The portable default sampler.
    pub fn new() -> Self {
        Self {
            label: Label::none(),
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod_min: 0.0,
            lod_max: 32.0,
            compare: None,
            max_anisotropy: 1,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }

    /// Sets all three address modes.
    pub fn with_address_modes(mut self, u: AddressMode, v: AddressMode, w: AddressMode) -> Self {
        self.address_u = u;
        self.address_v = v;
        self.address_w = w;
        self
    }

    /// Sets all three filters.
    pub fn with_filters(mut self, mag: FilterMode, min: FilterMode, mip: FilterMode) -> Self {
        self.mag_filter = mag;
        self.min_filter = min;
        self.mip_filter = mip;
        self
    }

    /// Sets the level-of-detail clamp.
    pub fn with_lod_clamp(mut self, min: f32, max: f32) -> Self {
        self.lod_min = min;
        self.lod_max = max;
        self
    }

    /// Sets the comparison function.
    pub fn with_compare(mut self, compare: CompareFunction) -> Self {
        self.compare = Some(compare);
        self
    }

    /// Sets the maximum anisotropy.
    pub fn with_max_anisotropy(mut self, value: u16) -> Self {
        self.max_anisotropy = value;
        self
    }

    /// Whether this descriptor is internally legal.
    pub fn is_well_formed(&self) -> bool {
        self.max_anisotropy >= 1
            && self.lod_min.is_finite()
            && self.lod_max.is_finite()
            && self.lod_min <= self.lod_max
    }

    /// Validates the portable creation invariants.
    ///
    /// This is the single lowering shared by the capability query, creation
    /// validation, and backend creation. `is_well_formed` answers the same
    /// question, but the message here is what a caller reads, so the two are
    /// kept apart rather than one being a boolean spelling of the other.
    pub(crate) fn normalize_and_validate(&self) -> RhiResult<()> {
        if self.max_anisotropy == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "sampler max anisotropy must be at least one",
            ));
        }
        if !self.lod_min.is_finite() || !self.lod_max.is_finite() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "sampler level-of-detail bounds must be finite",
            ));
        }
        if self.lod_min > self.lod_max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "sampler level-of-detail minimum is above its maximum",
            ));
        }
        Ok(())
    }

    /// The binding contract this sampler satisfies.
    ///
    /// A comparison sampler requires a comparison function; an unfiltered
    /// nearest sampler without anisotropy is non-filtering; everything else is a
    /// filtering sampler.
    pub fn binding_kind(&self) -> super::binding::SamplerKind {
        if self.compare.is_some() {
            super::binding::SamplerKind::Comparison
        } else if self.mag_filter == FilterMode::Nearest
            && self.min_filter == FilterMode::Nearest
            && self.mip_filter == FilterMode::Nearest
            && self.max_anisotropy == 1
        {
            super::binding::SamplerKind::NonFiltering
        } else {
            super::binding::SamplerKind::Filtering
        }
    }
}

/// A GPU sampler.
#[derive(Clone)]
pub struct Sampler {
    inner: Arc<dyn SamplerBackend>,
}

impl core::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sampler")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

impl Sampler {
    pub(crate) fn new(inner: Arc<dyn SamplerBackend>) -> Self {
        Self { inner }
    }

    /// The backing this handle retains, for the device's retirement registry.
    pub(crate) fn backing(&self) -> Arc<dyn SamplerBackend> {
        Arc::clone(&self.inner)
    }

    /// This sampler's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this sampler belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The descriptor this sampler was created from.
    pub fn descriptor(&self) -> &SamplerDescriptor {
        self.inner.descriptor()
    }
}

/// The backend half of a [`Sampler`].
pub(crate) trait SamplerBackend: Send + Sync + 'static {
    /// This sampler's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this sampler belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The descriptor this sampler was created from.
    fn descriptor(&self) -> &SamplerDescriptor;
}

/// A host-to-buffer upload.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BufferUploadDescriptor {
    /// A diagnostic label.
    pub label: Label,

    /// The destination buffer.
    pub dst: Buffer,
    /// The destination byte offset.
    pub dst_offset: u64,

    /// The retained, immutable source bytes.
    pub bytes: Arc<[u8]>,
}

/// A host-to-texture upload.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TextureUploadDescriptor {
    /// A diagnostic label.
    pub label: Label,

    /// The destination texture.
    pub dst: Texture,

    /// The destination subresource.
    pub subresource: TextureSubresourceLayers,
    /// The destination origin.
    pub origin: Origin3d,
    /// The copied extent.
    pub extent: Extent3d,

    /// The CPU source layout, not the native GPU copy layout.
    pub source_layout: HostTexelLayout,

    /// The retained, immutable source bytes.
    pub bytes: Arc<[u8]>,
}

impl BufferUploadDescriptor {
    /// Validates this upload against its destination buffer.
    ///
    /// The payload is retained by the job, so the check is that the retained
    /// bytes actually cover the range the job will write. An empty payload over
    /// an empty range is refused: it would be a GPU-visible mutation that
    /// mutates nothing, which a caller writing it almost always reached by
    /// mistake.
    pub(crate) fn validate(&self, device: DeviceIdentity) -> RhiResult<()> {
        if self.dst.device_identity() != device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "buffer upload destination belongs to another device identity",
            )
            .on(self.dst.id()));
        }
        if self.bytes.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "buffer upload payload must be non-empty",
            ));
        }
        let end = self
            .dst_offset
            .checked_add(self.bytes.len() as u64)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "buffer upload range overflows the address space",
                )
            })?;
        if end > self.dst.descriptor().size {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "buffer upload range is beyond the destination buffer",
            )
            .on(self.dst.id()));
        }
        Ok(())
    }
}

impl TextureUploadDescriptor {
    /// Validates this upload against its destination texture.
    ///
    /// Everything here is also checked for a copy, but it is checked again at
    /// creation because a job is retained and encoded later: a job that was
    /// accepted is a promise, and the promise has to hold before the caller has
    /// a chance to change anything.
    pub(crate) fn validate(&self, device: DeviceIdentity) -> RhiResult<()> {
        if self.dst.device_identity() != device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "texture upload destination belongs to another device identity",
            )
            .on(self.dst.id()));
        }
        let descriptor = self.dst.descriptor();
        let aspects = format_facts(descriptor.format).aspects();
        self.subresource.validate_for(
            descriptor.dimension,
            descriptor.mip_levels,
            descriptor.array_layers,
            aspects,
        )?;
        let region = Extent3d::d3(self.extent.width, self.extent.height, self.extent.depth);
        validate_region_within(descriptor, self.subresource.mip_level, self.origin, region)?;
        validate_host_layout(
            descriptor.format,
            region,
            self.source_layout,
            self.bytes.len(),
        )
    }
}

/// Checks that a copied region lies inside a texture's mip level.
pub(crate) fn validate_region_within(
    descriptor: &TextureDescriptor,
    mip_level: u32,
    origin: Origin3d,
    extent: Extent3d,
) -> RhiResult<()> {
    if !extent.is_non_zero() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "copy extent components must be non-zero",
        ));
    }
    let level = descriptor.extent.mip_extent(mip_level);
    let end_x = u64::from(origin.x) + u64::from(extent.width);
    let end_y = u64::from(origin.y) + u64::from(extent.height);
    let end_z = u64::from(origin.z) + u64::from(extent.depth);
    if end_x > u64::from(level.width)
        || end_y > u64::from(level.height)
        || end_z > u64::from(level.depth)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "copy region is beyond the destination subresource",
        ));
    }
    Ok(())
}

/// Either upload form.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum UploadDescriptor {
    /// A buffer upload.
    Buffer(BufferUploadDescriptor),
    /// A texture upload.
    Texture(TextureUploadDescriptor),
}

impl UploadDescriptor {
    /// The diagnostic label.
    pub fn label(&self) -> &Label {
        match self {
            Self::Buffer(descriptor) => &descriptor.label,
            Self::Texture(descriptor) => &descriptor.label,
        }
    }

    /// The retained source bytes.
    pub fn bytes(&self) -> &Arc<[u8]> {
        match self {
            Self::Buffer(descriptor) => &descriptor.bytes,
            Self::Texture(descriptor) => &descriptor.bytes,
        }
    }
}

/// A retained host-to-GPU mutation, encodable any number of times.
#[derive(Clone)]
pub struct UploadJob {
    inner: Arc<dyn UploadJobBackend>,
}

impl core::fmt::Debug for UploadJob {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UploadJob")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

impl UploadJob {
    pub(crate) fn new(inner: Arc<dyn UploadJobBackend>) -> Self {
        Self { inner }
    }

    /// This job's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this job belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The complete portable mutation descriptor, for capture and tooling.
    pub fn descriptor(&self) -> &UploadDescriptor {
        self.inner.descriptor()
    }
}

/// The backend half of an [`UploadJob`].
pub(crate) trait UploadJobBackend: Send + Sync + 'static {
    /// This job's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this job belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The complete portable mutation descriptor.
    fn descriptor(&self) -> &UploadDescriptor;
}

/// A readback request.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ReadbackRequest {
    /// Read a buffer range.
    Buffer {
        /// A diagnostic label.
        label: Label,
        /// The source buffer.
        src: Buffer,
        /// The source range.
        range: BufferRange,
    },

    /// Read a texture region.
    Texture {
        /// A diagnostic label.
        label: Label,

        /// The source texture.
        src: Texture,

        /// The source subresource.
        subresource: TextureSubresourceLayers,
        /// The source origin.
        origin: Origin3d,
        /// The read extent.
        extent: Extent3d,
    },
}

impl ReadbackRequest {
    /// The diagnostic label.
    pub fn label(&self) -> &Label {
        match self {
            Self::Buffer { label, .. } | Self::Texture { label, .. } => label,
        }
    }
}

/// The lifecycle of a readback ticket.
///
/// A ticket is never permanently stuck in `NotSubmitted` or `Pending`: dropping
/// the unsubmitted work moves it to `Abandoned`, and device loss moves it to
/// `DeviceLost`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReadbackStatus {
    /// Encoded into recorded work that has not been successfully submitted.
    NotSubmitted,

    /// Successfully submitted and awaiting terminal GPU completion.
    Pending,

    /// The CPU data is readable.
    Ready,

    /// The corresponding recorded work or submission plan was discarded before a
    /// successful submit.
    Abandoned,

    /// The device identity was lost.
    DeviceLost,

    /// The backend reported a terminal failure.
    Failed,
}

impl ReadbackStatus {
    /// Whether this status is terminal.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Ready | Self::Abandoned | Self::DeviceLost | Self::Failed
        )
    }
}

/// The byte layout of readback data.
///
/// Readback does not promise tightly packed texel bytes: a D3D12 copy footprint
/// distinguishes an unpadded row size from an aligned row pitch, backend staging
/// layouts differ, and forcing RHI to repack would add a cost the caller did not
/// ask for. A capture layer that needs a canonical packed blob repacks it there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReadbackTexelLayout {
    /// The byte distance between the starts of adjacent valid rows.
    pub bytes_per_row: u32,
    /// The number of rows between the starts of adjacent images, layers, or
    /// depth slices.
    pub rows_per_image: u32,
    /// The total length of the returned byte slice.
    pub total_size: u64,
}

/// The data a ready readback ticket exposes.
#[non_exhaustive]
#[derive(Debug)]
pub enum ReadbackData<'a> {
    /// Buffer bytes.
    Buffer {
        /// The read bytes.
        bytes: &'a [u8],
    },

    /// Texture bytes with their explicit layout.
    Texture {
        /// The read bytes.
        bytes: &'a [u8],
        /// The layout of those bytes.
        layout: ReadbackTexelLayout,
    },
}

/// A scoped, device-bound handle to an encoded readback.
#[derive(Clone)]
pub struct ReadbackTicket {
    inner: Arc<dyn ReadbackTicketBackend>,
}

impl core::fmt::Debug for ReadbackTicket {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReadbackTicket")
            .field("id", &self.id())
            .field("status", &self.status())
            .finish_non_exhaustive()
    }
}

impl ReadbackTicket {
    pub(crate) fn new(inner: Arc<dyn ReadbackTicketBackend>) -> Self {
        Self { inner }
    }

    /// This ticket's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this ticket belongs to.
    ///
    /// The device need not be passed again: the ticket already binds its own
    /// identity and internal state.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The original portable request.
    pub fn request(&self) -> &ReadbackRequest {
        self.inner.request()
    }

    /// This ticket's current lifecycle status.
    pub fn status(&self) -> ReadbackStatus {
        self.inner.status()
    }

    /// The terminal completion this ticket is associated with, once it has been
    /// successfully submitted.
    pub fn completion(&self) -> Option<CompletionPoint> {
        self.inner.completion()
    }

    /// The CPU-visible data, when the ticket is ready.
    ///
    /// The ticket does not spin or wait on the GPU: the host advances its own
    /// loop and calls [`Device::poll`].
    pub fn try_read(&self) -> RhiResult<Option<ReadbackData<'_>>> {
        self.inner.try_read()
    }
}

/// The backend half of a [`ReadbackTicket`].
pub(crate) trait ReadbackTicketBackend: Send + Sync + 'static {
    /// This ticket's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this ticket belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The original portable request.
    fn request(&self) -> &ReadbackRequest;

    /// This ticket's current lifecycle status.
    fn status(&self) -> ReadbackStatus;

    /// The associated terminal completion, once submitted.
    fn completion(&self) -> Option<CompletionPoint>;

    /// The CPU-visible data, when ready.
    fn try_read(&self) -> RhiResult<Option<ReadbackData<'_>>>;
}

/// The logical row footprint of one row of `format`, in bytes.
pub(crate) fn logical_row_bytes(format: TextureFormat, width: u32) -> Option<u64> {
    let facts = format_facts(format);
    let bytes = facts.logical_bytes_per_block()?;
    let blocks = width.div_ceil(facts.block_width());
    Some(u64::from(bytes) * u64::from(blocks))
}

/// Validates that a host layout covers the bytes of one copied region.
pub(crate) fn validate_host_layout(
    format: TextureFormat,
    extent: Extent3d,
    layout: HostTexelLayout,
    available: usize,
) -> RhiResult<()> {
    let facts = format_facts(format);
    let bytes_per_block = facts.logical_bytes_per_block().ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "this format has no portable host texel layout",
        )
    })?;
    let block_width = facts.block_width();
    let block_height = facts.block_height();
    if layout.bytes_per_row == 0 || layout.rows_per_image == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "host texel layout row and image strides must be non-zero",
        ));
    }
    let row_bytes = u64::from(bytes_per_block) * u64::from(extent.width.div_ceil(block_width));
    let block_rows = u64::from(extent.height.div_ceil(block_height));
    let image_rows = u64::from(layout.rows_per_image);
    let row_stride = u64::from(layout.bytes_per_row);
    if row_stride < row_bytes {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "host texel layout row stride is smaller than one logical row",
        ));
    }
    if row_stride % u64::from(bytes_per_block) != 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "host texel layout row stride is not aligned to the format block size",
        ));
    }
    if image_rows < block_rows {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "host texel layout image stride is smaller than one logical image",
        ));
    }
    let required = match extent.depth {
        1 => row_stride * (block_rows - 1) + row_bytes,
        depth => {
            let slices = u64::from(depth);
            (slices - 1) * image_rows * row_stride + row_stride * (block_rows - 1) + row_bytes
        }
    };
    if required > available as u64 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "host texel layout does not cover the requested region",
        ));
    }
    Ok(())
}

/// Validates a copy offset and size against `BufferCopyLayoutLimits`.
pub(crate) fn validate_buffer_copy_layout(
    limits: BufferCopyLayoutLimits,
    offset: u64,
    size: u64,
) -> RhiResult<()> {
    if limits.offset_alignment() != 0 && offset % limits.offset_alignment() != 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "buffer copy offset does not satisfy the route alignment",
        ));
    }
    if limits.size_alignment() != 0 && size % limits.size_alignment() != 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "buffer copy size does not satisfy the route alignment",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
