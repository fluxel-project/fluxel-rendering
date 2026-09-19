//! The binding model: slots, kinds, capability queries, and bind groups.
//!
//! This module owns rhi-design sections 20 to 22 and the compatibility tokens
//! from section 21.1.
//!
//! # What it is
//!
//! A [`BindGroup`] is an immutable resource packet validated against a logical
//! layout. It promises no native descriptor object: `VkDescriptorSet`, a D3D12
//! descriptor table, a Metal argument buffer, and a GL texture-unit packet are
//! all backend lowerings of the same logical packet.
//!
//! Binding legality is a query, not a pile of booleans. "The device supports
//! textures" does not imply that a storage texture with a cube view, a
//! read-write storage texture, a storage buffer visible to the vertex stage, a
//! fixed-length resource array, or a dynamic buffer offset is legal; each is a
//! separate [`BindingSupportQuery`] that a backend may refuse independently.
//!
//! # What it deliberately does not own
//!
//! Aggregate limits that span several groups, such as per-stage resource counts
//! and dynamic buffers per pipeline layout, cannot be decided at the individual
//! bind-group-layout stage. They are validated once, when a pipeline interface
//! is created.
//!
//! A compatibility id is a device-scoped interning token, never a correctness
//! proof: two objects still have to pass device-identity validation before they
//! may be used together, and an equal [`LayoutFingerprint`] never replaces full
//! canonical semantic comparison.

use std::sync::Arc;

use super::platform::{DeviceIdentity, Label, ObjectId, RhiError, RhiErrorKind, RhiResult};
use super::format::TextureFormat;
use super::resource::{BufferBinding, Sampler, TextureView, TextureViewDimension};
use super::shader::ShaderStages;

/// A Fluxel logical bind group index. Not a native set, space, or table index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindGroupIndex(u32);

impl BindGroupIndex {
    /// A group index.
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// The index.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A Fluxel logical slot within a bind group. Not a native register or binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingSlotId(u32);

impl BindingSlotId {
    /// A slot id.
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// The id.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// How many resources one logical binding holds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingCount {
    /// Exactly one resource.
    One,

    /// A fixed-length resource binding array. The value is at least 2.
    ///
    /// This is capability-gated vocabulary and does not imply runtime-sized,
    /// partially bound, update-after-bind, or arbitrary non-uniform descriptor
    /// indexing. A WebGPU-core backend may correctly answer
    /// [`BindingSupport::Unsupported`].
    Fixed(u32),
}

impl BindingCount {
    /// The number of elements this binding holds.
    pub fn elements(self) -> u32 {
        match self {
            Self::One => 1,
            Self::Fixed(count) => count,
        }
    }

    /// Whether the count is legal on its own.
    pub fn is_well_formed(self) -> bool {
        match self {
            Self::One => true,
            Self::Fixed(count) => count >= 2,
        }
    }
}

/// How a sampled texture binding may be filtered.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureSampleType {
    /// A filtering sampler is legal.
    Float,
    /// Only a non-filtering sampler is legal.
    UnfilterableFloat,
    /// A signed integer texture.
    Sint,
    /// An unsigned integer texture.
    Uint,
    /// A depth texture.
    Depth,
}

/// How a shader accesses a storage texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageAccess {
    /// Read-only storage access.
    ReadOnly,
    /// Write-only storage access.
    WriteOnly,
    /// Read-write storage access.
    ReadWrite,
}

/// The contract a sampler binding requires.
///
/// This describes the binding contract, not whether a particular sampler
/// descriptor currently uses linear filtering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SamplerKind {
    /// Filtering is used.
    Filtering,
    /// Only nearest sampling is used.
    NonFiltering,
    /// A comparison sampler.
    Comparison,
}

/// How a shader accesses a storage buffer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferBindingAccess {
    /// Read-only.
    ReadOnly,
    /// Read and write.
    ReadWrite,
}

/// The portable semantics of one binding.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BindingKind {
    /// A uniform buffer with a minimum visible byte range.
    UniformBuffer {
        /// The minimum visible byte range required by the shader or layout.
        /// Always greater than zero.
        min_size: u64,
    },

    /// A storage buffer with an access mode and minimum visible range.
    StorageBuffer {
        /// The access mode the shader uses.
        access: BufferBindingAccess,
        /// The minimum visible byte range. Always greater than zero.
        min_size: u64,
    },

    /// A sampled texture.
    SampledTexture {
        /// The required view dimension.
        dimension: TextureViewDimension,
        /// The sample type the shader uses.
        sample_type: TextureSampleType,
        /// Whether the view must be multisampled.
        multisampled: bool,
    },

    /// A storage texture.
    StorageTexture {
        /// The required view dimension.
        dimension: TextureViewDimension,
        /// The required view format.
        format: TextureFormat,
        /// The access mode the shader uses.
        access: StorageAccess,
    },

    /// A sampler.
    Sampler {
        /// The sampler contract.
        kind: SamplerKind,
    },
}

impl BindingKind {
    /// Whether this kind requires a buffer binding.
    pub fn is_buffer(&self) -> bool {
        matches!(self, Self::UniformBuffer { .. } | Self::StorageBuffer { .. })
    }

    /// The declared minimum visible byte range, when this is a buffer binding.
    pub fn min_size(&self) -> Option<u64> {
        match self {
            Self::UniformBuffer { min_size } | Self::StorageBuffer { min_size, .. } => {
                Some(*min_size)
            }
            _ => None,
        }
    }

    /// Whether the declared minimum sizes are legal.
    pub fn is_well_formed(&self) -> bool {
        match self {
            Self::UniformBuffer { min_size } | Self::StorageBuffer { min_size, .. } => *min_size > 0,
            _ => true,
        }
    }
}

/// A question about whether one binding shape is legal.
///
/// Total ordering exists so a capability database can key entries canonically
/// instead of depending on the order a backend happened to declare them.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BindingSupportQuery {
    /// Which stages may see the binding. Must be non-empty.
    pub visibility: ShaderStages,
    /// The binding semantics.
    pub kind: BindingKind,
    /// How many resources the binding holds.
    pub count: BindingCount,

    /// Valid only for uniform and storage buffers.
    pub dynamic_offset: bool,
}

impl BindingSupportQuery {
    /// A scalar, non-dynamic query.
    pub fn new(visibility: ShaderStages, kind: BindingKind) -> Self {
        Self {
            visibility,
            kind,
            count: BindingCount::One,
            dynamic_offset: false,
        }
    }

    /// Sets the resource count.
    pub fn with_count(mut self, count: BindingCount) -> Self {
        self.count = count;
        self
    }

    /// Requests a dynamic buffer offset.
    pub fn with_dynamic_offset(mut self, enabled: bool) -> Self {
        self.dynamic_offset = enabled;
        self
    }

    /// Whether this query is internally legal, independent of any device.
    pub fn is_well_formed(&self) -> bool {
        !self.visibility.is_empty()
            && self.kind.is_well_formed()
            && self.count.is_well_formed()
            && (!self.dynamic_offset || self.kind.is_buffer())
    }
}

/// The device's verdict on one binding shape.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingSupport {
    /// The shape is not usable.
    Unsupported,
    /// The shape is usable.
    Supported,
}

impl BindingSupport {
    /// Whether the shape is usable.
    pub fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }
}

/// The per-stage binding class a count limit applies to.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingLimitClass {
    /// Uniform buffers.
    UniformBuffers,
    /// Storage buffers.
    StorageBuffers,
    /// Sampled textures.
    SampledTextures,
    /// Storage textures.
    StorageTextures,
    /// Samplers.
    Samplers,
}

/// One entry of a bind group layout.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingSlot {
    /// The logical slot.
    pub slot: BindingSlotId,
    /// Which stages may see it.
    pub visibility: ShaderStages,
    /// The binding semantics.
    pub kind: BindingKind,
    /// How many resources it holds.
    pub count: BindingCount,

    /// Valid only for buffer bindings.
    pub dynamic_offset: bool,
}

impl BindingSlot {
    /// A scalar, non-dynamic slot.
    pub fn new(slot: BindingSlotId, visibility: ShaderStages, kind: BindingKind) -> Self {
        Self {
            slot,
            visibility,
            kind,
            count: BindingCount::One,
            dynamic_offset: false,
        }
    }

    /// Sets the resource count.
    pub fn with_count(mut self, count: BindingCount) -> Self {
        self.count = count;
        self
    }

    /// Requests a dynamic buffer offset.
    pub fn with_dynamic_offset(mut self, enabled: bool) -> Self {
        self.dynamic_offset = enabled;
        self
    }

    /// The capability query this slot implies.
    pub fn support_query(&self) -> BindingSupportQuery {
        BindingSupportQuery {
            visibility: self.visibility,
            kind: self.kind.clone(),
            count: self.count,
            dynamic_offset: self.dynamic_offset,
        }
    }
}

/// A device-scoped exact compatibility token for a bind group layout.
///
/// It is produced by the device interning canonical layout descriptors and
/// cannot be constructed by a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindGroupLayoutCompatibilityId(pub(crate) u64);

impl BindGroupLayoutCompatibilityId {
    /// The underlying value.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A device-scoped exact compatibility token for a pipeline interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineInterfaceCompatibilityId(pub(crate) u64);

impl PipelineInterfaceCompatibilityId {
    /// The underlying value.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A canonical descriptor fingerprint for caches, diagnostics, and capture
/// provenance.
///
/// Equal fingerprints cannot alone replace correctness validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayoutFingerprint(pub [u8; 32]);

/// A canonical bind group layout description.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindGroupLayoutDescriptor {
    /// A diagnostic label. It never participates in compatibility.
    pub label: Label,
    /// The entries, in ascending slot order once canonicalized.
    pub entries: Vec<BindingSlot>,
}

impl BindGroupLayoutDescriptor {
    /// A descriptor over `entries`.
    pub fn new(entries: Vec<BindingSlot>) -> Self {
        Self {
            label: Label::none(),
            entries,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }

    /// The entries sorted by ascending slot id.
    ///
    /// Canonicalization is by value, so a layout's identity cannot depend on the
    /// order the caller happened to declare its entries in.
    pub(crate) fn canonicalized(&self) -> Self {
        let mut entries = self.entries.clone();
        entries.sort_by_key(|entry| entry.slot);
        Self {
            label: self.label.clone(),
            entries,
        }
    }
}

/// A canonical, device-scoped bind group layout.
#[derive(Clone)]
pub struct BindGroupLayout {
    inner: Arc<dyn BindGroupLayoutBackend>,
}

impl core::fmt::Debug for BindGroupLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BindGroupLayout")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

impl BindGroupLayout {
    pub(crate) fn new(inner: Arc<dyn BindGroupLayoutBackend>) -> Self {
        Self { inner }
    }

    /// This layout's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this layout belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The canonicalized descriptor.
    pub fn descriptor(&self) -> &BindGroupLayoutDescriptor {
        self.inner.descriptor()
    }

    /// The device-scoped exact compatibility token.
    pub fn compatibility_id(&self) -> BindGroupLayoutCompatibilityId {
        self.inner.compatibility_id()
    }

    /// The canonical descriptor fingerprint.
    pub fn fingerprint(&self) -> LayoutFingerprint {
        self.inner.fingerprint()
    }

    /// The number of dynamic buffer elements in this layout.
    ///
    /// Their order is frozen as ascending slot id, then ascending element index
    /// within the same fixed-length binding. That one order is shared by raster
    /// and compute `set_bind_group`, the capture tooling IR, and backend
    /// lowering, so a dynamic offset list never means two different things.
    pub fn dynamic_offset_count(&self) -> u32 {
        self.inner.dynamic_offset_count()
    }
}

/// The backend half of a [`BindGroupLayout`].
pub(crate) trait BindGroupLayoutBackend: Send + Sync + 'static {
    /// This layout's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this layout belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The canonicalized descriptor.
    fn descriptor(&self) -> &BindGroupLayoutDescriptor;

    /// The device-scoped exact compatibility token.
    fn compatibility_id(&self) -> BindGroupLayoutCompatibilityId;

    /// The canonical descriptor fingerprint.
    fn fingerprint(&self) -> LayoutFingerprint;

    /// The number of dynamic buffer elements.
    fn dynamic_offset_count(&self) -> u32;
}

/// One resource a bind group binds.
#[non_exhaustive]
#[derive(Clone)]
pub enum BindingResource {
    /// A single buffer range.
    Buffer(BufferBinding),
    /// A single texture view.
    Texture(TextureView),
    /// A single sampler.
    Sampler(Sampler),

    /// A fixed-length buffer range array.
    BufferArray(Vec<BufferBinding>),
    /// A fixed-length texture view array.
    TextureArray(Vec<TextureView>),
    /// A fixed-length sampler array.
    SamplerArray(Vec<Sampler>),
}

impl BindingResource {
    /// The number of resources this value holds.
    pub fn len(&self) -> usize {
        match self {
            Self::Buffer(_) | Self::Texture(_) | Self::Sampler(_) => 1,
            Self::BufferArray(values) => values.len(),
            Self::TextureArray(values) => values.len(),
            Self::SamplerArray(values) => values.len(),
        }
    }

    /// Whether this value holds no resources.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this value is the array form.
    pub fn is_array(&self) -> bool {
        matches!(
            self,
            Self::BufferArray(_) | Self::TextureArray(_) | Self::SamplerArray(_)
        )
    }

    /// Whether this value's resource class matches `kind`.
    pub fn matches_kind(&self, kind: &BindingKind) -> bool {
        match (self, kind) {
            (Self::Buffer(_) | Self::BufferArray(_), BindingKind::UniformBuffer { .. })
            | (Self::Buffer(_) | Self::BufferArray(_), BindingKind::StorageBuffer { .. }) => true,
            (Self::Texture(_) | Self::TextureArray(_), BindingKind::SampledTexture { .. })
            | (Self::Texture(_) | Self::TextureArray(_), BindingKind::StorageTexture { .. }) => true,
            (Self::Sampler(_) | Self::SamplerArray(_), BindingKind::Sampler { .. }) => true,
            _ => false,
        }
    }
}

impl core::fmt::Debug for BindingResource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let class = match self {
            Self::Buffer(_) => "Buffer",
            Self::Texture(_) => "Texture",
            Self::Sampler(_) => "Sampler",
            Self::BufferArray(_) => "BufferArray",
            Self::TextureArray(_) => "TextureArray",
            Self::SamplerArray(_) => "SamplerArray",
        };
        write!(f, "{class}(len = {})", self.len())
    }
}

/// One slot's resource in a bind group descriptor.
#[derive(Clone, Debug)]
pub struct BindGroupEntry {
    /// The slot this entry fills.
    pub slot: BindingSlotId,
    /// The resource.
    pub resource: BindingResource,
}

impl BindGroupEntry {
    /// An entry binding `resource` to `slot`.
    pub fn new(slot: BindingSlotId, resource: BindingResource) -> Self {
        Self { slot, resource }
    }
}

/// An immutable logical resource packet.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BindGroupDescriptor {
    /// A diagnostic label.
    pub label: Label,
    /// The layout this packet is validated against.
    pub layout: BindGroupLayout,
    /// The entries, in ascending slot order once canonicalized.
    pub entries: Vec<BindGroupEntry>,
}

impl BindGroupDescriptor {
    /// A descriptor over `layout` with no entries yet.
    pub fn new(layout: BindGroupLayout) -> Self {
        Self {
            label: Label::none(),
            layout,
            entries: Vec::new(),
        }
    }

    /// Adds one entry.
    pub fn with_entry(mut self, entry: BindGroupEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Adds several entries.
    pub fn with_entries(mut self, entries: impl IntoIterator<Item = BindGroupEntry>) -> Self {
        self.entries.extend(entries);
        self
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }
}

/// An immutable, validated resource packet on a device.
#[derive(Clone)]
pub struct BindGroup {
    inner: Arc<dyn BindGroupBackend>,
}

impl core::fmt::Debug for BindGroup {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BindGroup")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

impl BindGroup {
    pub(crate) fn new(inner: Arc<dyn BindGroupBackend>) -> Self {
        Self { inner }
    }

    /// This bind group's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this packet belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The layout this packet was validated against.
    pub fn layout(&self) -> &BindGroupLayout {
        self.inner.layout()
    }

    /// The canonicalized descriptor.
    pub fn descriptor(&self) -> &BindGroupDescriptor {
        self.inner.descriptor()
    }
}

/// The backend half of a [`BindGroup`].
pub(crate) trait BindGroupBackend: Send + Sync + 'static {
    /// This bind group's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this packet belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The layout this packet was validated against.
    fn layout(&self) -> &BindGroupLayout;

    /// The canonicalized descriptor.
    fn descriptor(&self) -> &BindGroupDescriptor;
}

/// Validates that a layout's entries are individually legal.
///
/// Aggregate limits that span groups are deliberately not checked here; they
/// belong to pipeline interface creation.
pub(crate) fn validate_layout_entries(
    entries: &[BindingSlot],
    max_bindings_per_group: u64,
) -> RhiResult<()> {
    if entries.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "bind group layout must declare at least one entry",
        ));
    }
    if entries.len() as u64 > max_bindings_per_group {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "bind group layout declares more entries than MaxBindingsPerGroup",
        ));
    }
    for window in entries.windows(2) {
        if window[0].slot == window[1].slot {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("bind group layout declares slot {} twice", window[0].slot.get()),
            ));
        }
    }
    for entry in entries {
        if entry.slot.get() as u64 >= max_bindings_per_group {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "bind group layout slot {} is beyond MaxBindingsPerGroup",
                    entry.slot.get()
                ),
            ));
        }
        if entry.visibility.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("bind group layout slot {} has empty visibility", entry.slot.get()),
            ));
        }
        if !entry.kind.is_well_formed() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "bind group layout slot {} declares a zero minimum size",
                    entry.slot.get()
                ),
            ));
        }
        if entry.dynamic_offset && !entry.kind.is_buffer() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "bind group layout slot {} requests a dynamic offset on a non-buffer binding",
                    entry.slot.get()
                ),
            ));
        }
        if !entry.count.is_well_formed() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "bind group layout slot {} declares a fixed count below 2",
                    entry.slot.get()
                ),
            ));
        }
    }
    Ok(())
}
