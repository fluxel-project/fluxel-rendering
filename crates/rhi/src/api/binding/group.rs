//! Section 22: the bind group packet.
//!
//! The immutable resource packet and its validation: section 22.2's
//! canonicalization into ascending slot order, and section 22.3's per-resource
//! lists — buffer, sampled texture, storage texture, sampler. [`BindGroupLimits`]
//! lives here rather than being shared, because this file is its only consumer.
//!
//! Not owned here: the layout the packet is validated against (section 21,
//! `layout.rs`) and the device's answers. Whether a sampler and a sampled texture
//! are legal *together* is the shader interface's and the pipeline's question,
//! not this file's (section 22.3).

use core::fmt;
use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::external::ExternalTexture;
use crate::api::format::{TextureFormat, sample_type};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::platform::requirements::LimitKey;
use crate::api::resource::AccelerationStructure;
use crate::api::resource::buffer::{
    BufferBinding, BufferUsage, validate_buffer_ownership, validate_buffer_range,
};
use crate::api::resource::sampler::Sampler;
use crate::api::resource::subresource::TextureAspects;
use crate::api::resource::texture::TextureUsage;
use crate::api::resource::view::{TextureView, TextureViewDimension};

use crate::api::binding::backend::BindGroupBackend;

use super::layout::{BindGroupLayout, BindingSlot};
use super::vocabulary::{
    BindingCount, BindingKind, BindingSlotId, SamplerKind, StorageAccess, TextureSampleType,
};

/// The resource a bind group entry binds.
///
/// The scalar and array variants are separate rather than one `Vec` variant,
/// because section 22.1 makes the choice follow [`BindingCount`] exactly: `One`
/// takes a scalar, `Fixed(n)` takes the matching array, and an array of length 1
/// cannot stand in for a scalar. A single `Vec` variant would erase that rule from
/// the type and leave it to validation.
///
/// The element types are the resource module's own handles, held by clone: a bind
/// group is a logical owner of everything it binds, which is what keeps a bound
/// resource alive until the group is gone.
#[non_exhaustive]
#[derive(Clone)]
pub enum BindingResource {
    /// One buffer range.
    Buffer(BufferBinding),
    /// One texture view.
    Texture(TextureView),
    /// One sampler.
    Sampler(Sampler),
    /// One acceleration structure.
    AccelerationStructure(AccelerationStructure),
    /// One opaque external texture.
    ExternalTexture(ExternalTexture),

    /// A fixed-length array of buffer ranges.
    BufferArray(Vec<BufferBinding>),
    /// A fixed-length array of texture views.
    TextureArray(Vec<TextureView>),
    /// A fixed-length array of samplers.
    SamplerArray(Vec<Sampler>),
    /// A fixed-length array of acceleration structures.
    AccelerationStructureArray(Vec<AccelerationStructure>),
}

/// One slot of a bind group packet, paired with the resource that fills it.
#[derive(Clone)]
pub struct BindGroupEntry {
    /// The slot being filled.
    pub slot: BindingSlotId,
    /// The resource filling it.
    pub resource: BindingResource,
}

impl BindGroupEntry {
    /// Pairs a slot with a resource.
    pub fn new(slot: BindingSlotId, resource: BindingResource) -> Self {
        Self { slot, resource }
    }
}

/// Everything a caller states about a bind group before it exists.
///
/// Holds the [`BindGroupLayout`] by clone rather than by reference, because the
/// group is validated against that exact layout for its whole life: section 22.2
/// makes a P0 [`BindGroup`] immutable after creation, so the layout it was built
/// against must outlive the call.
#[non_exhaustive]
#[derive(Clone)]
pub struct BindGroupDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,

    /// The layout this packet is validated against.
    pub layout: BindGroupLayout,

    /// The resources. Canonicalized into ascending slot order on creation.
    pub entries: Vec<BindGroupEntry>,
}

impl BindGroupDescriptor {
    /// Describes an empty packet against a layout.
    pub fn new(layout: BindGroupLayout) -> Self {
        Self {
            label: Label::default(),
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

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// The entries in ascending [`BindingSlotId`] order.
    ///
    /// Section 22.2's canonicalization. Duplicate slots are refused by
    /// [`validate_bind_group_descriptor`] before this is called, so the sort has
    /// no tie to break and therefore no choice to make.
    pub(crate) fn canonicalized(&self) -> Self {
        let mut entries = self.entries.clone();
        entries.sort_by_key(|entry| entry.slot.get());
        Self {
            label: self.label.clone(),
            layout: self.layout.clone(),
            entries,
        }
    }
}

/// A created bind group: an immutable resource packet.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the [`DeviceIdentity`]
/// that created it. Section 22.2 makes it immutable after creation, so there is no
/// update verb here and no interior mutability: a change is a new group.
#[derive(Clone)]
pub struct BindGroup {
    inner: Arc<BindGroupInner>,
}

/// The one ownership domain of an immutable bind-group packet.
struct BindGroupInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: BindGroupDescriptor,
    /// The backend's own descriptor packet, in the same shape
    /// [`crate::api::resource::Buffer`] holds its allocation: it is directly
    /// owned by this handle's single shared inner domain, and behind `dyn` so
    /// that no native type reaches the exported surface (section 59).
    ///
    /// Section 22.2 makes a bind group a logical owner of everything it binds. The
    /// *logical* half of that is the descriptor above, which holds the resource
    /// handles; this field is where a backend keeps whatever its own API needs so
    /// that the addresses in a native descriptor stay valid for as long as the
    /// packet does — for Direct3D 12, a reference to each resource behind a GPU
    /// virtual address.
    ///
    /// Section 22.2 declares no accessor for it, and it is reached only by a
    /// command lowering, which downcasts inside its own backend.
    #[cfg_attr(not(feature = "dx12"), allow(dead_code))]
    native: Box<dyn BindGroupBackend>,
}

impl BindGroup {
    /// Assembles a created bind group.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only `Device::create_bind_group` may produce one.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        canonical: BindGroupDescriptor,
        native: Box<dyn BindGroupBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(BindGroupInner {
                id,
                device,
                descriptor: canonical,
                native,
            }),
        }
    }

    /// The backend's own descriptor packet.
    ///
    /// Crate-private because section 59 keeps native lowering out of the exported
    /// surface: the returned trait is `pub(crate)`, so this is the seam's ordinary
    /// inside face rather than a narrow door onto a public one. Its caller is the
    /// command lowering of the backend that created it, which downcasts to its own
    /// type — a group handed to another backend's device is refused portably, by
    /// device identity, long before a downcast is attempted.
    #[cfg_attr(not(feature = "dx12"), allow(dead_code))]
    pub(crate) fn native(&self) -> &dyn BindGroupBackend {
        self.inner.native.as_ref()
    }

    /// This group's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }

    /// The device that created this group.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }

    /// The layout this packet was validated against.
    pub fn layout(&self) -> &BindGroupLayout {
        &self.inner.descriptor.layout
    }

    /// The canonicalized descriptor.
    pub fn descriptor(&self) -> &BindGroupDescriptor {
        &self.inner.descriptor
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (defect D6 of the 0.16 plan): section 22.2
/// declares `#[derive(Clone)]` and no `Debug`, while a bind group is the natural
/// thing to name in a debug log. It prints the identity and the layout, not the
/// resources — a native handle in a log is a leak, and the descriptor is one call
/// away through [`BindGroup::descriptor`].
impl fmt::Debug for BindGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BindGroup")
            .field("id", &self.inner.id)
            .field("device", &self.inner.device)
            .field("layout", &self.inner.descriptor.layout)
            .finish_non_exhaustive()
    }
}
/// The device facts section 22.3 needs that are not properties of the objects.
///
/// A parameter bag rather than a device read, for the reason module 02 states for
/// its own validators: the four numbers are capability answers, and passing them
/// keeps the rule decidable and testable without a backend. The device façade
/// fills them from the four limits of the same names.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BindGroupLimits {
    max_uniform_buffer_binding_size: u64,
    max_storage_buffer_binding_size: u64,
    min_uniform_buffer_offset_alignment: u64,
    min_storage_buffer_offset_alignment: u64,
}

impl BindGroupLimits {
    /// States the four device limits a resource check depends on.
    pub(crate) fn new(
        max_uniform_buffer_binding_size: u64,
        max_storage_buffer_binding_size: u64,
        min_uniform_buffer_offset_alignment: u64,
        min_storage_buffer_offset_alignment: u64,
    ) -> Self {
        Self {
            max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size,
            min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment,
        }
    }
}

/// Checks everything about a packet that does not need a backend.
///
/// Section 22.2's canonicality rule and section 22.3's resource list, with the two
/// per-format questions taken as parameters:
///
/// ```text
/// entries canonicalized by slot    unique (a repeat is InvalidUsage)
/// every entry's slot               exists in the layout
/// BindingCount::One                -> scalar BindingResource
/// BindingCount::Fixed(n)           -> matching Array variant, len == n
/// Buffer           DeviceIdentity, range bounds, usage, size limits, alignment
/// SampledTexture   usage, aspects, dimension, sample type, multisampled
/// StorageTexture   usage, aspects, dimension, format, storage access
/// Sampler          kind compatible with the descriptor
/// ```
///
/// The two closures carry the format facts section 22.3 names as `FormatFacts`-
/// owned: `sample_type_of` answers
/// [`crate::api::format::FormatFacts::sample_type`] and
/// `storage_access_supported` answers
/// [`crate::api::format::StorageAccessSupport::supports`]. Both accessors exist,
/// so neither closure is a placeholder for a missing method, and they are here for
/// two different reasons:
///
/// * `storage_access_supported` stays a parameter because the storage answers are
///   *probed* device facts — [`crate::api::format::FormatFacts::storage_access`]
///   hands out a record a backend filled — and this validator holds only the
///   view's format name, never a device answer.
/// * `sample_type_of` is now redundant: the sample type is a pure table, read
///   through the same free function [`crate::api::format::FormatFacts::aspects`]
///   uses, so a parameter is a second copy of a mapping section 8 owns. It is kept
///   rather than withdrawn in this change because every caller in this crate
///   supplies one, and withdrawing it is an edit to those call sites.
///
/// Identity is checked before anything else, per resource, in the sense of section
/// 3.1: a buffer, view, or sampler from another device is
/// [`RhiErrorKind::WrongDevice`], and there is no implicit migration.
pub(crate) fn validate_bind_group_descriptor(
    desc: &BindGroupDescriptor,
    limits: BindGroupLimits,
    sample_type_of: impl Fn(TextureFormat) -> Option<TextureSampleType>,
    storage_access_supported: impl Fn(TextureFormat, StorageAccess) -> bool,
) -> RhiResult<()> {
    let target = desc.layout.device_identity();

    // Section 22.2: canonicalized by BindingSlotId, and a duplicate slot is
    // InvalidUsage. The check is a duplicate scan rather than an ordering check,
    // because §22.2 says "canonicalized", not "already canonical" — the Device
    // sorts the vector, so a caller's order is not a refusal.
    let mut slots = desc
        .entries
        .iter()
        .map(|entry| entry.slot.get())
        .collect::<Vec<_>>();
    slots.sort_unstable();
    if let Some(duplicate) = slots.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("a bind group fills slot {} twice", duplicate[0]),
        ));
    }

    for entry in &desc.entries {
        let Some(slot) = desc.layout.slot(entry.slot) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a bind group fills slot {}, which the layout does not declare",
                    entry.slot.get()
                ),
            ));
        };

        validate_resource(
            entry,
            slot,
            limits,
            target,
            &sample_type_of,
            &storage_access_supported,
        )?;
    }

    Ok(())
}

/// Checks one entry's resource against the slot it fills.
///
/// The outer match is on the resource variant and the inner one on the slot's
/// kind, so that each combination is resolved by the rule that actually applies
/// and a mismatched pairing gets the message that names both sides. No wildcard
/// arm appears on [`BindingKind`]: a new kind must be classified here, and the
/// arms that refuse a whole class say so explicitly.
fn validate_resource(
    entry: &BindGroupEntry,
    slot: &BindingSlot,
    limits: BindGroupLimits,
    target: DeviceIdentity,
    sample_type_of: &impl Fn(TextureFormat) -> Option<TextureSampleType>,
    storage_access_supported: &impl Fn(TextureFormat, StorageAccess) -> bool,
) -> RhiResult<()> {
    match &entry.resource {
        BindingResource::Buffer(binding) => match &slot.kind {
            BindingKind::UniformBuffer { min_size } => {
                validate_scalar_slot(slot)?;
                validate_buffer(binding, *min_size, BufferKind::Uniform, limits, target)
            }
            BindingKind::StorageBuffer { min_size, .. } => {
                validate_scalar_slot(slot)?;
                validate_buffer(binding, *min_size, BufferKind::Storage, limits, target)
            }
            BindingKind::SampledTexture { .. }
            | BindingKind::StorageTexture { .. }
            | BindingKind::Sampler { .. }
            | BindingKind::AccelerationStructure
            | BindingKind::ExternalTexture => Err(resource_mismatch(slot, "buffer")),
        },

        BindingResource::Texture(view) => match &slot.kind {
            BindingKind::SampledTexture {
                dimension,
                sample_type,
                multisampled,
            } => {
                validate_scalar_slot(slot)?;
                validate_sampled_texture(
                    view,
                    *dimension,
                    *sample_type,
                    *multisampled,
                    target,
                    sample_type_of,
                )
            }
            BindingKind::StorageTexture {
                dimension,
                format,
                access,
            } => {
                validate_scalar_slot(slot)?;
                validate_storage_texture(
                    view,
                    *dimension,
                    *format,
                    *access,
                    target,
                    storage_access_supported,
                )
            }
            BindingKind::UniformBuffer { .. }
            | BindingKind::StorageBuffer { .. }
            | BindingKind::Sampler { .. }
            | BindingKind::AccelerationStructure
            | BindingKind::ExternalTexture => Err(resource_mismatch(slot, "texture")),
        },

        BindingResource::Sampler(sampler) => match &slot.kind {
            BindingKind::Sampler { kind } => {
                validate_scalar_slot(slot)?;
                validate_sampler(sampler, *kind, target)
            }
            BindingKind::UniformBuffer { .. }
            | BindingKind::StorageBuffer { .. }
            | BindingKind::SampledTexture { .. }
            | BindingKind::StorageTexture { .. }
            | BindingKind::AccelerationStructure
            | BindingKind::ExternalTexture => Err(resource_mismatch(slot, "sampler")),
        },

        BindingResource::AccelerationStructure(structure) => match &slot.kind {
            BindingKind::AccelerationStructure => {
                validate_scalar_slot(slot)?;
                if structure.device_identity() != target {
                    return Err(RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "acceleration structure belongs to a different device",
                    )
                    .with_object(structure.id()));
                }
                Ok(())
            }
            _ => Err(resource_mismatch(slot, "acceleration structure")),
        },

        BindingResource::ExternalTexture(texture) => match &slot.kind {
            BindingKind::ExternalTexture => {
                validate_scalar_slot(slot)?;
                if texture.device_identity() != target {
                    return Err(RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "external texture belongs to a different device",
                    )
                    .with_object(texture.id()));
                }
                Ok(())
            }
            _ => Err(resource_mismatch(slot, "external texture")),
        },

        BindingResource::BufferArray(bindings) => match &slot.kind {
            BindingKind::UniformBuffer { min_size } => {
                validate_array_len(slot, bindings.len())?;
                for binding in bindings {
                    validate_buffer(binding, *min_size, BufferKind::Uniform, limits, target)?;
                }
                Ok(())
            }
            BindingKind::StorageBuffer { min_size, .. } => {
                validate_array_len(slot, bindings.len())?;
                for binding in bindings {
                    validate_buffer(binding, *min_size, BufferKind::Storage, limits, target)?;
                }
                Ok(())
            }
            BindingKind::SampledTexture { .. }
            | BindingKind::StorageTexture { .. }
            | BindingKind::Sampler { .. }
            | BindingKind::AccelerationStructure
            | BindingKind::ExternalTexture => Err(array_mismatch(slot, "buffer array")),
        },

        BindingResource::TextureArray(views) => match &slot.kind {
            BindingKind::SampledTexture {
                dimension,
                sample_type,
                multisampled,
            } => {
                validate_array_len(slot, views.len())?;
                for view in views {
                    validate_sampled_texture(
                        view,
                        *dimension,
                        *sample_type,
                        *multisampled,
                        target,
                        sample_type_of,
                    )?;
                }
                Ok(())
            }
            BindingKind::StorageTexture {
                dimension,
                format,
                access,
            } => {
                validate_array_len(slot, views.len())?;
                for view in views {
                    validate_storage_texture(
                        view,
                        *dimension,
                        *format,
                        *access,
                        target,
                        storage_access_supported,
                    )?;
                }
                Ok(())
            }
            BindingKind::UniformBuffer { .. }
            | BindingKind::StorageBuffer { .. }
            | BindingKind::Sampler { .. }
            | BindingKind::AccelerationStructure
            | BindingKind::ExternalTexture => Err(array_mismatch(slot, "texture array")),
        },

        BindingResource::SamplerArray(samplers) => match &slot.kind {
            BindingKind::Sampler { kind } => {
                validate_array_len(slot, samplers.len())?;
                for sampler in samplers {
                    validate_sampler(sampler, *kind, target)?;
                }
                Ok(())
            }
            BindingKind::UniformBuffer { .. }
            | BindingKind::StorageBuffer { .. }
            | BindingKind::SampledTexture { .. }
            | BindingKind::StorageTexture { .. }
            | BindingKind::AccelerationStructure
            | BindingKind::ExternalTexture => Err(array_mismatch(slot, "sampler array")),
        },

        BindingResource::AccelerationStructureArray(structures) => match &slot.kind {
            BindingKind::AccelerationStructure => {
                validate_array_len(slot, structures.len())?;
                for structure in structures {
                    if structure.device_identity() != target {
                        return Err(RhiError::new(
                            RhiErrorKind::WrongDevice,
                            "acceleration structure belongs to a different device",
                        )
                        .with_object(structure.id()));
                    }
                }
                Ok(())
            }
            _ => Err(array_mismatch(slot, "acceleration-structure array")),
        },
    }
}

/// A slot declared with `BindingCount::One` must take a scalar resource.
///
/// The check is on the variant rather than on `elements() == 1`, because section
/// 22.1's rule is about spelling: an array of length 1 cannot stand in for `One`.
fn validate_scalar_slot(slot: &BindingSlot) -> RhiResult<()> {
    match slot.count {
        BindingCount::One => Ok(()),
        BindingCount::Fixed(elements) => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "slot {} is declared with a fixed array of {elements} elements, so it takes the \
                 matching array, not one resource",
                slot.slot.get()
            ),
        )),
        BindingCount::RuntimeSized => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "slot {} is runtime-sized and takes an array packet, not one resource",
                slot.slot.get()
            ),
        )),
    }
}

/// A slot declared with `BindingCount::Fixed(n)` must take exactly `n` elements.
fn validate_array_len(slot: &BindingSlot, len: usize) -> RhiResult<()> {
    match slot.count {
        BindingCount::One => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "slot {} is declared with a single element, so an array cannot fill it even at \
                 length 1",
                slot.slot.get()
            ),
        )),
        BindingCount::Fixed(elements) if elements as usize == len => Ok(()),
        BindingCount::Fixed(elements) => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "slot {} is declared with a fixed array of {elements} elements, but the packet \
                 binds {len}",
                slot.slot.get()
            ),
        )),
        BindingCount::RuntimeSized if len > 0 => Ok(()),
        BindingCount::RuntimeSized => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "slot {} is runtime-sized but the packet binds no elements",
                slot.slot.get()
            ),
        )),
    }
}

/// The refusal for a resource whose variant does not match the slot's kind.
fn resource_mismatch(slot: &BindingSlot, bound: &str) -> RhiError {
    RhiError::new(
        RhiErrorKind::InvalidUsage,
        format!(
            "slot {} is declared with {:?}, but the packet binds a {bound}",
            slot.slot.get(),
            slot.kind
        ),
    )
}

/// The refusal for an array filling a slot declared with a non-array kind.
fn array_mismatch(slot: &BindingSlot, bound: &str) -> RhiError {
    RhiError::new(
        RhiErrorKind::InvalidUsage,
        format!(
            "slot {} is declared with {:?}, which takes one resource element, so a {bound} \
             cannot fill it",
            slot.slot.get(),
            slot.kind
        ),
    )
}

/// Which buffer limit and alignment a buffer binding is measured against.
#[derive(Clone, Copy)]
enum BufferKind {
    Uniform,
    Storage,
}

/// Section 22.3's buffer list.
fn validate_buffer(
    binding: &BufferBinding,
    required_min_size: u64,
    kind: BufferKind,
    limits: BindGroupLimits,
    target: DeviceIdentity,
) -> RhiResult<()> {
    // Identity first, in O(1), before any other rule (section 3.1).
    validate_buffer_ownership(&binding.buffer, target)?;

    let descriptor = binding.buffer.descriptor();
    let (usage, max_size, alignment, usage_name) = match kind {
        BufferKind::Uniform => (
            BufferUsage::UNIFORM,
            limits.max_uniform_buffer_binding_size,
            limits.min_uniform_buffer_offset_alignment,
            "UNIFORM",
        ),
        BufferKind::Storage => (
            BufferUsage::STORAGE,
            limits.max_storage_buffer_binding_size,
            limits.min_storage_buffer_offset_alignment,
            "STORAGE",
        ),
    };
    if !descriptor.usage.contains(usage) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a buffer binding requires a buffer created with BufferUsage::{usage_name}, \
                 which this buffer does not have"
            ),
        ));
    }

    // Range bounds, through the buffer module's own rule rather than a second copy
    // of "no overflow, inside the buffer, non-empty".
    validate_buffer_range(binding.range, descriptor.size)?;

    if binding.range.size < required_min_size {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a binding of {} bytes is smaller than the {required_min_size} bytes the layout \
                 requires",
                binding.range.size
            ),
        ));
    }
    if binding.range.size > max_size {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a binding of {} bytes is over the device maximum of {max_size}",
                binding.range.size
            ),
        ));
    }
    if alignment != 0 && binding.range.offset % alignment != 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a base offset of {} is not a multiple of the device's required alignment of \
                 {alignment}",
                binding.range.offset
            ),
        ));
    }
    Ok(())
}

/// Section 22.3's sampled-texture list.
fn validate_sampled_texture(
    view: &TextureView,
    dimension: TextureViewDimension,
    sample_type: TextureSampleType,
    multisampled: bool,
    target: DeviceIdentity,
    sample_type_of: &impl Fn(TextureFormat) -> Option<TextureSampleType>,
) -> RhiResult<()> {
    if view.device_identity() != target {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "texture view belongs to a different device",
        )
        .with_object(view.id()));
    }

    if !view
        .texture()
        .descriptor()
        .usage
        .contains(TextureUsage::SAMPLED)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a sampled texture binding requires a texture created with TextureUsage::SAMPLED",
        ));
    }
    if view.descriptor().dimension != dimension {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the layout expects a {dimension:?} sampled texture, but the view is {:?}",
                view.descriptor().dimension
            ),
        ));
    }

    // P0 has no sampled stencil semantics. The aspect rules below are equalities
    // rather than containments so that a depth-stencil view selecting DEPTH is
    // accepted and one selecting DEPTH|STENCIL is not: "forbids sampled STENCIL" is
    // a rule about what the view *selects*, not about what its format carries.
    let aspects = view.aspects();
    match sample_type {
        TextureSampleType::Depth => {
            if aspects != TextureAspects::DEPTH {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a Depth sampled texture must bind a view selecting exactly the DEPTH \
                     aspect; P0 forbids sampled STENCIL",
                ));
            }
        }
        TextureSampleType::Float
        | TextureSampleType::UnfilterableFloat
        | TextureSampleType::Sint
        | TextureSampleType::Uint => {
            if aspects != TextureAspects::COLOR {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a color or integer sampled texture must bind a view selecting exactly the \
                     COLOR aspect",
                ));
            }
        }
    }

    let format_sample_type = sample_type_of(view.format());
    if format_sample_type != Some(sample_type) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the layout expects a {sample_type:?} sampled texture, but {:?} samples as \
                 {format_sample_type:?}",
                view.format()
            ),
        ));
    }

    if multisampled != (view.sample_count() > 1) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the layout declares multisampled = {multisampled}, but the underlying texture \
                 has a sample count of {}",
                view.sample_count()
            ),
        ));
    }

    Ok(())
}

/// Section 22.3's storage-texture list.
fn validate_storage_texture(
    view: &TextureView,
    dimension: TextureViewDimension,
    format: TextureFormat,
    access: StorageAccess,
    target: DeviceIdentity,
    storage_access_supported: &impl Fn(TextureFormat, StorageAccess) -> bool,
) -> RhiResult<()> {
    if view.device_identity() != target {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "texture view belongs to a different device",
        )
        .with_object(view.id()));
    }

    if !view
        .texture()
        .descriptor()
        .usage
        .contains(TextureUsage::STORAGE)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a storage texture binding requires a texture created with TextureUsage::STORAGE",
        ));
    }
    if view.aspects() != TextureAspects::COLOR {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a storage texture must bind a view selecting exactly the COLOR aspect",
        ));
    }
    if view.descriptor().dimension != dimension {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the layout expects a {dimension:?} storage texture, but the view is {:?}",
                view.descriptor().dimension
            ),
        ));
    }
    // The view's *resolved* format, not the base texture's: an alternate view
    // format is what the shader reads and writes.
    if view.format() != format {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the layout expects a {format:?} storage texture, but the view is {:?}",
                view.format()
            ),
        ));
    }
    if !storage_access_supported(view.format(), access) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!(
                "this device does not support {access:?} access to a {:?} storage texture",
                view.format()
            ),
        ));
    }

    Ok(())
}

/// Section 22.3's sampler rule: the kind must be compatible with the descriptor.
///
/// Implemented as exactly the comparison half of the rule. A [`SamplerKind`]
/// describes what the *shader* expects, and the one expectation the descriptor can
/// contradict is "this sampler compares": [`SamplerKind::Comparison`] requires a
/// comparison function and every other kind requires that there is none. The
/// filter half is not decided here — section 22.3's own note says whether a
/// sampler and its texture are legal as a paired use belongs to shader interface
/// and pipeline validation, and a filtering interface satisfied by a nearest
/// sampler is not a device error.
fn validate_sampler(sampler: &Sampler, kind: SamplerKind, target: DeviceIdentity) -> RhiResult<()> {
    if sampler.device_identity() != target {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "sampler belongs to a different device",
        )
        .with_object(sampler.id()));
    }

    let compares = sampler.descriptor().compare.is_some();
    match kind {
        SamplerKind::Comparison if compares => Ok(()),
        SamplerKind::Comparison => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the layout declares a Comparison sampler, but the sampler descriptor has no \
             comparison function",
        )),
        SamplerKind::Filtering | SamplerKind::NonFiltering => {
            if compares {
                Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the layout declares a non-comparison sampler, but the sampler descriptor \
                     has a comparison function",
                ))
            } else {
                Ok(())
            }
        }
    }
}

/// Section 22.2's creation verb, defined in the chapter that owns the type it
/// produces.
///
/// The placement is the specification's own: section 22.2 writes this verb in an
/// `impl Device` in its own chapter, so the definition site is the owner.
impl Device {
    /// Creates a bind group on this device from a descriptor.
    ///
    /// Two steps before the stop, in this order:
    ///
    /// 1. Section 3.1's O(1) identity step, comparing the *layout's* device
    ///    against this device. The validator takes the layout's device as the
    ///    reference for every resource it checks, so proving that reference is
    ///    this device is the façade's half of the rule — without it, a packet
    ///    whose layout and resources all belong to another device would validate
    ///    against that device and say nothing about this one.
    /// 2. Section 22.2's canonicality rule and section 22.3's per-resource lists,
    ///    through `validate_bind_group_descriptor`, against this device's four
    ///    binding limits and its two per-format answers.
    ///
    /// The backend is then asked once, and is handed the *canonical* descriptor
    /// rather than the caller's. The two are the same packet — section 22.2 makes
    /// ascending slot order part of what a group is, so the sort is a
    /// normalization and not a change — but passing the canonical form means a
    /// lowering iterates entries in slot order without having to know that rule,
    /// and means the packet the backend wrote descriptors from is the packet this
    /// handle answers from.
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> RhiResult<BindGroup> {
        let layout_device = desc.layout.device_identity();
        if layout_device != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the layout this packet is validated against belongs to a different device; \
                 there is no implicit migration between devices",
            )
            .with_object(desc.layout.id()));
        }

        // Section 6.5's liveness verdict, after the ownership comparison above
        // and before any device fact is read. The order is the one section 3.1
        // and section 6.5 give: a packet whose layout belongs to another device
        // is `WrongDevice` even when this device is also lost, because the
        // caller's mistake is the packet.
        self.require_active()?;

        let capabilities = self.capabilities();

        // The four limits section 22.3 measures a buffer binding against. An
        // unexposed ceiling imposes none, and an unexposed alignment imposes none
        // either — `validate_buffer` skips an alignment of zero — which is the
        // "not applicable rather than zero" convention the pipeline validators
        // state for a limit the device does not expose. A ceiling of zero here
        // would refuse every buffer binding.
        let limits = BindGroupLimits::new(
            capabilities
                .limit(LimitKey::MaxUniformBufferBindingSize)
                .unwrap_or(u64::MAX),
            capabilities
                .limit(LimitKey::MaxStorageBufferBindingSize)
                .unwrap_or(u64::MAX),
            capabilities
                .limit(LimitKey::MinUniformBufferOffsetAlignment)
                .unwrap_or(0),
            capabilities
                .limit(LimitKey::MinStorageBufferOffsetAlignment)
                .unwrap_or(0),
        );

        // The sample type is a pure table, read through the same free function
        // `FormatFacts::aspects` uses; the storage-access answer is a *probed*
        // device fact, so it is read from the device's own format record rather
        // than derived. Both stay parameters of the validator: see its docs.
        validate_bind_group_descriptor(desc, limits, sample_type, |format, access| {
            capabilities
                .format(format)
                .is_some_and(|facts| facts.storage_access().supports(access))
        })?;

        // The one backend call, and the last statement that can fail. It takes the
        // canonical packet — see the note above — and is where the ownership
        // section 22.2 describes becomes concrete: a native descriptor holds
        // addresses, so the object that owns them must be the object returned here
        // and not the caller's buffers.
        let canonical = desc.canonicalized();
        let native = self.native().create_bind_group(&canonical)?;

        Ok(BindGroup::new(
            ObjectId::next(),
            self.identity(),
            canonical,
            native,
        ))
    }
}
