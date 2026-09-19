//! Section 21: the bind group layout.
//!
//! One declared binding, the layout that holds them, and the two distinct tokens
//! of section 21.1 — the exact same-Device compatibility token and the cache
//! fingerprint, which never substitutes for it. Section 20.5's layout validator
//! lives here because its subject is a layout, even though the list it checks is
//! numbered in section 20.
//!
//! Not owned here: the packet that is validated against a layout (section 22,
//! `group.rs`) and the aggregate limits that span groups. Section 20.5 excludes
//! those deliberately, and they are checked where a
//! `crate::api::pipeline::PipelineInterface` is created.

use core::fmt;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::shader::ShaderStages;

use super::vocabulary::{
    BindingCount, BindingKind, BindingSlotId, BindingSupport, BindingSupportQuery, is_buffer_kind,
    validate_binding_count, validate_binding_kind,
};

/// One declared binding in a bind group layout.
///
/// The `visibility` is a stage set rather than a single stage because a binding
/// used by both the vertex and fragment stages is one binding, not two; splitting
/// it would force two slots and two resources for the same data.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BindingSlot {
    /// The logical slot this declaration occupies.
    pub slot: BindingSlotId,

    /// The stages that may see this binding. Must not be empty.
    pub visibility: ShaderStages,

    /// The resource semantics of this binding.
    pub kind: BindingKind,

    /// How many resource elements this binding holds.
    pub count: BindingCount,

    /// Whether a dynamic offset is applied when the group is bound.
    ///
    /// "Valid only for buffer bindings" (section 20.5): a dynamic offset on a
    /// texture or sampler binding is refused at layout creation, because there is
    /// nothing in the corresponding native binding to offset.
    pub dynamic_offset: bool,
}

impl BindingSlot {
    /// Declares one binding with a single element and no dynamic offset.
    pub fn new(slot: BindingSlotId, visibility: ShaderStages, kind: BindingKind) -> Self {
        Self {
            slot,
            visibility,
            kind,
            count: BindingCount::One,
            dynamic_offset: false,
        }
    }

    /// Declares a fixed-length resource array here.
    pub fn with_count(mut self, count: BindingCount) -> Self {
        self.count = count;
        self
    }

    /// Declares that a dynamic offset is applied to this binding.
    pub fn with_dynamic_offset(mut self, enabled: bool) -> Self {
        self.dynamic_offset = enabled;
        self
    }
}

/// Device-scoped exact compatibility token for a canonical layout descriptor.
///
/// Produced by the Device interning canonical layout descriptors, and
/// deliberately unconstructable by a caller: section 21.1 introduced it to
/// replace a `CompatibilityId(pub u128)` that "could easily be misused as equal
/// 128-bit hash => equal correctness".
///
/// Two consequences are fixed by the specification and are not negotiable by a
/// caller or a backend:
///
/// ```text
/// two canonical BindGroupLayoutDescriptors completely identical
///     -> BindGroupLayoutCompatibilityId must be identical
///
/// different Device objects
///     -> DeviceIdentity validation still comes first
/// ```
///
/// So a compatibility token cannot make objects interoperable across devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutCompatibilityId(u64);

impl BindGroupLayoutCompatibilityId {
    /// Mints one token.
    ///
    /// Crate-private, because section 21.1 says a caller cannot construct it: the
    /// value means "this Device interned this exact descriptor", and only the
    /// interning Device knows that.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_bind_group_layout interns through this once api::platform \
                      is declared"
        )
    )]
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the interned value.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Canonical descriptor fingerprint for cache, diagnostics, and Capture provenance.
///
/// The field is public because section 21.1 declares it so, and that is
/// deliberate rather than an oversight: a fingerprint is *not* an identity, so
/// there is nothing to protect by hiding its bytes. The freeze rule attached to it
/// is what matters:
///
/// > Equal fingerprints cannot alone replace correctness validation.
///
/// Compatibility decisions use [`BindGroupLayoutCompatibilityId`] and the complete
/// canonical descriptor; this type is a hint.
///
/// The algorithm that produces it is not fixed in P0, so the value is supplied by
/// the Device's interning step rather than computed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayoutFingerprint(pub [u8; 32]);

/// Everything a caller states about a bind group layout before it exists.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor {
    /// Diagnostic label. Does not participate in compatibility, and is excluded
    /// from every canonical hash (section 19.8).
    pub label: Label,

    /// The declared bindings. Canonicalized into ascending slot order on creation.
    pub entries: Vec<BindingSlot>,
}

impl BindGroupLayoutDescriptor {
    /// Describes a layout. Checks nothing:
    /// `validate_bind_group_layout_descriptor` is the check, and it needs the
    /// device's capability answers.
    pub fn new(entries: Vec<BindingSlot>) -> Self {
        Self {
            label: Label::default(),
            entries,
        }
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// The entries in ascending [`BindingSlotId`] order.
    ///
    /// Section 21.2's canonicalization, and the reason it is a method rather than
    /// a free function: the canonical form is a property of the descriptor, and
    /// everything that stores, compares, or interns one stores this form. It sorts
    /// and does nothing else — a duplicate slot is refused by validation before
    /// this is ever called, so there is no merge rule to state here.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_bind_group_layout canonicalizes through this once \
                      api::platform is declared"
        )
    )]
    pub(crate) fn canonicalized(&self) -> Self {
        let mut entries = self.entries.clone();
        entries.sort_by_key(|entry| entry.slot.get());
        Self {
            label: self.label.clone(),
            entries,
        }
    }
}

/// A created bind group layout.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the [`DeviceIdentity`]
/// that created it. It stores the *canonicalized* descriptor, so
/// [`Self::descriptor`] answers what the layout actually is rather than what the
/// caller typed.
#[derive(Clone)]
pub struct BindGroupLayout {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: BindGroupLayoutDescriptor,
    compatibility_id: BindGroupLayoutCompatibilityId,
    fingerprint: LayoutFingerprint,
}

impl BindGroupLayout {
    /// Assembles a created layout.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only `Device::create_bind_group_layout` may produce one. It takes the
    /// canonicalized descriptor and the two interned tokens, because both are
    /// outputs of the interning step that the façade owns.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_bind_group_layout calls this once api::platform is declared"
        )
    )]
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        canonical: BindGroupLayoutDescriptor,
        compatibility_id: BindGroupLayoutCompatibilityId,
        fingerprint: LayoutFingerprint,
    ) -> Self {
        Self {
            id,
            device,
            descriptor: canonical,
            compatibility_id,
            fingerprint,
        }
    }

    /// This layout's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device that created this layout.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The canonicalized descriptor.
    ///
    /// Returns the canonical form, not the caller's input: section 21.2 makes
    /// ascending slot order part of what the layout *is*.
    pub fn descriptor(&self) -> &BindGroupLayoutDescriptor {
        &self.descriptor
    }

    /// The exact same-device compatibility token.
    pub fn compatibility_id(&self) -> BindGroupLayoutCompatibilityId {
        self.compatibility_id
    }

    /// The canonical descriptor fingerprint.
    ///
    /// A cache, diagnostics, and Capture-provenance hint. Section 21.1 is explicit
    /// that it cannot alone replace correctness validation, which is why
    /// compatibility is decided by [`Self::compatibility_id`] and by comparing the
    /// complete canonical descriptors.
    pub fn fingerprint(&self) -> LayoutFingerprint {
        self.fingerprint
    }

    /// How many dynamic buffer elements this layout declares.
    ///
    /// Section 21.3 freezes the consumption order these offsets are read in, and
    /// this count is what the recorder checks its `dynamic_offsets.len()` against:
    ///
    /// ```text
    /// ascending BindingSlotId
    ///     -> ascending element index within the same Fixed(n) binding
    /// ```
    ///
    /// A `Fixed(n)` binding with a dynamic offset contributes `n` offsets, not
    /// one, which is why this is a sum over element counts rather than a count of
    /// slots.
    pub fn dynamic_offset_count(&self) -> u32 {
        self.descriptor
            .entries
            .iter()
            .filter(|entry| entry.dynamic_offset)
            .map(|entry| entry.count.elements())
            .sum()
    }

    /// The declaration occupying `slot`, if this layout declares one.
    ///
    /// The lookup the packet validator uses to decide whether an entry fills a
    /// slot the layout actually has.
    pub fn slot(&self, slot: BindingSlotId) -> Option<&BindingSlot> {
        self.descriptor
            .entries
            .iter()
            .find(|entry| entry.slot == slot)
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (defect D6 of the 0.16 plan): section 21.3
/// declares `#[derive(Clone)]` and no `Debug`, while
/// [`crate::api::pipeline::PipelineInterfaceDescriptor`] derives `Debug` and
/// contains layouts. The resolution is that every public opaque handle implements
/// `Debug` portably — the identity rather than the contents, because section 7.1
/// describes an object by its identity, because the backend port will add a native
/// field that has no reason to be `Debug`, and because printing a native handle
/// into a log would leak it.
impl fmt::Debug for BindGroupLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BindGroupLayout")
            .field("id", &self.id)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}
/// Checks a layout descriptor against the capability answers it must respect.
///
/// Section 20.5's list, in its own order:
///
/// ```text
/// visibility is non-empty
/// entry count <= MaxBindingsPerGroup
/// every slot < MaxBindingsPerGroup
/// slots are unique within the layout
/// BindingSupportQuery == Supported
/// dynamic_offset allowed only for UniformBuffer / StorageBuffer
/// Fixed(n): n >= 2, and capability support is present
/// ```
///
/// The aggregate limits section 20.5 explicitly excludes — per-stage resource
/// count and dynamic-buffers-per-pipeline-layout — are not checked here, for the
/// section's own reason: they span multiple groups, so no single layout can decide
/// them.
///
/// Two error kinds, split the way module 02 splits them: a descriptor that
/// declares zero visibility, repeats a slot, or puts a dynamic offset on a texture
/// is the caller's mistake ([`RhiErrorKind::InvalidUsage`]), while a binding the
/// device cannot express is not ([`RhiErrorKind::Unsupported`]).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Device::create_bind_group_layout validates through this once api::platform \
                  is declared"
    )
)]
pub(crate) fn validate_bind_group_layout_descriptor(
    desc: &BindGroupLayoutDescriptor,
    max_bindings_per_group: u32,
    binding_support: impl Fn(&BindingSupportQuery) -> BindingSupport,
) -> RhiResult<()> {
    if desc.entries.len() > max_bindings_per_group as usize {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a layout declares {} bindings, over the device maximum of {}",
                desc.entries.len(),
                max_bindings_per_group
            ),
        ));
    }

    let mut slots = desc
        .entries
        .iter()
        .map(|entry| entry.slot.get())
        .collect::<Vec<_>>();
    slots.sort_unstable();
    if let Some(duplicate) = slots.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("a layout declares slot {} twice", duplicate[0]),
        ));
    }

    for entry in &desc.entries {
        if entry.slot.get() >= max_bindings_per_group {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "slot {} is at or past the device's MaxBindingsPerGroup of {}",
                    entry.slot.get(),
                    max_bindings_per_group
                ),
            ));
        }
        if entry.visibility.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "slot {} declares no visibility, so no stage could ever see it",
                    entry.slot.get()
                ),
            ));
        }
        validate_binding_kind(&entry.kind)?;
        validate_binding_count(entry.count)?;

        if entry.dynamic_offset && !is_buffer_kind(&entry.kind) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "slot {} declares a dynamic offset on a binding that is not a buffer",
                    entry.slot.get()
                ),
            ));
        }

        let query = BindingSupportQuery {
            visibility: entry.visibility,
            kind: entry.kind.clone(),
            count: entry.count,
            dynamic_offset: entry.dynamic_offset,
        };
        if binding_support(&query) == BindingSupport::Unsupported {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device cannot express the {:?} binding declared at slot {}",
                    entry.kind,
                    entry.slot.get()
                ),
            ));
        }
    }

    Ok(())
}
