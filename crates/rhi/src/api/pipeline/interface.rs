//! Section 23.1-23.2: `PipelineInterface` and its descriptor.
//!
//! The interface is the contract between shader entry points and bind-group
//! packets: the caller's ordered group-layout sequence, kept in order because
//! group numbering is semantic, plus the two tokens of section 23.2 that describe
//! it. This file owns the 23.1 aggregate-count validator and nothing else.
//!
//! Not owned here: section 23.3's requirement merge (`resources.rs`) and the
//! pipelines that consume an interface (`raster.rs`, `compute.rs`). The
//! interface states what a pipeline must satisfy; the pipeline is where it is
//! checked, so no rule about a specific pipeline appears below.

use core::fmt;

use crate::api::binding::{
    BindGroupIndex, BindGroupLayout, BindingKind, BindingLimitClass, LayoutFingerprint,
};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::platform::requirements::LimitKey;
use crate::api::shader::ShaderStage;
use crate::api::shader::vocabulary::stage_mask;

// ---------------------------------------------------------------------------
// Section 23 - PipelineInterface
// ---------------------------------------------------------------------------

/// Device-scoped exact compatibility token for a canonical interface descriptor.
///
/// The counterpart of [`crate::api::binding::BindGroupLayoutCompatibilityId`],
/// and separate from it on purpose: a layout and an *ordered sequence* of layouts
/// are different things, and one token type for both would make it possible to
/// compare an interface against a layout and get a plausible answer.
///
/// Produced by interning the complete ordered group-layout sequence (section
/// 23.2), so it is not constructible by a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineInterfaceCompatibilityId(u64);

impl PipelineInterfaceCompatibilityId {
    /// Mints one token.
    ///
    /// Crate-private for the same reason as the layout token: the value means
    /// "this Device interned this exact ordered sequence", and only the interning
    /// Device knows that. The one caller is [`Device::create_pipeline_interface`].
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the interned value.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Everything a caller states about a pipeline interface before it exists.
///
/// The vector index of `groups` *is* the [`BindGroupIndex`], which is why section
/// 23.1 requires an explicitly supplied empty layout rather than a sparse map:
/// "group 0 and group 2" is spelled by putting an empty layout at index 1, so
/// every backend gets stable, simple logical group numbering and no group is
/// renumbered by absence.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PipelineInterfaceDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,

    /// The group layouts, in [`BindGroupIndex`] order.
    pub groups: Vec<BindGroupLayout>,
}

impl PipelineInterfaceDescriptor {
    /// Describes an interface over an ordered group sequence.
    pub fn new(groups: Vec<BindGroupLayout>) -> Self {
        Self {
            label: Label::default(),
            groups,
        }
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// The canonical bytes section 23.2's interning is keyed on.
    ///
    /// The ordered group sequence, each group written as the canonical bytes of
    /// its descriptor behind its own length. The label is excluded, for the reason
    /// section 19.8 gives, and no group is skipped or reordered — section 23.1
    /// makes the vector index *be* the [`BindGroupIndex`], so the same two layouts
    /// in the other order are a different interface and must intern to a different
    /// id.
    ///
    /// Each group arrives already canonical: the descriptor read here is
    /// [`BindGroupLayout::descriptor`], which answers the canonical form the
    /// creating device stored rather than what the caller typed.
    pub(crate) fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.groups.len() as u64).to_le_bytes());
        for group in &self.groups {
            let bytes = group.descriptor().canonical_bytes();
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }
}

/// A created pipeline interface.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the [`DeviceIdentity`]
/// that created it. It stores the *canonicalized* descriptor, which for an
/// interface is the caller's ordered sequence interned as one value: unlike a
/// bind-group layout, the order is semantic, so nothing is sorted.
#[derive(Clone)]
pub struct PipelineInterface {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: PipelineInterfaceDescriptor,
    compatibility_id: PipelineInterfaceCompatibilityId,
    fingerprint: LayoutFingerprint,
}

impl PipelineInterface {
    /// Assembles a created interface.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, and
    /// both tokens are outcomes of the Device's interning step (section 23.2).
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: PipelineInterfaceDescriptor,
        compatibility_id: PipelineInterfaceCompatibilityId,
        fingerprint: LayoutFingerprint,
    ) -> Self {
        Self {
            id,
            device,
            descriptor,
            compatibility_id,
            fingerprint,
        }
    }

    /// This interface's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device that created this interface.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The descriptor this interface was created from.
    pub fn descriptor(&self) -> &PipelineInterfaceDescriptor {
        &self.descriptor
    }

    /// The exact same-device compatibility token.
    pub fn compatibility_id(&self) -> PipelineInterfaceCompatibilityId {
        self.compatibility_id
    }

    /// The canonical interface fingerprint.
    ///
    /// Section 23.2 states the two roles in that order and adds "not the
    /// reverse": the fingerprint is a cache and tooling hint, and compatibility is
    /// decided by [`Self::compatibility_id`] together with the complete ordered
    /// canonical descriptor.
    pub fn fingerprint(&self) -> LayoutFingerprint {
        self.fingerprint
    }

    /// The layout at one logical group index.
    ///
    /// An absent index is `None` rather than a default empty layout: section 23.1
    /// makes an empty group an explicitly supplied layout, so "no layout here" and
    /// "an empty layout here" are different facts and must stay distinguishable.
    pub fn group(&self, index: BindGroupIndex) -> Option<&BindGroupLayout> {
        self.descriptor.groups.get(index.get() as usize)
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (defect D6 of the 0.16 plan): section 23.2
/// declares `#[derive(Clone)]` and no `Debug`, while
/// [`RasterPipelineDescriptor`](super::RasterPipelineDescriptor) derives `Clone` and contains an interface. The
/// resolution is that every public opaque handle implements `Debug` portably —
/// identity rather than contents, because a native field would arrive with the
/// backend port and printing a native handle into a log would leak it.
impl fmt::Debug for PipelineInterface {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PipelineInterface")
            .field("id", &self.id)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}
/// Checks a pipeline interface descriptor against the device's aggregate limits.
///
/// Section 23.1's list, plus section 23.1's raster addendum where the vertex input
/// is available:
///
/// ```text
/// groups.len() <= MaxBindGroups
/// all BindGroupLayout DeviceIdentity values are identical
/// per ShaderStage + BindingLimitClass:
///     aggregate all visible group entries, count Fixed(n) as n elements
///     <= binding_limit(stage, class)
/// dynamic uniform/storage buffer elements
///     <= MaxDynamicUniformBuffersPerPipelineLayout
///     <= MaxDynamicStorageBuffersPerPipelineLayout
/// ```
///
/// Two readings worth stating, because the specification's list is terse:
///
/// * "Visible" means the stage bit is set in the slot's `visibility`, and the
///   count an entry contributes is `BindingCount::elements()` — the *element*
///   count, so a `Fixed(4)` binding contributes 4 to its class and 4 to the
///   dynamic-offset aggregate, not 1. That is what "count Fixed(n) as n binding
///   elements" buys.
/// * A limit the device does not expose is not checked. Section 23.1 says so
///   outright for `MaxBindGroupsPlusVertexBuffers` ("if Device exposes") and
///   section 27.3 repeats it ("if exposed by Device", "if present"), so the same
///   convention is applied to every key here rather than inventing a required set.
///
/// Section 23.1's group-identity rule is mutual equality between the groups, as
/// written. The comparison against the *creating* device's own identity is the
/// façade's O(1) step (section 3.1), exactly as section 12.3 leaves the buffer's
/// ownership comparison to [`crate::api::resource::buffer::validate_buffer_ownership`].
pub(crate) fn validate_pipeline_interface_descriptor(
    desc: &PipelineInterfaceDescriptor,
    limit: impl Fn(LimitKey) -> Option<u64>,
    binding_limit: impl Fn(ShaderStage, BindingLimitClass) -> Option<u32>,
) -> RhiResult<()> {
    if let Some(max) = limit(LimitKey::MaxBindGroups) {
        if desc.groups.len() as u64 > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a pipeline interface declares {} bind groups, over the device maximum of \
                     {max}",
                    desc.groups.len()
                ),
            ));
        }
    }

    if let Some(first) = desc.groups.first() {
        let device = first.device_identity();
        for group in desc.groups.iter().skip(1) {
            if group.device_identity() != device {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "a pipeline interface mixes bind group layouts from different devices",
                )
                .with_object(group.id()));
            }
        }
    }

    for stage in STAGES {
        let mask = stage_mask(stage);
        for class in CLASSES {
            let Some(max) = binding_limit(stage, class) else {
                continue;
            };
            let mut total = 0u64;
            for group in &desc.groups {
                for slot in &group.descriptor().entries {
                    if slot.visibility.contains(mask) && binding_class_of(&slot.kind) == class {
                        total += slot.count.elements() as u64;
                    }
                }
            }
            if total > max as u64 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "the {stage:?} stage uses {total} {class:?} bindings, over the device \
                         maximum of {max}"
                    ),
                ));
            }
        }
    }

    let mut dynamic_uniform = 0u64;
    let mut dynamic_storage = 0u64;
    for group in &desc.groups {
        for slot in &group.descriptor().entries {
            if !slot.dynamic_offset {
                continue;
            }
            match &slot.kind {
                BindingKind::UniformBuffer { .. } => {
                    dynamic_uniform += slot.count.elements() as u64;
                }
                BindingKind::StorageBuffer { .. } => {
                    dynamic_storage += slot.count.elements() as u64;
                }
                BindingKind::SampledTexture { .. }
                | BindingKind::StorageTexture { .. }
                | BindingKind::Sampler { .. } => {}
            }
        }
    }
    if let Some(max) = limit(LimitKey::MaxDynamicUniformBuffersPerPipelineLayout) {
        if dynamic_uniform > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the interface declares {dynamic_uniform} dynamic uniform buffer elements, \
                     over the device maximum of {max}"
                ),
            ));
        }
    }
    if let Some(max) = limit(LimitKey::MaxDynamicStorageBuffersPerPipelineLayout) {
        if dynamic_storage > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the interface declares {dynamic_storage} dynamic storage buffer elements, \
                     over the device maximum of {max}"
                ),
            ));
        }
    }

    Ok(())
}

/// The three stages a pipeline can draw from, in section 19.1's declaration order.
const STAGES: [ShaderStage; 3] = [
    ShaderStage::Vertex,
    ShaderStage::Fragment,
    ShaderStage::Compute,
];

/// The five aggregate classes of section 20.4, in declaration order.
const CLASSES: [BindingLimitClass; 5] = [
    BindingLimitClass::UniformBuffers,
    BindingLimitClass::StorageBuffers,
    BindingLimitClass::SampledTextures,
    BindingLimitClass::StorageTextures,
    BindingLimitClass::Samplers,
];

/// The class one binding kind is aggregated under.
///
/// Delegates to [`crate::api::binding::vocabulary::binding_kind_class`] rather than repeating
/// the match, so that section 20.4's five classes have exactly one definition and
/// section 23.1's aggregate counts cannot disagree with section 20.5's per-layout
/// counts about which class a binding belongs to.
fn binding_class_of(kind: &BindingKind) -> BindingLimitClass {
    crate::api::binding::vocabulary::binding_kind_class(kind)
}

/// Section 23.2's creation verb, defined in the chapter that owns the type it
/// produces.
///
/// The placement is the specification's own: section 23.2 writes this verb in an
/// `impl Device` in its own chapter, so the definition site is the owner.
impl Device {
    /// Creates a pipeline interface on this device from a descriptor.
    ///
    /// Two steps before the stop, in this order:
    ///
    /// 1. Section 3.1's O(1) identity step over the group layouts. Section 23.1's
    ///    group-identity rule is *mutual equality between the groups*, as written,
    ///    and the validator below implements exactly that; comparing the groups
    ///    against this device is the façade's half of it (section 3.1), in the
    ///    same shape section 12.3 leaves a buffer's ownership comparison to
    ///    `validate_buffer_ownership`. Without it, a sequence of layouts that all
    ///    agree with each other but belong to another device would validate and
    ///    say nothing about this one.
    /// 2. Section 23.1's aggregate counts, through
    ///    `validate_pipeline_interface_descriptor`, against this device's own
    ///    limit and binding-count answers.
    ///
    /// # Why there is no backend call
    ///
    /// The same Direct3D 12 fact that keeps
    /// [`Device::create_bind_group_layout`] off the seam: an interface is an
    /// ordered sequence of group layouts, and D3D12 has no native object for
    /// either. The native work of both is done once, at pipeline creation, where
    /// the whole sequence is lowered into the root signature that is a property of
    /// that pipeline. Section 23.2's interning is what is left, and it is portable.
    pub fn create_pipeline_interface(
        &self,
        desc: &PipelineInterfaceDescriptor,
    ) -> RhiResult<PipelineInterface> {
        let identity = self.identity();
        for group in &desc.groups {
            if group.device_identity() != identity {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "a group layout in this interface belongs to a different device; an \
                     interface binds this device's layouts and nothing else",
                )
                .with_object(group.id()));
            }
        }

        // Section 6.5's liveness verdict, after every group's ownership
        // comparison and before the device's limit answers are read. Section
        // 3.1 puts the identity comparisons first; section 6.5 gives them the
        // more specific answer, so a foreign layout on a lost device is still
        // `WrongDevice`.
        self.require_active()?;

        let capabilities = self.capabilities();
        validate_pipeline_interface_descriptor(
            desc,
            |key| capabilities.limit(key),
            |stage, class| capabilities.binding_limit(stage, class),
        )?;

        // Section 23.2's interning, in the same shape section 21.1's is written in
        // `binding/layout.rs`: one byte string, two tokens derived from it. Every
        // group read here already passed this device's ownership check above, so a
        // sequence interned against this device is a sequence of this device's
        // layouts.
        let bytes = desc.canonical_bytes();
        let compatibility_id =
            PipelineInterfaceCompatibilityId::new(self.interning().intern_interface(&bytes));
        let fingerprint = LayoutFingerprint(crate::api::internal::digest::sha256(&bytes));

        Ok(PipelineInterface::new(
            ObjectId::next(),
            identity,
            desc.clone(),
            compatibility_id,
            fingerprint,
        ))
    }
}
