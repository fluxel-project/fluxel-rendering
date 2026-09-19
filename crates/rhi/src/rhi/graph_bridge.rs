//! The portable bridge between recorded work and a graph compiler.
//!
//! This module owns rhi-design sections 37, 38.3, and 51.
//!
//! # What it is
//!
//! Three things that only a graph compiler needs, kept out of the direct-RHI
//! path: the actual-use vocabulary a recorded command produces, the
//! declared-versus-actual comparison, and the transient allocation service seam.
//!
//! # What it deliberately does not own
//!
//! Native state. There is no `VkImageLayout`, no `D3D12_RESOURCE_STATES`, and no
//! encoder state anywhere here. The graph decides resource versions, lifetimes,
//! culling, and alias packing; the RHI reports what its commands actually did
//! and realizes the packing the graph chose.
//!
//! # Definedness is not inferred from an access mask
//!
//! [`AccessMask`] answers "which hazard does this command carry", not "does the
//! shader fill the whole range". A storage buffer written by a shader may cover
//! any subset of its range, so initial contents, write coverage, and exported
//! definedness stay the graph's declaration, and
//! [`validate_recorded_work`] only rejects contradictions it can decide
//! statically.

use super::format::TextureFormat;
use super::platform::{DeviceIdentity, Label, RhiError, RhiErrorKind, RhiResult};
use super::presentation::AcquiredFrameId;
use super::resource::{
    Buffer, BufferDescriptor, BufferRange, Texture, TextureDescriptor, TextureSubresourceRange,
};

/// Which pipeline stages a resource use belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineScope(u32);

impl PipelineScope {
    /// The vertex stage.
    pub const VERTEX: Self = Self(1 << 0);
    /// The fragment stage.
    pub const FRAGMENT: Self = Self(1 << 1);
    /// The compute stage.
    pub const COMPUTE: Self = Self(1 << 2);
    /// A transfer or attachment command with no shader stage.
    pub const COPY: Self = Self(1 << 3);

    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Whether every bit of `other` is present.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two stage sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no stage is set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw bits, for canonical hashing and diagnostics.
    pub fn bits(self) -> u32 {
        self.0
    }
}

/// Which accesses a resource use performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AccessMask(u32);

impl AccessMask {
    /// Vertex fetch.
    pub const VERTEX_READ: Self = Self(1 << 0);
    /// Index fetch.
    pub const INDEX_READ: Self = Self(1 << 1);
    /// A uniform buffer read.
    pub const UNIFORM_READ: Self = Self(1 << 2);
    /// A shader read of any other kind.
    pub const SHADER_READ: Self = Self(1 << 3);
    /// A shader write.
    pub const SHADER_WRITE: Self = Self(1 << 4);
    /// A color attachment read.
    pub const COLOR_READ: Self = Self(1 << 5);
    /// A color attachment write.
    pub const COLOR_WRITE: Self = Self(1 << 6);
    /// A depth attachment read.
    pub const DEPTH_READ: Self = Self(1 << 7);
    /// A depth attachment write.
    pub const DEPTH_WRITE: Self = Self(1 << 8);
    /// A stencil attachment read.
    pub const STENCIL_READ: Self = Self(1 << 9);
    /// A stencil attachment write.
    pub const STENCIL_WRITE: Self = Self(1 << 10);
    /// A transfer read.
    pub const COPY_READ: Self = Self(1 << 11);
    /// A transfer write.
    pub const COPY_WRITE: Self = Self(1 << 12);
    /// A host observation of a range the GPU has finished writing.
    ///
    /// Reserved for graph and tooling bookkeeping. A recorder command never
    /// emits it: host staging and host completion are not recorder resource
    /// uses, and an upload is `COPY` with [`AccessMask::COPY_WRITE`].
    pub const HOST_READ: Self = Self(1 << 13);
    /// A host write into a range the GPU will later read.
    ///
    /// Reserved for graph and tooling bookkeeping, as with
    /// [`AccessMask::HOST_READ`].
    pub const HOST_WRITE: Self = Self(1 << 14);
    /// The presentation engine's read of a frame.
    pub const PRESENT: Self = Self(1 << 15);

    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Whether every bit of `other` is present.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two access sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no access is set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Every access that mutates a resource.
    const WRITE_BITS: u32 = AccessMask::SHADER_WRITE.0
        | AccessMask::COLOR_WRITE.0
        | AccessMask::DEPTH_WRITE.0
        | AccessMask::STENCIL_WRITE.0
        | AccessMask::COPY_WRITE.0
        | AccessMask::HOST_WRITE.0;

    /// Whether any write access is present.
    pub fn writes(self) -> bool {
        self.0 & Self::WRITE_BITS != 0
    }

    /// The raw bits, for canonical hashing and diagnostics.
    pub fn bits(self) -> u32 {
        self.0
    }
}

/// What a texture use is for, beyond its access bits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureUseIntent {
    /// Sampled by a shader.
    ShaderRead,
    /// Read and written by a shader.
    ShaderReadWrite,
    /// A color attachment.
    ColorAttachment,
    /// A depth/stencil attachment read without a write.
    DepthStencilRead,
    /// A depth/stencil attachment write.
    DepthStencilWrite,
    /// A transfer source.
    CopySrc,
    /// A transfer destination.
    CopyDst,
    /// A multisample resolve source.
    ResolveSrc,
    /// A multisample resolve destination.
    ResolveDst,
    /// Read by the presentation engine.
    Present,
}

/// A buffer use by one command.
#[derive(Clone, Debug)]
pub struct BufferUse {
    /// The buffer.
    pub buffer: Buffer,
    /// The byte range touched.
    pub range: BufferRange,
    /// The stages that touch it.
    pub stages: PipelineScope,
    /// The accesses performed.
    pub access: AccessMask,
}

/// A texture use by one command.
#[derive(Clone, Debug)]
pub struct TextureUse {
    /// The texture.
    pub texture: Texture,
    /// The mip/layer/aspect set touched.
    pub subresources: TextureSubresourceRange,
    /// The stages that touch it.
    pub stages: PipelineScope,
    /// The accesses performed.
    pub access: AccessMask,
    /// What the use is for.
    pub intent: TextureUseIntent,
}

/// A presentation frame use.
///
/// A frame is not an ordinary texture, so it enters the use model as its own
/// variant rather than being forced through a texture-shaped hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameAttachmentUse {
    /// The acquired frame.
    pub frame: AcquiredFrameId,
    /// The stages that touch it.
    pub stages: PipelineScope,
    /// The accesses performed.
    pub access: AccessMask,
}

/// One resource touched by a recorded command.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ResourceUse {
    /// A buffer use.
    Buffer(BufferUse),
    /// A texture use.
    Texture(TextureUse),
    /// A presentation frame use.
    Frame(FrameAttachmentUse),
}

/// The label a graph compiler gives one declared pass.
#[derive(Clone, Debug)]
pub struct DeclaredWorkContract {
    /// The pass's diagnostic label.
    pub pass_label: Label,
    /// The uses the graph declares for this pass.
    pub uses: Vec<ResourceUse>,
    /// The attachment, load/store, and definedness contract the graph
    /// generated.
    pub content: DeclaredContentContract,
}

/// The graph compiler's attachment and definedness contract.
///
/// It is opaque because the concrete graph types stay engine-internal; the RHI
/// bridge compares only the portable semantics it carries, which are reached
/// through this module's own accessors.
#[derive(Clone, Debug, Default)]
pub struct DeclaredContentContract {
    attachments: Vec<DeclaredAttachment>,
    buffers: Vec<DeclaredBufferWrite>,
}

impl DeclaredContentContract {
    /// An empty contract.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares one attachment's expected load/store/content behaviour.
    pub fn with_attachment(mut self, attachment: DeclaredAttachment) -> Self {
        self.attachments.push(attachment);
        self
    }

    /// Declares one buffer range the pass claims to write.
    pub fn with_buffer_write(mut self, write: DeclaredBufferWrite) -> Self {
        self.buffers.push(write);
        self
    }

    /// The declared attachments.
    pub fn attachments(&self) -> &[DeclaredAttachment] {
        &self.attachments
    }

    /// The declared buffer writes.
    pub fn buffer_writes(&self) -> &[DeclaredBufferWrite] {
        &self.buffers
    }
}

/// One attachment's declared behaviour.
#[derive(Clone, Debug)]
pub struct DeclaredAttachment {
    /// The color location, or `None` for the depth/stencil attachment.
    pub location: Option<u32>,
    /// The resource the graph binds there: a texture or an acquired frame.
    pub resource: DeclaredAttachmentResource,
    /// Whether the graph declares the previous contents as read.
    pub loads_existing: bool,
    /// Whether the graph declares the pass's result as exported.
    pub stores_result: bool,
    /// The format the graph expects.
    pub format: TextureFormat,
}

/// Which resource a declared attachment names.
#[derive(Clone, Debug)]
pub enum DeclaredAttachmentResource {
    /// A texture view's base texture.
    Texture(Texture),
    /// An acquired presentation frame.
    Frame(AcquiredFrameId),
}

/// One buffer range a pass declares it writes.
#[derive(Clone, Debug)]
pub struct DeclaredBufferWrite {
    /// The buffer.
    pub buffer: Buffer,
    /// The byte range the pass claims to define.
    pub range: BufferRange,
    /// The usage the declared range relies on.
    pub usage: super::resource::BufferUsage,
}

/// A transient resource the graph wants realized.
#[derive(Clone, Debug)]
pub enum TransientResourceDesc {
    /// A transient buffer.
    Buffer(BufferDescriptor),
    /// A transient texture.
    Texture(TextureDescriptor),
}

/// Which physical allocations may be aliased over one another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationCompatibilityClass(u64);

impl AllocationCompatibilityClass {
    pub(crate) const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// The raw class bits.
    pub fn bits(self) -> u64 {
        self.0
    }
}

/// What one transient resource needs from a physical allocation.
#[derive(Clone, Copy, Debug)]
pub struct AllocationRequirements {
    /// The byte size required.
    pub size: u64,
    /// The byte alignment required.
    pub alignment: u64,
    /// Which other requirements this one may share memory with.
    pub compatibility_class: AllocationCompatibilityClass,
    /// Whether the resource must not share memory with anything else.
    pub prefers_dedicated: bool,
}

/// A graph-generated logical packing plan.
#[derive(Clone, Debug, Default)]
pub struct TransientAllocationPlan {
    entries: Vec<TransientAllocationGroup>,
}

impl TransientAllocationPlan {
    /// An empty plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one group of resources that share a physical allocation.
    pub fn with_group(mut self, group: TransientAllocationGroup) -> Self {
        self.entries.push(group);
        self
    }

    /// The plan's allocation groups.
    pub fn groups(&self) -> impl Iterator<Item = &TransientAllocationGroup> {
        self.entries.iter()
    }
}

/// One physical allocation and the transient resources aliased into it.
#[derive(Clone, Debug)]
pub struct TransientAllocationGroup {
    /// The descriptors that share this allocation, in the graph's order.
    pub members: Vec<TransientResourceDesc>,
    /// The class every member agrees on.
    pub compatibility_class: AllocationCompatibilityClass,
}

/// The resources a plan realized.
pub struct TransientRealization {
    inner: Box<dyn TransientRealizationBackend>,
}

impl core::fmt::Debug for TransientRealization {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TransientRealization")
            .field("groups", &self.inner.group_count())
            .finish_non_exhaustive()
    }
}

impl TransientRealization {
    pub(crate) fn new(inner: Box<dyn TransientRealizationBackend>) -> Self {
        Self { inner }
    }

    /// The device identity the realized resources belong to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// How many physical allocations were created.
    pub fn group_count(&self) -> usize {
        self.inner.group_count()
    }
}

/// The backend half of a [`TransientRealization`].
pub(crate) trait TransientRealizationBackend: Send + Sync + 'static {
    /// The device identity the realized resources belong to.
    fn device_identity(&self) -> DeviceIdentity;

    /// How many physical allocations were created.
    fn group_count(&self) -> usize;
}

/// The seam a graph compiler uses to obtain and realize transient allocations.
///
/// This is a service implementation seam, not a capability trait: it answers
/// what a resource would need and then realizes the packing the graph chose. A
/// no-alias fallback is always legal, so an implementation that never shares
/// memory is a complete implementation rather than a degraded one.
pub trait TransientAllocationService {
    /// What one transient resource needs from a physical allocation.
    fn requirements(&self, desc: &TransientResourceDesc) -> RhiResult<AllocationRequirements>;

    /// Realizes a graph-generated packing plan.
    fn realize(&mut self, plan: &TransientAllocationPlan) -> RhiResult<TransientRealization>;
}

/// Which resource a recorded attachment actually bound.
#[derive(Clone, Debug)]
pub(crate) enum RecordedAttachmentResource {
    /// A texture view's base texture.
    Texture(Texture),
    /// An acquired presentation frame.
    Frame(AcquiredFrameId),
}

/// What a recorded attachment does with its contents when the scope begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttachmentLoad {
    /// The previous contents are preserved.
    Existing,
    /// The contents are replaced by a clear.
    Cleared,
    /// The contents are left undefined.
    Undefined,
}

/// One attachment a recorded raster scope actually bound.
///
/// This is the validator's view of a scope: it names the resource, the format,
/// and the load/store behaviour, and nothing about how a backend encoded it.
#[derive(Clone, Debug)]
pub(crate) struct RecordedAttachment {
    /// The color location, or `None` for the depth/stencil attachment.
    pub location: Option<u32>,
    /// The resource actually bound.
    pub resource: RecordedAttachmentResource,
    /// The format actually bound.
    pub format: TextureFormat,
    /// What happens to the contents when the scope begins.
    pub load: AttachmentLoad,
    /// Whether the result stays defined after the scope.
    pub stores: bool,
}

/// Checks a recorded work's structure against a graph declaration.
///
/// The comparison rejects only contradictions it can decide from the recorded
/// commands themselves: an actual use with no declared coverage, an attachment
/// the graph did not declare, a declared store against an actual discard, a
/// declared load against an actual clear, and a device mismatch.
///
/// It cannot and does not claim to prove that a shader wrote every byte of a
/// storage range. Write coverage and exported definedness stay the graph's
/// declaration, because no access mask can express them.
pub fn validate_recorded_work(
    work: &super::command::RecordedWork,
    declared: &DeclaredWorkContract,
) -> RhiResult<()> {
    let device = work.device_identity();

    // Declared uses first: a declaration from another device means the graph
    // and the recorder are talking about different devices, which no later
    // comparison can repair.
    for use_ in declared.uses.iter() {
        let Some(owner) = use_owner(use_) else {
            continue;
        };
        if owner != device {
            return Err(RhiError::wrong_device(
                "a declared use belongs to another device than the recorded work",
            ));
        }
    }

    // Actual uses must be covered by the declaration. A declaration may be
    // wider than reality; reality may never be wider than the declaration.
    for actual in work.resource_uses() {
        if !declared
            .uses
            .iter()
            .any(|candidate| declared_use_covers(candidate, actual))
        {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                "a recorded resource use is not covered by the graph declaration",
            ));
        }
    }

    // Every actual attachment must appear in the declaration with the same
    // resource, format, and load/store behaviour.
    for actual in work.recorded_attachments() {
        let Some(expected) = declared
            .content
            .attachments()
            .iter()
            .find(|candidate| candidate.location == actual.location)
        else {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "attachment {} is used by the recorded work but not declared",
                    describe_location(actual.location)
                ),
            ));
        };

        if !declared_attachment_matches(expected, actual) {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "attachment {} is a different resource than the graph declared",
                    describe_location(actual.location)
                ),
            ));
        }
        if expected.format != actual.format {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "attachment {} is {:?} but the graph declared {:?}",
                    describe_location(actual.location),
                    actual.format,
                    expected.format
                ),
            ));
        }
        if expected.loads_existing && actual.load == AttachmentLoad::Cleared {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "attachment {} declares loading existing contents but the recorded work clears it",
                    describe_location(actual.location)
                ),
            ));
        }
        if !expected.loads_existing && actual.load == AttachmentLoad::Existing {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "attachment {} declares a fresh write but the recorded work loads existing contents",
                    describe_location(actual.location)
                ),
            ));
        }
        if expected.stores_result && !actual.stores {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "attachment {} declares its result exported but the recorded work discards it",
                    describe_location(actual.location)
                ),
            ));
        }
    }

    // A declared exported attachment that the work never binds is a graph bug,
    // but only the declaration claiming a *stored* result makes it a
    // contradiction the recorded work can prove.
    for expected in declared.content.attachments().iter() {
        if !expected.stores_result {
            continue;
        }
        if work
            .recorded_attachments()
            .iter()
            .any(|actual| actual.location == expected.location)
        {
            continue;
        }
        return Err(RhiError::new(
            RhiErrorKind::IncompatibleInterface,
            format!(
                "the graph declares an exported result at {} but the recorded work never binds it",
                describe_location(expected.location)
            ),
        ));
    }

    Ok(())
}

/// The device that owns a use.
fn use_owner(use_: &ResourceUse) -> Option<DeviceIdentity> {
    match use_ {
        ResourceUse::Buffer(use_) => Some(use_.buffer.device_identity()),
        ResourceUse::Texture(use_) => Some(use_.texture.device_identity()),
        ResourceUse::Frame(use_) => Some(use_.frame.device_identity()),
    }
}

/// A human-readable name for an attachment location.
fn describe_location(location: Option<u32>) -> String {
    match location {
        Some(index) => format!("color location {index}"),
        None => "the depth/stencil attachment".to_string(),
    }
}

/// Whether a declared attachment names the same resource as a recorded one.
fn declared_attachment_matches(declared: &DeclaredAttachment, actual: &RecordedAttachment) -> bool {
    match (&declared.resource, &actual.resource) {
        (
            DeclaredAttachmentResource::Texture(declared),
            RecordedAttachmentResource::Texture(actual),
        ) => declared.id() == actual.id(),
        (DeclaredAttachmentResource::Frame(declared), RecordedAttachmentResource::Frame(actual)) => {
            *declared == *actual
        }
        _ => false,
    }
}

/// Whether one declared use covers one actual use, including ranges and access.
fn declared_use_covers(declared: &ResourceUse, actual: &ResourceUse) -> bool {
    match (declared, actual) {
        (ResourceUse::Buffer(declared), ResourceUse::Buffer(actual)) => {
            declared.buffer.id() == actual.buffer.id()
                && range_covers(declared.range, actual.range)
                && declared.stages.contains(actual.stages)
                && declared.access.contains(actual.access)
        }
        (ResourceUse::Texture(declared), ResourceUse::Texture(actual)) => {
            declared.texture.id() == actual.texture.id()
                && subresource_covers(declared.subresources, actual.subresources)
                && declared.stages.contains(actual.stages)
                && declared.access.contains(actual.access)
        }
        (ResourceUse::Frame(declared), ResourceUse::Frame(actual)) => {
            declared.frame == actual.frame
                && declared.stages.contains(actual.stages)
                && declared.access.contains(actual.access)
        }
        _ => false,
    }
}

/// Whether one byte range covers another within the same buffer.
fn range_covers(declared: BufferRange, actual: BufferRange) -> bool {
    match (declared.end(), actual.end()) {
        (Some(declared_end), Some(actual_end)) => {
            declared.offset <= actual.offset && actual_end <= declared_end
        }
        // An overflowing declaration is treated as open-ended rather than as a
        // silent non-cover, so a nonsense range surfaces as a mismatch.
        _ => declared.offset <= actual.offset,
    }
}

/// Whether one subresource range covers another across mips, layers, and aspects.
fn subresource_covers(declared: TextureSubresourceRange, actual: TextureSubresourceRange) -> bool {
    let mips = declared.base_mip <= actual.base_mip
        && actual.base_mip + actual.mip_count <= declared.base_mip + declared.mip_count;
    let layers = declared.base_layer <= actual.base_layer
        && actual.base_layer + actual.layer_count <= declared.base_layer + declared.layer_count;
    mips && layers && declared.aspects.contains(actual.aspects)
}

