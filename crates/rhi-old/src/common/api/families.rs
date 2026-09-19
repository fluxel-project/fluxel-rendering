//! The remaining capability families, as vocabulary.
//!
//! [`graphics`](super::graphics) is the first family and states the conventions
//! section 11.9 of the lead 3F plan fixes; these follow them exactly:
//!
//! # The rule that decides where a family is split
//!
//! **One independently negotiable batch of API per family, and one ledger row per
//! family.** The test is not whether two verbs sound alike; it is whether they
//! always appear, are always proved, and always fail together.
//!
//! Indirect draw and indirect dispatch fail that test: a platform can offer one
//! without the other, and the ledger already carries them as two rows. An earlier
//! version of this file merged them into one trait, which forced any backend with
//! only one of the two to write a refusing method for the other -- precisely the
//! lowest-common-denominator shape this layer exists to remove. They are two traits
//! here, and the same question must be asked of every family added later.
//!
//! - each is bounded on [`FamilyApi`](super::handle::FamilyApi), never the other
//!   way round;
//! - each declares its own `Error`, because backends disagree about what a command
//!   can fail on and one shared error would be too wide for every backend or too
//!   narrow for one;
//! - each takes or returns only the base's vocabulary — [`BufferId`], [`TextureId`]
//!   — and leaves anything else to an associated type, so no descriptor is invented
//!   here that a backend would have to accept;
//! - each adds **vocabulary only**. Whether a device may be asked for the family is
//!   the ledger's answer, read once by
//!   [`require`](super::negotiate::require); none of these traits can grant it.
//!
//! # Why the resource-role families are traits at all
//!
//! `StorageBuffer` and `StorageTexture` are resource roles rather than command
//! domains: they gate how a binding is built, not which verbs exist. They are still
//! traits, because a backend that cannot serve the role must be structurally unable
//! to build the binding — the same reason the command families are traits. What
//! they deliberately do **not** carry is a capability query: asking whether a role
//! is available is `require::<StorageBuffer>()`, and a second way to ask would be a
//! second answer.
//!
//! # What is absent, and stays absent until a consumer exists
//!
//! `Multiview`, `AsyncCompute` and `TransferQueue` have ledger rows and family
//! markers but no trait here: no retained recipe declares a multiview attachment,
//! and the execution model is one queue. Writing their methods now would be
//! vocabulary for work nobody has asked for, and the markers already let a caller
//! state the requirement.
//!
//! `BaseVertex` and `FirstInstance` join that list for the same shape of reason.
//! Their ledger rows are proved on a `Vulkan` 1.0 device, because the base offset
//! and the first instance are core parameters of the draw commands rather than
//! features, and the markers let a graph require them. The verbs that name the two
//! parameters are separate families by plan section 20.1 -- [`GraphicsApi`]'s draw
//! verbs fix both at zero -- and no retained recipe declares either, so they arrive
//! with the consumer that needs them rather than here.
//!
//! [`GraphicsApi`]: super::graphics::GraphicsApi

use fluxel_rendergraph::{BufferCopyRegion, TextureCopyRegion};

use crate::common::base::resource::{BufferId, TextureId};

use super::handle::FamilyApi;

/// The compute family: compute passes, pipelines and dispatches.
pub(crate) trait ComputeApi: FamilyApi {
    /// Why a command in this family was refused or failed.
    type Error;
    /// A compute pipeline this backend created.
    type Pipeline;
    /// A binding set this backend created.
    type Bindings;

    /// Opens a compute pass.
    fn begin_compute(&mut self) -> Result<(), Self::Error>;

    /// Closes the open compute pass.
    fn end_compute(&mut self) -> Result<(), Self::Error>;

    /// Selects the pipeline subsequent dispatches record through.
    fn set_compute_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<(), Self::Error>;

    /// Applies a binding set to the open pass.
    fn set_bindings(&mut self, bindings: &Self::Bindings) -> Result<(), Self::Error>;

    /// Records a dispatch of `groups` workgroups in each dimension.
    ///
    /// Each dimension must be non-zero: a zero-sized dispatch is a legal driver
    /// no-op, and one that arrives here is a caller's mistake rather than a request
    /// to do nothing.
    fn dispatch(&mut self, groups: [u32; 3]) -> Result<(), Self::Error>;
}

/// The storage-buffer family: a buffer a shader may read or write.
pub(crate) trait StorageBufferApi: FamilyApi {
    /// Why building a binding in this family was refused.
    type Error;
    /// A storage-buffer binding this backend built.
    type Binding;

    /// Builds a binding over the whole range `[offset, offset + size)` of `buffer`.
    ///
    /// The range is stated rather than derived, because a storage binding's extent
    /// is a property of the recipe that asked for it and a backend that guessed
    /// would bind bytes the caller never declared.
    fn create_storage_binding(
        &mut self,
        buffer: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Self::Binding, Self::Error>;
}

/// The storage-texture family: a texture a shader may read or write.
pub(crate) trait StorageTextureApi: FamilyApi {
    /// Why building a binding in this family was refused.
    type Error;
    /// A storage-texture binding this backend built.
    type Binding;

    /// Builds a binding over one whole texture level.
    ///
    /// One level, because that is what the storage-texture rows describe: a
    /// per-level view is a subresource question, and a binding that named several
    /// would be claiming support for addressing this vocabulary does not carry.
    fn create_storage_binding(
        &mut self,
        texture: TextureId,
    ) -> Result<Self::Binding, Self::Error>;
}

/// The indirect-draw family: a draw whose parameters are read from a buffer.
///
/// Separate from [`IndirectDispatchApi`] because the two are separately negotiable:
/// the ledger carries one row each, and a platform may offer one without the other.
pub(crate) trait IndirectDrawApi: FamilyApi {
    /// Why a command in this family was refused or failed.
    type Error;

    /// Records `count` draws whose parameters are read from `commands`.
    ///
    /// `stride` is stated rather than assumed to be the ABI's natural size: a
    /// caller may pack records with padding, and a backend that assumed tight
    /// packing would read the wrong offsets.
    fn draw_indirect(
        &mut self,
        commands: BufferId,
        offset: u64,
        count: u32,
        stride: u32,
    ) -> Result<(), Self::Error>;
}

/// The indirect-dispatch family: a dispatch whose workgroup counts come from a buffer.
pub(crate) trait IndirectDispatchApi: FamilyApi {
    /// Why a command in this family was refused or failed.
    type Error;

    /// Records one dispatch whose workgroup counts are read from `commands`.
    fn dispatch_indirect(
        &mut self,
        commands: BufferId,
        offset: u64,
    ) -> Result<(), Self::Error>;
}

/// The copy family: buffer and texture copies, as a family rather than as floor.
///
/// Every backend in the five-platform set serves copies today, and that is not a
/// reason to put them in the base. [`graphics`](super::graphics)'s membership test
/// says a base item is something all backends must agree on for Fluxel's own
/// semantics to hold; "all five happen to have it" is a fact about today's backends,
/// not a semantic requirement. Keeping copy a family also gives the facade migration
/// somewhere to point: the retired `CopyBackend` tier becomes "the device implements
/// `Copy`", which is a ledger row rather than a parallel type hierarchy.
pub(crate) trait CopyApi: FamilyApi {
    /// Why a copy was refused or failed.
    type Error;

    /// Copies `region` from one buffer to another.
    fn copy_buffer(
        &mut self,
        source: BufferId,
        destination: BufferId,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error>;

    /// Copies `region` from one texture to another.
    fn copy_texture(
        &mut self,
        source: TextureId,
        destination: TextureId,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error>;
}
