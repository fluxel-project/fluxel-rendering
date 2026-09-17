//! The remaining capability families, as vocabulary.
//!
//! [`graphics`](super::graphics) is the first family and states the conventions
//! section 11.9 of the lead 3F plan fixes; these four follow them exactly:
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

/// The indirect family: commands whose parameters are read from a buffer.
pub(crate) trait IndirectApi: FamilyApi {
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

    /// Records one dispatch whose workgroup counts are read from `commands`.
    fn dispatch_indirect(
        &mut self,
        commands: BufferId,
        offset: u64,
    ) -> Result<(), Self::Error>;
}
