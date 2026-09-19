//! The RenderGraph bridge (specification 04 §37, 04 §38.3, 06 §50, and 06 §51).
//!
//! **This file is being written in specification order and covers the whole of
//! the engine bridge.** It owns three things:
//!
//! ```text
//! §37    actual resource use    PipelineScope, AccessMask, TextureUseIntent,
//!                               BufferUse, TextureUse, FrameAttachmentUse,
//!                               ResourceUse
//! §38.3  declared vs actual     DeclaredWorkContract, DeclaredContentContract,
//!                               validate_recorded_work
//! §50    bridge semantics       what the graph owns and what the RHI owns
//! §51    transient allocation   TransientResourceDesc, AllocationCompatibilityClass,
//!                               AllocationRequirements, TransientAllocationPlan,
//!                               TransientRealization, TransientAllocationService
//! ```
//!
//! # Why one file holds two chapters
//!
//! Chapter 04 declares its own `pub mod graph_bridge` (at
//! `04-recording-resource-uses.md` L853) rather than putting the resource-use
//! vocabulary into the recording module, and chapter 06 declares another (at
//! `06-statistics-diagnostics-graph-bridge.md` L1168). The frozen layout in
//! `01-platform-device-capability.md` §2 names exactly **one** `graph_bridge`,
//! and `design-rhi.md` requires every public interface to be defined exactly
//! once, so the two blocks are one Rust module — this file. Chapter 04 keeps
//! ownership of the §37 and §38.3 definitions; the file follows the module the
//! specification puts them in.
//!
//! # What this module owns
//!
//! - The vocabulary in which a recorder *reports* what it actually touched
//!   ([`ResourceUse`] and the three use records under it).
//! - The declared-vs-actual check a graph compiler asks for
//!   ([`validate_recorded_work`]) and the contract it is checked against
//!   ([`DeclaredWorkContract`]).
//! - The physical-placement question a graph asks about a transient resource
//!   ([`AllocationRequirements`]) and the answer it gets back
//!   ([`TransientRealization`]), carried by [`TransientAllocationService`].
//!
//! # What it deliberately does not own
//!
//! Graph semantics. Resource versions, declared uses, the DAG, culling,
//! scheduling, logical lane assignment, lifetime, present relation, and alias
//! decisions belong to the graph compiler (section 50). This module carries
//! *questions across the boundary* — "what did you actually touch", "does this
//! match what I declared", "what does this transient need" — so that the RHI
//! never learns what a pass is, and the graph never learns what a barrier is.
//!
//! It also does not own hazard lowering, command recording, or the recorder's
//! command-level use sequence. Section 37.1 requires the recorder to retain a
//! `Command #N -> uses [...]` sequence internally; [`ResourceUse`] is the merged
//! summary that emerges from it, and section 37.1 is explicit that the summary
//! cannot be used to infer the internal synchronization of the work.
//!
//! # The invariant it enforces
//!
//! Every use record names a resource by its live logical handle, and every
//! declaration check is decided in the portable domain: [`validate_recorded_work`]
//! compares logical identity, ranges, subresources, and intents, and it never
//! consults a driver. Section 37.4 is the reason this matters — a `ResourceUse`
//! answers *hazard and access*, and nothing here may be read as proof that a
//! shader wrote a whole range. Definedness remains the graph's declaration, not
//! this module's inference.
//!
//! # Two entries of this file are not constructible by a caller
//!
//! [`DeclaredContentContract`], [`TransientAllocationPlan`], and
//! [`TransientRealization`] are all described by the specification as generated
//! by one side of the boundary rather than built by a caller, so none of them
//! has a public constructor — see each type's documentation for the gap that
//! leaves at the seam.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label};
use crate::api::resource::buffer::{Buffer, BufferDescriptor, BufferRange};
use crate::api::resource::subresource::TextureSubresourceRange;
use crate::api::resource::texture::{Texture, TextureDescriptor};

// ---------------------------------------------------------------------------
// Section 37 — actual resource use.
// ---------------------------------------------------------------------------

/// Which pipeline stages one resource use is visible to.
///
/// A bitset rather than an enum, because a use is not "one of" four stages: a
/// uniform buffer read by a vertex shader and a fragment shader is visible to
/// both, and section 37 has to express `VERTEX | FRAGMENT` without inventing a
/// combined variant for every pair.
///
/// The four bits are not [`crate::api::shader::ShaderStage`]. `COPY` has no
/// shader stage at all yet does have a pipeline scope, so a use that names
/// `COPY` is stating that a copy command touched the resource — which is
/// exactly how section 37.2 accounts for recorder upload and readback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineScope(u32);

impl PipelineScope {
    /// The vertex stage.
    pub const VERTEX: Self = Self(1 << 0);
    /// The fragment stage.
    pub const FRAGMENT: Self = Self(1 << 1);
    /// The compute stage.
    pub const COMPUTE: Self = Self(1 << 2);
    /// The copy domain, which has no shader stage and does have a scope.
    pub const COPY: Self = Self(1 << 3);

    /// Whether every bit set in `other` is also set in `self`.
    ///
    /// An empty `other` is contained in everything, matching
    /// [`crate::api::resource::buffer::BufferUsage::contains`]: nothing here
    /// rejects an empty *query*, and a use record with no stage is refused (if
    /// at all) by the validation that reads it, not by this predicate.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two scopes.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl core::fmt::Display for PipelineScope {
    /// Renders the scope as `VERTEX|FRAGMENT`, or `<none>` when empty.
    ///
    /// Section 37 does not declare this impl. It is added for the same reason
    /// [`crate::api::resource::buffer::BufferUsage`] carries one: a refused or
    /// uncovered use has to name *which* stage set was involved, and a raw `u32`
    /// cannot say it in an error message a caller can act on.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write_bit_names(
            formatter,
            self.0,
            &[
                (Self::VERTEX.0, "VERTEX"),
                (Self::FRAGMENT.0, "FRAGMENT"),
                (Self::COMPUTE.0, "COMPUTE"),
                (Self::COPY.0, "COPY"),
            ],
        )
    }
}

/// How a used resource is accessed.
///
/// A bitset rather than an enum, for the same reason as [`PipelineScope`]: one
/// use may be a read *and* a write — a storage buffer read-modify-written by one
/// dispatch, or a color attachment read and written by one draw — and hazard
/// lowering needs both bits at once.
///
/// Section 37 reserves [`AccessMask::HOST_READ`] and [`AccessMask::HOST_WRITE`]
/// for graph and tooling host observations. They are **not** emitted by normal
/// recorder commands: a recorder upload or readback reports `COPY` with
/// `COPY_WRITE` or `COPY_READ`, and host staging and host completion observation
/// are not represented as recorder resource uses at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AccessMask(u32);

impl AccessMask {
    /// Read as a vertex buffer.
    pub const VERTEX_READ: Self = Self(1 << 0);
    /// Read as an index buffer.
    pub const INDEX_READ: Self = Self(1 << 1);
    /// Read as a uniform buffer.
    pub const UNIFORM_READ: Self = Self(1 << 2);
    /// Read by a shader.
    pub const SHADER_READ: Self = Self(1 << 3);
    /// Written by a shader.
    pub const SHADER_WRITE: Self = Self(1 << 4);
    /// Read as a color attachment.
    pub const COLOR_READ: Self = Self(1 << 5);
    /// Written as a color attachment.
    pub const COLOR_WRITE: Self = Self(1 << 6);
    /// Read as a depth attachment.
    pub const DEPTH_READ: Self = Self(1 << 7);
    /// Written as a depth attachment.
    pub const DEPTH_WRITE: Self = Self(1 << 8);
    /// Read as a stencil attachment.
    pub const STENCIL_READ: Self = Self(1 << 9);
    /// Written as a stencil attachment.
    pub const STENCIL_WRITE: Self = Self(1 << 10);
    /// Read as a copy source.
    pub const COPY_READ: Self = Self(1 << 11);
    /// Written as a copy destination.
    pub const COPY_WRITE: Self = Self(1 << 12);
    /// Read by the host. Reserved for graph and tooling observation.
    pub const HOST_READ: Self = Self(1 << 13);
    /// Written by the host. Reserved for graph and tooling observation.
    pub const HOST_WRITE: Self = Self(1 << 14);
    /// Presented.
    ///
    /// Transcribed as written at `04-recording-resource-uses.md` L888. It is
    /// noted as an open question in the inventory (04-05 U5): the v1 closure
    /// corrections removed `HOST` and `PRESENT` from the P0 `ResourceUse`
    /// vocabulary, and a frame's present-time use is expressed as
    /// `ResourceUse::Frame` with `COLOR_WRITE` (05 L1349–L1354), not with this
    /// bit. It is kept because section 37 writes it, and removing it would be a
    /// specification change rather than an implementation of one.
    pub const PRESENT: Self = Self(1 << 15);

    /// Whether every bit set in `other` is also set in `self`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two access sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl core::fmt::Display for AccessMask {
    /// Renders the mask as `SHADER_READ|SHADER_WRITE`, or `<none>` when empty.
    ///
    /// Added for the same reason as [`PipelineScope`]'s `Display`: an uncovered
    /// use must be able to say which access was not declared.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write_bit_names(
            formatter,
            self.0,
            &[
                (Self::VERTEX_READ.0, "VERTEX_READ"),
                (Self::INDEX_READ.0, "INDEX_READ"),
                (Self::UNIFORM_READ.0, "UNIFORM_READ"),
                (Self::SHADER_READ.0, "SHADER_READ"),
                (Self::SHADER_WRITE.0, "SHADER_WRITE"),
                (Self::COLOR_READ.0, "COLOR_READ"),
                (Self::COLOR_WRITE.0, "COLOR_WRITE"),
                (Self::DEPTH_READ.0, "DEPTH_READ"),
                (Self::DEPTH_WRITE.0, "DEPTH_WRITE"),
                (Self::STENCIL_READ.0, "STENCIL_READ"),
                (Self::STENCIL_WRITE.0, "STENCIL_WRITE"),
                (Self::COPY_READ.0, "COPY_READ"),
                (Self::COPY_WRITE.0, "COPY_WRITE"),
                (Self::HOST_READ.0, "HOST_READ"),
                (Self::HOST_WRITE.0, "HOST_WRITE"),
                (Self::PRESENT.0, "PRESENT"),
            ],
        )
    }
}

/// What a texture use is *for*.
///
/// [`AccessMask`] answers what bits a command touched; this answers what kind of
/// attachment or copy role the texture played. The two are separate because the
/// same mask can mean different roles — `COLOR_READ | COLOR_WRITE` is a color
/// attachment, while `SHADER_READ | SHADER_WRITE` is a read-write storage image —
/// and the role, not the mask, is what attachment and copy-intent compatibility
/// is checked against.
///
/// [`Self::Present`] is transcribed as written at
/// `04-recording-resource-uses.md` L906. It is the same open question as
/// [`AccessMask::PRESENT`]: a frame's presentation use travels as
/// [`ResourceUse::Frame`], not as a texture intent. Kept because section 37
/// writes it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureUseIntent {
    /// Sampled by a shader.
    ShaderRead,
    /// Read and written by a shader, as in a storage image.
    ShaderReadWrite,
    /// Used as a color attachment.
    ColorAttachment,
    /// Read as a depth or stencil attachment.
    DepthStencilRead,
    /// Written as a depth or stencil attachment.
    DepthStencilWrite,
    /// Read as a copy source.
    CopySrc,
    /// Written as a copy destination.
    CopyDst,
    /// Read as a multisample resolve source.
    ResolveSrc,
    /// Written as a multisample resolve destination.
    ResolveDst,
    /// Presented.
    Present,
}

/// One buffer range used by recorded work.
///
/// The range is part of the record rather than implied by the buffer: buffer
/// coverage is one of the things section 38.3's check compares, and a record
/// that named only the buffer could not express a partial write.
#[derive(Clone)]
pub struct BufferUse {
    /// The buffer that was used. Held by clone, so the record survives the
    /// caller dropping its own handle (section 38.2).
    pub buffer: Buffer,
    /// The range within it that was touched.
    pub range: BufferRange,
    /// The stages that touched it.
    pub stages: PipelineScope,
    /// How they touched it.
    pub access: AccessMask,
}

/// One texture subresource range used by recorded work.
#[derive(Clone)]
pub struct TextureUse {
    /// The texture that was used. Held by clone, for the reason given on
    /// [`BufferUse::buffer`].
    pub texture: Texture,
    /// The mip/layer/aspect range that was touched.
    pub subresources: TextureSubresourceRange,
    /// The stages that touched it.
    pub stages: PipelineScope,
    /// How they touched it.
    pub access: AccessMask,
    /// What role the texture played.
    pub intent: TextureUseIntent,
}

/// An acquired frame used by recorded work.
///
/// A frame is not an ordinary [`Texture`] — the root specification makes
/// `FrameAttachment` neither a `Texture` nor a `TextureView` — so it enters the
/// use model as its own record instead of being folded into [`TextureUse`]. A
/// frame has no subresource range to name, because a caller does not choose
/// which layer of an acquired image it renders into.
#[derive(Clone, Copy, Debug)]
pub struct FrameAttachmentUse {
    /// The acquired frame that was used.
    pub frame: crate::api::presentation::AcquiredFrameId,
    /// The stages that touched it.
    pub stages: PipelineScope,
    /// How they touched it.
    pub access: AccessMask,
}

/// One resource the recorded work actually touched.
///
/// This is the merged actual-use summary section 38.1 exposes for a
/// `RecordedWork`. It answers hazard and access — and nothing more. Section 37.4
/// is explicit that a `SHADER_WRITE` on a storage buffer does **not** mean the
/// shader filled the whole range, so definedness, write coverage, and store or
/// discard expectations remain the graph's declaration. Nothing may read a
/// [`ResourceUse`] as proof of a write.
#[non_exhaustive]
#[derive(Clone)]
pub enum ResourceUse {
    /// A buffer range.
    Buffer(BufferUse),
    /// A texture subresource range.
    Texture(TextureUse),
    /// An acquired frame.
    Frame(FrameAttachmentUse),
}

// ---------------------------------------------------------------------------
// Section 38.3 — declared vs actual.
// ---------------------------------------------------------------------------

/// The uses a graph compiler declares for one pass, plus its content contract.
///
/// The graph side of section 38.3's check. Ordinary direct-RHI users do not
/// construct one: it exists so that a compiler which already knows what a pass
/// declared can have the recorder's actual uses checked against that
/// declaration, instead of the RHI re-deriving the graph's intent.
#[derive(Clone)]
pub struct DeclaredWorkContract {
    /// Diagnostic label of the pass. Excluded from every canonical hash.
    pub pass_label: Label,
    /// The uses the pass declared. May be more conservative than the actual
    /// uses; section 38.3 permits a declaration to cover more than is used and
    /// forbids it to cover less.
    pub uses: Vec<ResourceUse>,
    /// The attachment, load/store, and definedness contract the pass declared.
    pub content: DeclaredContentContract,
}

/// The attachment, load/store, and definedness contract a graph compiler
/// declares for one pass.
///
/// Opaque, and deliberately so: the concrete graph type stays engine-internal
/// and the bridge compares only portable semantics (section 38.3). What this
/// type promises is that the contract exists and travels with the pass; what it
/// contains is the graph's business.
///
/// # This type has no public constructor
///
/// It is described as *graph-generated*, so a caller outside the crate cannot
/// build one, and [`DeclaredWorkContract::content`] is nevertheless a public
/// field. That combination is a gap at the seam and is recorded rather than
/// papered over: the compiler that is supposed to mint this type lives outside
/// this crate, so it needs an entry point the specification does not declare,
/// and inventing one here would be adding public API the specification does not
/// have. See the module note in `api/mod.rs` for the standing rule that an
/// interface is transcribed before it is improved.
///
/// The constructor below is crate-private for the same reason: it exists so the
/// contract tests in `crate::api::tests` can exercise the shape of a
/// declaration, not so that a caller can mint one.
///
/// # `Clone` is a forced addition
///
/// Section 38.3 writes no derives on this type, and writes `#[derive(Clone)]` on
/// [`DeclaredWorkContract`], which holds one of these in a public field. The
/// derive is therefore pulled through rather than chosen: without it the
/// contract it belongs to could not be cloneable.
#[derive(Clone)]
pub struct DeclaredContentContract {
    /// Reserved for the graph compiler's declaration payload.
    ///
    /// Named rather than omitted so that the port that fills it adds a field
    /// here instead of reshaping the type.
    #[expect(
        dead_code,
        reason = "written by the constructor and read by nothing: the graph compiler that fills it is not written"
    )]
    domain: DeclaredContentDomain,
}

/// The backend-private declaration payload behind a [`DeclaredContentContract`].
///
/// Reserved, and modelled the same way as `platform::device`'s `DeviceDomain`:
/// the type is uninhabited of content until the graph compiler lands, and having
/// the seam named now means the port adds fields here rather than reshaping the
/// public type.
#[derive(Clone)]
struct DeclaredContentDomain;

impl DeclaredContentContract {
    /// Records an empty declaration.
    ///
    /// Crate-private: see the type documentation. The only intended minting side
    /// is the graph compiler, which cannot reach a `pub(crate)` constructor, so
    /// this is a placeholder for the tests and for the port.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the graph compiler mints this once the RenderGraph crate lands"
        )
    )]
    pub(crate) fn new() -> Self {
        Self {
            domain: DeclaredContentDomain,
        }
    }
}

/// Checks the actual uses of recorded work against the declaration for it.
///
/// Section 38.3's rule, in its own terms:
///
/// ```text
/// actual use must be covered by declared use
/// declared use may be more conservative
/// cross-DeviceIdentity / lost Device fails immediately
/// ```
///
/// Coverage is checked over the buffer byte range, the texture mip/layer/aspect
/// range, the stage and access sets, and attachment/copy/present intent
/// compatibility.
///
/// The failure kinds are:
///
/// ```text
/// wrong device             -> WrongDevice      (04 L1121)
/// lost device              -> DeviceLost       (04 L1121)
/// use not covered          -> InvalidUsage
/// ```
///
/// The third is not named by section 38.3; it is recorded in the inventory as
/// 04-05 U4. It is [`RhiErrorKind::InvalidUsage`] here because that is what the
/// case is: a declaration and a recording that cannot both be true, which
/// section 4 defines as a portable usage error. A backend is never consulted,
/// which is the point — section 37.4's check is a comparison of two portable
/// descriptions.
///
/// What it deliberately cannot check is shader data-dependent write coverage.
/// Section 37.4 forbids claiming that: a storage buffer's `SHADER_WRITE` says a
/// shader may write the range, not that it filled it.
pub fn validate_recorded_work(
    work: &crate::api::command::RecordedWork,
    declared: &DeclaredWorkContract,
) -> RhiResult<()> {
    let _ = (work, declared);
    unimplemented!(
        "the declared-vs-actual comparison reads RecordedWork's internal command \
         sequence, which module 04 has not built; the contract is fixed, the \
         comparison is not built"
    )
}

/// Which device a declared contract was written for.
///
/// Section 38.3 fails a cross-device comparison immediately, and that comparison
/// is O(1) and needs no backend, so it is stated here as the portable half of
/// the check even though the recording sequence it would be compared against is
/// not built yet.
///
/// Crate-private: only the code holding a recording can name the device it was
/// recorded on.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "validate_recorded_work compares this once module 04 records a device identity"
    )
)]
pub(crate) fn check_same_device(
    recorded: DeviceIdentity,
    declared: DeviceIdentity,
) -> RhiResult<()> {
    if recorded != declared {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "the declared work contract and the recorded work belong to different devices",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Section 51 — transient allocation.
// ---------------------------------------------------------------------------

/// A transient resource the graph is asking about.
///
/// The graph decides lifetime overlap, asks what each overlapping resource
/// needs, decides a compatible packing, and hands the packing to the RHI to
/// realize (section 51). This is the question half of that exchange: a
/// descriptor, not a created resource, because the point is to answer before
/// anything exists.
///
/// It is not `#[non_exhaustive]` and it carries no label or device: a transient
/// is described by exactly what it would be created as, and the graph is
/// expected to be able to match both arms exhaustively.
#[derive(Clone)]
pub enum TransientResourceDesc {
    /// A transient buffer, described exactly as it would be created.
    Buffer(BufferDescriptor),
    /// A transient texture, described exactly as it would be created.
    Texture(TextureDescriptor),
}

/// A target-specific class within which two transients may share one allocation.
///
/// Opaque, and compared by equality alone: two resources whose classes are equal
/// are placeable in the same memory, and the graph never needs to know *why*.
/// The class is target-specific because what makes two resources
/// interchangeable is a driver fact — tiling, compression, and aliasing rules —
/// that no portable formula can predict, which is precisely why section 51 makes
/// the graph ask instead of compute it.
///
/// A caller cannot construct one. An empty or guessed class would let a graph
/// pack resources the target cannot actually share, and the resulting failure
/// would surface as corruption rather than as an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationCompatibilityClass(u64);

impl AllocationCompatibilityClass {
    /// Records a class the target assigned.
    ///
    /// Crate-private: only the service that knows the target's placement rules
    /// may classify a resource.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "TransientAllocationService implementations mint these once they exist"
        )
    )]
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }
}

/// What a transient resource needs from an allocation.
///
/// The answer to [`TransientAllocationService::requirements`]. It is a
/// requirement, not an allocation: the graph uses it to decide a packing, and
/// the RHI realizes the packing later. Every field is a physical placement fact
/// that is target-specific, which is why none of it is derivable from the
/// descriptor on the portable side.
#[derive(Clone, Copy, Debug)]
pub struct AllocationRequirements {
    /// Bytes the resource needs.
    pub size: u64,
    /// Alignment the resource must start at.
    pub alignment: u64,
    /// The class within which two resources may share one allocation.
    pub compatibility_class: AllocationCompatibilityClass,
    /// Whether the target would place this resource in its own allocation even
    /// when the class would permit sharing.
    ///
    /// A preference, not a rule: a graph that ignores it still gets a legal
    /// realization, and one that honours it may get a better one.
    pub prefers_dedicated: bool,
}

/// The logical packing plan the graph produced for one frame's transients.
///
/// Opaque: what a packing plan is — which transients share which slot, at which
/// offsets — is the graph's representation, and the RHI's job is to realize it
/// rather than to read it. The graph owns the alias decision (section 50), so
/// this type is the decision travelling to the side that executes it.
///
/// # This type has no public constructor
///
/// It is described as *graph-generated*, and the graph compiler lives outside
/// this crate, so the minting entry point the specification does not declare is
/// the same open gap as [`DeclaredContentContract`]'s. The constructor below is
/// crate-private for the tests and for the port.
///
/// No derives, matching section 51. The plan crosses the boundary by reference
/// and is never copied, so a `Clone` would only invite a graph to hold two
/// packings for one frame — which is the state the graph owns and this type
/// merely carries.
pub struct TransientAllocationPlan {
    /// Reserved for the graph compiler's packing description.
    #[expect(
        dead_code,
        reason = "written by the constructor and read by nothing: the graph compiler that packs it is not written"
    )]
    domain: TransientPlanDomain,
}

/// The graph-private packing description behind a [`TransientAllocationPlan`].
struct TransientPlanDomain;

impl TransientAllocationPlan {
    /// Records an empty packing.
    ///
    /// Crate-private: see the type documentation.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the graph compiler mints this once the RenderGraph crate lands"
        )
    )]
    pub(crate) fn new() -> Self {
        Self {
            domain: TransientPlanDomain,
        }
    }
}

/// The resources a realized packing actually produced.
///
/// Opaque, and owned by the RHI: section 51 puts realization on this side of the
/// boundary, so the aliased allocations, their placement, and the objects
/// standing in for the transients are backend-private state that a caller holds
/// without reading. A no-alias fallback is always legal, so an implementation is
/// free to realize every transient separately.
///
/// `Debug` is written by hand rather than derived, per adjudication A16 in the
/// 0.16 plan: the realization holds native allocations, and printing a native
/// handle into a log is a leak. It prints nothing but its own name, because the
/// type has no portable identity to print.
///
/// No derives, matching section 51: a realization is produced once and owned by
/// whoever asked for it, so a `Clone` would let two owners retire the same
/// native allocations.
pub struct TransientRealization {
    /// Reserved for the backend's realized allocations.
    #[expect(
        dead_code,
        reason = "written by the constructor and read by nothing: the backend port that realizes it is not written"
    )]
    domain: TransientRealizationDomain,
}

/// The backend-private realized state behind a [`TransientRealization`].
struct TransientRealizationDomain;

impl TransientRealization {
    /// Records an empty realization.
    ///
    /// Crate-private: section 51 makes realization the RHI's half of the
    /// exchange, so only a service implementation may mint one.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "TransientAllocationService implementations mint these once they exist"
        )
    )]
    pub(crate) fn new() -> Self {
        Self {
            domain: TransientRealizationDomain,
        }
    }
}

impl core::fmt::Debug for TransientAllocationPlan {
    /// Prints the type name and the ellipsis that marks it incomplete.
    ///
    /// Hand-written, per adjudication A16. The plan holds a graph-private
    /// packing representation whose contents are not portable state, and the
    /// type carries no portable identity to show instead.
    ///
    /// The literal below is deliberate rather than a use of
    /// `finish_non_exhaustive()`. That method writes the `, ..` only when at
    /// least one field was added, so with no portable field to add it would
    /// print the bare type name — which asserts that this is all there is. The
    /// true statement is the opposite: there is more here, and it is not being
    /// shown.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("TransientAllocationPlan { .. }")
    }
}

impl core::fmt::Debug for TransientRealization {
    /// Prints the type name and the ellipsis, for the reason given on
    /// [`TransientAllocationPlan`]'s `Debug` — including why this is a literal
    /// rather than `finish_non_exhaustive()`.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("TransientRealization { .. }")
    }
}

/// The seam a graph compiler uses to place transient resources.
///
/// A **service implementation seam**, not a capability trait (section 51). The
/// difference matters: the RHI is not asked whether it "supports" aliasing, it
/// is told what a transient needs and answers with what it can actually place. A
/// capability trait would move the question into the type system, where the
/// answer has to be the same for every resource in the frame — and it is not,
/// because placement depends on the individual descriptor.
///
/// The two calls are the same exchange in order: ask what one resource requires,
/// then realize the packing the graph decided from those answers.
///
/// # The hand-off point is not specified
///
/// Section 51 declares this trait and nothing that returns an implementation of
/// it. There is no `Device::transient_allocation_service()` and no provider-side
/// equivalent in the specification, so a graph compiler has a seam but no
/// specified way to obtain one. That gap is recorded rather than closed here:
/// adding an accessor would be adding public API the specification does not
/// have, and the missing decision (whether the service is device-scoped,
/// provider-scoped, or handed to the graph by the host) is not this module's to
/// make.
pub trait TransientAllocationService {
    /// What this transient needs from an allocation.
    ///
    /// Target-specific by nature, so an implementation must consult the target;
    /// there is no portable formula for size, alignment, or compatibility class
    /// from a descriptor alone. A descriptor that is portable-invalid is still
    /// refused with [`RhiErrorKind::InvalidUsage`] rather than passed down, for
    /// the reason section 4 gives.
    fn requirements(&self, desc: &TransientResourceDesc) -> RhiResult<AllocationRequirements>;

    /// Realizes the packing the graph decided.
    ///
    /// Takes `&mut self` because realization is where the service's own state
    /// lives: the allocations it hands back have to be remembered so that a
    /// later plan can reuse them and so that they can be retired. What a
    /// realization's lifetime is, and whether one service may realize more than
    /// one plan, is not stated by section 51 and is recorded as an open question
    /// rather than assumed here.
    fn realize(&mut self, plan: &TransientAllocationPlan) -> RhiResult<TransientRealization>;
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

/// Renders a bit set as `A|B`, or `<none>` when no bit is set.
///
/// Shared by the two `Display` impls above so that an empty set is spelled the
/// same way everywhere and reads as a fact rather than as silence.
fn write_bit_names(
    formatter: &mut core::fmt::Formatter<'_>,
    value: u32,
    names: &[(u32, &str)],
) -> core::fmt::Result {
    let mut written = false;
    for (bit, name) in names {
        if value & bit == *bit {
            if written {
                formatter.write_str("|")?;
            }
            formatter.write_str(name)?;
            written = true;
        }
    }
    if !written {
        formatter.write_str("<none>")?;
    }
    Ok(())
}
