//! Specification section 56: the portable command IR and captured recorded work.
//!
//! One responsibility: **what a recording did, in portable semantic terms.** This
//! file is the middle of section 52's chain — the live
//! [`crate::api::command::RecordedWork`] retains a compact internal IR, and
//! [`CapturedRecordedWork`] is the expansion a ReplayRuntime reads. It is an
//! in-memory semantic IR and **not a file format** (section 56), which is why
//! nothing here has an opcode, a version, or a byte layout.
//!
//! Not owned here: the live recorder and its internal types (`api::command`,
//! module 04), the values the commands carry ([`super::mutation`]), the object
//! definitions they name ([`super::definition`]), and the plan they are submitted
//! in ([`super::plan`]).
//!
//! # The lowering is a normalization, and this is why
//!
//! Section 52.2 requires a live `RecordedWork` to make "portable commands" and
//! the "actual command/use sequence" retrievable, and permits a compact internal
//! IR. Module 04's IR is compact in one specific way that a reader must know
//! about: a draw does **not** record a preceding run of `SetRasterPipeline`,
//! `SetBindGroup`, `SetViewport`, and so on. It records the state that was
//! current when the draw was issued, in module 04's crate-private `RasterDraw`
//! record, because that is what a backend lowers —
//! a later state change must not retroactively rewrite an earlier draw.
//!
//! So the [`PortableCommand`] list for such a recording is a normalized
//! expansion:
//!
//! ```text
//! internal:  RasterDraw { pipeline, groups, viewport, .., range, instances }
//! captured:  SetRasterPipeline, SetBindGroup.., SetViewport, .., Draw
//! ```
//!
//! and two consecutive draws that used identical state therefore produce two
//! identical state-setting runs. That is faithful — replaying the expansion
//! produces the same GPU work — but it is not a byte-for-byte echo of the
//! caller's recording, and a tool that diffs a captured command list against its
//! own source of truth should expect the difference. The alternative, recording
//! state changes as they were issued, is a second recording of the same thing and
//! section 52.2 refuses it in as many words.
//!
//! # The two invariants a reader of this file should carry
//!
//! 1. **Order within `commands` is real GPU order.** It is the one place a
//!    capture can learn execution order, because [`super::SemanticEventId`] is
//!    CPU observation order and lane order plus explicit dependencies are the
//!    only other sources (§53.3).
//! 2. **A `PortableCommand` discriminant is not an opcode.** Section 56 is
//!    explicit that a Rust enum discriminant or memory layout must never be
//!    written directly as an artifact opcode: the Artifact Layer must supply its
//!    own tagged, versioned, bounds-checked, canonical encoding. Nothing in this
//!    file is `repr`, and nothing may be made `repr` for an encoding's
//!    convenience.

use core::ops::Range;

use crate::api::binding::BindGroupIndex;
use crate::api::command::{AccessMask, PipelineScope, TextureUseIntent};
use crate::api::command::{Color, IndexFormat, Rect, Viewport};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::presentation::AcquiredFrameId;
use crate::api::resource::buffer::BufferRange;
use crate::api::resource::subresource::TextureSubresourceRange;
use crate::api::submission::LaneWorkDomains;

use super::mutation::{
    CapturedBlit, CapturedBufferCopy, CapturedBufferTextureCopy, CapturedRasterScope,
    CapturedResolve, CapturedTextureCopy,
};

/// One command of a recording, in portable semantic terms.
///
/// The full vocabulary of what a recorder can have recorded. It is deliberately
/// *not* a mirror of a graphics API's command set: there is no descriptor-set
/// binding, no resource-barrier command, no root-signature command, and no
/// pipeline-cache command, because none of those is portable and none of them is
/// what a caller recorded. What a caller recorded is a pipeline, a group with its
/// dynamic offsets, buffers, viewport, scissor, and a draw.
///
/// # Why the draw carries instance and vertex ranges rather than counts
///
/// [`Self::Draw`] and [`Self::DrawIndexed`] carry [`Range`]s, because a draw's
/// range is a range rather than a count once `base_vertex` and a non-zero first
/// vertex are both in play. A count would force a replay to reconstruct an
/// offset from two other fields and would lose the `first_vertex` case entirely.
#[non_exhaustive]
#[derive(Clone)]
pub enum PortableCommand {
    /// A raster scope opened, with its attachment set.
    BeginRaster(CapturedRasterScope),

    /// The raster scope closed.
    EndRaster,

    /// The raster pipeline to draw with.
    SetRasterPipeline(ObjectId),

    /// The compute pipeline to dispatch with.
    SetComputePipeline(ObjectId),

    /// A bind group was bound at a group index.
    SetBindGroup {
        /// Which group index, as the interface numbers it.
        index: BindGroupIndex,
        /// The group's identity.
        group: ObjectId,
        /// The dynamic offsets consumed, in the order section 21.3 fixes.
        dynamic_offsets: Vec<u32>,
    },

    /// A vertex buffer was bound at a slot.
    SetVertexBuffer {
        /// The vertex buffer slot.
        slot: u32,
        /// The buffer's identity.
        buffer: ObjectId,
        /// The range of it that is bound.
        range: BufferRange,
    },

    /// An index buffer was bound.
    SetIndexBuffer {
        /// The buffer's identity.
        buffer: ObjectId,
        /// The range of it that is bound.
        range: BufferRange,
        /// The element type indices are cut with.
        format: IndexFormat,
    },

    /// The viewport was set.
    SetViewport(Viewport),

    /// The scissor rect was set.
    SetScissor(Rect),

    /// The blend constant was set.
    SetBlendConstant(Color),

    /// The stencil reference value was set.
    SetStencilReference(u32),

    /// A non-indexed draw.
    Draw {
        /// Which vertices to draw.
        vertices: Range<u32>,
        /// Which instances to draw them as.
        instances: Range<u32>,
    },

    /// An indexed draw.
    DrawIndexed {
        /// Which indices to draw.
        indices: Range<u32>,
        /// The value added to each index before it is fetched.
        base_vertex: i32,
        /// Which instances to draw them as.
        instances: Range<u32>,
    },

    /// A compute scope opened.
    BeginCompute {
        /// The scope's diagnostic label.
        label: Label,
    },

    /// The compute scope closed.
    EndCompute,

    /// A dispatch.
    Dispatch {
        /// Workgroups along x.
        x: u32,
        /// Workgroups along y.
        y: u32,
        /// Workgroups along z.
        z: u32,
    },

    /// An upload job was encoded into this recording.
    Upload {
        /// The upload job's identity.
        upload: ObjectId,
    },

    /// A readback was encoded into this recording.
    Readback {
        /// The readback ticket's identity.
        ticket: ObjectId,
    },

    /// A buffer-to-buffer copy.
    CopyBuffer(CapturedBufferCopy),

    /// A buffer-to-texture copy.
    CopyBufferToTexture(CapturedBufferTextureCopy),

    /// A texture-to-buffer copy.
    CopyTextureToBuffer(CapturedBufferTextureCopy),

    /// A texture-to-texture copy.
    CopyTexture(CapturedTextureCopy),

    /// A multisampled resolve.
    Resolve(CapturedResolve),

    /// A filtered blit.
    Blit(CapturedBlit),

    /// A debug group was opened.
    PushDebugGroup(String),

    /// A debug group was closed.
    PopDebugGroup,

    /// A debug marker was inserted.
    DebugMarker(String),
}

/// One captured command with the resource uses it produced.
///
/// The pairing is section 37.1's, carried through to the capture side: a flat use
/// list cannot say which command produced a use, and "this draw consumed that
/// binding" is exactly the question a diagnostics pass asks.
///
/// A command that reads nothing in particular — a viewport change, a debug
/// marker — has an empty `actual_uses`, because actual use is generated at draw
/// and dispatch rather than at the state-setting verbs.
#[derive(Clone)]
pub struct CapturedCommand {
    /// What the command was.
    pub command: PortableCommand,

    /// The actual uses this command produced.
    pub actual_uses: Vec<CapturedResourceUse>,
}

/// Everything a recording did, in portable semantic terms.
///
/// `merged_use_summary` and the per-command `actual_uses` are both present on
/// purpose, and are not redundant at two levels:
///
/// ```text
/// commands[].actual_uses   what each command consumed, in command order
/// merged_use_summary       what the recording as a whole touched, merged
/// ```
///
/// The merged summary is the live [`crate::api::command::RecordedWork`]'s own
/// public answer (`resource_uses`); the per-command lists are what a
/// *diagnostic* needs. A
/// capture that kept only the merged form could say that a texture was written
/// but not by which draw.
///
/// `domains` is [`LaneWorkDomains`] and not a lane: it is what the recording
/// contains, and it is the value a `SubmissionPlan` builder checks lanes against
/// (section 40.1). Recording it here is what lets a Replay decide which lane a
/// captured batch could run on without re-deriving it from the command list.
#[derive(Clone)]
pub struct CapturedRecordedWork {
    /// The recording's identity.
    pub work: ObjectId,

    /// The device every object in the recording belongs to.
    pub device: DeviceIdentity,

    /// Which execution domains the recording contains.
    pub domains: LaneWorkDomains,

    /// The commands, in recording order.
    pub commands: Vec<CapturedCommand>,

    /// The merged use summary the live recording reported.
    pub merged_use_summary: Vec<CapturedResourceUse>,
}

/// One resource use, named rather than held.
///
/// The tooling-side counterpart of
/// [`crate::api::command::ResourceUse`], and the difference is this chapter's
/// whole invariant: the live type holds `Buffer`, `Texture`, and `AcquiredFrameId`
/// — a live handle in two of the three arms — and this one holds [`ObjectId`]s.
/// Section 56 states the rule directly: a tooling-owned value may not retain a
/// live `Buffer` or `Texture` handle.
///
/// The stage and access fields reuse the RHI command vocabulary
/// ([`PipelineScope`], [`AccessMask`]) rather than creating tooling copies.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedResourceUse {
    /// A buffer was used.
    Buffer {
        /// The buffer's identity.
        buffer: ObjectId,
        /// The byte range used.
        range: BufferRange,
        /// The stages that used it.
        stages: PipelineScope,
        /// How they used it.
        access: AccessMask,
    },

    /// A texture was used.
    Texture {
        /// The texture's identity.
        texture: ObjectId,
        /// Which subresources were used.
        subresources: TextureSubresourceRange,
        /// The stages that used it.
        stages: PipelineScope,
        /// How they used it.
        access: AccessMask,
        /// What the use meant — attachment, sampled read, copy source, present.
        ///
        /// Carried in addition to `access` because the two answer different
        /// questions: `access` is a hazard class, and the intent is the role the
        /// use played. A snapshot decision needs the role and cannot recover it
        /// from the hazard class alone.
        intent: TextureUseIntent,
    },

    /// An acquired frame was used.
    Frame {
        /// The frame's identity.
        frame: AcquiredFrameId,
        /// The stages that used it.
        stages: PipelineScope,
        /// How they used it.
        access: AccessMask,
    },
}
