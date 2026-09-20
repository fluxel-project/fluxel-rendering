//! Recorded work: the internal command sequence and the opaque
//! [`RecordedWork`] a recorder produces (specification section 38.1 and 38.2).
//!
//! This module owns two things:
//!
//! 1. **The internal command sequence.** What a recorder actually recorded, in
//!    command order, with each command carrying the resource uses it produced.
//!    Section 37.1 keeps that sequence internal *on purpose*: the merged
//!    summary is the public answer, and a caller may not read a
//!    [`ResourceUse`] as proof that a write covered the whole range (section
//!    37.4).
//! 2. **The mapped [`RecordedWork`] summary.** One merged use list, one domain
//!    set, and the identity of the recording.
//!
//! # What this module does not own
//!
//! - Whether a lane accepts the work. Section 40.1's
//!   `lane.domains().contains(work.work_domains())` is the submission plan's
//!   check; this module only reports the domains.
//! - Copy, upload, and readback *validation*. Those verbs live in
//!   [`crate::api::command::copy`]; this module only stores what they recorded.
//!
//! # Strong ownership (section 38.2)
//!
//! A `RecordedWork` holds the logical objects it needs for execution — buffers,
//! textures, views, bind groups, pipelines, the upload payload, and the readback
//! state — by clone. The requirement is section 38.2's: a caller that drops its
//! own handle after recording must not change what is submitted. Every payload
//! below therefore owns its handles rather than borrowing them, and `finish()`
//! consumes the recorder rather than lending from it.

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::attachment::{ColorAttachment, DepthStencilAttachment};
use crate::api::command::copy::{
    BufferCopy, BufferTextureCopy, TextureBlit, TextureCopy, TextureResolve,
};
use crate::api::command::geometry::{Color, Rect, Viewport};
use crate::api::command::{IndexFormat, ResourceUse};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::pipeline::{ComputePipeline, RasterPipeline};
use crate::api::resource::buffer::BufferBinding;
use crate::api::resource::transfer::{ReadbackTicket, UploadJob};
use crate::api::submission::LaneWorkDomains;

/// One recording command, with the uses it produced.
///
/// The pairing is the point: section 37.1's command-level sequence is what makes
/// "this draw consumed that binding" answerable, and a flat use list cannot say
/// which command produced a use or in what order.
///
/// This carried an `expect(dead_code)` until the DX12 command spine began
/// matching on [`Self::payload`] and merging [`Self::uses`]. Its stated reason
/// named two readers, and the first of them arriving is what expired it — which
/// is the difference between an expectation that came true and one that was
/// wrong. There is no feature gate here for the same reason there is none on
/// `RecordedWork`: the recorder itself reads both fields in every build.
pub(crate) struct RecordedCommand {
    /// What the command was.
    ///
    /// This field carried a feature-gated `expect(dead_code)` for one round, on
    /// the reasoning that only a backend reads it. That was true of the *lowering*
    /// and false of the submit path: `Device::submit` walks the recorded commands
    /// to bind each readback ticket to its completion point (section 41.5), and
    /// that walk is portable code compiled in every configuration. The gate is
    /// therefore gone rather than widened — the reader that arrived is not a
    /// backend at all.
    pub(crate) payload: RecordedPayload,
    /// The actual uses this command produced.
    ///
    /// A command that reads nothing in particular — `set_viewport`, a debug
    /// marker — has an empty list, because section 32.4 generates actual use at
    /// draw and dispatch rather than at the state-setting verbs.
    pub(crate) uses: Vec<ResourceUse>,
}

/// What a recorded command was.
///
/// Every variant exists so that a recording can be replayed in command order by a
/// lowering backend, which is the only reader; until that backend exists, the
/// payloads are written and never read.
#[expect(dead_code, reason = "lowered by a backend port that is not built")]
pub(crate) enum RecordedPayload {
    /// A raster scope began, with its attachment set.
    RasterBegin(RasterBegin),
    /// A draw inside the open raster scope.
    RasterDraw(Box<RasterDraw>),
    /// The raster scope ended.
    RasterEnd,
    /// A compute scope began.
    ComputeBegin(ComputeBegin),
    /// A dispatch inside the open compute scope.
    ComputeDispatch(Box<ComputeDispatch>),
    /// The compute scope ended.
    ComputeEnd,
    /// One copy-family command.
    Copy(CopyRecord),
    /// An upload job was encoded.
    Upload(UploadJob),
    /// A readback was encoded.
    Readback(ReadbackTicket),
    /// A debug group was opened.
    ///
    /// Recorded rather than kept only in the scope's own stack, because a backend
    /// lowers a debug group as native debug-utils markup and needs to see where it
    /// began relative to the commands inside it.
    DebugPush(Label),
    /// A debug group was closed.
    DebugPop,
    /// A debug marker was inserted.
    DebugMarker(Label),
}

/// A raster scope's beginning, as recorded.
#[expect(dead_code, reason = "lowered by a backend port that is not built")]
pub(crate) struct RasterBegin {
    /// The scope's diagnostic label.
    pub(crate) label: Label,
    /// The canonicalized attachment set, by location.
    pub(crate) colors: Vec<(u32, ColorAttachment)>,
    /// The depth/stencil attachment, if any.
    pub(crate) depth_stencil: Option<DepthStencilAttachment>,
}

/// One raster draw, with the state that was current when it was issued.
///
/// The whole state is copied into the command rather than read from the scope at
/// end-of-scope time, because a draw is lowered with the state that was bound
/// when it was recorded; a later `set_pipeline` must not retroactively change an
/// earlier draw.
#[expect(dead_code, reason = "lowered by a backend port that is not built")]
pub(crate) struct RasterDraw {
    /// The pipeline that was bound. Section 32.3 requires one.
    pub(crate) pipeline: RasterPipeline,
    /// The bind groups that were bound.
    pub(crate) groups: Vec<BoundGroup>,
    /// The vertex buffers that were bound, by slot.
    pub(crate) vertex_buffers: Vec<(u32, BufferBinding)>,
    /// The index buffer and format, for an indexed draw.
    pub(crate) index: Option<BoundIndexBuffer>,
    /// The viewport, or `None` for the portable default.
    pub(crate) viewport: Option<Viewport>,
    /// The scissor rect, or `None` for the portable default.
    pub(crate) scissor: Option<Rect>,
    /// The blend constant in force.
    pub(crate) blend_constant: Color,
    /// The stencil reference in force.
    pub(crate) stencil_reference: u32,
    /// The vertex or index range drawn.
    pub(crate) range: core::ops::Range<u32>,
    /// The instance range drawn.
    pub(crate) instances: core::ops::Range<u32>,
    /// The base vertex of an indexed draw, or `0`.
    pub(crate) base_vertex: i32,
}

/// A compute scope's beginning, as recorded.
#[expect(dead_code, reason = "lowered by a backend port that is not built")]
pub(crate) struct ComputeBegin {
    /// The scope's diagnostic label.
    pub(crate) label: Label,
}

/// One dispatch, with the state that was current when it was issued.
#[expect(dead_code, reason = "lowered by a backend port that is not built")]
pub(crate) struct ComputeDispatch {
    /// The pipeline that was bound. Section 33 requires one.
    pub(crate) pipeline: ComputePipeline,
    /// The bind groups that were bound.
    pub(crate) groups: Vec<BoundGroup>,
    /// The workgroup counts.
    pub(crate) workgroups: (u32, u32, u32),
}

/// A bind group bound at an index, with its dynamic offsets.
#[derive(Clone)]
pub(crate) struct BoundGroup {
    /// The slot it was bound to.
    pub(crate) index: BindGroupIndex,
    /// The group itself.
    pub(crate) group: BindGroup,
    /// The dynamic offsets consumed by this binding.
    pub(crate) dynamic_offsets: Vec<u32>,
}

/// An index buffer bound with the format it is cut with.
#[derive(Clone)]
pub(crate) struct BoundIndexBuffer {
    /// The buffer and the range of it that is bound.
    pub(crate) binding: BufferBinding,
    /// The index element type.
    pub(crate) format: IndexFormat,
}

/// One copy-family command, as recorded.
pub(crate) enum CopyRecord {
    /// Buffer to buffer.
    Buffer(BufferCopy),
    /// Buffer to texture.
    BufferToTexture(BufferTextureCopy),
    /// Texture to buffer.
    TextureToBuffer(BufferTextureCopy),
    /// Texture to texture.
    Texture(TextureCopy),
    /// Multisampled to single-sampled resolve.
    Resolve(TextureResolve),
    /// Filtered blit.
    Blit(TextureBlit),
}

/// The result of a successful [`crate::api::command::CommandRecorder::finish`].
///
/// Opaque, and single-device by construction: every object that entered the
/// recording was identity-checked against the recorder's device, so a
/// `RecordedWork` cannot name more than one device and section 3.3's
/// cross-device rule has nothing left to refuse.
///
/// The two accessors that carry the interesting answers are
/// [`RecordedWork::work_domains`] and [`RecordedWork::resource_uses`]. The first
/// decides which lanes may accept the work (section 40.1); the second is the
/// merged actual-use summary consumed by submission validation and tooling.
/// Neither is a correctness proof of its own: section 37.4 is
/// explicit that a `SHADER_WRITE` use does not claim the shader filled the
/// range, so a caller may read coverage here only as a hazard statement.
pub struct RecordedWork {
    /// Process-local identity.
    id: ObjectId,
    /// The device every recorded object belongs to.
    device: DeviceIdentity,
    /// Which execution domains the recording actually contains.
    domains: LaneWorkDomains,
    /// The merged actual-use summary.
    uses: Vec<ResourceUse>,
    /// The command-ordered sequence, with per-command uses.
    commands: Vec<RecordedCommand>,
}

impl RecordedWork {
    /// Assembles the result of a finished recording.
    ///
    /// Crate-private for the reason every other constructor in this tree is:
    /// only a `CommandRecorder` that has just finished a recording can state
    /// these facts, and a caller-built `RecordedWork` would describe work that
    /// does not exist.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        domains: LaneWorkDomains,
        uses: Vec<ResourceUse>,
        commands: Vec<RecordedCommand>,
    ) -> Self {
        Self {
            id,
            device,
            domains,
            uses,
            commands,
        }
    }

    /// The process-local identity of this recording.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device every recorded object belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// Execution domains actually contained.
    ///
    /// Reported, not granted: section 40.1 requires a lane to contain these
    /// domains, and section 38.1 turns each of the recording's commands into one
    /// of `RASTER`, `COMPUTE`, or `COPY`. Several bits may be set at once, which
    /// is the ordinary case — a frame's recording rasterizes, copies, and
    /// uploads.
    pub fn work_domains(&self) -> LaneWorkDomains {
        self.domains
    }

    /// Merged actual-use summary.
    pub fn resource_uses(&self) -> &[ResourceUse] {
        &self.uses
    }

    /// The command-ordered sequence, with each command's own uses.
    ///
    /// Crate-private: section 37.1 keeps the command-level sequence *internal*,
    /// and it is read by the declared-versus-actual check, which needs to know
    /// not only that a use happened but which command produced it.
    ///
    /// The reader that arrived first is the DX12 command spine, which walks this
    /// slice to lower each payload and refuses the plan rather than skipping one
    /// it cannot lower, and the readback walk in `Device::submit`, which is how a
    /// ticket learns which point of the plan covers it. The declared-versus-actual
    /// comparison named in the old expectation is still unbuilt, and it is now the
    /// only reader missing — which is why there is no gate here at all: the two
    /// readers that arrived are portable code, compiled in every configuration.
    pub(crate) fn commands(&self) -> &[RecordedCommand] {
        &self.commands
    }
}

impl core::fmt::Debug for RecordedWork {
    /// Prints the recording's portable identity and its size, not its contents.
    ///
    /// Hand-written rather than derived, per adjudication A16: a derived `Debug`
    /// would print every cloned device-side handle, which is the one thing the
    /// portable surface may not expose. What is printed is what a caller can
    /// already read — the identity, the device, the domains, and how many uses
    /// and commands were recorded.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RecordedWork")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("work_domains", &self.domains)
            .field("resource_uses", &self.uses.len())
            .field("commands", &self.commands.len())
            .finish_non_exhaustive()
    }
}
