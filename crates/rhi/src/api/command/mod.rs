//! Recording and actual resource uses (specification sections 29 through 38).
//!
//! This is the chapter that turns a caller's intent into a portable recording: a
//! [`CommandRecorder`] is a CPU-side command builder that accepts copy, raster,
//! compute, upload, and readback commands in sequence and produces a
//! [`RecordedWork`] carrying the commands it recorded, the domains they belong to,
//! and the resources they actually touched.
//!
//! ```text
//! Device::create_recorder        -> CommandRecorder          (29.1)
//! begin_raster / begin_compute   -> an open scope            (29.2, 32, 33)
//! copy / upload / readback       -> recorded, recorder still open   (34.6, 35)
//! finish                         -> RecordedWork             (38.1)
//! ```
//!
//! # What this chapter deliberately is not
//!
//! Section 29 states four of these outright, and they are the reason the type
//! exists at all rather than being a thin wrapper over a native encoder:
//!
//! ```text
//! not a native command list / encoder
//! does not accept Graph declared uses
//! does not expose barriers / transitions
//! does not bind a submission lane
//! ```
//!
//! A recorder records *what happened*, never what a caller promised would happen.
//! The declared-versus-actual comparison is
//! [`crate::api::graph_bridge::validate_recorded_work`], which reads this chapter's
//! output rather than being trusted by it (section 37.4), and barrier and lane
//! decisions belong to the submission plan (section 40).
//!
//! # Submodules
//!
//! The chapter is split by what each file can independently vary:
//!
//! ```text
//! geometry    clear values, rects, viewports, load/store ops      (30)
//! attachment  raster attachment sets and their invariants         (31)
//! raster      RasterScope and the draw verbs                      (32)
//! compute     ComputeScope and dispatch                           (33)
//! copy        copy/resolve/blit descriptors and validators        (34)
//! record      the internal command sequence and RecordedWork      (38)
//! uses        which resource a command actually touched           (34.6, 37)
//! ```
//!
//! `IndexFormat` is the one type that stays at this root rather than moving into
//! `geometry`, where section 30 groups it: it is named by
//! [`crate::api::pipeline::PrimitiveState::strip_index_format`] (section 25.1), and
//! the pipeline chapter's import reads `crate::api::command::IndexFormat`.
//!
//! The submodules are crate-visible rather than private, for the reason
//! `resource::transfer` states for its own two halves: the validators they own are
//! crate-private entry points, and a `pub(crate) use` of one would be an unused
//! import in a non-test build. A caller outside the crate sees the re-exports
//! below and nothing else.
//!
//! # The invariant this chapter enforces
//!
//! **A recording is a state machine, and the borrow checker is half of it.** A
//! scope holds the recorder mutably, so two scopes cannot be open at once and no
//! command can land outside a scope's interior. The other half is the phase field:
//! a recorder that a dropped scope left unclosed is *poisoned* rather than
//! repaired, because a recording with a hole in it cannot be lowered and pretending
//! otherwise would put a malformed command sequence in front of a driver.

pub(crate) mod attachment;
pub(crate) mod compute;
pub(crate) mod copy;
pub(crate) mod geometry;
pub(crate) mod raster;
pub(crate) mod record;
pub(crate) mod uses;

pub use attachment::{
    ColorAttachment, ColorAttachmentView, DepthAttachmentMode, DepthStencilAttachment,
    RasterScopeDescriptor, StencilAttachmentMode,
};
pub use compute::{ComputeScope, ComputeScopeDescriptor};
pub use copy::{
    BlitFilter, BufferCopy, BufferTextureCopy, TextureBlit, TextureCopy, TextureResolve,
};
pub use geometry::{Color, ColorClearValue, LoadOp, Rect, StoreOp, Viewport};
pub use raster::RasterScope;
pub use record::RecordedWork;

use std::sync::Arc;

use crate::api::capability::EnabledCapabilities;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::graph_bridge::{AccessMask, PipelineScope, ResourceUse};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::resource::buffer::BufferRange;
use crate::api::resource::transfer::readback::{
    validate_buffer_readback, validate_texture_readback,
};
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackTicket, UploadDescriptor, UploadJob,
};
use crate::api::submission::LaneWorkDomains;

use self::copy::{
    buffer_copy_route, buffer_texture_route, resolve_route, texture_copy_route,
    texture_to_buffer_route, validate_buffer_copy, validate_buffer_texture_copy,
    validate_texture_blit, validate_texture_copy, validate_texture_resolve,
};
use self::record::{CopyRecord, RecordedCommand, RecordedPayload};
use self::uses::copy_uses;

/// The index element type a strip topology is cut with.
///
/// Named by [`crate::api::pipeline::PrimitiveState::strip_index_format`], and
/// meaningful only there: it selects where a primitive-restart index is
/// recognised, which is a property of the strip topology rather than of any
/// individual index buffer. A pipeline that draws a triangle list has no value
/// for it, which is why the field is optional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexFormat {
    /// A 16-bit index, whose primitive-restart value is `0xffff`.
    Uint16,
    /// A 32-bit index, whose primitive-restart value is `0xffff_ffff`.
    Uint32,
}

/// Everything a caller states about a recorder before it exists.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RecorderDescriptor {
    /// Diagnostic label.
    pub label: Label,
}

impl RecorderDescriptor {
    /// States a recorder with no label.
    pub fn new() -> Self {
        Self {
            label: Label::default(),
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}

impl Default for RecorderDescriptor {
    /// The same unlabelled descriptor as [`RecorderDescriptor::new`].
    ///
    /// `Default` is added because an unlabelled recorder is a legal descriptor
    /// rather than an incomplete one, and clippy's `new_without_default` lint is
    /// right that the two should agree.
    fn default() -> Self {
        Self::new()
    }
}

/// Which part of section 29.2's state machine a recorder is in.
///
/// Internal rather than public: the open state is expressed to a caller by which
/// verbs compile, since a scope holds the recorder mutably and nothing else can
/// be called while it lives. This enum exists for the two cases the borrow cannot
/// state — a scope abandoned by `Drop`, and a `finish` that must refuse rather than
/// return a recording with a hole in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecorderPhase {
    /// No scope is open; every recorder verb is available.
    Open,
    /// A raster scope is open.
    RasterScopeOpen,
    /// A compute scope is open.
    ///
    /// Reached by `begin_compute` once the device's [`OptionalFeature::Compute`]
    /// is present; a device without it refuses before the phase moves.
    ///
    /// [`OptionalFeature::Compute`]: crate::api::platform::requirements::OptionalFeature::Compute
    ComputeScopeOpen,
    /// A failure left the recording unusable.
    Poisoned,
}

/// A CPU-side portable command builder.
///
/// Opaque, and single-device by construction: every object that enters a recording
/// is identity-checked against [`CommandRecorder::device_identity`] before it is
/// recorded, so a recording cannot name more than one device and section 3.3's
/// cross-device rule has nothing left for a later stage to refuse.
///
/// The recorder holds no native encoder. That is deliberate and is what makes
/// section 29.2's "Drop does not perform a backend finalize that may fail" true:
/// there is nothing to finalize, so a dropped scope can only mark the recording
/// unusable, never leave a half-written native command list behind.
///
/// What it does hold is the device's capability snapshot, and the choice of *that*
/// rather than a [`Device`] is the same rule read a second time. Four verbs here
/// need a device answer — whether compute is enabled, the workgroup ceiling, and
/// the route plus copy-layout alignment of each copy family — and every one of
/// those answers is already in the snapshot the device interned at construction.
/// Holding a whole `Device` would be more than those verbs need, and the excess is
/// exactly the handle that would let a recorder reach a backend and lower, which
/// the rule above forbids. So the type makes the rule unfollowable rather than
/// merely documented.
pub struct CommandRecorder {
    /// Process-local identity.
    id: ObjectId,
    /// The device every recorded object must belong to.
    device: DeviceIdentity,
    /// The capability snapshot of that device.
    ///
    /// Shared, not copied: `Device` builds it once (section 7.2 makes an enabled
    /// contract immutable) and this is a second handle on the same value, so a
    /// recorder can never answer a capability question differently from the device
    /// that produced it.
    capabilities: Arc<EnabledCapabilities>,
    /// The descriptor's label, kept for diagnostics and capture.
    label: Label,
    /// Which part of the state machine this recorder is in.
    phase: RecorderPhase,
    /// Why the recording was poisoned, if it was.
    poison_reason: Option<RhiError>,
    /// The command-ordered interior of the recording.
    commands: Vec<RecordedCommand>,
    /// The domains recorded so far.
    ///
    /// `None` until the first command, because [`LaneWorkDomains`] has no way to
    /// name the empty set from outside `api::submission`. Section 10.1 makes an
    /// empty domain set a *refused* value rather than a valid empty one, so
    /// [`CommandRecorder::finish`] turns `None` into a refusal instead of
    /// unwrapping it into a recording that contains nothing executable.
    domains: Option<LaneWorkDomains>,
    /// This recorder's own debug-group stack, independent of any scope's.
    debug_stack: Vec<String>,
}

impl CommandRecorder {
    /// Assembles a recorder.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only [`Device::create_recorder`] may produce one, and a caller-built
    /// recorder would describe a device that never agreed to record.
    ///
    /// `capabilities` is the device's own snapshot, passed rather than derived so
    /// that a recorder cannot be built against a different device's facts than the
    /// identity it carries.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        capabilities: Arc<EnabledCapabilities>,
        label: Label,
    ) -> Self {
        Self {
            id,
            device,
            capabilities,
            label,
            phase: RecorderPhase::Open,
            poison_reason: None,
            commands: Vec::new(),
            domains: None,
            debug_stack: Vec::new(),
        }
    }

    /// The device every object in this recording belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The device facts a portable verb may decide against.
    ///
    /// Crate-private, and deliberately the narrowest thing that works: the four
    /// verbs that need a device answer ask *this*, so a verb added later that wants
    /// a backend handle cannot get one from here. See the type's documentation.
    pub(crate) fn capabilities(&self) -> &EnabledCapabilities {
        &self.capabilities
    }

    /// Finishes the recording and produces the work it describes.
    ///
    /// Section 38.1's mapping: one [`RecordedWork`] carrying the command
    /// sequence, the merged actual-use summary, and the domains the recording
    /// contains. The recorder is consumed, which is section 38.2's strong
    /// ownership — a caller that drops its own handles after recording cannot
    /// change what is submitted, because nothing here borrows from the caller.
    ///
    /// Three refusals, all of them "the recording is not a recording":
    ///
    /// - a poisoned recorder returns the failure that poisoned it, never a
    ///   partial result (section 29.2's `Poisoned` row);
    /// - an unclosed debug group is refused, because section 36 requires
    ///   `Recorder::finish` to see an empty `Open` stack;
    /// - a recording with no command at all is refused, because section 10.1 makes
    ///   an empty domain set a value that "records nothing executable" rather than a
    ///   valid empty one.
    ///
    /// An open *scope* is not on that list, and the reason is the borrow checker:
    /// a scope holds this recorder mutably, so `finish` cannot be reached while one
    /// lives. The phase check below exists for the case a future refactor breaks
    /// that, and it reports rather than repairs.
    pub fn finish(mut self) -> RhiResult<RecordedWork> {
        if let Some(reason) = self.poison_reason.take() {
            return Err(reason);
        }
        if !self.debug_stack.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "finish() needs an empty debug-group stack, and this recorder still has {} \
                     group(s) open",
                    self.debug_stack.len()
                ),
            ));
        }
        if self.phase != RecorderPhase::Open {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "finish() was reached with a scope still open, which the recorder's borrow should \
                 have made impossible",
            ));
        }
        let Some(domains) = self.domains else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "this recording contains no command, so it names no execution domain and nothing \
                 executable",
            ));
        };

        let uses = self
            .commands
            .iter()
            .flat_map(|command| command.uses.iter().cloned())
            .collect();
        Ok(RecordedWork::new(
            self.id,
            self.device,
            domains,
            uses,
            self.commands,
        ))
    }

    /// Copies a byte range between two buffers.
    ///
    /// Validated against section 34.1's list, then against the one question that is
    /// not portable: whether the device supports the buffer-to-buffer route, and
    /// with what offset and size alignment. Section 34.6 permits this command only
    /// while no scope is open, which is the first thing checked.
    pub fn copy_buffer(&mut self, copy: &BufferCopy) -> RhiResult<()> {
        self.require_open("copy_buffer")?;
        validate_buffer_copy(copy, self.device)?;
        self.route_pending(
            buffer_copy_route(),
            "copy_buffer",
            CopyRecord::Buffer(copy.clone()),
            Alignment::Buffer {
                src_offset: copy.src_offset,
                dst_offset: copy.dst_offset,
                size: copy.size,
            },
        )
    }

    /// Copies a buffer region into a texture region.
    pub fn copy_buffer_to_texture(&mut self, copy: &BufferTextureCopy) -> RhiResult<()> {
        self.require_open("copy_buffer_to_texture")?;
        validate_buffer_texture_copy(copy, self.device, true)?;
        self.route_pending(
            buffer_texture_route(copy, true),
            "copy_buffer_to_texture",
            CopyRecord::BufferToTexture(copy.clone()),
            Alignment::Texel {
                buffer_offset: copy.buffer_offset,
                bytes_per_row: copy.bytes_per_row,
            },
        )
    }

    /// Copies a texture region into a buffer region.
    pub fn copy_texture_to_buffer(&mut self, copy: &BufferTextureCopy) -> RhiResult<()> {
        self.require_open("copy_texture_to_buffer")?;
        validate_buffer_texture_copy(copy, self.device, false)?;
        self.route_pending(
            buffer_texture_route(copy, false),
            "copy_texture_to_buffer",
            CopyRecord::TextureToBuffer(copy.clone()),
            Alignment::Texel {
                buffer_offset: copy.buffer_offset,
                bytes_per_row: copy.bytes_per_row,
            },
        )
    }

    /// Copies a texel region between two textures.
    pub fn copy_texture(&mut self, copy: &TextureCopy) -> RhiResult<()> {
        self.require_open("copy_texture")?;
        validate_texture_copy(copy, self.device)?;
        self.route_pending(
            texture_copy_route(copy),
            "copy_texture",
            CopyRecord::Texture(copy.clone()),
            Alignment::None,
        )
    }

    /// Resolves a multisampled texture into a single-sampled one.
    pub fn resolve_texture(&mut self, resolve: &TextureResolve) -> RhiResult<()> {
        self.require_open("resolve_texture")?;
        validate_texture_resolve(resolve, self.device)?;
        self.route_pending(
            resolve_route(resolve),
            "resolve_texture",
            CopyRecord::Resolve(resolve.clone()),
            Alignment::None,
        )
    }

    /// Blits a region of one texture into another, optionally filtering.
    ///
    /// Section 34.5 makes the filter a route question rather than a promise: a blit
    /// with filtering is a different native operation from a scaled copy on some
    /// backends and absent on others, so the route key carries the filter and this
    /// verb cannot decide the answer.
    pub fn blit_texture(&mut self, blit: &TextureBlit) -> RhiResult<()> {
        self.require_open("blit_texture")?;
        validate_texture_blit(blit, self.device)?;
        self.route_pending(
            self::copy::blit_route(blit),
            "blit_texture",
            CopyRecord::Blit(blit.clone()),
            Alignment::None,
        )
    }

    /// Encodes a prepared upload.
    ///
    /// Fully portable, and therefore fully implemented. The job was already
    /// validated when it was created (section 17.3's checks run in
    /// `Device::create_buffer_upload` and `create_texture_upload`, and section
    /// 17.2 makes a job repeatable rather than one-shot), so what is left is
    /// section 35's mapping: the destination becomes a `COPY_WRITE` use in the
    /// `COPY` scope, and the job is recorded. HOST_WRITE is deliberately *not*
    /// emitted — section 37 reserves the host bits for graph and tooling
    /// observation, and an upload's GPU-side effect is the copy.
    pub fn encode_upload(&mut self, upload: &UploadJob) -> RhiResult<()> {
        self.require_open("encode_upload")?;
        if upload.device_identity() != self.device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the upload job belongs to a different device than the recorder",
            ));
        }

        let uses = upload_uses(upload);
        self.record_command(
            RecordedPayload::Upload(upload.clone()),
            uses,
            LaneWorkDomains::COPY,
        );
        Ok(())
    }

    /// Encodes a readback request and returns the ticket that will report it.
    ///
    /// Section 18.1's list, in two halves. Section 3.1's O(1) identity step comes
    /// first, so a cross-device readback is reported as a cross-device readback
    /// rather than as whatever the checks after it would have said. The rest —
    /// `COPY_SRC` usage, the range or region, and the copy-layout alignment — is
    /// [`crate::api::resource::transfer`]'s, and its two validators take the
    /// device's [`BufferCopyLayoutLimits`] as a parameter because the same
    /// alignment rules govern an upload and a readback. That parameter is the
    /// reason this verb needed the device snapshot: a permissive stand-in would
    /// accept a range the device never agreed to, which section 4 forbids.
    ///
    /// The order is section 4's discipline ①: every portable rule runs before any
    /// device question, so a caller's mistake is always reported as the caller's
    /// mistake rather than as a device limitation. Only then is the route asked —
    /// and a device that reports none answers `Unsupported` rather than the
    /// `InvalidUsage` a missing alignment would otherwise produce, which is the
    /// distinction section 9.4 draws between "this device cannot" and "you
    /// described it wrongly".
    ///
    /// [`BufferCopyLayoutLimits`]: crate::api::resource::route::BufferCopyLayoutLimits
    pub fn encode_readback(&mut self, request: ReadbackRequest) -> RhiResult<ReadbackTicket> {
        self.require_open("encode_readback")?;

        // Section 3.1's O(1) step, and the only part of readback validation that is
        // decidable without a device answer.
        match &request {
            ReadbackRequest::Buffer { src, .. } => {
                require_readback_device(src.device_identity(), self.device, "the readback buffer")?;
            }
            ReadbackRequest::Texture { src, .. } => {
                require_readback_device(
                    src.device_identity(),
                    self.device,
                    "the readback texture",
                )?;
            }
        }

        let uses = match &request {
            ReadbackRequest::Buffer { src, range, .. } => {
                validate_buffer_readback(src, *range, self.device)?;
                let limits = self.copy_layout_limits(buffer_copy_route(), "encode_readback")?;
                limits
                    .validate(range.offset, range.size)
                    .map_err(|e| e.at("encode_readback"))?;
                vec![ResourceUse::Buffer(crate::api::graph_bridge::BufferUse {
                    buffer: src.clone(),
                    range: *range,
                    stages: PipelineScope::COPY,
                    access: AccessMask::COPY_READ,
                })]
            }
            ReadbackRequest::Texture {
                src,
                subresource,
                origin,
                extent,
                ..
            } => {
                validate_texture_readback(src, *subresource, *origin, *extent, self.device)?;
                self.require_route(texture_to_buffer_route(src, subresource), "encode_readback")?;
                vec![ResourceUse::Texture(crate::api::graph_bridge::TextureUse {
                    texture: src.clone(),
                    subresources: crate::api::resource::subresource::TextureSubresourceRange {
                        aspects: crate::api::resource::subresource::aspect_bits(subresource.aspect),
                        base_mip: subresource.mip_level,
                        mip_count: 1,
                        base_layer: subresource.base_layer,
                        layer_count: subresource.layer_count,
                    },
                    stages: PipelineScope::COPY,
                    access: AccessMask::COPY_READ,
                    intent: crate::api::graph_bridge::TextureUseIntent::CopySrc,
                })]
            }
        };

        // The ticket is minted with this device's identity, so a caller that loses
        // the recorder still holds something that reports its own state. It is
        // recorded as well, because a readback is GPU work in the recording's
        // command order: a lowering backend has to see where the read of this
        // source happens relative to everything that wrote it.
        let ticket = ReadbackTicket::new(ObjectId::next(), self.device, request);
        self.record_command(
            RecordedPayload::Readback(ticket.clone()),
            uses,
            LaneWorkDomains::COPY,
        );
        Ok(ticket)
    }

    /// The copy-layout alignment a route imposes, or a refusal.
    ///
    /// A route that exists but reports no buffer-copy layout is not a device that
    /// imposes no alignment — it is a device answering that this route is not this
    /// kind of copy at all, which is what
    /// [`RouteCapabilities`](crate::api::resource::route::RouteCapabilities)'s own
    /// documentation says the `None` means. Treating it as "no constraint" would
    /// invert the answer, so it is a refusal in the same shape as a missing route.
    fn copy_layout_limits(
        &self,
        route: crate::api::resource::route::RouteQuery,
        what: &'static str,
    ) -> RhiResult<crate::api::resource::route::BufferCopyLayoutLimits> {
        self.capabilities()
            .route(&route)
            .capabilities()
            .and_then(|capabilities| capabilities.buffer_copy_layout())
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    format!(
                        "this device reports no buffer copy route with a copy layout, so it \
                         states no alignment a {what} could satisfy"
                    ),
                )
                .at(what)
            })
    }

    /// The texel-copy alignment a route imposes, or a refusal.
    ///
    /// The counterpart of [`Self::copy_layout_limits`] for the buffer/texture
    /// routes, and it refuses for the same reason: a route reporting no texel
    /// layout is stating that it is not a buffer/texture copy, which is not the
    /// same answer as "no alignment is imposed".
    fn texel_copy_layout_limits(
        &self,
        route: crate::api::resource::route::RouteQuery,
        what: &'static str,
    ) -> RhiResult<crate::api::resource::route::TexelCopyLayoutLimits> {
        self.capabilities()
            .route(&route)
            .capabilities()
            .and_then(|capabilities| capabilities.texel_copy_layout())
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    format!(
                        "this device reports no buffer/texture route with a texel copy layout, so \
                         it states no alignment a {what} could satisfy"
                    ),
                )
                .at(what)
            })
    }

    /// Refuses a route the device does not report.
    ///
    /// Section 9.4: a route that does not exist is `Unsupported`, and no
    /// substituted path may be recorded in its place.
    fn require_route(
        &self,
        route: crate::api::resource::route::RouteQuery,
        what: &'static str,
    ) -> RhiResult<()> {
        if self.capabilities().route(&route).is_supported() {
            return Ok(());
        }
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!(
                "this device reports no direct route for this {what}, and section 9.4 forbids \
                 substituting one, so it cannot be recorded"
            ),
        )
        .at(what))
    }

    /// Pushes a label onto the recorder's own debug-group stack.
    ///
    /// Section 36 gives the recorder, a raster scope, and a compute scope three
    /// independent stacks. This one describes the commands *outside* any pass; a
    /// stack pushed here is invisible to a scope's stack, and vice versa, because
    /// P0 debug groups do not span the boundary.
    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.require_open("push_debug_group")?;
        self.debug_stack.push(label.to_owned());
        self.record_command(
            RecordedPayload::DebugPush(Label(Some(label.to_owned()))),
            Vec::new(),
            LaneWorkDomains::COPY,
        );
        Ok(())
    }

    /// Pops the recorder's own debug-group stack.
    pub fn pop_debug_group(&mut self) -> RhiResult<()> {
        self.require_open("pop_debug_group")?;
        if self.debug_stack.pop().is_none() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "pop_debug_group has no matching push_debug_group on this recorder",
            ));
        }
        self.record_command(RecordedPayload::DebugPop, Vec::new(), LaneWorkDomains::COPY);
        Ok(())
    }

    /// Inserts a marker without changing the stack.
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.require_open("insert_debug_marker")?;
        self.record_command(
            RecordedPayload::DebugMarker(Label(Some(label.to_owned()))),
            Vec::new(),
            LaneWorkDomains::COPY,
        );
        Ok(())
    }

    /// Refuses a recorder verb that requires no scope to be open.
    pub(crate) fn require_open(&self, what: &'static str) -> RhiResult<()> {
        match self.phase {
            RecorderPhase::Open => Ok(()),
            RecorderPhase::RasterScopeOpen | RecorderPhase::ComputeScopeOpen => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("{what} can only be recorded while no scope is open"),
            )),
            RecorderPhase::Poisoned => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("{what} cannot be recorded because this recorder is poisoned"),
            )),
        }
    }

    /// Moves the recorder to another phase of section 29.2's state machine.
    pub(crate) fn set_phase(&mut self, phase: RecorderPhase) {
        self.phase = phase;
    }

    /// Marks the recording unusable, keeping the first reason.
    ///
    /// Section 29.3 makes a backend recording failure, a scope-finalization
    /// failure, and an internal-invariant failure poisoning rather than
    /// per-command. The *first* reason is kept because it is the cause: a scope
    /// dropped after a failed `end()` reports the dropped scope, and reporting that
    /// instead of whatever came first would hide the mistake that started it.
    pub(crate) fn poison(&mut self, reason: &str) {
        if self.phase == RecorderPhase::Poisoned {
            return;
        }
        self.phase = RecorderPhase::Poisoned;
        self.poison_reason = Some(RhiError::new(RhiErrorKind::InvalidUsage, reason));
    }

    /// Appends one command, its uses, and its execution domain.
    pub(crate) fn record_command(
        &mut self,
        payload: RecordedPayload,
        uses: Vec<ResourceUse>,
        domain: LaneWorkDomains,
    ) {
        self.domains = Some(match self.domains {
            Some(existing) => existing.union(domain),
            None => domain,
        });
        self.commands.push(RecordedCommand { payload, uses });
    }

    /// Answers the device half of a copy-family verb's rule list, then records it.
    ///
    /// Every copy verb does its portable validation and then reaches here, because
    /// the same two device facts gate all of them: whether the device supports the
    /// route, and what alignment that route's native copy accepts. Both come from
    /// the snapshot the recorder holds, so the answer is the device's own rather
    /// than one this layer assumed.
    ///
    /// Both refusals happen **before** the command is recorded, and that order is
    /// the contract rather than a preference. A refused copy must leave the
    /// recording exactly as it found it: recording first and refusing after would
    /// put a command the caller was told did not happen into a recording that
    /// `finish` would hand back as executable work.
    ///
    /// The route key is built and passed rather than derived here so that a reader
    /// sees exactly which question each verb asks.
    fn route_pending(
        &mut self,
        route: crate::api::resource::route::RouteQuery,
        what: &'static str,
        record: CopyRecord,
        alignment: Alignment,
    ) -> RhiResult<()> {
        self.require_route(route, what)?;

        // The alignment half of section 12.4. A route that exists but reports no
        // layout for the kind of copy it was asked about is answering that this is
        // not that kind of copy on this device — which is a refusal, not a pass.
        match alignment {
            Alignment::None => {}
            Alignment::Buffer {
                src_offset,
                dst_offset,
                size,
            } => {
                let limits = self.copy_layout_limits(route, what)?;
                limits.validate(src_offset, size).map_err(|e| e.at(what))?;
                limits.validate(dst_offset, size).map_err(|e| e.at(what))?;
            }
            Alignment::Texel {
                buffer_offset,
                bytes_per_row,
            } => {
                let limits = self.texel_copy_layout_limits(route, what)?;
                limits
                    .validate(buffer_offset, bytes_per_row)
                    .map_err(|e| e.at(what))?;
            }
        }

        let uses = copy_uses(&record);
        self.record_command(RecordedPayload::Copy(record), uses, LaneWorkDomains::COPY);
        Ok(())
    }
}

/// What a copy verb must check against the route's own copy layout.
///
/// Three shapes rather than one, because the routes genuinely differ: a
/// buffer-to-buffer copy aligns two offsets and a size, a buffer/texture copy
/// aligns a buffer offset and a row stride, and a texture-to-texture copy has no
/// buffer layout to align against at all. Passing the numbers in rather than
/// extracting them here is what keeps this function from having to know each
/// descriptor's field layout — and the verb is the side that holds the descriptor.
enum Alignment {
    /// No buffer layout applies.
    None,
    /// A buffer-to-buffer copy: both ends' offsets, and the shared size.
    Buffer {
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    },
    /// A buffer/texture copy: where the buffer side starts, and its row stride.
    Texel {
        buffer_offset: u64,
        bytes_per_row: u32,
    },
}

impl core::fmt::Debug for CommandRecorder {
    /// Prints the recorder's portable identity and how far it got.
    ///
    /// Hand-written rather than derived (adjudication A16): a derived `Debug` would
    /// walk every recorded command and print every cloned device-side handle, which
    /// is the one thing the portable surface may not expose. What is printed is what
    /// a caller can already read — the identity, the device, the phase, and how many
    /// commands were recorded.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CommandRecorder")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("label", &self.label)
            .field("phase", &self.phase)
            .field("commands", &self.commands.len())
            .field("work_domains", &self.domains)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Creates a recorder bound to this device.
    ///
    /// Section 29.1's creation verb. It is an inherent method written in the
    /// recording chapter rather than in `api::platform`, because section 29.1
    /// declares it there and because the type it produces is this chapter's: the
    /// definition site is the owner, and moving it would put the fact in a module
    /// that does not describe it.
    ///
    /// Nothing portable happens before the stop. A `RecorderDescriptor` carries only
    /// a label, and the device's identity is already decided by the object this is
    /// called on, so there is no caller-supplied value here that could be wrong.
    ///
    /// No backend is reached, and that is not an omission: this chapter's recorder
    /// holds no native encoder (see its documentation), so there is no native
    /// command builder to create here and none is created. A backend is first
    /// reached where the recording is submitted, exactly as the submission chapter
    /// records — the same shape, one chapter earlier in the caller's hands.
    pub fn create_recorder(&self, desc: &RecorderDescriptor) -> RhiResult<CommandRecorder> {
        // Section 6.5 lists `Recorder` among the handles a lost device refuses.
        self.require_active()?;

        Ok(CommandRecorder::new(
            ObjectId::next(),
            self.identity(),
            self.capabilities_arc(),
            desc.label.clone(),
        ))
    }
}

/// Refuses a readback source from another device.
///
/// Section 3.1's O(1) identity step, taken before anything else so that a
/// cross-device readback is [`RhiErrorKind::WrongDevice`] rather than whatever a
/// later check would have happened to say.
fn require_readback_device(
    actual: DeviceIdentity,
    expected: DeviceIdentity,
    what: &'static str,
) -> RhiResult<()> {
    if actual != expected {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            format!("{what} belongs to a different device than the recorder"),
        ));
    }
    Ok(())
}

/// Refuses a resource from another device.
///
/// Section 3.1's O(1) identity step, and the one check every recorded command
/// takes before any other: a cross-device argument must be reported as
/// [`RhiErrorKind::WrongDevice`] rather than as whatever else happens to be wrong
/// with it. Shared by the copy verbs, the raster scope, and the compute scope so
/// that all three answer the same question the same way.
pub(crate) fn require_device(
    actual: DeviceIdentity,
    expected: DeviceIdentity,
    what: &'static str,
) -> RhiResult<()> {
    if actual != expected {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            format!("{what} belongs to a different device than the recorder"),
        ));
    }
    Ok(())
}

/// The uses an upload job produces.
///
/// Section 35's mapping, read off the job's own descriptor: the destination is
/// written through a copy, so the access is `COPY_WRITE` in the `COPY` scope. The
/// texture case names the job's own subresource selection, because a texture upload
/// writes exactly one mip level of one layer run.
fn upload_uses(upload: &UploadJob) -> Vec<ResourceUse> {
    match upload.descriptor() {
        UploadDescriptor::Buffer(buffer) => {
            vec![ResourceUse::Buffer(crate::api::graph_bridge::BufferUse {
                buffer: buffer.dst.clone(),
                range: BufferRange::new(buffer.dst_offset, buffer.bytes.len() as u64),
                stages: PipelineScope::COPY,
                access: AccessMask::COPY_WRITE,
            })]
        }
        UploadDescriptor::Texture(texture) => {
            let aspect = texture.subresource.aspect;
            vec![ResourceUse::Texture(crate::api::graph_bridge::TextureUse {
                texture: texture.dst.clone(),
                subresources: crate::api::resource::subresource::TextureSubresourceRange {
                    aspects: crate::api::resource::subresource::aspect_bits(aspect),
                    base_mip: texture.subresource.mip_level,
                    mip_count: 1,
                    base_layer: texture.subresource.base_layer,
                    layer_count: texture.subresource.layer_count,
                },
                stages: PipelineScope::COPY,
                access: AccessMask::COPY_WRITE,
                intent: crate::api::graph_bridge::TextureUseIntent::CopyDst,
            })]
        }
    }
}
