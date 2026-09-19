//! Command recording: scopes, portable logical state, and recorded work.
//!
//! This module owns rhi-design sections 29 and 32 through 38.
//!
//! # What it is
//!
//! A recorder is a single-device, single-thread mutable object that turns
//! validated portable commands into [`RecordedWork`]. Every command is checked
//! against the recorder's own portable state before a backend sees it, so the
//! same recording is accepted or refused identically on every backend.
//!
//! # What it deliberately does not own
//!
//! Barriers and transitions, fences and events, native command lists, a
//! user-selected native queue, implicit shader or CPU-copy fallback, and
//! scope-auto-end-on-`Drop`. A scope ends explicitly; dropping one without
//! ending it poisons the recorder rather than silently finalizing work a
//! backend might reject.
//!
//! # Strong ownership
//!
//! Every accepted command stores the logical objects it needs — buffers,
//! textures, views, bind groups, pipelines, upload payloads, and readback
//! tickets — so dropping a caller's handle after recording cannot break a later
//! submission.
//!
//! # Two kinds of failure
//!
//! A parameter or capability error rejects *one command* and leaves the
//! recorder usable. A backend failure, a scope-finalization failure, or a
//! broken internal invariant poisons the recorder permanently, because after
//! one of those there is no way to know what the native stream contains.

pub mod attachment;
pub mod values;

mod validate;

use super::binding::{BindGroup, BindGroupIndex};
use super::format::LaneWorkDomains;
use super::graph_bridge::{
    AttachmentLoad, RecordedAttachment, RecordedAttachmentResource, ResourceUse,
};
use super::pipeline::{ComputePipeline, RasterPipeline};
use super::capability::DeviceLimits;
use super::platform::{
    DeviceIdentity, Label, LimitKey, ObjectId, RhiError, RhiErrorKind, RhiResult,
};
use super::resource::{BufferBinding, BufferUsage, ReadbackRequest, ReadbackTicket, UploadJob};

pub use attachment::{
    ColorAttachment, ColorAttachmentView, DepthAttachmentMode, DepthStencilAttachment,
    RasterScopeDescriptor, StencilAttachmentMode,
};
pub use values::{
    BlitFilter, BufferCopy, BufferTextureCopy, ClearValueClass, Color, ColorClearValue, IndexFormat,
    LoadOp, Rect, StoreOp, TextureBlit, TextureCopy, TextureResolve, Viewport,
};

/// A recorder's description.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RecorderDescriptor {
    /// A diagnostic label.
    pub label: Label,
}

impl Default for RecorderDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl RecorderDescriptor {
    /// An unlabelled recorder.
    pub fn new() -> Self {
        Self {
            label: Label::none(),
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }
}

/// The device facts a recorder validates against.
///
/// A recorder captures these when it is created, so every later command is
/// judged against the device's state at that moment rather than against a value
/// a backend changed underneath it.
#[derive(Clone, Debug)]
pub(crate) struct RecorderLimits {
    /// `LimitKey::MaxColorAttachments`.
    pub max_color_attachments: Option<u32>,
    /// `LimitKey::MaxComputeWorkgroupsPerDimension`.
    pub max_compute_workgroups_per_dimension: Option<u64>,
    /// Whether the device enables the compute feature.
    pub compute_enabled: bool,
}

impl RecorderLimits {
    /// Reads the recorder-relevant facts out of a device's limits.
    pub(crate) fn from_limits(limits: &DeviceLimits, compute_enabled: bool) -> Self {
        Self {
            max_color_attachments: limits
                .get(LimitKey::MaxColorAttachments)
                .map(|value| value.min(u32::MAX as u64) as u32),
            max_compute_workgroups_per_dimension: limits
                .get(LimitKey::MaxComputeWorkgroupsPerDimension),
            compute_enabled,
        }
    }
}

/// Where a recorder is in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecorderState {
    Open,
    Raster,
    Compute,
    Poisoned,
}

/// The portable logical state of the currently open scope.
///
/// Every field resets when a scope begins. No state is inherited from the prior
/// scope, which is what makes a scope's meaning depend only on its own
/// commands.
#[derive(Clone, Debug, Default)]
struct ScopeState {
    descriptor: Option<RasterScopeDescriptor>,
    raster_pipeline: Option<RasterPipeline>,
    compute_pipeline: Option<ComputePipeline>,
    bind_groups: Vec<Option<BindGroup>>,
    dynamic_offsets: Vec<Vec<u32>>,
    vertex_buffers: Vec<Option<BufferBinding>>,
    index_buffer: Option<(BufferBinding, IndexFormat)>,
    viewport: Option<Viewport>,
    scissor: Option<Rect>,
    blend_constant: Color,
    stencil_reference: u32,
    debug_depth: u32,
}

impl ScopeState {
    /// Drops everything the previous scope established.
    fn reset(&mut self) {
        *self = Self {
            blend_constant: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            },
            ..Self::default()
        };
    }
}

/// One recorded command, holding the live handles it needs.
///
/// The list of these is what makes [`RecordedWork`] independent of the caller's
/// handles, and it is also the command-level use sequence a tooling capture
/// reads. It is deliberately not public: a direct-RHI caller expresses work
/// through the scope API, not by assembling commands.
#[derive(Clone, Debug)]
pub(crate) enum RecordedCommand {
    BeginRaster(Box<RasterScopeDescriptor>),
    EndRaster,
    BeginCompute(Label),
    EndCompute,
    SetRasterPipeline(RasterPipeline),
    SetComputePipeline(ComputePipeline),
    SetBindGroup {
        index: BindGroupIndex,
        group: BindGroup,
        dynamic_offsets: Vec<u32>,
    },
    SetVertexBuffer {
        slot: u32,
        binding: BufferBinding,
    },
    SetIndexBuffer {
        binding: BufferBinding,
        format: IndexFormat,
    },
    SetViewport(Viewport),
    SetScissor(Rect),
    SetBlendConstant(Color),
    SetStencilReference(u32),
    Draw {
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    },
    DrawIndexed {
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    EncodeUpload(UploadJob),
    EncodeReadback(Box<ReadbackTicket>),
    CopyBuffer(Box<BufferCopy>),
    CopyBufferToTexture(Box<BufferTextureCopy>),
    CopyTextureToBuffer(Box<BufferTextureCopy>),
    CopyTexture(Box<TextureCopy>),
    ResolveTexture(Box<TextureResolve>),
    BlitTexture(Box<TextureBlit>),
    PushDebugGroup(String),
    PopDebugGroup,
    DebugMarker(String),
}

/// The backend half of a [`CommandRecorder`].
///
/// The recorder façade performs every portable check first, so a backend only
/// ever receives commands that are already legal in portable terms. A backend
/// still reports its own failures, and any of them poisons the recorder.
pub(crate) trait RecorderBackend: Send + Sync + 'static {
    /// The device identity this recorder belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// Opens a raster scope after its attachment set has been validated.
    fn begin_raster(
        &mut self,
        desc: &RasterScopeDescriptor,
        geometry: attachment::AttachmentGeometry,
    ) -> RhiResult<()>;

    /// Closes the open raster scope.
    fn end_raster(&mut self) -> RhiResult<()>;

    /// Opens a compute scope.
    fn begin_compute(&mut self, label: &Label) -> RhiResult<()>;

    /// Closes the open compute scope.
    fn end_compute(&mut self) -> RhiResult<()>;

    /// Records one already-validated command.
    fn record(&mut self, command: &RecordedCommand) -> RhiResult<()>;

    /// Allocates the ticket that will carry a readback's result.
    ///
    /// The ticket is a portable object, but only a backend knows the deferred
    /// copy it will be filled from, so the backend mints it.
    fn encode_readback(&mut self, request: &ReadbackRequest) -> RhiResult<ReadbackTicket>;

    /// Finalizes the recording into backend-owned work.
    fn finish(self: Box<Self>) -> RhiResult<Box<dyn RecordedWorkBackend>>;
}

/// The backend half of a [`RecordedWork`].
pub(crate) trait RecordedWorkBackend: Send + Sync + 'static {
    /// This work's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this work belongs to.
    fn device_identity(&self) -> DeviceIdentity;
}

/// Records portable commands into a single-device [`RecordedWork`].
pub struct CommandRecorder {
    inner: Option<Box<dyn RecorderBackend>>,
    device: DeviceIdentity,
    limits: RecorderLimits,
    state: RecorderState,
    scope: ScopeState,
    commands: Vec<RecordedCommand>,
    /// Per-command actual uses, in the same order as `commands`.
    command_uses: Vec<Vec<ResourceUse>>,
    /// Every attachment the recording actually bound.
    attachments: Vec<RecordedAttachment>,
    /// The domains the recording actually contains.
    domains: LaneWorkDomains,
    /// The recorder's own debug-group stack depth.
    debug_depth: u32,
}

impl core::fmt::Debug for CommandRecorder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CommandRecorder")
            .field("device", &self.device)
            .field("state", &self.state)
            .field("commands", &self.commands.len())
            .finish_non_exhaustive()
    }
}

impl CommandRecorder {
    /// Creates a recorder over a device's backend.
    pub(crate) fn new(inner: Box<dyn RecorderBackend>, limits: RecorderLimits) -> Self {
        let device = inner.device_identity();
        Self {
            inner: Some(inner),
            device,
            limits,
            state: RecorderState::Open,
            scope: ScopeState::default(),
            commands: Vec::new(),
            command_uses: Vec::new(),
            attachments: Vec::new(),
            domains: LaneWorkDomains::from_bits(0),
            debug_depth: 0,
        }
    }

    /// The device identity this recorder belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// Whether the recorder has been poisoned and can no longer record.
    pub fn is_poisoned(&self) -> bool {
        self.state == RecorderState::Poisoned
    }

    /// Rejects one command without poisoning the recorder.
    fn reject(operation: &'static str, error: RhiError) -> RhiError {
        error.at(operation)
    }

    /// Checks that the recorder is in its open state.
    fn require_open(&self, operation: &'static str) -> RhiResult<()> {
        match self.state {
            RecorderState::Open => Ok(()),
            RecorderState::Poisoned => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the recorder is poisoned",
            )
            .at(operation)),
            _ => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "this command is only legal outside a scope",
            )
            .at(operation)),
        }
    }

    /// Checks that an object belongs to this recorder's device.
    fn check_device(&self, owner: DeviceIdentity, operation: &'static str) -> RhiResult<()> {
        if owner != self.device {
            return Err(RhiError::wrong_device("the object belongs to another device").at(operation));
        }
        Ok(())
    }

    /// Poison the recorder and report the failure that caused it.
    fn poisoned(&mut self, error: RhiError) -> RhiError {
        self.state = RecorderState::Poisoned;
        error
    }

    /// Records an already-validated command and its uses.
    fn accept(&mut self, command: RecordedCommand, uses: Vec<ResourceUse>) -> RhiResult<()> {
        if self.state == RecorderState::Poisoned {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the recorder is poisoned and cannot accept more commands",
            ));
        }
        let outcome = match self.inner.as_mut() {
            Some(backend) => backend.record(&command),
            None => {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the recorder was already finished",
                ));
            }
        };
        if let Err(error) = outcome {
            return Err(self.poisoned(error));
        }
        self.commands.push(command);
        self.command_uses.push(uses);
        Ok(())
    }

    /// Opens a raster scope with the given attachments.
    pub fn begin_raster<'a>(
        &'a mut self,
        desc: &RasterScopeDescriptor,
    ) -> RhiResult<RasterScope<'a>> {
        self.require_open("begin_raster")?;
        let geometry =
            attachment::validate_scope_attachments(desc, self.limits.max_color_attachments)
                .map_err(|error| Self::reject("begin_raster", error))?;
        for slot in desc.colors.iter().flatten() {
            self.check_device(slot.view.device_identity(), "begin_raster")?;
        }
        if let Some(depth_stencil) = &desc.depth_stencil {
            self.check_device(depth_stencil.view.device_identity(), "begin_raster")?;
        }

        let outcome = match self.inner.as_mut() {
            Some(backend) => backend.begin_raster(desc, geometry),
            None => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the recorder was already finished",
            )),
        };
        if let Err(error) = outcome {
            return Err(self.poisoned(error));
        }
        self.scope.reset();
        self.scope.descriptor = Some(desc.clone());
        self.state = RecorderState::Raster;
        self.domains = self.domains.union(LaneWorkDomains::RASTER);
        self.record_attachments(desc);
        self.commands
            .push(RecordedCommand::BeginRaster(Box::new(desc.clone())));
        self.command_uses.push(Vec::new());
        Ok(RasterScope {
            recorder: self,
            ended: false,
        })
    }

    /// Records the attachments a raster scope actually bound.
    fn record_attachments(&mut self, desc: &RasterScopeDescriptor) {
        for (index, slot) in desc.colors.iter().enumerate() {
            let Some(slot) = slot else {
                continue;
            };
            let (resource, format) = match &slot.view {
                ColorAttachmentView::Texture(view) => (
                    RecordedAttachmentResource::Texture(view.texture().clone()),
                    view.format(),
                ),
                ColorAttachmentView::Frame(frame) => (
                    RecordedAttachmentResource::Frame(frame.frame_id()),
                    frame.format(),
                ),
            };
            self.attachments.push(RecordedAttachment {
                location: Some(index as u32),
                resource,
                format,
                load: match slot.load {
                    LoadOp::Load => AttachmentLoad::Existing,
                    LoadOp::Clear(_) => AttachmentLoad::Cleared,
                },
                stores: slot.store == StoreOp::Store,
            });
        }
        if let Some(depth_stencil) = &desc.depth_stencil {
            let mut load = AttachmentLoad::Undefined;
            let mut stores = false;
            if let Some(DepthAttachmentMode::ReadWrite { load: mode, store }) = depth_stencil.depth {
                load = match mode {
                    LoadOp::Load => AttachmentLoad::Existing,
                    LoadOp::Clear(_) => AttachmentLoad::Cleared,
                };
                stores |= store == StoreOp::Store;
            }
            if let Some(StencilAttachmentMode::ReadWrite { load: mode, store }) =
                depth_stencil.stencil
            {
                if load == AttachmentLoad::Undefined {
                    load = match mode {
                        LoadOp::Load => AttachmentLoad::Existing,
                        LoadOp::Clear(_) => AttachmentLoad::Cleared,
                    };
                }
                stores |= store == StoreOp::Store;
            }
            self.attachments.push(RecordedAttachment {
                location: None,
                resource: RecordedAttachmentResource::Texture(depth_stencil.view.texture().clone()),
                format: depth_stencil.view.format(),
                load,
                stores,
            });
        }
    }

    /// Closes the open raster scope.
    fn end_raster(&mut self) -> RhiResult<()> {
        let outcome = match self.inner.as_mut() {
            Some(backend) => backend.end_raster(),
            None => Ok(()),
        };
        if let Err(error) = outcome {
            return Err(self.poisoned(error));
        }
        self.state = RecorderState::Open;
        self.scope.reset();
        self.commands.push(RecordedCommand::EndRaster);
        self.command_uses.push(Vec::new());
        Ok(())
    }

    /// Opens a compute scope.
    pub fn begin_compute(&mut self) -> RhiResult<ComputeScope<'_>> {
        self.require_open("begin_compute")?;
        if !self.limits.compute_enabled {
            return Err(RhiError::unsupported(
                "the device does not enable the compute feature",
            )
            .at("begin_compute"));
        }
        let label = Label::none();
        let outcome = match self.inner.as_mut() {
            Some(backend) => backend.begin_compute(&label),
            None => Ok(()),
        };
        if let Err(error) = outcome {
            return Err(self.poisoned(error));
        }
        self.scope.reset();
        self.state = RecorderState::Compute;
        self.domains = self.domains.union(LaneWorkDomains::COMPUTE);
        self.commands.push(RecordedCommand::BeginCompute(label));
        self.command_uses.push(Vec::new());
        Ok(ComputeScope {
            recorder: self,
            ended: false,
        })
    }

    /// Closes the open compute scope.
    fn end_compute(&mut self) -> RhiResult<()> {
        let outcome = match self.inner.as_mut() {
            Some(backend) => backend.end_compute(),
            None => Ok(()),
        };
        if let Err(error) = outcome {
            return Err(self.poisoned(error));
        }
        self.state = RecorderState::Open;
        self.scope.reset();
        self.commands.push(RecordedCommand::EndCompute);
        self.command_uses.push(Vec::new());
        Ok(())
    }

    /// Encodes a host-to-GPU upload.
    pub fn encode_upload(&mut self, upload: &UploadJob) -> RhiResult<()> {
        self.require_open("encode_upload")?;
        self.check_device(upload.device_identity(), "encode_upload")?;
        let uses = validate::upload_uses(upload).map_err(|e| Self::reject("encode_upload", e))?;
        self.domains = self.domains.union(LaneWorkDomains::COPY);
        self.accept(RecordedCommand::EncodeUpload(upload.clone()), uses)
    }

    /// Encodes a GPU-to-host readback.
    pub fn encode_readback(&mut self, request: ReadbackRequest) -> RhiResult<ReadbackTicket> {
        self.require_open("encode_readback")?;
        let uses =
            validate::readback_uses(&request).map_err(|e| Self::reject("encode_readback", e))?;
        for owner in validate::readback_devices(&request) {
            self.check_device(owner, "encode_readback")?;
        }
        let outcome = match self.inner.as_mut() {
            Some(backend) => backend.encode_readback(&request),
            None => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the recorder was already finished",
            )),
        };
        let ticket = match outcome {
            Ok(ticket) => ticket,
            Err(error) => return Err(self.poisoned(error)),
        };
        self.domains = self.domains.union(LaneWorkDomains::COPY);
        self.commands
            .push(RecordedCommand::EncodeReadback(Box::new(ticket.clone())));
        self.command_uses.push(uses);
        Ok(ticket)
    }

    /// Records a buffer-to-buffer copy.
    pub fn copy_buffer(&mut self, copy: &BufferCopy) -> RhiResult<()> {
        self.require_open("copy_buffer")?;
        let uses = validate::copy_buffer_uses(copy).map_err(|e| Self::reject("copy_buffer", e))?;
        self.check_device(copy.src.device_identity(), "copy_buffer")?;
        self.check_device(copy.dst.device_identity(), "copy_buffer")?;
        self.domains = self.domains.union(LaneWorkDomains::COPY);
        self.accept(RecordedCommand::CopyBuffer(Box::new(copy.clone())), uses)
    }

    /// Records a buffer-to-texture copy.
    pub fn copy_buffer_to_texture(&mut self, copy: &BufferTextureCopy) -> RhiResult<()> {
        self.require_open("copy_buffer_to_texture")?;
        let uses = validate::buffer_texture_uses(copy, false)
            .map_err(|e| Self::reject("copy_buffer_to_texture", e))?;
        self.check_device(copy.buffer.device_identity(), "copy_buffer_to_texture")?;
        self.check_device(copy.texture.device_identity(), "copy_buffer_to_texture")?;
        self.domains = self.domains.union(LaneWorkDomains::COPY);
        self.accept(
            RecordedCommand::CopyBufferToTexture(Box::new(copy.clone())),
            uses,
        )
    }

    /// Records a texture-to-buffer copy.
    pub fn copy_texture_to_buffer(&mut self, copy: &BufferTextureCopy) -> RhiResult<()> {
        self.require_open("copy_texture_to_buffer")?;
        let uses = validate::buffer_texture_uses(copy, true)
            .map_err(|e| Self::reject("copy_texture_to_buffer", e))?;
        self.check_device(copy.buffer.device_identity(), "copy_texture_to_buffer")?;
        self.check_device(copy.texture.device_identity(), "copy_texture_to_buffer")?;
        self.domains = self.domains.union(LaneWorkDomains::COPY);
        self.accept(
            RecordedCommand::CopyTextureToBuffer(Box::new(copy.clone())),
            uses,
        )
    }

    /// Records a texture-to-texture copy.
    pub fn copy_texture(&mut self, copy: &TextureCopy) -> RhiResult<()> {
        self.require_open("copy_texture")?;
        let uses = validate::texture_copy_uses(copy).map_err(|e| Self::reject("copy_texture", e))?;
        self.check_device(copy.src.device_identity(), "copy_texture")?;
        self.check_device(copy.dst.device_identity(), "copy_texture")?;
        self.domains = self.domains.union(LaneWorkDomains::COPY);
        self.accept(RecordedCommand::CopyTexture(Box::new(copy.clone())), uses)
    }

    /// Records a multisample resolve.
    pub fn resolve_texture(&mut self, resolve: &TextureResolve) -> RhiResult<()> {
        self.require_open("resolve_texture")?;
        let uses = validate::resolve_uses(resolve).map_err(|e| Self::reject("resolve_texture", e))?;
        self.check_device(resolve.src.device_identity(), "resolve_texture")?;
        self.check_device(resolve.dst.device_identity(), "resolve_texture")?;
        self.domains = self.domains.union(LaneWorkDomains::COPY);
        self.accept(RecordedCommand::ResolveTexture(Box::new(resolve.clone())), uses)
    }

    /// Records a filtered or unfiltered blit.
    pub fn blit_texture(&mut self, blit: &TextureBlit) -> RhiResult<()> {
        self.require_open("blit_texture")?;
        let uses = validate::blit_uses(blit).map_err(|e| Self::reject("blit_texture", e))?;
        self.check_device(blit.src.device_identity(), "blit_texture")?;
        self.check_device(blit.dst.device_identity(), "blit_texture")?;
        self.domains = self.domains.union(LaneWorkDomains::COPY);
        self.accept(RecordedCommand::BlitTexture(Box::new(blit.clone())), uses)
    }

    /// Opens a debug group on the recorder's own stack.
    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.debug_depth += 1;
        self.accept(RecordedCommand::PushDebugGroup(label.to_string()), Vec::new())
    }

    /// Closes the recorder's innermost debug group.
    pub fn pop_debug_group(&mut self) -> RhiResult<()> {
        if self.debug_depth == 0 {
            return Err(Self::reject(
                "pop_debug_group",
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "no debug group is open on the recorder",
                ),
            ));
        }
        self.debug_depth -= 1;
        self.accept(RecordedCommand::PopDebugGroup, Vec::new())
    }

    /// Inserts a one-shot debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.accept(RecordedCommand::DebugMarker(label.to_string()), Vec::new())
    }

    /// Finalizes the recording.
    pub fn finish(mut self) -> RhiResult<RecordedWork> {
        self.require_open("finish")?;
        if self.debug_depth != 0 {
            return Err(Self::reject(
                "finish",
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the recorder's debug-group stack is not empty",
                ),
            ));
        }
        let Some(backend) = self.inner.take() else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the recorder was already finished",
            ));
        };
        let outcome = backend.finish();
        let inner = match outcome {
            Ok(inner) => inner,
            Err(error) => return Err(self.poisoned(error)),
        };
        let uses = validate::merge_uses(&self.command_uses);
        Ok(RecordedWork {
            inner,
            device: self.device,
            domains: self.domains,
            uses,
            attachments: self.attachments,
            commands: self.commands,
        })
    }
}

/// An open raster scope.
///
/// The scope holds `&mut CommandRecorder`, so while it is alive the recorder
/// cannot issue a command of another kind. Ending it is explicit: a scope that
/// is dropped without `end()` poisons the recorder, because the native stream
/// may already contain a partially opened pass.
#[derive(Debug)]
pub struct RasterScope<'a> {
    recorder: &'a mut CommandRecorder,
    ended: bool,
}

impl Drop for RasterScope<'_> {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        if let Some(backend) = self.recorder.inner.as_mut() {
            let _ = backend.end_raster();
        }
        let _ = self.recorder.poisoned(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a raster scope was dropped without end()",
        ));
    }
}

impl RasterScope<'_> {
    /// Binds a raster pipeline.
    pub fn set_pipeline(&mut self, pipeline: &RasterPipeline) -> RhiResult<()> {
        const OP: &str = "set_pipeline";
        self.recorder.check_device(pipeline.device_identity(), OP)?;
        let scope_signature = self
            .recorder
            .scope
            .descriptor
            .as_ref()
            .map(validate::target_signature_of);
        if scope_signature.as_ref() != Some(pipeline.target_signature()) {
            return Err(CommandRecorder::reject(
                OP,
                RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    "the pipeline's attachment signature differs from the open scope's",
                ),
            ));
        }
        self.recorder.scope.raster_pipeline = Some(pipeline.clone());
        self.recorder
            .accept(RecordedCommand::SetRasterPipeline(pipeline.clone()), Vec::new())
    }

    /// Binds a bind group at an index.
    pub fn set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        set_bind_group(&mut self.recorder.scope, index, group, dynamic_offsets)?;
        let command = RecordedCommand::SetBindGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        self.recorder.accept(command, Vec::new())
    }

    /// Binds a vertex buffer at a slot.
    pub fn set_vertex_buffer(&mut self, slot: u32, binding: &BufferBinding) -> RhiResult<()> {
        const OP: &str = "set_vertex_buffer";
        self.recorder
            .check_device(binding.buffer.device_identity(), OP)?;
        if !binding
            .buffer
            .descriptor()
            .usage
            .contains(BufferUsage::VERTEX)
        {
            return Err(CommandRecorder::reject(
                OP,
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the buffer was not created with VERTEX usage",
                ),
            ));
        }
        let index = slot as usize;
        if self.recorder.scope.vertex_buffers.len() <= index {
            self.recorder.scope.vertex_buffers.resize(index + 1, None);
        }
        self.recorder.scope.vertex_buffers[index] = Some(binding.clone());
        let command = RecordedCommand::SetVertexBuffer {
            slot,
            binding: binding.clone(),
        };
        self.recorder.accept(command, Vec::new())
    }

    /// Binds an index buffer.
    pub fn set_index_buffer(
        &mut self,
        binding: &BufferBinding,
        format: IndexFormat,
    ) -> RhiResult<()> {
        const OP: &str = "set_index_buffer";
        self.recorder
            .check_device(binding.buffer.device_identity(), OP)?;
        if !binding.buffer.descriptor().usage.contains(BufferUsage::INDEX) {
            return Err(CommandRecorder::reject(
                OP,
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the buffer was not created with INDEX usage",
                ),
            ));
        }
        self.recorder.scope.index_buffer = Some((binding.clone(), format));
        let command = RecordedCommand::SetIndexBuffer {
            binding: binding.clone(),
            format,
        };
        self.recorder.accept(command, Vec::new())
    }

    /// Sets the viewport.
    pub fn set_viewport(&mut self, viewport: Viewport) -> RhiResult<()> {
        viewport
            .validate()
            .map_err(|error| CommandRecorder::reject("set_viewport", error))?;
        self.recorder.scope.viewport = Some(viewport);
        self.recorder
            .accept(RecordedCommand::SetViewport(viewport), Vec::new())
    }

    /// Sets the scissor rectangle.
    pub fn set_scissor(&mut self, rect: Rect) -> RhiResult<()> {
        let overflow = rect
            .right()
            .and_then(|_| rect.bottom())
            .map_err(|error| CommandRecorder::reject("set_scissor", error));
        overflow?;
        self.recorder.scope.scissor = Some(rect);
        self.recorder
            .accept(RecordedCommand::SetScissor(rect), Vec::new())
    }

    /// Sets the blend constant.
    pub fn set_blend_constant(&mut self, color: Color) -> RhiResult<()> {
        self.recorder.scope.blend_constant = color;
        self.recorder
            .accept(RecordedCommand::SetBlendConstant(color), Vec::new())
    }

    /// Sets the stencil reference value.
    pub fn set_stencil_reference(&mut self, value: u32) -> RhiResult<()> {
        self.recorder.scope.stencil_reference = value;
        self.recorder
            .accept(RecordedCommand::SetStencilReference(value), Vec::new())
    }

    /// Draws non-indexed geometry.
    pub fn draw(
        &mut self,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> RhiResult<()> {
        const OP: &str = "draw";
        validate::validate_draw_ranges(&vertices, &instances)
            .map_err(|error| CommandRecorder::reject(OP, error))?;
        let uses = validate::draw_uses(
            &self.recorder.scope,
            &vertices,
            &instances,
            None,
        )
        .map_err(|error| CommandRecorder::reject(OP, error))?;
        self.recorder.accept(
            RecordedCommand::Draw {
                vertices,
                instances,
            },
            uses,
        )
    }

    /// Draws indexed geometry.
    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> RhiResult<()> {
        const OP: &str = "draw_indexed";
        validate::validate_indexed_draw_ranges(&indices, &instances)
            .map_err(|error| CommandRecorder::reject(OP, error))?;
        let uses = validate::draw_uses(
            &self.recorder.scope,
            // An indexed draw fetches vertices through the index buffer, so the
            // vertex-fetch bound is the index range, not a vertex range.
            &indices,
            &instances,
            Some(&indices),
        )
        .map_err(|error| CommandRecorder::reject(OP, error))?;
        self.recorder.accept(
            RecordedCommand::DrawIndexed {
                indices,
                base_vertex,
                instances,
            },
            uses,
        )
    }

    /// Opens a debug group on this scope's own stack.
    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.recorder.scope.debug_depth += 1;
        self.recorder
            .accept(RecordedCommand::PushDebugGroup(label.to_string()), Vec::new())
    }

    /// Closes this scope's innermost debug group.
    pub fn pop_debug_group(&mut self) -> RhiResult<()> {
        if self.recorder.scope.debug_depth == 0 {
            return Err(CommandRecorder::reject(
                "pop_debug_group",
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "no debug group is open on this scope",
                ),
            ));
        }
        self.recorder.scope.debug_depth -= 1;
        self.recorder.accept(RecordedCommand::PopDebugGroup, Vec::new())
    }

    /// Inserts a one-shot debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.recorder
            .accept(RecordedCommand::DebugMarker(label.to_string()), Vec::new())
    }

    /// Ends the scope.
    pub fn end(mut self) -> RhiResult<()> {
        if self.recorder.scope.debug_depth != 0 {
            return Err(CommandRecorder::reject(
                "end",
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the scope's debug-group stack is not empty",
                ),
            ));
        }
        self.ended = true;
        self.recorder.end_raster()
    }
}

/// An open compute scope.
///
/// As with [`RasterScope`], the borrow prevents mixing command kinds, and
/// dropping the scope without `end()` poisons the recorder.
#[derive(Debug)]
pub struct ComputeScope<'a> {
    recorder: &'a mut CommandRecorder,
    ended: bool,
}

impl Drop for ComputeScope<'_> {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        if let Some(backend) = self.recorder.inner.as_mut() {
            let _ = backend.end_compute();
        }
        let _ = self.recorder.poisoned(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a compute scope was dropped without end()",
        ));
    }
}

impl ComputeScope<'_> {
    /// Binds a compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) -> RhiResult<()> {
        self.recorder
            .check_device(pipeline.device_identity(), "set_pipeline")?;
        self.recorder.scope.compute_pipeline = Some(pipeline.clone());
        self.recorder.accept(
            RecordedCommand::SetComputePipeline(pipeline.clone()),
            Vec::new(),
        )
    }

    /// Binds a bind group at an index.
    pub fn set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        set_bind_group(&mut self.recorder.scope, index, group, dynamic_offsets)?;
        let command = RecordedCommand::SetBindGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        self.recorder.accept(command, Vec::new())
    }

    /// Dispatches a workgroup grid.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> RhiResult<()> {
        const OP: &str = "dispatch";
        let uses = validate::dispatch_uses(
            &self.recorder.scope,
            x,
            y,
            z,
            self.recorder.limits.max_compute_workgroups_per_dimension,
        )
        .map_err(|error| CommandRecorder::reject(OP, error))?;
        self.recorder.accept(RecordedCommand::Dispatch { x, y, z }, uses)
    }

    /// Opens a debug group on this scope's own stack.
    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.recorder.scope.debug_depth += 1;
        self.recorder
            .accept(RecordedCommand::PushDebugGroup(label.to_string()), Vec::new())
    }

    /// Closes this scope's innermost debug group.
    pub fn pop_debug_group(&mut self) -> RhiResult<()> {
        if self.recorder.scope.debug_depth == 0 {
            return Err(CommandRecorder::reject(
                "pop_debug_group",
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "no debug group is open on this scope",
                ),
            ));
        }
        self.recorder.scope.debug_depth -= 1;
        self.recorder.accept(RecordedCommand::PopDebugGroup, Vec::new())
    }

    /// Inserts a one-shot debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.recorder
            .accept(RecordedCommand::DebugMarker(label.to_string()), Vec::new())
    }

    /// Ends the scope.
    pub fn end(mut self) -> RhiResult<()> {
        if self.recorder.scope.debug_depth != 0 {
            return Err(CommandRecorder::reject(
                "end",
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the scope's debug-group stack is not empty",
                ),
            ));
        }
        self.ended = true;
        self.recorder.end_compute()
    }
}

/// Validates and installs a bind group in a scope's portable state.
fn set_bind_group(
    scope: &mut ScopeState,
    index: BindGroupIndex,
    group: &BindGroup,
    dynamic_offsets: &[u32],
) -> RhiResult<()> {
    const OP: &str = "set_bind_group";
    let expected = group.layout().dynamic_offset_count() as usize;
    if dynamic_offsets.len() != expected {
        return Err(CommandRecorder::reject(
            OP,
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the group needs {expected} dynamic offsets but {} were supplied",
                    dynamic_offsets.len()
                ),
            ),
        ));
    }
    validate::check_dynamic_offsets(group, dynamic_offsets)
        .map_err(|error| CommandRecorder::reject(OP, error))?;

    let slot = index.get() as usize;
    if scope.bind_groups.len() <= slot {
        scope.bind_groups.resize(slot + 1, None);
        scope.dynamic_offsets.resize(slot + 1, Vec::new());
    }
    scope.bind_groups[slot] = Some(group.clone());
    scope.dynamic_offsets[slot] = dynamic_offsets.to_vec();
    Ok(())
}

/// A finalized, single-device recording.
pub struct RecordedWork {
    inner: Box<dyn RecordedWorkBackend>,
    device: DeviceIdentity,
    domains: LaneWorkDomains,
    uses: Vec<ResourceUse>,
    attachments: Vec<RecordedAttachment>,
    commands: Vec<RecordedCommand>,
}

impl core::fmt::Debug for RecordedWork {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RecordedWork")
            .field("id", &self.id())
            .field("domains", &self.domains)
            .field("uses", &self.uses.len())
            .finish_non_exhaustive()
    }
}

impl RecordedWork {
    /// This work's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this work belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The execution domains this work actually contains.
    pub fn work_domains(&self) -> LaneWorkDomains {
        self.domains
    }

    /// The merged actual-use summary.
    ///
    /// This is a summary, not the command sequence: it cannot be used to infer
    /// the internal synchronization of the work, which is why the recorder
    /// keeps the per-command uses as well.
    pub fn resource_uses(&self) -> &[ResourceUse] {
        &self.uses
    }

    /// Every attachment this recording actually bound.
    pub(crate) fn recorded_attachments(&self) -> &[RecordedAttachment] {
        &self.attachments
    }

    /// The command-level use sequence, for tooling and diagnostics.
    pub(crate) fn commands(&self) -> &[RecordedCommand] {
        &self.commands
    }
}
