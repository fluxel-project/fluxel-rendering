//! Deterministic Layer 1 recorder.  It models Fluxel-owned identities, never GL names.

use std::collections::{BTreeMap, BTreeSet};

use super::*;

/// Exact observable mock operations, deliberately expressed in domain vocabulary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MockCall {
    CreateBuffer(BufferId),
    DestroyBuffer(BufferId),
    CreateTexture(TextureId),
    DestroyTexture(TextureId),
    CreateSampler(SamplerId),
    DestroySampler(SamplerId),
    CreateShader(ShaderId),
    DestroyShader(ShaderId),
    CreateProgram(ProgramId),
    DestroyProgram(ProgramId),
    CreateVertexArray(VertexArrayId),
    DestroyVertexArray(VertexArrayId),
    BindVertexArray(VertexArrayId),
    CreateFramebuffer(FramebufferId),
    DestroyFramebuffer(FramebufferId),
    BeginRenderPass(FramebufferId),
    EndRenderPass,
    SetRasterPipeline {
        program: ProgramId,
        vertex_array: VertexArrayId,
    },
    DrawRaster(GlDrawCommand),
    CopyBuffer {
        source: BufferId,
        destination: BufferId,
        size: u64,
    },
    CopyTexture {
        source: TextureId,
        destination: TextureId,
    },
    UploadTexture(TextureId),
    ReadTexture(TextureId),
    CreateFence(GlFenceLease),
    DestroyFence(GlFenceLease),
    PollFence(GlFenceLease),
    WaitFence(GlFenceLease),
    Flush,
    AcquireSurface(GlSurfaceLease),
    ResizeSurface(GlSurfaceSize),
    SuspendSurface,
    ResumeSurface,
    PresentSurface(GlSurfaceLease),
    CreateQuery(QueryId),
    DestroyQuery(QueryId),
    QueryResult(QueryId),
    BeginOcclusion(QueryId),
    EndOcclusion,
    Dispatch(GlDispatchGroups),
    BindStorageBuffer {
        binding: u32,
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    BindStorageImage {
        binding: u32,
        texture: TextureId,
    },
    ContextLost,
    ContextRestored(ContextStamp),
    Error(GlError),
}

/// Common-profile mock. It intentionally does not implement compute or storage traits.
#[derive(Debug)]
pub struct MockGlFamilyApi {
    discovery: GlDiscoverySnapshot,
    stamp: ContextStamp,
    lifecycle: GlContextLifecycle,
    owner: OwnerThreadIdentity,
    next_slot: u32,
    buffers: BTreeMap<BufferId, GlBufferDesc>,
    textures: BTreeMap<TextureId, GlTextureDesc>,
    samplers: BTreeSet<SamplerId>,
    shaders: BTreeSet<ShaderId>,
    programs: BTreeSet<ProgramId>,
    vaos: BTreeSet<VertexArrayId>,
    framebuffers: BTreeMap<FramebufferId, GlFramebufferDescriptor>,
    queries: BTreeSet<QueryId>,
    fences: GlFenceLeaseBook,
    syncs: BTreeSet<SyncId>,
    surface: GlSurfaceLeaseBook,
    surface_size: GlSurfaceSize,
    surface_suspended: bool,
    pass_active: bool,
    pixel_store: GlPixelStoreState,
    calls: Vec<MockCall>,
    next_error: Option<GlError>,
}
impl MockGlFamilyApi {
    pub fn from_discovery(discovery: GlDiscoverySnapshot) -> Self {
        Self {
            stamp: discovery.context_stamp(),
            discovery,
            lifecycle: GlContextLifecycle::Active,
            owner: OwnerThreadIdentity::current(),
            next_slot: 0,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            samplers: BTreeSet::new(),
            shaders: BTreeSet::new(),
            programs: BTreeSet::new(),
            vaos: BTreeSet::new(),
            framebuffers: BTreeMap::new(),
            queries: BTreeSet::new(),
            fences: GlFenceLeaseBook::default(),
            syncs: BTreeSet::new(),
            surface: GlSurfaceLeaseBook::new(),
            surface_size: GlSurfaceSize {
                width: 1,
                height: 1,
            },
            surface_suspended: false,
            pass_active: false,
            pixel_store: GlPixelStoreState::DEFAULT,
            calls: vec![],
            next_error: None,
        }
    }
    pub fn try_with_compute_storage(self) -> Result<MockComputeStorageApi, GlError> {
        MockComputeStorageApi::new(self)
    }
    pub fn calls(&self) -> &[MockCall] {
        &self.calls
    }
    pub fn clear_calls(&mut self) {
        self.calls.clear()
    }
    pub fn fail_next(&mut self, error: GlError) {
        self.next_error = Some(error)
    }
    fn owner(&self, op: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        (actual == self.owner)
            .then_some(())
            .ok_or(GlError::WrongThread {
                operation: op,
                expected: self.owner,
                actual,
            })
    }
    fn ready(&mut self, op: &'static str) -> Result<(), GlError> {
        self.owner(op)?;
        if let Some(e) = self.next_error.take() {
            self.error(e.clone());
            return Err(e);
        }
        match self.lifecycle {
            GlContextLifecycle::Active => Ok(()),
            GlContextLifecycle::Lost | GlContextLifecycle::Restoring => {
                self.error_result(GlError::ContextLost { operation: op })
            }
            GlContextLifecycle::Disposed => self.error_result(GlError::Disposed { operation: op }),
            GlContextLifecycle::Poisoned => self.error_result(GlError::Poisoned { operation: op }),
            _ => self.invalid(op, "context is not active"),
        }
    }
    fn error_result<T>(&mut self, e: GlError) -> Result<T, GlError> {
        self.error(e.clone());
        Err(e)
    }
    fn invalid<T>(&mut self, op: &'static str, message: &'static str) -> Result<T, GlError> {
        self.error_result(GlError::Validation {
            operation: op,
            message: message.into(),
        })
    }
    fn error(&mut self, e: GlError) {
        if matches!(e, GlError::ContextLost { .. }) {
            self.lifecycle = GlContextLifecycle::Lost;
        }
        self.calls.push(MockCall::Error(e));
    }
    fn slot(&mut self) -> Result<u32, GlError> {
        let slot = self.next_slot;
        self.next_slot = self.next_slot.checked_add(1).ok_or(GlError::OutOfMemory {
            operation: "mock-slot",
        })?;
        Ok(slot)
    }
    fn stamp(&mut self, op: &'static str, actual: ContextStamp) -> Result<(), GlError> {
        if actual.device != self.stamp.device {
            self.error_result(GlError::WrongContext {
                operation: op,
                object: actual,
                current: self.stamp,
            })
        } else if actual != self.stamp {
            self.error_result(GlError::StaleObject {
                operation: op,
                object: actual,
                current: self.stamp,
            })
        } else {
            Ok(())
        }
    }
    fn buffer(&mut self, op: &'static str, id: BufferId) -> Result<GlBufferDesc, GlError> {
        self.stamp(op, id.context)?;
        match self.buffers.get(&id).copied() {
            Some(desc) => Ok(desc),
            None => self.invalid(op, "buffer is not live"),
        }
    }
    fn texture(&mut self, op: &'static str, id: TextureId) -> Result<GlTextureDesc, GlError> {
        self.stamp(op, id.context)?;
        match self.textures.get(&id).copied() {
            Some(desc) => Ok(desc),
            None => self.invalid(op, "texture is not live"),
        }
    }
    fn live<K: GlObjectKind, F: FnOnce(&Self) -> bool>(
        &mut self,
        op: &'static str,
        id: ObjectIdentity<K>,
        exists: F,
    ) -> Result<(), GlError> {
        self.stamp(op, id.context)?;
        if exists(self) {
            Ok(())
        } else {
            self.invalid(op, "object is not live")
        }
    }
    fn reset_objects(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        self.shaders.clear();
        self.programs.clear();
        self.vaos.clear();
        self.framebuffers.clear();
        self.queries.clear();
        self.syncs.clear();
        self.fences.revoke_all();
        self.pass_active = false;
        let _ = self.surface.invalidate_generation();
    }
    fn frame_view(&mut self, op: &'static str, view: GlTextureView) -> Result<(), GlError> {
        match view.target {
            GlAttachmentTarget::Texture(id) => {
                self.texture(op, id)?;
                Ok(())
            }
            GlAttachmentTarget::SurfaceImage(id) => self.stamp(op, id.context),
        }
    }
    fn format_facts(
        &mut self,
        op: &'static str,
        desc: GlTextureDesc,
    ) -> Result<GlFormatCapabilities, GlError> {
        self.discovery
            .formats()
            .get_for(
                GlFormatResourceKind::Texture,
                desc.format,
                desc.sample_count,
            )
            .ok_or_else(|| GlError::Validation {
                operation: op,
                message: "no exact discovered format fact for texture allocation".into(),
            })
            .inspect_err(|error| self.error(error.clone()))
    }
    fn validate_buffer_allocation(&mut self, desc: GlBufferDesc) -> Result<(), GlError> {
        let capabilities = self.discovery.capabilities();
        if desc.usage.contains(GlBufferUsage::STORAGE)
            && !capabilities.supports(GlCapability::StorageBuffer)
        {
            return self.invalid(
                "create-buffer",
                "storage buffer usage lacks proved storage-buffer capability",
            );
        }
        if desc.usage.contains(GlBufferUsage::INDIRECT)
            && !capabilities.supports(GlCapability::IndirectDraw)
            && !capabilities.supports(GlCapability::IndirectDispatch)
        {
            return self.invalid(
                "create-buffer",
                "indirect buffer usage lacks proved indirect capability",
            );
        }
        Ok(())
    }
    fn validate_texture_allocation(&mut self, desc: GlTextureDesc) -> Result<(), GlError> {
        let facts = self.format_facts("create-texture", desc)?;
        if desc.usage.contains(GlTextureUsage::SAMPLED) && !facts.sampled {
            return self.invalid(
                "create-texture",
                "format is not sampled for this sample count",
            );
        }
        if desc.usage.contains(GlTextureUsage::RENDER_ATTACHMENT) && !facts.renderable {
            return self.invalid(
                "create-texture",
                "format is not renderable for this sample count",
            );
        }
        if desc.usage.contains(GlTextureUsage::COPY_SOURCE) && !facts.copy_source {
            return self.invalid("create-texture", "format is not a copy source");
        }
        if desc.usage.contains(GlTextureUsage::COPY_DESTINATION) && !facts.copy_destination {
            return self.invalid("create-texture", "format is not a copy destination");
        }
        if desc.usage.contains(GlTextureUsage::STORAGE_BINDING)
            && (!self
                .discovery
                .capabilities()
                .supports(GlCapability::StorageImage)
                || (!facts.storage_read && !facts.storage_write))
        {
            return self.invalid(
                "create-texture",
                "storage texture usage lacks proved image capability or exact format access",
            );
        }
        Ok(())
    }
    fn validate_storage_image_binding(
        &mut self,
        binding: u32,
        image: GlStorageImageBinding,
    ) -> Result<(), GlError> {
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::StorageImage)
        {
            return self.invalid(
                "bind-storage-image",
                "discovery did not prove storage-image capability",
            );
        }
        let desc = self.texture("bind-storage-image", image.texture)?;
        image.validate(
            binding,
            GlStorageImageLimits {
                max_image_units: self.discovery.limits().max_image_units,
            },
            self.discovery.formats(),
        )?;
        if !desc.usage.contains(GlTextureUsage::STORAGE_BINDING)
            || desc.format != image.format
            || desc.sample_count != image.sample_count
            || image.level >= desc.mip_level_count
        {
            return self.invalid(
                "bind-storage-image",
                "texture usage, format, sample count, or mip level is invalid for storage",
            );
        }
        let Some(mip_extent) = desc.mip_extent(image.level) else {
            return self.invalid("bind-storage-image", "storage image mip level is invalid");
        };
        let supports_layered = matches!(
            desc.dimension,
            GlTextureDimension::D3 | GlTextureDimension::D2Array | GlTextureDimension::Cube
        );
        if (image.layered && !supports_layered)
            || (!image.layered
                && image
                    .layer
                    .is_none_or(|layer| layer >= mip_extent.depth_or_layers))
        {
            return self.invalid(
                "bind-storage-image",
                "storage image layer selection is invalid for the texture shape",
            );
        }
        Ok(())
    }
}
impl GlFamilyApi for MockGlFamilyApi {
    fn profile(&self) -> GlFamilyProfile {
        self.discovery.context().profile()
    }
    fn context_stamp(&self) -> ContextStamp {
        self.stamp
    }
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }
    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner
    }
    fn assert_owner_thread(&self, op: &'static str) -> Result<(), GlError> {
        self.owner(op)
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.discovery
    }
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.ready("context-lost")?;
        self.lifecycle = GlContextLifecycle::Lost;
        self.reset_objects();
        self.calls.push(MockCall::ContextLost);
        Ok(())
    }
    fn context_restored(&mut self) -> Result<ContextStamp, GlError> {
        self.owner("context-restored")?;
        if self.lifecycle != GlContextLifecycle::Lost {
            return self.invalid("context-restored", "context is not lost");
        }
        if let Some(e) = self.next_error.take() {
            self.error(e.clone());
            return Err(e);
        }
        let Some(epoch) = self.stamp.epoch.checked_next() else {
            return self.error_result(GlError::Driver {
                operation: "context-restored",
                message: "context epoch exhausted".into(),
            });
        };
        self.lifecycle = GlContextLifecycle::Restoring;
        self.stamp = ContextStamp::new(self.stamp.device, epoch);
        self.discovery = self.discovery.rebind_for_test(self.stamp);
        self.next_slot = 0;
        self.lifecycle = GlContextLifecycle::Active;
        self.calls.push(MockCall::ContextRestored(self.stamp));
        Ok(self.stamp)
    }
}
impl GlResourceApi for MockGlFamilyApi {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        self.ready("create-buffer")?;
        desc.validate().map_err(|_| GlError::Validation {
            operation: "create-buffer",
            message: "invalid buffer descriptor".into(),
        })?;
        self.validate_buffer_allocation(desc)?;
        let id = BufferId::new(self.stamp, self.slot()?, 0);
        self.buffers.insert(id, desc);
        self.calls.push(MockCall::CreateBuffer(id));
        Ok(id)
    }
    fn create_texture_resource(&mut self, desc: GlTextureDesc) -> Result<TextureId, GlError> {
        self.ready("create-texture")?;
        desc.validate().map_err(|_| GlError::Validation {
            operation: "create-texture",
            message: "invalid texture descriptor".into(),
        })?;
        self.validate_texture_allocation(desc)?;
        let id = TextureId::new(self.stamp, self.slot()?, 0);
        self.textures.insert(id, desc);
        self.calls.push(MockCall::CreateTexture(id));
        Ok(id)
    }
    fn destroy_buffer_resource(&mut self, id: BufferId) -> Result<(), GlError> {
        self.ready("destroy-buffer")?;
        self.buffer("destroy-buffer", id)?;
        self.buffers.remove(&id);
        self.calls.push(MockCall::DestroyBuffer(id));
        Ok(())
    }
    fn destroy_texture_resource(&mut self, id: TextureId) -> Result<(), GlError> {
        self.ready("destroy-texture")?;
        self.texture("destroy-texture", id)?;
        self.textures.remove(&id);
        self.calls.push(MockCall::DestroyTexture(id));
        Ok(())
    }
}
impl GlSamplerApi for MockGlFamilyApi {
    fn create_sampler(&mut self, d: GlSamplerDesc) -> Result<SamplerId, GlError> {
        self.ready("create-sampler")?;
        d.validate_for(&self.discovery)
            .map_err(|_| GlError::Validation {
                operation: "create-sampler",
                message: "invalid sampler".into(),
            })?;
        let id = SamplerId::new(self.stamp, self.slot()?, 0);
        self.samplers.insert(id);
        self.calls.push(MockCall::CreateSampler(id));
        Ok(id)
    }
    fn destroy_sampler(&mut self, id: SamplerId) -> Result<(), GlError> {
        self.ready("destroy-sampler")?;
        self.live("destroy-sampler", id, |this| this.samplers.contains(&id))?;
        self.samplers.remove(&id);
        self.calls.push(MockCall::DestroySampler(id));
        Ok(())
    }
}
impl GlShaderApi for MockGlFamilyApi {
    fn create_shader(&mut self, s: &GlShaderSource) -> Result<ShaderId, GlError> {
        self.ready("create-shader")?;
        s.validate_for(self.profile())
            .map_err(|_| GlError::Validation {
                operation: "create-shader",
                message: "shader source does not target this profile".into(),
            })?;
        let id = ShaderId::new(self.stamp, self.slot()?, 0);
        self.shaders.insert(id);
        self.calls.push(MockCall::CreateShader(id));
        Ok(id)
    }
    fn destroy_shader(&mut self, id: ShaderId) -> Result<(), GlError> {
        self.ready("destroy-shader")?;
        self.live("destroy-shader", id, |this| this.shaders.contains(&id))?;
        self.shaders.remove(&id);
        self.calls.push(MockCall::DestroyShader(id));
        Ok(())
    }
    fn create_program(
        &mut self,
        d: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection), GlError> {
        self.ready("create-program")?;
        d.validate_for(self.profile())
            .map_err(|_| GlError::Validation {
                operation: "create-program",
                message: "invalid program descriptor".into(),
            })?;
        let id = ProgramId::new(self.stamp, self.slot()?, 0);
        self.programs.insert(id);
        self.calls.push(MockCall::CreateProgram(id));
        Ok((
            id,
            GlProgramReflection {
                vertex_inputs: vec![],
                fragment_outputs: vec![],
                assignments: vec![],
            },
        ))
    }
    fn destroy_program(&mut self, id: ProgramId) -> Result<(), GlError> {
        self.ready("destroy-program")?;
        self.live("destroy-program", id, |this| this.programs.contains(&id))?;
        self.programs.remove(&id);
        self.calls.push(MockCall::DestroyProgram(id));
        Ok(())
    }
}
impl GlVertexApi for MockGlFamilyApi {
    fn create_vertex_array(&mut self, l: &GlVertexLayout) -> Result<VertexArrayId, GlError> {
        self.ready("create-vertex-array")?;
        l.validate().map_err(|_| GlError::Validation {
            operation: "create-vertex-array",
            message: "invalid vertex layout".into(),
        })?;
        let id = VertexArrayId::new(self.stamp, self.slot()?, 0);
        self.vaos.insert(id);
        self.calls.push(MockCall::CreateVertexArray(id));
        Ok(id)
    }
    fn destroy_vertex_array(&mut self, id: VertexArrayId) -> Result<(), GlError> {
        self.ready("destroy-vertex-array")?;
        self.live("destroy-vertex-array", id, |this| this.vaos.contains(&id))?;
        self.vaos.remove(&id);
        self.calls.push(MockCall::DestroyVertexArray(id));
        Ok(())
    }
    fn bind_vertex_array(
        &mut self,
        id: VertexArrayId,
        buffers: &[GlVertexBufferBinding],
        index: Option<GlIndexBinding>,
    ) -> Result<(), GlError> {
        self.ready("bind-vertex-array")?;
        self.live("bind-vertex-array", id, |this| this.vaos.contains(&id))?;
        for b in buffers {
            self.buffer("bind-vertex-array", b.buffer)?;
        }
        if let Some(i) = index {
            self.buffer("bind-vertex-array", i.buffer)?;
        }
        self.calls.push(MockCall::BindVertexArray(id));
        Ok(())
    }
}
impl GlFramebufferApi for MockGlFamilyApi {
    fn create_framebuffer(
        &mut self,
        d: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        self.ready("create-framebuffer")?;
        for v in d
            .color_attachments
            .iter()
            .copied()
            .chain(d.depth_stencil_attachment)
        {
            self.frame_view("create-framebuffer", v)?;
        }
        d.validate(self.discovery.limits().max_color_attachments, self.stamp)
            .map_err(|_| GlError::Validation {
                operation: "create-framebuffer",
                message: "invalid framebuffer descriptor".into(),
            })?;
        let id = FramebufferId::new(self.stamp, self.slot()?, 0);
        self.framebuffers.insert(id, d.clone());
        self.calls.push(MockCall::CreateFramebuffer(id));
        Ok(id)
    }
    fn destroy_framebuffer(&mut self, id: FramebufferId) -> Result<(), GlError> {
        self.ready("destroy-framebuffer")?;
        self.live("destroy-framebuffer", id, |this| {
            this.framebuffers.contains_key(&id)
        })?;
        self.framebuffers.remove(&id);
        self.calls.push(MockCall::DestroyFramebuffer(id));
        Ok(())
    }
    fn begin_render_pass(&mut self, d: &GlRenderPassDescriptor) -> Result<(), GlError> {
        self.ready("begin-render-pass")?;
        self.live("begin-render-pass", d.framebuffer, |this| {
            this.framebuffers.contains_key(&d.framebuffer)
        })?;
        if self.pass_active {
            return self.invalid("begin-render-pass", "render pass already active");
        }
        let framebuffer = match self.framebuffers.get(&d.framebuffer).cloned() {
            Some(framebuffer) => framebuffer,
            None => return self.invalid("begin-render-pass", "framebuffer is not live"),
        };
        d.validate(
            &framebuffer,
            self.discovery.limits().max_color_attachments,
            self.stamp,
        )
        .map_err(|_| GlError::Validation {
            operation: "begin-render-pass",
            message: "render pass does not match its framebuffer descriptor".into(),
        })?;
        for a in &d.color_attachments {
            self.frame_view("begin-render-pass", a.view)?;
            if let Some(v) = a.resolve_target {
                self.frame_view("begin-render-pass", v)?;
            }
        }
        self.pass_active = true;
        self.calls.push(MockCall::BeginRenderPass(d.framebuffer));
        Ok(())
    }
    fn end_render_pass(&mut self) -> Result<(), GlError> {
        self.ready("end-render-pass")?;
        if !self.pass_active {
            return self.invalid("end-render-pass", "no active render pass");
        }
        self.pass_active = false;
        self.calls.push(MockCall::EndRenderPass);
        Ok(())
    }
}
impl GlRasterCommandApi for MockGlFamilyApi {
    fn set_raster_pipeline(&mut self, p: &GlRasterPipeline) -> Result<(), GlError> {
        self.ready("set-raster-pipeline")?;
        if !self.pass_active {
            return self.invalid("set-raster-pipeline", "no active render pass");
        }
        self.live("set-raster-pipeline", p.program, |this| {
            this.programs.contains(&p.program)
        })?;
        self.live("set-raster-pipeline", p.vertex_array, |this| {
            this.vaos.contains(&p.vertex_array)
        })?;
        self.calls.push(MockCall::SetRasterPipeline {
            program: p.program,
            vertex_array: p.vertex_array,
        });
        Ok(())
    }
    fn draw_raster(&mut self, d: GlDrawCommand) -> Result<(), GlError> {
        self.ready("draw-raster")?;
        if !self.pass_active {
            return self.invalid("draw-raster", "no active render pass");
        }
        let zero = match d {
            GlDrawCommand::NonIndexed(x) => x.vertex_count == 0 || x.instance_count == 0,
            GlDrawCommand::Indexed(x) => x.index_count == 0 || x.instance_count == 0,
        };
        if zero {
            return self.invalid("draw-raster", "draw count and instances must be nonzero");
        }
        self.calls.push(MockCall::DrawRaster(d));
        Ok(())
    }
}
impl GlCopyDomainApi for MockGlFamilyApi {
    fn copy_buffer_range(&mut self, s: GlBufferRange, d: GlBufferRange) -> Result<(), GlError> {
        self.ready("copy-buffer")?;
        let sd = self.buffer("copy-buffer", s.buffer)?;
        let dd = self.buffer("copy-buffer", d.buffer)?;
        s.validate_for(sd)
            .and_then(|_| d.validate_for(dd))
            .map_err(|_| GlError::Validation {
                operation: "copy-buffer",
                message: "invalid buffer range".into(),
            })?;
        if s.size != d.size {
            return self.invalid("copy-buffer", "copy sizes differ");
        }
        if !sd.usage.contains(GlBufferUsage::COPY_SOURCE)
            || !dd.usage.contains(GlBufferUsage::COPY_DESTINATION)
        {
            return self.invalid("copy-buffer", "copy source or destination usage is missing");
        }
        self.calls.push(MockCall::CopyBuffer {
            source: s.buffer,
            destination: d.buffer,
            size: s.size,
        });
        Ok(())
    }
    fn copy_texture_region(
        &mut self,
        s: GlTextureRegion,
        d: GlTextureRegion,
    ) -> Result<(), GlError> {
        self.ready("copy-texture")?;
        let sd = self.texture("copy-texture", s.subresource.texture)?;
        let dd = self.texture("copy-texture", d.subresource.texture)?;
        validate_texture_copy(s, sd, d, dd).map_err(|_| GlError::Validation {
            operation: "copy-texture",
            message: "invalid texture copy".into(),
        })?;
        let source_facts = self.format_facts("copy-texture", sd)?;
        let destination_facts = self.format_facts("copy-texture", dd)?;
        if !sd.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !dd.usage.contains(GlTextureUsage::COPY_DESTINATION)
            || !source_facts.copy_source
            || !destination_facts.copy_destination
        {
            return self.invalid(
                "copy-texture",
                "copy source or destination usage/exact format fact is missing",
            );
        }
        self.calls.push(MockCall::CopyTexture {
            source: s.subresource.texture,
            destination: d.subresource.texture,
        });
        Ok(())
    }
    fn upload_texture(
        &mut self,
        d: GlTextureRegion,
        l: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        self.ready("upload-texture")?;
        let desc = self.texture("upload-texture", d.subresource.texture)?;
        d.validate_for(desc).map_err(|_| GlError::Validation {
            operation: "upload-texture",
            message: "invalid texture upload".into(),
        })?;
        let n = l.required_bytes(d).map_err(|_| GlError::Validation {
            operation: "upload-texture",
            message: "invalid texture upload".into(),
        })?;
        if bytes.len() != usize::try_from(n).unwrap_or(usize::MAX) {
            return self.invalid("upload-texture", "byte length mismatch");
        }
        self.calls
            .push(MockCall::UploadTexture(d.subresource.texture));
        Ok(())
    }
    fn read_texture(
        &mut self,
        s: GlTextureRegion,
        l: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        self.ready("read-texture")?;
        let desc = self.texture("read-texture", s.subresource.texture)?;
        s.validate_for(desc).map_err(|_| GlError::Validation {
            operation: "read-texture",
            message: "invalid read region".into(),
        })?;
        let n = l.required_bytes(s).map_err(|_| GlError::Validation {
            operation: "read-texture",
            message: "invalid layout".into(),
        })?;
        let bytes = vec![
            0;
            usize::try_from(n).map_err(|_| GlError::OutOfMemory {
                operation: "read-texture"
            })?
        ];
        self.calls
            .push(MockCall::ReadTexture(s.subresource.texture));
        Ok(GlReadback { layout: l, bytes })
    }
    fn pixel_store(&self) -> GlPixelStoreState {
        self.pixel_store
    }
}
impl GlSyncApi for MockGlFamilyApi {
    fn create_fence(&mut self) -> Result<GlFenceLease, GlError> {
        self.ready("create-fence")?;
        let id = SyncId::new(self.stamp, self.slot()?, 0);
        self.syncs.insert(id);
        let lease = self.fences.issue(id)?;
        self.calls.push(MockCall::CreateFence(lease));
        Ok(lease)
    }
    fn destroy_fence(&mut self, l: GlFenceLease) -> Result<(), GlError> {
        self.ready("destroy-fence")?;
        self.stamp("destroy-fence", l.fence.context)?;
        self.fences.validate(l)?;
        self.syncs.remove(&l.fence);
        self.fences.revoke(l);
        self.calls.push(MockCall::DestroyFence(l));
        Ok(())
    }
    fn poll_fence(&mut self, l: GlFenceLease) -> Result<GlFenceStatus, GlError> {
        self.ready("poll-fence")?;
        self.stamp("poll-fence", l.fence.context)?;
        self.fences.validate(l)?;
        self.calls.push(MockCall::PollFence(l));
        Ok(GlFenceStatus::Pending)
    }
    fn wait_fence(&mut self, l: GlFenceLease, _: GlWaitBound) -> Result<GlFenceStatus, GlError> {
        self.ready("wait-fence")?;
        self.stamp("wait-fence", l.fence.context)?;
        self.fences.validate(l)?;
        self.calls.push(MockCall::WaitFence(l));
        Ok(GlFenceStatus::Pending)
    }
    fn flush(&mut self) -> Result<(), GlError> {
        self.ready("flush")?;
        self.calls.push(MockCall::Flush);
        Ok(())
    }
}
impl GlSurfacePresentationApi for MockGlFamilyApi {
    fn acquire_surface_image(&mut self) -> Result<GlSurfaceAcquire, GlError> {
        self.ready("acquire-surface-image")?;
        if self.surface_suspended || self.surface_size.is_zero() {
            return Ok(GlSurfaceAcquire::Suspended);
        }
        let image = SurfaceImageId::new(self.stamp, self.slot()?, 0);
        let lease = self.surface.acquire(image, self.surface_size)?;
        self.calls.push(MockCall::AcquireSurface(lease));
        Ok(GlSurfaceAcquire::Lease(lease))
    }
    fn resize_surface(&mut self, size: GlSurfaceSize) -> Result<(), GlError> {
        self.ready("resize-surface")?;
        self.surface.invalidate_generation()?;
        self.surface_size = size;
        self.calls.push(MockCall::ResizeSurface(size));
        Ok(())
    }
    fn suspend_surface(&mut self) -> Result<(), GlError> {
        self.ready("suspend-surface")?;
        self.surface.invalidate_generation()?;
        self.surface_suspended = true;
        self.calls.push(MockCall::SuspendSurface);
        Ok(())
    }
    fn resume_surface(&mut self) -> Result<(), GlError> {
        self.ready("resume-surface")?;
        self.surface.invalidate_generation()?;
        self.surface_suspended = false;
        self.calls.push(MockCall::ResumeSurface);
        Ok(())
    }
    fn present_surface(&mut self, l: GlSurfaceLease) -> Result<(), GlError> {
        self.ready("present-surface")?;
        self.stamp("present-surface", l.image.context)?;
        self.surface.consume(l)?;
        self.calls.push(MockCall::PresentSurface(l));
        Ok(())
    }
}
impl GlQueryObjectsApi for MockGlFamilyApi {
    fn create_query(&mut self) -> Result<QueryId, GlError> {
        self.ready("create-query")?;
        let id = QueryId::new(self.stamp, self.slot()?, 0);
        self.queries.insert(id);
        self.calls.push(MockCall::CreateQuery(id));
        Ok(id)
    }
    fn destroy_query(&mut self, id: QueryId) -> Result<(), GlError> {
        self.ready("destroy-query")?;
        self.live("destroy-query", id, |this| this.queries.contains(&id))?;
        self.queries.remove(&id);
        self.calls.push(MockCall::DestroyQuery(id));
        Ok(())
    }
    fn query_result(&mut self, id: QueryId) -> Result<GlQueryResult, GlError> {
        self.ready("query-result")?;
        self.live("query-result", id, |this| this.queries.contains(&id))?;
        self.calls.push(MockCall::QueryResult(id));
        Ok(GlQueryResult::Pending)
    }
}
impl GlOcclusionQueryApi for MockGlFamilyApi {
    fn begin_occlusion_query(&mut self, id: QueryId) -> Result<(), GlError> {
        self.ready("begin-occlusion-query")?;
        self.live("begin-occlusion-query", id, |this| {
            this.queries.contains(&id)
        })?;
        self.calls.push(MockCall::BeginOcclusion(id));
        Ok(())
    }
    fn end_occlusion_query(&mut self) -> Result<(), GlError> {
        self.ready("end-occlusion-query")?;
        self.calls.push(MockCall::EndOcclusion);
        Ok(())
    }
}

/// Explicit opt-in domains, available only after their snapshot proved both capabilities.
#[derive(Debug)]
pub struct MockComputeStorageApi {
    inner: MockGlFamilyApi,
}
impl MockComputeStorageApi {
    pub fn new(inner: MockGlFamilyApi) -> Result<Self, GlError> {
        let caps = inner.discovery().capabilities();
        if caps.supports(GlCapability::Compute) && caps.supports(GlCapability::StorageBuffer) {
            Ok(Self { inner })
        } else {
            Err(GlError::Unsupported {
                operation: "mock-compute-storage",
                reason: "discovery did not prove compute and storage-buffer support",
            })
        }
    }
    pub fn calls(&self) -> &[MockCall] {
        self.inner.calls()
    }
    pub fn into_inner(self) -> MockGlFamilyApi {
        self.inner
    }
}
impl GlFamilyApi for MockComputeStorageApi {
    fn profile(&self) -> GlFamilyProfile {
        self.inner.profile()
    }
    fn context_stamp(&self) -> ContextStamp {
        self.inner.context_stamp()
    }
    fn lifecycle(&self) -> GlContextLifecycle {
        self.inner.lifecycle()
    }
    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.inner.owner_thread()
    }
    fn assert_owner_thread(&self, op: &'static str) -> Result<(), GlError> {
        self.inner.assert_owner_thread(op)
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        self.inner.discovery()
    }
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.inner.context_lost()
    }
    fn context_restored(&mut self) -> Result<ContextStamp, GlError> {
        self.inner.context_restored()
    }
}
impl GlComputeDispatchApi for MockComputeStorageApi {
    fn dispatch(&mut self, g: GlDispatchGroups) -> Result<(), GlError> {
        self.inner.ready("dispatch")?;
        g.validate(GlComputeLimits {
            max_group_count: self.inner.discovery.limits().max_compute_work_group_count,
            max_group_size: self.inner.discovery.limits().max_compute_work_group_size,
            max_group_invocations: self
                .inner
                .discovery
                .limits()
                .max_compute_work_group_invocations,
        })?;
        self.inner.calls.push(MockCall::Dispatch(g));
        Ok(())
    }
    fn memory_barrier(&mut self, b: GlMemoryBarrier) -> Result<(), GlError> {
        self.inner.ready("memory-barrier")?;
        b.validate_nonempty()
    }
}
impl GlStorageBufferApi for MockComputeStorageApi {
    fn bind_storage_buffer(
        &mut self,
        binding: u32,
        r: GlStorageBufferRange,
    ) -> Result<(), GlError> {
        self.inner.ready("bind-storage-buffer")?;
        let desc = self.inner.buffer("bind-storage-buffer", r.buffer)?;
        r.validate(
            binding,
            GlStorageBufferLimits {
                max_bindings: self.inner.discovery.limits().max_storage_buffer_bindings,
                max_block_size: self.inner.discovery.limits().max_storage_block_size,
                offset_alignment: self
                    .inner
                    .discovery
                    .limits()
                    .storage_buffer_offset_alignment,
            },
        )?;
        GlBufferRange {
            buffer: r.buffer,
            offset: r.offset,
            size: r.size,
        }
        .validate_for(desc)
        .map_err(|_| GlError::Validation {
            operation: "bind-storage-buffer",
            message: "storage buffer range is outside the allocation".into(),
        })?;
        if !desc.usage.contains(GlBufferUsage::STORAGE) {
            return self
                .inner
                .invalid("bind-storage-buffer", "buffer lacks storage usage");
        }
        self.inner.calls.push(MockCall::BindStorageBuffer {
            binding,
            buffer: r.buffer,
            offset: r.offset,
            size: r.size,
        });
        Ok(())
    }
}
impl GlStorageImageApi for MockComputeStorageApi {
    fn bind_storage_image(
        &mut self,
        binding: u32,
        i: GlStorageImageBinding,
    ) -> Result<(), GlError> {
        self.inner.ready("bind-storage-image")?;
        self.inner.validate_storage_image_binding(binding, i)?;
        self.inner.calls.push(MockCall::BindStorageImage {
            binding,
            texture: i.texture,
        });
        Ok(())
    }
}
