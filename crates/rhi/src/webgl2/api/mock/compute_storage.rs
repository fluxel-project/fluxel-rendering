//! Opt-in mock wrapper over the proved compute and storage domains.
//!
//! The wrapper exists so ordinary compute-free fixtures cannot accidentally
//! reach optional domains: constructing it fails unless the bound discovery
//! snapshot proved both capabilities, mirroring the provider-side rule.
//!
//! Indirect dispatch lives here rather than with the other indirect domains
//! because it is the one indirect command that is not a raster command: it needs
//! the installed compute program, which only this wrapper carries, exactly as
//! the native provider's single compute type carries it.

use super::*;

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
    fn set_compute_program(&mut self, program: ProgramId) -> Result<(), GlError> {
        self.inner.ready("set-compute-program")?;
        self.inner.live("set-compute-program", program, |this| {
            this.programs.contains(&program)
        })?;
        self.inner.installed_compute_program = Some(program);
        // The verb installs as well as records, so the modelled driver's current
        // program moves here too.  A recorder that only recorded the intent would
        // accept a dispatch the real provider cannot perform.
        self.inner.select_program(program);
        self.inner.calls.push(MockCall::SetComputeProgram(program));
        Ok(())
    }
    fn dispatch(&mut self, g: GlDispatchGroups) -> Result<(), GlError> {
        self.inner.ready("dispatch")?;
        let Some(program) = self.inner.installed_compute_program else {
            return self
                .inner
                .invalid("dispatch", "no compute program is installed");
        };
        // A raster install since the compute install took the current-program
        // slot, so the selection is re-asserted, exactly as the provider does.
        self.inner.select_program(program);
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
impl GlDispatchIndirectApi for MockComputeStorageApi {
    fn dispatch_indirect(&mut self, command: GlDispatchIndirectCommand) -> Result<(), GlError> {
        const OP: &str = "dispatch-indirect";
        self.inner.ready(OP)?;
        self.inner.require_indirect_capability(
            OP,
            GlCapability::IndirectDispatch,
            "this context did not prove the indirect-dispatch capability",
        )?;
        // The installed program is what makes the record's work-group triple
        // meaningful, so the provider checks it after the capability row and
        // before the record layout: a context that never proved dispatch must
        // not be told to install a program first.
        let Some(program) = self.inner.installed_compute_program else {
            return self.inner.invalid(OP, "no compute program is installed");
        };
        self.inner
            .live(OP, program, |this| this.programs.contains(&program))?;
        if let Err(error) = command.validate(OP) {
            return self.inner.error_result(error);
        }
        self.inner.indirect_buffer(OP, command.range)?;
        self.inner.select_program(program);
        self.inner.calls.push(MockCall::DispatchIndirect(command));
        Ok(())
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

// ---------------------------------------------------------------------------
// The required command domains, forwarded.
// ---------------------------------------------------------------------------
//
// The wrapper holds every *optional* domain, so a fixture that did not ask for
// compute cannot name one.  That was its whole purpose and it is worth keeping.
// What it also used to be is a *half* provider: `GlStateBackend` requires
// fifteen traits beyond the family one, this type implemented none of them, and
// so no state machine could be built over it -- which left the machine's
// optional-domain entry points with no test at all (the plan's P1-18).
//
// Completing it does not weaken the opt-in.  What a fixture proves by writing
// `MockComputeStorageApi::new(inner)` is unchanged: it still has to hand over a
// recorder whose snapshot proved both capabilities, and an ordinary fixture over
// `MockGlFamilyApi` still cannot name a compute verb.  What changes is that the
// wrapper is now the *whole* provider for the context it wraps, which is closer
// to the driver's shape than the previous split was -- on real hardware one
// provider carries every command domain, and only the capability row decides
// which of them a context can serve.
//
// Every forwarder below is one line of body on purpose.  These are not
// re-implementations: each one hands its arguments to the same method on the
// inner recorder, unchanged, so the trace the wrapper produces is the trace the
// inner recorder would have produced.  Nothing here validates, defaults or
// reorders anything, and a forwarder that did would be a second definition of a
// rule the inner recorder already owns.
//
// How far that claim is *checked* rather than merely written down is worth being
// exact about, because "these are only forwarders" is the kind of sentence that
// stops a reader looking.  A wrong method name, or a body whose arguments do not
// match the trait's, does not compile: the trait declares the signature and the
// inner recorder is the only thing that can satisfy it.  A right name with a
// *wrong argument inside it* compiles, and the only cure is a test that reads the
// trace back.  Two roles have one: `bind_storage_buffer` and `bind_storage_image`
// are driven through a state machine over this wrapper and their recorded words
// are compared exactly, so a corrupted forward there fails a test -- confirmed by
// corrupting one on purpose and watching seven tests across two suites fail.  The
// remaining fifty are reached by no test, and among them three take a pair of
// same-typed parameters that could be swapped without the compiler noticing:
// `copy_buffer_range`, `copy_texture_region` and `blit_framebuffer`, each a
// source/destination pair.  Those three are right by inspection and by nothing
// else, and a change to any of them deserves a reader's attention rather than
// this comment's assurance.

impl GlResourceApi for MockComputeStorageApi {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        self.inner.create_buffer_resource(desc)
    }
    fn create_texture_resource(&mut self, desc: GlTextureDesc) -> Result<TextureId, GlError> {
        self.inner.create_texture_resource(desc)
    }
    fn create_render_buffer(
        &mut self,
        desc: GlRenderBufferDesc,
    ) -> Result<RenderbufferId, GlError> {
        self.inner.create_render_buffer(desc)
    }
    fn destroy_buffer_resource(&mut self, buffer: BufferId) -> Result<(), GlError> {
        self.inner.destroy_buffer_resource(buffer)
    }
    fn destroy_texture_resource(&mut self, texture: TextureId) -> Result<(), GlError> {
        self.inner.destroy_texture_resource(texture)
    }
    fn destroy_render_buffer(&mut self, render_buffer: RenderbufferId) -> Result<(), GlError> {
        self.inner.destroy_render_buffer(render_buffer)
    }
}
impl GlSamplerApi for MockComputeStorageApi {
    fn create_sampler(&mut self, desc: GlSamplerDesc) -> Result<SamplerId, GlError> {
        self.inner.create_sampler(desc)
    }
    fn destroy_sampler(&mut self, sampler: SamplerId) -> Result<(), GlError> {
        self.inner.destroy_sampler(sampler)
    }
}
impl GlCopyDomainApi for MockComputeStorageApi {
    fn copy_buffer_range(
        &mut self,
        source: GlBufferRange,
        destination: GlBufferRange,
    ) -> Result<(), GlError> {
        self.inner.copy_buffer_range(source, destination)
    }
    fn copy_texture_region(
        &mut self,
        source: GlTextureRegion,
        destination: GlTextureRegion,
    ) -> Result<(), GlError> {
        self.inner.copy_texture_region(source, destination)
    }
    fn upload_buffer(&mut self, destination: GlBufferRange, bytes: &[u8]) -> Result<(), GlError> {
        self.inner.upload_buffer(destination, bytes)
    }
    fn read_buffer(&mut self, source: GlBufferRange) -> Result<Vec<u8>, GlError> {
        self.inner.read_buffer(source)
    }
    fn upload_texture(
        &mut self,
        destination: GlTextureRegion,
        layout: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        self.inner.upload_texture(destination, layout, bytes)
    }
    fn read_texture(
        &mut self,
        source: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        self.inner.read_texture(source, layout)
    }
    fn pixel_store(&self) -> GlPixelStoreState {
        self.inner.pixel_store()
    }
}
impl GlFramebufferApi for MockComputeStorageApi {
    fn create_framebuffer(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        self.inner.create_framebuffer(descriptor)
    }
    fn destroy_framebuffer(&mut self, framebuffer: FramebufferId) -> Result<(), GlError> {
        self.inner.destroy_framebuffer(framebuffer)
    }
    fn begin_render_pass(&mut self, descriptor: &GlRenderPassDescriptor) -> Result<(), GlError> {
        self.inner.begin_render_pass(descriptor)
    }
    fn end_render_pass(&mut self) -> Result<(), GlError> {
        self.inner.end_render_pass()
    }
    fn blit_framebuffer(
        &mut self,
        source: FramebufferId,
        destination: FramebufferId,
        region: GlBlitRegion,
        filter: GlFilterMode,
        masks: GlBlitMask,
    ) -> Result<(), GlError> {
        self.inner
            .blit_framebuffer(source, destination, region, filter, masks)
    }
}
impl GlRasterCommandApi for MockComputeStorageApi {
    fn set_raster_pipeline(&mut self, pipeline: &GlRasterPipeline) -> Result<(), GlError> {
        self.inner.set_raster_pipeline(pipeline)
    }
    fn draw_raster(&mut self, draw: GlDrawCommand) -> Result<(), GlError> {
        self.inner.draw_raster(draw)
    }
}
impl GlShaderApi for MockComputeStorageApi {
    fn create_shader(&mut self, source: &GlShaderSource) -> Result<ShaderId, GlError> {
        self.inner.create_shader(source)
    }
    fn destroy_shader(&mut self, shader: ShaderId) -> Result<(), GlError> {
        self.inner.destroy_shader(shader)
    }
    fn create_program(
        &mut self,
        descriptor: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection), GlError> {
        self.inner.create_program(descriptor)
    }
    fn destroy_program(&mut self, program: ProgramId) -> Result<(), GlError> {
        self.inner.destroy_program(program)
    }
}
impl GlBindingApi for MockComputeStorageApi {
    fn active_texture(&mut self, unit: u32) -> Result<(), GlError> {
        self.inner.active_texture(unit)
    }
    fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) -> Result<(), GlError> {
        self.inner.bind_texture(unit, target, texture)
    }
    fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) -> Result<(), GlError> {
        self.inner.bind_sampler(unit, sampler)
    }
    fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
    ) -> Result<(), GlError> {
        self.inner.bind_uniform_buffer(index, buffer, offset, size)
    }
}
impl GlVertexApi for MockComputeStorageApi {
    fn create_vertex_array(&mut self, layout: &GlVertexLayout) -> Result<VertexArrayId, GlError> {
        self.inner.create_vertex_array(layout)
    }
    fn destroy_vertex_array(&mut self, vertex_array: VertexArrayId) -> Result<(), GlError> {
        self.inner.destroy_vertex_array(vertex_array)
    }
    fn bind_vertex_array(
        &mut self,
        vertex_array: VertexArrayId,
        buffers: &[GlVertexBufferBinding],
        index: Option<GlIndexBinding>,
    ) -> Result<(), GlError> {
        self.inner.bind_vertex_array(vertex_array, buffers, index)
    }
}
impl GlSyncApi for MockComputeStorageApi {
    fn create_fence(&mut self) -> Result<GlFenceLease, GlError> {
        self.inner.create_fence()
    }
    fn destroy_fence(&mut self, fence: GlFenceLease) -> Result<(), GlError> {
        self.inner.destroy_fence(fence)
    }
    fn poll_fence(&mut self, fence: GlFenceLease) -> Result<GlFenceStatus, GlError> {
        self.inner.poll_fence(fence)
    }
    fn wait_fence(
        &mut self,
        fence: GlFenceLease,
        bound: GlWaitBound,
    ) -> Result<GlFenceStatus, GlError> {
        self.inner.wait_fence(fence, bound)
    }
    fn flush(&mut self) -> Result<(), GlError> {
        self.inner.flush()
    }
}
impl GlQueryObjectsApi for MockComputeStorageApi {
    fn create_query(&mut self) -> Result<QueryId, GlError> {
        self.inner.create_query()
    }
    fn destroy_query(&mut self, query: QueryId) -> Result<(), GlError> {
        self.inner.destroy_query(query)
    }
    fn query_result(&mut self, query: QueryId) -> Result<GlQueryResult, GlError> {
        self.inner.query_result(query)
    }
}
impl GlOcclusionQueryApi for MockComputeStorageApi {
    fn begin_occlusion_query(&mut self, query: QueryId) -> Result<(), GlError> {
        self.inner.begin_occlusion_query(query)
    }
    fn end_occlusion_query(&mut self) -> Result<(), GlError> {
        self.inner.end_occlusion_query()
    }
}
impl GlElapsedQueryApi for MockComputeStorageApi {
    fn begin_elapsed_query(&mut self, query: QueryId) -> Result<(), GlError> {
        self.inner.begin_elapsed_query(query)
    }
    fn end_elapsed_query(&mut self) -> Result<(), GlError> {
        self.inner.end_elapsed_query()
    }
}
impl GlTimestampQueryApi for MockComputeStorageApi {
    fn query_timestamp(&mut self, query: QueryId) -> Result<(), GlError> {
        self.inner.query_timestamp(query)
    }
}
impl GlMultiDrawApi for MockComputeStorageApi {
    fn multi_draw(&mut self, command: &GlMultiDraw) -> Result<(), GlError> {
        self.inner.multi_draw(command)
    }
}
impl GlSurfacePresentationApi for MockComputeStorageApi {
    fn acquire_surface_image(&mut self) -> Result<GlSurfaceAcquire, GlError> {
        self.inner.acquire_surface_image()
    }
    fn resize_surface(&mut self, size: GlSurfaceSize) -> Result<(), GlError> {
        self.inner.resize_surface(size)
    }
    fn suspend_surface(&mut self) -> Result<(), GlError> {
        self.inner.suspend_surface()
    }
    fn resume_surface(&mut self) -> Result<(), GlError> {
        self.inner.resume_surface()
    }
    fn present_surface(&mut self, lease: GlSurfaceLease) -> Result<(), GlError> {
        self.inner.present_surface(lease)
    }
}
