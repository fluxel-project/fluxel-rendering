//! Browser object records and generation-safe lookup helpers.
//!
//! Slot/generation identities are Fluxel-owned; the raw JS handles live only
//! inside these records and never participate in identity comparisons.

use web_sys::{
    WebGlFramebuffer, WebGlProgram, WebGlQuery, WebGlRenderbuffer, WebGlShader, WebGlSync,
    WebGlVertexArrayObject,
};

use super::super::GlFamilyApi as _;
use super::super::{
    FramebufferId, GlError, GlFramebufferDescriptor, GlIndexBinding, GlPrimitiveTopology,
    GlProgramDescriptor, GlRenderBufferDesc, GlVertexLayout, ProgramId, QueryId, RenderbufferId,
    ShaderId, SyncId, VertexArrayId,
};
use super::discovery::WebGl2BrowserDiscovery;

pub(super) struct BrowserShader {
    pub(super) generation: u32,
    pub(super) raw: WebGlShader,
}
pub(super) struct BrowserProgram {
    pub(super) generation: u32,
    pub(super) raw: WebGlProgram,
    pub(super) descriptor: GlProgramDescriptor,
}
pub(super) struct BrowserVertexArray {
    pub(super) generation: u32,
    pub(super) raw: WebGlVertexArrayObject,
    pub(super) layout: GlVertexLayout,
    pub(super) index: Option<GlIndexBinding>,
}
pub(super) struct BrowserFramebuffer {
    pub(super) generation: u32,
    pub(super) raw: WebGlFramebuffer,
    /// Creation facts, retained for pass validation and blit bounds.
    pub(super) descriptor: GlFramebufferDescriptor,
}
pub(super) struct BrowserRenderbuffer {
    pub(super) generation: u32,
    pub(super) raw: WebGlRenderbuffer,
    /// Creation facts, retained because nothing else records what was allocated.
    pub(super) desc: GlRenderBufferDesc,
}
pub(super) struct BrowserQuery {
    pub(super) generation: u32,
    pub(super) raw: WebGlQuery,
    /// The target the query last recorded a measurement for.
    pub(super) target: Option<u32>,
}
pub(super) struct BrowserSync {
    pub(super) generation: u32,
    pub(super) raw: WebGlSync,
}

/// Facts of the render pass currently recording on this context.
pub(super) struct ActivePass {
    pub(super) framebuffer: FramebufferId,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) samples: u32,
    /// Per color attachment; `true` when `end_render_pass` must invalidate.
    pub(super) discard_color: Vec<bool>,
    pub(super) discard_depth_stencil: Option<bool>,
}

/// The raster pipeline installed for the active pass.
pub(super) struct ActiveRaster {
    pub(super) program: ProgramId,
    /// The topology this pipeline installed.
    ///
    /// The vertex array is deliberately not recorded here even though the install
    /// names one: GL's vertex-array binding is a single slot the input domain also
    /// writes, and its owner is `WebGl2BrowserDiscovery::bound_vertex_array`.  A
    /// per-pipeline copy went stale the moment the geometry domain reconciled
    /// inputs, which under the uncached execution mode is on every request.
    pub(super) topology: GlPrimitiveTopology,
}

impl WebGl2BrowserDiscovery {
    pub(super) fn shader(
        &self,
        operation: &'static str,
        id: ShaderId,
    ) -> Result<&BrowserShader, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.shaders.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "shader is not live")),
        }
    }

    pub(super) fn program(
        &self,
        operation: &'static str,
        id: ProgramId,
    ) -> Result<&BrowserProgram, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.programs.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "program is not live")),
        }
    }

    pub(super) fn vertex_array(
        &self,
        operation: &'static str,
        id: VertexArrayId,
    ) -> Result<&BrowserVertexArray, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.vertex_arrays.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "vertex array is not live")),
        }
    }

    pub(super) fn framebuffer(
        &self,
        operation: &'static str,
        id: FramebufferId,
    ) -> Result<&BrowserFramebuffer, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.framebuffers.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "framebuffer is not live")),
        }
    }

    pub(super) fn renderbuffer(
        &self,
        operation: &'static str,
        id: RenderbufferId,
    ) -> Result<&BrowserRenderbuffer, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.renderbuffers.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(
                operation,
                "renderbuffer allocation is not live",
            )),
        }
    }

    pub(super) fn query(
        &self,
        operation: &'static str,
        id: QueryId,
    ) -> Result<&BrowserQuery, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.queries.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "query is not live")),
        }
    }

    pub(super) fn sync(
        &self,
        operation: &'static str,
        id: SyncId,
    ) -> Result<&BrowserSync, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.syncs.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "fence is not live")),
        }
    }

    /// Shared framebuffer-completeness observation for copy and pass work.
    pub(super) fn require_complete(&self, operation: &'static str) -> Result<(), GlError> {
        use web_sys::WebGl2RenderingContext as Gl;
        let status = self.raw.check_framebuffer_status(Gl::FRAMEBUFFER);
        if status == Gl::FRAMEBUFFER_COMPLETE {
            Ok(())
        } else {
            Err(GlError::IncompleteFramebuffer { operation, status })
        }
    }

    /// Clears every executable table. Context loss must never call methods on
    /// the browser handles, so only the Rust-side tables are dropped.
    pub(super) fn clear_executable_state(&mut self) {
        self.shaders.clear();
        self.programs.clear();
        self.vertex_arrays.clear();
        self.framebuffers.clear();
        self.queries.clear();
        self.syncs.clear();
        self.fences.revoke_all();
        self.pass = None;
        self.raster = None;
        self.bound_vertex_array = None;
        self.active_query = None;
        self.next_shader_slot = 0;
        self.next_program_slot = 0;
        self.next_vertex_array_slot = 0;
        self.next_framebuffer_slot = 0;
        self.next_query_slot = 0;
        self.next_sync_slot = 0;
        self.next_surface_slot = 0;
        let _ = self.surface.invalidate_generation();
    }
}
