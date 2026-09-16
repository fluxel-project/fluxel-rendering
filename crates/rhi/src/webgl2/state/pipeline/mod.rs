//! The pipeline state domain: which raster pipeline and which program are
//! installed, and the rasterization state a pipeline install carries.
//!
//! Responsibility: make the backend's installed program, raster pipeline, and
//! per-pipeline rasterization values agree with what the caller asked for.
//!
//! Not owned here: the pass boundary (session), the vertex bindings (geometry),
//! the texture and buffer bindings (textures, buffers), and the linked-program
//! objects themselves (Layer 1's shader tables).  This domain holds the derived
//! program cache and the installed-state mirror, and nothing else.
//!
//! Layer 1 fact this domain exists for: `end_render_pass` forgets the installed
//! pipeline, which is why the session reports [`super::session::SessionEffects`]
//! and the machine applies it here.
