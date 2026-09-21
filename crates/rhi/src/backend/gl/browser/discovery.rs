//! Browser-owned WebGL2 discovery evidence and the RHI-owned context record.
//!
//! The Host/JS bridge owns canvas creation, DOM events, RAF, and context-loss
//! listeners. RHI creates and owns the WebGL2 context associated with that
//! Host-provided canvas, then gathers immutable evidence for its `ContextStamp`.

use core::cell::Cell;
use js_sys::{Array, Object, Reflect};
use std::collections::BTreeMap;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlSampler, WebGlTexture};

use super::super::api::{
    BufferId, ContextStamp, CoreOrExtension, GlBufferDesc, GlCapability, GlContextFlags,
    GlContextInfo, GlContextLifecycle, GlDiscoveryBuilder, GlDiscoveryError, GlDiscoverySnapshot,
    GlError, GlExtensionSet, GlFamilyApi as _, GlFamilyProfile, GlFenceLeaseBook, GlFiniteF32,
    GlKnownExtension, GlLimits, GlOperationProbe, GlPixelStoreState, GlSurfaceFacts,
    GlSurfaceLeaseBook, GlTextureDesc, OwnerThreadIdentity, TextureId, VertexArrayId,
};
use super::exec_multidraw::BrowserMultiDraw;
use super::exec_timer::{self, BrowserTimerQuery};
use super::objects::{
    ActivePass, ActiveRaster, BrowserFramebuffer, BrowserProgram, BrowserQuery,
    BrowserRenderbuffer, BrowserShader, BrowserSync, BrowserVertexArray,
};

/// Immutable WebGL2 discovery evidence plus the callable browser context.
///
/// RHI retains the Host-provided canvas and the context it created. `glow`
/// receives a cloned JS context handle, while `raw` remains the RHI-owned
/// browser API handle for discovery and provider work.
/// It is called only after the fallible WebGL2 version check below succeeds.
/// The remaining assumptions made by glow are WebGL binding invariants (a real
/// WebGL2 context returns a string `VERSION` and an array-or-null extension
/// list); violations are browser defects rather than recoverable application
/// input.  All application-observable queries in this module remain fallible.
pub(crate) struct WebGl2BrowserDiscovery {
    pub(super) canvas: HtmlCanvasElement,
    pub(super) raw: WebGl2RenderingContext,
    pub(super) glow: glow::Context,
    pub(super) snapshot: GlDiscoverySnapshot,
    pub(super) owner_thread: OwnerThreadIdentity,
    pub(super) lifecycle: Cell<GlContextLifecycle>,
    pub(super) buffers: BTreeMap<u32, BrowserBuffer>,
    pub(super) textures: BTreeMap<u32, BrowserTexture>,
    pub(super) samplers: BTreeMap<u32, BrowserSampler>,
    pub(super) renderbuffers: BTreeMap<u32, BrowserRenderbuffer>,
    pub(super) shaders: BTreeMap<u32, BrowserShader>,
    pub(super) programs: BTreeMap<u32, BrowserProgram>,
    pub(super) vertex_arrays: BTreeMap<u32, BrowserVertexArray>,
    pub(super) framebuffers: BTreeMap<u32, BrowserFramebuffer>,
    pub(super) queries: BTreeMap<u32, BrowserQuery>,
    pub(super) syncs: BTreeMap<u32, BrowserSync>,
    pub(super) fences: GlFenceLeaseBook,
    pub(super) surface: GlSurfaceLeaseBook,
    pub(super) surface_suspended: bool,
    pub(super) pass: Option<ActivePass>,
    pub(super) raster: Option<ActiveRaster>,
    /// The vertex array the modelled driver holds, or `None` when it holds none.
    ///
    /// GL has one vertex-array binding slot, and this provider models the slot
    /// rather than the intent of each verb: a pipeline install and an input
    /// reconcile both reach it, and the install's choice is the older of the two
    /// by the time a draw runs.  A draw resolves *this* record and not the array
    /// the pipeline install named, because the geometry domain replaces that
    /// array on every request under the uncached execution mode.  The two verbs
    /// that bind a real array write it -- the input domain's `bind_vertex_array`
    /// and a raster pipeline install -- and `prepare_draw` is what reads it.
    pub(super) bound_vertex_array: Option<VertexArrayId>,
    /// The batch commands retained from the acquired batch extension object.
    ///
    /// `Some` exactly when that object exposed every command the batch domain
    /// issues, so it is also exactly when the batch capability resolved (audit
    /// P1-8). Dropped with the rest of the executable state on loss, because the
    /// object belongs to the dead context.
    pub(super) multi_draw: Option<BrowserMultiDraw>,
    /// The timer commands retained from the acquired timer-query object.
    ///
    /// `Some` exactly when that object exposed every command the timer domain
    /// issues, so it is also exactly when the timer capability could resolve
    /// (audit P1-5). Dropped with the rest of the executable state on loss,
    /// because the object belongs to the dead context.
    pub(super) timer: Option<BrowserTimerQuery>,
    /// `WEBGL_compressed_texture_astc` exposes LDR and HDR as profile strings
    /// on the extension object.  HDR must not inherit LDR admission merely
    /// because the object exists.
    pub(super) astc_hdr: bool,
    /// Slot of the query currently recording, if any.
    pub(super) active_query: Option<u32>,
    pub(super) next_buffer_slot: u32,
    pub(super) next_texture_slot: u32,
    pub(super) next_sampler_slot: u32,
    pub(super) next_renderbuffer_slot: u32,
    pub(super) next_shader_slot: u32,
    pub(super) next_program_slot: u32,
    pub(super) next_vertex_array_slot: u32,
    pub(super) next_framebuffer_slot: u32,
    pub(super) next_query_slot: u32,
    pub(super) next_sync_slot: u32,
    pub(super) next_surface_slot: u32,
    pub(super) pixel_store: GlPixelStoreState,
}

pub(super) struct BrowserBuffer {
    pub(super) generation: u32,
    pub(super) raw: WebGlBuffer,
    pub(super) desc: GlBufferDesc,
}
pub(super) struct BrowserTexture {
    pub(super) generation: u32,
    pub(super) raw: WebGlTexture,
    pub(super) desc: GlTextureDesc,
}
pub(super) struct BrowserSampler {
    pub(super) generation: u32,
    pub(super) raw: WebGlSampler,
}

impl WebGl2BrowserDiscovery {
    /// Creates and owns a WebGL2 context for a Host-provided canvas, then
    /// discovers its current generation without issuing a rendering command.
    ///
    /// This deliberately preserves the 0.14 readback/residency-oracle setting:
    /// the default framebuffer remains available after presentation.
    pub(crate) fn open(stamp: ContextStamp, canvas: HtmlCanvasElement) -> Result<Self, GlError> {
        // Context creation is itself a browser side effect, so ownership is
        // captured before configuring or touching the canvas.
        let owner_thread = OwnerThreadIdentity::current();
        let options = Object::new();
        Reflect::set(
            &options,
            &JsValue::from_str("preserveDrawingBuffer"),
            &JsValue::TRUE,
        )
        .map_err(|value| js_error("configure WebGL2 context", value))?;
        let value = canvas
            .get_context_with_context_options("webgl2", &options)
            .map_err(|value| js_error("create WebGL2 context", value))?
            .ok_or_else(|| driver("create WebGL2 context", "browser did not provide WebGL2"))?;
        let raw = value.dyn_into::<WebGl2RenderingContext>().map_err(|_| {
            driver(
                "create WebGL2 context",
                "browser returned a non-WebGL2 context",
            )
        })?;
        Self::from_owned_context(stamp, canvas, raw, owner_thread)
    }

    /// Internal ownership-transfer seam for RHI context restoration only.
    ///
    /// It is intentionally private: Host never creates or owns the WebGL2
    /// context in the Fluxel boundary model.
    fn from_owned_context(
        stamp: ContextStamp,
        canvas: HtmlCanvasElement,
        raw: WebGl2RenderingContext,
        owner_thread: OwnerThreadIdentity,
    ) -> Result<Self, GlError> {
        ensure_context_live(&raw, "discover WebGL2 context")?;
        let version = string_parameter(&raw, WebGl2RenderingContext::VERSION, "VERSION")?;
        require_webgl2_version(&version)?;

        // Do this after validating the only glow constructor precondition that
        // comes from application-visible browser state.  We deliberately do
        // not use glow for discovery: its browser constructor panics on JS
        // binding invariant violations instead of returning `Result`.
        let glow = glow::Context::from_webgl2_context(raw.clone());
        let (extensions, commands) = discover_extensions(&raw)?;
        // The masked vendor/renderer strings above are the context's own answers
        // and stay verbatim; the unmasked pair, when the context exposes the
        // optional debug route, is recorded beside them as flag markers (audit
        // P2-6) rather than replacing an answer the browser chose to give.
        let mut context_flags = discover_context_flags(&raw)?;
        super::driver_identity::record_markers(
            &mut context_flags,
            super::driver_identity::unmasked_identity(&raw),
        );
        let context = GlContextInfo::new(
            GlFamilyProfile::WebGl2,
            version,
            string_parameter(
                &raw,
                WebGl2RenderingContext::SHADING_LANGUAGE_VERSION,
                "SHADING_LANGUAGE_VERSION",
            )?,
            string_parameter(&raw, WebGl2RenderingContext::VENDOR, "VENDOR")?,
            string_parameter(&raw, WebGl2RenderingContext::RENDERER, "RENDERER")?,
            browser_identity()?,
            context_flags,
        );
        // The timer counter width is not a context limit: it is answered by the
        // acquired timer object, and it is zero whenever that object is absent
        // or does not answer both timer targets.
        let limits = discover_limits(&raw, &extensions, commands.timer.as_ref())?;
        // Float and depth format facts are answered by real framebuffer
        // completeness probes on this exact context (audit P1-6); the probes
        // use scratch objects and leave no state behind.
        let mut formats = super::format_map::webgl2_baseline_formats(&raw, &extensions)?;
        if commands.astc_hdr {
            super::format_map::record_astc_hdr_formats(&mut formats)?;
        }
        // Renderbuffer facts are answered the same way, by allocating scratch
        // storage and reading attachment completeness back, so a renderbuffer
        // allocation is admitted only from an observation on this context.
        super::renderbuffer_facts::record_facts(&raw, &limits, &mut formats)?;
        let mut builder = GlDiscoveryBuilder::new(stamp, context, extensions, limits, formats)
            .map_err(discovery_error)?;
        // The drawing buffer is this family's FBO 0, and it is observed through
        // the same drawing-buffer parameters the native path reads. The answers
        // depend on the context attributes the canvas was created with, so they
        // are read rather than assumed from the specification: a context with no
        // alpha really does report a zero alpha width, and a presenter that
        // assumed RGBA would claim a format this context does not have. A
        // parameter that cannot be read leaves the facts unavailable, which is
        // the answer that rejects work rather than the one that guesses.
        builder.surface_facts(drawing_buffer_facts(&raw));

        // Timer queries are the sole currently normalized browser extension
        // domain, and their oracle is the complete, callable entry-point set
        // acquired below together with a nonzero counter width read from the
        // same object: the extension alone would prove nothing about whether a
        // measurement can ever be reported (audit P1-5).
        builder.resolve(
            GlCapability::TimerQuery,
            CoreOrExtension {
                desktop_core: None,
                embedded_core: None,
                extension: Some(GlKnownExtension::ExtDisjointTimerQueryWebgl2),
                extension_requires_probe: false,
            },
            GlOperationProbe::NotRequired,
        );
        // OVR_multiview2 is acquired when the browser exposes the extension
        // object, but no multiview attachment probe exists yet: the pass state
        // that would have to carry a view count is not modelled, so a pass that
        // claims several views would attach a single layer and render the wrong
        // picture. The row therefore records the real evidence and stays
        // permanently disabled until the probe and the multiview attach path
        // land together (audit P1-7).
        builder.resolve(
            GlCapability::Multiview,
            CoreOrExtension {
                desktop_core: None,
                embedded_core: None,
                extension: Some(GlKnownExtension::OvrMultiview2),
                extension_requires_probe: true,
            },
            GlOperationProbe::NotRun,
        );
        // WEBGL_multi_draw is the browser's only per-draw parameter batch. Its
        // oracle is the complete, callable entry-point set acquired below
        // together with the per-draw validation each draw already goes through;
        // a batch therefore enables here once the extension object proves its
        // commands, and `multi_draw` falls back to single draws when it does not
        // (audit P1-8).
        builder.resolve(
            GlCapability::MultiDraw,
            CoreOrExtension {
                desktop_core: None,
                embedded_core: None,
                extension: Some(GlKnownExtension::WebglMultiDraw),
                extension_requires_probe: false,
            },
            GlOperationProbe::NotRequired,
        );
        // A context may be lost between any two browser calls. Do not publish
        // a discovery snapshot across that boundary.
        ensure_context_live(&raw, "finish WebGL2 discovery")?;
        Ok(Self {
            canvas,
            raw,
            glow,
            snapshot: builder.build(),
            owner_thread,
            lifecycle: Cell::new(GlContextLifecycle::Active),
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
            renderbuffers: BTreeMap::new(),
            shaders: BTreeMap::new(),
            programs: BTreeMap::new(),
            vertex_arrays: BTreeMap::new(),
            framebuffers: BTreeMap::new(),
            queries: BTreeMap::new(),
            syncs: BTreeMap::new(),
            fences: GlFenceLeaseBook::default(),
            surface: GlSurfaceLeaseBook::new(),
            surface_suspended: false,
            pass: None,
            raster: None,
            bound_vertex_array: None,
            multi_draw: commands.multi_draw,
            timer: commands.timer,
            astc_hdr: commands.astc_hdr,
            active_query: None,
            next_buffer_slot: 0,
            next_texture_slot: 0,
            next_sampler_slot: 0,
            next_renderbuffer_slot: 0,
            next_shader_slot: 0,
            next_program_slot: 0,
            next_vertex_array_slot: 0,
            next_framebuffer_slot: 0,
            next_query_slot: 0,
            next_sync_slot: 0,
            next_surface_slot: 0,
            pixel_store: GlPixelStoreState::DEFAULT,
        })
    }

    /// Returns the discovery evidence bound to the supplied stamp.
    pub(crate) fn snapshot(&self) -> &GlDiscoverySnapshot {
        &self.snapshot
    }

    /// Preflight every future browser provider call inside this module.
    ///
    /// Raw WebGL and glow handles deliberately have no crate-visible borrowing
    /// method, so an executable seam cannot bypass this owner/loss check.
    pub(super) fn assert_provider_ready(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_owner_thread(operation)?;
        let lifecycle = self.lifecycle.get();
        if lifecycle != GlContextLifecycle::Active {
            return Err(GlError::InvalidLifecycle {
                operation,
                lifecycle,
            });
        }
        if self.raw.is_context_lost() {
            // Host owns loss events, while RHI owns the execution boundary.
            // The first provider preflight that observes loss durably records
            // it before returning, preventing another browser API call.
            self.lifecycle.set(GlContextLifecycle::Lost);
            Err(GlError::ContextLost { operation })
        } else {
            Ok(())
        }
    }

    pub(super) fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner_thread {
            Ok(())
        } else {
            Err(GlError::WrongThread {
                operation,
                expected: self.owner_thread,
                actual,
            })
        }
    }

    pub(super) fn allocate_slot(next: &mut u32, operation: &'static str) -> Result<u32, GlError> {
        let slot = *next;
        *next = next.checked_add(1).ok_or_else(|| GlError::Driver {
            operation,
            message: "object allocation slots exhausted".into(),
        })?;
        Ok(slot)
    }

    pub(super) fn validation(operation: &'static str, message: &'static str) -> GlError {
        GlError::Validation {
            operation,
            message: message.into(),
        }
    }

    pub(super) fn driver_error(&self, operation: &'static str) -> Result<(), GlError> {
        let error = self.raw.get_error();
        if error == WebGl2RenderingContext::NO_ERROR {
            Ok(())
        } else {
            Err(GlError::Driver {
                operation,
                message: format!("WebGL error 0x{error:04x}"),
            })
        }
    }

    pub(super) fn buffer(
        &self,
        operation: &'static str,
        id: BufferId,
    ) -> Result<&BrowserBuffer, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.buffers.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "buffer allocation is not live")),
        }
    }

    pub(super) fn texture(
        &self,
        operation: &'static str,
        id: TextureId,
    ) -> Result<&BrowserTexture, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.textures.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(
                operation,
                "texture allocation is not live",
            )),
        }
    }

    /// Maps a JS exception from a catch-typed binding into a driver error.
    pub(super) fn js_failure(operation: &'static str, value: JsValue) -> GlError {
        js_error(operation, value)
    }
}

fn ensure_context_live(
    raw: &WebGl2RenderingContext,
    operation: &'static str,
) -> Result<(), GlError> {
    (!raw.is_context_lost())
        .then_some(())
        .ok_or(GlError::ContextLost { operation })
}

/// The extension objects whose commands must be retained for later calls.
///
/// Both are `Some` exactly when their object exposed every command its domain
/// issues, which is also exactly when that domain's capability could resolve.
struct RetainedCommands {
    multi_draw: Option<BrowserMultiDraw>,
    timer: Option<BrowserTimerQuery>,
    astc_hdr: bool,
}

/// Records evidence for every reported extension this contract models.
///
/// Returns the ledger together with the acquired objects whose commands must be
/// retained: those extensions define their commands on the object itself, so no
/// context method can reach them later.
fn discover_extensions(
    raw: &WebGl2RenderingContext,
) -> Result<(GlExtensionSet, RetainedCommands), GlError> {
    let listed = raw.get_supported_extensions().ok_or_else(|| {
        driver(
            "getSupportedExtensions",
            "browser returned null for a live WebGL2 context",
        )
    })?;
    let mut extensions = GlExtensionSet::default();
    for value in listed.iter() {
        let name = value.as_string().ok_or_else(|| {
            driver(
                "getSupportedExtensions",
                "browser returned a non-string extension name",
            )
        })?;
        extensions.report_raw(name);
    }

    let mut commands = RetainedCommands {
        multi_draw: None,
        timer: None,
        astc_hdr: false,
    };
    for known in [
        GlKnownExtension::ExtDisjointTimerQueryWebgl2,
        GlKnownExtension::ExtColorBufferFloat,
        GlKnownExtension::ExtFloatBlend,
        GlKnownExtension::OesTextureFloatLinear,
        GlKnownExtension::ExtTextureFilterAnisotropic,
        GlKnownExtension::WebglMultiDraw,
        GlKnownExtension::OvrMultiview2,
        GlKnownExtension::KhrParallelShaderCompile,
        GlKnownExtension::CompressedTextureS3tc,
        GlKnownExtension::CompressedTextureS3tcSrgb,
        GlKnownExtension::CompressedTextureBptc,
        GlKnownExtension::CompressedTextureRgtc,
        GlKnownExtension::CompressedTextureAstc,
        GlKnownExtension::CompressedTextureEtc,
    ] {
        if extensions.provenance(known).is_none() {
            continue;
        }
        let object = match raw.get_extension(acquisition_name(known, &extensions)) {
            Ok(Some(object)) => object,
            Ok(None) | Err(_) => {
                // A reported name without a usable extension object is
                // explicitly failed and can never enable a capability.
                extensions.fail(known);
                continue;
            }
        };
        if known == GlKnownExtension::WebglMultiDraw {
            // Its commands live on the object, so the object is the oracle: an
            // object that does not expose all of them fails the acquisition and
            // leaves the batch domain on its single-draw route (audit P1-8).
            match BrowserMultiDraw::acquire(&object) {
                Some(batch) => {
                    extensions.acquire(known);
                    commands.multi_draw = Some(batch);
                }
                None => {
                    extensions.fail(known);
                }
            }
            continue;
        }
        if known == GlKnownExtension::ExtDisjointTimerQueryWebgl2 {
            // Same oracle shape as the batch domain: the timer commands are
            // defined on the object, so a reported name without all five of them
            // fails the acquisition and leaves every timer operation rejected
            // (audit P1-5).
            match BrowserTimerQuery::acquire(&object) {
                Some(timer) => {
                    extensions.acquire(known);
                    commands.timer = Some(timer);
                }
                None => {
                    extensions.fail(known);
                }
            }
            continue;
        }
        if known == GlKnownExtension::CompressedTextureAstc {
            // The ASTC extension object answers `supportedProfiles`; `hdr` is
            // independent from an LDR object acquisition.  This reads the
            // exact object before publishing the private admission bit.
            commands.astc_hdr = astc_hdr_profile(&object);
        }
        extensions.acquire(known);
    }
    Ok((extensions, commands))
}

fn astc_hdr_profile(object: &JsValue) -> bool {
    let Ok(value) = Reflect::get(object, &JsValue::from_str("supportedProfiles")) else {
        return false;
    };
    Array::is_array(&value)
        && Array::from(&value)
            .iter()
            .any(|profile| profile.as_string().is_some_and(|profile| profile == "hdr"))
}

fn acquisition_name(known: GlKnownExtension, extensions: &GlExtensionSet) -> &'static str {
    if known == GlKnownExtension::ExtTextureFilterAnisotropic {
        for alias in [
            "EXT_texture_filter_anisotropic",
            "WEBKIT_EXT_texture_filter_anisotropic",
            "MOZ_EXT_texture_filter_anisotropic",
        ] {
            if extensions
                .raw_reported_names()
                .any(|reported| reported == alias)
            {
                return alias;
            }
        }
    }
    known.raw_name()
}

fn discover_limits(
    raw: &WebGl2RenderingContext,
    extensions: &GlExtensionSet,
    timer: Option<&BrowserTimerQuery>,
) -> Result<GlLimits, GlError> {
    let max_samples = u32_parameter(raw, WebGl2RenderingContext::MAX_SAMPLES, "MAX_SAMPLES")?;
    Ok(GlLimits {
        max_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_TEXTURE_SIZE,
            "MAX_TEXTURE_SIZE",
        )?,
        max_3d_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_3D_TEXTURE_SIZE,
            "MAX_3D_TEXTURE_SIZE",
        )?,
        max_array_texture_layers: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_ARRAY_TEXTURE_LAYERS,
            "MAX_ARRAY_TEXTURE_LAYERS",
        )?,
        max_cube_map_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_CUBE_MAP_TEXTURE_SIZE,
            "MAX_CUBE_MAP_TEXTURE_SIZE",
        )?,
        max_renderbuffer_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_RENDERBUFFER_SIZE,
            "MAX_RENDERBUFFER_SIZE",
        )?,
        max_color_attachments: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COLOR_ATTACHMENTS,
            "MAX_COLOR_ATTACHMENTS",
        )?,
        max_draw_buffers: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_DRAW_BUFFERS,
            "MAX_DRAW_BUFFERS",
        )?,
        max_vertex_attributes: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_ATTRIBS,
            "MAX_VERTEX_ATTRIBS",
        )?,
        max_viewport_dimensions: viewport_dimensions(raw)?,
        max_viewports: 0,
        max_vertex_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_TEXTURE_IMAGE_UNITS,
            "MAX_VERTEX_TEXTURE_IMAGE_UNITS",
        )?,
        max_fragment_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_TEXTURE_IMAGE_UNITS,
            "MAX_TEXTURE_IMAGE_UNITS",
        )?,
        max_combined_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
            "MAX_COMBINED_TEXTURE_IMAGE_UNITS",
        )?,
        max_uniform_buffer_bindings: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_UNIFORM_BUFFER_BINDINGS,
            "MAX_UNIFORM_BUFFER_BINDINGS",
        )?,
        max_uniform_block_size: u64_parameter(
            raw,
            WebGl2RenderingContext::MAX_UNIFORM_BLOCK_SIZE,
            "MAX_UNIFORM_BLOCK_SIZE",
        )?,
        uniform_buffer_offset_alignment: u64_parameter(
            raw,
            WebGl2RenderingContext::UNIFORM_BUFFER_OFFSET_ALIGNMENT,
            "UNIFORM_BUFFER_OFFSET_ALIGNMENT",
        )?,
        max_vertex_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_UNIFORM_BLOCKS,
            "MAX_VERTEX_UNIFORM_BLOCKS",
        )?,
        max_fragment_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_FRAGMENT_UNIFORM_BLOCKS,
            "MAX_FRAGMENT_UNIFORM_BLOCKS",
        )?,
        max_compute_uniform_blocks: 0,
        max_combined_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COMBINED_UNIFORM_BLOCKS,
            "MAX_COMBINED_UNIFORM_BLOCKS",
        )?,
        max_storage_buffer_bindings: 0,
        max_storage_block_size: 0,
        storage_buffer_offset_alignment: 0,
        max_vertex_storage_blocks: 0,
        max_fragment_storage_blocks: 0,
        max_compute_storage_blocks: 0,
        max_combined_storage_blocks: 0,
        max_image_units: 0,
        max_combined_image_units: 0,
        max_samples,
        // WebGL2 has renderbuffer multisampling but no multisample textures.
        // Do not turn MAX_SAMPLES into a texture capability.
        max_color_texture_samples: 0,
        max_depth_texture_samples: 0,
        max_integer_samples: 0,
        max_compute_work_group_count: [0; 3],
        max_compute_work_group_size: [0; 3],
        max_compute_work_group_invocations: 0,
        max_multi_draw_indirect_count: None,
        max_multiview_view_count: multiview_view_limit(raw, extensions),
        // Zero here is the capability row's limit half: it means at least one
        // timer target did not answer a usable width, so the row stays disabled
        // while the ledger keeps recording what the entry-point oracle saw.
        query_counter_bits: exec_timer::recorded_counter_width(raw, timer),
        max_texture_anisotropy: anisotropy_limit(raw, extensions)?,
    })
}

/// The number of views one attachment may serve in one pass.
///
/// The token is defined by the multiview extension family and is invalid until
/// an extension object has been acquired, so a context without it records 0 and
/// can never satisfy the multiview floor. One view is the plain single-view
/// attachment every WebGL2 context already has and is not a multiview proof.
///
/// An acquired object that does not answer the token with a number also records
/// 0 rather than failing discovery: Chrome on an ANGLE/SwiftShader context
/// lists and returns the object while answering the token with `null`, and
/// turning that into a discovery error makes the whole provider unopenable for
/// a capability that is disabled either way (observed while adding the browser
/// tests, not a hypothesis).
fn multiview_view_limit(raw: &WebGl2RenderingContext, extensions: &GlExtensionSet) -> u32 {
    const MAX_VIEWS_OVR: u32 = 0x9632;
    if !extensions.is_acquired(GlKnownExtension::OvrMultiview2) {
        return 0;
    }
    u32_parameter(raw, MAX_VIEWS_OVR, "MAX_VIEWS_OVR")
        // An unanswered token may also be answered with a driver error; that
        // error belongs to this question and must not surface as the failure of
        // the next unrelated provider call.
        .inspect_err(|_| {
            let _ = raw.get_error();
        })
        .unwrap_or(0)
}

fn anisotropy_limit(
    raw: &WebGl2RenderingContext,
    extensions: &GlExtensionSet,
) -> Result<Option<GlFiniteF32>, GlError> {
    // `MAX_TEXTURE_MAX_ANISOTROPY_EXT` has this registry value for all three
    // spellings. It is invalid until an extension object has been acquired.
    const MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;
    if !extensions.is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic) {
        return Ok(None);
    }
    let value = number_parameter(
        raw,
        MAX_TEXTURE_MAX_ANISOTROPY_EXT,
        "MAX_TEXTURE_MAX_ANISOTROPY_EXT",
    )? as f32;
    GlFiniteF32::new(value)
        .ok_or_else(|| {
            driver(
                "getParameter",
                "MAX_TEXTURE_MAX_ANISOTROPY_EXT was not finite f32",
            )
        })
        .map(Some)
}

/// Reads the live context-attribute answers back into the evidence record.
///
/// Every accepted attribute is recorded as an exact `name=value` entry so the
/// snapshot proves what the browser actually granted, including attributes the
/// requester set but the browser could not honor (audit P1-10).
fn discover_context_flags(raw: &WebGl2RenderingContext) -> Result<GlContextFlags, GlError> {
    let mut flags = GlContextFlags::default();
    let Some(attributes) = raw.get_context_attributes() else {
        // A `None` answer gets an explicit marker instead of pretending
        // defaults were observed.
        flags
            .other
            .insert("webgl.context-attributes-unavailable=true".into());
        return Ok(flags);
    };
    let record = |flags: &mut GlContextFlags, name: &str, value: Option<bool>| {
        if let Some(value) = value {
            flags.other.insert(format!("webgl.{name}={value}"));
        }
    };
    record(&mut flags, "alpha", attributes.get_alpha());
    record(&mut flags, "antialias", attributes.get_antialias());
    record(&mut flags, "depth", attributes.get_depth());
    record(&mut flags, "stencil", attributes.get_stencil());
    record(
        &mut flags,
        "premultiplied-alpha",
        attributes.get_premultiplied_alpha(),
    );
    record(
        &mut flags,
        "preserve-drawing-buffer",
        attributes.get_preserve_drawing_buffer(),
    );
    record(
        &mut flags,
        "fail-if-major-performance-caveat",
        attributes.get_fail_if_major_performance_caveat(),
    );
    Ok(flags)
}

fn string_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<String, GlError> {
    raw.get_parameter(pname)
        .map_err(|value| js_error("getParameter", value))?
        .as_string()
        .ok_or_else(|| driver("getParameter", &format!("{name} was not a string")))
}

/// Reads the drawing buffer's colour component widths.
///
/// One observation, and the typed half of what the native path reports as
/// `gl.surface-color-bits`. The browser records none of the `gl.surface-*` keys:
/// that channel exists on the native side to name *why* an observation failed,
/// which is a distinction the typed value deliberately does not carry, and a
/// second renderer with no reasons to report would be a reporting channel with
/// nothing in it.
///
/// A parameter that cannot be read answers [`GlSurfaceFacts::Unavailable`] rather
/// than failing discovery, which is the native path's rule for the same
/// observation: a drawable a presenter cannot describe is a missing surface, not
/// a context that failed to be discovered. The depth, stencil, sample-buffer and
/// sample-count parameters the native path also reads are left to it, because
/// they are read for keys this side does not write and for no field of the typed
/// value.
fn drawing_buffer_facts(raw: &WebGl2RenderingContext) -> GlSurfaceFacts {
    const PARAMETERS: [(u32, &str); 4] = [
        (WebGl2RenderingContext::RED_BITS, "RED_BITS"),
        (WebGl2RenderingContext::GREEN_BITS, "GREEN_BITS"),
        (WebGl2RenderingContext::BLUE_BITS, "BLUE_BITS"),
        (WebGl2RenderingContext::ALPHA_BITS, "ALPHA_BITS"),
    ];
    let mut color_bits = [0u32; PARAMETERS.len()];
    for (slot, (pname, name)) in color_bits.iter_mut().zip(PARAMETERS) {
        match u32_parameter(raw, pname, name) {
            Ok(value) => *slot = value,
            Err(_) => return GlSurfaceFacts::Unavailable,
        }
    }
    GlSurfaceFacts::Observed { color_bits }
}

fn u32_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<u32, GlError> {
    let value = number_parameter(raw, pname, name)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > f64::from(u32::MAX) {
        return Err(driver("getParameter", &format!("{name} was not a u32")));
    }
    Ok(value as u32)
}

fn u64_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<u64, GlError> {
    let value = number_parameter(raw, pname, name)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > 9_007_199_254_740_991.0
    {
        return Err(driver(
            "getParameter",
            &format!("{name} was not an exact JavaScript integer"),
        ));
    }
    Ok(value as u64)
}

fn number_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<f64, GlError> {
    raw.get_parameter(pname)
        .map_err(|value| js_error("getParameter", value))?
        .as_f64()
        .ok_or_else(|| driver("getParameter", &format!("{name} was not a number")))
}

fn viewport_dimensions(raw: &WebGl2RenderingContext) -> Result<[u32; 2], GlError> {
    let value = raw
        .get_parameter(WebGl2RenderingContext::MAX_VIEWPORT_DIMS)
        .map_err(|value| js_error("getParameter", value))?;
    let values = Array::from(&value);
    if values.length() != 2 {
        return Err(driver(
            "getParameter",
            "MAX_VIEWPORT_DIMS did not contain two values",
        ));
    }
    let width = values.get(0).as_f64();
    let height = values.get(1).as_f64();
    match (width, height) {
        (Some(width), Some(height))
            if width.is_finite()
                && height.is_finite()
                && width.fract() == 0.0
                && height.fract() == 0.0
                && width >= 0.0
                && height >= 0.0
                && width <= f64::from(u32::MAX)
                && height <= f64::from(u32::MAX) =>
        {
            Ok([width as u32, height as u32])
        }
        _ => Err(driver(
            "getParameter",
            "MAX_VIEWPORT_DIMS was not a u32 pair",
        )),
    }
}

fn browser_identity() -> Result<String, GlError> {
    let global = js_sys::global();
    let navigator = Reflect::get(&global, &JsValue::from_str("navigator"))
        .map_err(|value| js_error("navigator", value))?;
    let user_agent = Reflect::get(&navigator, &JsValue::from_str("userAgent"))
        .map_err(|value| js_error("navigator.userAgent", value))?;
    user_agent.as_string().ok_or_else(|| {
        driver(
            "navigator.userAgent",
            "browser returned a non-string identity",
        )
    })
}

/// Accepts exactly the `WebGL 2.x` family and rejects look-alike strings.
///
/// The major component must be the single digit `2` followed by `.` or the
/// end of the string, so `"WebGL 20"` and `"WebGL 2foo"` both fail (audit
/// P2-9); minor and vendor text are retained verbatim elsewhere.
pub(super) fn require_webgl2_version(version: &str) -> Result<(), GlError> {
    let accepted = version.strip_prefix("WebGL ").and_then(|remainder| {
        let (major, rest) = remainder.split_at(1.min(remainder.len()));
        (major == "2" && (rest.is_empty() || rest.starts_with('.'))).then_some(())
    });
    accepted.ok_or_else(|| driver("getParameter(VERSION)", "context did not report WebGL 2"))
}

fn discovery_error(error: GlDiscoveryError) -> GlError {
    driver("build WebGL2 discovery", &format!("{error:?}"))
}

fn js_error(operation: &'static str, value: JsValue) -> GlError {
    driver(operation, &format!("browser exception: {value:?}"))
}

fn driver(operation: &'static str, message: &str) -> GlError {
    GlError::Driver {
        operation,
        message: message.to_owned(),
    }
}
