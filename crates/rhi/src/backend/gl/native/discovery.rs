//! Native GL/GLES discovery over a context made current by the Host.
//!
//! This module never creates a window, display, surface, or context.  Its one
//! unsafe boundary is the `glow` adapter: callers must keep the supplied
//! context current on its owning thread for the entire call.  The resulting
//! snapshot is data only and remains bound to the caller supplied stamp.

use super::probes::{
    GlowProbes, NativeGlProbes, ProbeAnswer, ProbeReport, record_extension_probes,
    run_operation_probes,
};
use super::surface_facts;
use crate::backend::gl::api::{
    ContextStamp, CoreOrExtension, GlCapability, GlContextFlags, GlContextInfo, GlDiscoveryBuilder,
    GlDiscoveryError, GlDiscoverySnapshot, GlExtensionSet, GlFamilyProfile, GlFiniteF32, GlFormat,
    GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable, GlKnownExtension,
    GlLimits, GlOperationProbe, GlVersion,
};
use core::ffi::c_void;

/// Narrows one immutable native discovery record to the v13 routes which the
/// native owner actually executes today.
///
/// This deliberately lives beside discovery, rather than on the platform
/// adapters: WGL and EGL must publish the same answer for the same GL record.
/// In particular, an entry point found by discovery is not a public feature.
/// Compute, indirect execution, queries, barriers, and compressed upload stay
/// closed until `driver.rs` has an end-to-end submission route for them.
pub(crate) fn v13_capability_snapshot(
    discovery: &GlDiscoverySnapshot,
) -> crate::backend::gl::capabilities::GlCapabilitySnapshot {
    use crate::api::format::{TextureFormat, TextureSupportLimits, TextureSupportQuery};
    use crate::api::resource::texture::{Extent3d, TextureDimension, TextureUsage};
    use crate::backend::gl::capabilities::{
        GlCapabilitySnapshot, GlFormatEvidence as V13FormatEvidence, GlLoweringClosure,
        GlTextureEvidence,
    };
    use crate::backend::gl::facts::{GlExtension, GlFeatureProbe, GlFunction};

    let limits = discovery.limits();
    let texture_limits = TextureSupportLimits::new(
        Extent3d::d2(limits.max_texture_size, limits.max_texture_size),
        limits
            .max_texture_size
            .checked_ilog2()
            .map_or(1, |value| value + 1),
        1,
    );
    let mut formats = Vec::new();
    let mut textures = Vec::new();
    for format in TextureFormat::all() {
        let Ok(gl_format) = crate::backend::gl::translate::texture_format(format) else {
            continue;
        };
        let Some(evidence) =
            discovery
                .formats()
                .get_for(GlFormatResourceKind::Texture, gl_format, 1)
        else {
            continue;
        };
        // `NativeGlProvider::create_texture_resource` currently owns only
        // single-sample 2D texture storage.  Do not advertise array, 3D, or
        // multisample forms merely because the discovery table contains a
        // driver fact for a related GL object class.
        let depth_or_stencil = matches!(
            format,
            TextureFormat::Depth16Unorm
                | TextureFormat::Depth24Plus
                | TextureFormat::Depth24PlusStencil8
                | TextureFormat::Depth32Float
                | TextureFormat::Depth32FloatStencil8
                | TextureFormat::Stencil8
        );
        formats.push(V13FormatEvidence {
            format,
            storage_read: false,
            storage_write: false,
            storage_read_write: false,
            color_attachment: evidence.renderable && !depth_or_stencil,
            // Native v13 raster lowering currently owns the offscreen color
            // route only. The lower GL framebuffer layer can represent depth
            // and stencil, but Phase A deliberately refuses those attachments;
            // capability facts must follow that executable closure.
            depth_attachment: false,
            stencil_attachment: false,
            blendable: evidence.blendable,
            filterable: evidence.filterable,
            storage_atomic: false,
        });
        for usage in TextureUsage::all() {
            if usage.is_empty() || usage.contains(TextureUsage::STORAGE) {
                continue;
            }
            let rgba8 = matches!(
                format,
                TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
            );
            let compressed = crate::backend::gl::translate::is_compressed_texture_format(format);
            // COPY_SRC includes the public readback route, whose current GL
            // lowering is deliberately RGBA8-only.  Do not let broader native
            // copy-image support advertise an operation the execution driver
            // will reject.  Compressed COPY_DST is the strict whole-mip upload
            // path and remains independently admissible.
            if usage.contains(TextureUsage::COPY_SRC) && (!rgba8 || !evidence.copy_source) {
                continue;
            }
            if usage.contains(TextureUsage::COPY_DST)
                && !(rgba8 && evidence.copy_destination)
                && !compressed
            {
                continue;
            }
            if usage.contains(TextureUsage::COLOR_ATTACHMENT)
                && (!evidence.renderable || depth_or_stencil)
            {
                continue;
            }
            if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
                continue;
            }
            textures.push(GlTextureEvidence {
                query: TextureSupportQuery::new(TextureDimension::D2, format, usage, 1),
                limits: texture_limits,
            });
        }
    }
    let mut feature_probe = GlFeatureProbe::new(discovery.context().profile());
    // `glCompressedTexImage2D` and `glCompressedTexSubImage2D` are core GL
    // entry points at every profile floor we admit.  Exact extension evidence
    // below selects the compressed *internal format*; it does not manufacture
    // a command entry point.  The executor currently uses the whole-mip image
    // route, while recording both core symbols keeps the feature registry's
    // callable-route predicate honest for the shared vocabulary.
    feature_probe.report_function(GlFunction::CompressedTexImage2d);
    feature_probe.report_function(GlFunction::CompressedTexSubImage2d);
    if discovery
        .extensions()
        .is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic)
    {
        feature_probe.report_extension(GlExtension::ExtTextureFilterAnisotropic);
        feature_probe.report_function(GlFunction::TexParameterAnisotropy);
    }
    if discovery
        .capabilities()
        .supports(GlCapability::IndirectDraw)
    {
        // Discovery's capability fact already includes the operation probe.
        // Feed the shared registry the exact callable command pair as well;
        // setting only `lowering.raster_indirect` would otherwise leave a real
        // GL 4.0/GLES 3.1 route conservatively but incorrectly unpublished.
        feature_probe.report_function(GlFunction::DrawArraysIndirect);
        feature_probe.report_function(GlFunction::DrawElementsIndirect);
        if discovery
            .extensions()
            .is_acquired(GlKnownExtension::ArbDrawIndirect)
        {
            feature_probe.report_extension(GlExtension::ArbDrawIndirect);
        }
    }
    if discovery.capabilities().supports(GlCapability::Compute) {
        feature_probe.report_function(GlFunction::DispatchCompute);
        if discovery
            .extensions()
            .is_acquired(GlKnownExtension::ArbComputeShader)
        {
            feature_probe.report_extension(GlExtension::ArbComputeShader);
        }
    }
    for (known, feature) in [
        (
            GlKnownExtension::ExtTextureCompressionS3tc,
            GlExtension::ExtTextureCompressionS3tc,
        ),
        (
            GlKnownExtension::CompressedTextureS3tcSrgb,
            GlExtension::CompressedTextureS3tcSrgb,
        ),
        (
            GlKnownExtension::ExtTextureCompressionRgtc,
            GlExtension::ExtTextureCompressionRgtc,
        ),
        (
            GlKnownExtension::ArbTextureCompressionBptc,
            GlExtension::ArbTextureCompressionBptc,
        ),
        (
            GlKnownExtension::ArbEs3Compatibility,
            GlExtension::ArbEs3Compatibility,
        ),
        (
            GlKnownExtension::OesCompressedEtc2Rgb8Texture,
            GlExtension::OesCompressedEtc2Rgb8Texture,
        ),
        (
            GlKnownExtension::KhrTextureCompressionAstcLdr,
            GlExtension::KhrTextureCompressionAstcLdr,
        ),
        (
            GlKnownExtension::KhrTextureCompressionAstcHdr,
            GlExtension::KhrTextureCompressionAstcHdr,
        ),
    ] {
        if discovery.extensions().is_acquired(known) {
            feature_probe.report_extension(feature);
        }
    }
    GlCapabilitySnapshot {
        profile: discovery.context().profile(),
        // This snapshot is intentionally not an entry-point inventory.  The
        // lowering closure below is the public contract, so no optional
        // feature receives positive evidence before its submit route exists.
        feature_probe,
        limits,
        // Every native allocation is checked against GLsizei before the GL
        // call.  The driver exposes no larger v13 buffer route today.
        maximum_buffer_size: i32::MAX as u64,
        formats,
        textures,
        lowering: GlLoweringClosure {
            buffers: true,
            textures: true,
            buffer_copy: true,
            texture_copy: true,
            // Shader ABI 1.0 has no authoritative logical-binding-to-GL-name
            // table. Resource-bearing programs therefore fail in Phase A and
            // their public binding capabilities must remain unpublished.
            bindings: false,
            // Resource-free compute programs have a complete create/dispatch
            // route. Storage resources remain independently closed by binding
            // facts and the shader-write Phase-A gate.
            compute: true,
            // The Phase-B driver lowers exactly one `Draw*Indirect` record;
            // multi/count variants remain closed because their extension
            // entry points are not part of the common glow route.
            raster_indirect: true,
            // GL query objects can record measurements, but baseline GL4 / ES3
            // has no portable asynchronous query-result-to-buffer command.
            // Do not publish a query feature until QueryResolve can preserve
            // v13 submission ordering without a CPU stall.
            occlusion_query: false,
            timestamp_query: false,
            sampler_anisotropy: limits
                .max_texture_anisotropy
                .is_some_and(|value| value.get() > 1.0),
            // Creation allocates an exact compressed object lazily and the
            // copy domain defines complete mips through compressedTexImage2D.
            // Per-format discovery rows still decide admission.
            compressed_upload: true,
            ..Default::default()
        },
    }
}

/// Failure to obtain a complete native discovery record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeDiscoveryError {
    /// The current context already carried an error before discovery began.
    ///
    /// Discovery never clears an unknown driver error and continues, because
    /// that would make later observations impossible to attribute.
    PreExistingGlError,
    /// A required driver string was absent or malformed.
    InvalidContextString(&'static str),
    /// The driver reported a profile outside Fluxel's native GL-family scope.
    UnsupportedProfile(String),
    /// A required numeric observation failed.
    QueryFailed(&'static str),
    /// The common snapshot contract rejected otherwise collected facts.
    Snapshot(GlDiscoveryError),
}

/// Small mockable subset of native GL used by discovery.
///
/// Implementations must return `None` for a failed query, including a GL
/// error.  This is deliberately not a general command interface.
pub(super) trait NativeGlQuery {
    /// Consumes exactly one pending driver error, returning true when it was
    /// not `GL_NO_ERROR`.
    fn take_error(&self) -> bool;
    fn string(&self, name: u32) -> Option<String>;
    fn integer(&self, name: u32) -> Option<i64>;
    fn integer_pair(&self, name: u32) -> Option<[i64; 2]>;
    fn indexed_integer(&self, name: u32, index: u32) -> Option<i64>;
    fn float(&self, name: u32) -> Option<f32>;
    fn indexed_string(&self, name: u32, index: u32) -> Option<String>;
    /// A parameter of the draw framebuffer's attachment, or `None` when the
    /// driver refused the query.
    ///
    /// A separate entry point rather than an argument to `integer`, because it
    /// is a different GL command and not a different parameter: this is the one
    /// place a core profile still answers what the default framebuffer's
    /// component widths are. The target is always the draw framebuffer, so it
    /// is not a parameter -- `DRAW_FRAMEBUFFER_BINDING` above is the query that
    /// says which object that is.
    fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64>;
}

/// The driver identity recorded when no platform layer supplied one.
///
/// A recorded absence rather than a stand-in. The GL version string is already
/// recorded as `version`, and copying it here is exactly what made the previous
/// record claim a driver identity nobody had observed (audit P2-6): a report
/// that reads `4.6.0 NVIDIA` as the driver cannot tell an observed platform
/// string from a repeated GL one. The marker follows the
/// `gl.context-flags-unavailable` style so an absence stays greppable.
pub(crate) const DRIVER_IDENTITY_UNAVAILABLE: &str = "gl.driver-identity-unavailable=true";

/// Normalizes a platform-supplied driver identity.
///
/// The platform layer is the only place a WGL or EGL driver string exists, so it
/// supplies one or supplies nothing. A blank string is nothing: recording it
/// verbatim would show an empty driver where the truth is that none was read.
pub(super) fn normalized_driver_identity(supplied: &str) -> String {
    let supplied = supplied.trim();
    if supplied.is_empty() {
        DRIVER_IDENTITY_UNAVAILABLE.to_owned()
    } else {
        supplied.to_owned()
    }
}

/// Discovers native facts using an already-current `glow` context.
///
/// Uses the same Host loader that built `glow` for the one registry query glow
/// 0.18 does not bind (`glGetQueryiv`).
///
/// # Safety contract
///
/// The Host/RHI provider must have made `context` current on its owner thread,
/// must keep it current throughout this call, and must serialize access to the
/// context.  `glow` forwards to the current native context; violating that
/// contract is outside Rust's type system and can call an unrelated driver.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) unsafe fn discover_current_glow(
    context: &glow::Context,
    stamp: ContextStamp,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    // SAFETY: the absent loader only keeps `glGetQueryiv` probes closed; no
    // probe calls an entry point the context did not promise.
    unsafe { discover_current_glow_with_loader(context, stamp, |_| core::ptr::null()) }
}

/// [`discover_current_glow`] plus the Host proc loader, used only to resolve
/// `glGetQueryiv` for the timer-query counter width.
///
/// # Safety contract
///
/// Same as [`discover_current_glow`]. The loader must return valid entry
/// points for the currently bound context or null.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) unsafe fn discover_current_glow_with_loader(
    context: &glow::Context,
    stamp: ContextStamp,
    proc_loader: impl Fn(&str) -> *const c_void,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    // SAFETY: forwarded from this function's documented caller contract.
    unsafe {
        discover_current_glow_identified(context, stamp, proc_loader, DRIVER_IDENTITY_UNAVAILABLE)
    }
}

/// [`discover_current_glow_with_loader`] plus the platform driver identity.
///
/// The driver identity is the one fact discovery cannot read from GL: on this
/// family it belongs to the platform layer (the ICD behind WGL, the EGL driver
/// behind an EGL display), so the provider that owns that layer supplies it.
/// Leaving the string empty records [`DRIVER_IDENTITY_UNAVAILABLE`] instead, so
/// a context whose platform layer did not report an identity is visible as an
/// absence rather than as whatever GL happened to answer.
///
/// # Safety contract
///
/// Same as [`discover_current_glow_with_loader`].
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) unsafe fn discover_current_glow_identified(
    context: &glow::Context,
    stamp: ContextStamp,
    proc_loader: impl Fn(&str) -> *const c_void,
    driver_identity: &str,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    use glow::HasContext as _;

    struct GlowQuery<'a>(&'a glow::Context);
    impl NativeGlQuery for GlowQuery<'_> {
        fn take_error(&self) -> bool {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            unsafe { self.0.get_error() != glow::NO_ERROR }
        }
        fn string(&self, name: u32) -> Option<String> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_string(name) };
            (!self.take_error() && !value.is_empty()).then_some(value)
        }
        fn integer(&self, name: u32) -> Option<i64> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_i32(name) };
            // SAFETY: get_error reads the same current context and makes failed
            // optional queries fail closed instead of turning them into support.
            (!self.take_error()).then_some(i64::from(value))
        }
        fn integer_pair(&self, name: u32) -> Option<[i64; 2]> {
            let mut values = [0_i32; 2];
            // SAFETY: upheld by discover_current_glow's current-context contract.
            unsafe { self.0.get_parameter_i32_slice(name, &mut values) };
            // SAFETY: see integer.
            (!self.take_error()).then_some([i64::from(values[0]), i64::from(values[1])])
        }
        fn indexed_integer(&self, name: u32, index: u32) -> Option<i64> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_indexed_i32(name, index) };
            // SAFETY: see integer.
            (!self.take_error()).then_some(i64::from(value))
        }
        fn float(&self, name: u32) -> Option<f32> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_f32(name) };
            // SAFETY: see integer.
            (!self.take_error()).then_some(value)
        }
        fn indexed_string(&self, name: u32, index: u32) -> Option<String> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_indexed_string(name, index) };
            (!self.take_error() && !value.is_empty()).then_some(value)
        }
        fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe {
                self.0.get_framebuffer_attachment_parameter_i32(
                    glow_const::DRAW_FRAMEBUFFER,
                    attachment,
                    name,
                )
            };
            // SAFETY: see integer. The draw-framebuffer target is named rather
            // than left to a default because there is no default: a query
            // against the wrong target would answer about a framebuffer nobody
            // bound and read as an observation of the drawable.
            (!self.take_error()).then_some(i64::from(value))
        }
    }

    let raw_get_query_iv = proc_loader("glGetQueryiv");
    // SAFETY: `glGetQueryiv` has this exact ABI; the null-pointer case only
    // disables the timer-query counter probe.
    let get_query_iv = (!raw_get_query_iv.is_null()).then(|| unsafe {
        core::mem::transmute::<*const c_void, super::probes::GetQueryivFn>(raw_get_query_iv)
    });

    // SAFETY: forwarded from this function's documented caller contract.
    let probes = GlowProbes::new(context, get_query_iv);
    let query = GlowQuery(context);
    // Both wrappers read the same current context and never run concurrently.
    discover_with_query_pair(&query, &probes, stamp, driver_identity)
}

pub(super) fn discover_with_query(
    query: &(impl NativeGlQuery + NativeGlProbes),
    stamp: ContextStamp,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    discover_with_query_identified(query, stamp, DRIVER_IDENTITY_UNAVAILABLE)
}

/// [`discover_with_query`] with a platform driver identity supplied.
pub(super) fn discover_with_query_identified(
    query: &(impl NativeGlQuery + NativeGlProbes),
    stamp: ContextStamp,
    driver_identity: &str,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    discover_with_query_pair(&NativeOnly(query), query, stamp, driver_identity)
}

/// Adapts a combined test fake to the query half only.
struct NativeOnly<'a, Q: NativeGlQuery + NativeGlProbes>(&'a Q);
impl<Q: NativeGlQuery + NativeGlProbes> NativeGlQuery for NativeOnly<'_, Q> {
    fn take_error(&self) -> bool {
        self.0.take_error()
    }
    fn string(&self, name: u32) -> Option<String> {
        self.0.string(name)
    }
    fn integer(&self, name: u32) -> Option<i64> {
        self.0.integer(name)
    }
    fn integer_pair(&self, name: u32) -> Option<[i64; 2]> {
        self.0.integer_pair(name)
    }
    fn indexed_integer(&self, name: u32, index: u32) -> Option<i64> {
        self.0.indexed_integer(name, index)
    }
    fn float(&self, name: u32) -> Option<f32> {
        self.0.float(name)
    }
    fn indexed_string(&self, name: u32, index: u32) -> Option<String> {
        self.0.indexed_string(name, index)
    }
    fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64> {
        self.0.drawable_attachment(attachment, name)
    }
}

fn discover_with_query_pair(
    query: &impl NativeGlQuery,
    probes: &impl NativeGlProbes,
    stamp: ContextStamp,
    driver_identity: &str,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    if query.take_error() {
        return Err(NativeDiscoveryError::PreExistingGlError);
    }
    let version = required_string(query, glow_const::VERSION, "GL_VERSION")?;
    let profile = parse_native_profile(&version)
        .ok_or_else(|| NativeDiscoveryError::UnsupportedProfile(version.clone()))?;
    let glsl = required_string(
        query,
        glow_const::SHADING_LANGUAGE_VERSION,
        "GL_SHADING_LANGUAGE_VERSION",
    )?;
    let vendor = required_string(query, glow_const::VENDOR, "GL_VENDOR")?;
    let renderer = required_string(query, glow_const::RENDERER, "GL_RENDERER")?;
    let mut extensions = extensions(query, profile)?;
    let report = run_operation_probes(probes, profile, &extensions);
    // Acquisition says the entry points were loaded, never that they work, so
    // the ledger advances only for the extensions whose probe really ran and
    // passed. This runs before `limits` and `native_formats`, which both read
    // acquisition, and `is_acquired` covers `Probed`, so neither can regress.
    record_extension_probes(profile, &report, &mut extensions);
    let limits = limits(query, profile, &extensions, &report)?;
    let formats = native_formats(profile, &extensions, &limits, &report)?;
    let mut flags = context_flags(query, profile)?;
    // The drawable's observed format rides the context-flags record, which is
    // the snapshot's only durable free-form fact set. It is an observation about
    // the surface rather than about the context, and it is recorded at all
    // because FBO 0 has no `GlFormatTable` row that could carry it.
    let (surface, surface_strings) = surface_facts::observe(query);
    flags.other.extend(surface_strings);
    // The desktop requirement is recorded next to the observed facts so a report
    // that shows a desktop context can see, in the same record, the GL 4.0
    // family floor (audit P2-12). It is stamped only where it applies: the
    // embedded profile has no such floor, and inventing one there would make the
    // marker describe a requirement that does not exist.
    if matches!(profile, GlFamilyProfile::Desktop { .. }) {
        flags.other.insert(desktop_context_floor_marker());
    }
    let mut builder = GlDiscoveryBuilder::new(
        stamp,
        GlContextInfo::new(
            profile,
            &version,
            glsl,
            vendor,
            renderer,
            // The driver identity is the platform layer's, never a copy of the
            // GL version: the version is already its own field, and a duplicate
            // here reads as an observed driver string to every later report.
            normalized_driver_identity(driver_identity),
            flags,
        ),
        extensions,
        limits,
        formats,
    )
    .map_err(NativeDiscoveryError::Snapshot)?;
    // The typed half of the observation above, recorded beside the strings it
    // came out with rather than read back out of them.
    builder.surface_facts(surface);

    // Capability enablement: a resolved core-or-extension route is necessary
    // but never sufficient. Every optional command domain also records the
    // outcome of its real operation probe from `run_operation_probes`.
    builder.resolve(
        GlCapability::Compute,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbComputeShader),
            extension_requires_probe: true,
        },
        report.compute.to_operation_probe(),
    );
    builder.resolve(
        GlCapability::StorageBuffer,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderStorageBufferObject),
            extension_requires_probe: true,
        },
        report.storage_buffer.to_operation_probe(),
    );
    builder.resolve(
        GlCapability::StorageImage,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 2)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderImageLoadStore),
            extension_requires_probe: true,
        },
        report.storage_image.to_operation_probe(),
    );
    builder.resolve(
        GlCapability::IndirectDraw,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 0)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: None,
            extension_requires_probe: false,
        },
        report.indirect_draw.to_operation_probe(),
    );
    builder.resolve(
        GlCapability::IndirectDispatch,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: None,
            extension_requires_probe: false,
        },
        report.indirect_dispatch.to_operation_probe(),
    );
    // No probe (and no glow entry point) exists for multi-draw-indirect, so
    // the fact stays `NotRun` and the capability can never silently enable.
    builder.resolve(
        GlCapability::MultiDrawIndirect,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: None,
            extension: None,
            extension_requires_probe: false,
        },
        report.multi_draw_indirect.to_operation_probe(),
    );
    // The normalized batch domain has no native route in this contract: no
    // accepted core version supplies a combined per-draw batch command here,
    // and the browser extension that does is WebGL2-only. The row is resolved
    // explicitly so a native context records the absence as a fact instead of
    // leaving it unasked, and every native batch therefore takes the
    // provider's single-draw path (audit P1-8).
    builder.resolve(
        GlCapability::MultiDraw,
        CoreOrExtension {
            desktop_core: None,
            embedded_core: None,
            extension: Some(GlKnownExtension::WebglMultiDraw),
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRun,
    );
    // Multiview has no core route on any accepted profile: the desktop core
    // profile it would need does not exist, and the embedded route is the OVR
    // extension, which is additionally gated on its acquisition. No multiview
    // attachment probe exists, so the fact stays `NotRun` and every profile
    // rejects a multiview pass (audit P1-7).
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
    builder.resolve(
        GlCapability::TimerQuery,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(3, 3)),
            embedded_core: Some(GlVersion::new(3, 0)),
            extension: None,
            extension_requires_probe: false,
        },
        // The counter-width observation is a limit read, not a command probe;
        // `limits.query_counter_bits` carries the real `glGetQueryiv` answer.
        GlOperationProbe::NotRequired,
    );
    Ok(builder.build())
}

/// Records the actual `GL_CONTEXT_FLAGS`/profile mask as evidence context
/// flags, falling back to explicit "unavailable" markers instead of defaults.
fn context_flags(
    query: &impl NativeGlQuery,
    profile: GlFamilyProfile,
) -> Result<GlContextFlags, NativeDiscoveryError> {
    let mut flags = GlContextFlags::default();
    const CONTEXT_FLAGS: u32 = 0x821E;
    const CONTEXT_PROFILE_MASK: u32 = 0x9126;
    const CONTEXT_CORE_PROFILE_BIT: i64 = 0x0000_0001;
    const CONTEXT_COMPATIBILITY_PROFILE_BIT: i64 = 0x0000_0002;
    const CONTEXT_ROBUST_ACCESS: u32 = 0x90F3;
    match profile {
        GlFamilyProfile::Desktop { .. } => {
            if let Some(bits) = query.integer(CONTEXT_FLAGS) {
                flags.other.insert(format!("gl.context-flags=0x{bits:08x}"));
                flags.debug = bits & 0x0000_0002 != 0;
                flags.forward_compatible = bits & 0x0000_0001 != 0;
                flags.no_error = bits & 0x0000_0008 != 0;
            } else {
                flags
                    .other
                    .insert("gl.context-flags-unavailable=true".into());
            }
            if let Some(mask) = query.integer(CONTEXT_PROFILE_MASK) {
                if mask & CONTEXT_CORE_PROFILE_BIT != 0 {
                    flags.other.insert("gl.profile=core".into());
                } else if mask & CONTEXT_COMPATIBILITY_PROFILE_BIT != 0 {
                    flags.other.insert("gl.profile=compatibility".into());
                }
            }
            if let Some(robust) = query.integer(CONTEXT_ROBUST_ACCESS) {
                flags.robust_access = robust != 0;
            }
        }
        GlFamilyProfile::Embedded { .. } | GlFamilyProfile::WebGl2 => {
            // ES and WebGL2 expose no context-flags query to discovery, and
            // robustness context creation is Host-owned; record the absence
            // explicitly instead of pretending defaults were observed.
            flags.other.insert("gl.context-flags=unavailable".into());
        }
    }
    Ok(flags)
}

pub(super) fn required_string(
    query: &impl NativeGlQuery,
    token: u32,
    name: &'static str,
) -> Result<String, NativeDiscoveryError> {
    query
        .string(token)
        .filter(|value| !value.trim().is_empty())
        .ok_or(NativeDiscoveryError::InvalidContextString(name))
}

fn extensions(
    query: &impl NativeGlQuery,
    profile: GlFamilyProfile,
) -> Result<GlExtensionSet, NativeDiscoveryError> {
    let count = nonnegative(
        query.integer(glow_const::NUM_EXTENSIONS),
        "GL_NUM_EXTENSIONS",
    )?;
    let mut result = GlExtensionSet::default();
    for index in 0..count {
        let name = query.indexed_string(glow_const::EXTENSIONS, index).ok_or(
            NativeDiscoveryError::QueryFailed("glGetStringi(GL_EXTENSIONS)"),
        )?;
        result.report_raw(name);
    }
    // Native GL entry points are loaded by the provider before it constructs
    // glow. Record acquisition only for typed, legal names; the operation
    // probes in `run_operation_probes` remain the real enablement evidence.
    //
    // The multiview extension is deliberately absent from this list: the
    // native boundary binds no multiview attachment entry point, so acquisition
    // cannot be proved here and the name stays `Reported`. It is still typed
    // and still recorded, which is what keeps a driver that reports it visible
    // in evidence without letting the reported name enable anything.
    for known in [
        GlKnownExtension::ArbComputeShader,
        GlKnownExtension::ArbShaderStorageBufferObject,
        GlKnownExtension::ArbShaderImageLoadStore,
        GlKnownExtension::ExtTextureFilterAnisotropic,
        GlKnownExtension::OesTextureFloatLinear,
        GlKnownExtension::KhrRobustness,
        GlKnownExtension::KhrDebug,
    ] {
        if known.is_legal_for(profile) && result.provenance(known).is_some() {
            let _ = result.acquire(known);
        }
    }
    Ok(result)
}

/// The desktop GL core version this native family requires, and why.
///
/// This is a recorded policy decision rather than a discovered fact (audit
/// P2-12). The native provider requests the highest available core context and
/// falls back through GL 4.x to 4.0. Individual 4.1+ command domains remain
/// disabled unless their exact core-or-extension route, entry points, limits,
/// and v13 lowering were all observed; accepting GL 4.0 never implies the GL
/// 4.3 feature set.
pub(crate) const REQUIRED_DESKTOP_VERSION: GlVersion = GlVersion::new(4, 0);

/// The recorded marker for [`REQUIRED_DESKTOP_VERSION`].
///
/// It rides the context-flags record so that observed facts travel with the
/// requirement they are read against: a report showing any desktop GL context
/// can see the 4.0 family floor instead of learning it from a second constant.
///
/// The marker builder and the version it is built from are read by the WGL
/// provider too, which asks for this version, enforces it on the actual version
/// string and then requires the snapshot to carry this exact marker.  All three
/// read one value rather than three copies of it (audit P2-12).
pub(crate) fn desktop_context_floor_marker() -> String {
    format!(
        "gl.desktop-context-floor={}.{}",
        REQUIRED_DESKTOP_VERSION.major, REQUIRED_DESKTOP_VERSION.minor
    )
}

/// Whether this profile can allocate multisample texture storage at all.
///
/// Two-dimensional multisample texture storage is a 4.3 / 3.1 core feature.
/// Below that floor the sample-count queries still answer, because they are
/// renderbuffer facts there, so a recorded ceiling is not by itself proof that
/// the allocation exists. The floor is recorded rather than discovered by
/// trying because there is nothing to try: the loader never resolves the entry
/// point below the floor, and the dispatch table calls it through a null
/// pointer, which is not a GL error any caller can be handed.
pub(super) const fn supports_multisample_texture_storage(profile: GlFamilyProfile) -> bool {
    profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
}

/// The sample-count class one texture format belongs to.
///
/// GL bounds a multisample texture's sample count by a ceiling chosen from the
/// format's class rather than by one global number, so the class is what turns
/// a recorded limit into the bound that actually applies. `GlFormat` has no
/// integer format, so the integer arm is unreachable today; it is modelled
/// instead of folded into the color arm because folding it would silently
/// accept an integer format at the color ceiling the day one is added.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TextureSampleClass {
    Color,
    DepthStencil,
    Integer,
}

/// The sample class of one format. Both the record and the allocation gate read
/// this one mapping, so a fact can never be written under one ceiling and
/// checked against another.
pub(super) const fn texture_sample_class(format: GlFormat) -> TextureSampleClass {
    match format {
        GlFormat::Depth16Unorm | GlFormat::Depth24PlusStencil8 | GlFormat::Depth32Float => {
            TextureSampleClass::DepthStencil
        }
        _ => TextureSampleClass::Color,
    }
}

/// The recorded ceiling that governs one sample class.
pub(super) const fn texture_sample_ceiling(class: TextureSampleClass, limits: &GlLimits) -> u32 {
    match class {
        TextureSampleClass::Color => limits.max_color_texture_samples,
        TextureSampleClass::DepthStencil => limits.max_depth_texture_samples,
        TextureSampleClass::Integer => limits.max_integer_samples,
    }
}

fn limits(
    query: &impl NativeGlQuery,
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    report: &ProbeReport,
) -> Result<GlLimits, NativeDiscoveryError> {
    let u = |token, name| nonnegative(query.integer(token), name);
    let pair = query
        .integer_pair(glow_const::MAX_VIEWPORT_DIMS)
        .ok_or(NativeDiscoveryError::QueryFailed("GL_MAX_VIEWPORT_DIMS"))?;
    let optional_routes = optional_limit_routes(profile, extensions);
    // A ceiling is recorded only where the allocation it bounds exists: below
    // the storage floor the value stays at its one-sample fail-closed floor
    // instead of advertising a multisample texture the context cannot make.
    let texture_multisample = supports_multisample_texture_storage(profile);
    // Optional domains must not turn a context that lacks one optional limit
    // token into a failed baseline discovery.  A legal route is only permission
    // to ask, not proof the driver can answer; a refused query therefore
    // records the fail-closed zero fact.  This also contains EGL stacks which
    // report a newer ES version string for an ES 3.0 request but reject the
    // ES3.1-only storage/compute tokens in that actual context.
    let optional = |enabled, token, _name| {
        if enabled {
            Ok(query
                .integer(token)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(0))
        } else {
            Ok(0)
        }
    };
    let indexed = |enabled, token, index, _name| {
        if enabled {
            // See `optional`: an unavailable optional indexed limit is a
            // closed capability fact, not a reason to reject ES3 baseline.
            Ok(query
                .indexed_integer(token, index)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(0))
        } else {
            Ok(0)
        }
    };
    let anisotropy = extensions
        .is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic)
        .then(|| query.float(glow_const::MAX_TEXTURE_MAX_ANISOTROPY_EXT))
        .flatten()
        .and_then(GlFiniteF32::new);
    Ok(GlLimits {
        max_texture_size: u(glow_const::MAX_TEXTURE_SIZE, "GL_MAX_TEXTURE_SIZE")?,
        max_3d_texture_size: u(glow_const::MAX_3D_TEXTURE_SIZE, "GL_MAX_3D_TEXTURE_SIZE")?,
        max_array_texture_layers: u(
            glow_const::MAX_ARRAY_TEXTURE_LAYERS,
            "GL_MAX_ARRAY_TEXTURE_LAYERS",
        )?,
        max_cube_map_texture_size: u(
            glow_const::MAX_CUBE_MAP_TEXTURE_SIZE,
            "GL_MAX_CUBE_MAP_TEXTURE_SIZE",
        )?,
        max_renderbuffer_size: u(
            glow_const::MAX_RENDERBUFFER_SIZE,
            "GL_MAX_RENDERBUFFER_SIZE",
        )?,
        max_color_attachments: u(
            glow_const::MAX_COLOR_ATTACHMENTS,
            "GL_MAX_COLOR_ATTACHMENTS",
        )?,
        max_draw_buffers: u(glow_const::MAX_DRAW_BUFFERS, "GL_MAX_DRAW_BUFFERS")?,
        max_vertex_attributes: u(glow_const::MAX_VERTEX_ATTRIBS, "GL_MAX_VERTEX_ATTRIBS")?,
        max_viewport_dimensions: [
            to_u32(pair[0], "GL_MAX_VIEWPORT_DIMS")?,
            to_u32(pair[1], "GL_MAX_VIEWPORT_DIMS")?,
        ],
        max_viewports: optional(
            matches!(profile, GlFamilyProfile::Desktop { major: 4, minor } if minor >= 1),
            glow_const::MAX_VIEWPORTS,
            "GL_MAX_VIEWPORTS",
        )?,
        max_vertex_texture_image_units: u(
            glow_const::MAX_VERTEX_TEXTURE_IMAGE_UNITS,
            "GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS",
        )?,
        max_fragment_texture_image_units: u(
            glow_const::MAX_TEXTURE_IMAGE_UNITS,
            "GL_MAX_TEXTURE_IMAGE_UNITS",
        )?,
        max_combined_texture_image_units: u(
            glow_const::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
            "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS",
        )?,
        max_uniform_buffer_bindings: u(
            glow_const::MAX_UNIFORM_BUFFER_BINDINGS,
            "GL_MAX_UNIFORM_BUFFER_BINDINGS",
        )?,
        max_uniform_block_size: u64::from(u(
            glow_const::MAX_UNIFORM_BLOCK_SIZE,
            "GL_MAX_UNIFORM_BLOCK_SIZE",
        )?),
        uniform_buffer_offset_alignment: u64::from(u(
            glow_const::UNIFORM_BUFFER_OFFSET_ALIGNMENT,
            "GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT",
        )?),
        max_vertex_uniform_blocks: u(
            glow_const::MAX_VERTEX_UNIFORM_BLOCKS,
            "GL_MAX_VERTEX_UNIFORM_BLOCKS",
        )?,
        max_fragment_uniform_blocks: u(
            glow_const::MAX_FRAGMENT_UNIFORM_BLOCKS,
            "GL_MAX_FRAGMENT_UNIFORM_BLOCKS",
        )?,
        max_compute_uniform_blocks: optional(
            optional_routes.compute,
            glow_const::MAX_COMPUTE_UNIFORM_BLOCKS,
            "GL_MAX_COMPUTE_UNIFORM_BLOCKS",
        )?,
        max_combined_uniform_blocks: u(
            glow_const::MAX_COMBINED_UNIFORM_BLOCKS,
            "GL_MAX_COMBINED_UNIFORM_BLOCKS",
        )?,
        max_storage_buffer_bindings: optional(
            optional_routes.storage,
            glow_const::MAX_SHADER_STORAGE_BUFFER_BINDINGS,
            "GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS",
        )?,
        max_storage_block_size: u64::from(optional(
            optional_routes.storage,
            glow_const::MAX_SHADER_STORAGE_BLOCK_SIZE,
            "GL_MAX_SHADER_STORAGE_BLOCK_SIZE",
        )?),
        storage_buffer_offset_alignment: u64::from(optional(
            optional_routes.storage,
            glow_const::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT,
            "GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT",
        )?),
        max_vertex_storage_blocks: optional(
            optional_routes.storage,
            glow_const::MAX_VERTEX_SHADER_STORAGE_BLOCKS,
            "GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS",
        )?,
        max_fragment_storage_blocks: optional(
            optional_routes.storage,
            glow_const::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,
            "GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS",
        )?,
        max_compute_storage_blocks: optional(
            optional_routes.storage,
            glow_const::MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
            "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS",
        )?,
        max_combined_storage_blocks: optional(
            optional_routes.storage,
            glow_const::MAX_COMBINED_SHADER_STORAGE_BLOCKS,
            "GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS",
        )?,
        max_image_units: optional(
            optional_routes.image,
            glow_const::MAX_IMAGE_UNITS,
            "GL_MAX_IMAGE_UNITS",
        )?,
        max_combined_image_units: optional(
            optional_routes.image,
            glow_const::MAX_COMBINED_IMAGE_UNIFORMS,
            "GL_MAX_COMBINED_IMAGE_UNIFORMS",
        )?,
        max_samples: u(glow_const::MAX_SAMPLES, "GL_MAX_SAMPLES")?,
        max_color_texture_samples: optional(
            texture_multisample,
            glow_const::MAX_COLOR_TEXTURE_SAMPLES,
            "GL_MAX_COLOR_TEXTURE_SAMPLES",
        )?
        .max(1),
        max_depth_texture_samples: optional(
            texture_multisample,
            glow_const::MAX_DEPTH_TEXTURE_SAMPLES,
            "GL_MAX_DEPTH_TEXTURE_SAMPLES",
        )?
        .max(1),
        max_integer_samples: optional(
            texture_multisample,
            glow_const::MAX_INTEGER_SAMPLES,
            "GL_MAX_INTEGER_SAMPLES",
        )?
        .max(1),
        max_compute_work_group_count: [
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                0,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[0]",
            )?,
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                1,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[1]",
            )?,
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                2,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[2]",
            )?,
        ],
        max_compute_work_group_size: [
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                0,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[0]",
            )?,
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                1,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[1]",
            )?,
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                2,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[2]",
            )?,
        ],
        max_compute_work_group_invocations: optional(
            optional_routes.compute,
            glow_const::MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
            "GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS",
        )?,
        // No GL family exposes a portable multi-draw-indirect count limit query,
        // so the fact is `None` here -- and this is the row that decides the
        // domain, not a note beside it: `GlLimits::supports_multi_draw_indirect`
        // is exactly "a queried count of at least one", so with no query to read
        // one from, the limit half answers false on every real context and the
        // capability cannot enable however the route is resolved. The route and
        // the probe are false here too and independently: `GlowProbes` has no
        // multi-draw-indirect probe because glow 0.18 does not bind the entry
        // point, and no executor implements the verb (plan P2-15, narrowly
        // resolved as a fixture-only domain rather than left to be discovered a
        // fourth time).
        max_multi_draw_indirect_count: None,
        // The view count is only queryable on a context that acquired the
        // multiview extension; without it the fact is 0 and the multiview floor
        // cannot be met by any route.
        max_multiview_view_count: optional(
            extensions.is_acquired(GlKnownExtension::OvrMultiview2),
            glow_const::MAX_MULTIVIEW_VIEWS,
            "GL_MAX_VIEWS_OVR",
        )?,
        query_counter_bits: report.query_counter_bits.unwrap_or(0),
        max_texture_anisotropy: anisotropy,
    })
}

/// The resolved legal route for every optional limit family.
///
/// GLES does not inherit desktop `GL_ARB_*` alternatives.  In particular an
/// ES 3.0 context may report vendor aggregate extensions whose names describe
/// ES 3.1 functionality, but it has no legal SSBO/image/compute limit tokens.
/// Keep the route decision before every query so a rejected token cannot be
/// mistaken for a missing device capability after the fact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OptionalLimitRoutes {
    compute: bool,
    storage: bool,
    image: bool,
}

fn optional_limit_routes(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
) -> OptionalLimitRoutes {
    let desktop_extension = |extension: GlKnownExtension| {
        matches!(profile, GlFamilyProfile::Desktop { .. })
            && extension.is_legal_for(profile)
            && extensions.is_acquired(extension)
    };
    OptionalLimitRoutes {
        compute: profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
            || desktop_extension(GlKnownExtension::ArbComputeShader),
        storage: profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
            || desktop_extension(GlKnownExtension::ArbShaderStorageBufferObject),
        image: profile.meets(Some(GlVersion::new(4, 2)), Some(GlVersion::new(3, 1)))
            || desktop_extension(GlKnownExtension::ArbShaderImageLoadStore),
    }
}

#[cfg(test)]
mod optional_limit_route_tests {
    use super::optional_limit_routes;
    use crate::backend::gl::api::{GlExtensionSet, GlFamilyProfile};

    #[test]
    fn gles30_has_no_es31_limit_route() {
        let routes = optional_limit_routes(
            GlFamilyProfile::Embedded { major: 3, minor: 0 },
            &GlExtensionSet::default(),
        );
        assert!(!routes.compute);
        assert!(!routes.storage);
        assert!(!routes.image);
    }

    #[test]
    fn gles31_enables_its_core_limit_routes() {
        let routes = optional_limit_routes(
            GlFamilyProfile::Embedded { major: 3, minor: 1 },
            &GlExtensionSet::default(),
        );
        assert!(routes.compute);
        assert!(routes.storage);
        assert!(routes.image);
    }
}

/// Builds the complete native format table: core baselines plus every fact the
/// operation probes actually observed.
///
/// Float facts are probe facts. Neither RGBA16F nor RGBA32F renderability is
/// core-guaranteed on every accepted native profile (ES 3.x requires
/// `EXT_color_buffer_float` and desktop depends on the exact version), and the
/// same holds for float depth rendering, so all of them are recorded only with
/// `OperationProbed` evidence from real framebuffer-completeness probes.
///
/// Every sample count recorded here is bounded by the limit that governs its
/// class, so a caller can read the presence of a fact as the proof that the
/// context both accepts that count and can allocate it.
pub(super) fn native_formats(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    limits: &GlLimits,
    report: &ProbeReport,
) -> Result<GlFormatTable, NativeDiscoveryError> {
    let mut table = GlFormatTable::default();
    let mut record = |facts: GlFormatCapabilities| {
        table.record(facts).map_err(|error| {
            NativeDiscoveryError::Snapshot(GlDiscoveryError::InvalidFormats(error))
        })
    };
    // RGBA8 sRGB baseline is an unconditional guarantee of every accepted
    // profile. `Rgba8Unorm` is recorded below together with its probed
    // storage-image facts so the table never sees two conflicting records.
    record(GlFormatCapabilities {
        format: GlFormat::Rgba8Srgb,
        resource_kind: GlFormatResourceKind::Texture,
        sample_count: 1,
        evidence: GlFormatEvidence::CoreGuaranteed,
        sampled: true,
        filterable: true,
        renderable: true,
        blendable: true,
        storage_read: false,
        storage_write: false,
        copy_source: true,
        copy_destination: true,
    })?;
    // RGBA8 storage-image facts: recorded when the image load/store probe ran,
    // regardless of outcome, so a failed probe is distinguishable from absent
    // evidence and can never silently enable the capability.
    let rgba8_storage = report.rgba8_storage == ProbeAnswer::Passed;
    let mut rgba8_unorm = rgba8_texture_facts();
    if report.rgba8_storage.ran() {
        rgba8_unorm.evidence = GlFormatEvidence::OperationProbed;
        rgba8_unorm.storage_read = rgba8_storage;
        rgba8_unorm.storage_write = rgba8_storage;
    }
    record(rgba8_unorm)?;
    // RGBA8 renderbuffer facts: color-renderable storage for every accepted
    // profile, at single sample and at the portable multisample counts.
    for sample_count in [1, 4, 8] {
        if sample_count > limits.max_samples {
            continue;
        }
        for format in [GlFormat::Rgba8Unorm, GlFormat::Rgba8Srgb] {
            record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Renderbuffer,
                sample_count,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: false,
                filterable: false,
                renderable: true,
                blendable: true,
                storage_read: false,
                storage_write: false,
                copy_source: false,
                copy_destination: false,
            })?;
        }
    }
    // Multisample texture facts: the same color and depth formats that are
    // guaranteed renderable single-sample storage, recorded up to the ceiling
    // GL records for their class. A count above that ceiling, or a format
    // outside the three, gets no fact at all, which is what makes an
    // unsupported request fail closed at allocation instead of reaching the
    // driver as a plausible-looking allocation that the driver then refuses.
    if supports_multisample_texture_storage(profile) {
        for (format, class) in [
            (GlFormat::Rgba8Unorm, TextureSampleClass::Color),
            (GlFormat::Rgba8Srgb, TextureSampleClass::Color),
            (GlFormat::Depth32Float, TextureSampleClass::DepthStencil),
        ] {
            let ceiling = texture_sample_ceiling(class, limits);
            for sample_count in [2, 4, 8, 16] {
                if sample_count > ceiling {
                    continue;
                }
                record(GlFormatCapabilities {
                    format,
                    resource_kind: GlFormatResourceKind::Texture,
                    sample_count,
                    evidence: GlFormatEvidence::CoreGuaranteed,
                    // Multisample storage is never read by a sampler and never
                    // filtered: it is read by a resolve, which is a framebuffer
                    // blit between whole attachments. The copy fields stay false
                    // for the same reason -- the shared copy word is
                    // single-sample, and claiming it here would describe an
                    // operation that no path in this layer performs.
                    sampled: false,
                    filterable: false,
                    renderable: true,
                    blendable: class == TextureSampleClass::Color,
                    storage_read: false,
                    storage_write: false,
                    copy_source: false,
                    copy_destination: false,
                })?;
            }
        }
    }
    // Float depth facts come only from real attachment probes; without probe
    // evidence renderability stays false instead of an optimistic core claim.
    record(depth_fact(
        GlFormatResourceKind::Texture,
        report.depth_texture_attachment,
    )?)?;
    if report.depth_renderbuffer_attachment.ran() {
        record(depth_fact(
            GlFormatResourceKind::Renderbuffer,
            report.depth_renderbuffer_attachment,
        )?)?;
    }
    // RGBA8 storage-image facts: (recorded above with the unorm baseline)
    // Float color facts are recorded only when their attachment probe ran:
    // a failed probe records `renderable: false` with operation evidence, and
    // an absent probe backend leaves the format out of the table entirely
    // (the contract rejects an unsupported core guarantee for float formats).
    let float32_filterable = match profile {
        // Desktop GL 4.x core lists both float formats as texture-filterable.
        GlFamilyProfile::Desktop { .. } => true,
        // ES 3.x requires OES_texture_float_linear for 32F filtering.
        _ => extensions.is_acquired(GlKnownExtension::OesTextureFloatLinear),
    };
    if report.rgba16f_attachment.ran() {
        record(float_fact(
            GlFormat::Rgba16Float,
            // Half-float textures are texture-filterable in every accepted core.
            true,
            report.rgba16f_attachment,
            // ES forbids blending with float attachments until EXT_float_blend;
            // desktop core permits 16F blending wherever rendering is legal.
            matches!(profile, GlFamilyProfile::Desktop { .. }),
        )?)?;
    }
    if report.rgba32f_attachment.ran() {
        record(float_fact(
            GlFormat::Rgba32Float,
            float32_filterable,
            report.rgba32f_attachment,
            // 32F blending additionally requires EXT_float_blend, whose typed
            // route exists only for the browser profile; native ES stays false.
            matches!(profile, GlFamilyProfile::Desktop { .. }),
        )?)?;
    }
    // Compressed textures are exact format facts, never a family-level flag.
    // GLES 3.x supplies ETC2/EAC in core; all other native families below are
    // admitted only when their matching extension was acquired on this same
    // context.  Allocation and compressedTexImage2D share the token mapping
    // in `provider`, so a positive row is an executable upload route rather
    // than a version-string promise.
    for format in [
        GlFormat::Etc2Rgb8Unorm,
        GlFormat::Etc2Rgb8Srgb,
        GlFormat::Etc2Rgba8Unorm,
        GlFormat::Etc2Rgba8Srgb,
        GlFormat::Etc2Rgb8A1Unorm,
        GlFormat::Etc2Rgb8A1Srgb,
        GlFormat::EacR11Unorm,
        GlFormat::EacRg11Unorm,
        GlFormat::EacR11Snorm,
        GlFormat::EacRg11Snorm,
    ] {
        if format.is_core_compressed_for(profile) {
            let direct_copy = compressed_direct_copy(profile);
            record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: false,
                renderable: false,
                blendable: false,
                storage_read: false,
                storage_write: false,
                copy_source: direct_copy,
                copy_destination: direct_copy,
            })?;
        }
    }
    for format in native_optional_compressed_formats() {
        if format.is_core_compressed_for(profile) {
            continue;
        }
        let Some(extension) = native_compressed_extension(format, extensions) else {
            continue;
        };
        // `glCopyImageSubData` is a real compressed-to-compressed route only
        // on desktop GL 4.3+. GLES 3.0 and WebGL2 deliberately keep these
        // fields false: neither has this v13 route, and framebuffer copying
        // cannot attach compressed images.
        let direct_copy = compressed_direct_copy(profile);
        record(GlFormatCapabilities {
            format,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::ExtensionAcquired(extension),
            sampled: true,
            // GL's compressed formats are sampleable and filterable. They are
            // never FBO attachments or storage images in this contract.
            filterable: true,
            renderable: false,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: direct_copy,
            copy_destination: direct_copy,
        })?;
    }
    Ok(table)
}

/// Exact optional compressed formats understood by the native storage map.
/// Keeping this list next to native discovery makes additions require review
/// of evidence, allocation and upload together.
fn native_optional_compressed_formats() -> Vec<GlFormat> {
    use crate::backend::gl::api::{GlAstcBlock as B, GlCompressedColorSpace as C};
    let mut formats = vec![
        GlFormat::Bc1RgbUnorm,
        GlFormat::Bc1RgbaUnorm,
        GlFormat::Bc2RgbaUnorm,
        GlFormat::Bc3RgbaUnorm,
        GlFormat::Bc1RgbSrgb,
        GlFormat::Bc1RgbaSrgb,
        GlFormat::Bc2RgbaSrgb,
        GlFormat::Bc3RgbaSrgb,
        GlFormat::Bc4RUnorm,
        GlFormat::Bc4RSnorm,
        GlFormat::Bc5RgUnorm,
        GlFormat::Bc5RgSnorm,
        GlFormat::Bc6hRgbUfloat,
        GlFormat::Bc6hRgbSfloat,
        GlFormat::Bc7RgbaUnorm,
        GlFormat::Bc7RgbaSrgb,
    ];
    for block in [
        B::B4x4,
        B::B5x4,
        B::B5x5,
        B::B6x5,
        B::B6x6,
        B::B8x5,
        B::B8x6,
        B::B8x8,
        B::B10x5,
        B::B10x6,
        B::B10x8,
        B::B10x10,
        B::B12x10,
        B::B12x12,
    ] {
        formats.push(GlFormat::Astc {
            block,
            color_space: C::Linear,
        });
        formats.push(GlFormat::Astc {
            block,
            color_space: C::Srgb,
        });
        formats.push(GlFormat::Astc {
            block,
            color_space: C::Hdr,
        });
    }
    formats.extend([
        GlFormat::Etc2Rgb8Unorm,
        GlFormat::Etc2Rgb8Srgb,
        GlFormat::Etc2Rgba8Unorm,
        GlFormat::Etc2Rgba8Srgb,
        GlFormat::Etc2Rgb8A1Unorm,
        GlFormat::Etc2Rgb8A1Srgb,
        GlFormat::EacR11Unorm,
        GlFormat::EacRg11Unorm,
        GlFormat::EacR11Snorm,
        GlFormat::EacRg11Snorm,
    ]);
    formats
}

fn native_compressed_extension(
    format: GlFormat,
    extensions: &GlExtensionSet,
) -> Option<GlKnownExtension> {
    use crate::backend::gl::api::GlCompressedColorSpace as C;
    let candidates: &[GlKnownExtension] = match format {
        GlFormat::Bc1RgbUnorm
        | GlFormat::Bc1RgbaUnorm
        | GlFormat::Bc2RgbaUnorm
        | GlFormat::Bc3RgbaUnorm => &[GlKnownExtension::ExtTextureCompressionS3tc],
        GlFormat::Bc1RgbSrgb
        | GlFormat::Bc1RgbaSrgb
        | GlFormat::Bc2RgbaSrgb
        | GlFormat::Bc3RgbaSrgb => &[GlKnownExtension::CompressedTextureS3tcSrgb],
        GlFormat::Bc4RUnorm | GlFormat::Bc4RSnorm | GlFormat::Bc5RgUnorm | GlFormat::Bc5RgSnorm => {
            &[GlKnownExtension::ExtTextureCompressionRgtc]
        }
        GlFormat::Bc6hRgbUfloat
        | GlFormat::Bc6hRgbSfloat
        | GlFormat::Bc7RgbaUnorm
        | GlFormat::Bc7RgbaSrgb => &[GlKnownExtension::ArbTextureCompressionBptc],
        GlFormat::Astc {
            color_space: C::Linear | C::Srgb,
            ..
        } => &[GlKnownExtension::KhrTextureCompressionAstcLdr],
        GlFormat::Astc {
            color_space: C::Hdr,
            ..
        } => &[GlKnownExtension::KhrTextureCompressionAstcHdr],
        GlFormat::Etc2Rgb8Unorm
        | GlFormat::Etc2Rgb8Srgb
        | GlFormat::Etc2Rgba8Unorm
        | GlFormat::Etc2Rgba8Srgb
        | GlFormat::Etc2Rgb8A1Unorm
        | GlFormat::Etc2Rgb8A1Srgb
        | GlFormat::EacR11Unorm
        | GlFormat::EacRg11Unorm
        | GlFormat::EacR11Snorm
        | GlFormat::EacRg11Snorm => &[
            GlKnownExtension::ArbEs3Compatibility,
            GlKnownExtension::OesCompressedEtc2Rgb8Texture,
        ],
        _ => &[],
    };
    candidates
        .iter()
        .copied()
        .find(|extension| extensions.is_acquired(*extension))
}

/// The only GL-family image-copy command accepted for compressed storage.
/// Framebuffer copying has no compressed attachment route, so GLES 3.0/3.1
/// intentionally publish upload-only compressed facts.
const fn compressed_direct_copy(profile: GlFamilyProfile) -> bool {
    match profile {
        GlFamilyProfile::Desktop { major, minor } => major > 4 || (major == 4 && minor >= 3),
        GlFamilyProfile::Embedded { major, minor } => major > 3 || (major == 3 && minor >= 2),
        GlFamilyProfile::WebGl2 => false,
    }
}

fn rgba8_texture_facts() -> GlFormatCapabilities {
    GlFormatCapabilities {
        format: GlFormat::Rgba8Unorm,
        resource_kind: GlFormatResourceKind::Texture,
        sample_count: 1,
        evidence: GlFormatEvidence::CoreGuaranteed,
        sampled: true,
        filterable: true,
        renderable: true,
        blendable: true,
        storage_read: false,
        storage_write: false,
        copy_source: true,
        copy_destination: true,
    }
}

/// Builds the `Depth32Float` fact for one resource kind from its probe.
///
/// The common contract requires a single-sample depth fact in every snapshot;
/// when no probe evidence exists the record stays conservative (`renderable`
/// false) instead of claiming the profile guarantees float depth rendering.
fn depth_fact(
    resource_kind: GlFormatResourceKind,
    probe: ProbeAnswer,
) -> Result<GlFormatCapabilities, NativeDiscoveryError> {
    let renderable = probe == ProbeAnswer::Passed;
    let evidence = if probe.ran() {
        GlFormatEvidence::OperationProbed
    } else {
        GlFormatEvidence::CoreGuaranteed
    };
    Ok(GlFormatCapabilities {
        format: GlFormat::Depth32Float,
        resource_kind,
        sample_count: 1,
        evidence,
        sampled: resource_kind == GlFormatResourceKind::Texture,
        filterable: false,
        renderable,
        blendable: false,
        storage_read: false,
        storage_write: false,
        copy_source: false,
        copy_destination: false,
    })
}

/// Builds one float color format fact from its attachment probe. Callers only
/// invoke this when the probe ran, so the evidence is always operation based.
fn float_fact(
    format: GlFormat,
    filterable: bool,
    probe: ProbeAnswer,
    blend_when_renderable: bool,
) -> Result<GlFormatCapabilities, NativeDiscoveryError> {
    let renderable = probe == ProbeAnswer::Passed;
    Ok(GlFormatCapabilities {
        format,
        resource_kind: GlFormatResourceKind::Texture,
        sample_count: 1,
        // Float textures are samplable on every accepted core; the record's
        // evidence marks the concrete attachment probe that proved rendering.
        evidence: GlFormatEvidence::OperationProbed,
        sampled: true,
        filterable,
        renderable,
        blendable: renderable && blend_when_renderable,
        storage_read: false,
        storage_write: false,
        copy_source: renderable,
        copy_destination: renderable,
    })
}

fn nonnegative(value: Option<i64>, name: &'static str) -> Result<u32, NativeDiscoveryError> {
    value
        .ok_or(NativeDiscoveryError::QueryFailed(name))
        .and_then(|value| to_u32(value, name))
}
fn to_u32(value: i64, name: &'static str) -> Result<u32, NativeDiscoveryError> {
    u32::try_from(value).map_err(|_| NativeDiscoveryError::QueryFailed(name))
}

/// Parses only Fluxel's supported native profiles: GL 4.x and GLES 3.x.
pub(crate) fn parse_native_profile(version: &str) -> Option<GlFamilyProfile> {
    let embedded = version.strip_prefix("OpenGL ES ");
    let text = embedded.unwrap_or(version);
    let mut digits = text
        .split(|character: char| !character.is_ascii_digit() && character != '.')
        .find(|part| part.contains('.'))?
        .split('.');
    let major = digits.next()?.parse().ok()?;
    let minor = digits.next()?.parse().ok()?;
    match (embedded.is_some(), major) {
        (false, 4) => Some(GlFamilyProfile::Desktop { major, minor }),
        (true, 3) => Some(GlFamilyProfile::Embedded { major, minor }),
        _ => None,
    }
}

pub(super) mod glow_const {
    pub const VENDOR: u32 = 0x1F00;
    pub const RENDERER: u32 = 0x1F01;
    pub const VERSION: u32 = 0x1F02;
    pub const EXTENSIONS: u32 = 0x1F03;
    pub const SHADING_LANGUAGE_VERSION: u32 = 0x8B8C;
    pub const NUM_EXTENSIONS: u32 = 0x821D;
    pub const MAX_TEXTURE_SIZE: u32 = 0x0D33;
    pub const MAX_3D_TEXTURE_SIZE: u32 = 0x8073;
    pub const MAX_ARRAY_TEXTURE_LAYERS: u32 = 0x88FF;
    pub const MAX_CUBE_MAP_TEXTURE_SIZE: u32 = 0x851C;
    pub const MAX_RENDERBUFFER_SIZE: u32 = 0x84E8;
    pub const MAX_COLOR_ATTACHMENTS: u32 = 0x8CDF;
    pub const MAX_DRAW_BUFFERS: u32 = 0x8824;
    pub const MAX_VERTEX_ATTRIBS: u32 = 0x8869;
    pub const MAX_VIEWPORT_DIMS: u32 = 0x0D3A;
    pub const MAX_VIEWPORTS: u32 = 0x825B;
    pub const MAX_VERTEX_TEXTURE_IMAGE_UNITS: u32 = 0x8B4C;
    pub const MAX_TEXTURE_IMAGE_UNITS: u32 = 0x8872;
    pub const MAX_COMBINED_TEXTURE_IMAGE_UNITS: u32 = 0x8B4D;
    pub const MAX_UNIFORM_BUFFER_BINDINGS: u32 = 0x8A2F;
    pub const MAX_UNIFORM_BLOCK_SIZE: u32 = 0x8A30;
    pub const UNIFORM_BUFFER_OFFSET_ALIGNMENT: u32 = 0x8A34;
    pub const MAX_VERTEX_UNIFORM_BLOCKS: u32 = 0x8A2B;
    pub const MAX_FRAGMENT_UNIFORM_BLOCKS: u32 = 0x8A2D;
    pub const MAX_COMPUTE_UNIFORM_BLOCKS: u32 = 0x91BB;
    pub const MAX_COMBINED_UNIFORM_BLOCKS: u32 = 0x8A2E;
    pub const MAX_SHADER_STORAGE_BUFFER_BINDINGS: u32 = 0x90DD;
    pub const MAX_SHADER_STORAGE_BLOCK_SIZE: u32 = 0x90DE;
    pub const SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT: u32 = 0x90DF;
    pub const MAX_VERTEX_SHADER_STORAGE_BLOCKS: u32 = 0x90D6;
    pub const MAX_FRAGMENT_SHADER_STORAGE_BLOCKS: u32 = 0x90DA;
    pub const MAX_COMPUTE_SHADER_STORAGE_BLOCKS: u32 = 0x90DB;
    pub const MAX_COMBINED_SHADER_STORAGE_BLOCKS: u32 = 0x90DC;
    pub const MAX_IMAGE_UNITS: u32 = 0x8F38;
    pub const MAX_COMBINED_IMAGE_UNIFORMS: u32 = 0x90CF;
    pub const MAX_SAMPLES: u32 = 0x8D57;
    pub const MAX_COLOR_TEXTURE_SAMPLES: u32 = 0x910E;
    pub const MAX_DEPTH_TEXTURE_SAMPLES: u32 = 0x910F;
    pub const MAX_INTEGER_SAMPLES: u32 = 0x9110;
    /// Views one attachment may serve in one pass, as defined by the multiview
    /// extension family (the same registry value WebGL2 exposes).
    pub const MAX_MULTIVIEW_VIEWS: u32 = 0x9632;
    pub const MAX_COMPUTE_WORK_GROUP_COUNT: u32 = 0x91BE;
    pub const MAX_COMPUTE_WORK_GROUP_SIZE: u32 = 0x91BF;
    pub const MAX_COMPUTE_WORK_GROUP_INVOCATIONS: u32 = 0x90EB;
    pub const MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;
    /// The framebuffer bound for drawing, which tells the drawable queries
    /// below whether they would answer about the surface at all.
    pub const DRAW_FRAMEBUFFER_BINDING: u32 = 0x8CA6;
    /// The draw framebuffer itself, as the target an attachment query is asked
    /// about. The same object the binding query above reads the identity of.
    pub const DRAW_FRAMEBUFFER: u32 = 0x8CA9;
    pub const SAMPLE_BUFFERS: u32 = 0x80A8;
    pub const SAMPLES: u32 = 0x80A9;
    /// The drawable's colour buffers and its depth and stencil attachments.
    ///
    /// The component widths are read from an attachment rather than from the
    /// default-framebuffer `GL_*_BITS` queries this table used to carry: those
    /// were deprecated in 3.0 and removed from the core profile in 3.1, so on
    /// the core profile this crate accepts they answer `GL_INVALID_ENUM`, and
    /// `glGetFramebufferAttachmentParameteriv` is the question the profile
    /// still answers. The samples pair above is unaffected -- the core profile
    /// kept it, which is visible on the hardware this was found on, where the
    /// six bit queries failed and these two answered.
    ///
    /// Values cross-checked against two independent bindings, because a wrong
    /// token here is a question asked about nothing: `glow` 0.18 and `web-sys`
    /// 0.3 agree on all six sizes and on the object type, and the two colour
    /// buffers are consistent with the `FRONT`/`BACK` values the latter records
    /// (`GL_FRONT` is `0x0404`, so `GL_FRONT_LEFT`/`GL_BACK_LEFT` are the two
    /// entries below it).
    pub const BACK_LEFT: u32 = 0x0402;
    pub const FRONT_LEFT: u32 = 0x0400;
    pub const DEPTH: u32 = 0x1801;
    pub const STENCIL: u32 = 0x1802;
    pub const FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE: u32 = 0x8CD0;
    pub const FRAMEBUFFER_ATTACHMENT_RED_SIZE: u32 = 0x8212;
    pub const FRAMEBUFFER_ATTACHMENT_GREEN_SIZE: u32 = 0x8213;
    pub const FRAMEBUFFER_ATTACHMENT_BLUE_SIZE: u32 = 0x8214;
    pub const FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: u32 = 0x8215;
    pub const FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: u32 = 0x8216;
    pub const FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: u32 = 0x8217;
}
