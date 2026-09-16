//! Native GL/GLES discovery over a context made current by the Host.
//!
//! This module never creates a window, display, surface, or context.  Its one
//! unsafe boundary is the `glow` adapter: callers must keep the supplied
//! context current on its owning thread for the entire call.  The resulting
//! snapshot is data only and remains bound to the caller supplied stamp.

use core::ffi::c_void;

use super::super::{
    ContextStamp, CoreOrExtension, GlCapability, GlContextFlags, GlContextInfo, GlDiscoveryBuilder,
    GlDiscoveryError, GlDiscoverySnapshot, GlExtensionSet, GlFamilyProfile, GlFiniteF32, GlFormat,
    GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable, GlKnownExtension,
    GlLimits, GlOperationProbe, GlVersion,
};
use super::probes::{
    GlowProbes, NativeGlProbes, ProbeAnswer, ProbeReport, record_extension_probes,
    run_operation_probes,
};

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
    discover_with_query_pair(&query, &probes, stamp)
}

pub(super) fn discover_with_query(
    query: &(impl NativeGlQuery + NativeGlProbes),
    stamp: ContextStamp,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    discover_with_query_pair(&NativeOnly(query), query, stamp)
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
}

fn discover_with_query_pair(
    query: &impl NativeGlQuery,
    probes: &impl NativeGlProbes,
    stamp: ContextStamp,
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
    let formats = native_formats(profile, &extensions, limits.max_samples, &report)?;
    let mut builder = GlDiscoveryBuilder::new(
        stamp,
        GlContextInfo::new(
            profile,
            &version,
            glsl,
            vendor,
            renderer,
            version.clone(),
            context_flags(query, profile)?,
        ),
        extensions,
        limits,
        formats,
    )
    .map_err(NativeDiscoveryError::Snapshot)?;

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
    let compute = profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
        || extensions.is_acquired(GlKnownExtension::ArbComputeShader);
    let storage = profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
        || extensions.is_acquired(GlKnownExtension::ArbShaderStorageBufferObject);
    let image = profile.meets(Some(GlVersion::new(4, 2)), Some(GlVersion::new(3, 1)))
        || extensions.is_acquired(GlKnownExtension::ArbShaderImageLoadStore);
    let texture_multisample = matches!(profile, GlFamilyProfile::Desktop { .. })
        || profile.meets(None, Some(GlVersion::new(3, 1)));
    let optional = |enabled, token, name| if enabled { u(token, name) } else { Ok(0) };
    let indexed = |enabled, token, index, name| {
        if enabled {
            nonnegative(query.indexed_integer(token, index), name)
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
            compute,
            glow_const::MAX_COMPUTE_UNIFORM_BLOCKS,
            "GL_MAX_COMPUTE_UNIFORM_BLOCKS",
        )?,
        max_combined_uniform_blocks: u(
            glow_const::MAX_COMBINED_UNIFORM_BLOCKS,
            "GL_MAX_COMBINED_UNIFORM_BLOCKS",
        )?,
        max_storage_buffer_bindings: optional(
            storage,
            glow_const::MAX_SHADER_STORAGE_BUFFER_BINDINGS,
            "GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS",
        )?,
        max_storage_block_size: u64::from(optional(
            storage,
            glow_const::MAX_SHADER_STORAGE_BLOCK_SIZE,
            "GL_MAX_SHADER_STORAGE_BLOCK_SIZE",
        )?),
        storage_buffer_offset_alignment: u64::from(optional(
            storage,
            glow_const::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT,
            "GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT",
        )?),
        max_vertex_storage_blocks: optional(
            storage,
            glow_const::MAX_VERTEX_SHADER_STORAGE_BLOCKS,
            "GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS",
        )?,
        max_fragment_storage_blocks: optional(
            storage,
            glow_const::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,
            "GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS",
        )?,
        max_compute_storage_blocks: optional(
            storage,
            glow_const::MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
            "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS",
        )?,
        max_combined_storage_blocks: optional(
            storage,
            glow_const::MAX_COMBINED_SHADER_STORAGE_BLOCKS,
            "GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS",
        )?,
        max_image_units: optional(image, glow_const::MAX_IMAGE_UNITS, "GL_MAX_IMAGE_UNITS")?,
        max_combined_image_units: optional(
            image,
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
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                0,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[0]",
            )?,
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                1,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[1]",
            )?,
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                2,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[2]",
            )?,
        ],
        max_compute_work_group_size: [
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                0,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[0]",
            )?,
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                1,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[1]",
            )?,
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                2,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[2]",
            )?,
        ],
        max_compute_work_group_invocations: optional(
            compute,
            glow_const::MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
            "GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS",
        )?,
        // No GL family exposes a portable multi-draw-indirect count limit
        // query; the honest fact stays `None` (unbounded/unqueried), and
        // enablement is governed by route evidence plus the operation probe.
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

/// Builds the complete native format table: core baselines plus every fact the
/// operation probes actually observed.
///
/// Float facts are probe facts. Neither RGBA16F nor RGBA32F renderability is
/// core-guaranteed on every accepted native profile (ES 3.x requires
/// `EXT_color_buffer_float` and desktop depends on the exact version), and the
/// same holds for float depth rendering, so all of them are recorded only with
/// `OperationProbed` evidence from real framebuffer-completeness probes.
pub(super) fn native_formats(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    max_samples: u32,
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
        if sample_count > max_samples {
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
    // Core compressed guarantees are unchanged.
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
                copy_source: false,
                copy_destination: false,
            })?;
        }
    }
    Ok(table)
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
}
