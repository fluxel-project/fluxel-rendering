//! Native GL/GLES discovery over a context made current by the Host.
//!
//! This module never creates a window, display, surface, or context.  Its one
//! unsafe boundary is the `glow` adapter: callers must keep the supplied
//! context current on its owning thread for the entire call.  The resulting
//! snapshot is data only and remains bound to the caller supplied stamp.

use super::super::{
    ContextStamp, CoreOrExtension, GlCapability, GlContextFlags, GlContextInfo, GlDiscoveryBuilder,
    GlDiscoveryError, GlDiscoverySnapshot, GlExtensionSet, GlFamilyProfile, GlFiniteF32, GlFormat,
    GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable, GlKnownExtension,
    GlLimits, GlOperationProbe, GlVersion,
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

    // SAFETY: forwarded from this function's documented caller contract.
    discover_with_query(&GlowQuery(context), stamp)
}

pub(super) fn discover_with_query(
    query: &impl NativeGlQuery,
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
    let extensions = extensions(query, profile)?;
    let limits = limits(query, profile, &extensions)?;
    let formats = baseline_formats(profile)?;
    let mut builder = GlDiscoveryBuilder::new(
        stamp,
        GlContextInfo::new(
            profile,
            &version,
            glsl,
            vendor,
            renderer,
            version.clone(),
            GlContextFlags::default(),
        ),
        extensions,
        limits,
        formats,
    )
    .map_err(NativeDiscoveryError::Snapshot)?;

    // Version and limit reads are evidence only, not executable operation
    // probes.  Keep every optional command domain disabled until the provider
    // performs a separately recorded compile/link/bind/dispatch probe.
    builder.resolve(
        GlCapability::Compute,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbComputeShader),
            extension_requires_probe: true,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::StorageBuffer,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderStorageBufferObject),
            extension_requires_probe: true,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::StorageImage,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 2)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderImageLoadStore),
            extension_requires_probe: true,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::IndirectDraw,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 0)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: None,
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::IndirectDispatch,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: None,
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::MultiDrawIndirect,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: None,
            extension: None,
            extension_requires_probe: false,
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
        GlOperationProbe::NotRequired,
    );
    Ok(builder.build())
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
    // glow.  Record that acquisition only for typed, legal names; command
    // probes are intentionally left absent and therefore cannot enable an
    // extension-only capability.
    for known in [
        GlKnownExtension::ArbComputeShader,
        GlKnownExtension::ArbShaderStorageBufferObject,
        GlKnownExtension::ArbShaderImageLoadStore,
        GlKnownExtension::ExtTextureFilterAnisotropic,
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
        max_multi_draw_indirect_count: None,
        // GL_QUERY_COUNTER_BITS is queried with glGetQueryiv(target, pname),
        // not glGetIntegerv. The small discovery trait intentionally has no
        // query-object API, so preserve it as unavailable instead of issuing
        // an invalid query or inferring timer support from a version string.
        query_counter_bits: 0,
        max_texture_anisotropy: anisotropy,
    })
}

pub(super) fn baseline_formats(
    profile: GlFamilyProfile,
) -> Result<GlFormatTable, NativeDiscoveryError> {
    let mut table = GlFormatTable::default();
    for format in [
        GlFormat::Rgba8Unorm,
        GlFormat::Rgba8Srgb,
        GlFormat::Depth32Float,
    ] {
        table
            .record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                // These are the unconditional profile baseline, not driver
                // operation probes.  Optional formats stay absent until a
                // provider records a real operation probe.
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: format != GlFormat::Depth32Float,
                renderable: true,
                blendable: format != GlFormat::Depth32Float,
                storage_read: false,
                storage_write: false,
                copy_source: format != GlFormat::Depth32Float,
                copy_destination: format != GlFormat::Depth32Float,
            })
            .map_err(|error| {
                NativeDiscoveryError::Snapshot(GlDiscoveryError::InvalidFormats(error))
            })?;
    }
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
            table
                .record(GlFormatCapabilities {
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
                })
                .map_err(|error| {
                    NativeDiscoveryError::Snapshot(GlDiscoveryError::InvalidFormats(error))
                })?;
        }
    }
    Ok(table)
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
    pub const MAX_COMPUTE_WORK_GROUP_COUNT: u32 = 0x91BE;
    pub const MAX_COMPUTE_WORK_GROUP_SIZE: u32 = 0x91BF;
    pub const MAX_COMPUTE_WORK_GROUP_INVOCATIONS: u32 = 0x90EB;
    pub const MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;
}
