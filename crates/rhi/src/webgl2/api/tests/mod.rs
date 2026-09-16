//! Shared fixtures for the api contract suites, plus the suites themselves.
//!
//! This module owns only fixture construction: the limits literal, the context
//! builder, the format table, the context stamp, the six `CoreOrExtension`
//! routes the shipped rows use, and the ledger/builder/snapshot helpers built on
//! top of them. The files below are composition entries in the same sense: each
//! covers one contract of the module under test and owns no fixture beyond what
//! that contract needs.
//!
//! The fixture is deliberately `pub(crate)`: a suite outside this module may
//! share these builders rather than restate the same limits and tables, because
//! two copies of a fixture can drift apart without any test noticing.

pub(crate) use super::*;

pub(crate) fn limits() -> GlLimits {
    GlLimits {
        max_texture_size: 2_048,
        max_3d_texture_size: 256,
        max_array_texture_layers: 256,
        max_cube_map_texture_size: 2_048,
        max_renderbuffer_size: 4_096,
        max_color_attachments: 4,
        max_draw_buffers: 4,
        max_vertex_attributes: 16,
        max_viewport_dimensions: [2_048; 2],
        max_viewports: 1,
        max_vertex_texture_image_units: 16,
        max_fragment_texture_image_units: 16,
        max_combined_texture_image_units: 16,
        max_uniform_buffer_bindings: 24,
        max_uniform_block_size: 16_384,
        uniform_buffer_offset_alignment: 256,
        max_vertex_uniform_blocks: 12,
        max_fragment_uniform_blocks: 12,
        max_compute_uniform_blocks: 12,
        max_combined_uniform_blocks: 24,
        max_storage_buffer_bindings: 8,
        max_storage_block_size: 1 << 27,
        storage_buffer_offset_alignment: 256,
        max_vertex_storage_blocks: 4,
        max_fragment_storage_blocks: 4,
        max_compute_storage_blocks: 4,
        max_combined_storage_blocks: 8,
        max_image_units: 4,
        max_combined_image_units: 4,
        max_samples: 4,
        max_color_texture_samples: 4,
        max_depth_texture_samples: 4,
        max_integer_samples: 4,
        max_compute_work_group_count: [65_535; 3],
        max_compute_work_group_size: [1_024, 1_024, 64],
        max_compute_work_group_invocations: 1_024,
        // A context that answered the multiview query with two views: the
        // numeric half is satisfiable while the capability still needs its
        // route and probe evidence.
        max_multiview_view_count: 2,
        max_multi_draw_indirect_count: Some(1),
        query_counter_bits: 32,
        max_texture_anisotropy: GlFiniteF32::new(16.0),
    }
}

pub(crate) fn desktop_limits() -> GlLimits {
    let mut limits = limits();
    limits.max_texture_size = 16_384;
    limits.max_3d_texture_size = 2_048;
    limits.max_array_texture_layers = 2_048;
    limits.max_cube_map_texture_size = 16_384;
    limits.max_renderbuffer_size = 16_384;
    limits.max_color_attachments = 8;
    limits.max_draw_buffers = 8;
    limits.max_viewport_dimensions = [16_384; 2];
    limits.max_uniform_buffer_bindings = 36;
    limits
}

pub(crate) fn context(profile: GlFamilyProfile) -> GlContextInfo {
    GlContextInfo::new(
        profile,
        "version",
        "glsl",
        "vendor",
        "renderer",
        "driver",
        GlContextFlags::default(),
    )
}

pub(crate) fn formats(storage: bool) -> GlFormatTable {
    let mut t = GlFormatTable::default();
    for format in [
        GlFormat::Rgba8Unorm,
        GlFormat::Rgba8Srgb,
        GlFormat::Depth32Float,
    ] {
        t.record(GlFormatCapabilities {
            format,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: if storage && format == GlFormat::Rgba8Unorm {
                GlFormatEvidence::OperationProbed
            } else {
                GlFormatEvidence::CoreGuaranteed
            },
            sampled: true,
            filterable: true,
            renderable: true,
            blendable: true,
            storage_read: storage && format == GlFormat::Rgba8Unorm,
            storage_write: storage && format == GlFormat::Rgba8Unorm,
            copy_source: true,
            copy_destination: true,
        })
        .expect("unique fact");
    }
    t
}

pub(crate) fn stamp(epoch: ContextEpoch) -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(7).expect("identity"), epoch)
}

pub(crate) fn compute() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: Some(GlVersion::new(4, 3)),
        embedded_core: Some(GlVersion::new(3, 1)),
        extension: Some(GlKnownExtension::ArbComputeShader),
        extension_requires_probe: true,
    }
}

/// The shipped batch requirement: one extension route whose oracle is the
/// acquired, complete command set of the extension object (audit P1-8).
pub(crate) fn batch() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: None,
        embedded_core: None,
        extension: Some(GlKnownExtension::WebglMultiDraw),
        extension_requires_probe: false,
    }
}

/// The shipped multiview requirement: the second-revision extension route,
/// which needs a queried view count and a successful operation probe.
pub(crate) fn multiview() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: None,
        embedded_core: None,
        extension: Some(GlKnownExtension::OvrMultiview2),
        extension_requires_probe: true,
    }
}

/// The shipped non-indexed/indexed indirect-draw requirement: core on desktop
/// 4.0 and embedded 3.1, with no extension route and a probe that must pass.
pub(crate) fn indirect_draw() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: Some(GlVersion::new(4, 0)),
        embedded_core: Some(GlVersion::new(3, 1)),
        extension: None,
        extension_requires_probe: false,
    }
}

/// The shipped indirect-dispatch requirement, which is core one desktop version
/// later than the draw half.
pub(crate) fn indirect_dispatch() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: Some(GlVersion::new(4, 3)),
        embedded_core: Some(GlVersion::new(3, 1)),
        extension: None,
        extension_requires_probe: false,
    }
}

/// The shipped multi-draw-indirect requirement: desktop-only, and a row whose
/// probe no provider answers.
pub(crate) fn multi_draw_indirect() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: Some(GlVersion::new(4, 3)),
        embedded_core: None,
        extension: None,
        extension_requires_probe: false,
    }
}

/// A ledger that has reported exactly these raw runtime names.
pub(crate) fn ledger(names: &[&str]) -> GlExtensionSet {
    let mut extensions = GlExtensionSet::default();
    for name in names {
        extensions.report_raw(*name);
    }
    extensions
}

/// A builder bound to one profile and one extension ledger.
pub(crate) fn builder(
    profile: GlFamilyProfile,
    extensions: GlExtensionSet,
    limits: GlLimits,
) -> GlDiscoveryBuilder {
    GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(profile),
        extensions,
        limits,
        formats(false),
    )
    .expect("test discovery")
}

pub(crate) fn snapshot(profile: GlFamilyProfile) -> GlDiscoverySnapshot {
    let limits = match profile {
        GlFamilyProfile::Desktop { .. } => desktop_limits(),
        _ => limits(),
    };
    GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(profile),
        GlExtensionSet::default(),
        limits,
        formats(false),
    )
    .expect("test discovery")
    .build()
}

/// One extra exact fact beyond the single-sample baseline table.
pub(crate) fn snapshot_with_fact(
    resource_kind: GlFormatResourceKind,
    sample_count: u32,
) -> GlDiscoverySnapshot {
    let texture_facts = resource_kind == GlFormatResourceKind::Texture;
    let mut formats = formats(false);
    formats
        .record(GlFormatCapabilities {
            format: GlFormat::Rgba8Unorm,
            resource_kind,
            sample_count,
            evidence: GlFormatEvidence::CoreGuaranteed,
            sampled: texture_facts,
            filterable: texture_facts,
            renderable: true,
            blendable: texture_facts,
            storage_read: false,
            storage_write: false,
            copy_source: texture_facts,
            copy_destination: texture_facts,
        })
        .expect("extra exact fact");
    GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        limits(),
        formats,
    )
    .expect("test discovery")
    .build()
}

/// A WebGL2 snapshot over a caller-supplied format table.
///
/// The three snapshots above all start from the baseline table and add to it, so
/// none of them can express a format that is *recorded and not renderable* --
/// which is a reading the common contract distinguishes from an absent row. That
/// case needs a table the caller composed, and building it here keeps the profile,
/// the stamp and the limits from being restated at the call site.
pub(crate) fn snapshot_with_formats(formats: GlFormatTable) -> GlDiscoverySnapshot {
    GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        limits(),
        formats,
    )
    .expect("test discovery")
    .build()
}

/// A WebGL2 baseline snapshot whose drawable was observed at these widths.
///
/// The three snapshots above are all built by a provider that never asked what
/// the drawable is, which is why they report no surface; this one is the same
/// context with the one extra observation a presenter needs.
pub(crate) fn snapshot_with_surface(color_bits: [u32; 4]) -> GlDiscoverySnapshot {
    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        limits(),
        formats(false),
    )
    .expect("test discovery");
    builder.surface_facts(GlSurfaceFacts::Observed { color_bits });
    builder.build()
}

pub(crate) fn compute_storage_snapshot(storage_image: bool) -> GlDiscoverySnapshot {
    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(storage_image),
    )
    .expect("desktop discovery");
    builder.resolve(GlCapability::Compute, compute(), GlOperationProbe::Passed);
    builder.resolve(
        GlCapability::StorageBuffer,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderStorageBufferObject),
            extension_requires_probe: true,
        },
        GlOperationProbe::Passed,
    );
    if storage_image {
        builder.resolve(
            GlCapability::StorageImage,
            CoreOrExtension {
                desktop_core: Some(GlVersion::new(4, 2)),
                embedded_core: Some(GlVersion::new(3, 1)),
                extension: Some(GlKnownExtension::ArbShaderImageLoadStore),
                extension_requires_probe: true,
            },
            GlOperationProbe::Passed,
        );
    }
    builder.build()
}

/// A desktop context that resolved exactly one indirect-command row.
///
/// The route is the shipped one, restated here for the reason the compute,
/// batch and multiview routes above are: a fixture is representative only when
/// it builds the ledger the shipped discovery would have built, and a row
/// enabled by a route no provider resolves would prove a state the shipped path
/// cannot reach.
///
/// The probe answer is the caller's, because it is the one half of a row's
/// evidence the shipped path does not answer the same way for all three rows.
/// A row is enabled by its evidence *and* its limits *and* a passed probe, so a
/// caller that passes `NotRun` is asking for the fail-closed answer.
pub(crate) fn indirect_snapshot(
    capability: GlCapability,
    operation_probe: GlOperationProbe,
) -> GlDiscoverySnapshot {
    let route = match capability {
        GlCapability::IndirectDraw => indirect_draw(),
        GlCapability::IndirectDispatch => indirect_dispatch(),
        GlCapability::MultiDrawIndirect => multi_draw_indirect(),
        other => panic!("{other:?} is not an indirect-command row"),
    };
    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(false),
    )
    .expect("desktop discovery");
    builder.resolve(capability, route, operation_probe);
    builder.build()
}

pub(crate) fn mock_texture_desc() -> GlTextureDesc {
    texture_desc(1)
}

pub(crate) fn texture_desc(sample_count: u32) -> GlTextureDesc {
    GlTextureDesc {
        dimension: GlTextureDimension::D2,
        extent: GlExtent3d {
            width: 1,
            height: 1,
            depth_or_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        format: GlFormat::Rgba8Unorm,
        usage: GlTextureUsage::RENDER_ATTACHMENT,
    }
}

pub(crate) fn texture_view(texture: TextureId, sample_count: u32) -> GlTextureView {
    GlTextureView {
        target: GlAttachmentTarget::Texture(texture),
        format: GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        layer_count: 1,
        width: 1,
        height: 1,
        sample_count,
    }
}

mod binding;
mod discovery;
mod formats;
mod framebuffer;
mod indirect;
mod multi_draw;
mod multiview;
mod query;
mod raster;
mod recorder;
mod shader;
mod storage;
mod transfer;
