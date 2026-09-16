//! Contract tests and shared fixtures for the mock recorder's own oracle.
//!
//! These assert what the recorder is allowed to *claim* about work it never
//! executed, which is a property of the recorder rather than of the Layer 1
//! contract, so they live next to the recorder.  The fixtures are here and the
//! evidence is split by subject -- the oracle's own answers in `oracle`, the
//! optional command domains in `command`, the multiview gate in `multiview` --
//! because the fixtures are the one part every group must agree on while the
//! groups themselves change for unrelated reasons.
//!
//! The discovery fixture below is deliberately a copy of the API-level one:
//! `api/tests/mod.rs` now exposes the same builders `pub(crate)`, but this suite
//! keeps its own copy so its evidence cannot be invalidated by an edit to a file
//! another work package is still moving.  Two copies can drift; the assertions
//! inside each recorder fixture below are what make a drifted copy fail loudly
//! instead of quietly weakening a test.

mod command;
mod multiview;
mod oracle;

use super::*;

fn limits() -> GlLimits {
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
        max_multiview_view_count: 2,
        max_multi_draw_indirect_count: Some(1),
        query_counter_bits: 32,
        max_texture_anisotropy: GlFiniteF32::new(16.0),
    }
}

fn formats() -> GlFormatTable {
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
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: true,
                renderable: true,
                blendable: true,
                storage_read: false,
                storage_write: false,
                copy_source: true,
                copy_destination: true,
            })
            .expect("unique fact");
    }
    table
}

/// The desktop limits a 4.3 context reports: the same floors as `limits`, raised
/// where the desktop profile requires more.
fn desktop_limits() -> GlLimits {
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

/// The context stamp every fixture in this suite is bound to.
fn stamp() -> ContextStamp {
    ContextStamp::new(
        DeviceIdentity::new(7).expect("identity"),
        ContextEpoch::INITIAL,
    )
}

/// One context's identity facts, with the raw strings standing in for a driver.
fn context(profile: GlFamilyProfile) -> GlContextInfo {
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

/// The compute route both providers resolve: a desktop/embedded core version
/// plus the ARB extension, gated on its own operation probe.
fn compute() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: Some(GlVersion::new(4, 3)),
        embedded_core: Some(GlVersion::new(3, 1)),
        extension: Some(GlKnownExtension::ArbComputeShader),
        extension_requires_probe: true,
    }
}

/// The storage-buffer route, gated the same way as compute.
fn storage_buffer() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: Some(GlVersion::new(4, 3)),
        embedded_core: Some(GlVersion::new(3, 1)),
        extension: Some(GlKnownExtension::ArbShaderStorageBufferObject),
        extension_requires_probe: true,
    }
}

/// A route satisfied by core versions alone, with no extension alternative.
///
/// This is how both providers resolve the single indirect commands: the feature
/// is core on the versions that have it, and a profile without one resolves no
/// evidence at all rather than reaching for an extension that does not exist.
fn core_route(desktop: Option<GlVersion>, embedded: Option<GlVersion>) -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: desktop,
        embedded_core: embedded,
        extension: None,
        extension_requires_probe: false,
    }
}

/// The batch route: one extension whose oracle is the acquired, complete command
/// set of the extension object, with no queryable limit and no probe.
fn batch() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: None,
        embedded_core: None,
        extension: Some(GlKnownExtension::WebglMultiDraw),
        extension_requires_probe: false,
    }
}

/// The multiview route: the second-revision extension, which needs both its
/// acquisition and a successful operation probe.
fn multiview() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: None,
        embedded_core: None,
        extension: Some(GlKnownExtension::OvrMultiview2),
        extension_requires_probe: true,
    }
}

/// A recorder on a desktop 4.3 context that proved the optional rows only a
/// desktop family can reach: compute, storage buffers and the indirect commands.
///
/// Each row is resolved through the route its own provider uses, and the
/// single-indirect rows carry the probe outcome a real probe backend produces,
/// so a command these tests accept is one a real desktop context could accept.
/// The multi-draw-indirect row is the deliberate exception and says so at the
/// call site below.
fn desktop_recorder() -> MockGlFamilyApi {
    let mut builder = GlDiscoveryBuilder::new(
        stamp(),
        context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(),
    )
    .expect("desktop discovery");
    builder.resolve(GlCapability::Compute, compute(), GlOperationProbe::Passed);
    builder.resolve(
        GlCapability::StorageBuffer,
        storage_buffer(),
        GlOperationProbe::Passed,
    );
    builder.resolve(
        GlCapability::IndirectDraw,
        core_route(Some(GlVersion::new(4, 0)), Some(GlVersion::new(3, 1))),
        GlOperationProbe::Passed,
    );
    builder.resolve(
        GlCapability::IndirectDispatch,
        core_route(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1))),
        GlOperationProbe::Passed,
    );
    // No shipped provider can currently enable this row: glow 0.18 binds no
    // multi-draw-indirect entry point, so native records `NotRun` and the
    // capability stays closed, and WebGL2 has no indirect mapping at all.  The
    // recorder still has to model the domain -- that is the whole point of the
    // missing-domain audit -- so this fixture injects the probe outcome a future
    // provider would produce rather than pretending the row is reachable today.
    builder.resolve(
        GlCapability::MultiDrawIndirect,
        core_route(Some(GlVersion::new(4, 3)), None),
        GlOperationProbe::Passed,
    );
    let snapshot = builder.build();
    for capability in [
        GlCapability::Compute,
        GlCapability::StorageBuffer,
        GlCapability::IndirectDraw,
        GlCapability::IndirectDispatch,
        GlCapability::MultiDrawIndirect,
    ] {
        assert!(
            snapshot.capabilities().supports(capability),
            "the fixture resolved {capability:?} through its shipped route"
        );
    }
    assert_eq!(
        snapshot.limits().max_multi_draw_indirect_count,
        Some(1),
        "the count limit the batch tests measure against"
    );
    MockGlFamilyApi::from_discovery(snapshot)
}

/// A WebGL2 recorder that proved the two optional rows only this family has: the
/// acquired batch command set, and the probed multiview route.
///
/// Both are resolved exactly as the browser discovery resolves them.  Multiview
/// records `NotRun` there today because no attachment probe exists yet, so the
/// probe outcome here is the one that would follow from the missing probe
/// landing; the gate itself reads only the resolved row, which is what makes the
/// recorder testable before the probe does.
fn webgl2_recorder() -> MockGlFamilyApi {
    let mut extensions = GlExtensionSet::default();
    extensions.report_raw("WEBGL_multi_draw");
    extensions.report_raw("OVR_multiview2");
    assert!(extensions.acquire(GlKnownExtension::WebglMultiDraw));
    assert!(extensions.acquire(GlKnownExtension::OvrMultiview2));
    assert!(extensions.probe(GlKnownExtension::OvrMultiview2));
    let mut builder = GlDiscoveryBuilder::new(
        stamp(),
        context(GlFamilyProfile::WebGl2),
        extensions,
        limits(),
        formats(),
    )
    .expect("webgl2 discovery");
    builder.resolve(
        GlCapability::MultiDraw,
        batch(),
        GlOperationProbe::NotRequired,
    );
    builder.resolve(
        GlCapability::Multiview,
        multiview(),
        GlOperationProbe::Passed,
    );
    let snapshot = builder.build();
    assert!(snapshot.capabilities().supports(GlCapability::MultiDraw));
    assert!(snapshot.capabilities().supports(GlCapability::Multiview));
    assert_eq!(
        snapshot.max_multiview_view_count(),
        2,
        "the view count the multiview tests measure against"
    );
    MockGlFamilyApi::from_discovery(snapshot)
}

/// A recorder bound to a WebGL2 snapshot at exactly the profile minimums.
fn recorder() -> MockGlFamilyApi {
    let snapshot = GlDiscoveryBuilder::new(
        stamp(),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        limits(),
        formats(),
    )
    .expect("test discovery")
    .build();
    MockGlFamilyApi::from_discovery(snapshot)
}

/// A recorder whose WebGL2 ledger acquired S3TC and whose format table
/// therefore carries one exact compressed fact.
fn compressed_recorder() -> MockGlFamilyApi {
    let extension = GlKnownExtension::CompressedTextureS3tc;
    let mut extensions = GlExtensionSet::default();
    extensions.report_raw("WEBGL_compressed_texture_s3tc");
    assert!(extensions.acquire(extension), "the ledger acquires S3TC");
    let mut table = formats();
    table
        .record(GlFormatCapabilities {
            format: GlFormat::Bc1RgbaUnorm,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::ExtensionAcquired(extension),
            sampled: true,
            filterable: true,
            renderable: false,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: false,
        })
        .expect("exact compressed fact");
    MockGlFamilyApi::from_discovery(
        GlDiscoveryBuilder::new(
            stamp(),
            context(GlFamilyProfile::WebGl2),
            extensions,
            limits(),
            table,
        )
        .expect("test discovery")
        .build(),
    )
}

/// The client layout a compressed upload is *not* measured by.
fn rgba8_layout() -> GlPixelLayout {
    GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 64,
        rows_per_image: 4,
        offset: 0,
        alignment: 4,
        repack: GlRepackPolicy::Disallow,
    }
}

/// Asserts one trace section is exactly these calls.
fn assert_trace(trace: &[MockCall], expected: &[MockCall]) {
    assert_eq!(trace, expected, "the trace records exactly what happened");
}

/// Asserts one trace section is a single rejection and nothing else.
///
/// This is the property every guard in the recorder has to have.  An error in
/// the trace with no accepted call beside it means the check ran *before* the
/// side effect; an accepted call beside it would mean the recorder did the work
/// and then reported a failure, which is exactly what a state-machine test
/// cannot distinguish from a correct guard.
fn assert_rejected(trace: &[MockCall]) {
    assert_eq!(trace.len(), 1, "only the rejection was recorded: {trace:?}");
    assert!(
        matches!(trace[0], MockCall::Error(_)),
        "the rejection reached the trace as an error"
    );
}

/// A non-indexed draw of `vertex_count` vertices from `first_vertex`.
fn non_indexed_at(first_vertex: u32, vertex_count: u32) -> GlDrawCommand {
    GlDrawCommand::NonIndexed(GlNonIndexedDraw {
        first_vertex,
        vertex_count,
        instance_count: 1,
    })
}

/// One non-indexed draw of three vertices and one instance.
fn non_indexed() -> GlDrawCommand {
    non_indexed_at(0, 3)
}

/// A one-layer color attachment view over a freshly allocated target.
///
/// The allocation is a plain D2 texture because that is the only dimensionality
/// this slice attaches (both providers reject every other one), so a view asking
/// for several layers is the only shape a multiview request can take here.
fn target_view(api: &mut MockGlFamilyApi, layer_count: u32) -> GlTextureView {
    let texture = api
        .create_texture_resource(GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 1,
                height: 1,
                depth_or_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::RENDER_ATTACHMENT,
        })
        .expect("render target");
    GlTextureView {
        target: GlAttachmentTarget::Texture(texture),
        format: GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        layer_count,
        width: 1,
        height: 1,
        sample_count: 1,
    }
}

/// A framebuffer descriptor holding exactly this view.
fn framebuffer_descriptor(view: GlTextureView) -> GlFramebufferDescriptor {
    GlFramebufferDescriptor {
        color_attachments: vec![view],
        depth_stencil_attachment: None,
        draw_buffers: vec![],
    }
}

/// A pass descriptor naming exactly the view its framebuffer was built from.
fn pass_descriptor(framebuffer: FramebufferId, view: GlTextureView) -> GlRenderPassDescriptor {
    GlRenderPassDescriptor {
        framebuffer,
        color_attachments: vec![GlColorAttachment {
            view,
            resolve_target: None,
            load: GlLoadOp::Clear,
            store: GlStoreOp::Store,
            clear: GlColorClearValue {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0,
            },
        }],
        depth_stencil_attachment: None,
    }
}

/// Opens a single-layer 1x1 pass on this recorder.
///
/// Every raster and indirect command requires an active pass, and the recorder
/// models no installed pipeline, so this is the whole precondition those domains
/// have here.
fn begin_pass(api: &mut MockGlFamilyApi) -> FramebufferId {
    let view = target_view(api, 1);
    let framebuffer = api
        .create_framebuffer(&framebuffer_descriptor(view))
        .expect("framebuffer");
    api.begin_render_pass(&pass_descriptor(framebuffer, view))
        .expect("pass");
    framebuffer
}

/// A buffer created for one role, so a replay of its trace says which role the
/// domain under test was asked to read.
fn buffer_with(api: &mut MockGlFamilyApi, size: u64, usage: GlBufferUsage) -> BufferId {
    api.create_buffer_resource(GlBufferDesc { size, usage })
        .expect("role buffer")
}

/// A non-indexed command range of `draw_count` 16-byte records inside `size`.
fn command_range(buffer: BufferId, size: u64, draw_count: u32) -> GlIndirectCommandRange {
    GlIndirectCommandRange {
        range: GlBufferRange {
            buffer,
            offset: 0,
            size,
        },
        command_offset: 0,
        draw_count,
        stride: 0,
        abi: GlIndirectAbi::NonIndexed,
    }
}

/// The count word one counted batch reads before it decides how many records to
/// issue, inside a `size`-byte range.
fn count_range(buffer: BufferId, size: u64, max_draw_count: u32) -> GlIndirectCountRange {
    GlIndirectCountRange {
        range: GlBufferRange {
            buffer,
            offset: 0,
            size,
        },
        count_offset: 0,
        max_draw_count,
    }
}

/// One dispatch record wholly inside a range of `size` bytes.
fn dispatch_command(buffer: BufferId, size: u64) -> GlDispatchIndirectCommand {
    GlDispatchIndirectCommand {
        range: GlBufferRange {
            buffer,
            offset: 0,
            size,
        },
        command_offset: 0,
    }
}

/// A compute program descriptor, which is the only kind that may be installed
/// for indirect dispatch.
fn compute_program() -> GlProgramDescriptor {
    GlProgramDescriptor {
        kind: GlProgramKind::Compute {
            shader: GlShaderSource {
                stage: GlShaderStage::Compute,
                dialect: GlShaderDialect::Desktop { version: 430 },
                entry_point: "main".into(),
                source_hash: ShaderSourceHash([9; 32]),
                text: "void main() {}".into(),
                debug_name: None,
            },
        },
        layout: GlPipelineLayout { bindings: vec![] },
        debug_name: None,
    }
}
