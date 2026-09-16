//! Contract tests for indirect ranges against the allocation they name.
//!
//! An indirect range has two bounds and this file covers both. The record
//! validators own the inner one (the record must fit inside the range) and are
//! covered with the rest of the record arithmetic; the outer one -- the range
//! must fit inside the buffer it names -- is the rule this file exists for,
//! because it is the rule that had no call site on any indirect path.
//!
//! The expected error is written out here rather than read from the
//! implementation, so this file states the contract instead of quoting it: an
//! implementation that keeps rejecting but invents its own wording fails here.
//! Every implementation the host can execute is presented with the same
//! malformed request and has to answer with that one error.
//!
//! Honest limit of that evidence: the native provider cannot be constructed
//! without a live GL context, so it cannot appear in a host test. Its two verbs
//! share one buffer-resolution funnel (`exec_compute::indirect_command_buffer`)
//! which calls the same shared rule, so the native half is held by construction
//! and by reading rather than by a failing case here. The recorder is the
//! differential oracle for the domain, which is what makes this the strongest
//! reachable evidence rather than the strongest possible one.

use super::*;

/// The one error this contract defines for a range that leaves its allocation.
///
/// Spelled once, in the test, so the implementations are compared against a
/// single expectation instead of each being compared against itself.
fn outside_allocation(operation: &'static str) -> GlError {
    GlError::Validation {
        operation,
        message: "indirect buffer range is outside the allocation".into(),
    }
}

/// A core-only capability route, which is how both providers resolve the single
/// indirect commands: core on the versions that have it, nothing else.
fn core_route(desktop: Option<GlVersion>, embedded: Option<GlVersion>) -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: desktop,
        embedded_core: embedded,
        extension: None,
        extension_requires_probe: false,
    }
}

/// A recorder that proved the whole indirect domain plus the compute pair the
/// dispatch verb needs, so one fixture addresses every verb under test.
fn indirect_recorder() -> MockGlFamilyApi {
    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(false),
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
    builder.resolve(
        GlCapability::MultiDrawIndirect,
        core_route(Some(GlVersion::new(4, 3)), None),
        GlOperationProbe::Passed,
    );
    let snapshot = builder.build();
    for capability in [
        GlCapability::IndirectDraw,
        GlCapability::IndirectDispatch,
        GlCapability::MultiDrawIndirect,
    ] {
        assert!(
            snapshot.capabilities().supports(capability),
            "the fixture resolved {capability:?}"
        );
    }
    MockGlFamilyApi::from_discovery(snapshot)
}

/// A command buffer of exactly `size` bytes.
fn command_buffer(api: &mut MockGlFamilyApi, size: u64) -> BufferId {
    api.create_buffer_resource(GlBufferDesc {
        size,
        usage: GlBufferUsage::INDIRECT,
    })
    .expect("command buffer")
}

/// One non-indexed record inside a `size`-byte range.
///
/// The record itself fits whenever `size` is at least the 16-byte ABI, so a
/// range wider than its allocation fails on the allocation bound and nothing
/// else -- which is what makes these cases name the rule under test.
fn command_range(buffer: BufferId, size: u64) -> GlIndirectCommandRange {
    GlIndirectCommandRange {
        range: GlBufferRange {
            buffer,
            offset: 0,
            size,
        },
        command_offset: 0,
        draw_count: 1,
        stride: 0,
        abi: GlIndirectAbi::NonIndexed,
    }
}

/// One count word inside a `size`-byte range.
fn count_range(buffer: BufferId, size: u64) -> GlIndirectCountRange {
    GlIndirectCountRange {
        range: GlBufferRange {
            buffer,
            offset: 0,
            size,
        },
        count_offset: 0,
        max_draw_count: 1,
    }
}

/// One dispatch record inside a `size`-byte range.
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

/// Opens a single-layer 1x1 pass, the only precondition the recorder models for
/// a raster indirect verb.
fn begin_pass(api: &mut MockGlFamilyApi) {
    let texture = api
        .create_texture_resource(mock_texture_desc())
        .expect("render target");
    let view = texture_view(texture, 1);
    let framebuffer = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![view],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("framebuffer");
    api.begin_render_pass(&GlRenderPassDescriptor {
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
    })
    .expect("pass");
}

/// A compute program descriptor, the only kind an indirect dispatch may read.
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

/// The allocation bound is exact and it fails closed at the arithmetic edge.
///
/// A window ending exactly at the allocation end is legal; one byte further is
/// not; and a window whose end cannot be represented in `u64` must be refused
/// rather than wrapped into an address that happens to be inside the buffer.
#[test]
fn the_allocation_bound_is_exact_and_overflow_safe() {
    let desc = GlBufferDesc {
        size: 16,
        usage: GlBufferUsage::INDIRECT,
    };
    let buffer = BufferId::new(stamp(ContextEpoch::INITIAL), 0, 0);
    let range = |offset, size| GlBufferRange {
        buffer,
        offset,
        size,
    };
    let check = |range| validate_indirect_allocation("draw-indirect", range, desc);

    assert_eq!(check(range(0, 16)), Ok(()), "a whole-buffer window");
    assert_eq!(check(range(8, 8)), Ok(()), "a window ending on the bound");
    assert_eq!(
        check(range(16, 0)),
        Err(outside_allocation("draw-indirect"))
    );
    assert_eq!(
        check(range(8, 9)),
        Err(outside_allocation("draw-indirect")),
        "one byte past the end is past the end"
    );
    assert_eq!(
        check(range(0, 17)),
        Err(outside_allocation("draw-indirect"))
    );
    assert_eq!(
        check(range(16, 4)),
        Err(outside_allocation("draw-indirect")),
        "a window starting past the allocation holds nothing"
    );
    assert_eq!(
        check(range(u64::MAX - 2, 4)),
        Err(outside_allocation("draw-indirect")),
        "an unrepresentable end fails closed"
    );
}

/// Every reachable indirect verb bounds its range by the allocation it names.
///
/// Each verb is presented a request that is well formed on its own terms --
/// nonempty, 4-byte aligned, wide enough for the record it reads -- but wider
/// than the buffer it names. A provider that accepted one would bind that buffer
/// and let the driver read records past the end of the allocation, which is a
/// silent out-of-bounds read rather than a rejection, so the refusal has to
/// happen before the binding changes. Each rejection must also carry the same
/// structured error, since a caller cannot repair a range it cannot recognize.
#[test]
fn every_raster_indirect_verb_bounds_its_range_by_the_allocation() {
    let mut api = indirect_recorder();
    begin_pass(&mut api);
    let over = command_buffer(&mut api, 16);
    api.clear_calls();

    let error = api
        .draw_indirect(command_range(over, 32))
        .expect_err("a 32-byte range cannot fit a 16-byte allocation");
    assert_eq!(error, outside_allocation("draw-indirect"));

    let error = api
        .multi_draw_indirect(command_range(over, 32))
        .expect_err("a batch reads the same bound range");
    assert_eq!(error, outside_allocation("multi-draw-indirect"));

    let error = api
        .multi_draw_indirect_count(command_range(over, 32), count_range(over, 16))
        .expect_err("the record range is bounded");
    assert_eq!(error, outside_allocation("multi-draw-indirect-count"));

    // The count word is a second range into the same allocation, reached
    // independently of the record range, so it needs the bound of its own: a
    // trace that only bounded the records would still let the driver read a
    // count word past the end of the buffer.
    let error = api
        .multi_draw_indirect_count(command_range(over, 16), count_range(over, 32))
        .expect_err("the count range is bounded");
    assert_eq!(error, outside_allocation("multi-draw-indirect-count"));

    assert_eq!(
        api.calls().len(),
        4,
        "one rejection per call and nothing else: {:?}",
        api.calls()
    );
    // The trace is read as errors rather than merely counted: a funnel that
    // recorded an accepted command beside the rejection would pass a length
    // check, and this is the assertion that rules it out.
    let recorded: Vec<GlError> = api
        .calls()
        .iter()
        .map(|call| match call {
            MockCall::Error(error) => error.clone(),
            other => panic!("a rejection must not be recorded beside a command: {other:?}"),
        })
        .collect();
    assert_eq!(
        recorded,
        vec![
            outside_allocation("draw-indirect"),
            outside_allocation("multi-draw-indirect"),
            outside_allocation("multi-draw-indirect-count"),
            outside_allocation("multi-draw-indirect-count"),
        ],
        "each verb rejects in order, and the counted batch bounds both of its ranges"
    );
}

/// The indirect-dispatch verb is bounded by the same rule.
///
/// It is the one indirect verb that is not a raster command, so it is reached
/// through the wrapper carrying the installed program the record needs; the
/// bound itself is the same rule and the same error.
#[test]
fn an_indirect_dispatch_bounds_its_range_by_the_allocation() {
    let mut recorder = indirect_recorder();
    let over = command_buffer(&mut recorder, 16);
    let (program, _) = recorder
        .create_program(&compute_program())
        .expect("compute program");
    let mut api = recorder
        .try_with_compute_storage()
        .expect("the fixture proved both optional rows");
    api.set_compute_program(program).expect("install");
    let from = api.calls().len();

    let error = api
        .dispatch_indirect(dispatch_command(over, 32))
        .expect_err("a 32-byte range cannot fit a 16-byte allocation");
    assert_eq!(error, outside_allocation("dispatch-indirect"));
    // The slice is asserted to hold exactly the rejection rather than merely to
    // contain no command: an empty trace satisfies "nothing was accepted", so
    // the length is what makes the trace itself the evidence.
    assert_eq!(
        api.calls()[from..].len(),
        1,
        "the rejection is recorded: {:?}",
        &api.calls()[from..]
    );
    assert!(
        matches!(
            api.calls().last(),
            Some(MockCall::Error(error)) if *error == outside_allocation("dispatch-indirect")
        ),
        "the record was never read: {:?}",
        &api.calls()[from..]
    );
}

/// The bound is a bound, not a refusal: the same requests inside the allocation
/// are accepted and recorded.
///
/// Without this half the rejection test would pass against an implementation
/// that refused every indirect command, which is a different contract.
#[test]
fn indirect_ranges_inside_their_allocation_are_accepted() {
    let mut api = indirect_recorder();
    begin_pass(&mut api);
    let whole = command_buffer(&mut api, 16);
    let counted = command_buffer(&mut api, 32);
    api.clear_calls();

    let command = command_range(whole, 16);
    assert_eq!(api.draw_indirect(command), Ok(()));
    assert!(matches!(
        api.calls().last(),
        Some(MockCall::DrawIndirect(recorded)) if *recorded == command
    ));

    let batch = command_range(counted, 32);
    assert_eq!(api.multi_draw_indirect(batch), Ok(()));
    let count = count_range(counted, 16);
    assert_eq!(api.multi_draw_indirect_count(batch, count), Ok(()));
    assert!(matches!(
        api.calls().last(),
        Some(MockCall::MultiDrawIndirectCount { commands, count: recorded })
            if *commands == batch && *recorded == count
    ));
}
