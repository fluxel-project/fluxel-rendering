//! Tests for the common execution adapter over a GL-family state machine.
//!
//! Two kinds of test live here and they answer different questions.  The
//! *lowering* tests call [`transient`]'s functions directly, because the target
//! a dimension and a layer count lower to is not observable through the adapter
//! -- Layer 1 stores the descriptor it was handed and reports only the identity.
//! Every other test goes through `ExecutionBackend` on a real
//! [`GlCompatibilityDevice`] over `MockGlFamilyApi`, because what those check is
//! the contract: which calls the adapter makes, in which order, and what it
//! answers when it is asked about a submission.
//!
//! Fence completion cannot be simulated here -- the recorder has no submission
//! queue, so nothing it recorded can finish -- and the tests therefore inject it,
//! which is the seam Layer 1 already documents for exactly this reason.  What is
//! being tested is what the adapter does with an observed outcome, and an
//! outcome the test names is as much an observation as one a driver produced.

mod commands;

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, CompletionFailure, CompletionStatus,
    ExecutionBackend, Extent3d, FrameExecutor, QueueId, TextureDesc, TextureDimension,
    TextureFormat, TextureUsage, TextureUsageKind,
};

use super::super::capabilities::capabilities;
use super::transient;
use super::{
    GlCompatibilityDevice, UnsupportedBindings, UnsupportedComputePipeline,
    UnsupportedRasterPipeline,
};
use crate::webgl2::api::tests::{compute_storage_snapshot, snapshot};
use crate::webgl2::api::{
    GlBufferUsage, GlError, GlExtent3d, GlFamilyApi, GlFamilyProfile, GlFenceStatus, GlFormat,
    GlTextureDimension, GlTextureUsage, MockCall, MockGlFamilyApi,
};

/// The adapter under test, over the WebGL2 snapshot.
type Adapter = GlCompatibilityDevice<MockGlFamilyApi>;

/// The peer check F1 owed F2: the boundary's own consumer accepts this adapter.
///
/// F1's record deferred it in as many words -- "the 'consumer that names the
/// adapter as `B: ExecutionBackend`' check moves to F2 with it" -- and it is why
/// the associated types are F2's first act rather than something that could be
/// fixed later.  What it proves is narrow and worth stating precisely: the eleven
/// types are a *set* `FrameExecutor` accepts through its generic, with no `dyn
/// ExecutionBackend` and no adapter-side shim between them.
///
/// It is a compile-time check and not a run.  A frame's first real act is
/// `emit_transitions`, which this adapter refuses until its copy slice lands, so
/// running one would assert the refusal rather than the acceptance -- and the
/// acceptance is the question that was open.
#[allow(
    dead_code,
    reason = "the check is that this type-checks, not that it runs"
)]
fn the_executor_accepts_the_adapter(adapter: Adapter) -> FrameExecutor<Adapter> {
    FrameExecutor::new(adapter)
}

fn adapter() -> Adapter {
    GlCompatibilityDevice::new(MockGlFamilyApi::from_discovery(snapshot(
        GlFamilyProfile::WebGl2,
    )))
}

/// The adapter over the one snapshot that proved compute, storage and a storage
/// image, which is the only one whose format table can carry a storage request.
fn desktop() -> Adapter {
    GlCompatibilityDevice::new(MockGlFamilyApi::from_discovery(compute_storage_snapshot(
        true,
    )))
}

/// A common texture description, with the two things this adapter decides
/// together -- the dimension and the array-layer count -- left to the caller.
fn texture(dimension: TextureDimension, array_layers: u32, depth: u32) -> TextureDesc {
    TextureDesc {
        dimension,
        extent: Extent3d {
            width: 4,
            height: 4,
            depth,
        },
        mip_levels: 1,
        array_layers,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

/// The plain case: one two-dimensional layer.
fn plain_texture() -> TextureDesc {
    texture(TextureDimension::D2, 1, 1)
}

/// A usage set an attachment-shaped transient is compiled with.
fn colour_usage() -> TextureUsage {
    TextureUsage::from_kinds([
        TextureUsageKind::ColorAttachment,
        TextureUsageKind::CopySource,
    ])
}

/// Every mock call the adapter made, in order.
fn calls(adapter: &mut Adapter) -> Vec<MockCall> {
    adapter.machine.backend().calls().to_vec()
}

/// How many of the mock calls so far satisfy `matched`.
fn count(adapter: &mut Adapter, matched: fn(&MockCall) -> bool) -> usize {
    calls(adapter).iter().filter(|call| matched(call)).count()
}

fn is_destroy_texture(call: &MockCall) -> bool {
    matches!(call, MockCall::DestroyTexture(_))
}

fn is_destroy_buffer(call: &MockCall) -> bool {
    matches!(call, MockCall::DestroyBuffer(_))
}

fn is_destroy_fence(call: &MockCall) -> bool {
    matches!(call, MockCall::DestroyFence(_))
}

/// Forgets the trace so far, so a later count is about what follows.
fn trace_from_here(adapter: &mut Adapter) {
    adapter.machine.backend().clear_calls();
}

/// The operation name of the refusal `result` carries.
///
/// Every fail-closed verb in this slice refuses with `GlError::Unsupported` and
/// names itself, so a test that only checked for an error would accept a
/// validation failure or a context error as evidence that the verb refuses.  The
/// name is checked because the name is what an operator reads.
fn refused<T>(result: Result<T, GlError>) -> &'static str {
    match result {
        Ok(_) => panic!("the adapter was expected to refuse this"),
        Err(GlError::Unsupported { operation, .. }) => operation,
        Err(other) => panic!("expected a fail-closed refusal, got {other:?}"),
    }
}

/// The operation name of the validation failure `result` carries.
///
/// The counterpart of [`refused`] for the verbs that *do* reach a provider: what
/// a bad request gets back from there is a validation failure and not a
/// fail-closed refusal, and a test that accepted either would not be able to
/// tell "this adapter does not implement it" from "this adapter tried and the
/// request was wrong".
fn invalid<T>(result: Result<T, GlError>) -> &'static str {
    match result {
        Ok(_) => panic!("the adapter was expected to reject this"),
        Err(GlError::Validation { operation, .. }) => operation,
        Err(other) => panic!("expected a validation failure, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// What the adapter describes itself as.
// ---------------------------------------------------------------------------

#[test]
fn the_adapter_reports_the_lowered_capabilities_of_its_own_context() {
    let adapter = adapter();
    let expected = capabilities(&snapshot(GlFamilyProfile::WebGl2));

    assert_eq!(
        adapter.capabilities(),
        &expected,
        "the description is the lowering of this context's own discovery snapshot"
    );
    assert_eq!(
        adapter.device_identity(),
        adapter.device_identity(),
        "and it does not move between questions"
    );
}

#[test]
fn the_three_object_types_this_adapter_cannot_produce_are_uninhabited() {
    // Each of these only compiles while its type has no variant.  A value of an
    // uninhabited type cannot be written down anywhere, which is a stronger
    // statement than "no verb here returns one": no implementation in any crate
    // could hand one back.  The first step of implementing the raster or compute
    // slice adds a variant and breaks this build, rather than silently widening
    // what the adapter claims to accept.
    #[allow(
        dead_code,
        reason = "the check is that this compiles, not that it runs"
    )]
    fn no_raster_pipeline(value: UnsupportedRasterPipeline) -> ! {
        match value {}
    }
    #[allow(
        dead_code,
        reason = "the check is that this compiles, not that it runs"
    )]
    fn no_compute_pipeline(value: UnsupportedComputePipeline) -> ! {
        match value {}
    }
    #[allow(
        dead_code,
        reason = "the check is that this compiles, not that it runs"
    )]
    fn no_bindings(value: UnsupportedBindings) -> ! {
        match value {}
    }
}

// ---------------------------------------------------------------------------
// The encoder: one queue, one command, and no vocabulary beyond it.
// ---------------------------------------------------------------------------

#[test]
fn the_encoder_opens_on_the_one_queue_and_refuses_any_other() {
    let mut adapter = adapter();

    assert!(
        adapter.begin_encoder(QueueId::new(0)).is_ok(),
        "one immediate context is one ordered queue, which the contract names zero"
    );
    let encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    assert!(
        adapter.finish_encoder(encoder).is_ok(),
        "finishing records nothing, so it cannot fail"
    );

    assert_eq!(
        refused(adapter.begin_encoder(QueueId::new(3))),
        "begin-encoder",
        "a second queue does not exist here, and the refusal names the verb"
    );
}

#[test]
fn every_raster_and_compute_verb_refuses_and_names_itself() {
    use fluxel_rendergraph::{IndexFormat, RasterPassDescriptor, ScissorRect, Viewport};

    let mut adapter = adapter();
    let buffer = adapter
        .create_transient_buffer(
            BufferDesc { size: 64 },
            BufferUsage::from_kinds([
                BufferUsageKind::CopySource,
                BufferUsageKind::CopyDestination,
            ]),
        )
        .expect("a transient buffer");
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let pass = RasterPassDescriptor {
        label: "unrecorded",
        colors: &[],
        depth_stencil: None,
    };

    // What is left of the fail-closed set after the copy slice landed: the two
    // scopes and the every command that would record into one.  `set-raster-
    // pipeline` and its two siblings are absent rather than untested -- their
    // argument types are uninhabited, so a call to one cannot be written down
    // (`the_three_object_types_this_adapter_cannot_produce_are_uninhabited`).
    assert_eq!(
        refused(adapter.begin_raster(&mut encoder, &pass)),
        "begin-raster"
    );
    assert_eq!(refused(adapter.end_raster(&mut encoder)), "end-raster");
    assert_eq!(
        refused(adapter.begin_compute(&mut encoder, "unrecorded")),
        "begin-compute"
    );
    assert_eq!(refused(adapter.end_compute(&mut encoder)), "end-compute");
    assert_eq!(
        refused(adapter.set_vertex_buffer(&mut encoder, 0, &buffer.physical, 0)),
        "set-vertex-buffer"
    );
    assert_eq!(
        refused(adapter.set_index_buffer(&mut encoder, &buffer.physical, 0, IndexFormat::Uint16)),
        "set-index-buffer"
    );
    assert_eq!(
        refused(adapter.set_viewport(
            &mut encoder,
            Viewport {
                x: 0.0,
                y: 0.0,
                width: 4.0,
                height: 4.0,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        )),
        "set-viewport"
    );
    assert_eq!(
        refused(adapter.set_scissor(
            &mut encoder,
            ScissorRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
        )),
        "set-scissor"
    );
    assert_eq!(refused(adapter.draw(&mut encoder, 0..3, 0..1)), "draw");
    assert_eq!(
        refused(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..1)),
        "draw-indexed"
    );
    assert_eq!(
        refused(adapter.dispatch(&mut encoder, [1, 1, 1])),
        "dispatch"
    );
}

#[test]
fn a_copy_pass_opens_and_closes_over_a_copy_with_no_scope_of_its_own() {
    use fluxel_rendergraph::{BufferCopyRegion, PresentationSubmission};

    let mut adapter = adapter();
    let buffer = adapter
        .create_transient_buffer(
            BufferDesc { size: 64 },
            BufferUsage::from_kinds([
                BufferUsageKind::CopySource,
                BufferUsageKind::CopyDestination,
            ]),
        )
        .expect("a transient buffer");
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    trace_from_here(&mut adapter);

    // The brackets are not a stub: this family issues a copy as a direct
    // command with no scope around it, so there is no state for them to
    // establish and none to tear down, and they emit nothing.
    assert!(adapter.begin_copy(&mut encoder, "unrecorded").is_ok());
    assert!(adapter.end_copy(&mut encoder).is_ok());
    assert!(
        calls(&mut adapter).is_empty(),
        "a copy scope has no GL counterpart to open or close"
    );

    // Which is exactly why the copy itself carries the whole cost: it is the
    // one thing between the brackets that reaches the driver.
    assert!(
        adapter
            .copy_buffer(
                &mut encoder,
                &buffer.physical,
                &buffer.physical,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 32,
                    size: 32,
                },
            )
            .is_ok()
    );
    assert!(
        matches!(
            calls(&mut adapter).as_slice(),
            [MockCall::CopyBuffer { .. }]
        ),
        "one copy between two no-op brackets is one command: {:?}",
        calls(&mut adapter)
    );

    // No presentation token can exist either, so this adapter can only ever be
    // handed an empty presentation list.  That is a fact about the type rather
    // than a check, and it is the reason `submit` below never sees a token.
    let presentations: Vec<PresentationSubmission<super::UnsupportedPresentationToken>> = vec![];
    assert!(presentations.is_empty());
}

// ---------------------------------------------------------------------------
// The completion path.
// ---------------------------------------------------------------------------

/// Submits one empty command buffer and returns the completion for it.
fn submit_empty(adapter: &mut Adapter) -> crate::webgl2::api::GlFenceLease {
    let encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let command_buffer = adapter.finish_encoder(encoder).expect("a command buffer");
    adapter
        .submit(QueueId::new(0), command_buffer, vec![])
        .expect("a submission")
}

#[test]
fn a_submission_flushes_and_then_creates_its_fence() {
    let mut adapter = adapter();
    trace_from_here(&mut adapter);

    let completion = submit_empty(&mut adapter);

    // The order is the whole contract of this pair: a fence reports the
    // commands issued before it, so creating it before the flush would make it
    // report the previous submission instead of this one.
    assert!(
        matches!(
            calls(&mut adapter).as_slice(),
            [MockCall::Flush, MockCall::CreateFence(_)]
        ),
        "a submission flushes and then fences: {:?}",
        calls(&mut adapter)
    );
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Pending,
        "nothing has observed it yet, and pending is the honest answer for work just issued"
    );
}

#[test]
fn a_submission_on_a_queue_that_does_not_exist_is_refused() {
    let mut adapter = adapter();
    let encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let command_buffer = adapter.finish_encoder(encoder).expect("a command buffer");

    assert_eq!(
        refused(adapter.submit(QueueId::new(1), command_buffer, vec![])),
        "submit"
    );
}

#[test]
fn a_completed_submission_releases_its_leases_and_destroys_its_fence() {
    let mut adapter = adapter();
    let bound = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient texture");
    let held = bound.lease.clone();
    let completion = submit_empty(&mut adapter);
    adapter.retire(completion, vec![held]);
    // The other handle goes, and the object must survive it: the submission is
    // still in flight and the ledger is holding its own clone.
    drop(bound);
    trace_from_here(&mut adapter);

    adapter
        .machine
        .backend()
        .inject_fence_status(completion.fence, GlFenceStatus::Complete);

    assert_eq!(adapter.collect_retired().expect("a poll"), 1);
    assert_eq!(
        count(&mut adapter, is_destroy_texture),
        1,
        "the frame that learned the work is done is the frame that frees it"
    );
    assert_eq!(
        count(&mut adapter, is_destroy_fence),
        1,
        "and the fence is a driver object that nothing will ask about again"
    );
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Complete,
        "the record is what a later question is answered from"
    );
}

#[test]
fn a_pending_submission_keeps_everything_it_holds() {
    let mut adapter = adapter();
    let bound = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient texture");
    let completion = submit_empty(&mut adapter);
    adapter.retire(completion, vec![bound.lease.clone()]);
    drop(bound);
    trace_from_here(&mut adapter);

    // No injection: the recorder's own default is `Pending`, which is the only
    // honest default for a context with no submission queue.
    assert_eq!(adapter.collect_retired().expect("a poll"), 0);
    assert_eq!(
        count(&mut adapter, is_destroy_texture),
        0,
        "nothing was released"
    );
    assert_eq!(
        count(&mut adapter, is_destroy_fence),
        0,
        "and no fence was destroyed"
    );
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Pending
    );
}

#[test]
fn an_unobserved_fence_leaves_its_submission_quarantined() {
    let mut adapter = adapter();
    let bound = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient texture");
    let completion = submit_empty(&mut adapter);
    adapter.retire(completion, vec![bound.lease.clone()]);
    drop(bound);
    trace_from_here(&mut adapter);

    adapter
        .machine
        .backend()
        .inject_fence_status(completion.fence, GlFenceStatus::Unknown);

    // `Unknown` is not a terminal outcome in this contract, and treating it as
    // one would release physical objects that submitted work may still
    // reference.  The submission keeps everything it holds.
    assert_eq!(adapter.collect_retired().expect("a poll"), 0);
    assert_eq!(count(&mut adapter, is_destroy_texture), 0);
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Unknown
    );
}

#[test]
fn a_failed_submission_releases_and_reports_which_failure_it_was() {
    let mut adapter = adapter();
    let bound = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient texture");
    let completion = submit_empty(&mut adapter);
    adapter.retire(completion, vec![bound.lease.clone()]);
    drop(bound);
    trace_from_here(&mut adapter);

    adapter
        .machine
        .backend()
        .inject_fence_status(completion.fence, GlFenceStatus::Failed);

    assert_eq!(adapter.collect_retired().expect("a poll"), 1);
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Failed(CompletionFailure::ExecutionFailed),
        "the context is still active, so this is work that failed rather than a device that went away"
    );
    assert_eq!(count(&mut adapter, is_destroy_texture), 1);
}

#[test]
fn retiring_a_submission_that_already_settled_releases_where_it_is_handed_over() {
    let mut adapter = adapter();
    let completion = submit_empty(&mut adapter);
    adapter
        .machine
        .backend()
        .inject_fence_status(completion.fence, GlFenceStatus::Complete);
    assert_eq!(
        adapter.collect_retired().expect("a poll"),
        1,
        "one submission settled, and nothing had been retired into it: the count is settlements, not leases"
    );

    // The common case, and the reason `Retirement::Release` exists: a caller
    // that polls its own submission to a terminal outcome never retires it at
    // all, so by the time this call arrives nothing here is waiting for it.
    let bound = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient texture");
    trace_from_here(&mut adapter);
    adapter.retire(completion, vec![bound.lease.clone()]);
    drop(bound);

    // Nothing has destroyed it yet: the release is a record, and the record is
    // acted on at the adapter's next entry point.
    assert_eq!(count(&mut adapter, is_destroy_texture), 0);
    adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    assert_eq!(
        count(&mut adapter, is_destroy_texture),
        1,
        "the next entry point destroys what the release recorded"
    );
}

// ---------------------------------------------------------------------------
// Transient lifetime, which is a handle count rather than a frame.
// ---------------------------------------------------------------------------

#[test]
fn a_transient_lives_until_the_last_handle_that_retains_it_drops() {
    let mut adapter = adapter();
    let bound = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient texture");
    let retained = bound.lease.clone();
    let physical = bound.physical;
    assert_eq!(
        bound.identity,
        transient::resource_identity(physical.slot, physical.generation),
        "the common identity is the Layer 1 slot and generation, and nothing else"
    );
    drop(bound);
    trace_from_here(&mut adapter);

    adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    assert_eq!(
        count(&mut adapter, is_destroy_texture),
        0,
        "a clone is still held, and the executor's pool holds one for a slot's whole life"
    );

    drop(retained);
    adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    assert_eq!(count(&mut adapter, is_destroy_texture), 1);
}

#[test]
fn a_buffer_transient_travels_the_same_path() {
    let mut adapter = adapter();
    let bound = adapter
        .create_transient_buffer(
            BufferDesc { size: 256 },
            BufferUsage::from_kinds([
                BufferUsageKind::CopySource,
                BufferUsageKind::CopyDestination,
            ]),
        )
        .expect("a transient buffer");
    assert!(
        adapter
            .create_transient_buffer(BufferDesc { size: 256 }, BufferUsage::empty())
            .is_err()
    );
    trace_from_here(&mut adapter);

    drop(bound);
    adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    assert_eq!(count(&mut adapter, is_destroy_buffer), 1);
}

#[test]
fn a_context_generation_change_forgets_what_the_previous_one_released() {
    let mut adapter = adapter();
    let identity = adapter.device_identity();
    let bound = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient texture");
    // Releases it while the old context is still the current one, so the record
    // is in the queue when the change is noticed.
    drop(bound);
    trace_from_here(&mut adapter);

    adapter
        .machine
        .backend()
        .context_lost()
        .expect("context loss");
    adapter
        .machine
        .backend()
        .context_restored()
        .expect("context restoration");

    assert!(
        adapter.begin_encoder(QueueId::new(0)).is_ok(),
        "the restored context accepts work again"
    );
    assert_eq!(
        count(&mut adapter, is_destroy_texture),
        0,
        "the restored context invalidated every object of the previous epoch, so destroying one would be a call against a dead generation"
    );
    assert_ne!(
        adapter.device_identity(),
        identity,
        "a new context generation is a new common device identity: the objects of the old one are not this device's"
    );
}

// ---------------------------------------------------------------------------
// The lowering, called directly because the target it picks is not observable
// through the adapter.
// ---------------------------------------------------------------------------

#[test]
fn the_array_layer_count_decides_the_target_and_the_third_extent_follows() {
    let plain = transient::texture_descriptor(plain_texture(), colour_usage()).expect("lowered");
    assert_eq!(plain.dimension, GlTextureDimension::D2);
    assert_eq!(
        plain.extent,
        GlExtent3d {
            width: 4,
            height: 4,
            depth_or_layers: 1
        }
    );

    let arrayed =
        transient::texture_descriptor(texture(TextureDimension::D2, 3, 1), colour_usage())
            .expect("lowered");
    assert_eq!(
        arrayed.dimension,
        GlTextureDimension::D2Array,
        "a layer count above one makes the array its own target here"
    );
    assert_eq!(
        arrayed.extent.depth_or_layers, 3,
        "and the count becomes its depth"
    );

    let volume = transient::texture_descriptor(texture(TextureDimension::D3, 1, 6), colour_usage())
        .expect("lowered");
    assert_eq!(volume.dimension, GlTextureDimension::D3);
    assert_eq!(
        volume.extent.depth_or_layers, 6,
        "a volume keeps the third component as texels"
    );
}

#[test]
fn a_shape_the_gl_family_has_no_target_for_is_refused() {
    // A layered one-dimensional texture, and a volume with layers: neither has a
    // target, and the refusal is at the lowering rather than in the driver.
    assert_eq!(
        refused(transient::texture_descriptor(
            texture(TextureDimension::D1, 2, 1),
            colour_usage()
        )),
        "create-transient-texture"
    );
    assert_eq!(
        refused(transient::texture_descriptor(
            texture(TextureDimension::D3, 4, 4),
            colour_usage()
        )),
        "create-transient-texture"
    );
    assert_eq!(
        refused(transient::texture_descriptor(
            texture(TextureDimension::D2, 0, 1),
            colour_usage()
        )),
        "create-transient-texture",
        "no dimension can carry a layer count of zero"
    );
}

#[test]
fn a_request_with_nothing_gl_usage_can_express_has_no_descriptor() {
    let mut presentation = plain_texture();
    presentation.format = TextureFormat::Bgra8Unorm;

    assert_eq!(
        refused(transient::texture_descriptor(
            presentation,
            TextureUsage::from_kinds([TextureUsageKind::Sampled])
        )),
        "create-transient-texture",
        "no accepted GL-family profile has a BGRA texture format, so there is nothing to lower to"
    );
    assert_eq!(
        refused(transient::texture_descriptor(
            plain_texture(),
            TextureUsage::from_kinds([TextureUsageKind::Present])
        )),
        "create-transient-texture",
        "presentation is a statement about what happens after execution and not a creation fact"
    );
    assert_eq!(
        refused(transient::texture_descriptor(
            plain_texture(),
            TextureUsage::empty()
        )),
        "create-transient-texture"
    );
    assert_eq!(
        refused(transient::buffer_descriptor(
            BufferDesc { size: 64 },
            BufferUsage::empty()
        )),
        "create-transient-buffer",
        "and a buffer is created for some operation or not at all"
    );
}

#[test]
fn the_derived_usage_is_wider_than_the_request_where_the_gl_set_is_coarser() {
    let mut adapter = desktop();
    let bound = adapter
        .create_transient_texture(
            plain_texture(),
            TextureUsage::from_kinds([TextureUsageKind::StorageRead]),
        )
        .expect("a storage transient");

    // One GL bit covers shader storage in both directions and the physical
    // object really does permit both, so reporting only what was asked for
    // would understate the object.  The contract asks for the physical facts.
    assert!(bound.usage.contains(TextureUsageKind::StorageRead));
    assert!(
        bound.usage.contains(TextureUsageKind::StorageWrite),
        "one GL storage bit is both directions, and the physical object permits both"
    );

    // The attachment side comes from the format rather than from the request,
    // for the same reason: the GL bit does not distinguish the two sides.
    let attachment = transient::texture_usage(
        GlTextureUsage::RENDER_ATTACHMENT | GlTextureUsage::SAMPLED,
        GlFormat::Rgba8Unorm,
    );
    assert!(attachment.contains(TextureUsageKind::ColorAttachment));
    assert!(!attachment.contains(TextureUsageKind::DepthStencilAttachment));
    assert!(attachment.contains(TextureUsageKind::Sampled));

    let depth = transient::texture_usage(GlTextureUsage::RENDER_ATTACHMENT, GlFormat::Depth32Float);
    assert!(depth.contains(TextureUsageKind::DepthStencilAttachment));
    assert!(!depth.contains(TextureUsageKind::ColorAttachment));

    // The buffer side widens in exactly the same one place.
    let storage = transient::buffer_usage(GlBufferUsage::STORAGE);
    assert!(storage.contains(BufferUsageKind::StorageRead));
    assert!(storage.contains(BufferUsageKind::StorageWrite));
    assert!(
        !storage.contains(BufferUsageKind::Uniform),
        "and widens in no other: what the object does not permit is not reported"
    );
}

#[test]
fn every_common_buffer_operation_has_a_gl_role() {
    // The buffer side of the lowering is total on the operation set.  This is
    // what lets `buffer_descriptor` say that the only way to an empty
    // descriptor is a request with no operations in it, rather than a request
    // some operation of which went unrecognized.
    let every = BufferUsage::from_kinds([
        BufferUsageKind::Uniform,
        BufferUsageKind::StorageRead,
        BufferUsageKind::StorageWrite,
        BufferUsageKind::Vertex,
        BufferUsageKind::Index,
        BufferUsageKind::Indirect,
        BufferUsageKind::CopySource,
        BufferUsageKind::CopyDestination,
    ]);
    let lowered = transient::buffer_descriptor(BufferDesc { size: 64 }, every)
        .expect("every operation lowers");
    for bit in [
        GlBufferUsage::UNIFORM,
        GlBufferUsage::STORAGE,
        GlBufferUsage::VERTEX,
        GlBufferUsage::INDEX,
        GlBufferUsage::INDIRECT,
        GlBufferUsage::COPY_SOURCE,
        GlBufferUsage::COPY_DESTINATION,
    ] {
        assert!(
            lowered.usage.contains(bit),
            "every requested operation contributed its role"
        );
    }
    assert!(
        !lowered.usage.contains(GlBufferUsage::MAP_READ)
            && !lowered.usage.contains(GlBufferUsage::MAP_WRITE),
        "and no role was invented: a transient is device-local and the common contract has no mapping operation"
    );
}

#[test]
fn the_common_identity_separates_slots_and_generations() {
    assert_ne!(
        transient::resource_identity(0, 1),
        transient::resource_identity(1, 1),
        "two slots are two objects"
    );
    assert_ne!(
        transient::resource_identity(1, 1),
        transient::resource_identity(1, 2),
        "and one slot reused is a new object, which the generation is there to say"
    );
}

#[test]
fn the_format_correspondence_is_read_from_the_lowering_that_already_states_it() {
    assert_eq!(
        transient::gl_format(TextureFormat::Rgba8Unorm),
        Some(GlFormat::Rgba8Unorm)
    );
    assert_eq!(
        transient::gl_format(TextureFormat::Rgba8UnormSrgb),
        Some(GlFormat::Rgba8Srgb)
    );
    assert_eq!(
        transient::gl_format(TextureFormat::Rgba16Float),
        Some(GlFormat::Rgba16Float)
    );
    assert_eq!(
        transient::gl_format(TextureFormat::Depth32Float),
        Some(GlFormat::Depth32Float)
    );
    assert_eq!(
        transient::gl_format(TextureFormat::Bgra8Unorm),
        None,
        "the one common format no accepted GL-family profile has a counterpart for"
    );
}
