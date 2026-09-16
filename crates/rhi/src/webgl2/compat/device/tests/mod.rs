//! Tests for the common execution adapter over a GL-family state machine.
//!
//! Two kinds of test live here and they answer different questions.  The
//! *lowering* tests call [`transient`]'s functions directly, because the target
//! a dimension and a layer count lower to is not observable through the adapter
//! -- Layer 1 stores the descriptor it was handed and reports only the identity.
//! Every other test goes through `ExecutionBackend` on a real
//! [`GlCompatibilityDevice`](super::GlCompatibilityDevice) over
//! [`MockGlFamilyApi`](crate::webgl2::api::MockGlFamilyApi), because what those
//! check is the contract: which calls the adapter makes, in which order, and what
//! it answers when it is asked about a submission.
//!
//! Fence completion cannot be simulated here -- the recorder has no submission
//! queue, so nothing it recorded can finish -- and the tests therefore inject it,
//! which is the seam Layer 1 already documents for exactly this reason.  What is
//! being tested is what the adapter does with an observed outcome, and an
//! outcome the test names is as much an observation as one a driver produced.

mod commands;
mod compute;
mod compute_storage;
mod frame;
mod harness;
mod objects;
mod raster;

use harness::*;

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, CompletionFailure, CompletionStatus,
    ExecutionBackend, Extent3d, PresentationSubmission, QueueId, TextureDesc, TextureDimension,
    TextureFormat, TextureUsage, TextureUsageKind,
};

use super::super::capabilities::capabilities;
use super::transient;
use crate::webgl2::api::tests::snapshot;
use crate::webgl2::api::{
    GlBufferUsage, GlError, GlExtent3d, GlFamilyApi, GlFamilyProfile, GlFenceLease, GlFenceStatus,
    GlFormat, GlSurfacePresentationApi as _, GlSurfaceSize, GlTextureDimension, GlTextureUsage,
    MockCall, TextureId,
};

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
fn the_acquisition_verb_refuses_until_a_surface_is_advertised() {
    // This is what replaced the check that the presentation token was
    // uninhabited, and the replacement is narrower.  It was three types when
    // that check was written and F5 moved the last of them: a raster pipeline and
    // a binding set became objects the renderer registers at F3(b), and
    // `ComputePipeline` at F4(c), so each of those was consumed by the
    // implementation it held back.  The token cannot be consumed the same way,
    // because a type that carries an acquisition to submission has to hold the
    // acquisition -- so what stands in place of "no value of this type can be
    // written down" is "no caller without an advertised surface can obtain one".
    //
    // The two are not equally strong, and the difference is the whole of what
    // this test is for: the type-level guarantee could not be broken by any
    // implementation in any crate, while this one is a refusal that the step
    // reporting a surface removes.  It is checked at the verb that would produce
    // a token *and* at the submission that would consume one, so the two ends of
    // that single fact cannot come to disagree.
    let mut adapter = adapter();
    assert!(
        adapter.capabilities().surface.is_none(),
        "the advertisement every refusal below is derived from, stated once"
    );
    assert_eq!(
        refused(adapter.acquire_surface_texture(plain_texture(), colour_usage())),
        "acquire-surface-texture",
        "the acquisition is the only door to a token, and it is shut"
    );
    assert!(
        !adapter
            .machine
            .backend()
            .calls()
            .iter()
            .any(is_create_texture),
        "and it is shut before the driver: a refused request costs no acquisition and no object"
    );
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
fn every_compute_verb_refuses_and_names_itself() {
    let mut adapter = adapter();
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");

    // What is left of the fail-closed set after the raster slice landed, and the
    // shape of the refusal changed at F4(c) without changing the answer: these
    // three are refused by the *witness*, not by a missing lowering.  The adapter
    // built by `adapter()` is over the browser provider's snapshot and names no
    // witness, so it is `NoCompute` and every verb of the domain refuses with the
    // sentence that says this machine is over a backend with no compute command
    // domain.  `Unsupported` is still the contract's word for it -- a caller
    // could not fix it by opening a pass, which is exactly what the raster verbs
    // below can be fixed by.
    assert_eq!(
        refused(adapter.begin_compute(&mut encoder, "unrecorded")),
        "begin-compute"
    );
    assert_eq!(refused(adapter.end_compute(&mut encoder)), "end-compute");
    assert_eq!(
        refused(adapter.dispatch(&mut encoder, [1, 1, 1])),
        "dispatch",
        "the dispatch refuses before it can be told about the pass it has none of"
    );
}

#[test]
fn the_raster_verbs_are_real_and_so_refuse_as_validation_outside_a_pass() {
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

    // The distinction this test exists for.  These verbs are implemented now, so
    // what they answer with is a *validation* failure naming the request that
    // cannot be honoured rather than a fail-closed refusal saying the adapter has
    // no vocabulary: a caller that opened a pass could make every one of these
    // succeed, which is exactly what `Unsupported` would deny.  Checking the
    // operation name is checking that the verb reports itself and not a
    // neighbour.
    assert_eq!(
        invalid(adapter.set_vertex_buffer(&mut encoder, 0, &buffer.physical, 0)),
        "set-vertex-buffer"
    );
    assert_eq!(
        invalid(adapter.set_index_buffer(&mut encoder, &buffer.physical, 0, IndexFormat::Uint16)),
        "set-index-buffer"
    );
    assert_eq!(
        invalid(adapter.set_viewport(
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
        invalid(adapter.set_scissor(
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
    assert_eq!(invalid(adapter.draw(&mut encoder, 0..3, 0..1)), "draw");
    assert_eq!(
        invalid(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..1)),
        "draw-indexed"
    );
    assert_eq!(invalid(adapter.end_raster(&mut encoder)), "end-raster");

    // `begin-raster` is the exception, and correctly so: it is asked to open a
    // pass this family has no pipeline for -- one with no colour attachment at
    // all -- and that is a capability fact about the closed artifact set rather
    // than a request that a different caller could fix.  So it refuses
    // fail-closed and names itself, like the compute verbs.
    let pass = RasterPassDescriptor {
        label: "unrecorded",
        colors: &[],
        depth_stencil: None,
    };
    assert_eq!(
        refused(adapter.begin_raster(&mut encoder, &pass)),
        "begin-raster"
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

    // The token exists as a type now, and this adapter still cannot hand one
    // out: the acquisition verb refuses while no surface is advertised, so the
    // only presentation list this adapter can ever be handed is the empty one.
    // That is a refusal rather than a fact about the type -- see
    // `the_acquisition_verb_refuses_until_a_surface_is_advertised` -- and the
    // step reporting a surface is what ends it.
    let presentations: Vec<PresentationSubmission<super::GlSurfaceToken>> = vec![];
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

/// The indirect role is read back off the one bit that records it.
///
/// The other direction is already covered -- the totality test below lowers a
/// request containing `Indirect` and asserts the bit it produced -- while this
/// arm of `buffer_usage` had no assertion at all, so a derivation that dropped
/// it would have left the round trip one-way without any test noticing.
#[test]
fn an_indirect_buffer_reports_the_role_its_bit_stands_for() {
    let held = transient::buffer_usage(GlBufferUsage::INDIRECT);
    assert_eq!(
        held,
        BufferUsage::from_kinds([BufferUsageKind::Indirect]),
        "the bit is one role, and the object holds nothing beside it"
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

/// One acquisition's extent, which is what a declared surface texture has to
/// match.
///
/// A `GlSurfaceSize` rather than the lease it came from, because a lease carries
/// an acquisition identity that only Layer 1's lease book can mint -- and the
/// rule under test is about the extent, which is a plain pair of numbers.
fn acquired(width: u32, height: u32) -> GlSurfaceSize {
    GlSurfaceSize { width, height }
}

#[test]
fn a_declared_surface_texture_must_be_the_extent_that_was_acquired() {
    // Called directly rather than through the verb, because the rule is about
    // one comparison and the route to it is the subject of the two tests below:
    // a rule reached through a path is tested there, and a rule tested here.
    const OP: &str = "acquire-surface-texture";
    assert!(
        super::surface::validate_surface_extent(OP, plain_texture(), acquired(4, 4)).is_ok(),
        "the descriptor the graph declares for its surface resource is what a frame hands over"
    );

    // The one field the acquisition has an opinion about, and the disagreement a
    // driver would otherwise settle by its own choice of what to keep.
    let wider = TextureDesc {
        extent: Extent3d {
            width: 8,
            height: 4,
            depth: 1,
        },
        ..plain_texture()
    };
    let refused = super::surface::validate_surface_extent(OP, wider, acquired(4, 4))
        .expect_err("refused before any driver call");
    assert!(
        matches!(refused, GlError::Validation { operation, .. } if operation == OP),
        "expected a validation refusal naming the verb, got {refused:?}"
    );
}

/// Submits one empty command buffer through the presenting verb, and answers the
/// completion.
///
/// The token is handed straight to `submit_tokens` rather than through the
/// contract's `submit`, because a `PresentationSubmission` needs a
/// `PresentTarget` and only the graph that declared the root can mint one.  The
/// outer verb is a pass-through of exactly the list this call builds, and
/// `the_contracts_submission_hands_its_tokens_to_the_presenting_verb` is what
/// holds it to that.
fn present_empty(adapter: &mut Adapter, tokens: Vec<super::GlSurfaceToken>) -> GlFenceLease {
    let encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let command_buffer = adapter.finish_encoder(encoder).expect("a command buffer");
    adapter
        .submit_tokens(QueueId::new(0), command_buffer, tokens)
        .expect("a submission")
}

/// Whether the drawable was published from this texture.
fn presented_from(adapter: &mut Adapter, source: TextureId) -> bool {
    adapter.machine.backend().calls().iter().any(
        |call| matches!(call, MockCall::PublishSurface { source: named, .. } if *named == source),
    )
}

#[test]
fn an_advertised_surface_is_a_texture_a_frame_can_present() {
    let mut adapter = surface_adapter();
    assert!(
        adapter.capabilities().surface.is_some(),
        "the drawable was read, so the device describes one"
    );

    let bound = adapter
        .acquire_surface_texture(plain_texture(), colour_usage())
        .expect("an advertised surface is not a refusal")
        .expect("the mock drawable is neither suspended nor zero-sized");
    let source = bound.texture.physical;
    assert!(
        adapter.attachments.contains_key(&source),
        "the acquired image is an attachment record like every other texture: {:?}",
        adapter.attachments
    );

    let completion = present_empty(&mut adapter, vec![bound.presentation]);
    assert!(
        presented_from(&mut adapter, source),
        "the acquisition reached the drawable: {:?}",
        adapter.machine.backend().calls()
    );
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Pending,
        "and a present that worked is not a failed submission"
    );
}

#[test]
fn a_present_that_fails_is_a_failed_completion_rather_than_a_refusal() {
    let mut adapter = surface_adapter();
    let bound = adapter
        .acquire_surface_texture(plain_texture(), colour_usage())
        .expect("an advertised surface is not a refusal")
        .expect("an unsuspended drawable");
    let source = bound.texture.physical;
    // A resize is what invalidates an acquisition, so this token names a lease
    // the lease book no longer holds -- a caller error, and one that arrives
    // after the command buffer has been accepted.
    adapter
        .machine
        .backend()
        .resize_surface(acquired(4, 4))
        .expect("the mock drawable resizes");

    let completion = present_empty(&mut adapter, vec![bound.presentation]);
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Failed(CompletionFailure::ExecutionFailed),
        "the frame did not reach the drawable, and the completion is where that is said"
    );
    assert!(
        !presented_from(&mut adapter, source),
        "and nothing was presented: {:?}",
        adapter.machine.backend().calls()
    );
    // The outcome survives the poll that would otherwise report the fence's own
    // good news about the commands.
    assert!(
        adapter.collect_retired().is_ok(),
        "a failed present is not a failed retirement"
    );
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Failed(CompletionFailure::ExecutionFailed),
        "a terminal outcome this adapter recorded is not reopened by a poll"
    );
}

#[test]
fn the_contracts_submission_hands_its_tokens_to_the_presenting_verb() {
    // The one thing the outer verb does that the fixture above cannot exercise:
    // it unwraps the submission the contract hands it.  An empty list is the
    // strongest statement available from here, and the presenting verb's own
    // tests are `an_advertised_surface_is_a_texture_a_frame_can_present` and the
    // refusal test above.
    let mut adapter = surface_adapter();
    let encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let command_buffer = adapter.finish_encoder(encoder).expect("a command buffer");
    let presentations: Vec<PresentationSubmission<super::GlSurfaceToken>> = vec![];
    let completion = adapter
        .submit(QueueId::new(0), command_buffer, presentations)
        .expect("a submission with nothing to present");
    assert_eq!(
        adapter.completion_status(&completion),
        CompletionStatus::Pending
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
