//! The compute pass kind, the dispatch that commits it, and the witness refusal.
//!
//! F4(c)'s subject is the optional command domain, and what these tests check is
//! the half a raster pass cannot show: that the brackets emit nothing, because
//! this family's pass boundary is a *framebuffer's*; that the two pass kinds
//! refuse each other's verbs in both directions at every verb that could be
//! reached; that a dispatch links, installs, binds and issues in one order, where
//! the order is the claim; and that an adapter whose *type* has no compute domain
//! refuses every verb while still registering the objects a frame described.
//!
//! [`super::compute_storage`] owns the sibling subject -- how a storage block and
//! an image unit lower against the device's own record of what it created --
//! and both suites share [`super::harness`]'s fixtures and readers.
//!
//! # Which adapter each test runs on, and why there are three
//!
//! [`compute_adapter`] is the only one whose *type* can dispatch: it names
//! [`WithCompute`](super::super::compute::WithCompute) and its snapshot proved
//! compute and storage buffers.  [`compute_adapter_without_storage_images`] is
//! the same type over a snapshot that proved those two and **not** storage
//! images, which is the only way to reach the second refusal -- the witness has
//! the domain and the context did not prove the capability.  [`desktop`] is the
//! mirror image: a context whose profile *can* lower a compute program and whose
//! witness is the default [`NoCompute`], which is the first refusal.  The second
//! of the three is exercised next door, at the one capability it withholds.
//!
//! # What the trace is evidence for
//!
//! [`a_dispatch_links_installs_binds_and_dispatches_in_that_order`] compares a
//! whole dispatch position by position, because in this domain the order *is* the
//! claim: the program has to be linked before anything is bound against it, and a
//! link invalidates the mirror's installed pipeline, so a different sequence is a
//! different -- and wrong -- set of driver calls.  The ids inside that trace are
//! then compared against each other, because a trace whose ids were merely
//! present would accept a dispatch that ran one program over another's buffer.

use fluxel_rendergraph::{
    BindingSetId, BufferDesc, BufferUsage, BufferUsageKind, ComputePipelineId, ExecutionBackend,
    QueueId, RasterPassDescriptor, RasterPipelineId, RenderObjectProvider, Viewport,
};

use super::super::compute::NoCompute;
use super::super::object::Recipe;
use super::{
    cleared, colour_usage, compute_adapter, compute_calls, compute_count, compute_trace_from_here,
    desktop, invalid, is_create_program, is_dispatch, is_install_program, is_pass_boundary,
    plain_texture, read_write, refusal_reason, refused, registered, storage_buffer,
};
use crate::resource::{ComputeKernel, RasterKernel};
use crate::webgl2::api::{GlDispatchGroups, GlError, MockCall};

#[test]
fn a_compute_pass_opens_and_closes_with_no_layer_one_boundary() {
    let mut adapter = compute_adapter();
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    compute_trace_from_here(&mut adapter);

    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("the context proved compute");
    adapter.end_compute(&mut encoder).expect("the pass closes");

    // The brackets are not a stub, and the empty trace is the evidence rather
    // than a spelling of one: this family's pass boundary is a *framebuffer's*, so
    // `begin_pass` and `end_pass` are about attachments and a dispatch has none.
    // A raster pass at these two points issues `BeginRenderPass` and
    // `EndRenderPass`; a compute pass issues nothing at all.
    assert!(
        compute_calls(&mut adapter).is_empty(),
        "a compute pass has no counterpart in Layer 1 to open or to close"
    );

    // Which is the same fact said the other way round, and worth saying: the
    // verbs were not merely quiet, they did not reach the boundary pair either.
    assert_eq!(compute_count(&mut adapter, is_pass_boundary), 0);
}

#[test]
fn a_command_buffer_cannot_be_finished_inside_a_compute_pass_either() {
    let mut adapter = compute_adapter();
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("the context proved compute");

    assert_eq!(
        invalid(adapter.finish_encoder(encoder)),
        "finish-encoder",
        "an abandoned pass is the same mistake whichever kind it was"
    );

    // And the *unwind* is the one thing that differs, which is why it is
    // conditioned on the shape: a raster pass found here is closed for the
    // backend's sake (`tests/raster.rs` holds that half), while a compute pass
    // opened nothing in Layer 1 and so has nothing to close.  Closing one anyway
    // would end a pass this adapter never began.
    assert_eq!(
        compute_count(&mut adapter, is_pass_boundary),
        0,
        "an unwound compute pass issues no boundary, because it never opened one"
    );
}

#[test]
fn the_two_pass_kinds_are_closed_by_their_own_verb_and_refuse_each_other_s() {
    let mut adapter = compute_adapter();
    let target = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient colour attachment");
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };

    // A compute pass reached by the raster close.  The refusal names the verb
    // that *does* close it, because the frame's next step is the whole of what it
    // needs to know -- and the pass is put back rather than taken, so that the
    // close this message names still has something to find.
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("the context proved compute");
    assert_eq!(invalid(adapter.end_raster(&mut encoder)), "end-raster");
    adapter
        .end_compute(&mut encoder)
        .expect("the pass was put back, so its own close still finds it");

    // And the other direction, which would otherwise leave a raster pass open
    // with no verb left to close it.
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    assert_eq!(invalid(adapter.end_compute(&mut encoder)), "end-compute");
    adapter.end_raster(&mut encoder).expect("the pass closes");

    // A verb of the other kind, and these are the two directions that matter:
    // rasterization state recorded in a compute pass would be recorded where
    // nothing reads it, and a dispatch inside a raster pass has no pass to be
    // recorded in.
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("a compute pass");
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
    adapter.end_compute(&mut encoder).expect("the pass closes");

    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    assert_eq!(
        invalid(adapter.dispatch(&mut encoder, [1, 1, 1])),
        "dispatch"
    );
    adapter.end_raster(&mut encoder).expect("the pass closes");

    // A raster recipe recorded in a compute pass is the remaining pair, and it is
    // the one that would be silently wrong rather than loudly: the next dispatch
    // would install it.
    let mut objects = adapter.object_registry();
    objects
        .register_raster_pipeline(RasterPipelineId::new(0), RasterKernel::Triangle)
        .expect("the triangle lowers on this profile");
    let triangle = objects
        .raster_pipeline(RasterPipelineId::new(0))
        .expect("it was registered");
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("a compute pass");
    assert_eq!(
        invalid(adapter.set_raster_pipeline(&mut encoder, &triangle.physical)),
        "set-raster-pipeline"
    );
    adapter.end_compute(&mut encoder).expect("the pass closes");
}

// ---------------------------------------------------------------------------
// The dispatch: the link, the install, the bindings and the command.
// ---------------------------------------------------------------------------

#[test]
fn a_dispatch_links_installs_binds_and_dispatches_in_that_order() {
    let mut adapter = compute_adapter();
    // The artifact with one binding, which makes the trace the dispatch's own
    // shape rather than a recipe's: `WrappingAdd` declares one read-write storage
    // block at binding zero and nothing else.
    let values = storage_buffer(&mut adapter, 256);
    let resources = [read_write(&values.physical)];
    let (pipeline, bindings) = registered(&mut adapter, ComputeKernel::WrappingAdd, &resources);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    compute_trace_from_here(&mut adapter);

    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("the context proved compute");
    adapter
        .set_compute_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set resolved for the installed recipe");
    adapter
        .dispatch(&mut encoder, [2, 1, 1])
        .expect("a dispatch of 65535-or-fewer workgroups per axis");
    adapter.end_compute(&mut encoder).expect("the pass closes");

    let trace = compute_calls(&mut adapter);
    match trace.as_slice() {
        [
            MockCall::CreateProgram(linked),
            MockCall::SelectProgram(selected),
            MockCall::SetComputeProgram(installed),
            MockCall::BindStorageBuffer {
                binding,
                buffer,
                offset,
                size,
            },
            MockCall::Dispatch(groups),
        ] => {
            assert_eq!(
                linked, installed,
                "the program that was linked is the one that was installed"
            );
            assert_eq!(
                linked, selected,
                "and installing it is what puts it in the driver's one current-program slot -- a link cleared that slot, so the selection is a call the frame never asked for"
            );
            assert_eq!(*binding, 0, "the artifact's own binding number");
            assert_eq!(
                buffer, &values.physical,
                "and the buffer the frame resolved behind it"
            );
            assert_eq!(*offset, 0, "a whole range starts at zero");
            assert_eq!(
                *size, 256,
                "and names the whole allocation, which is the size this device recorded when it created the buffer"
            );
            assert_eq!(
                groups,
                &GlDispatchGroups([2, 1, 1]),
                "the dispatch is the work-group triple the frame asked for, lowered without scale"
            );
        }
        other => panic!("the dispatch issued a different sequence: {other:?}"),
    }
    // Implied by the match above, and stated because it is the whole of what
    // makes this a *compute* trace: none of the five calls is a pass boundary.
    assert!(!trace.iter().any(is_pass_boundary));
}

#[test]
fn the_program_is_linked_once_and_re_asserted_at_every_dispatch() {
    let mut adapter = compute_adapter();
    let values = storage_buffer(&mut adapter, 256);
    let resources = [read_write(&values.physical)];
    let (pipeline, bindings) = registered(&mut adapter, ComputeKernel::WrappingAdd, &resources);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    compute_trace_from_here(&mut adapter);

    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("a compute pass");
    adapter
        .set_compute_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
    adapter
        .dispatch(&mut encoder, [1, 1, 1])
        .expect("a dispatch");
    adapter
        .dispatch(&mut encoder, [1, 1, 1])
        .expect("a dispatch");
    adapter.end_compute(&mut encoder).expect("the pass closes");

    // Two dispatches of one pass, which is the case the install order inside is
    // for: this family has one current program shared with the pipeline domain,
    // and any link between two dispatches moves that slot -- so the install is
    // re-asserted per dispatch rather than once per pass.  Here nothing moved it,
    // and the re-assertion is still issued: a mirror that believed the slot was
    // still its own would be relying on every other domain's good behaviour.
    assert_eq!(
        compute_count(&mut adapter, is_create_program),
        1,
        "the cache retains the program, so the second dispatch links nothing: {:?}",
        compute_calls(&mut adapter)
    );
    assert_eq!(
        compute_count(&mut adapter, is_install_program),
        2,
        "and installs it once per dispatch"
    );
    assert_eq!(compute_count(&mut adapter, is_dispatch), 2);
}

// ---------------------------------------------------------------------------
// The first refusal: an adapter whose type has no compute command domain.
// ---------------------------------------------------------------------------

#[test]
fn a_compute_pipeline_registers_where_the_profile_lowers_it_and_the_witness_refuses_it() {
    // The other refusal, and the pair is only testable side by side: `desktop()` is
    // a context whose profile has a compute shading language and whose adapter
    // type names no witness, so registration lowers and every verb refuses.
    let mut adapter = desktop();
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");

    // Registration is *inherent* and not gated on the witness -- see
    // `GlObjectRegistry`'s documentation -- and this is what that buys: a frame
    // can be described against this adapter and refused at the verb that would
    // have recorded a command, where the refusal can name itself.  The alternative
    // ordering would fail the registration, which is a fact about a *description*
    // rather than about a command.
    let mut objects = adapter.object_registry();
    let pipeline_id = ComputePipelineId::new(0);
    objects
        .register_compute_pipeline(pipeline_id, ComputeKernel::WrappingAdd)
        .expect("a desktop profile lowers a compute program");
    let pipeline = objects
        .compute_pipeline(pipeline_id)
        .expect("and the registry resolves it like any other object");

    let values = adapter
        .create_transient_buffer(
            BufferDesc { size: 256 },
            BufferUsage::from_kinds([BufferUsageKind::StorageRead, BufferUsageKind::StorageWrite]),
        )
        .expect("a transient storage buffer");
    let set_id = BindingSetId::new(0);
    objects.register_bindings(set_id, Recipe::Compute(ComputeKernel::WrappingAdd));
    let bindings = objects
        .bindings(set_id, &[read_write(&values.physical)], &[])
        .expect("the set is validated, not dispatched");

    assert_eq!(
        refusal_reason(adapter.begin_compute(&mut encoder, "compute")),
        NoCompute::REFUSAL,
        "the pass bracket refuses first, and names the adapter's type rather than the request"
    );
    assert_eq!(
        refused(adapter.set_compute_pipeline(&mut encoder, &pipeline.physical)),
        "set-compute-pipeline",
        "and so does every verb of the domain, each naming itself"
    );
    assert_eq!(
        refused(adapter.dispatch(&mut encoder, [1, 1, 1])),
        "dispatch"
    );

    // The shared verb is the exception, and the reason is that it is *shared*: a
    // binding set is recorded into either kind of pass, so `set_bindings` never
    // reaches the witness.  What it reports instead is what is actually missing
    // here -- the bracket above left no pass behind -- and that is a validation
    // failure rather than a refusal, because a caller that opened a pass could
    // make the same call succeed.
    let error = adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect_err("the shared verb reaches this encoder with no pass open");
    assert!(matches!(
        error,
        GlError::Validation { operation, .. } if operation == "set-bindings"
    ));
    // And the *sentence* is asserted, because this call site is the reason it
    // changed: the refusal used to name a raster pass and a draw's missing
    // framebuffer, and a compute encoder reaching a shared verb is exactly the
    // caller that made both halves false.  Nothing else in the trace can show
    // that -- the operation name was already right.
    let GlError::Validation { message, .. } = &error else {
        unreachable!("matched above")
    };
    assert_eq!(
        message, "no pass is open on this encoder",
        "the shared verb is told what is missing and not which kind of pass it wanted"
    );
    adapter
        .finish_encoder(encoder)
        .expect("an encoder with no pass open finishes");
}
