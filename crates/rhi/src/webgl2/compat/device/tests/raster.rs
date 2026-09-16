//! The raster pass, and the draws that commit it.
//!
//! F3(c)'s subject is the one place this adapter *drives* Layer 2 instead of
//! relaying to it.  A pass is recorded as the frame's verbs arrive and issued at
//! the draw, because a GL-family pipeline names its vertex array in the same
//! value as its rasterization state and the contract supplies those pieces in
//! the other order; so what these tests check is the seam that decision creates.
//! Which calls a frame's verb order becomes, what a pass nobody draws does, and
//! which requests are refused before anything is installed for them.
//!
//! # What a trace here asserts, and what it deliberately does not
//!
//! [`a_pass_that_draws_issues_the_calls_layer_two_needs_in_the_order_it_needs_them`]
//! compares a whole pass position by position, because in this family the order
//! *is* the claim: a program installed before it was linked, or geometry bound
//! before its array existed, would produce a different sequence.  The ids inside
//! that trace are then compared against each other -- the framebuffer opened is
//! the one that was created, the array the draw bound is the one the pipeline
//! names -- because a trace whose ids were merely present would accept a frame
//! that rendered into one attachment and drew with another's geometry.
//!
//! Every other trace here anchors on the calls its own subject is about and lets
//! the rest be absorbed, either by a leading wildcard or by `trace_from_here`.  A
//! second full-sequence comparison would be a copy of the first that any change
//! to the lowering breaks twice, and what the later tests ask is whether a
//! *particular* call happened, in what order, with which arguments -- not whether
//! the pass has the shape the first test already pinned.
//!
//! What is *not* asserted is that a pass's framebuffer is destroyed when it
//! closes.  Layer 2 reports whether a pass owns what it derived, and the machine
//! this adapter always builds retains it (`ExecutionMode::Optimized`, whose
//! budget fits everything these tests make), so the owning path is unreachable
//! from [`super::Adapter`]'s only constructor and is exercised by Layer 2's own
//! suite.  What these tests pin is the half that is reachable: that this adapter
//! frees nothing it was not told it owned.
//!
//! # The two refusal kinds, kept apart
//!
//! A pass this family has no pipeline for is `refused` -- a capability fact about
//! the closed artifact set that no caller could fix -- while a request that
//! disagrees with the artifact or the object it names is `invalid`, because a
//! caller that changed it could make the same call succeed.  Each test below says
//! which it is asserting and why, and the numbers they turn on are read off the
//! artifacts' own identity table rather than written here.

use fluxel_rendergraph::{
    BindingResourceSemantic, BindingSetId, BoundBindings, BoundBuffer, BoundRasterPipeline,
    BoundTexture, BufferDesc, BufferRange, BufferReadUse, BufferUsage, BufferUsageKind,
    ExecutionBackend, LoadOp, QueueId, RasterPassDescriptor, RasterPipelineId,
    RenderObjectProvider, ResolvedBindingResource, ScissorRect, StoreOp, TextureAspect,
    TextureRange, TextureReadUse, Viewport,
};

use super::super::object::{Bindings, RasterPipeline};
use super::super::retention::GlRetentionLease;
use super::{
    Adapter, adapter, attachment, calls, cleared, colour_usage, count, depth, invalid,
    plain_texture, refused, resolved, sampled, sampled_usage, trace_from_here, uniform,
};
use crate::resource::RasterKernel;
use crate::webgl2::api::{
    BufferId, GlDrawCommand, GlFamilyApi, GlNonIndexedDraw, GlTextureTarget, MockCall, TextureId,
};

/// A viewport over the whole of the 4x4 attachments every test here renders into,
/// which is also the viewport a pass defaults to.
fn whole_attachment() -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: 4.0,
        height: 4.0,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

/// The whole of a 4x4 attachment as a scissor rectangle.
fn whole_scissor() -> ScissorRect {
    ScissorRect {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    }
}

/// The colour attachment one pass renders into.
///
/// Handed back whole rather than as its identity because the lease has to outlive
/// every use of the id: the record this adapter lowers attachments from is
/// dropped when the object is destroyed, and a test that let the lease go early
/// would be attaching a texture the device no longer holds.
fn colour_target(adapter: &mut Adapter) -> BoundTexture<TextureId, GlRetentionLease> {
    adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient colour attachment")
}

/// The pipeline and binding set a frame would be handed for `kernel`.
///
/// Registered through the adapter's own registry rather than built directly, so
/// that every test below drives the two objects the way a frame does: F1 left
/// "the boundary's consumer accepts this adapter" as a compile-time check, and
/// resolving through the registry is what turns it into a run.
///
/// `resources` is the resolution for `kernel`; an artifact that reads nothing
/// still gets a set, because a frame resolves one per draw whatever the recipe
/// declares and the adapter refuses a draw whose pass recorded none.
fn registered(
    adapter: &mut Adapter,
    kernel: RasterKernel,
    resources: &[ResolvedBindingResource<'_, TextureId, BufferId>],
) -> (
    BoundRasterPipeline<RasterPipeline, GlRetentionLease>,
    BoundBindings<Bindings, GlRetentionLease>,
) {
    let mut objects = adapter.object_registry();
    let pipeline_id = RasterPipelineId::new(0);
    let set_id = BindingSetId::new(0);
    objects
        .register_raster_pipeline(pipeline_id, kernel)
        .unwrap_or_else(|error| panic!("{kernel:?} is lowered on WebGL2: {error:?}"));
    objects.register_bindings(set_id, kernel);
    let pipeline = objects
        .raster_pipeline(pipeline_id)
        .unwrap_or_else(|error| panic!("{kernel:?} was registered: {}", error.context.detail));
    let bindings = objects
        .bindings(set_id, resources, &[])
        .unwrap_or_else(|error| panic!("a set for {kernel:?}: {}", error.context.detail));
    (pipeline, bindings)
}

/// A transient buffer in the one usage the caller names.
///
/// Handed back whole, like the attachment, and for a sharper reason: a binding
/// resource a frame resolved is *live* for as long as its lease is held, and the
/// geometry domain refuses an array built over a buffer the device has released.
/// A helper returning the bare identity would make every test here depend on the
/// release queue happening not to have drained yet.
fn buffer(
    adapter: &mut Adapter,
    size: u64,
    kind: BufferUsageKind,
) -> BoundBuffer<BufferId, GlRetentionLease> {
    adapter
        .create_transient_buffer(BufferDesc { size }, BufferUsage::from_kinds([kind]))
        .unwrap_or_else(|error| panic!("a transient {kind:?} buffer: {error:?}"))
}

/// A transient sampled texture, as a binding's physical resource.
fn sampled_texture(adapter: &mut Adapter) -> BoundTexture<TextureId, GlRetentionLease> {
    adapter
        .create_transient_texture(plain_texture(), sampled_usage())
        .expect("a transient sampled texture")
}

/// Every mock call that destroyed an object a pass could have derived.
fn destroys(call: &MockCall) -> bool {
    matches!(
        call,
        MockCall::DestroyFramebuffer(_)
            | MockCall::DestroyVertexArray(_)
            | MockCall::DestroyProgram(_)
    )
}

/// How many of the calls so far installed a raster pipeline.
fn installs(call: &MockCall) -> bool {
    matches!(call, MockCall::SetRasterPipeline { .. })
}

/// Whether a call created one of the things a pass derives, or a shader.
fn derives_anything(call: &MockCall) -> bool {
    matches!(
        call,
        MockCall::CreateFramebuffer(_)
            | MockCall::CreateProgram(_)
            | MockCall::CreateVertexArray(_)
            | MockCall::CreateShader(_)
    )
}

#[test]
fn a_pass_that_draws_issues_the_calls_layer_two_needs_in_the_order_it_needs_them() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    // The bare triangle: no vertex buffers, no index buffer, no bindings.  It is
    // the smallest artifact that can be drawn, which makes the trace below the
    // pass's own shape rather than a recipe's.
    let (pipeline, bindings) = registered(&mut adapter, RasterKernel::Triangle, &[]);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    trace_from_here(&mut adapter);

    adapter
        .begin_raster(&mut encoder, &pass)
        .expect("one colour attachment at index zero");
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set resolved for the installed recipe");
    adapter
        .set_viewport(&mut encoder, whole_attachment())
        .expect("a viewport over the attachment");
    adapter
        .set_scissor(&mut encoder, whole_scissor())
        .expect("a scissor over the attachment");
    adapter
        .draw(&mut encoder, 0..3, 0..1)
        .expect("the bare triangle");
    adapter.end_raster(&mut encoder).expect("the pass closes");

    let trace = calls(&mut adapter);
    match trace.as_slice() {
        [
            MockCall::CreateFramebuffer(created),
            MockCall::BeginRenderPass(begun),
            MockCall::CreateProgram(linked),
            MockCall::CreateVertexArray(derived),
            MockCall::BindVertexArray(bound),
            MockCall::SelectProgram(selected),
            MockCall::SetRasterPipeline {
                program,
                vertex_array,
            },
            MockCall::BindVertexArray(rebound),
            MockCall::DrawRaster(draw),
            MockCall::EndRenderPass,
        ] => {
            assert_eq!(
                created, begun,
                "the pass opened the framebuffer that was derived for its attachment"
            );
            assert_eq!(
                linked, selected,
                "the program was linked, then made current"
            );
            assert_eq!(
                linked, program,
                "and the pipeline installed is that program"
            );
            assert_eq!(
                derived, bound,
                "the array derived for the recipe is the one that was bound"
            );
            assert_eq!(
                derived, vertex_array,
                "and the array the pipeline names, which is why the derivation precedes the install"
            );
            assert_eq!(
                derived, rebound,
                "and the one re-enabled after the install took the claim: a pipeline bind does not re-emit the attribute description"
            );
            assert_eq!(
                draw,
                &GlDrawCommand::NonIndexed(GlNonIndexedDraw {
                    first_vertex: 0,
                    vertex_count: 3,
                    instance_count: 1,
                }),
                "the draw is the command the frame asked for, lowered without scale"
            );
        }
        other => panic!("the pass issued a different sequence: {other:?}"),
    }
}

#[test]
fn a_second_pass_reuses_the_framebuffer_it_derived_and_re_installs_the_pipeline() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    let (pipeline, bindings) = registered(&mut adapter, RasterKernel::Triangle, &[]);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };

    // The first pass, drawn, so that the second one starts from a machine that
    // has already derived everything this attachment and this recipe need.
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
    adapter.draw(&mut encoder, 0..3, 0..1).expect("a draw");
    adapter.end_raster(&mut encoder).expect("the pass closes");
    trace_from_here(&mut adapter);

    // The same pass again, on the same encoder.  What is asserted is what the
    // second pass *costs*: the caches that retain (the session's framebuffers and
    // the pipeline domain's programs) answer from their records rather than
    // deriving again, while the installed pipeline is re-issued because a pass
    // end is a boundary the driver forgets -- Layer 1's `end_render_pass` drops
    // the pipeline it had, so a mirror that kept believing one was installed
    // would skip the install this draw needs.
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("the same pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("the same set");
    adapter
        .draw(&mut encoder, 0..3, 0..1)
        .expect("a draw again");
    adapter.end_raster(&mut encoder).expect("the pass closes");

    let trace = calls(&mut adapter);
    assert!(
        !trace.iter().any(derives_anything),
        "a second pass over a retained framebuffer, program and array derives nothing: {trace:?}"
    );
    assert!(
        !trace.iter().any(destroys),
        "and frees nothing, because this machine's caches own everything it derived: {trace:?}"
    );
    assert_eq!(
        count(&mut adapter, installs),
        1,
        "the pipeline is installed once, because the pass before it ended: {trace:?}"
    );
    assert!(
        matches!(
            trace.as_slice(),
            [
                MockCall::BeginRenderPass(_),
                ..,
                MockCall::SetRasterPipeline { .. },
                MockCall::BindVertexArray(_),
                MockCall::DrawRaster(_),
                MockCall::EndRenderPass
            ]
        ),
        "and the pass runs in the same order, from a framebuffer that was not re-derived: {trace:?}"
    );
}

#[test]
fn a_pass_boundary_the_frame_did_not_cross_is_refused_in_both_directions() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };

    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    assert_eq!(
        invalid(adapter.begin_raster(&mut encoder, &pass)),
        "begin-raster",
        "a second pass while one is open: this context runs one at a time, and a caller that closed the first could open this one"
    );

    adapter.end_raster(&mut encoder).expect("the pass closes");
    assert_eq!(
        invalid(adapter.end_raster(&mut encoder)),
        "end-raster",
        "and the other direction, which a caller that opened a pass could not reach"
    );

    // A command buffer cannot be finished inside a pass.  The executor brackets
    // every pass it opens, so an open one here is a frame that abandoned it --
    // and the refusal is reported *and* the pass unwound, which the next frame is
    // what proves: a provider whose pass is still open refuses to begin another
    // one, so leaving it would turn one abandoned pass into a context nothing can
    // render on again.
    adapter
        .begin_raster(&mut encoder, &pass)
        .expect("a pass again");
    assert_eq!(invalid(adapter.finish_encoder(encoder)), "finish-encoder");
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    adapter
        .begin_raster(&mut encoder, &pass)
        .expect("the abandoned pass was closed, so the next frame can open its own");
    adapter.end_raster(&mut encoder).expect("the pass closes");
}

#[test]
fn an_attachment_set_no_artifact_can_run_in_is_refused_before_a_framebuffer_exists() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    let other = colour_target(&mut adapter);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");

    // Five passes, one per attachment fact this family's pass vocabulary has no
    // case for.  Each is `refused` and not `invalid`: none of them is a request a
    // caller could fix by naming a different object -- the closed artifact set
    // writes one colour target at index zero and declares no depth state, so a
    // pass with a different set has no pipeline that could run in it.
    //
    // The empty set is the sixth and is covered where the verb itself lives
    // (`super`'s raster-verb test), so it is not repeated here.
    let two_colours = [
        cleared(&target.physical),
        attachment(1, &other.physical, LoadOp::Load, StoreOp::Store),
    ];
    let shifted = [attachment(
        1,
        &target.physical,
        LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
        StoreOp::Store,
    )];
    let plain = [cleared(&target.physical)];
    let dont_care = [attachment(
        0,
        &target.physical,
        LoadOp::DontCare,
        StoreOp::Store,
    )];
    let discard = [attachment(
        0,
        &target.physical,
        LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
        StoreOp::Discard,
    )];
    let with_depth = [cleared(&target.physical)];
    let depth_stencil = depth(&other.physical);

    let cases = [
        (
            "a second colour attachment",
            RasterPassDescriptor {
                label: "raster",
                colors: &two_colours,
                depth_stencil: None,
            },
        ),
        (
            "a colour attachment not at index zero",
            RasterPassDescriptor {
                label: "raster",
                colors: &shifted,
                depth_stencil: None,
            },
        ),
        (
            "a depth-stencil attachment",
            RasterPassDescriptor {
                label: "raster",
                colors: &with_depth,
                depth_stencil: Some(depth_stencil),
            },
        ),
        (
            "a don't-care load",
            RasterPassDescriptor {
                label: "raster",
                colors: &dont_care,
                depth_stencil: None,
            },
        ),
        (
            "a discarded store",
            RasterPassDescriptor {
                label: "raster",
                colors: &discard,
                depth_stencil: None,
            },
        ),
    ];
    for (case, descriptor) in &cases {
        assert_eq!(
            refused(adapter.begin_raster(&mut encoder, descriptor)),
            "begin-raster",
            "{case}"
        );
    }
    // The control: the shape every case varied from does open a pass, so the five
    // refusals above are about what each case changed and not about this encoder
    // refusing everything.
    let admitted = RasterPassDescriptor {
        label: "raster",
        colors: &plain,
        depth_stencil: None,
    };
    adapter
        .begin_raster(&mut encoder, &admitted)
        .expect("the one attachment set this family runs");
    adapter.end_raster(&mut encoder).expect("the pass closes");
}

#[test]
fn an_attachment_this_device_no_longer_holds_is_refused_rather_than_attached() {
    let mut adapter = adapter();
    let released = colour_target(&mut adapter);
    let id = released.physical;
    drop(released);
    // The release is drained at the adapter's next entry point, which is where
    // the attachment record goes with the object.  So the identity below names a
    // texture this device created and no longer holds -- the case a caller
    // reaches by keeping an id past its lease.
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    trace_from_here(&mut adapter);

    let colours = [cleared(&id)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    assert_eq!(
        invalid(adapter.begin_raster(&mut encoder, &pass)),
        "begin-raster",
        "a request naming an object the device does not hold: a caller holding its lease could make the same call succeed"
    );
    assert!(
        !calls(&mut adapter).iter().any(derives_anything),
        "and the refusal precedes the derivation, so no framebuffer was created for it: {:?}",
        calls(&mut adapter)
    );
}

#[test]
fn a_draw_is_refused_when_it_is_not_the_verb_its_artifact_is_drawn_with() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    trace_from_here(&mut adapter);

    // The bare triangle is not indexed.  Every refusal below is made before the
    // commit, which is the point of making them here at all: a commit installs a
    // program, an array and a pipeline, so a draw refused after one would leave
    // the driver holding state for a command that never happened.
    let (pipeline, _) = registered(&mut adapter, RasterKernel::Triangle, &[]);
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    assert_eq!(
        invalid(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..1)),
        "draw-indexed",
        "the artifact draws its vertices without an index buffer"
    );
    assert_eq!(
        refused(adapter.draw(&mut encoder, 0..3, 0..2)),
        "draw",
        "and its instanced form is an optional verb this adapter has no lowering for, which no caller could fix by changing the request"
    );
    assert_eq!(
        invalid(adapter.draw(&mut encoder, 0..3, 0..0)),
        "draw",
        "a draw that runs no instance is refused by the verb rather than by the provider, whose name for it the frame never issued"
    );
    assert_eq!(
        invalid(adapter.draw(&mut encoder, 0..0, 0..1)),
        "draw",
        "a draw with no vertices has nothing to rasterize"
    );

    // The mirror image, on an artifact that is indexed.
    let (pipeline, _) = registered(&mut adapter, RasterKernel::IndexedPositionFloat32x3, &[]);
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    assert_eq!(
        invalid(adapter.draw(&mut encoder, 0..3, 0..1)),
        "draw",
        "the artifact takes its vertices from an index buffer, so it is drawn with draw-indexed"
    );
    assert_eq!(
        refused(adapter.draw_indexed(&mut encoder, 0..3, 7, 0..1)),
        "draw-indexed",
        "a base vertex is an optional verb here: the family adds it inside the shader pipeline, so the offset would be a draw the caller asked for and the driver never made"
    );
    assert_eq!(
        invalid(adapter.draw_indexed(&mut encoder, 0..0, 0, 0..1)),
        "draw-indexed",
        "an indexed draw with no indices has nothing to rasterize"
    );
    assert_eq!(
        refused(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..2)),
        "draw-indexed",
        "and its instanced form is refused by the same rule"
    );
    assert!(
        !calls(&mut adapter).iter().any(derives_anything),
        "and nothing was installed for any of them: {:?}",
        calls(&mut adapter)
    );
}

#[test]
fn a_binding_set_resolved_for_another_artifact_is_refused_at_the_bind_and_again_at_the_draw() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");

    // Two indexed artifacts, so the drawing verb is the same one for both and the
    // only thing that can disagree is the recipe.
    let other = RasterKernel::IndexedPositionFloat32x3;
    let (installed, agreeing) = registered(&mut adapter, other, &[]);
    let (mismatched, mismatched_set) =
        registered(&mut adapter, RasterKernel::IndexedPositionColor, &[]);
    assert_ne!(other, mismatched.physical.kernel(), "the two differ");

    adapter
        .set_raster_pipeline(&mut encoder, &installed.physical)
        .expect("a registered pipeline");
    assert_eq!(
        invalid(adapter.set_bindings(&mut encoder, &mismatched_set.physical)),
        "set-bindings",
        "the set is checked at the call that got it wrong, while the pipeline it disagrees with is the one still installed"
    );

    // And the other order: a set that agrees with the pipeline installed when it
    // was recorded, and a pipeline installed after it.  Only the draw can catch
    // this one, which is why the check exists in both places.
    trace_from_here(&mut adapter);
    adapter
        .set_raster_pipeline(&mut encoder, &installed.physical)
        .expect("the artifact the set below was resolved for");
    adapter
        .set_bindings(&mut encoder, &agreeing.physical)
        .expect("a set for the installed recipe");
    adapter
        .set_raster_pipeline(&mut encoder, &mismatched.physical)
        .expect("a pipeline installed after the set");
    assert_eq!(
        invalid(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..1)),
        "draw-indexed",
        "the set in this pass was resolved for the artifact installed when it was recorded"
    );
    assert!(
        !calls(&mut adapter).iter().any(derives_anything),
        "and the disagreement is reported before anything is installed for the draw: {:?}",
        calls(&mut adapter)
    );
}

#[test]
fn a_textured_artifacts_bindings_arrive_one_at_a_time_in_the_recipes_own_order() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    // The one artifact whose bindings are a uniform block *and* a texture, which
    // is the smallest set that can show the ordering: the texture domain's request
    // is single-valued, so a set applied in one pass would bind only its last
    // unit.
    let kernel = RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture;
    let identity = kernel.portable_identity();
    assert_eq!(
        identity.uniform_binding,
        Some(0),
        "the frame block is at binding zero"
    );
    assert_eq!(
        identity.texture_binding,
        Some(1),
        "and the sampled texture at binding one"
    );

    // The block's size and the sub-range at the end of this test are both derived
    // from the alignment this device discovered rather than written here.  A byte
    // range that is not a multiple of it is refused by the binding validation
    // before any draw, so a literal offset would be asserting this snapshot's
    // alignment number rather than the adapter carrying the authorized range
    // through -- and the offset must also stay inside the allocation.
    let alignment = adapter
        .capabilities()
        .limits
        .min_uniform_buffer_offset_alignment;
    assert!(
        alignment > 0,
        "a device that discovered no alignment refuses every block range, and a legal sub-range is what this test is about"
    );
    let frame = buffer(&mut adapter, alignment * 2, BufferUsageKind::Uniform);
    let texture = sampled_texture(&mut adapter);
    let vertex = buffer(&mut adapter, 64, BufferUsageKind::Vertex);
    let indices = buffer(&mut adapter, 64, BufferUsageKind::Index);
    let format = identity
        .index_format
        .expect("an indexed artifact declares the element format its indices are read as");
    let slot = super::super::super::shader::vertex_layout(kernel).buffers[0].slot;

    let resources = resolved(kernel, &texture.physical, &frame.physical);
    let (pipeline, bindings) = registered(&mut adapter, kernel, &resources);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set resolved for the installed recipe");
    // The slot is the artifact's own, read off its layout rather than written
    // here: a recipe whose vertex input changed would change this with it.
    adapter
        .set_vertex_buffer(&mut encoder, slot, &vertex.physical, 0)
        .expect("a slot the artifact declares");
    adapter
        .set_index_buffer(&mut encoder, &indices.physical, 0, format)
        .expect("the index buffer an indexed artifact is drawn from");
    trace_from_here(&mut adapter);

    adapter
        .draw_indexed(&mut encoder, 0..3, 0, 0..1)
        .expect("a draw of the textured artifact");
    adapter.end_raster(&mut encoder).expect("the pass closes");

    let trace = calls(&mut adapter);
    match trace.as_slice() {
        [
            ..,
            MockCall::BindUniformBuffer {
                index: 0,
                buffer: Some(bound),
                offset: 0,
                size: 0,
            },
            MockCall::BindTexture {
                unit: 1,
                target: GlTextureTarget::D2,
                texture: Some(bound_texture),
            },
            MockCall::DrawRaster(GlDrawCommand::Indexed(draw)),
            MockCall::EndRenderPass,
        ] => {
            assert_eq!(
                *bound, frame.physical,
                "the block bound is the buffer the graph resolved"
            );
            assert_eq!(
                *bound_texture, texture.physical,
                "and the texture bound is the one it resolved, at the recipe's own binding number"
            );
            assert_eq!(draw.first_index, 0);
            assert_eq!(draw.index_count, 3);
            assert_eq!(draw.instance_count, 1);
        }
        other => panic!("the bindings did not arrive one at a time in recipe order: {other:?}"),
    }
    assert_eq!(
        count(&mut adapter, |call| matches!(
            call,
            MockCall::BindUniformBuffer { .. }
        )),
        1,
        "and each logical binding was applied exactly once: a set applied as a whole would have left only its last unit bound"
    );

    // The authorized range is carried through rather than widened to the whole
    // allocation, and a range this family's 32-bit binding point cannot express is
    // reported by the draw instead of truncated.
    for (offset, refusal) in [
        (alignment, None),
        (u64::from(u32::MAX) + 1, Some("an authorized offset beyond")),
    ] {
        let ranged = Bindings::new(
            kernel,
            &[
                ResolvedBindingResource::Buffer {
                    physical: &frame.physical,
                    range: BufferRange::Bytes { offset, size: 16 },
                    semantic: BindingResourceSemantic::BufferRead(BufferReadUse::Uniform),
                },
                sampled(&texture.physical),
            ],
        )
        .expect("the same recipe at the same numbers");
        let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
        adapter.begin_raster(&mut encoder, &pass).expect("a pass");
        adapter
            .set_raster_pipeline(&mut encoder, &pipeline.physical)
            .expect("a registered pipeline");
        adapter
            .set_bindings(&mut encoder, &ranged)
            .expect("a set for the installed recipe");
        adapter
            .set_index_buffer(&mut encoder, &indices.physical, 0, format)
            .expect("the element format the artifact declares");
        adapter
            .set_vertex_buffer(&mut encoder, slot, &vertex.physical, 0)
            .expect("the slot the artifact declares");
        trace_from_here(&mut adapter);
        match refusal {
            Some(what) => assert_eq!(
                invalid(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..1)),
                "draw-indexed",
                "{what} 32-bit binding points can address is reported, not truncated into a range the graph never authorized"
            ),
            None => {
                adapter
                    .draw_indexed(&mut encoder, 0..3, 0, 0..1)
                    .expect("a draw with an authorized sub-range");
                // Exactly one block binding, and it names the range the graph
                // authorized rather than the whole allocation.  That the block
                // arrives before the texture, one binding at a time, is the first
                // half of this test's subject and is pinned there; what is asked
                // here is only what the block binding was given.
                let bound: Vec<(u32, u32)> = calls(&mut adapter)
                    .into_iter()
                    .filter_map(|call| match call {
                        MockCall::BindUniformBuffer {
                            index: 0,
                            offset,
                            size,
                            ..
                        } => Some((offset, size)),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    bound,
                    [(
                        u32::try_from(offset)
                            .expect("an offset the graph authorized as a 32-bit binding point"),
                        16
                    )],
                    "the authorized byte range is the one bound, not the whole allocation"
                );
            }
        }
        adapter.end_raster(&mut encoder).expect("the pass closes");
    }
}

#[test]
fn a_sampled_binding_that_names_subresources_has_no_lowering() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    let kernel = RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture;
    let texture = sampled_texture(&mut adapter);
    let frame = buffer(&mut adapter, 256, BufferUsageKind::Uniform);
    let vertex = buffer(&mut adapter, 64, BufferUsageKind::Vertex);
    let indices = buffer(&mut adapter, 64, BufferUsageKind::Index);
    let format = kernel
        .portable_identity()
        .index_format
        .expect("an indexed artifact declares its index format");
    let slot = super::super::super::shader::vertex_layout(kernel).buffers[0].slot;

    // The uniform is authorized whole and the texture for a mip range: this family
    // has no texture view, so the level a sampled binding names would be dropped
    // silently and the shader would read the wrong one.
    let resources = [
        uniform(&frame.physical),
        ResolvedBindingResource::Texture {
            physical: &texture.physical,
            range: TextureRange::Subresources {
                base_mip_level: 0,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: TextureAspect::All,
            },
            semantic: BindingResourceSemantic::TextureRead(TextureReadUse::Sampled),
        },
    ];
    let (pipeline, bindings) = registered(&mut adapter, kernel, &resources);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set resolved for the installed recipe");
    // Everything the draw checks before the bindings, so that what refuses below
    // is the range and not an earlier gap in the pass: the geometry and the recipe
    // are in order, and only the sampled binding's selected subresources are not.
    adapter
        .set_vertex_buffer(&mut encoder, slot, &vertex.physical, 0)
        .expect("a slot the artifact declares");
    adapter
        .set_index_buffer(&mut encoder, &indices.physical, 0, format)
        .expect("the index buffer an indexed artifact is drawn from");
    assert_eq!(
        refused(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..1)),
        "draw-indexed",
        "no caller could make this call succeed by naming another object: the family has no texture view at all"
    );
}

#[test]
fn a_vertex_slot_the_recipe_and_the_frame_disagree_about_is_refused_at_the_draw() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    let kernel = RasterKernel::IndexedPositionFloat32x3;
    let declared = super::super::super::shader::vertex_layout(kernel).buffers[0].slot;
    let indices = buffer(&mut adapter, 64, BufferUsageKind::Index);
    let vertex = buffer(&mut adapter, 64, BufferUsageKind::Vertex);
    let format = kernel
        .portable_identity()
        .index_format
        .expect("an indexed artifact declares its index format");
    let (pipeline, bindings) = registered(&mut adapter, kernel, &[]);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
    adapter
        .set_index_buffer(&mut encoder, &indices.physical, 0, format)
        .expect("the element format the artifact declares");
    adapter
        .set_viewport(&mut encoder, whole_attachment())
        .expect("a viewport over the attachment");
    trace_from_here(&mut adapter);

    // A slot the artifact reads and the frame never set: refused rather than left
    // reading the array's default, which a driver would accept and a caller would
    // see as a misshapen mesh.
    assert_eq!(
        invalid(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..1)),
        "draw-indexed",
        "the artifact reads a slot the pass never bound, and the refusal names the verb the frame called"
    );

    // The record is complete, so the draw runs.  This is the control that keeps
    // the two refusals around it from being a pass that refuses everything.
    adapter
        .set_vertex_buffer(&mut encoder, declared, &vertex.physical, 0)
        .expect("the slot the artifact declares");
    adapter
        .draw_indexed(&mut encoder, 0..3, 0, 0..1)
        .expect("every slot the artifact reads is bound");
    assert_eq!(
        count(&mut adapter, |call| matches!(
            call,
            MockCall::DrawRaster(GlDrawCommand::Indexed(_))
        )),
        1,
        "and it reached the driver"
    );

    // The other direction: a slot the frame bound and the artifact does not
    // declare.  Refused for the same reason, from the other side -- the recipe
    // decides what may be bound, not the frame.  It is checked here, after the
    // successful draw, because a pass's record only ever grows: a slot set once
    // stays set, so this is the last thing this pass can assert.
    adapter
        .set_vertex_buffer(&mut encoder, declared + 1, &vertex.physical, 0)
        .expect("a slot, recorded before anything checks it against the recipe");
    assert_eq!(
        invalid(adapter.draw_indexed(&mut encoder, 0..3, 0, 0..1)),
        "draw-indexed",
        "the frame bound a vertex slot the artifact does not declare"
    );
    assert_eq!(
        count(&mut adapter, |call| matches!(
            call,
            MockCall::DrawRaster(GlDrawCommand::Indexed(_))
        )),
        1,
        "and the refused draw never reached the driver"
    );
}

#[test]
fn a_context_replaced_during_a_pass_closes_nothing_and_the_frame_records_again() {
    let mut adapter = adapter();
    let target = colour_target(&mut adapter);
    let (pipeline, bindings) = registered(&mut adapter, RasterKernel::Triangle, &[]);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let colours = [cleared(&target.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
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

    // The pass belonged to the epoch the context replaced, and the restored
    // context released it before it reported its new stamp -- so there is no
    // boundary left for this verb to cross, and the close reports one.  What makes
    // that correct rather than convenient is the two assertions after it: the
    // frame's next verb still refuses, and nothing of the dead generation was
    // destroyed.
    adapter
        .end_raster(&mut encoder)
        .expect("a pass the restored context already released is not a boundary this verb can fail to close");
    assert!(
        !calls(&mut adapter)
            .iter()
            .any(|call| matches!(call, MockCall::EndRenderPass)),
        "and it never reached the driver: a pass of the superseded generation has no end to emit: {:?}",
        calls(&mut adapter)
    );
    assert_eq!(
        count(&mut adapter, destroys),
        0,
        "nothing of that generation is destroyed through an identity the backend no longer accepts: {:?}",
        calls(&mut adapter)
    );

    // The loss is not swallowed, it is reported where the frame can act on it:
    // the command buffer carries the generation it was recorded against, and
    // submission is the verb that compares it.
    let buffer = adapter
        .finish_encoder(encoder)
        .expect("the pass is closed, so the encoder finishes");
    assert!(
        adapter.submit(QueueId::new(0), buffer, Vec::new()).is_err(),
        "a frame that recorded across a context replacement cannot submit what it recorded"
    );

    // The frame can render again, over objects the restored context minted.  The
    // attachment is a new texture because that is the only kind the restored
    // context has: the previous one's identities were invalidated with its
    // objects, and the adapter forgot their shapes along with them.
    let fresh = colour_target(&mut adapter);
    let colours = [cleared(&fresh.physical)];
    let pass = RasterPassDescriptor {
        label: "raster",
        colors: &colours,
        depth_stencil: None,
    };
    let (pipeline, bindings) = registered(&mut adapter, RasterKernel::Triangle, &[]);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    adapter.begin_raster(&mut encoder, &pass).expect("a pass");
    adapter
        .set_raster_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
    adapter
        .set_viewport(&mut encoder, whole_attachment())
        .expect("a viewport over the restored attachment");
    adapter.draw(&mut encoder, 0..3, 0..1).expect("a draw");
    adapter.end_raster(&mut encoder).expect("the pass closes");
    assert!(
        calls(&mut adapter)
            .iter()
            .any(|call| matches!(call, MockCall::DrawRaster(_))),
        "the restored context rasterized the frame's draw"
    );
}
