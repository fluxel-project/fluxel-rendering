//! One whole frame, driven by the executor rather than by the verbs.
//!
//! The sibling suites call the recording verbs directly, which pins what each
//! verb lowers to but not that a *frame* can drive them.  A frame is a caller
//! contract -- a compiled graph, a resource provider, an object provider, and a
//! pass closure that resolves its bindings before it draws -- and this suite is
//! that caller.  What it checks is the closed loop: the capability check, the
//! transient the graph created, the transition the executor handed the backend,
//! the pass, the draw, the submission, and the export that came back.  The
//! transition is worth naming because it leaves *no* trace -- this adapter issues
//! nothing for one -- so the only evidence that it happened is that the frame
//! completed, which is exactly the check the verb suites cannot make.
//!
//! It is also where the peer check F1 owed F2 finally lands in executable form.
//! That check was "the consumer that names the adapter as `B: ExecutionBackend`
//! accepts it", and it lived in the harness as a function whose only content was
//! `FrameExecutor::new(adapter)` -- a compile-time check that nothing ran, kept
//! alive by an `allow(dead_code)`.  This suite builds the same executor and then
//! drives it, so the placeholder is gone and the check it stood for is now made
//! by a test that would fail if the associated types stopped fitting.
//!
//! # Why the frames here carry no imports
//!
//! A provider is consulted for *imports* only: a graph's own textures and
//! buffers arrive through `create_transient_texture` / `create_transient_buffer`
//! on the backend, and the executor never asks a provider for one.  So a frame
//! that imports nothing still exercises the whole adapter boundary, and it is
//! the only frame this slice can build honestly -- a caller-owned object has no
//! producer on this backend yet.  The adapter's two creation verbs are both
//! transient ones, and the retained path that would hand a renderer a lease over
//! its own buffer is the migration F6 owns.  [`NoImports`] therefore answers
//! every request with the contract's own "not supplied" error rather than
//! standing in as a provider that pretends to have one.
//!
//! That gap is wider than it looks, and it is worth stating where it lands,
//! because it is the reason this suite has one frame and not two.  A frame that
//! *binds geometry* -- the vertex-input path, which the single frame below
//! cannot reach -- needs a vertex buffer, and on this backend the two ways to
//! get one are both closed:
//!
//! - an **import** needs a caller-owned buffer, which no verb of this adapter
//!   creates; and
//! - a **graph-created** buffer is refused at compile time before it can be
//!   read: `create_buffer` leaves the resource uninitialized, and the
//!   compiler's own initialization pass rejects a read with
//!   `CompileErrorKind::ReadBeforeInitialization`.  Seeding it in-frame takes a
//!   copy pass, whose source is -- an import.
//!
//! So the imported half of the frame contract is not untested because it was
//! skipped; it is unreachable until something can produce a caller-owned object.
//! What this suite covers is the half that is reachable end to end.

use fluxel_rendergraph::{
    AttachmentOps, BindingSetId, BoundBuffer, BoundTexture, BufferBindingId, ColorAttachmentDesc,
    ExecutionBackend, ExportTextureContract, FrameBindingError, FrameBindingErrorKind,
    FrameExecutor, FrameInputs, FrameResourceProvider, LoadOp, RasterPipelineId, RenderGraph,
    ResourceAccessState, StoreOp, TextureBindingId, TextureRange, WriteCoverage,
};

use super::super::retention::GlRetentionLease;
use super::{Adapter, adapter, plain_texture};
use crate::resource::RasterKernel;
use crate::webgl2::api::{BufferId, GlDrawCommand, GlNonIndexedDraw, MockCall, TextureId};

/// The frame resources this suite serves: none.
///
/// Not a stub, and the module documentation says why: no frame this slice can
/// build imports anything, so the provider is reached only if a test asks the
/// executor for a resource the graph was supposed to create.  Answering that
/// with the contract's structured error is the honest response, and it is also
/// what makes the frame below evidence that the executor resolved the graph's
/// own transients rather than taking a shortcut through this value.
struct NoImports;

impl FrameResourceProvider<Adapter> for NoImports {
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<TextureId, GlRetentionLease>, FrameBindingError> {
        Err(missing(
            FrameBindingErrorKind::MissingTexture,
            format!("this backend has no caller-owned texture to resolve for {id:?}"),
        ))
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<BufferId, GlRetentionLease>, FrameBindingError> {
        Err(missing(
            FrameBindingErrorKind::MissingBuffer,
            format!("this backend has no caller-owned buffer to resolve for {id:?}"),
        ))
    }
}

/// The refusal [`NoImports`] answers with.
fn missing(kind: FrameBindingErrorKind, detail: String) -> FrameBindingError {
    FrameBindingError {
        kind,
        texture_slot: None,
        buffer_slot: None,
        resource: None,
        surface_binding: None,
        detail,
    }
}

/// The colour attachment this frame renders into: index zero, cleared, stored.
///
/// The same four fields the sibling verb suite states through its own helper;
/// this suite cannot reuse that one because it borrows a `TextureId` while a
/// graph pass declares a version, and the two are different types on purpose.
fn cleared() -> ColorAttachmentDesc {
    ColorAttachmentDesc {
        index: 0,
        range: TextureRange::Whole,
        operations: AttachmentOps {
            load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Full,
        },
    }
}

/// The calls the adapter made for one frame, in order.
fn trace(executor: &FrameExecutor<Adapter>) -> Vec<MockCall> {
    let mut backend = executor
        .try_backend()
        .expect("no frame is executing once `execute` has returned");
    backend.machine.backend().calls().to_vec()
}

// ---------------------------------------------------------------------------
// The smallest whole frame.
// ---------------------------------------------------------------------------

#[test]
fn a_whole_frame_of_this_adapter_runs_to_the_calls_the_verb_suite_pins() {
    let mut graph = RenderGraph::new();
    let colour = graph.create_texture("colour", plain_texture());
    let raster = graph.add_raster_pass(
        "triangle",
        |pass| {
            let colour = pass.color_attachment(colour, cleared());
            (colour, ())
        },
        |commands, resolver, (), _frame| {
            commands.set_pipeline(RasterPipelineId::new(1))?;
            // The adapter refuses a draw in a pass that recorded no binding set,
            // and this artifact reads nothing -- so the set it applies is the
            // empty one, resolved through the frame's own resolver rather than
            // conjured to satisfy the refusal.
            let bindings = resolver.resolve_bindings(BindingSetId::new(0), &[], &[])?;
            commands.set_bindings(&bindings)?;
            commands.draw(0..3, 0..1)?;
            Ok(())
        },
    );
    let slot = graph.export_texture(
        raster.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ColorAttachmentWrite,
        },
    );

    let mut device = adapter();
    let compiled = graph
        .compile(device.capabilities())
        .expect("the graph asks this device only for what its own capabilities prove")
        .graph;
    let mut objects = device.object_registry();
    objects
        .register_raster_pipeline(RasterPipelineId::new(1), RasterKernel::Triangle)
        .expect("WebGL2 writes a dialect for the triangle artifact");
    objects.register_bindings(BindingSetId::new(0), RasterKernel::Triangle);

    let executor = FrameExecutor::new(device);
    let frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &NoImports,
            &objects,
        )
        .expect("every resource this frame names is one the graph creates");

    let exported = frame
        .exports
        .texture(slot)
        .expect("the graph's one export came back");
    assert_eq!(
        frame.exports.textures().count(),
        1,
        "and it is the frame's only one"
    );
    assert_eq!(
        exported.descriptor,
        plain_texture(),
        "the descriptor handed back is the graph's, not the lowered form this backend created"
    );
    match trace(&executor).as_slice() {
        [
            MockCall::CreateTexture(created),
            MockCall::CreateFramebuffer(derived),
            MockCall::BeginRenderPass(begun),
            MockCall::CreateProgram(linked),
            MockCall::CreateVertexArray(array),
            MockCall::BindVertexArray(bound),
            MockCall::SelectProgram(selected),
            MockCall::SetRasterPipeline {
                program,
                vertex_array,
            },
            MockCall::BindVertexArray(rebound),
            MockCall::DrawRaster(draw),
            MockCall::EndRenderPass,
            MockCall::Flush,
            MockCall::CreateFence(_),
        ] => {
            assert_eq!(
                *created, exported.physical,
                "the transient the frame created is the texture it exported"
            );
            assert_eq!(
                derived, begun,
                "the framebuffer derived for that attachment is the one the boundary opened"
            );
            assert_eq!(
                linked, selected,
                "the program was linked, then made current"
            );
            assert_eq!(linked, program, "and it is the one the pipeline installs");
            assert_eq!(array, bound, "the derived array is the one bound");
            assert_eq!(
                array, vertex_array,
                "and the one the pipeline names, which is why the derivation precedes the install"
            );
            assert_eq!(
                array, rebound,
                "and the one re-enabled after the install took the claim"
            );
            assert_eq!(
                draw,
                &GlDrawCommand::NonIndexed(GlNonIndexedDraw {
                    first_vertex: 0,
                    vertex_count: 3,
                    instance_count: 1,
                }),
                "the draw is the one the frame asked for, and it is the last thing the pass did"
            );
        }
        other => panic!("the frame issued a different sequence: {other:?}"),
    }
}
