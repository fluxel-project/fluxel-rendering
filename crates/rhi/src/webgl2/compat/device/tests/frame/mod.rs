//! Whole frames, driven by the executor rather than by the verbs.
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
//! # One file per scenario, and why
//!
//! The subject is one -- a whole frame -- but a frame is also the cheapest thing
//! in this crate to vary, because every axis of the frame contract (imports,
//! bindings, attachments) is a place the adapter can be wrong in a way no verb
//! suite would notice.  This file holds the shared fixtures and the frame that
//! consults no provider at all; each further scenario gets its own submodule
//! beside it rather than a section appended here, so that a file keeps one
//! sentence of responsibility as the scenarios accumulate.
//!
//! This module owns the fixtures more than one scenario needs: the refusal
//! helper, the colour attachment, the trace reader, and the provider that imports
//! nothing.  A submodule reaches them through `super`, which is also why they
//! carry no visibility annotation -- a child module sees its ancestors' private
//! items, and nothing outside this suite has any business calling them.
//!
//! # The finding this suite was built around, and its correction
//!
//! The first frame below carries no imports, and for a while it was the only
//! frame this suite could build.  The reason was recorded here as a finding
//! rather than a skip.  A provider is consulted for *imports* only: a graph's own
//! textures and buffers arrive through `create_transient_texture` /
//! `create_transient_buffer` on the backend, and the executor never asks a
//! provider for one.  So a frame that imports nothing still exercises the whole
//! adapter boundary, while a frame that *binds geometry* needs a caller-owned
//! buffer, and the finding said there were two ways to want one and both were
//! closed:
//!
//! - an **import** needs a caller-owned buffer, which no verb of this adapter
//!   creates; and
//! - a **graph-created** buffer is refused at compile time before it can be
//!   read: `create_buffer` leaves the resource uninitialized, and the
//!   compiler's own initialization pass rejects a read with
//!   `CompileErrorKind::ReadBeforeInitialization`.  Seeding it in-frame takes a
//!   copy pass, whose source is -- an import.
//!
//! The first clause was true of the *frame path's* vocabulary and false of the
//! backend.  `create_transient_buffer` is a public contract verb, callable
//! outside a frame, and it returns exactly the `BoundBuffer` a provider must
//! return.  What was genuinely missing was a way to put host bytes into the
//! buffer it made, because no verb of the execution contract writes any -- every
//! resource verb there is about *existence*.  That is
//! [`GlCompatibilityDevice::upload_buffer`](super::super::GlCompatibilityDevice),
//! and [`imported`] is the imported half made reachable.

use fluxel_rendergraph::{
    AttachmentOps, BindingSetId, BoundBuffer, BoundTexture, BufferBindingId, BufferDesc,
    BufferUsage, BufferUsageKind, ColorAttachmentDesc, ExecutionBackend, ExportTextureContract,
    ExternalOwnership, FrameBindingError, FrameBindingErrorKind, FrameExecutor, FrameInputs,
    FrameResourceProvider, ImportBufferContract, ImportedBuffer, InitialContents, LoadOp,
    RasterPipelineId, RenderGraph, ResourceAccessState, StoreOp, TextureBindingId, TextureRange,
    WriteCoverage,
};

use super::super::object::Recipe;
use super::super::retention::GlRetentionLease;
use super::{Adapter, adapter, plain_texture};
use crate::resource::RasterKernel;
use crate::webgl2::api::{BufferId, GlDrawCommand, GlNonIndexedDraw, MockCall, TextureId};

mod imported;
mod uniform;

/// The frame resources the first frame serves: none.
///
/// Not a stub, and the module documentation says why: that frame imports
/// nothing, so this provider is reached only if a test asks the executor for a
/// resource the graph was supposed to create.  Answering that with the contract's
/// structured error is the honest response, and it is also what makes that frame
/// evidence that the executor resolved the graph's own transients rather than
/// taking a shortcut through a provider that would have supplied them.
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

/// The refusal a provider in this suite answers with.
///
/// Shared by every provider here, because the shape of the refusal is not the
/// interesting part of any of them: what a scenario varies is *which* bindings it
/// can answer, and a provider that can answer none and one that can answer two of
/// five differ only in that.
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

/// The colour attachment these frames render into: index zero, cleared, stored.
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
// Import plumbing, shared by every frame that draws from a caller-owned object.
// ---------------------------------------------------------------------------

/// The three positions the frames here draw, built from the floats they stand
/// for.
///
/// Built rather than written out as a byte literal because the numbers are the
/// only part of this a reader can check: nothing here has a driver to read the
/// bytes back, so what the array has to be is *a* defined vertex stream of the
/// shape the artifact declares -- three `Float32x3` positions at a stride of
/// twelve, which is what the indexed position artifacts record.  A wall of hex
/// would hide exactly that.
fn position_stream() -> Vec<u8> {
    const POSITIONS: [[f32; 3]; 3] = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
    POSITIONS
        .iter()
        .flatten()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

/// The same three vertices as one triangle of `u32` indices.
///
/// The element stream is here because the vertex-input slice is the whole
/// vertex-array state and not just its attribute half: every artifact in this
/// family's table that reads a vertex stream is an indexed one, so a frame that
/// binds attributes and cannot bind elements is a frame no recipe accepts.
fn index_stream() -> Vec<u8> {
    const INDICES: [u32; 3] = [0, 1, 2];
    INDICES
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect()
}

/// One imported buffer slot, declared the way every import in this suite is.
///
/// `Undefined` and not a domain state, because it is what the adapter's creation
/// verb answers with and the executor requires the two to agree exactly.  It is
/// also the honest state: the bytes are the caller's, and the buffer has no prior
/// access on this context to name.
fn imported(graph: &mut RenderGraph, name: &str, size: u64) -> ImportedBuffer {
    graph.import_buffer_slot(
        name,
        ImportBufferContract {
            descriptor: BufferDesc { size },
            initial_state: ResourceAccessState::Undefined,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    )
}

/// The same resolved buffer, restated so a provider can hand it out.
///
/// The lease is cloned rather than moved, and the clone shares the one
/// `LeaseInner`: the contract lets a provider answer the same object more than
/// once within a frame, so the retained objects are released when the last holder
/// drops rather than when this one answers.
fn reissue(
    buffer: &BoundBuffer<BufferId, GlRetentionLease>,
) -> BoundBuffer<BufferId, GlRetentionLease> {
    BoundBuffer {
        device: buffer.device,
        identity: buffer.identity,
        physical: buffer.physical,
        descriptor: buffer.descriptor,
        usage: buffer.usage,
        initial_state: buffer.initial_state,
        lease: buffer.lease.clone(),
    }
}

/// Creates one caller-owned buffer and fills it, before any frame exists.
///
/// The two steps are the whole claim every frame here rests on: neither verb
/// needs a frame, an encoder or a provider, so an object a frame will import can
/// be made and backed with nothing but the adapter.
fn own(
    device: &mut Adapter,
    size: u64,
    usage: BufferUsageKind,
    bytes: &[u8],
) -> BoundBuffer<BufferId, GlRetentionLease> {
    let buffer = device
        .create_transient_buffer(BufferDesc { size }, BufferUsage::from_kinds([usage]))
        .expect("a caller-owned buffer");
    device
        .upload_buffer(buffer.physical, 0, bytes)
        .expect("the bytes reach the buffer this device created");
    buffer
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
    objects.register_bindings(BindingSetId::new(0), Recipe::Raster(RasterKernel::Triangle));

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
