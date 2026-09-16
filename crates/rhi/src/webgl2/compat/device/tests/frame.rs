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
//! # Two frames: the half that was reachable, and the half that was not
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
//! and the second frame below is the imported half made reachable: the two
//! caller-owned buffers a vertex-array state needs, created by this device before
//! the frame begins, uploaded with one triangle's positions and its elements,
//! handed back through a provider, and drawn indexed from.
//!
//! The pair is the evidence, and each is worth what the other is not.  The first
//! frame proves a frame runs while consulting no provider at all -- [`NoImports`]
//! answers every request with the contract's own "not supplied" error, so
//! reaching it would fail the frame.  The second proves the provider is the whole
//! mechanism for a caller-owned object, and that nothing on the vertex-input path
//! had to change to admit one: `RasterCommandSink::set_vertex_buffer` takes a
//! graph handle, and `raster.rs`'s `buffer_physical` resolves that handle against
//! whichever physical the provider returned.

use fluxel_rendergraph::{
    AttachmentOps, BindingSetId, BoundBuffer, BoundTexture, BufferBindingId, BufferDesc,
    BufferRange, BufferReadUse, BufferUsage, BufferUsageKind, ColorAttachmentDesc,
    ExecutionBackend, ExportTextureContract, ExternalOwnership, FrameBindingError,
    FrameBindingErrorKind, FrameExecutor, FrameInputs, FrameResourceProvider, ImportBufferContract,
    ImportedBuffer, IndexFormat, InitialContents, LoadOp, RasterPipelineId, RenderGraph,
    ResourceAccessState, StoreOp, TextureBindingId, TextureRange, WriteCoverage,
};

use super::super::object::Recipe;
use super::super::retention::GlRetentionLease;
use super::{Adapter, adapter, plain_texture};
use crate::resource::RasterKernel;
use crate::webgl2::api::{
    BufferId, GlDrawCommand, GlIndexedDraw, GlNonIndexedDraw, MockCall, TextureId,
};

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

// ---------------------------------------------------------------------------
// The half that needed a caller-owned object.
// ---------------------------------------------------------------------------

/// The three positions the second frame draws, built from the floats they stand
/// for.
///
/// Built rather than written out as a byte literal because the numbers are the
/// only part of this a reader can check: nothing here has a driver to read the
/// bytes back, so what the array has to be is *a* defined vertex stream of the
/// shape the artifact declares -- three `Float32x3` positions at a stride of
/// twelve, which is what `IndexedPositionFloat32x3` records.  A wall of hex would
/// hide exactly that.
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

/// A provider that answers two bindings with objects *this device* made.
///
/// It holds the `BoundBuffer`s the adapter's own creation verb returned, and that
/// is the whole of what a provider for an import has to be: the objects are
/// created outside any frame, and answering with them is what makes the graph's
/// imported slots resolve.  Two rather than one, because the answer has to be
/// *per binding* -- a provider that returned whatever it happened to hold would
/// pass a one-import frame and be wrong the moment two imports differ in usage,
/// which is exactly the pair here.
struct UploadedBuffers {
    vertices: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
    indices: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
}

impl FrameResourceProvider<Adapter> for UploadedBuffers {
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<TextureId, GlRetentionLease>, FrameBindingError> {
        Err(missing(
            FrameBindingErrorKind::MissingTexture,
            format!("this frame imports two buffers and no texture, so {id:?} has no answer"),
        ))
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<BufferId, GlRetentionLease>, FrameBindingError> {
        if id == self.vertices.0 {
            return Ok(reissue(&self.vertices.1));
        }
        if id == self.indices.0 {
            return Ok(reissue(&self.indices.1));
        }
        Err(missing(
            FrameBindingErrorKind::MissingBuffer,
            format!("this frame binds two buffers, so {id:?} has no answer"),
        ))
    }
}

#[test]
fn a_frame_draws_from_caller_owned_buffers_this_device_created_and_uploaded() {
    let positions = position_stream();
    let elements = index_stream();
    let (vertex_bytes, index_bytes) = (positions.len() as u64, elements.len() as u64);

    let mut graph = RenderGraph::new();
    let vertices = imported(&mut graph, "triangle-vertices", vertex_bytes);
    let indices = imported(&mut graph, "triangle-indices", index_bytes);
    let colour = graph.create_texture("colour", plain_texture());
    let raster = graph.add_raster_pass(
        "triangle",
        |pass| {
            let vertices = pass.read_buffer(
                &vertices.version,
                BufferReadUse::Vertex,
                BufferRange::whole(),
            );
            let indices =
                pass.read_buffer(&indices.version, BufferReadUse::Index, BufferRange::whole());
            let colour = pass.color_attachment(colour, cleared());
            (colour, (vertices, indices))
        },
        |commands, resolver, (vertices, indices), _frame| {
            commands.set_pipeline(RasterPipelineId::new(1))?;
            // Both commands take the graph's *handles*, not physical objects: the
            // buffers the frame draws from are resolved by the recording path out
            // of whatever the provider answered with, which is why a caller-owned
            // object needed no new verb anywhere on the vertex-input path.
            commands.set_vertex_buffer(0, vertices)?;
            commands.set_index_buffer(indices, IndexFormat::Uint32)?;
            let bindings = resolver.resolve_bindings(BindingSetId::new(0), &[], &[])?;
            commands.set_bindings(&bindings)?;
            commands.draw_indexed(0..3, 0, 0..1)?;
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
    // Created and filled *before* the executor is built, which is the claim the
    // whole slice rests on: neither verb needs a frame, an encoder or a provider.
    let own = |device: &mut Adapter, size: u64, usage: BufferUsageKind, bytes: &[u8]| {
        let buffer = device
            .create_transient_buffer(BufferDesc { size }, BufferUsage::from_kinds([usage]))
            .expect("a caller-owned buffer");
        device
            .upload_buffer(buffer.physical, 0, bytes)
            .expect("the bytes reach the buffer this device created");
        buffer
    };
    let vertex_buffer = own(
        &mut device,
        vertex_bytes,
        BufferUsageKind::Vertex,
        &positions,
    );
    let index_buffer = own(&mut device, index_bytes, BufferUsageKind::Index, &elements);
    let (vertex_physical, index_physical) = (vertex_buffer.physical, index_buffer.physical);

    let compiled = graph
        .compile(device.capabilities())
        .expect("the graph asks this device only for what its own capabilities prove")
        .graph;
    let mut objects = device.object_registry();
    let kernel = RasterKernel::IndexedPositionFloat32x3;
    objects
        .register_raster_pipeline(RasterPipelineId::new(1), kernel)
        .expect("WebGL2 writes a dialect for this artifact");
    objects.register_bindings(BindingSetId::new(0), Recipe::Raster(kernel));

    let provider = UploadedBuffers {
        vertices: (BufferBindingId::new(1), vertex_buffer),
        indices: (BufferBindingId::new(2), index_buffer),
    };
    let executor = FrameExecutor::new(device);
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(vertices.slot, provider.vertices.0);
    inputs.bind_buffer(indices.slot, provider.indices.0);
    let frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &provider,
            &objects,
        )
        .expect("both imports resolve, because the provider answers each with an object the contract accepts");

    let exported = frame
        .exports
        .texture(slot)
        .expect("the graph's one export came back");
    assert_eq!(
        exported.descriptor,
        plain_texture(),
        "the graph's own transient is unaffected by the imports beside it"
    );

    match trace(&executor).as_slice() {
        [
            MockCall::CreateBuffer(created_vertex),
            MockCall::UploadBuffer {
                buffer: filled_vertex,
                offset: vertex_offset,
                size: uploaded_vertex,
            },
            MockCall::CreateBuffer(created_index),
            MockCall::UploadBuffer {
                buffer: filled_index,
                offset: index_offset,
                size: uploaded_index,
            },
            MockCall::CreateTexture(created_texture),
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
            // The first four calls are the whole point, and they are the ones the
            // other frame has none of: all of them happened before the frame's
            // own transient was created, so both objects the frame drew from were
            // already backed when the frame began.
            assert_eq!(
                (*created_vertex, *created_index),
                (vertex_physical, index_physical),
                "the buffers the frame drew from are the two this caller created"
            );
            assert_ne!(
                *created_vertex, *created_index,
                "and they are two allocations rather than one shared between the roles"
            );
            assert_eq!(
                (*filled_vertex, *filled_index),
                (vertex_physical, index_physical),
                "and the ones the uploads filled, rather than buffers of their own"
            );
            assert_eq!(
                (*vertex_offset, *index_offset),
                (0, 0),
                "each upload landed at the caller's offset"
            );
            assert_eq!(
                (*uploaded_vertex, *uploaded_index),
                (vertex_bytes, index_bytes),
                "and each covered the caller's own bytes, which are the size its range was derived from"
            );
            assert_eq!(
                *created_texture, exported.physical,
                "the frame's own transient is still the texture it exported"
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
                &GlDrawCommand::Indexed(GlIndexedDraw {
                    first_index: 0,
                    index_count: 3,
                    instance_count: 1,
                }),
                "the draw is the one the frame asked for, over the element stream the caller uploaded"
            );
        }
        other => panic!("the frame issued a different sequence: {other:?}"),
    }
}
