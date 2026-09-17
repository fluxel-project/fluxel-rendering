//! The frame whose geometry is a caller-owned object.
//!
//! The sibling frame consults no provider, so it can be driven by the graph's own
//! transients alone.  This one draws from two buffers the *caller* made, which is
//! the other half of the frame contract: an import is resolved by asking a
//! [`FrameResourceProvider`], and a provider can only answer with something that
//! already exists.  What this file proves is that such an object needs nothing
//! new anywhere on the vertex-input path -- reproduce the two commands a
//! vertex-array state needs, hand back the two objects, and the ordinary raster
//! path draws from them.
//!
//! Two imports rather than one, and that is the vertex-input slice being the whole
//! vertex-array state instead of its attribute half: every artifact in this
//! family's table that reads a vertex stream is an indexed one (`Triangle`, the
//! one kernel with no vertex stream, is also the one that declares no vertex
//! layout), so an attribute binding with no element binding is a frame no recipe
//! accepts.  The pair is also strictly better evidence than one buffer would be,
//! because the provider then has to answer *per binding id* rather than returning
//! whatever it happens to hold.

use fluxel_rendergraph::{
    BindingSetId, BoundBuffer, BoundTexture, BufferBindingId, BufferRange, BufferReadUse,
    BufferUsageKind, ExecutionBackend, ExportTextureContract, FrameBindingError,
    FrameBindingErrorKind, FrameExecutor, FrameInputs, FrameResourceProvider, IndexFormat,
    RasterPipelineId, RenderGraph, ResourceAccessState, TextureBindingId,
};

use super::super::super::object::Recipe;
use super::super::super::retention::GlRetentionLease;
use super::super::{Adapter, adapter, plain_texture};
use super::{cleared, imported, index_stream, missing, own, position_stream, reissue, trace};
use crate::resource::RasterKernel;
use crate::webgl2::api::{BufferId, GlDrawCommand, GlIndexedDraw, MockCall, TextureId};

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
