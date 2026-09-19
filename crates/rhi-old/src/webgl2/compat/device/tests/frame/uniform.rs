//! The frame whose uniform is a caller-owned object.
//!
//! The sibling frames cover the two extremes of the binding seam: one resolves no
//! resources at all, and one draws from caller-owned geometry.  This one is the
//! third arm and the first that binds something a *shader* reads -- a frame
//! uniform supplied as an imported buffer, which is what every artifact in the
//! camera/material family declares (`portable_identity` gives it
//! `binding_count: 1`, `uniform_binding: Some(0)` and an 80-byte block).
//!
//! # This slice needed no new surface, and that is the finding
//!
//! The vertex-input frame had to wait for a verb ([`super::super::super::GlCompatibilityDevice`]'s
//! `upload_buffer`) because nothing in the execution contract writes host bytes.
//! The uniform slice is the case that shows how narrow that gap was: a frame
//! uniform is *also* just bytes in a caller-owned buffer, so the verb that closed
//! the vertex slice closes this one too, and nothing at any layer changed to
//! admit it.  The two slices differ in what the *recipe* declares -- a binding
//! count of one instead of zero, and a block at binding zero that the shader
//! reads -- not in what the caller has to produce.
//!
//! What this file therefore checks is that the recipe half is actually carried:
//! the pass declares a uniform read, the set resolves to exactly one resource,
//! and the draw that comes out names the imported buffer at the block's own
//! binding number.  The single call that proves it is the `BindUniformBuffer`
//! between the vertex input and the draw, and it is worth pinning *there* rather
//! than in the verb suite because only a whole frame can show that the recipe, the
//! import and the provider's answer meet at the draw.

use fluxel_rendergraph::{
    BindingResource, BindingSetId, BoundBuffer, BoundTexture, BufferBindingId, BufferRange,
    BufferReadUse, BufferUsageKind, ExecutionBackend, ExportTextureContract, FrameBindingError,
    FrameBindingErrorKind, FrameExecutor, FrameInputs, FrameResourceProvider, IndexFormat,
    RasterPipelineId, RenderGraph, ResourceAccessState, TextureBindingId,
};

use super::super::super::object::Recipe;
use super::super::super::retention::GlRetentionLease;
use super::super::{Adapter, adapter, plain_texture};
use super::{cleared, imported, index_stream, missing, own, position_stream, reissue, trace};
use crate::resource::RasterKernel;
use crate::webgl2::api::{BufferId, GlDrawCommand, GlIndexedDraw, MockCall, TextureId};

/// The binding recipe this frame registers its one set under.
///
/// Distinct from the empty set the sibling frames use, and the distinction is
/// load-bearing rather than cosmetic: the adapter refuses a set resolved for one
/// artifact when another is installed (`set_bindings`), so a frame that wants the
/// recipe check to have anything to compare needs its own number.
fn camera_bindings() -> BindingSetId {
    BindingSetId::new(1)
}

/// The 80 bytes the camera/material artifacts declare as their frame block.
///
/// Built from the same two fields the artifact's own struct names -- a
/// `mat4x4<f32>` view-projection and a `vec4<f32>` base colour, which is where
/// eighty bytes comes from -- rather than from a byte literal, so the length is
/// something a reader can check against the declaration instead of trusting.
/// The test below asserts this equals the artifact's own `uniform_binding_size`,
/// which is what keeps the two from drifting apart silently.
///
/// Reachable from the sibling scenario because the textured artifact declares the
/// *same* block: it is the camera/material struct with one more binding beside it,
/// so `uniform_binding_size` is eighty there too and a second spelling here would
/// be a second thing to keep in step with the artifact.  A sibling is not a
/// descendant, so the visibility has to be said; it is still `super` and not
/// `pub(crate)`, because nothing outside this suite has any business with it.
pub(super) fn frame_uniforms() -> Vec<u8> {
    const VIEW_PROJECTION: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ];
    const BASE_COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    VIEW_PROJECTION
        .iter()
        .chain(BASE_COLOR.iter())
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

/// A provider that answers three bindings with objects *this device* made.
///
/// Three rather than two, and the third is the point of this file: a uniform is
/// an imported buffer like any other, so the provider dispatch that the sibling
/// frame proves for two *usage contracts* is proved here for three -- and the
/// uniform is the one whose usage the adapter refuses to bind without
/// (`GlBufferUsage::UNIFORM`), which is a check no geometry binding makes.
struct UploadedFrame {
    vertices: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
    indices: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
    uniform: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
}

impl FrameResourceProvider<Adapter> for UploadedFrame {
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<TextureId, GlRetentionLease>, FrameBindingError> {
        Err(missing(
            FrameBindingErrorKind::MissingTexture,
            format!("this frame imports three buffers and no texture, so {id:?} has no answer"),
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
        if id == self.uniform.0 {
            return Ok(reissue(&self.uniform.1));
        }
        Err(missing(
            FrameBindingErrorKind::MissingBuffer,
            format!("this frame binds three buffers, so {id:?} has no answer"),
        ))
    }
}

#[test]
fn a_frame_binds_an_imported_uniform_at_the_binding_number_the_artifact_declares() {
    let kernel = RasterKernel::IndexedPositionFloat32x3CameraMaterial;
    let identity = kernel.portable_identity();
    let uniform_bytes = frame_uniforms();
    assert_eq!(
        uniform_bytes.len() as u64,
        identity.uniform_binding_size,
        "the bytes this frame uploads are the block the artifact declares, not a length chosen here"
    );
    let uniform_binding = identity
        .uniform_binding
        .expect("a camera/material artifact is the case that has a frame block at all");

    let positions = position_stream();
    let elements = index_stream();
    let (vertex_bytes, index_bytes) = (positions.len() as u64, elements.len() as u64);
    let uniform_size = uniform_bytes.len() as u64;

    let mut graph = RenderGraph::new();
    let vertices = imported(&mut graph, "triangle-vertices", vertex_bytes);
    let indices = imported(&mut graph, "triangle-indices", index_bytes);
    let uniform = imported(&mut graph, "frame-uniforms", uniform_size);
    let colour = graph.create_texture("colour", plain_texture());
    let raster = graph.add_raster_pass(
        "camera-material-triangle",
        |pass| {
            let vertices = pass.read_buffer(
                &vertices.version,
                BufferReadUse::Vertex,
                BufferRange::whole(),
            );
            let indices =
                pass.read_buffer(&indices.version, BufferReadUse::Index, BufferRange::whole());
            // `Uniform` and not `Vertex`: the read use is what the plan turns into
            // the usage the import's provider must have covered, so declaring the
            // wrong one here is refused at resolution rather than at the draw.
            let uniform = pass.read_buffer(
                &uniform.version,
                BufferReadUse::Uniform,
                BufferRange::whole(),
            );
            let colour = pass.color_attachment(colour, cleared());
            (colour, (vertices, indices, uniform))
        },
        |commands, resolver, (vertices, indices, uniform), _frame| {
            commands.set_pipeline(RasterPipelineId::new(1))?;
            // The set is resolved with one resource, because the artifact declares
            // a binding count of one -- a set of any other size is refused by the
            // adapter before the uniform arm is reached, which is why the empty
            // set the sibling frames use cannot stand in here.
            let bindings = resolver.resolve_bindings(
                camera_bindings(),
                &[BindingResource::BufferRead(uniform)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.set_vertex_buffer(0, vertices)?;
            commands.set_index_buffer(indices, IndexFormat::Uint32)?;
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
    let vertex_buffer = own(
        &mut device,
        vertex_bytes,
        BufferUsageKind::Vertex,
        &positions,
    );
    let index_buffer = own(&mut device, index_bytes, BufferUsageKind::Index, &elements);
    let uniform_buffer = own(
        &mut device,
        uniform_size,
        BufferUsageKind::Uniform,
        &uniform_bytes,
    );
    let (vertex_physical, index_physical, uniform_physical) = (
        vertex_buffer.physical,
        index_buffer.physical,
        uniform_buffer.physical,
    );

    let compiled = graph
        .compile(device.capabilities())
        .expect("the graph asks this device only for what its own capabilities prove")
        .graph;
    let mut objects = device.object_registry();
    objects
        .register_raster_pipeline(RasterPipelineId::new(1), kernel)
        .expect("WebGL2 writes a dialect for this artifact");
    objects.register_bindings(camera_bindings(), Recipe::Raster(kernel));

    let provider = UploadedFrame {
        vertices: (BufferBindingId::new(1), vertex_buffer),
        indices: (BufferBindingId::new(2), index_buffer),
        uniform: (BufferBindingId::new(3), uniform_buffer),
    };
    let executor = FrameExecutor::new(device);
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(vertices.slot, provider.vertices.0);
    inputs.bind_buffer(indices.slot, provider.indices.0);
    inputs.bind_buffer(uniform.slot, provider.uniform.0);
    let frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &provider,
            &objects,
        )
        .expect("all three imports resolve, including the one the shader reads");

    let exported = frame
        .exports
        .texture(slot)
        .expect("the graph's one export came back");
    assert_eq!(
        exported.descriptor,
        plain_texture(),
        "the graph's own transient is unaffected by the three imports beside it"
    );

    match trace(&executor).as_slice() {
        [
            MockCall::CreateBuffer(created_vertex),
            MockCall::UploadBuffer {
                buffer: filled_vertex,
                ..
            },
            MockCall::CreateBuffer(created_index),
            MockCall::UploadBuffer {
                buffer: filled_index,
                ..
            },
            MockCall::CreateBuffer(created_uniform),
            MockCall::UploadBuffer {
                buffer: filled_uniform,
                ..
            },
            MockCall::CreateTexture(created_texture),
            MockCall::CreateFramebuffer(_),
            MockCall::BeginRenderPass(_),
            MockCall::CreateProgram(linked),
            MockCall::CreateVertexArray(array),
            MockCall::BindVertexArray(bound),
            MockCall::SelectProgram(selected),
            MockCall::SetRasterPipeline {
                program,
                vertex_array,
            },
            MockCall::BindVertexArray(rebound),
            MockCall::BindUniformBuffer {
                index,
                buffer: Some(bound_uniform),
                offset,
                size,
            },
            MockCall::DrawRaster(draw),
            MockCall::EndRenderPass,
            MockCall::Flush,
            MockCall::CreateFence(_),
        ] => {
            assert_eq!(
                (*created_vertex, *created_index, *created_uniform),
                (vertex_physical, index_physical, uniform_physical),
                "the three buffers the frame bound are the three this caller created"
            );
            assert_eq!(
                (*filled_vertex, *filled_index, *filled_uniform),
                (vertex_physical, index_physical, uniform_physical),
                "and the three the uploads filled, rather than buffers of their own"
            );
            assert_eq!(
                *created_uniform, uniform_physical,
                "the block the frame read is the caller's uniform buffer"
            );
            assert_eq!(
                *created_texture, exported.physical,
                "the frame's own transient is still the texture it exported"
            );
            assert_eq!(
                *bound_uniform, uniform_physical,
                "the block bound at the draw is the buffer the provider answered with"
            );
            assert_eq!(
                *index, uniform_binding,
                "bound at the binding number the artifact declares, not a number chosen by the frame"
            );
            assert_eq!(
                (*offset, *size),
                (0, 0),
                "a whole-range block is (0, 0) at the binding point, which is the range the pass declared"
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
                "and the draw follows the block it reads, which is why the bind precedes it"
            );
        }
        other => panic!("the frame issued a different sequence: {other:?}"),
    }
}
