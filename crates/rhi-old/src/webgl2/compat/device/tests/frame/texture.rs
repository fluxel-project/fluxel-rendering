//! The frame that samples a caller-owned image.
//!
//! This is the last of the three binding arms and the only one that needed new
//! surface.  The sibling frames cover a recipe that reads nothing and one that
//! reads a block of bytes; a sampled texture is the binding whose *object* is not
//! expressible as bytes in a buffer, and it is therefore the one where the frame
//! contract and the adapter's vocabulary actually disagreed.  A provider can only
//! answer with something that already exists, and before
//! [`GlCompatibilityDevice::upload_texture`](super::super::super::GlCompatibilityDevice)
//! there was no verb anywhere on the adapter that could put host pixels into a
//! texture it had made.
//!
//! # What the record already carried, and what had to be added
//!
//! Worth stating because the reconnaissance that opened this slice expected the
//! opposite: the adapter turned out to be missing *one* verb and no state at all.
//! `create_transient_texture` already retains an
//! [`Attachment`](super::super::super::pass::Attachment) -- the projection of the
//! creation descriptor that the pass lowering reads -- and that record is
//! exactly what a region derivation needs: the extent, the format and the
//! dimension.  So the second of the two questions this slice opened with ("can
//! the adapter check the region at all, or must the creation path start keeping a
//! descriptor record?") answers itself from the record that was already there, and
//! neither `create_transient_texture` nor `refresh`'s purge list changed.
//!
//! Everything downstream of the verb was likewise already in place, and this file
//! is what proves it rather than asserting it: the texture a provider answers with
//! is one `create_transient_texture` made, so it carries `Undefined` as its
//! initial state and matches an import contract declared the same way, and it is
//! in the attachment map the bind path checks against, so the draw's sampled bind
//! needs no new check either.
//!
//! # What is pinned, and why it is pinned here
//!
//! The texture is bound at `unit: 1` and no sampler is bound at all.  The first is
//! the recipe's own binding number rather than a unit the frame chose, and the
//! second is the fold: this artifact's sampler has no logical counterpart, so the
//! backend folds it into the texture and the trace has no `BindSampler` to show.
//! Both are facts a verb suite cannot state -- the verb suite binds through the
//! object registry and never reaches a draw -- so they belong to the frame that
//! makes them matter.

use fluxel_rendergraph::{
    BindingResource, BindingSetId, BoundBuffer, BoundTexture, BufferBindingId, BufferRange,
    BufferReadUse, BufferUsageKind, ExecutionBackend, ExportTextureContract, Extent3d,
    ExternalOwnership, FrameBindingError, FrameBindingErrorKind, FrameExecutor, FrameInputs,
    FrameResourceProvider, ImportTextureContract, IndexFormat, InitialContents, RasterPipelineId,
    RenderGraph, ResourceAccessState, TextureBindingId, TextureDesc, TextureDimension,
    TextureFormat, TextureRange, TextureReadUse, TextureUsage, TextureUsageKind,
};

use super::super::super::object::Recipe;
use super::super::super::retention::GlRetentionLease;
use super::super::{Adapter, adapter, plain_texture};
use super::{cleared, imported, index_stream, missing, own, position_stream, reissue, trace};
use crate::resource::RasterKernel;
use crate::webgl2::api::{
    BufferId, GlDrawCommand, GlIndexedDraw, GlTextureTarget, MockCall, TextureId,
};

/// The binding recipe this frame registers its one set under.
///
/// A third number, distinct from the empty set and from the frame-uniform set,
/// for the reason stated beside the other two: the adapter refuses a set resolved
/// for one artifact when another is installed, so a frame that wants the recipe
/// check to have anything to compare needs a number of its own.
fn sampled_bindings() -> BindingSetId {
    BindingSetId::new(2)
}

/// The image this frame imports: two by two, one layer, one mip.
///
/// Deliberately not [`plain_texture`], which is the graph's own colour target.
/// The two differing in extent is what makes the trace readable: a reader can see
/// that the texture the provider answered with is the one the upload filled and
/// not the attachment the pass rendered into, which two identically shaped
/// textures would leave to the identity comparison alone.
fn sampled_image() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 2,
            height: 2,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

/// The image's four texels, as the tightly packed RGBA8 bytes one upload covers.
///
/// Written as texels rather than as sixteen bytes because that is what the array
/// is: a red, a green, a blue and a yellow, one per corner.  Nothing here reads
/// them back -- these tests record calls rather than pixels -- so what the values
/// have to be is *a* defined image of the shape the descriptor declares, and the
/// shape is the only part a reader can check.
fn image_pixels() -> Vec<u8> {
    const TEXELS: [[u8; 4]; 4] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    TEXELS.iter().flatten().copied().collect()
}

/// The same resolved texture, restated so a provider can hand it out.
///
/// The buffer twin of this lives in [`super`] because two scenarios answer with
/// buffers; this one is here because exactly one scenario answers with a texture,
/// and a fixture with a single user belongs beside it.  The lease is cloned for
/// the reason stated there: the contract lets a provider answer the same object
/// more than once within a frame, so the retained object is released when the last
/// holder drops rather than when this one answers.
fn reissue_texture(
    texture: &BoundTexture<TextureId, GlRetentionLease>,
) -> BoundTexture<TextureId, GlRetentionLease> {
    BoundTexture {
        device: texture.device,
        identity: texture.identity,
        physical: texture.physical,
        descriptor: texture.descriptor,
        usage: texture.usage,
        initial_state: texture.initial_state,
        lease: texture.lease.clone(),
    }
}

/// A provider that answers four bindings with objects *this device* made.
///
/// Four rather than one, and the fourth is the point of this file: the three
/// buffers are the geometry and the frame block the sibling scenarios already
/// prove, so the only new claim is the one the texture carries.  Everything else
/// is here because this artifact declares all four -- it is the richest recipe in
/// the table, which is why it is the one that shows the whole set arriving in
/// order at one draw.
struct UploadedFrame {
    vertices: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
    indices: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
    uniform: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
    image: (TextureBindingId, BoundTexture<TextureId, GlRetentionLease>),
}

impl FrameResourceProvider<Adapter> for UploadedFrame {
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<TextureId, GlRetentionLease>, FrameBindingError> {
        if id == self.image.0 {
            return Ok(reissue_texture(&self.image.1));
        }
        Err(missing(
            FrameBindingErrorKind::MissingTexture,
            format!("this frame imports one texture, so {id:?} has no answer"),
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

/// Creates one caller-owned texture and fills it, before any frame exists.
///
/// [`own`]'s twin on the texture side, and the pair is the whole claim the import
/// half of the frame contract rests on: a caller makes the object, fills it, and
/// everything in both steps happens outside any frame, encoder or executor.
fn own_image(
    device: &mut Adapter,
    descriptor: TextureDesc,
    bytes: &[u8],
) -> BoundTexture<TextureId, GlRetentionLease> {
    let texture = device
        .create_transient_texture(
            descriptor,
            TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        )
        .expect("a caller-owned image");
    device
        .upload_texture(texture.physical, bytes)
        .expect("the pixels reach the texture this device created");
    texture
}

#[test]
fn a_frame_draws_from_caller_owned_buffers_and_samples_a_caller_owned_image() {
    let kernel = RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture;
    let identity = kernel.portable_identity();
    let (texture_binding, uniform_binding) = (
        identity
            .texture_binding
            .expect("a textured artifact is the case that names a sampled binding"),
        identity
            .uniform_binding
            .expect("and it declares the frame block too, as every camera/material artifact does"),
    );
    assert_eq!(
        identity.binding_count, 2,
        "one block and one image: the set below is resolved with exactly these two, in this order"
    );

    let positions = position_stream();
    let elements = index_stream();
    let uniforms = super::uniform::frame_uniforms();
    let pixels = image_pixels();
    let descriptor = sampled_image();
    assert_eq!(
        pixels.len() as u64,
        u64::from(descriptor.extent.width) * u64::from(descriptor.extent.height) * 4,
        "the bytes this frame uploads are the whole level the descriptor declares, not a length chosen here"
    );

    let mut graph = RenderGraph::new();
    let vertices = imported(&mut graph, "triangle-vertices", positions.len() as u64);
    let indices = imported(&mut graph, "triangle-indices", elements.len() as u64);
    let uniform = imported(&mut graph, "frame-uniforms", uniforms.len() as u64);
    let image = graph.import_texture_slot(
        "sampled-image",
        ImportTextureContract {
            descriptor,
            // `Undefined` for the same reason the buffers declare it, and it is
            // the choice this file checks rather than assumes: the verb that
            // fills the image is a caller's and leaves no state behind, so the
            // record the executor compares against is the creation record's.
            initial_state: ResourceAccessState::Undefined,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let colour = graph.create_texture("colour", plain_texture());
    let raster = graph.add_raster_pass(
        "camera-material-textured-triangle",
        |pass| {
            let vertices = pass.read_buffer(
                &vertices.version,
                BufferReadUse::Vertex,
                BufferRange::whole(),
            );
            let indices =
                pass.read_buffer(&indices.version, BufferReadUse::Index, BufferRange::whole());
            let uniform = pass.read_buffer(
                &uniform.version,
                BufferReadUse::Uniform,
                BufferRange::whole(),
            );
            // `Sampled` and not `CopySource`: the read use is what the plan turns
            // into the usage the import's provider must have covered, so declaring
            // the wrong one is refused at resolution rather than at the draw.
            let image =
                pass.read_texture(&image.version, TextureReadUse::Sampled, TextureRange::Whole);
            let colour = pass.color_attachment(colour, cleared());
            (colour, (vertices, indices, uniform, image))
        },
        |commands, resolver, (vertices, indices, uniform, image), _frame| {
            commands.set_pipeline(RasterPipelineId::new(1))?;
            // Resolved in the recipe's own order -- block zero, then image one --
            // because the adapter applies a set one binding at a time and checks
            // each against the number the artifact declares.
            let bindings = resolver.resolve_bindings(
                sampled_bindings(),
                &[
                    BindingResource::BufferRead(uniform),
                    BindingResource::TextureRead(image),
                ],
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
        positions.len() as u64,
        BufferUsageKind::Vertex,
        &positions,
    );
    let index_buffer = own(
        &mut device,
        elements.len() as u64,
        BufferUsageKind::Index,
        &elements,
    );
    let uniform_buffer = own(
        &mut device,
        uniforms.len() as u64,
        BufferUsageKind::Uniform,
        &uniforms,
    );
    let sampled = own_image(&mut device, descriptor, &pixels);
    let (vertex_physical, index_physical, uniform_physical, image_physical) = (
        vertex_buffer.physical,
        index_buffer.physical,
        uniform_buffer.physical,
        sampled.physical,
    );

    let compiled = graph
        .compile(device.capabilities())
        .expect("the graph asks this device only for what its own capabilities prove")
        .graph;
    let mut objects = device.object_registry();
    objects
        .register_raster_pipeline(RasterPipelineId::new(1), kernel)
        .expect("WebGL2 writes a dialect for this artifact");
    objects.register_bindings(sampled_bindings(), Recipe::Raster(kernel));

    let provider = UploadedFrame {
        vertices: (BufferBindingId::new(1), vertex_buffer),
        indices: (BufferBindingId::new(2), index_buffer),
        uniform: (BufferBindingId::new(3), uniform_buffer),
        image: (TextureBindingId::new(1), sampled),
    };
    let executor = FrameExecutor::new(device);
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(vertices.slot, provider.vertices.0);
    inputs.bind_buffer(indices.slot, provider.indices.0);
    inputs.bind_buffer(uniform.slot, provider.uniform.0);
    inputs.bind_texture(image.slot, provider.image.0);
    let frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &provider,
            &objects,
        )
        .expect("all four imports resolve, including the image the shader samples");

    let exported = frame
        .exports
        .texture(slot)
        .expect("the graph's one export came back");
    assert_eq!(
        exported.descriptor,
        plain_texture(),
        "the graph's own transient is unaffected by the four imports beside it"
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
            MockCall::CreateTexture(created_image),
            MockCall::UploadTexture(filled_image),
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
            MockCall::BindTexture {
                unit,
                target,
                texture: Some(bound_image),
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
                (*created_image, *filled_image),
                (image_physical, image_physical),
                "the image is one object: created and then filled, rather than a second texture the upload made"
            );
            assert_eq!(
                (*filled_vertex, *filled_index, *filled_uniform),
                (vertex_physical, index_physical, uniform_physical),
                "and the three uploads filled the buffers they were created for, rather than buffers of their own"
            );
            assert_eq!(
                *created_texture, exported.physical,
                "and the graph's own transient is still the texture it exported"
            );
            assert_eq!(
                *bound_uniform, uniform_physical,
                "the block bound at the draw is the buffer the provider answered with"
            );
            assert_eq!(
                *bound_image, image_physical,
                "and the image sampled at the draw is the texture the caller created and filled"
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
                *unit, texture_binding,
                "and the image is bound at *its* binding number used as the texture unit, which is the fold's whole visible consequence"
            );
            assert_eq!(
                *target,
                GlTextureTarget::D2,
                "the target comes from the recorded attachment rather than from the frame, which is why the record has to carry a dimension"
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
                "and the draw follows both binds, which is why the whole set precedes it"
            );
        }
        other => panic!("the frame issued a different sequence: {other:?}"),
    }
}
