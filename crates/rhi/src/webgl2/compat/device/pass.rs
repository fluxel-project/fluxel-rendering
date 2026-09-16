//! Lowering one recorded raster pass onto Layer 1's pass vocabulary.
//!
//! Responsibility: turn the common contract's pass description, and the state a
//! frame recorded inside it, into the values Layer 2's `begin_pass`,
//! `set_pipeline` and `set_vertex_input` take.  Every function here is pure: it
//! reads contract and Layer 1 values and produces Layer 1 values, and it neither
//! touches the machine nor issues anything.  That is what makes the decisions
//! below checkable without a context, and it is why the adapter's verbs stay
//! short enough to read as the order they drive the machine in.
//!
//! Not owned here: the framebuffer's *lifetime* (Layer 2's session cache derives
//! and evicts it, and the verb honours the ownership answer it is given), the
//! machine calls themselves, and whether a pass may be open at all (the encoder
//! records that).
//!
//! # Where an attachment's shape comes from
//!
//! Layer 1's framebuffer and pass descriptors are built from [`GlTextureView`]s,
//! which carry the attachment's format, extent, sample count and the coordinate
//! the view selects.  The contract's [`RasterColorAttachment`] carries only a
//! texture identity and a range, and Layer 1 exposes no query that answers the
//! rest -- a [`TextureId`] names an object, it does not describe one.
//!
//! So the adapter remembers.  [`Attachment`] is what it records when it creates
//! a texture, and remembering is complete here rather than merely convenient:
//! the contract's `Texture` associated type is this adapter's own, and
//! `create_transient_texture` is the only thing that mints one, so a frame
//! cannot name a texture this record does not describe.  The record is dropped
//! where the object is destroyed, so an identity reused after a deletion cannot
//! resolve to the shape of its previous occupant.
//!
//! # The three attachment facts that have no lowering
//!
//! [`LoadOp::DontCare`] and [`StoreOp::Discard`] are refused rather than mapped.
//! Layer 1's [`GlLoadOp`] is `Load` or `Clear` and has no third case, so
//! lowering a don't-care means choosing which of the two the caller meant -- and
//! a pass told its attachment contents are undefined did not ask for either.
//! `Discard` does exist in Layer 1, but `GlRenderPassDescriptor::validate`
//! requires a resolve target for it and this adapter has no resolve path to give
//! it one.  Both refusals agree with the retained path, which refuses the same
//! two facts for its own backends.
//!
//! A depth-stencil attachment is refused for a third reason: none of the ten
//! closed artifacts declares depth state, so a pass that attaches depth storage
//! would attach it to a pipeline that can neither test nor write it.  Refusing
//! keeps that a named gap rather than a silently dropped attachment, and closing
//! it is a recipe question and not a verb one.

use fluxel_rendergraph::{
    IndexFormat, LoadOp, RasterColorAttachment, RasterDepthStencilAttachment, ScissorRect, StoreOp,
    TextureRange, Viewport,
};

use crate::resource::RasterKernel;
use crate::webgl2::api::{
    GlAttachmentTarget, GlColorAttachment, GlColorClearValue, GlColorTargetState, GlCullMode,
    GlError, GlFormat, GlFramebufferDescriptor, GlFrontFace, GlIndexBinding, GlIndexFormat,
    GlLoadOp, GlMultisampleState, GlPrimitiveTopology, GlRasterState, GlScissorRect, GlStoreOp,
    GlTextureDesc, GlTextureDimension, GlTextureTarget, GlTextureView, GlVertexBufferBinding,
    GlViewport, TextureId,
};
use crate::webgl2::state::geometry::VertexInput;

use super::object::RasterPipeline;

/// What the adapter recorded about one texture it created.
///
/// A projection of the creation descriptor and not a copy of it: the usage bits
/// and the mip count are facts no pass asks for, and carrying them would make
/// this record look like a second descriptor that has to be kept in step with
/// the first.
#[derive(Clone, Copy, Debug)]
pub(super) struct Attachment {
    dimension: GlTextureDimension,
    format: GlFormat,
    width: u32,
    height: u32,
    sample_count: u32,
    layers: u32,
}

impl Attachment {
    /// The facts a pass needs, read off the descriptor the texture was created
    /// from.
    pub(super) fn of(desc: &GlTextureDesc) -> Self {
        Self {
            dimension: desc.dimension,
            format: desc.format,
            width: desc.extent.width,
            height: desc.extent.height,
            sample_count: desc.sample_count,
            layers: desc.extent.depth_or_layers,
        }
    }

    /// The extents a pass over this attachment renders into, which is also the
    /// viewport a pipeline is defaulted to when the frame sets none.
    pub(super) fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The sample count a pipeline over this attachment declares.
    pub(super) fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// The texture target a bind of this attachment uses.
    ///
    /// Refused for a one-dimensional texture, which is the one dimension this
    /// family has that no binding point accepts: a `D1` texture can be created
    /// and copied but not sampled or attached, so a recipe that named one would
    /// have no way to read it.
    pub(super) fn target(&self) -> Result<GlTextureTarget, GlError> {
        Ok(match self.dimension {
            GlTextureDimension::D2 => GlTextureTarget::D2,
            GlTextureDimension::D3 => GlTextureTarget::D3,
            GlTextureDimension::Cube => GlTextureTarget::Cube,
            GlTextureDimension::D2Array => GlTextureTarget::D2Array,
            GlTextureDimension::D1 => {
                return Err(unsupported(
                    "set-bindings",
                    "this family's binding points accept no one-dimensional texture",
                ));
            }
        })
    }
}

/// One attachment's view, at the coordinate `range` selects.
pub(super) fn view(id: TextureId, facts: Attachment, range: TextureRange) -> GlTextureView {
    let (mip_level, array_layer, layer_count) = match range {
        // The whole allocation, which for a non-arrayed texture is one layer.
        TextureRange::Whole => (0, 0, facts.layers),
        TextureRange::Subresources {
            base_mip_level,
            base_array_layer,
            array_layer_count,
            ..
        } => (base_mip_level, base_array_layer, array_layer_count),
    };
    GlTextureView {
        target: GlAttachmentTarget::Texture(id),
        format: facts.format,
        mip_level,
        array_layer,
        layer_count,
        width: facts.width,
        height: facts.height,
        sample_count: facts.sample_count,
    }
}

/// The framebuffer descriptor for one pass's attachment views.
///
/// The draw buffers are the identity mapping over the colour attachments: given
/// `n` views, attachment `i` is written by fragment output `i`.  A fixed recipe
/// writes one output and [`admit`] admits one colour attachment, so this is `[0]`
/// in every case the adapter can reach -- and it is built from the count rather
/// than written as a literal, so that a pass with a different count is refused by
/// Layer 1's framebuffer validation instead of silently writing the wrong output.
pub(super) fn framebuffer(
    colours: Vec<GlTextureView>,
    depth_stencil: Option<GlTextureView>,
) -> GlFramebufferDescriptor {
    GlFramebufferDescriptor {
        draw_buffers: (0..colours.len() as u32).collect(),
        color_attachments: colours,
        depth_stencil_attachment: depth_stencil,
    }
}

/// The reason a pass this family cannot run is refused.
///
/// Stated once because two arms of [`admit`] report the same fact -- there is no
/// colour attachment, and the one there is is not at index zero -- and a caller
/// reading the error should not be able to tell which arm it came from, because
/// the mistake is the same one.
const ONE_COLOUR_TARGET: &str = "every closed raster artifact writes one colour target at index zero, so a pass with a different attachment set has no pipeline that could run in it";

/// The one colour attachment a pass this family can run has.
///
/// Returns the attachment rather than merely admitting the set, so that the
/// caller's next step -- looking up what this device recorded about the texture
/// -- has no second place to decide what "one attachment" means and no index to
/// trust.  The refusals are the three facts this family's pass vocabulary has no
/// case for: zero colour attachments, more than one, one that is not at index
/// zero, and any depth-stencil attachment at all.
pub(super) fn admit<'a>(
    colours: &'a [RasterColorAttachment<'a, TextureId>],
    depth_stencil: Option<&RasterDepthStencilAttachment<'_, TextureId>>,
) -> Result<&'a RasterColorAttachment<'a, TextureId>, GlError> {
    let [attachment] = colours else {
        return Err(unsupported("begin-raster", ONE_COLOUR_TARGET));
    };
    if attachment.index != 0 {
        return Err(unsupported("begin-raster", ONE_COLOUR_TARGET));
    }
    if depth_stencil.is_some() {
        return Err(unsupported(
            "begin-raster",
            "no closed raster artifact declares depth-stencil state, so a pass that attaches depth storage would run a pipeline that can neither test nor write it",
        ));
    }
    Ok(attachment)
}

/// The colour attachments one admitted pass lowers to.
///
/// Returns the *attachments* and not a whole [`GlRenderPassDescriptor`], and that
/// is a lifetime decision rather than a taste one: the descriptor needs a
/// framebuffer, the framebuffer is derived by asking Layer 2 for one, and Layer 2
/// may answer that the pass owns the framebuffer it just derived.  So a refusal
/// made *after* the derivation would leave an owned framebuffer with no pass to
/// close it -- nothing would name it again, because the frame only ever sees the
/// error.  Keeping the descriptor's construction in the verb, after the
/// derivation, puts every refusal this lowering can make in front of it.
///
/// `views` is the sequence the caller built from the very attachments it passes
/// here, and that is required rather than convenient: Layer 1's pass validation
/// compares each attachment's view against the framebuffer's at the same index,
/// so a pass built from independently constructed views would be refused as a
/// duplicate attachment.
pub(super) fn attachments(
    colours: &[RasterColorAttachment<'_, TextureId>],
    views: &[GlTextureView],
) -> Result<Vec<GlColorAttachment>, GlError> {
    let mut lowered = Vec::with_capacity(colours.len());
    for (attachment, view) in colours.iter().zip(views) {
        let operations = attachment.operations;
        // The load and the value are two fields of Layer 1's attachment rather
        // than one, so a load that does not clear still has to state a value.  It
        // states zero, which is the one a caller that reads it anyway cannot
        // mistake for a request -- and the refusal above is what keeps a
        // don't-care load from reaching here at all, since choosing a value for
        // one would be choosing what the caller declined to choose.
        let (load, clear) = match operations.load {
            LoadOp::Load => (
                GlLoadOp::Load,
                GlColorClearValue {
                    red: 0,
                    green: 0,
                    blue: 0,
                    alpha: 0,
                },
            ),
            LoadOp::Clear(value) => (GlLoadOp::Clear, clear_value(value)),
            LoadOp::DontCare => {
                return Err(unsupported(
                    "begin-raster",
                    "this family's attachment load is either load or clear, so a don't-care load has no lowering that would not be choosing one of the two",
                ));
            }
        };
        let store = match operations.store {
            StoreOp::Store => GlStoreOp::Store,
            StoreOp::Discard => {
                return Err(unsupported(
                    "begin-raster",
                    "this family's attachment discard requires a resolve target, and this adapter has no resolve path to give it one",
                ));
            }
        };
        lowered.push(GlColorAttachment {
            view: *view,
            resolve_target: None,
            load,
            store,
            clear,
        });
    }
    Ok(lowered)
}

/// One colour clear value, as the IEEE-754 bits Layer 1 stores.
fn clear_value(value: [f32; 4]) -> GlColorClearValue {
    GlColorClearValue {
        red: value[0].to_bits(),
        green: value[1].to_bits(),
        blue: value[2].to_bits(),
        alpha: value[3].to_bits(),
    }
}

/// The viewport a pass defaults to when the frame sets none.
pub(super) fn whole_extent(facts: Attachment) -> GlViewport {
    GlViewport {
        x: 0,
        y: 0,
        width: facts.width,
        height: facts.height,
        // The family's clip-space depth range, as bits, in the order Layer 1
        // declares them.
        min_depth: 0.0f32.to_bits(),
        max_depth: 1.0f32.to_bits(),
    }
}

/// The contract's viewport, as Layer 1 declares one.
///
/// The truncation is the family's shape and not a lost fact: Layer 1's viewport
/// is integral, and `GlRasterState::validate` then bounds-checks the result
/// against the pass extent, so a fractional or out-of-range viewport is refused
/// at the pipeline rather than silently rounded into range.
pub(super) fn viewport(viewport: Viewport) -> GlViewport {
    GlViewport {
        x: viewport.x as u32,
        y: viewport.y as u32,
        width: viewport.width as u32,
        height: viewport.height as u32,
        min_depth: viewport.min_depth.to_bits(),
        max_depth: viewport.max_depth.to_bits(),
    }
}

/// The contract's scissor, as Layer 1 declares one.
pub(super) fn scissor(scissor: ScissorRect) -> GlScissorRect {
    GlScissorRect {
        x: scissor.x,
        y: scissor.y,
        width: scissor.width,
        height: scissor.height,
    }
}

/// The raster state one closed artifact is installed with.
///
/// The artifact fixes everything but the viewport, the scissor and the sample
/// count, and each of those three is a property of the *pass* rather than of the
/// recipe, which is why they arrive as arguments and the rest does not.  So the
/// claim this function makes is "these seven fields are the same for all ten
/// artifacts", and the guard is [`topology`]'s match, which lists all ten and
/// has no wildcard arm -- a new artifact is a compile error there rather than a
/// recipe that silently draws as a triangle list.
pub(super) fn raster_state(
    kernel: RasterKernel,
    viewport: GlViewport,
    scissor: Option<GlScissorRect>,
    sample_count: u32,
) -> GlRasterState {
    GlRasterState {
        topology: topology(kernel),
        // Nothing in this family's closed set culls, tests depth or blends: the
        // artifacts are the renderer's validation slice, and a face that
        // disappeared under a cull mode would be indistinguishable from a
        // missing draw in exactly the evidence this slice exists to produce.
        cull_mode: GlCullMode::None,
        front_face: GlFrontFace::CounterClockwise,
        depth_stencil: None,
        color_targets: vec![GlColorTargetState {
            write_mask: 0x0f,
            blend: None,
        }],
        multisample: GlMultisampleState {
            sample_count,
            alpha_to_coverage_enabled: false,
            sample_mask: u32::MAX,
        },
        viewport,
        scissor,
        blend_constant: [0; 4],
    }
}

/// The primitive topology one closed artifact draws as.
///
/// Listed rather than wildcarded, so that the fact is a claim about the whole
/// set and a new artifact has to decide rather than inherit.
fn topology(kernel: RasterKernel) -> GlPrimitiveTopology {
    match kernel {
        RasterKernel::Triangle
        | RasterKernel::IndexedPositionColor
        | RasterKernel::IndexedPositionFloat32x3
        | RasterKernel::IndexedPositionFloat32x3CameraMaterial
        | RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
        | RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
        | RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        | RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
        | RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
        | RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor => {
            GlPrimitiveTopology::Triangles
        }
    }
}

/// Whether one closed artifact takes its vertices from an index buffer.
///
/// The one non-indexed artifact is the bare triangle; the other nine are meshes.
/// Like [`topology`], this lists the set rather than testing a property, because
/// the identity table carries no such flag and a guessed one would be a second
/// copy of the artifact table.
pub(super) fn indexed(kernel: RasterKernel) -> bool {
    !matches!(kernel, RasterKernel::Triangle)
}

/// The vertex input one draw binds, from what the frame recorded.
///
/// The recipe decides what may be bound and not the frame: the layout is the
/// artifact's own, so a slot the artifact does not declare is refused rather
/// than bound to nothing, and a slot it does declare but the frame never set is
/// refused rather than left reading whatever the array held.  That second check
/// is the one a driver would not make -- a missing attribute is a legal array
/// with its own default, so a draw that forgot a buffer would render rather than
/// fail.
///
/// `operation` is the verb that asked, and it is a parameter rather than a
/// constant here because three of the four refusals below are reachable from
/// both draw verbs: the check is about the artifact and the frame's record, and
/// neither of them says which verb was called.  A refusal naming `draw` for a
/// `draw-indexed` would send an operator to a call the frame never made.
pub(super) fn vertex_input(
    pipeline: &RasterPipeline,
    buffers: &[GlVertexBufferBinding],
    index: Option<GlIndexBinding>,
    operation: &'static str,
) -> Result<VertexInput, GlError> {
    let layout = pipeline.vertex_layout();
    for declared in &layout.buffers {
        if !buffers.iter().any(|bound| bound.slot == declared.slot) {
            return Err(malformed(
                operation,
                "the artifact reads a vertex slot the frame never bound",
            ));
        }
    }
    if buffers
        .iter()
        .any(|bound| !layout.buffers.iter().any(|d| d.slot == bound.slot))
    {
        return Err(malformed(
            operation,
            "the frame bound a vertex slot the artifact does not declare",
        ));
    }
    let kernel = pipeline.kernel();
    let index = match (index, indexed(kernel)) {
        (Some(binding), true) => Some(binding),
        (None, true) => {
            return Err(malformed(
                operation,
                "the artifact takes its vertices from an index buffer, and the frame bound none",
            ));
        }
        (Some(_), false) => {
            return Err(malformed(
                operation,
                "the artifact draws its vertices without an index buffer, and the frame bound one",
            ));
        }
        (None, false) => None,
    };
    Ok(VertexInput {
        layout: layout.clone(),
        bindings: buffers.to_vec(),
        index,
    })
}

/// The contract's index format, as Layer 1 declares one.
pub(super) fn index_format(format: IndexFormat) -> GlIndexFormat {
    match format {
        IndexFormat::Uint16 => GlIndexFormat::Uint16,
        IndexFormat::Uint32 => GlIndexFormat::Uint32,
    }
}

/// The two error constructors are shared with the verbs rather than local, so
/// that a refusal this file makes and a refusal the adapter's own verb makes for
/// the same fact read the same way.
use super::failure::{malformed, unsupported};
