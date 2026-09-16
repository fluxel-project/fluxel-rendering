//! The two objects the common contract asks this adapter to register.
//!
//! Responsibility: hold what a raster pipeline and a binding set *are* on this
//! family, once each, in the shape the two call sites that receive them read.
//! Both are descriptions, and neither names a browser or platform type -- which
//! is the boundary's requirement and is checkable here rather than promised:
//! [`RasterPipeline`] holds Layer 1 shader and vertex data, and [`Bindings`]
//! holds Layer 1 object identities, the same two identities the contract already
//! declares as this adapter's `Texture` and `Buffer`.
//!
//! Not owned here: the native objects a binding names (Layer 1 created them and
//! the frame resolves them), the linking that turns a
//! [`GlProgramDescriptor`](crate::webgl2::api::GlProgramDescriptor) into a
//! program, and the retention both are handed out with (the registry mints it,
//! and the module documentation of
//! [`retention`](super::retention) says why it covers nothing).
//!
//! # Why the pipeline carries a recipe and not resolved ids
//!
//! The obvious shape for [`RasterPipeline`] is the triple Layer 2's
//! `set_pipeline` takes -- a `ProgramId`, a `VertexArrayId` and a
//! `GlRasterState` -- resolved once, at registration.  That shape is wrong here,
//! and the reason is a lifetime fact rather than a preference: **a resolved id
//! is not stable across calls.**  Layer 2's program cache is bounded and evicts,
//! and eviction *destroys* the program it removes
//! (`state/pipeline/cache.rs`'s `retain`, which destroys everything the budget
//! pushed out).  A `RasterPipeline` holding a `ProgramId` would therefore name a
//! destroyed object as soon as some later call linked a different recipe and
//! pushed this one out, and `set_pipeline` would hand a dead name to the driver
//! -- on a real context, at a draw, with nothing in this crate able to notice.
//! The same applies to a vertex array, whose domain caches on the same terms.
//!
//! So the object holds what is stable: the recipe, and the lowered descriptor
//! and vertex layout derived from it.  Those are stable because they are
//! *inputs* -- a pure function of the kernel and the context's profile, which is
//! why [`RasterPipeline::new`] takes both and needs no device at all -- and
//! because holding them is what keeps a draw from re-lowering two shader sources
//! per frame.  Resolving the ids is [`super`]'s job at the pass that installs the
//! pipeline, where `&mut` access to the machine and a place to honour Layer 2's
//! own "the caller owns this object" answer both exist.
//!
//! `Rc` rather than a plain struct for the same reason the retained path's seven
//! binding objects are `Arc`-shared records: the contract hands the object out
//! *owned* (`BoundRasterPipeline.physical`) from a registry that has to keep it,
//! so the registry needs a handle whose clone is not the object.

use std::rc::Rc;

use fluxel_rendergraph::{
    BindingResourceSemantic, BufferRange, ResolvedBindingResource, TextureRange,
};

use crate::resource::RasterKernel;
use crate::webgl2::api::{
    BufferId, GlFamilyProfile, GlProgramDescriptor, GlVertexLayout, TextureId,
};

use super::super::shader;

/// One registered raster pipeline: the recipe and the lowering it was built as.
///
/// `Clone` is a handle clone and not a copy of the recipe, which is what lets
/// the registry both keep it and hand it out.
#[derive(Clone, Debug)]
pub(crate) struct RasterPipeline(Rc<RasterRecipe>);

/// What a [`RasterPipeline`] shares between its handles.
#[derive(Debug)]
struct RasterRecipe {
    kernel: RasterKernel,
    descriptor: GlProgramDescriptor,
    vertex_layout: GlVertexLayout,
}

impl RasterPipeline {
    /// Lowers one fixed artifact for one context profile.
    ///
    /// The profile is a parameter rather than a field read from a machine,
    /// because lowering needs nothing else: the two things it produces are
    /// functions of the kernel and the profile alone, and a caller that had to
    /// borrow a device to build a description would be able to do it only inside
    /// a frame.  The error is a profile this family has no dialect for, which is
    /// the same set Layer 1's own dialect rule refuses -- `shader::program`
    /// owns that agreement and its tests hold both directions of it.
    ///
    /// The vertex layout is derived here rather than in the registry because it
    /// cannot fail: it is a pure match on the recipe's own identity table, and
    /// the one validation it could be checked against
    /// (`GlVertexLayout::validate`) is a property of the recipe that the
    /// lowering's own suite already holds for all ten artifacts.
    pub(super) fn new(
        kernel: RasterKernel,
        profile: GlFamilyProfile,
    ) -> Result<Self, shader::UnsupportedProfile> {
        Ok(Self(Rc::new(RasterRecipe {
            kernel,
            descriptor: shader::program(kernel, profile)?,
            vertex_layout: shader::vertex_layout(kernel),
        })))
    }

    /// The fixed artifact this pipeline is.
    pub(super) fn kernel(&self) -> RasterKernel {
        self.0.kernel
    }

    /// The lowered program descriptor, which Layer 2 links.
    pub(super) fn descriptor(&self) -> &GlProgramDescriptor {
        &self.0.descriptor
    }

    /// The vertex input shape, which Layer 2 derives the vertex array from.
    pub(super) fn vertex_layout(&self) -> &GlVertexLayout {
        &self.0.vertex_layout
    }
}

/// One logical binding of a registered set, and the physical resource it names.
///
/// One entry per *logical* binding, which is not one per WGSL binding: a GLSL
/// `sampler2D` is a texture and its sampler at one unit, so the linear-clamp
/// recipe's two WGSL entries are one entry here.  Which WGSL binding a sampler
/// arrived on is therefore not representable, which is exactly why
/// [`shader::layout`] takes the fold as a fact about the contract and the
/// identity's `sampler_binding` keeps the number.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BindingSlot {
    /// The frame uniform block, which a draw binds as a uniform buffer.
    Uniform {
        /// The recipe's own binding number, kept so a diagnostic can name it.
        binding: u32,
        /// The physical buffer the frame resolved.
        buffer: BufferId,
        /// The range the graph authorized, which the bind must respect.
        range: BufferRange,
    },
    /// A sampled texture, which a draw binds at a texture unit.
    Texture {
        /// The recipe's own binding number.
        binding: u32,
        /// The physical texture the frame resolved.
        texture: TextureId,
        /// The subresources the graph authorized.
        range: TextureRange,
    },
}

/// One registered binding set, resolved against a frame's resources.
///
/// Built per resolution rather than registered once, because it names physical
/// objects that differ frame to frame: what the registry holds is the *recipe*
/// a set id stands for, and what the recorder receives is this.  It is the
/// validation the contract asks a provider for -- "responsible for validating
/// its opaque recipe against that information and its native pipeline metadata"
/// -- and it is answered here, where both halves are in hand, rather than left
/// to the pass that binds it.
#[derive(Debug)]
pub(crate) struct Bindings {
    kernel: RasterKernel,
    slots: Vec<BindingSlot>,
}

impl Bindings {
    /// Validates one recipe against the resources a frame resolved for it.
    ///
    /// The three checks are the three ways a recipe and its resolution can
    /// disagree, and each is a refusal rather than a coercion.
    ///
    /// 1. **Count.**  The identity's `binding_count` is how many WGSL bindings
    ///    the artifact declares, and the graph lists one resource per binding.
    ///    A different count means the set id was registered against another
    ///    recipe, or the graph's binding recipe drifted from the artifact's.
    /// 2. **Kind at the recipe's own numbers.**  Each of the artifact's three
    ///    binding numbers is read at its own index, and has to be the kind the
    ///    identity records there: a buffer behind `uniform_binding`, a texture
    ///    behind `texture_binding`.  This is the assumption a positional read
    ///    rests on, so it is checked in both directions rather than trusted --
    ///    and it is what makes a mismatch a refusal instead of a bind of the
    ///    wrong resource.
    /// 3. **Semantics.**  A uniform block is read and a sampled texture is read,
    ///    so a resolved resource authorized for writing cannot be bound as one.
    ///    The contract asks the provider to use the resolved semantic "rather
    ///    than treating the complete physical allocation as implicitly
    ///    authorized", and this family has no writable form of either binding to
    ///    offer it, so the only sound answer is the refusal.
    ///
    /// The sampler's WGSL binding is deliberately not read.  It has no logical
    /// counterpart (the fold above), and reading it as though it did would make
    /// this function require a second texture where the family has one.
    pub(super) fn new(
        kernel: RasterKernel,
        resources: &[ResolvedBindingResource<'_, TextureId, BufferId>],
    ) -> Result<Self, &'static str> {
        let identity = kernel.portable_identity();
        if resources.len() as u32 != identity.binding_count {
            return Err(
                "the binding set names a different number of resources than the artifact declares",
            );
        }
        let mut slots = Vec::new();
        if let Some(binding) = identity.uniform_binding {
            let Some(ResolvedBindingResource::Buffer {
                physical,
                range,
                semantic,
            }) = resources.get(binding as usize)
            else {
                return Err(
                    "the artifact reads a uniform block at this binding, but the frame resolved another kind of resource there",
                );
            };
            if !matches!(semantic, BindingResourceSemantic::BufferRead(_)) {
                return Err(
                    "a uniform block is read, so the resource bound to it cannot be authorized for writing",
                );
            }
            slots.push(BindingSlot::Uniform {
                binding,
                buffer: **physical,
                range: *range,
            });
        }
        if let Some(binding) = identity.texture_binding {
            let Some(ResolvedBindingResource::Texture {
                physical,
                range,
                semantic,
            }) = resources.get(binding as usize)
            else {
                return Err(
                    "the artifact samples a texture at this binding, but the frame resolved another kind of resource there",
                );
            };
            if !matches!(semantic, BindingResourceSemantic::TextureRead(_)) {
                return Err(
                    "a sampled texture is read, so the resource bound to it cannot be authorized for writing",
                );
            }
            slots.push(BindingSlot::Texture {
                binding,
                texture: **physical,
                range: *range,
            });
        }
        Ok(Self { kernel, slots })
    }

    /// The fixed artifact whose bindings these are.
    pub(super) fn kernel(&self) -> RasterKernel {
        self.kernel
    }

    /// The resolved logical bindings, in the artifact's own binding order.
    pub(super) fn slots(&self) -> &[BindingSlot] {
        &self.slots
    }
}
