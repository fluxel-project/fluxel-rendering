//! The objects the common contract asks this adapter to register.
//!
//! Responsibility: hold what a raster pipeline, a compute pipeline and a binding
//! set *are* on this family, once each, in the shape the call sites that receive
//! them read.  All three are descriptions, and none names a browser or platform
//! type -- which is the boundary's requirement and is checkable here rather than
//! promised: the two pipelines hold Layer 1 shader and vertex data, and
//! [`Bindings`] holds Layer 1 object identities, the same two identities the
//! contract already declares as this adapter's `Texture` and `Buffer`.
//!
//! Not owned here: the native objects a binding names (Layer 1 created them and
//! the frame resolves them), the linking that turns a
//! [`GlProgramDescriptor`](crate::webgl2::api::GlProgramDescriptor) into a
//! program, and the retention all three are handed out with (the registry mints
//! it, and the module documentation of [`retention`](super::retention) says why
//! it covers nothing).
//!
//! # Why a binding set names a [`Recipe`] and not a raster kernel
//!
//! `set_bindings` is one contract verb, shared by both families, so the value it
//! receives has to be able to describe either one -- and the two are validated
//! against different sources.  That difference is a fact about the two identity
//! types rather than a shortcut: a `RasterArtifactIdentity` carries its binding
//! numbers (`uniform_binding`, `texture_binding`, `binding_count`), while a
//! `ComputeArtifactIdentity` carries only a `binding_recipe_version`, so the
//! compute half reads its arrangement off
//! [`shader::compute_program`](super::super::shader::compute_program)'s layout.
//! The alternative was a per-kernel access table beside the body and the layout
//! already in `compat::shader`, which would be a third statement about the same
//! kernel.
//!
//! # Why a pipeline carries a recipe and not resolved ids
//!
//! The obvious shape for [`RasterPipeline`] -- and for [`ComputePipeline`],
//! which is built the same way for the same reason -- is the pair Layer 2's
//! `set_pipeline` takes: a `ProgramId` and the rest of the pipeline state,
//! resolved once at registration.  That shape is wrong here, and the reason is a
//! lifetime fact rather than a preference: **a resolved id is not stable across
//! calls.**  Layer 2's program cache is bounded and evicts, and eviction
//! *destroys* the program it removes (`state/pipeline/cache.rs`'s `retain`, which
//! destroys everything the budget pushed out).  A pipeline holding a `ProgramId`
//! would therefore name a destroyed object as soon as some later call linked a
//! different recipe and pushed this one out, and `set_pipeline` would hand a dead
//! name to the driver -- on a real context, at a draw or a dispatch, with nothing
//! in this crate able to notice.  The same applies to a vertex array, whose
//! domain caches on the same terms.
//!
//! So the objects hold what is stable: the recipe, and the lowered descriptor
//! derived from it.  Those are stable because they are *inputs* -- a pure
//! function of the kernel and the context's profile, which is why the two
//! constructors take both and need no device at all -- and because holding them
//! is what keeps a draw from re-lowering two shader sources per frame.
//! Resolving the ids is [`super`]'s job at the pass that installs the pipeline,
//! where `&mut` access to the machine and a place to honour Layer 2's own "the
//! caller owns this object" answer both exist.
//!
//! `Rc` rather than a plain struct for the same reason the retained path's seven
//! binding objects are `Arc`-shared records: the contract hands the object out
//! *owned* (`BoundRasterPipeline.physical`) from a registry that has to keep it,
//! so the registry needs a handle whose clone is not the object.

use std::rc::Rc;

use fluxel_rendergraph::{
    BindingResourceSemantic, BufferRange, BufferReadWriteUse, ResolvedBindingResource,
    TextureRange, TextureReadWriteUse, TextureWriteUse,
};

use crate::resource::{ComputeKernel, RasterKernel};
use crate::webgl2::api::{
    BufferId, GlFamilyProfile, GlLogicalBinding, GlProgramDescriptor, GlShaderResourceKind,
    GlStorageBufferUsage, GlStorageImageAccess, GlVertexLayout, TextureId,
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
/// [`shader::program`](super::super::shader::program) takes the fold as a fact
/// about the contract and the identity's `sampler_binding` keeps the number.
/// Nothing folds in a compute layout: its five artifacts declare a texture or an
/// image and then a storage block, and no sampler of their own.
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
    /// A shader storage buffer, which a dispatch binds at a storage binding
    /// point.
    ///
    /// The access is the one the artifact's declaration states and not the one
    /// the graph authorized: they are different facts, and [`Bindings::new`]
    /// refuses the pair when the second is weaker than the first.
    StorageBuffer {
        /// The recipe's own binding number.
        binding: u32,
        /// The physical buffer the frame resolved.
        buffer: BufferId,
        /// The range the graph authorized, which the bind must respect.
        range: BufferRange,
        /// The access the artifact's declaration states.
        usage: GlStorageBufferUsage,
    },
    /// A shader storage image, which a dispatch binds at an image unit.
    ///
    /// The level, the layer, the format and the sample count an image unit needs
    /// are *not* here: the first two come from the graph's range and the last two
    /// from the device's own record of what it created, and neither is available
    /// where this value is built.  [`super::encoder`] lowers them at the bind,
    /// which is why a slot holds the access and the range rather than a ready
    /// `GlStorageImageBinding`.
    StorageImage {
        /// The recipe's own binding number.
        binding: u32,
        /// The physical texture the frame resolved.
        texture: TextureId,
        /// The subresources the graph authorized.
        range: TextureRange,
        /// The access the artifact's declaration states.
        access: GlStorageImageAccess,
    },
}

/// Which fixed artifact a binding set belongs to.
///
/// [`Bindings`] holds one of these rather than a raster kernel because
/// `set_bindings` is one contract verb both families reach.  It is the *recipe*
/// and not the lowered descriptor: what a set is checked against at the pass is
/// which artifact it was resolved for, and a descriptor is a value two recipes
/// could share (the linear-clamp pair does).
///
/// `pub(crate)` rather than `pub(super)` because a caller outside this module
/// tree names it: the registry's `register_bindings` takes one, and that method
/// is `pub(crate)`.  The reach is unchanged in fact -- the module holding this is
/// private -- so the wider marker widens nothing and only stops the compiler
/// from calling the method's own signature private-in-public (E0446).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Recipe {
    /// One of the ten closed raster artifacts.
    Raster(RasterKernel),
    /// One of the five closed compute artifacts.
    Compute(ComputeKernel),
}

/// One registered compute pipeline: the recipe and the lowering it was built as.
///
/// The same shape and the same reason as [`RasterPipeline`], minus the vertex
/// layout -- a compute program has no vertex input, and a field that was always
/// empty would be a second place to look for one.
#[derive(Clone, Debug)]
pub(crate) struct ComputePipeline(Rc<ComputeRecipe>);

/// What a [`ComputePipeline`] shares between its handles.
#[derive(Debug)]
struct ComputeRecipe {
    kernel: ComputeKernel,
    descriptor: GlProgramDescriptor,
}

impl ComputePipeline {
    /// Lowers one fixed artifact for one context profile.
    ///
    /// The error is a profile whose shading language has no compute stage, which
    /// is a *narrowing* of the dialect rule rather than a hole in it -- see
    /// [`shader::UnsupportedComputeProfile`](super::super::shader::UnsupportedComputeProfile),
    /// which owns that argument and whose tests hold both directions of it.
    pub(super) fn new(
        kernel: ComputeKernel,
        profile: GlFamilyProfile,
    ) -> Result<Self, shader::UnsupportedComputeProfile> {
        Ok(Self(Rc::new(ComputeRecipe {
            kernel,
            descriptor: shader::compute_program(kernel, profile)?,
        })))
    }

    /// The fixed artifact this pipeline is.
    pub(super) fn kernel(&self) -> ComputeKernel {
        self.0.kernel
    }

    /// The lowered program descriptor, which Layer 2 links.
    pub(super) fn descriptor(&self) -> &GlProgramDescriptor {
        &self.0.descriptor
    }
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
    recipe: Recipe,
    slots: Vec<BindingSlot>,
}

impl Bindings {
    /// Validates one recipe against the resources a frame resolved for it.
    ///
    /// Which half runs is the recipe's answer, and the two halves read different
    /// sources for the arrangement they check against -- [`Recipe`] states why.
    /// What both share is the shape of the three checks, and each is a refusal
    /// rather than a coercion:
    ///
    /// 1. **Count.**  The artifact declares a fixed number of bindings and the
    ///    graph lists one resource per binding.  A different count means the set
    ///    id was registered against another recipe, or the graph's binding recipe
    ///    drifted from the artifact's.
    /// 2. **Kind at the artifact's own numbers.**  Each declared binding number
    ///    is read at its own index, and has to hold the kind the recipe declares
    ///    there: a buffer behind a storage block, a texture behind a texture or
    ///    an image.  This is the assumption a positional read rests on, so it is
    ///    checked in both directions rather than trusted -- and it is what makes
    ///    a mismatch a refusal instead of a bind of the wrong resource.
    /// 3. **Semantics.**  The contract asks the provider to use the resolved
    ///    semantic "rather than treating the complete physical allocation as
    ///    implicitly authorized", and this family has no writable form of a
    ///    read-only binding to offer a frame that authorized one, so the only
    ///    sound answer is the refusal.
    pub(super) fn new(
        recipe: Recipe,
        resources: &[ResolvedBindingResource<'_, TextureId, BufferId>],
    ) -> Result<Self, &'static str> {
        let slots = match recipe {
            Recipe::Raster(kernel) => raster_slots(kernel, resources)?,
            Recipe::Compute(kernel) => compute_slots(kernel, resources)?,
        };
        Ok(Self { recipe, slots })
    }

    /// The fixed artifact whose bindings these are.
    pub(super) fn recipe(&self) -> Recipe {
        self.recipe
    }

    /// The resolved logical bindings, in the artifact's own binding order.
    pub(super) fn slots(&self) -> &[BindingSlot] {
        &self.slots
    }
}

/// The slots of one raster artifact, read against the numbers its identity
/// records.
///
/// The sampler's WGSL binding is deliberately not read.  It has no logical
/// counterpart (the fold above), and reading it as though it did would make this
/// function require a second texture where the family has one.
fn raster_slots(
    kernel: RasterKernel,
    resources: &[ResolvedBindingResource<'_, TextureId, BufferId>],
) -> Result<Vec<BindingSlot>, &'static str> {
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
    Ok(slots)
}

/// The slots of one compute artifact, read against the lowering's own layout.
///
/// The arrangement comes from [`shader::compute_layout`](super::super::shader::compute_layout)
/// rather than from the identity, which is the difference [`Recipe`] documents:
/// a `ComputeArtifactIdentity` records a binding recipe *version* and no numbers,
/// and a second table here would be a third statement about the same kernel
/// beside the body and the layout.  The lowering's suite is what holds that
/// layout against the artifact's own WGSL.
///
/// Nothing folds in this half -- the five artifacts declare a texture or an
/// image and then a storage block, and no sampler of their own -- so the count
/// the graph sends is the number of bindings the layout declares.
fn compute_slots(
    kernel: ComputeKernel,
    resources: &[ResolvedBindingResource<'_, TextureId, BufferId>],
) -> Result<Vec<BindingSlot>, &'static str> {
    let layout = shader::compute_layout(kernel);
    if resources.len() != layout.bindings.len() {
        return Err(
            "the binding set names a different number of resources than the artifact declares",
        );
    }
    let mut slots = Vec::with_capacity(layout.bindings.len());
    for binding in &layout.bindings {
        let resolved = resources
            .get(binding.location.binding as usize)
            .ok_or("the artifact declares a binding the frame resolved no resource for")?;
        slots.push(compute_slot(binding, resolved)?);
    }
    Ok(slots)
}

/// One compute binding, checked against the resource the frame resolved for it.
fn compute_slot(
    binding: &GlLogicalBinding,
    resolved: &ResolvedBindingResource<'_, TextureId, BufferId>,
) -> Result<BindingSlot, &'static str> {
    let number = binding.location.binding;
    match binding.kind {
        GlShaderResourceKind::StorageBuffer(usage) => {
            let ResolvedBindingResource::Buffer {
                physical,
                range,
                semantic,
            } = resolved
            else {
                return Err(
                    "the artifact declares a storage buffer at this binding, but the frame resolved another kind of resource there",
                );
            };
            if !permits_storage_buffer(*semantic, usage) {
                return Err(
                    "a storage buffer is bound at a storage binding point, so the resource behind it must be authorized for the access its declaration states",
                );
            }
            Ok(BindingSlot::StorageBuffer {
                binding: number,
                buffer: **physical,
                range: *range,
                usage,
            })
        }
        GlShaderResourceKind::StorageImage(access) => {
            let ResolvedBindingResource::Texture {
                physical,
                range,
                semantic,
            } = resolved
            else {
                return Err(
                    "the artifact declares a storage image at this binding, but the frame resolved another kind of resource there",
                );
            };
            if !permits_storage_image(*semantic, access) {
                return Err(
                    "a storage image is bound at an image unit, so the resource behind it must be authorized for the access its declaration states",
                );
            }
            Ok(BindingSlot::StorageImage {
                binding: number,
                texture: **physical,
                range: *range,
                access,
            })
        }
        GlShaderResourceKind::CombinedTextureSampler => {
            let ResolvedBindingResource::Texture {
                physical,
                range,
                semantic,
            } = resolved
            else {
                return Err(
                    "the artifact samples a texture at this binding, but the frame resolved another kind of resource there",
                );
            };
            // The same check, and the same reason, as the raster half's: this is
            // the one logical binding kind both halves declare, so a rule that
            // differed between them would be indefensible.
            if !matches!(semantic, BindingResourceSemantic::TextureRead(_)) {
                return Err(
                    "a sampled texture is read, so the resource bound to it cannot be authorized for writing",
                );
            }
            Ok(BindingSlot::Texture {
                binding: number,
                texture: **physical,
                range: *range,
            })
        }
        // A compute layout declares none of these, and one that did would be the
        // drift the lowering's own suite exists to catch.  It is a refusal here
        // rather than a panic because what reaches this function is a frame's
        // input.
        GlShaderResourceKind::UniformBuffer
        | GlShaderResourceKind::Sampler
        | GlShaderResourceKind::Texture => {
            Err("the artifact declares a binding this family's compute path has no lowering for")
        }
    }
}

/// Whether a graph-authorized semantic permits the access a storage buffer's
/// declaration states.
///
/// The rule is stated once and applies to both storage kinds: **a write must
/// name the storage role, and a read need only be a read.**  A read is a read
/// because the ordering a graph tracks for it is the same question whatever its
/// sub-role -- what a use *is* comes from the compiler's declaration, so a
/// texture read at a sampled binding and one read at an image unit are the same
/// hazard -- while a write authorized as a copy destination is a different
/// command from a write a shader performs, and binding it at a storage point
/// would be a write this frame never declared.  The specific variants below are
/// where that distinction lives: `BufferWriteUse` and `TextureWriteUse` both
/// have a `CopyDestination`, and a storage binding is never one.
fn permits_storage_buffer(semantic: BindingResourceSemantic, usage: GlStorageBufferUsage) -> bool {
    match usage {
        GlStorageBufferUsage::ReadOnly => matches!(
            semantic,
            BindingResourceSemantic::BufferRead(_) | BindingResourceSemantic::BufferReadWrite(_)
        ),
        GlStorageBufferUsage::ReadWrite => matches!(
            semantic,
            BindingResourceSemantic::BufferReadWrite(BufferReadWriteUse::Storage)
        ),
    }
}

/// Whether a graph-authorized semantic permits the access a storage image's
/// declaration states, on [`permits_storage_buffer`]'s rule.
fn permits_storage_image(semantic: BindingResourceSemantic, access: GlStorageImageAccess) -> bool {
    match access {
        GlStorageImageAccess::ReadOnly => matches!(
            semantic,
            BindingResourceSemantic::TextureRead(_) | BindingResourceSemantic::TextureReadWrite(_)
        ),
        GlStorageImageAccess::WriteOnly => matches!(
            semantic,
            BindingResourceSemantic::TextureWrite(TextureWriteUse::Storage)
                | BindingResourceSemantic::TextureReadWrite(TextureReadWriteUse::Storage)
        ),
        GlStorageImageAccess::ReadWrite => matches!(
            semantic,
            BindingResourceSemantic::TextureReadWrite(TextureReadWriteUse::Storage)
        ),
    }
}
