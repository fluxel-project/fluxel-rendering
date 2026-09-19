//! The five closed compute artifacts, lowered onto Layer 1's shader contract.
//!
//! Responsibility: turn one [`ComputeKernel`] and the profile of the context it
//! will run on into the [`GlProgramDescriptor`] a provider accepts -- the
//! program's single compute stage, its declared local workgroup shape, and its
//! logical binding layout.  The dialect rule and the source composition are
//! [`super`]'s, because the raster lowering needs the same two and neither
//! family decides them.
//!
//! Not owned here: the native resources a binding names (the adapter's binding
//! factories do that), the dispatch dimensions a call supplies, and the pass a
//! dispatch happens in (the session's).
//!
//! # This family narrows the dialect rule rather than inheriting it
//!
//! GLSL ES 300 has no compute stage at all, so the two profiles [`super`]'s
//! dialect rule admits most readily -- the WebGL2 profile and embedded 3.0 --
//! are refused here, and so are desktop versions below 4.30.  That is what
//! [`UnsupportedComputeProfile`] means, and it is a *narrowing*: every profile
//! this module admits is one the dialect rule admits too, so a descriptor from
//! here cannot be refused later for its dialect.
//!
//! # What the binding names are, and why they are not the WGSL ones
//!
//! A storage binding is reflected **by name**, because a provider cannot
//! enumerate a linked program's storage blocks the way it enumerates its
//! uniform blocks -- see the two reflection loops this is lowered for.  So each
//! name here has to be the identifier the GLSL actually declares, which for a
//! storage block is the *block* name: GLSL has no instance name in these
//! declarations, so the block name is the only name a program reports.
//!
//! The member keeps the WGSL resource's own spelling (`values`,
//! `destination`), and the block takes the PascalCase form the frame uniform
//! block already uses (`FrameUniforms`), so a reader comparing a body against
//! its WGSL original sees the same words in the same places.  Two kernels share
//! the block name `Destination` and they never share a program, so nothing is
//! ambiguous.

use crate::resource::ComputeKernel;
use crate::webgl2::api::{
    GlBindingLocation, GlFamilyProfile, GlLogicalBinding, GlPipelineLayout, GlProgramDescriptor,
    GlProgramKind, GlShaderResourceKind, GlShaderStage, GlStorageBufferUsage, GlStorageImageAccess,
};

use super::text;
use super::{UnsupportedComputeProfile, dialect_for, source};

/// Lowers one fixed compute artifact for one context profile.
///
/// The error case is a profile that cannot run a compute stage, and the set is
/// stated in [`supports_compute`] rather than inferred from a version number
/// here.
///
/// The entry points here are `pub(in crate::webgl2::compat)` rather than
/// `pub(super)`, for [`super::raster`]'s reason: the module that *composes* them
/// is [`super`] and the layer that consumes them is `compat`, which is where
/// they were visible when the parent defined them.
pub(in crate::webgl2::compat) fn compute_program(
    kernel: ComputeKernel,
    profile: GlFamilyProfile,
) -> Result<GlProgramDescriptor, UnsupportedComputeProfile> {
    let (dialect, header) = supports_compute(profile).ok_or(UnsupportedComputeProfile)?;
    let shader = source(
        GlShaderStage::Compute,
        dialect,
        format!("{header}{}", body(kernel)),
    );
    Ok(GlProgramDescriptor {
        kind: GlProgramKind::Compute { shader },
        layout: layout(kernel),
        debug_name: Some(format!("fixed-compute:{}", kernel.entry_point())),
    })
}

/// The dialect a profile must have for a compute stage to exist in it.
///
/// Expressed as a filter over [`dialect_for`]'s answer rather than as a second
/// table, so that the two cannot disagree: a profile that function has no
/// dialect for is refused here for the same reason and with the same result.
///
/// Both embedded versions that admit compute do so through the local workgroup
/// shape, which ES 310 added; desktop 4.30 is where the same shape arrives on
/// that family.  Versions above these are admitted rather than enumerated,
/// because the stage and the syntax this module uses are unchanged by them.
fn supports_compute(profile: GlFamilyProfile) -> Option<(super::GlShaderDialect, String)> {
    let (dialect, header) = dialect_for(profile)?;
    let admitted = match profile {
        GlFamilyProfile::Embedded { major: 3, minor } => minor >= 1,
        GlFamilyProfile::Desktop { major: 4, minor } => minor >= 3,
        _ => false,
    };
    admitted.then_some((dialect, header))
}

/// The logical bindings one fixed artifact declares, in program order.
///
/// The arrangement is this module's to state and not the artifact's: a
/// [`ComputeKernel`]'s identity carries a `binding_recipe_version` and no
/// enumeration, so the numbers below are the lowering's reading of that version
/// rather than a copy of anything.  What ties the two together is the WGSL the
/// artifact also carries, which is where these numbers come from -- group 0,
/// the sampler or image first, the storage buffer second -- and
/// `the_layout_is_the_arrangement_the_wgsl_declares` is what holds them there.
///
/// Each storage kind carries the access its body declares, and the two
/// arithmetic bodies declare none: a GLSL `buffer` block with no qualifier is
/// read-write, which is what those kernels do with it.  The two image bodies are
/// explicit (`writeonly` and `readonly`), and the access is stated here rather
/// than parsed out of the text because the text is the lowering's output and
/// this is its input.
pub(in crate::webgl2::compat) fn layout(kernel: ComputeKernel) -> GlPipelineLayout {
    let binding = |name: &str, number: u32, kind: GlShaderResourceKind| GlLogicalBinding {
        name: name.to_owned(),
        location: GlBindingLocation {
            group: 0,
            binding: number,
        },
        kind,
        array_count: 1,
    };
    let storage = |name: &str, number: u32| {
        binding(
            name,
            number,
            GlShaderResourceKind::StorageBuffer(GlStorageBufferUsage::ReadWrite),
        )
    };
    let image = |name: &str, number: u32, access: GlStorageImageAccess| {
        binding(name, number, GlShaderResourceKind::StorageImage(access))
    };
    let bindings = match kernel {
        ComputeKernel::WrappingAdd | ComputeKernel::WrappingMultiply => {
            vec![storage(text::COMPUTE_VALUES_BLOCK, 0)]
        }
        ComputeKernel::TexturePackRgba8 => vec![
            binding(
                text::COMPUTE_SOURCE_TEXTURE,
                0,
                GlShaderResourceKind::CombinedTextureSampler,
            ),
            storage(text::COMPUTE_DESTINATION_BLOCK, 1),
        ],
        ComputeKernel::TextureStoreRgba8 => vec![image(
            text::COMPUTE_OUTPUT_IMAGE,
            0,
            GlStorageImageAccess::WriteOnly,
        )],
        ComputeKernel::TextureLoadRgba8 => vec![
            image(
                text::COMPUTE_SOURCE_IMAGE,
                0,
                GlStorageImageAccess::ReadOnly,
            ),
            storage(text::COMPUTE_DESTINATION_BLOCK, 1),
        ],
    };
    GlPipelineLayout { bindings }
}

/// The one stage body a fixed artifact is built from.
fn body(kernel: ComputeKernel) -> String {
    let local_size = local_size_declaration(kernel);
    let source = match kernel {
        ComputeKernel::WrappingAdd => text::COMPUTE_WRAPPING_ADD,
        ComputeKernel::WrappingMultiply => text::COMPUTE_WRAPPING_MULTIPLY,
        ComputeKernel::TexturePackRgba8 => text::COMPUTE_TEXTURE_PACK_RGBA8,
        ComputeKernel::TextureStoreRgba8 => text::COMPUTE_TEXTURE_STORE_RGBA8,
        ComputeKernel::TextureLoadRgba8 => text::COMPUTE_TEXTURE_LOAD_RGBA8,
    };
    format!("{local_size}{source}")
}

/// The workgroup shape one artifact declares, as GLSL writes it.
///
/// The numbers come from the artifact's own identity rather than from numbers
/// written here, so a recipe whose shape changed would change this with it --
/// and the shape is a declaration the driver reads, not a hint: a body that
/// indexed past it would be reading work no invocation was given.
///
/// The `z` axis is always written even where it is one, because the WGSL
/// origins all state three axes and a reader comparing the two should not have
/// to know which of them defaulted.
fn local_size_declaration(kernel: ComputeKernel) -> String {
    let [x, y, z] = kernel.workgroup_size();
    format!("layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\n")
}
