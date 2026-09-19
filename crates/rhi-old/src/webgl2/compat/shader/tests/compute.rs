//! Tests for the five compute artifacts' lowering.
//!
//! The same limit the raster tests state applies here and is not restated at
//! length: the mock links nothing, so what these measure is structure.  The two
//! seams they hold are the ones the raster suite holds, transposed to this
//! family -- the artifact's own WGSL, which is where the binding arrangement and
//! the workgroup shape come from, and the GLSL bodies, which are what a driver
//! will read.  Every assertion compares those two against each other rather than
//! restating either.

use super::super::compute::compute_program;
use super::super::text;
use crate::resource::ComputeKernel;
use crate::webgl2::api::{
    GlFamilyProfile, GlProgramKind, GlShaderDialect, GlShaderResourceKind, GlShaderStage,
    GlStorageBufferUsage, GlStorageImageAccess,
};

/// The five closed recipes, in the order the artifact module declares them.
///
/// A new variant fails `body()`'s match in the lowering, which has no wildcard
/// arm and is therefore a compile error; this list is what lets the cross-checks
/// below cover all of them.
const KERNELS: [ComputeKernel; 5] = [
    ComputeKernel::WrappingAdd,
    ComputeKernel::WrappingMultiply,
    ComputeKernel::TexturePackRgba8,
    ComputeKernel::TextureStoreRgba8,
    ComputeKernel::TextureLoadRgba8,
];

/// The profiles compute is admitted on, and the dialect each must select.
///
/// The dialect is written out rather than derived from the version, because the
/// two families number versions differently and the whole point of the check is
/// that this lowering picks the one Layer 1's rule would: a desktop profile is
/// not lowered as an embedded version of the same number, and a table that
/// assumed it could would pass while every desktop link failed.
const ADMITTED: [(GlFamilyProfile, GlShaderDialect); 4] = [
    (
        GlFamilyProfile::Embedded { major: 3, minor: 1 },
        GlShaderDialect::Embedded { version: 310 },
    ),
    (
        GlFamilyProfile::Embedded { major: 3, minor: 2 },
        GlShaderDialect::Embedded { version: 320 },
    ),
    (
        GlFamilyProfile::Desktop { major: 4, minor: 3 },
        GlShaderDialect::Desktop { version: 430 },
    ),
    (
        GlFamilyProfile::Desktop { major: 4, minor: 5 },
        GlShaderDialect::Desktop { version: 450 },
    ),
];

/// The profiles the dialect rule admits and this family must still refuse.
///
/// Every one of these is a profile `dialect_for` answers for, which is what
/// makes them the interesting set: the refusal is a narrowing of an answer the
/// parent gives, not a gap in it.  A profile the parent itself refuses is not
/// listed, because `every_refused_profile_is_one_the_dialect_rule_answers_for`
/// would then be testing something else.
const REFUSED: [GlFamilyProfile; 4] = [
    GlFamilyProfile::WebGl2,
    GlFamilyProfile::Embedded { major: 3, minor: 0 },
    GlFamilyProfile::Desktop { major: 4, minor: 0 },
    GlFamilyProfile::Desktop { major: 4, minor: 2 },
];

/// The declaration each storage or sampling kind opens with in the bodies.
///
/// The storage arms include the access qualifier, which is what makes this the
/// guard for the payload those two kinds carry: a layout that named the wrong
/// access would look for a declaration the body does not contain, and the
/// assertion fails naming the kernel and the binding.  Read-write is GLSL's
/// unqualified form in both places -- a `buffer` block with no qualifier is
/// read-write, and so is an `image2D` with neither `readonly` nor `writeonly` --
/// so those are the two arms with nothing in front of the keyword.
fn declaration_of(kind: GlShaderResourceKind) -> &'static str {
    match kind {
        GlShaderResourceKind::StorageBuffer(GlStorageBufferUsage::ReadWrite) => "buffer ",
        GlShaderResourceKind::StorageBuffer(GlStorageBufferUsage::ReadOnly) => "readonly buffer ",
        GlShaderResourceKind::CombinedTextureSampler => "uniform sampler2D ",
        GlShaderResourceKind::StorageImage(GlStorageImageAccess::ReadOnly) => {
            "readonly uniform image2D "
        }
        GlShaderResourceKind::StorageImage(GlStorageImageAccess::WriteOnly) => {
            "writeonly uniform image2D "
        }
        GlShaderResourceKind::StorageImage(GlStorageImageAccess::ReadWrite) => "uniform image2D ",
        // A compute layout declares none of these three, so a body reached with
        // one would be the drift this test exists to catch: the empty prefix
        // matches nothing and the assertion fails naming the binding.
        GlShaderResourceKind::UniformBuffer
        | GlShaderResourceKind::Sampler
        | GlShaderResourceKind::Texture => "",
    }
}

/// Every `@binding(n)` number one artifact's WGSL declares, in source order.
fn wgsl_binding_numbers(kernel: ComputeKernel) -> Vec<u32> {
    let source = kernel.wgsl_source();
    let mut numbers = Vec::new();
    let mut rest = source;
    while let Some(start) = rest.find("@binding(") {
        let after = &rest[start + "@binding(".len()..];
        let end = after.find(')').expect("an unclosed binding attribute");
        numbers.push(after[..end].parse().expect("a non-numeric binding"));
        rest = &after[end..];
    }
    numbers
}

#[test]
fn every_profile_that_admits_compute_lowers_and_every_other_is_refused() {
    for kernel in KERNELS {
        for (profile, dialect) in ADMITTED {
            let descriptor = compute_program(kernel, profile)
                .unwrap_or_else(|_| panic!("{kernel:?} must lower for {profile:?}"));
            let GlProgramKind::Compute { shader } = &descriptor.kind else {
                panic!("{kernel:?} lowered to a program that is not a compute program");
            };
            assert_eq!(shader.stage, GlShaderStage::Compute, "{kernel:?}");
            assert_eq!(
                shader.dialect, dialect,
                "{kernel:?} must select the dialect {profile:?} requires"
            );
            // The lowering's answer and Layer 1's own profile rule have to
            // agree: a descriptor this admits and the provider then refuses for
            // its dialect would be a link-time failure on a real context and
            // nowhere else.
            assert!(
                descriptor.validate_for(profile).is_ok(),
                "{kernel:?} lowered to a descriptor Layer 1 refuses for {profile:?}"
            );
        }
        for profile in REFUSED {
            assert!(
                compute_program(kernel, profile).is_err(),
                "{kernel:?} must not lower for {profile:?}"
            );
        }
    }
}

#[test]
fn every_refused_profile_is_one_the_dialect_rule_answers_for() {
    // The point of the narrowing, stated as a test rather than as a comment: a
    // profile this family refuses is one the parent answers for, so the refusal
    // is compute's own judgement about the stage and not a hole in the dialect
    // rule.  If the parent ever stopped admitting one of these, `REFUSED` would
    // contain a profile refused for the wrong reason and this would say so.
    for profile in REFUSED {
        let raster_kernel = crate::resource::RasterKernel::Triangle;
        assert!(
            super::super::raster::program(raster_kernel, profile).is_ok(),
            "{profile:?} is refused by compute and by the dialect rule, so it is not evidence"
        );
    }
}

#[test]
fn the_layout_is_the_arrangement_the_wgsl_declares() {
    for kernel in KERNELS {
        let descriptor = compute_program(kernel, ADMITTED[0].0).expect("an admitted profile");
        let numbers: Vec<u32> = descriptor
            .layout
            .bindings
            .iter()
            .map(|binding| binding.location.binding)
            .collect();
        // The arrangement is the lowering's reading of the artifact's binding
        // recipe version, and the WGSL is the only place that recipe is written
        // down.  So this is the assertion that keeps the two from drifting.
        assert_eq!(
            numbers,
            wgsl_binding_numbers(kernel),
            "{kernel:?} lays out bindings the WGSL does not declare"
        );
        for binding in &descriptor.layout.bindings {
            assert_eq!(binding.location.group, 0, "{kernel:?}");
            assert_eq!(binding.array_count, 1, "{kernel:?}");
        }
    }
}

#[test]
fn every_declared_name_is_one_the_body_declares() {
    for kernel in KERNELS {
        let descriptor = compute_program(kernel, ADMITTED[0].0).expect("an admitted profile");
        let GlProgramKind::Compute { shader } = &descriptor.kind else {
            panic!("{kernel:?} lowered to a program that is not a compute program");
        };
        // A provider reflects a storage binding by the name the linked program
        // reports, so a layout naming an identifier the text does not declare
        // would fail on a real context and nowhere else.
        for binding in &descriptor.layout.bindings {
            let declaration = format!("{}{}", declaration_of(binding.kind), binding.name);
            assert!(
                shader.text.contains(&declaration),
                "{kernel:?} names {:?} in its layout but declares no `{declaration}`",
                binding.name
            );
        }
    }
}

#[test]
fn the_local_size_declaration_is_the_shape_the_artifact_records() {
    for kernel in KERNELS {
        let descriptor = compute_program(kernel, ADMITTED[0].0).expect("an admitted profile");
        let GlProgramKind::Compute { shader } = &descriptor.kind else {
            panic!("{kernel:?} lowered to a program that is not a compute program");
        };
        let [x, y, z] = kernel.workgroup_size();
        let declared =
            format!("layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;");
        assert!(
            shader.text.contains(&declared),
            "{kernel:?} must declare `{declared}`"
        );
    }
}

#[test]
fn the_two_arithmetic_bodies_differ_in_one_expression() {
    // The pair is two full bodies rather than one parameterized template,
    // because composing a shader through `format!` would double every brace and
    // stop the text being readable beside the WGSL.  What that costs is two
    // copies that can drift, and this is what holds them: the difference must be
    // exactly the operator and its operand, so an edit to one body alone fails
    // here instead of producing two artifacts that compute the same thing.
    assert_eq!(
        text::COMPUTE_WRAPPING_ADD.replace("+ 1u", "* 3u"),
        text::COMPUTE_WRAPPING_MULTIPLY
    );
    assert_ne!(text::COMPUTE_WRAPPING_ADD, text::COMPUTE_WRAPPING_MULTIPLY);
}

#[test]
fn the_two_packing_bodies_differ_only_in_where_the_texel_came_from() {
    // The same guard for the other near-pair: one fetches a sampled texture and
    // the other reads a storage image, and the arithmetic after that is
    // identical because the WGSL originals are.  The declarations differ too --
    // a storage block in both, and a sampler in one against an image in the
    // other -- so the guard is on the part below them.
    let tail = |body: &str| {
        body.split_once("uint r = uint(round(")
            .expect("the packing expression")
            .1
            .to_owned()
    };
    assert_eq!(
        tail(text::COMPUTE_TEXTURE_PACK_RGBA8),
        tail(text::COMPUTE_TEXTURE_LOAD_RGBA8)
    );
}

#[test]
fn the_same_kernel_lowers_to_the_same_bytes_twice() {
    for kernel in KERNELS {
        let first = compute_program(kernel, ADMITTED[0].0).expect("an admitted profile");
        let second = compute_program(kernel, ADMITTED[0].0).expect("an admitted profile");
        // The descriptor is what Layer 2's program cache keys on, so two
        // lowerings of one recipe have to be one value -- a hash of the text is
        // part of it, and a body built from an unordered source would make the
        // cache miss forever without ever failing.
        assert_eq!(first, second, "{kernel:?}");
    }
}
