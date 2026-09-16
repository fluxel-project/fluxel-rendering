//! Tests for the fixed artifacts' lowering.
//!
//! Every test here is *structural*, and the module doc says why that is the
//! whole of what can be measured without a GL context: the mock provider links
//! nothing, so a suite that passes here has shown that the lowering produces the
//! right shape for the right profile and has shown nothing at all about whether
//! the GLSL is valid or computes what its WGSL original computes.  Those are
//! hardware-evidence questions and are recorded as open in the series plan.
//!
//! What the tests are for, given that limit, is the seam between two tables
//! that must not drift: the artifact identity (`crate::resource`, the retained
//! path's own record of what each recipe is) and the GLSL bodies ([`text`]).
//! Every assertion below compares one against the other rather than restating
//! either, so a recipe whose bindings or streams changed would fail here instead
//! of on a device.
//!
//! The same seam runs over the other five recipes in [`compute`], which is a
//! module of its own because those tests name none of the raster vocabulary.

use super::*;

// The compute family's cross-checks, in a module of their own because they hold
// the other half of the same seam -- the artifact's WGSL against the body --
// over a set of recipes no raster test names.
mod compute;

// The names the lowering's own contract does not carry, imported here rather
// than reached through [`super`]: the parent composes the two families and no
// longer names a raster kernel or a pipeline layout itself, so a glob over it
// stopped being a source for these when the families were split out.
use crate::resource::{RasterKernel, RasterVertexLayout};
use crate::webgl2::api::{GlPipelineLayout, GlProgramKind, GlShaderResourceKind};

/// The ten closed recipes, in the order the artifact module declares them.
///
/// A new variant does not fail this list -- it fails `bodies()`, which is a
/// match with no wildcard arm and therefore a compile error.  That is the real
/// guard; this list is what lets the cross-checks below cover all of them.
const KERNELS: [RasterKernel; 10] = [
    RasterKernel::Triangle,
    RasterKernel::IndexedPositionColor,
    RasterKernel::IndexedPositionFloat32x3,
    RasterKernel::IndexedPositionFloat32x3CameraMaterial,
    RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture,
    RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv,
    RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
    RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
    RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert,
    RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor,
];

/// Every profile this family lowers for, one per dialect it can emit.
const ADMITTED: [GlFamilyProfile; 7] = [
    GlFamilyProfile::WebGl2,
    GlFamilyProfile::Embedded { major: 3, minor: 0 },
    GlFamilyProfile::Embedded { major: 3, minor: 1 },
    GlFamilyProfile::Embedded { major: 3, minor: 2 },
    GlFamilyProfile::Desktop { major: 4, minor: 0 },
    GlFamilyProfile::Desktop { major: 4, minor: 3 },
    GlFamilyProfile::Desktop { major: 4, minor: 6 },
];

/// Profiles whose core version is below every dialect this family writes.
///
/// Layer 1 admits no profile below GLSL ES 3.00 or desktop GLSL 4.00, and the
/// lowering has to agree with it rather than guess.
const REFUSED: [GlFamilyProfile; 4] = [
    GlFamilyProfile::Embedded { major: 2, minor: 0 },
    GlFamilyProfile::Embedded { major: 3, minor: 3 },
    GlFamilyProfile::Desktop { major: 3, minor: 3 },
    GlFamilyProfile::Desktop { major: 2, minor: 1 },
];

/// One recipe's lowered raster program.
fn raster(
    kernel: RasterKernel,
    profile: GlFamilyProfile,
) -> (GlShaderSource, GlShaderSource, GlPipelineLayout) {
    let descriptor =
        program(kernel, profile).expect("this profile is one the lowering answers for");
    match descriptor.kind {
        GlProgramKind::Raster { vertex, fragment } => (vertex, fragment, descriptor.layout),
        GlProgramKind::Compute { .. } => {
            panic!("a fixed raster artifact lowers to a raster program, never a compute one")
        }
    }
}

#[test]
fn every_kernel_lowers_to_a_program_layer_one_accepts_on_every_admitted_profile() {
    for profile in ADMITTED {
        for kernel in KERNELS {
            let descriptor = program(kernel, profile).expect("an admitted profile is lowered");
            descriptor.validate_for(profile).unwrap_or_else(|error| {
                panic!("{kernel:?} on {profile:?} is refused by layer 1: {error:?}")
            });
        }
    }
}

#[test]
fn a_refused_profile_is_refused_before_a_program_exists_and_layer_one_refuses_it_too() {
    // Both halves matter.  The first says the lowering does not answer for a
    // profile it has no dialect for; the second says the set it answers for is
    // the set Layer 1 accepts, so the two tables cannot drift into disagreement
    // about which contexts are supported.
    let probe = GlShaderSource {
        stage: GlShaderStage::Vertex,
        dialect: GlShaderDialect::Embedded { version: 300 },
        entry_point: ENTRY_POINT.to_owned(),
        source_hash: content_hash("void main() {}"),
        text: "void main() {}".to_owned(),
        debug_name: None,
    };
    for profile in REFUSED {
        for kernel in KERNELS {
            assert_eq!(
                program(kernel, profile).err(),
                Some(UnsupportedProfile),
                "{kernel:?} has no lowering for {profile:?}"
            );
        }
        assert!(
            probe.validate_for(profile).is_err(),
            "layer 1 refuses {profile:?} as well, so the two tables agree"
        );
    }
}

#[test]
fn a_body_that_reads_the_frame_block_is_given_one_in_both_stages() {
    // A GLSL uniform block is visible only to the stage that declares it, so
    // this is not a formatting preference: a fragment body reading `base_color`
    // while only the vertex stage declared the block would fail to link on a
    // real context and nowhere in this suite.
    for kernel in KERNELS {
        let identity = kernel.portable_identity();
        let (vertex, fragment, _) = raster(kernel, GlFamilyProfile::WebGl2);
        let declared = vertex.text.contains(text::FRAME_UNIFORMS)
            && fragment.text.contains(text::FRAME_UNIFORMS);
        let read = vertex.text.contains("view_projection") || fragment.text.contains("base_color");
        assert_eq!(
            declared,
            identity.uniform_binding.is_some(),
            "{kernel:?} declares the frame block exactly when its recipe has one"
        );
        assert!(
            !read || declared,
            "{kernel:?} reads the frame block, so both of its stages must declare it"
        );
    }
}

#[test]
fn the_layout_claims_exactly_the_bindings_the_identity_records() {
    for kernel in KERNELS {
        let identity = kernel.portable_identity();
        let (vertex, fragment, layout) = raster(kernel, GlFamilyProfile::WebGl2);
        layout
            .validate()
            .expect("the layout is one layer 1 accepts");

        let uniforms: Vec<_> = layout
            .bindings
            .iter()
            .filter(|binding| binding.kind == GlShaderResourceKind::UniformBuffer)
            .collect();
        let textures: Vec<_> = layout
            .bindings
            .iter()
            .filter(|binding| binding.kind == GlShaderResourceKind::CombinedTextureSampler)
            .collect();
        assert_eq!(
            uniforms.len(),
            usize::from(identity.uniform_binding.is_some()),
            "{kernel:?} declares one uniform binding exactly when its recipe has one"
        );
        assert_eq!(
            textures.len(),
            usize::from(identity.texture_binding.is_some()),
            "{kernel:?} declares one combined texture binding exactly when its recipe has one"
        );
        assert!(
            layout
                .bindings
                .iter()
                .all(|binding| binding.array_count == 1 && binding.location.group == 0),
            "{kernel:?} binds single non-arrayed resources in group zero"
        );

        if let Some(binding) = identity.uniform_binding {
            assert_eq!(
                uniforms[0].location.binding, binding,
                "{kernel:?} keeps the recipe's own binding number"
            );
            assert_eq!(uniforms[0].name, text::FRAME_UNIFORMS_NAME);
        }
        if let Some(binding) = identity.texture_binding {
            assert_eq!(
                textures[0].location.binding, binding,
                "{kernel:?} keeps the recipe's own binding number"
            );
            assert_eq!(textures[0].name, text::TEXTURE_NAME);
        }

        // The names are what a provider reflects by, so each one has to be the
        // identifier the text actually declares -- not merely a name the layout
        // and the declaration happen to share by coincidence of spelling.
        for binding in &layout.bindings {
            let declared = match binding.kind {
                GlShaderResourceKind::UniformBuffer => {
                    vertex.text.contains(&format!("uniform {}", binding.name))
                        && fragment.text.contains(&format!("uniform {}", binding.name))
                }
                GlShaderResourceKind::CombinedTextureSampler => fragment
                    .text
                    .contains(&format!("uniform sampler2D {};", binding.name)),
                // The two storage kinds are a compute program's, and no raster
                // recipe declares one, so a raster layout naming either would be
                // exactly the drift this loop exists to catch: `false` fails the
                // assertion below naming the kernel and the binding.
                GlShaderResourceKind::Sampler
                | GlShaderResourceKind::Texture
                | GlShaderResourceKind::StorageBuffer
                | GlShaderResourceKind::StorageImage => false,
            };
            assert!(
                declared,
                "{kernel:?} names {:?} in its layout but does not declare it",
                binding.name
            );
        }
    }
}

#[test]
fn the_sampler_folds_into_the_combined_texture_binding_rather_than_a_second_one() {
    // A GLSL `sampler2D` is a texture and its sampler at one unit, so the
    // recipes that address a texture with a separate WGSL sampler binding lose
    // exactly one logical binding here -- and the layout cannot carry which
    // WGSL binding the sampler arrived on, which is why the fold is asserted
    // against the identity instead of against a count written here.
    for kernel in KERNELS {
        let identity = kernel.portable_identity();
        let (_, _, layout) = raster(kernel, GlFamilyProfile::WebGl2);
        let folded = u32::from(identity.sampler_binding.is_some());
        assert_eq!(
            layout.bindings.len() as u32,
            identity.binding_count - folded,
            "{kernel:?} folds its sampler binding into the texture's"
        );
    }
}

#[test]
fn every_vertex_layout_is_one_layer_one_accepts() {
    for kernel in KERNELS {
        let identity = kernel.portable_identity();
        let layout = vertex_layout(kernel);
        layout
            .validate()
            .unwrap_or_else(|error| panic!("{kernel:?} has an invalid vertex layout: {error:?}"));

        // Three of the six layouts have a second stream, and only one of those
        // three -- the explicit-UV recipe -- records the stream's own layout,
        // stride and shader location in the identity.  So "has a slot-one
        // attribute" is read from the layout variant, not from those optional
        // fields: the normal and vertex-color streams are equally real and
        // simply carry no coordinate mapping to describe.
        let second: Vec<_> = layout
            .attributes
            .iter()
            .filter(|attribute| attribute.buffer_slot == 1)
            .collect();
        let two_streams = matches!(
            identity.vertex_layout,
            RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2
                | RasterVertexLayout::PositionFloat32x3AndNormalFloat32x3
                | RasterVertexLayout::PositionFloat32x3AndColorUnorm8x4
        );
        assert_eq!(
            second.len(),
            usize::from(two_streams),
            "{kernel:?} declares a slot-one attribute exactly when its recipe has a second stream"
        );
        if two_streams {
            assert_eq!(
                layout.buffers.len(),
                2,
                "{kernel:?} has one buffer slot per stream"
            );
            if let Some(location) = identity.texture_coordinate_shader_location {
                assert_eq!(
                    second[0].location, location,
                    "{kernel:?} reads slot one at the location its recipe records"
                );
            }
        }

        // The optional fields are a subset of what has a second stream: a
        // recipe that recorded one is one of the three, and the one that
        // records a *stride* is the explicit-UV recipe alone.
        if identity.texture_coordinate_vertex_layout.is_some() {
            assert!(
                two_streams,
                "{kernel:?} recorded a slot-one layout, so it must have a second stream"
            );
            assert_eq!(
                identity.vertex_layout,
                RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2,
                "{kernel:?} records a slot-one layout only as the explicit-UV recipe"
            );
        }
    }
}

#[test]
fn the_second_stream_is_tightly_packed_exactly_as_the_recipe_declares() {
    for kernel in KERNELS {
        let identity = kernel.portable_identity();
        let layout = vertex_layout(kernel);
        assert_eq!(
            layout.buffers.first().map(|buffer| buffer.stride),
            (!layout.buffers.is_empty()).then_some(identity.vertex_stride),
            "{kernel:?} takes its first stream's stride from its own identity"
        );
        if let Some(buffer) = layout.buffers.get(1) {
            assert_eq!(
                buffer.stride,
                identity.texture_coordinate_vertex_stride.unwrap_or(0),
                "{kernel:?} leaves tight packing to the recipe's own record"
            );
        }
    }
}

#[test]
fn the_triangle_takes_its_positions_from_the_vertex_index() {
    // The one recipe with no vertex input at all, and the two facts that make
    // it that: an empty layout, and a vertex stage that reads the index.
    let layout = vertex_layout(RasterKernel::Triangle);
    assert!(layout.buffers.is_empty() && layout.attributes.is_empty());
    for kernel in KERNELS {
        let empty = vertex_layout(kernel).attributes.is_empty();
        assert_eq!(
            empty,
            kernel == RasterKernel::Triangle,
            "{kernel:?} has no vertex input exactly when it is the index-driven recipe"
        );
    }
    let (vertex, _, _) = raster(RasterKernel::Triangle, GlFamilyProfile::WebGl2);
    assert!(
        vertex.text.contains("gl_VertexID"),
        "the index-driven recipe reads the vertex index"
    );
}

#[test]
fn the_linear_clamp_pair_share_one_program_body() {
    // One pair of the ten is one program, and it is this pair.  The sRGB kernel
    // is not a second lowering: the decode is its texture's internal format,
    // and writing it into the shader would decode twice.
    let shared = [
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
    ];
    let (first_vertex, first_fragment, first_layout) = raster(shared[0], GlFamilyProfile::WebGl2);
    let (second_vertex, second_fragment, second_layout) =
        raster(shared[1], GlFamilyProfile::WebGl2);
    assert_eq!(first_vertex.text, second_vertex.text);
    assert_eq!(first_fragment.text, second_fragment.text);
    assert_eq!(first_layout, second_layout);

    // And no *other* pair does.  Counting the recipes per body pair is what
    // makes "exactly one shared" a claim rather than an impression.
    let mut by_body: std::collections::BTreeMap<(String, String), Vec<RasterKernel>> =
        std::collections::BTreeMap::new();
    for kernel in KERNELS {
        let (vertex, fragment, _) = raster(kernel, GlFamilyProfile::WebGl2);
        by_body
            .entry((vertex.text, fragment.text))
            .or_default()
            .push(kernel);
    }
    let shared_pairs: Vec<_> = by_body
        .into_iter()
        .filter(|(_, kernels)| kernels.len() > 1)
        .collect();
    assert_eq!(
        shared_pairs.len(),
        1,
        "exactly one pair of the ten recipes is one program"
    );
    assert_eq!(shared_pairs[0].1, shared.to_vec());
}

#[test]
fn lowering_twice_produces_the_same_bytes() {
    // The digest and the text are both functions of the recipe and the profile,
    // and Layer 2's program cache is keyed on the descriptor's contents, so a
    // lowering that varied between calls would compile a second program for a
    // recipe already linked.
    for profile in ADMITTED {
        for kernel in KERNELS {
            assert_eq!(
                program(kernel, profile),
                program(kernel, profile),
                "{kernel:?} on {profile:?} lowers to the same bytes twice"
            );
        }
    }
}

#[test]
fn the_digest_separates_profiles_and_agrees_with_the_text() {
    // Two things at once, because they are the two things a content digest has
    // to do: distinguish different text, and be equal for equal text.  Nothing
    // in this crate reads the value back, so this is the whole of its contract.
    let (webgl2_vertex, _, _) = raster(RasterKernel::Triangle, GlFamilyProfile::WebGl2);
    let (desktop_vertex, _, _) = raster(
        RasterKernel::Triangle,
        GlFamilyProfile::Desktop { major: 4, minor: 6 },
    );
    assert_ne!(webgl2_vertex.text, desktop_vertex.text);
    assert_ne!(webgl2_vertex.source_hash, desktop_vertex.source_hash);
    assert_eq!(
        webgl2_vertex.source_hash,
        content_hash(&webgl2_vertex.text),
        "the digest is a function of the text alone"
    );
    for kernel in KERNELS {
        let (vertex, fragment, _) = raster(kernel, GlFamilyProfile::WebGl2);
        assert_eq!(vertex.source_hash, content_hash(&vertex.text));
        assert_eq!(fragment.source_hash, content_hash(&fragment.text));
        assert_ne!(
            vertex.source_hash, fragment.source_hash,
            "{kernel:?} lowers two stages to two different texts"
        );
    }
}
