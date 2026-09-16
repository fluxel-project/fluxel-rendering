//! The two registered objects, and the registry that answers for them.
//!
//! F3(b)'s subject is the seam between a *renderer* and this adapter: what a
//! renderer registers before a frame, what the executor is handed back when it
//! asks for one of those identities, and what happens when the two disagree.
//! None of it issues a command -- that is F3(c)'s subject -- so the tests here
//! are about refusals and about shapes, and every shape is compared against the
//! lowering the sibling suite already owns rather than restated here.
//!
//! # The claim that is not a comparison
//!
//! Both objects are handed out with a lease, and the claim is that the lease
//! covers nothing.  That is a claim about *this* backend rather than about the
//! contract, and the evidence for it is not the implementation's own comment: it
//! is that releasing either object destroys nothing, while releasing a transient
//! -- the same lease type, dropped the same way -- destroys what it named.  The
//! test asserts both halves together, because an empty release is worth nothing
//! as evidence unless something else in the same test did fill it.

use fluxel_rendergraph::{
    BindingResourceSemantic, BindingSetId, BufferDesc, BufferRange, BufferUsage, BufferUsageKind,
    BufferWriteUse, DeviceIdentity, ExecutionBackend, QueueId, RasterPipelineId,
    RecordingErrorKind, RenderObjectProvider, ResolvedBindingResource, TextureRange, TextureUsage,
    TextureUsageKind, TextureWriteUse,
};

use super::super::object::{BindingSlot, Bindings};
use super::super::registry::GlObjectRegistry;
use super::super::retention::ReleaseQueue;
use super::{
    Adapter, adapter, colour_usage, count, is_destroy_buffer, is_destroy_texture, plain_texture,
    refused, resolved, sampled, trace_from_here,
};
use crate::resource::RasterKernel;
use crate::webgl2::api::{BufferId, GlFamilyApi, GlFamilyProfile, MockGlFamilyApi, TextureId};

/// The ten closed recipes, in the order the artifact module declares them.
///
/// A copy of the sibling lowering suite's list, and deliberately a copy: the two
/// suites check different things about the same ten artifacts, and sharing one
/// list would need a constant in the artifact module whose only consumers are
/// tests.  What a stale copy here costs is coverage of the new recipe in *this*
/// suite, and nothing else -- the completeness guard is the lowering's own
/// `bodies()`, a match with no wildcard arm, so a new variant is a compile error
/// wherever it is first felt.
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

/// Every profile this family has a dialect for, one per version it can write.
const ADMITTED: [GlFamilyProfile; 7] = [
    GlFamilyProfile::WebGl2,
    GlFamilyProfile::Embedded { major: 3, minor: 0 },
    GlFamilyProfile::Embedded { major: 3, minor: 1 },
    GlFamilyProfile::Embedded { major: 3, minor: 2 },
    GlFamilyProfile::Desktop { major: 4, minor: 0 },
    GlFamilyProfile::Desktop { major: 4, minor: 3 },
    GlFamilyProfile::Desktop { major: 4, minor: 6 },
];

/// Profiles this family writes no dialect for, one per way of having none: an
/// embedded core below the floor, an embedded core above the ceiling, and a
/// desktop core that is not 4.x at all.
const REFUSED: [GlFamilyProfile; 4] = [
    GlFamilyProfile::Embedded { major: 2, minor: 0 },
    GlFamilyProfile::Embedded { major: 3, minor: 3 },
    GlFamilyProfile::Desktop { major: 3, minor: 3 },
    GlFamilyProfile::Desktop { major: 2, minor: 1 },
];

/// A registry with a device identity and a queue of its own.
///
/// Used where the profile is the variable: the adapter's own registry is always
/// built from the context it runs on, so a profile sweep needs one that is not.
fn detached(profile: GlFamilyProfile) -> GlObjectRegistry<MockGlFamilyApi> {
    GlObjectRegistry::new(DeviceIdentity::new(1), profile, ReleaseQueue::new())
}

#[test]
fn every_fixed_artifact_registers_and_resolves_to_the_lowering_of_the_context_profile() {
    let mut adapter = adapter();
    let identity = adapter.device_identity();
    let mut objects = adapter.object_registry();
    for (index, kernel) in KERNELS.into_iter().enumerate() {
        let id = RasterPipelineId::new(index as u64);
        objects
            .register_raster_pipeline(id, kernel)
            .unwrap_or_else(|error| panic!("{kernel:?} is lowered on WebGL2: {error:?}"));
        let bound = objects
            .raster_pipeline(id)
            .unwrap_or_else(|error| panic!("{kernel:?} was registered: {}", error.context.detail));

        // What came back, checked against the lowering the sibling suite owns
        // rather than against a second copy of it: if the registry ever handed
        // out another recipe's object, or one lowered for another profile, this
        // is where it shows.
        assert_eq!(
            bound.device, identity,
            "{kernel:?} resolves for the device the registry was built on"
        );
        assert_eq!(bound.physical.kernel(), kernel);
        assert_eq!(
            bound.physical.descriptor(),
            &super::super::super::shader::program(kernel, GlFamilyProfile::WebGl2)
                .expect("WebGL2 is an admitted profile")
        );
    }
}

#[test]
fn registration_refuses_exactly_the_profiles_the_lowering_has_no_dialect_for() {
    // Both directions over both sets.  "Refused at registration" is only worth
    // having if the set refused is the set that cannot be lowered: a registry
    // that refused too much would push the same failure to a later frame, which
    // is the ordering this refusal exists to avoid, and one that refused too
    // little would report it at a draw instead of at the call that established
    // the fact.
    for profile in ADMITTED {
        for kernel in KERNELS {
            assert!(
                super::super::super::shader::program(kernel, profile).is_ok(),
                "the lowering writes a dialect for {profile:?}"
            );
            let mut objects = detached(profile);
            objects
                .register_raster_pipeline(RasterPipelineId::new(0), kernel)
                .unwrap_or_else(|error| panic!("{kernel:?} registers on {profile:?}: {error:?}"));
        }
    }
    for profile in REFUSED {
        for kernel in KERNELS {
            assert!(
                super::super::super::shader::program(kernel, profile).is_err(),
                "the lowering writes no dialect for {profile:?}"
            );
            let mut objects = detached(profile);
            assert_eq!(
                refused(objects.register_raster_pipeline(RasterPipelineId::new(0), kernel)),
                "register-raster-pipeline",
                "{kernel:?} on {profile:?} names the call that refused it"
            );
        }
    }
}

#[test]
fn an_identity_that_was_never_registered_is_refused_rather_than_resolved() {
    let mut adapter = adapter();
    let mut objects = adapter.object_registry();

    let error = objects
        .raster_pipeline(RasterPipelineId::new(4))
        .err()
        .expect("nothing was registered under this identity");
    assert!(matches!(
        error.kind,
        RecordingErrorKind::IncompatibleBindingRecipe
    ));
    // The context carries no pass: a provider is not inside the recorder and does
    // not know which pass asked, and inventing one would be a worse diagnostic
    // than an empty list.
    assert!(error.context.passes.is_empty());
    assert!(
        !error.context.detail.is_empty(),
        "the refusal says what is missing"
    );

    // The neighbouring identity resolves, so the refusal above was about the
    // identity and not about the registry being empty of everything.
    objects
        .register_raster_pipeline(RasterPipelineId::new(3), KERNELS[0])
        .expect("an admitted profile");
    assert!(objects.raster_pipeline(RasterPipelineId::new(3)).is_ok());

    let error = objects
        .bindings(BindingSetId::new(0), &[], &[])
        .err()
        .expect("nothing was registered under this set identity");
    assert!(matches!(
        error.kind,
        RecordingErrorKind::IncompatibleBindingRecipe
    ));
}

#[test]
fn a_binding_set_is_validated_against_the_resources_the_frame_resolved_for_it() {
    let mut adapter = adapter();
    let (texture, buffer) = matched_resources(&mut adapter);
    let mut objects = adapter.object_registry();

    for (index, kernel) in KERNELS.into_iter().enumerate() {
        let identity = kernel.portable_identity();
        let resources = resolved(kernel, &texture, &buffer);
        let id = BindingSetId::new(index as u64);
        objects.register_bindings(id, kernel);
        let bound = objects
            .bindings(id, &resources, &[])
            .unwrap_or_else(|error| panic!("{kernel:?}: {}", error.context.detail));

        assert_eq!(bound.physical.kernel(), kernel, "the set kept its recipe");
        // One slot per *logical* binding, which is one per WGSL binding minus the
        // sampler that a GLSL `sampler2D` folds into its texture.  Read off the
        // identity's own numbers rather than counted here, so the two facts are
        // compared instead of the same fact written twice.
        let expected = usize::from(identity.uniform_binding.is_some())
            + usize::from(identity.texture_binding.is_some());
        assert_eq!(bound.physical.slots().len(), expected, "{kernel:?}");
        let numbers: Vec<u32> = bound
            .physical
            .slots()
            .iter()
            .map(|slot| match slot {
                BindingSlot::Uniform { binding, .. } | BindingSlot::Texture { binding, .. } => {
                    *binding
                }
            })
            .collect();
        let mut declared = Vec::new();
        if let Some(binding) = identity.uniform_binding {
            declared.push(binding);
        }
        if let Some(binding) = identity.texture_binding {
            declared.push(binding);
        }
        assert_eq!(
            numbers, declared,
            "{kernel:?} keeps the recipe's own binding numbers"
        );
        assert_eq!(
            identity.binding_count - bound.physical.slots().len() as u32,
            u32::from(identity.sampler_binding.is_some()),
            "{kernel:?} folds exactly its sampler binding and nothing else"
        );
    }
}

#[test]
fn a_binding_set_that_disagrees_with_its_resolution_is_refused() {
    let mut adapter = adapter();
    let (texture, buffer) = matched_resources(&mut adapter);
    // The texture-sampling recipe, chosen because it is the one with all three
    // binding numbers to disagree about.
    let kernel = RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture;
    let identity = kernel.portable_identity();
    let uniform_at = identity
        .uniform_binding
        .expect("the recipe reads the frame block");
    assert!(
        identity.texture_binding.is_some(),
        "the recipe samples a texture"
    );

    let good = resolved(kernel, &texture, &buffer);
    assert!(Bindings::new(kernel, &good).is_ok(), "the agreeing case");

    // One resource short: the count is the first thing that can disagree, and it
    // is what catches a set id registered against another artifact.
    assert!(Bindings::new(kernel, &good[..good.len() - 1]).is_err());

    // The uniform's own number carrying a texture.  Built by hand rather than by
    // permuting `good`, so that the case is the one named in the assertion.
    let swapped: Vec<_> = (0..identity.binding_count)
        .map(|_| sampled(&texture))
        .collect();
    assert!(Bindings::new(kernel, &swapped).is_err());

    // The right kinds at the right numbers and the wrong authorization: a
    // uniform block and a sampled texture are both read, so a resource the graph
    // resolved as a write cannot be bound as either.
    let written: Vec<_> = (0..identity.binding_count)
        .map(|index| {
            if index == uniform_at {
                ResolvedBindingResource::Buffer {
                    physical: &buffer,
                    range: BufferRange::Whole,
                    semantic: BindingResourceSemantic::BufferWrite(BufferWriteUse::Storage),
                }
            } else if identity.texture_binding == Some(index) {
                ResolvedBindingResource::Texture {
                    physical: &texture,
                    range: TextureRange::Whole,
                    semantic: BindingResourceSemantic::TextureWrite(TextureWriteUse::Storage),
                }
            } else {
                sampled(&texture)
            }
        })
        .collect();
    assert!(Bindings::new(kernel, &written).is_err());
}

#[test]
fn a_dynamic_offset_has_nothing_to_apply_to_in_a_fixed_binding_set() {
    let mut adapter = adapter();
    let (texture, buffer) = matched_resources(&mut adapter);
    let kernel = RasterKernel::IndexedPositionFloat32x3CameraMaterial;
    let resources = resolved(kernel, &texture, &buffer);
    let mut objects = adapter.object_registry();
    let id = BindingSetId::new(0);
    objects.register_bindings(id, kernel);

    assert!(objects.bindings(id, &resources, &[]).is_ok());
    let error = objects
        .bindings(id, &resources, &[0])
        .err()
        .expect("a fixed binding set declares no arrayed binding");
    assert!(matches!(
        error.kind,
        RecordingErrorKind::IncompatibleBindingRecipe
    ));
    assert!(
        error.context.detail.contains("dynamic offset"),
        "the refusal says which fact it is about, not merely that something is wrong"
    );
}

#[test]
fn the_two_registered_objects_retain_nothing_so_releasing_them_destroys_nothing() {
    let mut adapter = adapter();
    let (texture, buffer) = matched_resources(&mut adapter);
    let kernel = RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture;
    let mut objects = adapter.object_registry();
    let pipeline_id = RasterPipelineId::new(0);
    let set_id = BindingSetId::new(0);
    objects
        .register_raster_pipeline(pipeline_id, kernel)
        .expect("an admitted profile");
    objects.register_bindings(set_id, kernel);
    let resources = resolved(kernel, &texture, &buffer);

    // The two resources were released when `matched_resources` dropped their
    // leases.  Draining that first leaves the trace about the objects under test.
    adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    trace_from_here(&mut adapter);

    // The control: a transient's lease is the same type, dropped the same way,
    // and it does destroy what it named.
    let control = adapter
        .create_transient_texture(plain_texture(), colour_usage())
        .expect("a transient texture");
    drop(control);

    let pipeline = objects.raster_pipeline(pipeline_id).expect("registered");
    let bindings = objects
        .bindings(set_id, &resources, &[])
        .expect("registered");
    drop(pipeline);
    drop(bindings);

    adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    assert_eq!(
        count(&mut adapter, is_destroy_texture),
        1,
        "only the control's texture is destroyed: a non-empty group here would be the binding set pushing the frame's own texture a second time, which release_pending would then destroy twice"
    );
    assert_eq!(
        count(&mut adapter, is_destroy_buffer),
        0,
        "and the frame's buffer likewise, though the set names it"
    );
}

#[test]
fn a_registry_left_over_from_a_replaced_context_reports_the_superseded_generation() {
    let mut adapter = adapter();
    let before = adapter.device_identity();
    let mut objects = adapter.object_registry();
    let id = RasterPipelineId::new(0);
    objects
        .register_raster_pipeline(id, KERNELS[0])
        .expect("an admitted profile");
    assert_eq!(
        objects.raster_pipeline(id).expect("registered").device,
        before
    );

    adapter
        .machine
        .backend()
        .context_lost()
        .expect("context loss");
    adapter
        .machine
        .backend()
        .context_restored()
        .expect("context restoration");
    adapter.begin_encoder(QueueId::new(0)).expect("an encoder");

    // The claim is about the *registry*, not about the recipe it holds: the
    // recipe is still a valid lowering, and the generation it reports is what a
    // frame's own cross-device check compares.  So a stale registry is refused
    // rather than allowed to bind against a dead epoch.
    assert_ne!(
        adapter.device_identity(),
        before,
        "the restored context is a new common device"
    );
    let stale = objects
        .raster_pipeline(id)
        .expect("the registry still holds the recipe it was given");
    assert_eq!(
        stale.device, before,
        "it reports the generation it captured"
    );
    assert_ne!(
        stale.device,
        adapter.device_identity(),
        "which is the one the resolver will compare against and refuse"
    );
}

/// The one texture and one buffer a frame would resolve.
///
/// Made through the adapter rather than by hand, so that the two identities are
/// the ones a real creation produces; their leases are dropped here, which is
/// what makes the caller of this function the one that has to account for the
/// release.
fn matched_resources(adapter: &mut Adapter) -> (TextureId, BufferId) {
    let texture = adapter
        .create_transient_texture(
            plain_texture(),
            TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        )
        .expect("a transient texture")
        .physical;
    let buffer = adapter
        .create_transient_buffer(
            BufferDesc { size: 256 },
            BufferUsage::from_kinds([BufferUsageKind::Uniform]),
        )
        .expect("a transient buffer")
        .physical;
    (texture, buffer)
}
