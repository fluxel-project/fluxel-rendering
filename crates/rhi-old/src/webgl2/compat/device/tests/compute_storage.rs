//! How the storage half of a binding set lowers onto Layer 1's two forms.
//!
//! A compute artifact's bindings are not only textures and uniform blocks: a
//! storage block and an image unit are two binding forms the raster half never
//! reaches, and this family spells each of them in a way that needs a fact no
//! other layer holds.  That fact is the device's own record of what it created --
//! a buffer's extent, a texture's shape, format and sample count -- and the tests
//! here are about a lowering that reads it.
//!
//! `tests/compute.rs` owns the sibling subject: the pass kind, the order a
//! dispatch records in, and the refusal an adapter with no compute domain gives.
//! Both suites drive the same three adapters and both are refused by the same
//! two facts, which is why the fixtures and readers they share live in
//! [`super::harness`] rather than in either of them.
//!
//! # What a trace here is evidence for, and what it is not
//!
//! `MockCall::BindStorageBuffer` carries the offset and the size, so the
//! whole-allocation lowering is readable straight off the trace.  `MockCall::BindStorageImage`
//! carries the binding and the texture and **nothing else**, so the level, the
//! layer, the format and the sample count an image unit was given are not.
//! What stands in their place is that the recorder validates all four against its
//! own record of the texture *before* it records the call
//! (`api/mock/mod.rs::validate_storage_image_binding`), so a bind that reaches the
//! trace at all is a bind whose lowering agreed with the object -- and a lowering
//! that disagreed is a refusal, which
//! [`an_unproved_storage_image_refuses_at_the_dispatch_while_a_storage_buffer_binds`]
//! shows is observable from here.

use fluxel_rendergraph::{
    BindingResourceSemantic, BindingSetId, BufferRange, BufferReadUse, BufferWriteUse,
    ComputePipelineId, ExecutionBackend, QueueId, RenderObjectProvider, ResolvedBindingResource,
    TextureDimension, TextureRange, TextureReadUse, TextureUsage, TextureUsageKind,
    TextureWriteUse,
};

use super::super::object::Recipe;
use super::super::retention::GlRetentionLease;
use super::{
    ComputeAdapter, compute_adapter, compute_adapter_without_storage_images, compute_calls,
    compute_count, compute_trace_from_here, is_dispatch, plain_texture, read_write, refused,
    registered, sampled_usage, storage_buffer, texture,
};
use crate::resource::ComputeKernel;
use crate::webgl2::api::{BufferId, GlError, MockCall, TextureId};

/// A transient storage texture, as a binding's physical resource.
fn storage_texture(
    adapter: &mut ComputeAdapter,
    layers: u32,
) -> fluxel_rendergraph::BoundTexture<TextureId, GlRetentionLease> {
    adapter
        .create_transient_texture(
            texture(TextureDimension::D2, layers, 1),
            TextureUsage::from_kinds([TextureUsageKind::StorageWrite]),
        )
        .unwrap_or_else(|error| panic!("a transient storage texture: {error:?}"))
}

/// One storage buffer, authorized as a read-only shader access.
///
/// The half of the rule's asymmetry that a read-write declaration refuses, which
/// is why it is built beside [`read_write`] rather than inline at its one use.
fn read_only(buffer: &BufferId) -> ResolvedBindingResource<'_, TextureId, BufferId> {
    ResolvedBindingResource::Buffer {
        physical: buffer,
        range: BufferRange::Whole,
        semantic: BindingResourceSemantic::BufferRead(BufferReadUse::Storage),
    }
}

/// One texture, authorized as a read-only image unit's declaration states.
fn image_read(source: &TextureId) -> ResolvedBindingResource<'_, TextureId, BufferId> {
    ResolvedBindingResource::Texture {
        physical: source,
        range: TextureRange::Whole,
        semantic: BindingResourceSemantic::TextureRead(TextureReadUse::Storage),
    }
}

/// One texture, authorized as a write-only image unit's declaration states.
fn image_write(destination: &TextureId) -> ResolvedBindingResource<'_, TextureId, BufferId> {
    ResolvedBindingResource::Texture {
        physical: destination,
        range: TextureRange::Whole,
        semantic: BindingResourceSemantic::TextureWrite(TextureWriteUse::Storage),
    }
}

#[test]
fn the_storage_range_a_whole_binding_lowers_to_is_the_whole_allocation() {
    let mut adapter = compute_adapter();
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");

    // Two allocations, so that the size in the trace cannot be a constant this
    // test would accept either way.
    for size in [256_u64, 4_096] {
        let values = storage_buffer(&mut adapter, size);
        let resources = [read_write(&values.physical)];
        let (pipeline, bindings) = registered(&mut adapter, ComputeKernel::WrappingAdd, &resources);
        compute_trace_from_here(&mut adapter);
        adapter
            .begin_compute(&mut encoder, "compute")
            .expect("a compute pass");
        adapter
            .set_compute_pipeline(&mut encoder, &pipeline.physical)
            .expect("a registered pipeline");
        adapter
            .set_bindings(&mut encoder, &bindings.physical)
            .expect("a set for the installed recipe");
        adapter
            .dispatch(&mut encoder, [1, 1, 1])
            .expect("a dispatch");
        adapter.end_compute(&mut encoder).expect("the pass closes");

        // `Whole` is the one place this differs from the uniform path next door.
        // An indexed uniform binding point spells "to the end" as an offset of
        // zero with a size of zero; a storage binding is validated against the
        // real allocation, so a size of zero would be refused -- which is why
        // this adapter records each buffer's extent as it creates it.
        assert!(
            compute_calls(&mut adapter).iter().any(|call| matches!(
                call,
                MockCall::BindStorageBuffer { offset: 0, size: bound, .. } if *bound == size
            )),
            "a whole range lowers to the whole {size}-byte allocation: {:?}",
            compute_calls(&mut adapter)
        );
    }
}

// ---------------------------------------------------------------------------
// The storage image, whose four shape facts come from the device's own record.
// ---------------------------------------------------------------------------

#[test]
fn a_storage_image_binding_carries_the_shape_the_device_created() {
    let mut adapter = compute_adapter();
    let destination = storage_texture(&mut adapter, 1);
    let source = storage_texture(&mut adapter, 1);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");

    // The write-only artifact first, which is the smallest set that names an
    // image unit at all.
    let resources = [image_write(&destination.physical)];
    let (pipeline, bindings) =
        registered(&mut adapter, ComputeKernel::TextureStoreRgba8, &resources);
    compute_trace_from_here(&mut adapter);
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("a compute pass");
    adapter
        .set_compute_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
    adapter
        .dispatch(&mut encoder, [1, 1, 1])
        .expect("a dispatch");
    adapter.end_compute(&mut encoder).expect("the pass closes");
    assert!(
        compute_calls(&mut adapter).iter().any(|call| matches!(
            call,
            MockCall::BindStorageImage { binding: 0, texture } if *texture == destination.physical
        )),
        "the image unit names the texture the frame resolved: {:?}",
        compute_calls(&mut adapter)
    );

    // Then the artifact that reads one and writes a block through the *shared*
    // arm of the binding loop.  It is the reason that loop can serve both
    // families: `TextureLoadRgba8` samples nothing and stores nothing, yet its
    // first binding is a texture and its second a buffer, so a compute-only
    // second loop would have had to repeat the texture arm to reach it.
    let values = storage_buffer(&mut adapter, 256);
    let resources = [image_read(&source.physical), read_write(&values.physical)];
    let (pipeline, bindings) =
        registered(&mut adapter, ComputeKernel::TextureLoadRgba8, &resources);
    compute_trace_from_here(&mut adapter);
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("a compute pass");
    adapter
        .set_compute_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
    adapter
        .dispatch(&mut encoder, [1, 1, 1])
        .expect("a dispatch");
    adapter.end_compute(&mut encoder).expect("the pass closes");
    let trace = compute_calls(&mut adapter);
    assert!(
        trace.iter().any(|call| matches!(
            call,
            MockCall::BindStorageImage { binding: 0, texture } if *texture == source.physical
        )),
        "the image unit is binding zero, and it names the source: {trace:?}"
    );
    assert!(
        trace.iter().any(|call| matches!(
            call,
            MockCall::BindStorageBuffer { binding: 1, buffer, offset: 0, size: 256 } if *buffer == values.physical
        )),
        "while the storage block keeps its own number beside it: {trace:?}"
    );
}

#[test]
fn a_storage_image_range_the_family_cannot_name_is_refused_at_the_dispatch() {
    let mut adapter = compute_adapter();
    let destination = storage_texture(&mut adapter, 4);
    let id = BindingSetId::new(0);
    let pipeline_id = ComputePipelineId::new(0);
    let mut objects = adapter.object_registry();
    objects
        .register_compute_pipeline(pipeline_id, ComputeKernel::TextureStoreRgba8)
        .expect("the artifact lowers on this profile");
    objects.register_bindings(id, Recipe::Compute(ComputeKernel::TextureStoreRgba8));
    let pipeline = objects
        .compute_pipeline(pipeline_id)
        .expect("it was registered");
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("a compute pass");
    adapter
        .set_compute_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");

    // Two ranges an image unit has no way to name, and both are refused by the
    // *lowering* rather than by the recorder -- which is why each is asserted at
    // the verb that asked, and why neither is a validation failure: a caller that
    // selected one subresource could make the same call succeed.
    let several_mips = TextureRange::Subresources {
        base_mip_level: 0,
        mip_level_count: 2,
        base_array_layer: 0,
        array_layer_count: 1,
        aspect: fluxel_rendergraph::TextureAspect::All,
    };
    let partial_layers = TextureRange::Subresources {
        base_mip_level: 0,
        mip_level_count: 1,
        base_array_layer: 0,
        array_layer_count: 3,
        aspect: fluxel_rendergraph::TextureAspect::All,
    };
    for range in [several_mips, partial_layers] {
        let resources = [ResolvedBindingResource::Texture {
            physical: &destination.physical,
            range,
            semantic: BindingResourceSemantic::TextureWrite(TextureWriteUse::Storage),
        }];
        let bindings = objects
            .bindings(id, &resources, &[])
            .expect("the set validates: the range is a fact the bind lowers, not one it checks");
        adapter
            .set_bindings(&mut encoder, &bindings.physical)
            .expect("a set for the installed recipe");
        assert_eq!(
            refused(adapter.dispatch(&mut encoder, [1, 1, 1])),
            "dispatch",
            "{range:?} names neither one subresource nor all of them"
        );
    }
}

// ---------------------------------------------------------------------------
// The two refusals, side by side, and the rule the storage half adds.
// ---------------------------------------------------------------------------

#[test]
fn an_unproved_storage_image_refuses_at_the_dispatch_while_a_storage_buffer_binds() {
    let mut adapter = compute_adapter_without_storage_images();
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");

    // The positive control, and the reason this test is not simply a refusal: an
    // adapter of this type exists, its witness has the whole domain, and the two
    // capabilities its snapshot *did* prove are served.  Without this, a
    // "refusal" below could be a fact about the type rather than about the
    // snapshot -- which is the first refusal, and a different one.
    let values = storage_buffer(&mut adapter, 256);
    let resources = [read_write(&values.physical)];
    let (pipeline, bindings) = registered(&mut adapter, ComputeKernel::WrappingAdd, &resources);
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("compute was proved");
    adapter
        .set_compute_pipeline(&mut encoder, &pipeline.physical)
        .expect("a registered pipeline");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
    adapter
        .dispatch(&mut encoder, [1, 1, 1])
        .expect("storage buffers were proved");
    adapter.end_compute(&mut encoder).expect("the pass closes");
    assert_eq!(
        compute_count(&mut adapter, is_dispatch),
        1,
        "the storage-buffer half of the domain runs"
    );

    // The refusal, at the same verb and for the one capability the snapshot never
    // proved.  The sentence is the half that distinguishes it from the first
    // refusal: the vocabulary is here and this context is not one it may be used
    // on.
    //
    // The texture is created *without* the storage role, and that is forced rather
    // than chosen: a context that did not prove storage images cannot create a
    // storage texture at all (Layer 1's recorder refuses the usage), so the only
    // image a frame on this adapter could bind is one of these.  It makes the
    // request wrong in two ways, and what the assertion below holds is which of
    // them is reported -- the capability, because `C::admit` is asked at every
    // verb before anything else is looked at, and the object's own disagreement
    // with the binding would otherwise be the answer.
    let destination = adapter
        .create_transient_texture(plain_texture(), sampled_usage())
        .expect("a sampled transient");
    let resources = [image_write(&destination.physical)];
    let (pipeline, bindings) =
        registered(&mut adapter, ComputeKernel::TextureStoreRgba8, &resources);
    adapter
        .begin_compute(&mut encoder, "compute")
        .expect("compute was proved, so the pass still opens");
    adapter
        .set_compute_pipeline(&mut encoder, &pipeline.physical)
        .expect("a compute program needs no storage image");
    adapter
        .set_bindings(&mut encoder, &bindings.physical)
        .expect("a set for the installed recipe");
    match adapter.dispatch(&mut encoder, [1, 1, 1]) {
        Err(GlError::Unsupported { operation, reason }) => {
            assert_eq!(operation, "dispatch");
            assert!(
                reason.contains("did not prove the storage-image capability"),
                "the reason names the capability the snapshot never proved, rather than the adapter's type: {reason}"
            );
        }
        other => panic!("expected a fail-closed refusal naming the capability, got {other:?}"),
    }
}

#[test]
fn a_storage_binding_the_graph_authorized_for_something_else_is_refused() {
    let mut adapter = compute_adapter();
    let values = storage_buffer(&mut adapter, 256);
    let destination = storage_texture(&mut adapter, 1);
    let mut objects = adapter.object_registry();
    let set_id = BindingSetId::new(0);

    // The rule, stated once in `object.rs` and applying to both storage kinds: a
    // write must name the storage role, and a read need only be a read.  The
    // raster half of it is held by `tests/objects.rs`; what is new here is that
    // the compute family reads its declarations off the lowering's *layout* rather
    // than off an identity, so the rule is being applied from a second source.
    objects.register_bindings(set_id, Recipe::Compute(ComputeKernel::WrappingAdd));
    let copy_destination = [ResolvedBindingResource::Buffer {
        physical: &values.physical,
        range: BufferRange::Whole,
        semantic: BindingResourceSemantic::BufferWrite(BufferWriteUse::CopyDestination),
    }];
    assert!(
        objects.bindings(set_id, &copy_destination, &[]).is_err(),
        "a write authorized as a copy destination is a different command from a write a shader performs, and a storage binding is never one"
    );
    let read_only_range = [read_only(&values.physical)];
    assert!(
        objects.bindings(set_id, &read_only_range, &[]).is_err(),
        "and the other half of the asymmetry: the rule says a *read* need only be a read, not that a read satisfies a read-write declaration"
    );
    assert!(
        objects
            .bindings(set_id, &[read_write(&values.physical)], &[])
            .is_ok(),
        "while the access the artifact actually declares is accepted"
    );

    objects.register_bindings(set_id, Recipe::Compute(ComputeKernel::TextureStoreRgba8));
    for refused_semantic in [
        BindingResourceSemantic::TextureRead(TextureReadUse::Sampled),
        BindingResourceSemantic::TextureWrite(TextureWriteUse::CopyDestination),
    ] {
        let resources = [ResolvedBindingResource::Texture {
            physical: &destination.physical,
            range: TextureRange::Whole,
            semantic: refused_semantic,
        }];
        assert!(
            objects.bindings(set_id, &resources, &[]).is_err(),
            "a write-only image unit needs a write that names the storage role, and {refused_semantic:?} is not one"
        );
    }
    assert!(
        objects
            .bindings(set_id, &[image_write(&destination.physical)], &[])
            .is_ok(),
        "while the declared access is accepted"
    );

    // The read-only artifact takes a read, and its storage block is checked
    // separately -- which is the case that shows the two kinds are checked at
    // their own numbers rather than by one rule applied to the shape.
    objects.register_bindings(set_id, Recipe::Compute(ComputeKernel::TextureLoadRgba8));
    let source = storage_texture(&mut adapter, 1);
    let pair = [image_read(&source.physical), read_write(&values.physical)];
    assert!(
        objects.bindings(set_id, &pair, &[]).is_ok(),
        "a read authorization is what a read-only image unit needs"
    );
    let wrong_block = [image_read(&source.physical), read_only(&values.physical)];
    assert!(
        objects.bindings(set_id, &wrong_block, &[]).is_err(),
        "and its storage block is still refused a read-only authorization"
    );
}
