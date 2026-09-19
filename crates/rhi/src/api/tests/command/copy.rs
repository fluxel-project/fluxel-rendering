//! Sections 33 and 34: the compute scope, and copy, resolve, and blit.

use super::*;
use crate::api::binding::BindGroupIndex;
use crate::api::command::{
    BlitFilter, BufferCopy, ComputeScopeDescriptor, TextureBlit, TextureCopy,
};
use crate::api::pipeline::{ComputePipeline, ComputePipelineDescriptor};
use crate::api::tests::fixture;
use crate::base::mock::compute_pipeline_backend_for_test;

#[test]
fn begin_compute_stops_at_the_device_capability() {
    let mut recorder = recorder();
    assert_kind(
        recorder
            .begin_compute(&ComputeScopeDescriptor::new().with_label("dispatch"))
            .map(|_| ()),
        RhiErrorKind::Unsupported,
    );
}

/// Section 33's validation list, reached through a scope a test cannot open because
/// `begin_compute` stops at the device capability first.
///
/// Compiled, never called. What it reviews is the call shape: a dispatch takes
/// three workgroup counts and states nothing else, because the pipeline and the
/// groups were bound before it and the interface checks belong to the dispatch
/// rather than to the caller.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_dispatch_states_only_its_workgroups(recorder: &mut CommandRecorder) {
    let mut scope = recorder
        .begin_compute(&ComputeScopeDescriptor::new().with_label("shape"))
        .expect("the device has compute enabled");
    scope
        .set_pipeline(&compute_pipeline())
        .expect("the pipeline belongs to this device");
    scope
        .set_bind_group(
            BindGroupIndex::new(0),
            &uniform_group(uniform_layout(1)),
            &[],
        )
        .expect("the offsets match the layout");
    scope
        .dispatch(8, 8, 1)
        .expect("the counts are within the device's limits");
    scope.end().expect("the scope is complete");
}

fn compute_pipeline() -> ComputePipeline {
    let descriptor =
        ComputePipelineDescriptor::new(vertex_module(67), interface_of(uniform_layout(1)));
    // Section 28 hands the backend the caller's descriptor rather than a
    // canonical form, so this is one descriptor cloned once, not two packets that
    // happen to agree.
    let native = compute_pipeline_backend_for_test(descriptor.clone());
    ComputePipeline::new(object(57), device(), descriptor, native)
}

#[test]
fn a_copy_must_name_a_source_and_a_destination_that_allow_it() {
    let mut recorder = recorder();

    // The destination must be a copy destination.
    let wrong_dst = BufferCopy {
        dst: buffer_with(BufferUsage::COPY_SRC, 64),
        ..buffer_copy()
    };
    assert_kind(recorder.copy_buffer(&wrong_dst), RhiErrorKind::InvalidUsage);

    // The range must fit both sides.
    let too_long = BufferCopy {
        size: 128,
        ..buffer_copy()
    };
    assert_kind(recorder.copy_buffer(&too_long), RhiErrorKind::InvalidUsage);

    // An empty copy is not a copy.
    let empty = BufferCopy {
        size: 0,
        ..buffer_copy()
    };
    assert_kind(recorder.copy_buffer(&empty), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_copy_from_another_device_is_wrong_device() {
    let mut recorder = recorder();
    let foreign = BufferCopy {
        src: fixture::buffer(
            object(11),
            other_device(),
            BufferDescriptor::new(64, BufferUsage::COPY_SRC),
        ),
        ..buffer_copy()
    };
    assert_kind(recorder.copy_buffer(&foreign), RhiErrorKind::WrongDevice);
}

#[test]
fn a_legal_buffer_copy_stops_at_the_device_route() {
    // The portable validation passed; what is missing is the device's answer to
    // `RouteQuery::BufferToBuffer` and its copy-layout alignment. This device
    // reports no route at all, so section 9.4's answer is a refusal rather than a
    // substituted path.
    let mut recorder = recorder();
    assert_kind(
        recorder.copy_buffer(&buffer_copy()),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn a_buffer_copy_the_device_route_accepts_is_recorded() {
    // The same descriptor that the empty device refuses, on a device that reports
    // the route with a 4-byte alignment its offsets and size satisfy. This is the
    // half of the pair that keeps the refusals from being unconditional.
    let mut recorder = recorder_reporting(facts_with_buffer_copy_route(4, 4));
    assert!(
        recorder.copy_buffer(&buffer_copy()).is_ok(),
        "a 64-byte copy at offset 0 satisfies a 4-byte alignment"
    );
}

#[test]
fn a_buffer_copy_that_breaks_the_route_alignment_is_refused() {
    // Section 12.4's alignment rule, which no test could reach before a recorder
    // could hold the device's copy-layout limits. The copy is legal in every
    // portable respect — the range fits, the usages are right — and the device's
    // own alignment is what refuses it.
    let mut recorder = recorder_reporting(facts_with_buffer_copy_route(4, 4));
    let misaligned = BufferCopy {
        src_offset: 1,
        ..buffer_copy()
    };
    assert_kind(
        recorder.copy_buffer(&misaligned),
        RhiErrorKind::InvalidUsage,
    );

    // And on the destination side, which is a separate check: an implementation
    // that validated only one end would pass the first case above and this one
    // would still be caught, but a copy misaligned at the destination only would
    // not have been.
    let mut recorder = recorder_reporting(facts_with_buffer_copy_route(4, 4));
    let misaligned_dst = BufferCopy {
        dst_offset: 2,
        size: 60,
        ..buffer_copy()
    };
    assert_kind(
        recorder.copy_buffer(&misaligned_dst),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_buffer_texture_copy_that_breaks_the_texel_alignment_is_refused() {
    // One row of the region is 16 bytes. The device reports a 256-byte row pitch
    // alignment, so the tight pitch is refused — and the generously padded one,
    // whose footprint the buffer was sized for, is accepted. Both directions,
    // because an alignment check that refused everything would also pass a test
    // that only asserted the refusal.
    let mut recorder = recorder_reporting(facts_with_texel_copy_route(256, 256));
    assert_kind(
        recorder.copy_buffer_to_texture(&buffer_texture_copy(16, 1024)),
        RhiErrorKind::InvalidUsage,
    );

    let mut recorder = recorder_reporting(facts_with_texel_copy_route(256, 256));
    assert!(
        recorder
            .copy_buffer_to_texture(&buffer_texture_copy(256, 1024))
            .is_ok(),
        "a 256-byte row pitch satisfies the alignment this device reports"
    );
}

#[test]
fn a_texture_copy_must_agree_on_shape() {
    let mut recorder = recorder();
    let copy = TextureCopy {
        src: renderable_texture(TextureFormat::Rgba8Unorm),
        src_subresource: color_layers(1),
        src_origin: origin(),
        dst: renderable_texture(TextureFormat::Rgba8Unorm),
        // The destination names two layers where the source names one.
        dst_subresource: color_layers(2),
        dst_origin: origin(),
        extent: Extent3d::d2(4, 4),
    };
    assert_kind(recorder.copy_texture(&copy), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_blit_that_reaches_past_its_source_is_refused() {
    let mut recorder = recorder();
    let blit = TextureBlit {
        src: renderable_texture(TextureFormat::Rgba8Unorm),
        src_subresource: color_layers(1),
        src_origin: origin(),
        src_extent: Extent3d::d2(8, 8),
        dst: renderable_texture(TextureFormat::Rgba8Unorm),
        dst_subresource: color_layers(1),
        dst_origin: origin(),
        dst_extent: Extent3d::d2(4, 4),
        filter: BlitFilter::Linear,
    };

    // The source region is 8x8 and the texture is 4x4. The filter is part of the
    // route key rather than of this refusal: section 34.5 leaves "is a linear blit
    // supported here" to the device.
    assert_kind(recorder.blit_texture(&blit), RhiErrorKind::InvalidUsage);
}
