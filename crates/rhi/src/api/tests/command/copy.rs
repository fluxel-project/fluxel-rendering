//! Sections 33 and 34: the compute scope, and copy, resolve, and blit.

use super::*;
use crate::api::binding::BindGroupIndex;
use crate::api::command::{
    BlitFilter, BufferCopy, ComputeScopeDescriptor, TextureBlit, TextureCopy,
};
use crate::api::pipeline::{ComputePipeline, ComputePipelineDescriptor};
use crate::api::resource::subresource::{Origin3d, TextureAspect, TextureSubresourceLayers};

/// A one-mip, one-layer color selection at the origin.
fn color_layers(layer_count: u32) -> TextureSubresourceLayers {
    TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count,
    }
}

fn origin() -> Origin3d {
    Origin3d { x: 0, y: 0, z: 0 }
}

#[test]
#[should_panic(expected = "OptionalFeature::Compute")]
fn begin_compute_stops_at_the_device_capability() {
    let mut recorder = recorder();
    let _ = recorder.begin_compute(&ComputeScopeDescriptor::new().with_label("dispatch"));
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
    ComputePipeline::new(
        object(57),
        device(),
        ComputePipelineDescriptor::new(vertex_module(67), interface_of(uniform_layout(1))),
    )
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
        src: Buffer::new(
            object(11),
            other_device(),
            BufferDescriptor::new(64, BufferUsage::COPY_SRC),
        ),
        ..buffer_copy()
    };
    assert_kind(recorder.copy_buffer(&foreign), RhiErrorKind::WrongDevice);
}

#[test]
#[should_panic(expected = "copy_buffer must ask the device")]
fn a_legal_buffer_copy_stops_at_the_device_route() {
    // The portable validation passed; what is missing is the device's answer to
    // `RouteQuery::BufferToBuffer` and its copy-layout alignment.
    let mut recorder = recorder();
    let _ = recorder.copy_buffer(&buffer_copy());
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
