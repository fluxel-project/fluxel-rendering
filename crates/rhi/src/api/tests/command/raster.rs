//! Section 32: the raster scope and the draw verbs.

use super::*;
use crate::api::binding::BindGroupIndex;
use crate::api::command::{AccessMask, PipelineScope, ResourceUse, TextureUseIntent};
use crate::api::command::{IndexFormat, Rect, Viewport};
use crate::api::pipeline::{
    PrimitiveState, PrimitiveTopology, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexInputState, VertexStepMode,
};
use crate::api::submission::LaneWorkDomains;

/// A pipeline whose vertex input reads an 8-byte stride of two floats at slot 0.
fn pipeline_reading_a_vertex_buffer(
    id: u64,
    module: u64,
    layout: BindGroupLayout,
) -> RasterPipeline {
    RasterPipeline::new(
        object(id),
        device(),
        RasterPipelineDescriptor::new(vertex_module(module), interface_of(layout))
            .with_color_target(
                ShaderLocation::new(0),
                ColorTargetState::new(TextureFormat::Rgba8Unorm),
            )
            .with_vertex_input(VertexInputState::new().with_buffer(
                VertexBufferLayout::new(8, VertexStepMode::Vertex).with_attribute(
                    VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x2, 0),
                ),
            )),
        crate::api::tests::mock::raster_pipeline_backend_for_test(),
    )
}

/// A pipeline drawing a strip, optionally declaring the index format it is cut
/// with.
fn strip_pipeline(id: u64, module: u64, format: Option<IndexFormat>) -> RasterPipeline {
    let primitive = match format {
        Some(format) => {
            PrimitiveState::new(PrimitiveTopology::TriangleStrip).with_strip_index_format(format)
        }
        None => PrimitiveState::new(PrimitiveTopology::TriangleStrip),
    };
    RasterPipeline::new(
        object(id),
        device(),
        RasterPipelineDescriptor::new(vertex_module(module), interface_of(uniform_layout(1)))
            .with_color_target(
                ShaderLocation::new(0),
                ColorTargetState::new(TextureFormat::Rgba8Unorm),
            )
            .with_primitive(primitive),
        crate::api::tests::mock::raster_pipeline_backend_for_test(),
    )
}

/// Binds slot 0 of a scope to the uniform group its pipeline's interface uses.
fn bind_the_uniform(scope: &mut crate::api::command::RasterScope<'_>, layout: BindGroupLayout) {
    scope
        .set_bind_group(BindGroupIndex::new(0), &uniform_group(layout), &[])
        .expect("the offsets match the layout");
}

fn index_buffer_binding() -> BufferBinding {
    BufferBinding::new(buffer_with(BufferUsage::INDEX, 64), BufferRange::new(0, 64))
}

#[test]
fn a_pipeline_from_another_device_is_wrong_device() {
    let mut recorder = recorder();
    let mut scope = recorder
        .begin_raster(&color_scope("cross device"))
        .expect("the attachment set is legal");

    let foreign = RasterPipeline::new(
        object(52),
        other_device(),
        RasterPipelineDescriptor::new(vertex_module(62), interface_of(uniform_layout(1)))
            .with_color_target(
                ShaderLocation::new(0),
                ColorTargetState::new(TextureFormat::Rgba8Unorm),
            ),
        crate::api::tests::mock::raster_pipeline_backend_for_test(),
    );
    assert_kind(scope.set_pipeline(&foreign), RhiErrorKind::WrongDevice);
}

#[test]
fn a_pipeline_for_a_different_target_set_is_incompatible() {
    let mut recorder = recorder();
    let mut scope = recorder
        .begin_raster(&color_scope("mismatch"))
        .expect("the attachment set is legal");

    assert_kind(
        scope.set_pipeline(&mismatched_pipeline(uniform_layout(1))),
        RhiErrorKind::IncompatibleInterface,
    );

    assert!(
        scope
            .set_pipeline(&raster_pipeline(uniform_layout(1)))
            .is_ok()
    );
}

#[test]
fn a_draw_needs_a_pipeline() {
    let mut recorder = recorder();
    let mut scope = recorder
        .begin_raster(&color_scope("no pipeline"))
        .expect("the attachment set is legal");

    assert_kind(scope.draw(0..3, 0..1), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_draw_needs_the_groups_the_interface_uses() {
    let mut recorder = recorder();
    let mut scope = recorder
        .begin_raster(&color_scope("unbound group"))
        .expect("the attachment set is legal");
    scope
        .set_pipeline(&raster_pipeline(uniform_layout(1)))
        .expect("the target signatures agree");

    assert_kind(scope.draw(0..3, 0..1), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_group_of_the_wrong_layout_is_incompatible() {
    let mut recorder = recorder();
    let mut scope = recorder
        .begin_raster(&color_scope("wrong layout"))
        .expect("the attachment set is legal");
    scope
        .set_pipeline(&raster_pipeline(uniform_layout(1)))
        .expect("the target signatures agree");
    scope
        .set_bind_group(
            BindGroupIndex::new(0),
            &uniform_group(uniform_layout(2)),
            &[],
        )
        .expect("binding a group is legal on its own");

    // The compatibility identity, not the slot contents, is what a draw compares.
    assert_kind(scope.draw(0..3, 0..1), RhiErrorKind::IncompatibleInterface);
}

#[test]
fn a_draw_records_its_attachment_and_binding_uses() {
    let mut recorder = recorder();
    let layout = uniform_layout(1);
    {
        let mut scope = recorder
            .begin_raster(&color_scope("uses"))
            .expect("the attachment set is legal");
        scope
            .set_pipeline(&raster_pipeline(layout.clone()))
            .expect("the target signatures agree");
        bind_the_uniform(&mut scope, layout);
        scope.draw(0..3, 0..1).expect("the draw is legal");
        scope.end().expect("the scope had no debug group open");
    }

    let work = recorder.finish().expect("the recording is complete");
    assert!(work.work_domains().contains(LaneWorkDomains::RASTER));
    assert_eq!(work.device_identity(), device());

    let uniform = work.resource_uses().iter().find_map(|use_| match use_ {
        ResourceUse::Buffer(buffer) => {
            (buffer.access == AccessMask::UNIFORM_READ).then_some(buffer.clone())
        }
        ResourceUse::Texture(_) | ResourceUse::Frame(_) | ResourceUse::AccelerationStructure(_) => {
            None
        }
    });
    let uniform = uniform.expect("a bound uniform produces a use");
    assert_eq!(uniform.stages, PipelineScope::VERTEX);
    assert_eq!(uniform.range, BufferRange::new(0, 16));

    // Section 37.3: a draw reads and writes its color attachments, and the clear
    // at scope begin was a write of its own — so the same texture appears twice,
    // with different accesses.
    assert!(work.resource_uses().iter().any(|use_| matches!(
        use_,
        ResourceUse::Texture(texture)
            if texture.access == AccessMask::COLOR_WRITE
                && texture.intent == TextureUseIntent::ColorAttachment
    )));
    assert!(work.resource_uses().iter().any(|use_| matches!(
        use_,
        ResourceUse::Texture(texture)
            if texture.access == AccessMask::COLOR_READ.union(AccessMask::COLOR_WRITE)
    )));
}

#[test]
fn a_vertex_buffer_slot_the_input_state_reads_must_be_bound() {
    let mut recorder = recorder();
    let layout = uniform_layout(1);
    let mut scope = recorder
        .begin_raster(&color_scope("vertex input"))
        .expect("the attachment set is legal");
    scope
        .set_pipeline(&pipeline_reading_a_vertex_buffer(53, 63, layout.clone()))
        .expect("the target signatures agree");
    bind_the_uniform(&mut scope, layout);

    assert_kind(scope.draw(0..3, 0..1), RhiErrorKind::InvalidUsage);

    scope
        .set_vertex_buffer(0, &vertex_binding())
        .expect("a VERTEX buffer at slot 0 is legal");
    scope.draw(0..3, 0..1).expect("the draw is now complete");
}

#[test]
fn a_draw_that_reaches_past_a_vertex_buffer_range_is_refused() {
    let mut recorder = recorder();
    let layout = uniform_layout(1);
    let mut scope = recorder
        .begin_raster(&color_scope("out of range"))
        .expect("the attachment set is legal");
    scope
        .set_pipeline(&pipeline_reading_a_vertex_buffer(54, 64, layout.clone()))
        .expect("the target signatures agree");
    bind_the_uniform(&mut scope, layout);
    scope
        .set_vertex_buffer(
            0,
            &BufferBinding::new(buffer_with(BufferUsage::VERTEX, 64), BufferRange::new(0, 8)),
        )
        .expect("the buffer itself is a legal vertex source");

    // Three vertices at an 8-byte stride need 24 bytes and the range has 8.
    assert_kind(scope.draw(0..3, 0..1), RhiErrorKind::InvalidUsage);
    scope.draw(0..1, 0..1).expect("one vertex fits");
}

#[test]
fn an_indexed_draw_needs_an_index_buffer() {
    let mut recorder = recorder();
    let layout = uniform_layout(1);
    let mut scope = recorder
        .begin_raster(&color_scope("no indices"))
        .expect("the attachment set is legal");
    scope
        .set_pipeline(&raster_pipeline(layout.clone()))
        .expect("the target signatures agree");
    bind_the_uniform(&mut scope, layout);

    assert_kind(
        scope.draw_indexed(0..3, 0, 0..1),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn an_indexed_draw_records_the_index_read() {
    let mut recorder = recorder();
    let layout = uniform_layout(1);
    {
        let mut scope = recorder
            .begin_raster(&color_scope("indices"))
            .expect("the attachment set is legal");
        scope
            .set_pipeline(&raster_pipeline(layout.clone()))
            .expect("the target signatures agree");
        bind_the_uniform(&mut scope, layout);
        scope
            .set_index_buffer(&index_buffer_binding(), IndexFormat::Uint16)
            .expect("the format is a property of the binding, not of the pipeline");
        scope
            .draw_indexed(0..3, 0, 0..1)
            .expect("three 16-bit indices fit 64 bytes");
        scope.end().expect("the scope had no debug group open");
    }

    let work = recorder.finish().expect("the recording is complete");
    assert!(work.resource_uses().iter().any(|use_| matches!(
        use_,
        ResourceUse::Buffer(buffer) if buffer.access == AccessMask::INDEX_READ
    )));
}

#[test]
fn a_strip_topology_requires_the_index_format_it_declared() {
    let silent = strip_pipeline(55, 65, None);
    let declaring = strip_pipeline(56, 66, Some(IndexFormat::Uint32));

    // A strip pipeline that declares no index format cannot be index-drawn at all,
    // and one that declares Uint32 refuses a Uint16 draw.
    for (pipeline, format) in [
        (&silent, IndexFormat::Uint16),
        (&declaring, IndexFormat::Uint16),
    ] {
        let mut recorder = recorder();
        let layout = uniform_layout(1);
        let mut scope = recorder
            .begin_raster(&color_scope("strip"))
            .expect("the attachment set is legal");
        scope
            .set_pipeline(pipeline)
            .expect("the target signatures agree");
        bind_the_uniform(&mut scope, layout);
        scope
            .set_index_buffer(&index_buffer_binding(), format)
            .expect("the binding is legal on its own");
        assert_kind(
            scope.draw_indexed(0..3, 0, 0..1),
            RhiErrorKind::InvalidUsage,
        );
    }

    // The declared format matches the binding, so the indexed strip draw is legal.
    let mut recorder = recorder();
    let layout = uniform_layout(1);
    let mut scope = recorder
        .begin_raster(&color_scope("strip match"))
        .expect("the attachment set is legal");
    scope
        .set_pipeline(&declaring)
        .expect("the target signatures agree");
    bind_the_uniform(&mut scope, layout);
    scope
        .set_index_buffer(&index_buffer_binding(), IndexFormat::Uint32)
        .expect("Uint32 is what the pipeline declared");
    scope
        .draw_indexed(0..3, 0, 0..1)
        .expect("the strip draw is legal");
}

fn indirect_facts(multi: bool, base_vertex: bool) -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    facts.record_feature(crate::api::platform::OptionalFeature::IndirectDraw);
    if multi {
        facts.record_feature(crate::api::platform::OptionalFeature::MultiDrawIndirect);
    }
    if base_vertex {
        facts.record_feature(crate::api::platform::OptionalFeature::BaseVertex);
    }
    facts
}

#[test]
fn indirect_draw_records_the_precise_argument_range_and_actual_read() {
    let mut recorder = recorder_reporting(indirect_facts(false, false));
    let layout = uniform_layout(1);
    {
        let mut scope = recorder.begin_raster(&color_scope("indirect")).unwrap();
        scope
            .set_pipeline(&raster_pipeline(layout.clone()))
            .unwrap();
        bind_the_uniform(&mut scope, layout);
        let arguments = buffer_with(BufferUsage::INDIRECT, 32);
        scope.draw_indirect(&arguments, 4).unwrap();
        scope.end().unwrap();
    }
    let work = recorder.finish().unwrap();
    assert!(
        work.resource_uses()
            .iter()
            .any(|use_| matches!(use_, ResourceUse::Buffer(buffer)
        if buffer.range == BufferRange::new(4, 16)
            && buffer.stages == PipelineScope::VERTEX
            && buffer.access == AccessMask::INDIRECT_READ))
    );
}

#[test]
fn indexed_indirect_requires_a_bound_and_strip_compatible_index_buffer() {
    let mut recorder = recorder_reporting(indirect_facts(false, false));
    let layout = uniform_layout(1);
    let strip = strip_pipeline(91, 92, Some(IndexFormat::Uint32));
    let arguments = buffer_with(BufferUsage::INDIRECT, 20);
    let mut scope = recorder
        .begin_raster(&color_scope("indirect strip"))
        .unwrap();
    scope.set_pipeline(&strip).unwrap();
    bind_the_uniform(&mut scope, layout);
    assert_kind(
        scope.draw_indexed_indirect(&arguments, 0),
        RhiErrorKind::InvalidUsage,
    );
    scope
        .set_index_buffer(&index_buffer_binding(), IndexFormat::Uint16)
        .unwrap();
    assert_kind(
        scope.draw_indexed_indirect(&arguments, 0),
        RhiErrorKind::InvalidUsage,
    );
    scope
        .set_index_buffer(&index_buffer_binding(), IndexFormat::Uint32)
        .unwrap();
    scope.draw_indexed_indirect(&arguments, 0).unwrap();
}

#[test]
fn multi_indirect_checks_capability_stride_count_and_full_span() {
    let arguments = buffer_with(BufferUsage::INDIRECT, 32);
    let layout = uniform_layout(1);
    let mut missing = recorder_reporting(indirect_facts(false, false));
    let mut missing_scope = missing.begin_raster(&color_scope("no multi")).unwrap();
    missing_scope
        .set_pipeline(&raster_pipeline(layout.clone()))
        .unwrap();
    bind_the_uniform(&mut missing_scope, layout.clone());
    assert_kind(
        missing_scope.multi_draw_indirect(&arguments, 0, 1, 16),
        RhiErrorKind::Unsupported,
    );

    let mut recorder = recorder_reporting(indirect_facts(true, false));
    let mut scope = recorder.begin_raster(&color_scope("multi")).unwrap();
    scope
        .set_pipeline(&raster_pipeline(layout.clone()))
        .unwrap();
    bind_the_uniform(&mut scope, layout);
    assert_kind(
        scope.multi_draw_indirect(&arguments, 0, 0, 16),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        scope.multi_draw_indirect(&arguments, 0, 2, 15),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        scope.multi_draw_indirect(&arguments, 16, 2, 16),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        scope.multi_draw_indirect(&arguments, u64::MAX - 3, 1, 16),
        RhiErrorKind::InvalidUsage,
    );
    scope.multi_draw_indirect(&arguments, 0, 2, 16).unwrap();
    scope.end().unwrap();

    // Indexed multi-draw has the wider native argument record and retains the
    // ordinary indexed-strip compatibility rule.
    let indexed_arguments = buffer_with(BufferUsage::INDIRECT, 40);
    let indexed_layout = uniform_layout(1);
    let indexed_pipeline = strip_pipeline(101, 102, Some(IndexFormat::Uint32));
    let mut indexed_scope = recorder
        .begin_raster(&color_scope("multi indexed"))
        .unwrap();
    indexed_scope.set_pipeline(&indexed_pipeline).unwrap();
    bind_the_uniform(&mut indexed_scope, indexed_layout);
    indexed_scope
        .set_index_buffer(&index_buffer_binding(), IndexFormat::Uint32)
        .unwrap();
    indexed_scope
        .multi_draw_indexed_indirect(&indexed_arguments, 0, 2, 20)
        .unwrap();
    indexed_scope.end().unwrap();
}

#[test]
fn base_vertex_is_capability_gated_but_zero_remains_portable() {
    let layout = uniform_layout(1);
    let mut absent = recorder();
    let mut absent_scope = absent
        .begin_raster(&color_scope("base vertex absent"))
        .unwrap();
    absent_scope
        .set_pipeline(&raster_pipeline(layout.clone()))
        .unwrap();
    bind_the_uniform(&mut absent_scope, layout.clone());
    absent_scope
        .set_index_buffer(&index_buffer_binding(), IndexFormat::Uint16)
        .unwrap();
    absent_scope.draw_indexed(0..1, 0, 0..1).unwrap();
    assert_kind(
        absent_scope.draw_indexed(0..1, -1, 0..1),
        RhiErrorKind::Unsupported,
    );

    let mut present = recorder_reporting(indirect_facts(false, true));
    let mut present_scope = present
        .begin_raster(&color_scope("base vertex present"))
        .unwrap();
    present_scope
        .set_pipeline(&raster_pipeline(layout.clone()))
        .unwrap();
    bind_the_uniform(&mut present_scope, layout);
    present_scope
        .set_index_buffer(&index_buffer_binding(), IndexFormat::Uint16)
        .unwrap();
    present_scope.draw_indexed(0..1, -1, 0..1).unwrap();
}

#[test]
fn first_instance_is_capability_gated_but_zero_remains_portable() {
    let layout = uniform_layout(1);
    let mut absent = recorder();
    let mut absent_scope = absent
        .begin_raster(&color_scope("base instance absent"))
        .unwrap();
    absent_scope
        .set_pipeline(&raster_pipeline(layout.clone()))
        .unwrap();
    bind_the_uniform(&mut absent_scope, layout.clone());
    absent_scope.draw(0..1, 0..1).unwrap();
    assert_kind(absent_scope.draw(0..1, 1..2), RhiErrorKind::Unsupported);

    let mut facts = crate::api::capability::CapabilityFacts::empty();
    facts.record_feature(crate::api::platform::OptionalFeature::BaseInstance);
    let mut present = recorder_reporting(facts);
    let mut present_scope = present
        .begin_raster(&color_scope("base instance present"))
        .unwrap();
    present_scope
        .set_pipeline(&raster_pipeline(layout.clone()))
        .unwrap();
    bind_the_uniform(&mut present_scope, layout);
    present_scope.draw(0..1, 1..2).unwrap();
}

#[test]
fn an_open_debug_group_refuses_end_and_poisons_the_recording() {
    let mut recorder = recorder();
    {
        let mut scope = recorder
            .begin_raster(&color_scope("open group"))
            .expect("the attachment set is legal");
        scope
            .push_debug_group("inside the pass")
            .expect("pushing is always legal");

        // Section 29.3: a failed scope finalization poisons, and the drop that
        // follows this refusal carries it out.
        assert_kind(scope.end(), RhiErrorKind::InvalidUsage);
    }

    assert_kind(recorder.finish().map(|_| ()), RhiErrorKind::InvalidUsage);
}

#[test]
fn the_portable_viewport_and_scissor_default_to_the_attachment_extent() {
    // Section 32.3 fixes the default as the attachment extent, and not as whatever
    // a backend happens to select.
    let mut recorder = recorder();
    let mut scope = recorder
        .begin_raster(&color_scope("defaults"))
        .expect("the attachment set is legal");

    assert_eq!(
        scope.effective_viewport(),
        Viewport::new(0.0, 0.0, 4.0, 4.0, 0.0, 1.0)
    );
    assert_eq!(scope.effective_scissor(), Rect::new(0, 0, 4, 4));
    assert_eq!(scope.stencil_reference(), 0);

    scope
        .set_scissor(Rect::new(1, 1, 2, 2))
        .expect("a rect inside the extent is legal");
    assert_eq!(scope.effective_scissor(), Rect::new(1, 1, 2, 2));

    assert_kind(
        scope.set_viewport(Viewport::new(0.0, 0.0, 2.0, 2.0, 1.0, 0.0)),
        RhiErrorKind::InvalidUsage,
    );
}
