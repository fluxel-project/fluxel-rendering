//! Section 33: draw range and vertex/index validation.
//!
//! Every rule here is a range that would otherwise be handed to a backend as an
//! out-of-bounds read. The recorder refuses it while the recording can still
//! continue, so each test also asserts that the recorder stayed usable.

use super::*;
// ---------------------------------------------------------------------------
// Section 33: draw range validation
// ---------------------------------------------------------------------------

#[test]
fn an_inverted_draw_range_is_refused() {
    let mock = Mock::new();
    let view = color_view(&mock);
    let pipeline = plain_pipeline(&mock, COLOR_FORMAT);
    let descriptor = one_color_scope(&view);

    let mut recorder = mock.recorder();
    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope.set_pipeline(&pipeline).expect("the pipeline binds");

        let error = scope
            .draw(range(3, 1), range(0, 1))
            .expect_err("an inverted vertex range is refused");
        assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
        assert_eq!(error.operation(), Some("draw"));

        let error = scope
            .draw(range(0, 1), range(2, 1))
            .expect_err("an inverted instance range is refused");
        assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);

        scope.end().expect("the scope still closes");
    }
    assert!(!recorder.is_poisoned());
    assert_eq!(
        journal_count(&mock, "recorder:record:Draw"),
        0,
        "both draws were refused before the backend saw them"
    );
}

#[test]
fn a_draw_without_a_bound_pipeline_is_refused() {
    let mock = Mock::new();
    let view = color_view(&mock);
    let descriptor = one_color_scope(&view);

    let mut recorder = mock.recorder();
    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        let error = scope
            .draw(0..1, 0..1)
            .expect_err("a draw with no pipeline bound is refused");
        assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
        scope.end().expect("the scope still closes");
    }
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_vertex_fetch_past_the_end_of_its_binding_is_refused() {
    let mock = Mock::new();
    let view = color_view(&mock);
    let pipeline = fetch_pipeline(&mock, COLOR_FORMAT);
    let descriptor = one_color_scope(&view);
    // Eight bytes of stride, sixteen bytes exposed: two vertices fit.
    let vertices = mock.buffer(16, BufferUsage::VERTEX);

    let mut recorder = mock.recorder();
    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope.set_pipeline(&pipeline).expect("the pipeline binds");
        scope
            .set_vertex_buffer(0, &BufferBinding::new(vertices.clone(), BufferRange::new(0, 16)))
            .expect("a VERTEX buffer with an explicit range binds");

        let error = scope
            .draw(0..4, 0..1)
            .expect_err("four vertices at stride eight need thirty-two bytes");
        assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
        assert_eq!(error.operation(), Some("draw"));

        scope
            .draw(0..2, 0..1)
            .expect("two vertices fit inside the exposed range");
        scope.end().expect("the scope closes");
    }
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_draw_with_a_declared_vertex_slot_left_unbound_is_refused() {
    let mock = Mock::new();
    let view = color_view(&mock);
    let pipeline = fetch_pipeline(&mock, COLOR_FORMAT);
    let descriptor = one_color_scope(&view);

    let mut recorder = mock.recorder();
    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope.set_pipeline(&pipeline).expect("the pipeline binds");
        let error = scope
            .draw(0..1, 0..1)
            .expect_err("vertex slot 0 is declared by the pipeline but not bound");
        assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
        scope.end().expect("the scope still closes");
    }
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_strip_pipeline_refuses_a_mismatched_index_format() {
    let mock = Mock::new();
    let view = color_view(&mock);
    let pipeline = strip_pipeline(&mock, COLOR_FORMAT);
    let descriptor = one_color_scope(&view);
    let indices = mock.buffer(64, BufferUsage::INDEX);

    let mut recorder = mock.recorder();
    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope.set_pipeline(&pipeline).expect("the pipeline binds");
        scope
            .set_index_buffer(
                &BufferBinding::new(indices.clone(), BufferRange::new(0, 64)),
                IndexFormat::Uint32,
            )
            .expect("an INDEX buffer binds");

        let error = scope
            .draw_indexed(0..3, 0, 0..1)
            .expect_err("the pipeline was built for 16-bit strip indices");
        assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
        assert_eq!(error.operation(), Some("draw_indexed"));
        scope.end().expect("the scope still closes");
    }
    assert!(!recorder.is_poisoned());
}

#[test]
fn an_indexed_draw_without_an_index_buffer_is_refused() {
    let mock = Mock::new();
    let view = color_view(&mock);
    let pipeline = plain_pipeline(&mock, COLOR_FORMAT);
    let descriptor = one_color_scope(&view);

    let mut recorder = mock.recorder();
    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope.set_pipeline(&pipeline).expect("the pipeline binds");
        let error = scope
            .draw_indexed(0..3, 0, 0..1)
            .expect_err("an indexed draw needs a bound index buffer");
        assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
        scope.end().expect("the scope still closes");
    }
    assert!(!recorder.is_poisoned());
}

