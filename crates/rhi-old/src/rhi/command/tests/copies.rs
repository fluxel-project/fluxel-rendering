//! Section 34: copy route validation.
//!
//! A copy is the one command family whose operands are byte ranges and texel
//! extents rather than a pipeline, so its rules are arithmetic: a range must fit
//! inside its buffer, a command may not read and write the same bytes, a row
//! stride must cover one texture row, and a multi-slice copy must describe the
//! gap between slices.

use super::*;
// ---------------------------------------------------------------------------
// Section 34: copy route validation
// ---------------------------------------------------------------------------

#[test]
fn a_buffer_copy_range_outside_its_buffer_is_refused() {
    let mock = Mock::new();
    let (src, dst) = copy_pair(&mock, 64);
    let mut recorder = mock.recorder();

    let error = recorder
        .copy_buffer(&BufferCopy {
            src,
            src_offset: 32,
            dst,
            dst_offset: 0,
            size: 64,
        })
        .expect_err("a copy range must fit inside both buffers");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_buffer_copy_that_overlaps_itself_is_refused() {
    let mock = Mock::new();
    let buffer = mock.buffer(64, BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST));
    let mut recorder = mock.recorder();

    let error = recorder
        .copy_buffer(&BufferCopy {
            src: buffer.clone(),
            src_offset: 0,
            dst: buffer,
            dst_offset: 32,
            size: 64,
        })
        .expect_err("one command may not read and write the same bytes");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_copy_whose_row_is_smaller_than_one_texture_row_is_refused() {
    let mock = Mock::new();
    let texture = mock.texture(4, 4, COLOR_FORMAT, TextureUsage::COPY_DST);
    let buffer = mock.buffer(4096, BufferUsage::COPY_SRC);
    let mut recorder = mock.recorder();

    // Four texels of Rgba8Unorm are sixteen bytes, so eight bytes per row cannot
    // describe a single row of this texture.
    let error = recorder
        .copy_buffer_to_texture(&BufferTextureCopy {
            buffer,
            buffer_offset: 0,
            bytes_per_row: 8,
            rows_per_image: 16,
            texture,
            texture_subresource: color_layers(),
            texture_origin: Origin3d::ZERO,
            extent: Extent3d::d2(4, 4),
        })
        .expect_err("bytes_per_row must cover one texture row");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_copy_whose_row_pitch_is_not_aligned_is_refused() {
    let mock = Mock::new();
    let texture = mock.texture(4, 4, COLOR_FORMAT, TextureUsage::COPY_DST);
    let buffer = mock.buffer(4096, BufferUsage::COPY_SRC);
    let mut recorder = mock.recorder();

    // Twenty bytes covers one sixteen-byte row of this texture, so the
    // row-stride check above is satisfied; what this input violates is the
    // route's `TexelCopyLayoutLimits::bytes_per_row_alignment`, the row pitch a
    // command-buffer texel copy may have.
    let error = recorder
        .copy_buffer_to_texture(&BufferTextureCopy {
            buffer,
            buffer_offset: 0,
            bytes_per_row: 20,
            rows_per_image: 16,
            texture,
            texture_subresource: color_layers(),
            texture_origin: Origin3d::ZERO,
            extent: Extent3d::d2(4, 4),
        })
        .expect_err("bytes_per_row must satisfy the route's row alignment");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("copy_buffer_to_texture"));
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_copy_whose_buffer_offset_is_not_aligned_is_refused() {
    let mock = Mock::new();
    let texture = mock.texture(4, 4, COLOR_FORMAT, TextureUsage::COPY_DST);
    let buffer = mock.buffer(4096, BufferUsage::COPY_SRC);
    let mut recorder = mock.recorder();

    // The offset is inside the buffer and the row pitch is aligned; what is
    // missing is the route's `TexelCopyLayoutLimits::buffer_offset_alignment`.
    let error = recorder
        .copy_buffer_to_texture(&BufferTextureCopy {
            buffer,
            buffer_offset: 8,
            bytes_per_row: 256,
            rows_per_image: 16,
            texture,
            texture_subresource: color_layers(),
            texture_origin: Origin3d::ZERO,
            extent: Extent3d::d2(4, 4),
        })
        .expect_err("buffer_offset must satisfy the route's offset alignment");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("copy_buffer_to_texture"));
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_copy_whose_route_the_device_does_not_declare_is_unsupported() {
    let mock = Mock::new();
    let texture = mock.texture(4, 4, COLOR_FORMAT, TextureUsage::COPY_DST);
    let buffer = mock.buffer(4096, BufferUsage::COPY_SRC);
    let mut recorder = mock.recorder();

    // The route key includes the aspect, and this device declares routes for the
    // color and depth aspects of this format but none for its stencil aspect.
    // Creation is not gated on the route table — a texture's aspect set and its
    // copy routes are separate declarations — so every descriptor fact below is
    // creatable, and the refusal can only come from the route. Section 34.4 is
    // why it must come from there rather than from a fallback: an unsupported
    // route is `Unsupported`, and a backend may not quietly substitute one.
    //
    // The accepted copy first is the control: it has the same buffer, offset,
    // stride, extent, and texture, and differs only in the aspect named. Without
    // it, an `Unsupported` from some other check would look like this one.
    recorder
        .copy_buffer_to_texture(&BufferTextureCopy {
            buffer: buffer.clone(),
            buffer_offset: 0,
            bytes_per_row: 256,
            rows_per_image: 16,
            texture: texture.clone(),
            texture_subresource: color_layers(),
            texture_origin: Origin3d::ZERO,
            extent: Extent3d::d2(4, 4),
        })
        .expect("the color aspect of this copy has a declared route");

    let error = recorder
        .copy_buffer_to_texture(&BufferTextureCopy {
            buffer,
            buffer_offset: 0,
            bytes_per_row: 256,
            rows_per_image: 16,
            texture,
            texture_subresource: TextureSubresourceLayers {
                aspect: TextureAspect::Stencil,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            texture_origin: Origin3d::ZERO,
            extent: Extent3d::d2(4, 4),
        })
        .expect_err("a route the device does not declare is not a legal copy");
    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert_eq!(error.operation(), Some("copy_buffer_to_texture"));
    assert_eq!(
        journal_count(&mock, "recorder:record:CopyBufferToTexture"),
        1,
        "only the control reached the backend; the refused copy did not"
    );
    assert!(
        !recorder.is_poisoned(),
        "a capability error rejects one command and leaves the recorder usable"
    );
}

#[test]
fn a_multi_slice_copy_without_rows_per_image_is_refused() {
    let mock = Mock::new();
    let texture = mock
        .device()
        .create_texture(&TextureDescriptor::new_3d(
            4,
            4,
            2,
            COLOR_FORMAT,
            TextureUsage::COPY_DST,
        ))
        .expect("the mock device creates a 3D texture");
    let buffer = mock.buffer(4096, BufferUsage::COPY_SRC);
    let mut recorder = mock.recorder();

    let error = recorder
        .copy_buffer_to_texture(&BufferTextureCopy {
            buffer,
            buffer_offset: 0,
            bytes_per_row: 16,
            rows_per_image: 0,
            texture,
            texture_subresource: color_layers(),
            texture_origin: Origin3d::ZERO,
            extent: Extent3d::d3(4, 4, 2),
        })
        .expect_err("a multi-slice copy needs an inter-slice stride");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("copy_buffer_to_texture"));
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_texture_copy_that_changes_format_is_refused() {
    let mock = Mock::new();
    let src = mock.texture(4, 4, COLOR_FORMAT, TextureUsage::COPY_SRC);
    let dst = mock.texture(4, 4, TextureFormat::Rgba8UnormSrgb, TextureUsage::COPY_DST);
    let mut recorder = mock.recorder();

    let error = recorder
        .copy_texture(&TextureCopy {
            src,
            src_subresource: color_layers(),
            src_origin: Origin3d::ZERO,
            dst,
            dst_subresource: color_layers(),
            dst_origin: Origin3d::ZERO,
            extent: Extent3d::d2(4, 4),
        })
        .expect_err("a texture copy cannot change format");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("copy_texture"));
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_resolve_of_a_texture_onto_itself_is_refused() {
    let mock = Mock::new();
    let texture = mock.texture(
        4,
        4,
        COLOR_FORMAT,
        TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
    );
    let mut recorder = mock.recorder();

    let error = recorder
        .resolve_texture(&TextureResolve {
            src: texture.clone(),
            src_subresource: color_layers(),
            src_origin: Origin3d::ZERO,
            dst: texture,
            dst_subresource: color_layers(),
            dst_origin: Origin3d::ZERO,
            extent: Extent3d::d2(4, 4),
        })
        .expect_err("a resolve source and destination must differ");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(!recorder.is_poisoned());
}

#[test]
fn a_copy_between_scopes_is_recorded_alongside_them() {
    let mock = Mock::new();
    let (src, dst) = copy_pair(&mock, 64);
    let view = color_view(&mock);
    let descriptor = one_color_scope(&view);
    let pipeline = plain_pipeline(&mock, COLOR_FORMAT);

    let mut recorder = mock.recorder();
    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope.set_pipeline(&pipeline).expect("the pipeline binds");
        scope.draw(0..3, 0..1).expect("a draw is recorded");
        scope.end().expect("the scope closes");
    }
    recorder
        .copy_buffer(&BufferCopy {
            src,
            src_offset: 0,
            dst,
            dst_offset: 0,
            size: 64,
        })
        .expect("a copy is recorded between scopes");

    let work = recorder.finish().expect("the recording finalizes");
    assert!(work.work_domains().contains(LaneWorkDomains::RASTER));
    assert!(work.work_domains().contains(LaneWorkDomains::COPY));
    assert_eq!(
        work.commands().len(),
        5,
        "begin, set pipeline, draw, end, then the copy"
    );
    assert!(mock.saw("recorder:record:Draw"));
    assert!(mock.saw("recorder:record:CopyBuffer"));
}

