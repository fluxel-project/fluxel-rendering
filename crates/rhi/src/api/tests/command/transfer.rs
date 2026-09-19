//! Section 35: upload and readback encoding.

use super::*;
use crate::api::resource::transfer::ReadbackRequest;
use crate::api::tests::fixture;

#[test]
fn a_readback_from_another_device_is_wrong_device_before_anything_else() {
    // Section 3.1's O(1) step runs first, so a cross-device readback is reported as
    // a cross-device readback and not as whatever the later checks would say.
    let mut recorder = recorder();
    let request = ReadbackRequest::Buffer {
        label: Label::default(),
        src: fixture::buffer(
            object(13),
            other_device(),
            BufferDescriptor::new(64, BufferUsage::COPY_SRC),
        ),
        range: BufferRange::new(0, 64),
    };
    assert_kind(
        recorder.encode_readback(request).map(|_| ()),
        RhiErrorKind::WrongDevice,
    );
}

#[test]
fn a_readback_on_a_device_without_the_route_is_unsupported() {
    // The identity step passed and every portable rule passed; what is left is the
    // device's own answer, and this device reports no route at all. `Unsupported`
    // rather than `InvalidUsage`, because section 9.4 makes "this device cannot"
    // and "you described it wrongly" different answers.
    let mut recorder = recorder();
    let request = ReadbackRequest::Buffer {
        label: Label::default(),
        src: buffer_with(BufferUsage::COPY_SRC, 64),
        range: BufferRange::new(0, 64),
    };
    assert_kind(
        recorder.encode_readback(request).map(|_| ()),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn a_readback_whose_range_breaks_the_device_alignment_is_refused() {
    // The same request on a device that *does* report the route, with a 4-byte
    // alignment. The range is valid for the buffer and misaligned for the copy,
    // which is section 12.4's alignment rule and the reason `encode_readback`
    // needed the device snapshot at all.
    let mut recorder = recorder_reporting(facts_with_buffer_copy_route(4, 4));
    let request = ReadbackRequest::Buffer {
        label: Label::default(),
        src: buffer_with(BufferUsage::COPY_SRC, 64),
        range: BufferRange::new(1, 64),
    };
    assert_kind(
        recorder.encode_readback(request).map(|_| ()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn an_aligned_readback_is_accepted_and_its_ticket_reports_the_device() {
    let mut recorder = recorder_reporting(facts_with_buffer_copy_route(4, 4));
    let request = ReadbackRequest::Buffer {
        label: Label::default(),
        src: buffer_with(BufferUsage::COPY_SRC, 64),
        range: BufferRange::new(0, 64),
    };
    let ticket = recorder
        .encode_readback(request)
        .expect("the range satisfies the alignment this device reports");

    assert_eq!(ticket.device_identity(), device());
    // Section 18.2: an encoded but unsubmitted readback is `NotSubmitted`, and the
    // ticket says so rather than reporting a state no submission produced.
    assert_eq!(
        ticket.status(),
        crate::api::resource::transfer::ReadbackStatus::NotSubmitted
    );
}

#[test]
fn a_readback_source_without_copy_src_is_refused_before_the_route() {
    // The portable half of section 18.1 runs before the device's route question,
    // so the answer is `InvalidUsage` even on a device that reports no route: the
    // caller's mistake is reported as the caller's mistake.
    let mut recorder = recorder();
    let request = ReadbackRequest::Buffer {
        label: Label::default(),
        src: buffer_with(BufferUsage::COPY_DST, 64),
        range: BufferRange::new(0, 64),
    };
    assert_kind(
        recorder.encode_readback(request).map(|_| ()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_texture_readback_asks_the_texture_to_buffer_route() {
    // The buffer route is present and the texture route is not, so this proves the
    // texture readback asked about the texture route rather than about the one
    // fact table entry that happens to exist.
    let mut recorder = recorder_reporting(facts_with_buffer_copy_route(4, 4));
    let request = ReadbackRequest::Texture {
        label: Label::default(),
        src: renderable_texture(TextureFormat::Rgba8Unorm),
        subresource: color_layers(1),
        origin: origin(),
        extent: Extent3d::d2(4, 4),
    };
    assert_kind(
        recorder.encode_readback(request).map(|_| ()),
        RhiErrorKind::Unsupported,
    );
}
