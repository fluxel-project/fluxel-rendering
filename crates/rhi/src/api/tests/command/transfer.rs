//! Section 35: upload and readback encoding.

use super::*;
use crate::api::resource::transfer::ReadbackRequest;

#[test]
fn a_readback_from_another_device_is_wrong_device_before_anything_else() {
    // Section 3.1's O(1) step runs first, so a cross-device readback is reported as
    // a cross-device readback and not as whatever the later checks would say.
    let mut recorder = recorder();
    let request = ReadbackRequest::Buffer {
        label: Label::default(),
        src: Buffer::new(
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
#[should_panic(expected = "encode_readback must check")]
fn a_readback_on_this_device_stops_at_the_device_layout() {
    // The identity step passed; what is missing is the device's copy-layout
    // alignment, which `resource::transfer::readback` takes as a parameter.
    let mut recorder = recorder();
    let request = ReadbackRequest::Buffer {
        label: Label::default(),
        src: buffer_with(BufferUsage::COPY_SRC, 64),
        range: BufferRange::new(0, 64),
    };
    let _ = recorder.encode_readback(request);
}
