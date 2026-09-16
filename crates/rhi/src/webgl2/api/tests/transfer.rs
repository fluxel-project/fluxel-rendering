//! Contract tests for buffer upload, readback and copy validation,
//! including that a rejected transfer never records a driver command.

use super::*;

#[test]
fn mock_buffer_upload_and_readback_check_bounds_and_exact_lengths() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let buffer = api
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE | GlBufferUsage::COPY_DESTINATION,
        })
        .expect("buffer");
    let range = |offset, size| GlBufferRange {
        buffer,
        offset,
        size,
    };

    api.upload_buffer(range(0, 16), &[7; 16])
        .expect("subrange upload");
    let bytes = api.read_buffer(range(16, 16)).expect("readback");
    assert_eq!(bytes.len(), 16);
    assert!(api.calls().ends_with(&[
        MockCall::UploadBuffer {
            buffer,
            offset: 0,
            size: 16,
        },
        MockCall::ReadBuffer {
            buffer,
            offset: 16,
            size: 16,
        },
    ]));

    assert!(api.upload_buffer(range(0, 0), &[]).is_err());
    assert!(api.upload_buffer(range(16, 16), &[0; 15]).is_err());
    assert!(api.upload_buffer(range(48, 17), &[0; 17]).is_err());
    assert!(api.read_buffer(range(48, 17)).is_err());
    assert!(api.read_buffer(range(0, 0)).is_err());
}

#[test]
fn mock_resource_and_copy_validation_do_not_emit_driver_commands() {
    let mut unsupported = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    assert!(
        unsupported
            .create_buffer_resource(GlBufferDesc {
                size: 64,
                usage: GlBufferUsage::STORAGE,
            })
            .is_err()
    );
    let first = unsupported
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("rejected allocation did not consume a slot");
    assert_eq!(first.slot, 0);

    let source = unsupported
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("source");
    let destination = unsupported
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("destination without copy-destination usage");
    let trace_len = unsupported.calls().len();
    assert!(
        unsupported
            .copy_buffer_range(
                GlBufferRange {
                    buffer: source,
                    offset: 0,
                    size: 16,
                },
                GlBufferRange {
                    buffer: destination,
                    offset: 0,
                    size: 16,
                },
            )
            .is_err()
    );
    assert!(
        !unsupported.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::CopyBuffer { .. }))
    );
}
