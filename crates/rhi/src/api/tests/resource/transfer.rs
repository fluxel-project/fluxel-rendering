//! Sections 17 and 18: upload and readback.

use super::*;
use crate::api::error::RhiErrorKind;
use crate::api::format::TextureFormat;
use crate::api::identity::Label;
use crate::api::resource::buffer::{
    BufferRange, BufferSupport, BufferSupportLimits, BufferSupportQuery, BufferUsage,
};
use crate::api::resource::subresource::{
    HostTexelLayout, Origin3d, TextureAspect, TextureSubresourceLayers,
};
use crate::api::resource::texture::{Extent3d, TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::readback::{
    validate_buffer_readback, validate_texture_readback,
};
use crate::api::resource::transfer::upload::{validate_buffer_upload, validate_texture_upload};
use crate::api::resource::transfer::{
    BufferUploadDescriptor, ReadbackRequest, ReadbackStatus, ReadbackTexelLayout, ReadbackTicket,
    ReadbackViewData, TextureUploadDescriptor, UploadDescriptor, UploadJob,
};
use crate::api::submission::CompletionPoint;
use crate::api::tests::fixture;
use crate::base::mock::{device_for_test, paired_device_for_test};

// Shape-check the frozen async boundary without requiring a particular async
// runtime in this contract-test crate.
#[allow(dead_code)]
async fn async_readback_shape(ticket: &ReadbackTicket) -> crate::api::RhiResult<usize> {
    let view = ticket.read().await?;
    Ok(match view.data() {
        ReadbackViewData::Buffer { bytes } | ReadbackViewData::Texture { bytes, .. } => bytes.len(),
    })
}

#[test]
fn a_legal_buffer_upload_is_accepted() {
    let dst = buffer_with(BufferUsage::COPY_DST, 64);
    let descriptor = BufferUploadDescriptor {
        label: Label::default(),
        dst,
        dst_offset: 0,
        bytes: vec![0u8; 16].into(),
    };
    assert!(validate_buffer_upload(&descriptor, device(), &copy_limits()).is_ok());
}

#[test]
fn a_buffer_upload_must_name_a_copy_destination() {
    // Section 11.1: "COPY_DST not declared -> cannot use Upload / copy
    // destination", and the backend may not bypass it.
    let dst = buffer_with(BufferUsage::COPY_SRC, 64);
    let descriptor = BufferUploadDescriptor {
        label: Label::default(),
        dst,
        dst_offset: 0,
        bytes: vec![0u8; 16].into(),
    };
    assert_kind(
        validate_buffer_upload(&descriptor, device(), &copy_limits()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_buffer_upload_must_carry_bytes_and_fit_the_destination() {
    let dst = buffer_with(BufferUsage::COPY_DST, 64);

    let empty = BufferUploadDescriptor {
        label: Label::default(),
        dst: dst.clone(),
        dst_offset: 0,
        bytes: Vec::new().into(),
    };
    assert_kind(
        validate_buffer_upload(&empty, device(), &copy_limits()),
        RhiErrorKind::InvalidUsage,
    );

    // Exactly at the end is legal; one byte past it is not.
    let fits = BufferUploadDescriptor {
        label: Label::default(),
        dst: dst.clone(),
        dst_offset: 48,
        bytes: vec![0u8; 16].into(),
    };
    assert!(validate_buffer_upload(&fits, device(), &copy_limits()).is_ok());

    let overflows = BufferUploadDescriptor {
        label: Label::default(),
        dst: dst.clone(),
        dst_offset: 52,
        bytes: vec![0u8; 16].into(),
    };
    assert_kind(
        validate_buffer_upload(&overflows, device(), &copy_limits()),
        RhiErrorKind::InvalidUsage,
    );

    // The alignment rules of section 17.3, on both sides.
    let misaligned_offset = BufferUploadDescriptor {
        label: Label::default(),
        dst: dst.clone(),
        dst_offset: 1,
        bytes: vec![0u8; 16].into(),
    };
    assert_kind(
        validate_buffer_upload(&misaligned_offset, device(), &copy_limits()),
        RhiErrorKind::InvalidUsage,
    );

    let misaligned_size = BufferUploadDescriptor {
        label: Label::default(),
        dst: dst.clone(),
        dst_offset: 0,
        bytes: vec![0u8; 6].into(),
    };
    assert_kind(
        validate_buffer_upload(&misaligned_size, device(), &copy_limits()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_buffer_upload_from_another_device_is_wrong_device() {
    let dst = buffer_with(BufferUsage::COPY_DST, 64);
    let descriptor = BufferUploadDescriptor {
        label: Label::default(),
        dst,
        dst_offset: 0,
        bytes: vec![0u8; 16].into(),
    };
    assert_kind(
        validate_buffer_upload(&descriptor, identity(9), &copy_limits()),
        RhiErrorKind::WrongDevice,
    );
}

#[test]
fn the_upload_verb_refuses_a_foreign_buffer_before_it_asks_the_route() {
    // The validator test above cannot see this, because the ordering it checks is
    // the *verb's*. `create_buffer_upload` has to read the copy route before it can
    // call the validator at all — the validator takes the alignment limits that read
    // produces — so section 3.1's comparison has to be established by the verb, which
    // is what its `# Errors` section promises.
    //
    // The difference is observable in the kind, not just in the order: with the
    // identity comparison ahead of the route read a foreign buffer is `WrongDevice`,
    // and with it only inside the validator the same call would answer `Unsupported`
    // on a device stating no buffer-to-buffer route — a verdict about the device's
    // routes, handed to a caller whose actual error was passing someone else's buffer.
    //
    // It is one of three refusals on these verbs that are reachable on today's
    // tree: the other two are the lost-device refusal tested below and the two
    // route refusals, which became reachable once the capability snapshot was
    // built and this verb could ask a device its own copy route. What still stops
    // an otherwise-legal upload is the absent backend staging path.
    let live = device_for_test(device());
    let foreign = fixture::buffer(
        object(91),
        identity(9),
        BufferDescriptor::new(64, BufferUsage::COPY_DST),
    );
    let error = match live.create_buffer_upload(BufferUploadDescriptor {
        label: Label::default(),
        dst: foreign,
        dst_offset: 0,
        bytes: vec![0u8; 16].into(),
    }) {
        Ok(_) => panic!("a buffer from another device must be refused"),
        Err(error) => error,
    };
    assert_eq!(
        error.kind(),
        RhiErrorKind::WrongDevice,
        "{}",
        error.message()
    );
}

/// Ownership is answered before liveness, and the difference is observable.
///
/// Section 6.5 asks two different questions about a handle that reaches a
/// device — "is this yours" and "are you still alive" — and gives two different
/// answers: `WrongDevice` when the handle is passed to a device that is not its
/// own, `DeviceLost` when it is used through its lost original. Both are true of
/// a foreign buffer on a lost device, so the order decides which kind the caller
/// is handed, and a caller that branches on the kind acts differently.
///
/// Section 3.1 settles that order for the wrong-device half: the O(1) identity
/// comparison is the first thing a public operation does. So this test asserts
/// the *kind* on the doubly-wrong call, which is the only way the ordering is
/// visible from outside the crate.
#[test]
fn a_foreign_buffer_stays_wrong_device_even_on_a_lost_device() {
    let (lost, native) = paired_device_for_test(device());
    native.mark_lost(crate::api::platform::DeviceLossInfo::new(
        "the device was lost before the call".to_string(),
    ));

    let foreign = fixture::buffer(
        object(92),
        identity(9),
        BufferDescriptor::new(64, BufferUsage::COPY_DST),
    );
    let error = match lost.create_buffer_upload(BufferUploadDescriptor {
        label: Label::default(),
        dst: foreign,
        dst_offset: 0,
        bytes: vec![0u8; 16].into(),
    }) {
        Ok(_) => panic!("a buffer from another device must be refused"),
        Err(error) => error,
    };
    assert_eq!(
        error.kind(),
        RhiErrorKind::WrongDevice,
        "the caller's mistake is the packet, so the ownership verdict comes first: {}",
        error.message()
    );
}

/// A 4x2 RGBA8 upload fixture: one image, a tightly packed row pitch, 32 bytes.
fn texture_upload_fixture(bytes: usize) -> TextureUploadDescriptor {
    TextureUploadDescriptor {
        label: Label::default(),
        dst: texture_with(TextureDescriptor::new_2d(
            4,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_DST,
        )),
        subresource: TextureSubresourceLayers {
            aspect: TextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        origin: Origin3d { x: 0, y: 0, z: 0 },
        extent: Extent3d::d2(4, 2),
        source_layout: HostTexelLayout {
            bytes_per_row: 16,
            rows_per_image: 2,
        },
        bytes: vec![0u8; bytes].into(),
    }
}

#[test]
fn a_legal_texture_upload_is_accepted() {
    assert!(validate_texture_upload(&texture_upload_fixture(32), device()).is_ok());
}

#[test]
fn a_texture_upload_must_fit_its_source_and_its_region() {
    // One byte short of the last copied texel.
    assert_kind(
        validate_texture_upload(&texture_upload_fixture(31), device()),
        RhiErrorKind::InvalidUsage,
    );

    // A region past the edge of the mip level.
    let mut past_the_edge = texture_upload_fixture(32);
    past_the_edge.origin = Origin3d { x: 1, y: 0, z: 0 };
    assert_kind(
        validate_texture_upload(&past_the_edge, device()),
        RhiErrorKind::InvalidUsage,
    );

    // A mip level the texture does not have.
    let mut past_the_mips = texture_upload_fixture(32);
    past_the_mips.subresource.mip_level = 1;
    assert_kind(
        validate_texture_upload(&past_the_mips, device()),
        RhiErrorKind::InvalidUsage,
    );

    // An aspect the format does not carry.
    let mut wrong_aspect = texture_upload_fixture(32);
    wrong_aspect.subresource.aspect = TextureAspect::Depth;
    assert_kind(
        validate_texture_upload(&wrong_aspect, device()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_texture_upload_must_name_a_copy_destination_and_a_single_sample() {
    let mut no_copy_dst = texture_upload_fixture(32);
    no_copy_dst.dst = texture_with(simple_texture_descriptor(TextureUsage::SAMPLED));
    assert_kind(
        validate_texture_upload(&no_copy_dst, device()),
        RhiErrorKind::InvalidUsage,
    );

    // A multisampled destination cannot be an upload target for these routes.
    let mut multisampled = texture_upload_fixture(32);
    multisampled.dst = texture_with(
        TextureDescriptor::new_2d(4, 2, TextureFormat::Rgba8Unorm, TextureUsage::COPY_DST)
            .with_sample_count(4),
    );
    assert_kind(
        validate_texture_upload(&multisampled, device()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_texture_upload_is_not_held_to_native_staging_alignment() {
    // Section 17.3: "Upload does not require the caller to meet native staging
    // alignment". A 16-byte row pitch is legal even though every native copy
    // footprint wants 256, because upload may repack through private staging.
    assert!(validate_texture_upload(&texture_upload_fixture(32), device()).is_ok());
}

#[test]
fn an_upload_job_carries_its_descriptor_and_identity() {
    let descriptor = UploadDescriptor::Buffer(BufferUploadDescriptor {
        label: Label::default(),
        dst: buffer_with(BufferUsage::COPY_DST, 64),
        dst_offset: 0,
        bytes: vec![1u8; 8].into(),
    });
    let job = UploadJob::new(object(61), device(), descriptor);

    assert_eq!(job.id(), object(61));
    assert_eq!(job.device_identity(), device());
    match job.descriptor() {
        UploadDescriptor::Buffer(buffer) => {
            assert_eq!(buffer.bytes.len(), 8);
            assert_eq!(buffer.dst_offset, 0);
        }
        UploadDescriptor::Texture(_) => panic!("the job described a buffer upload"),
    }

    // The retained payload is shareable: encoding the same job twice is two
    // independent mutations of the same bytes (section 17.2).
    let clone = job.clone();
    assert_eq!(clone.id(), job.id());
    match (job.descriptor(), clone.descriptor()) {
        (UploadDescriptor::Buffer(first), UploadDescriptor::Buffer(second)) => {
            assert!(std::sync::Arc::ptr_eq(&first.bytes, &second.bytes));
        }
        _ => panic!("both descriptors name a buffer upload"),
    }
}

#[test]
fn a_readback_request_names_a_buffer_range_or_a_texture_region() {
    let buffer_request = ReadbackRequest::Buffer {
        label: Label::default(),
        src: buffer_with(BufferUsage::COPY_SRC, 64),
        range: BufferRange::new(0, 16),
    };
    match &buffer_request {
        ReadbackRequest::Buffer { range, .. } => assert_eq!(*range, BufferRange::new(0, 16)),
        ReadbackRequest::Texture { .. } => panic!("this request names a buffer"),
    }

    let texture_request = ReadbackRequest::Texture {
        label: Label::default(),
        src: texture_with(TextureDescriptor::new_2d(
            4,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_SRC,
        )),
        subresource: TextureSubresourceLayers {
            aspect: TextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        origin: Origin3d { x: 0, y: 0, z: 0 },
        extent: Extent3d::d2(4, 2),
    };
    match &texture_request {
        ReadbackRequest::Texture { extent, .. } => assert_eq!(*extent, Extent3d::d2(4, 2)),
        ReadbackRequest::Buffer { .. } => panic!("this request names a texture"),
    }
}

#[test]
fn a_buffer_readback_needs_copy_source_usage_and_a_valid_range() {
    let src = buffer_with(BufferUsage::COPY_SRC, 64);
    assert!(validate_buffer_readback(&src, BufferRange::new(0, 64), device()).is_ok());

    assert_kind(
        validate_buffer_readback(&src, BufferRange::new(0, 65), device()),
        RhiErrorKind::InvalidUsage,
    );

    // A readback of a buffer nobody can read: section 11.1's "COPY_SRC not
    // declared -> cannot use Readback / copy source".
    let write_only = buffer_with(BufferUsage::COPY_DST, 64);
    assert_kind(
        validate_buffer_readback(&write_only, BufferRange::new(0, 16), device()),
        RhiErrorKind::InvalidUsage,
    );

    // Ownership, which the O(1) identity step also covers — asserted here as well
    // because a validator that dropped it would still pass the verb's own test.
    assert_kind(
        validate_buffer_readback(&src, BufferRange::new(0, 16), identity(3)),
        RhiErrorKind::WrongDevice,
    );

    // The third item on section 18.1's list — the route's alignment — is *not*
    // here, because it is not portable and this function no longer takes a device
    // layout. `a_readback_whose_range_breaks_the_device_alignment_is_refused` in
    // the command chapter drives it through the verb, which is where it belongs:
    // that test states a device answer and gets an answer back, while a fixture
    // here would only prove the validator can compare two numbers.
}

#[test]
fn a_texture_readback_needs_copy_source_usage_and_a_single_sample() {
    let region = (
        TextureSubresourceLayers {
            aspect: TextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        Origin3d { x: 0, y: 0, z: 0 },
        Extent3d::d2(4, 2),
    );
    let src = texture_with(TextureDescriptor::new_2d(
        4,
        2,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COPY_SRC,
    ));
    assert!(validate_texture_readback(&src, region.0, region.1, region.2, device()).is_ok());

    // Section 18.1 requires a single-sampled source even though the texture
    // cannot be uploaded to at that sample count.
    let multisampled = texture_with(
        TextureDescriptor::new_2d(
            4,
            2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_SRC.union(TextureUsage::COLOR_ATTACHMENT),
        )
        .with_sample_count(4),
    );
    assert_kind(
        validate_texture_readback(&multisampled, region.0, region.1, region.2, device()),
        RhiErrorKind::InvalidUsage,
    );

    let unresolvable = texture_with(simple_texture_descriptor(TextureUsage::SAMPLED));
    assert_kind(
        validate_texture_readback(&unresolvable, region.0, region.1, region.2, device()),
        RhiErrorKind::InvalidUsage,
    );

    assert_kind(
        validate_texture_readback(&src, region.0, region.1, region.2, identity(4)),
        RhiErrorKind::WrongDevice,
    );
}

#[test]
fn a_ticket_walks_its_state_machine_and_reports_data_only_when_ready() {
    // Section 18.2's machine, driven from the device side the same way the
    // backend port will drive it.
    let ticket = ReadbackTicket::new(
        object(71),
        device(),
        ReadbackRequest::Buffer {
            label: Label::default(),
            src: buffer_with(BufferUsage::COPY_SRC, 64),
            range: BufferRange::new(0, 4),
        },
    );

    assert_eq!(ticket.id(), object(71));
    assert_eq!(ticket.device_identity(), device());
    assert!(matches!(ticket.request(), ReadbackRequest::Buffer { .. }));
    assert_eq!(ticket.status(), ReadbackStatus::NotSubmitted);
    assert!(ticket.try_read().expect("not an error yet").is_none());

    ticket.set_status(ReadbackStatus::Pending);
    assert!(ticket.try_read().expect("still not an error").is_none());

    ticket.publish(vec![1, 2, 3, 4], None);
    assert_eq!(ticket.status(), ReadbackStatus::Ready);
    match ticket.try_read().expect("ready").expect("data").data() {
        ReadbackViewData::Buffer { bytes } => assert_eq!(bytes, &[1, 2, 3, 4]),
        ReadbackViewData::Texture { .. } => panic!("a buffer request returns buffer bytes"),
    }
}

#[test]
fn a_ticket_reports_the_point_its_work_was_accepted_under() {
    // Section 18.2's binding. The unsubmitted case is the interesting one: a
    // ticket must report `None` rather than mint a point for work the device has
    // not accepted, so the only way to reach `Some` is the submit path's own
    // record — here driven directly, the same way `publish` is.
    let ticket = ReadbackTicket::new(
        object(73),
        device(),
        ReadbackRequest::Buffer {
            label: Label::default(),
            src: buffer_with(BufferUsage::COPY_SRC, 64),
            range: BufferRange::new(0, 4),
        },
    );

    assert!(
        ticket.completion().is_none(),
        "no submission has recorded a point yet, so there is nothing to wait on"
    );

    ticket.set_completion(CompletionPoint::new(device(), 1));
    assert_eq!(ticket.completion(), Some(CompletionPoint::new(device(), 1)));

    // First write wins. Overwriting would move a caller's wait onto work it
    // never encoded, which is the failure the crate-private setter exists to
    // make impossible from outside.
    ticket.set_completion(CompletionPoint::new(device(), 2));
    assert_eq!(ticket.completion(), Some(CompletionPoint::new(device(), 1)));
}

#[test]
fn a_cloned_ticket_observes_the_same_state() {
    // Section 18.4's "shared device-scoped state": a clone that snapshotted its
    // status would disagree with the original the moment the device advanced.
    let ticket = ReadbackTicket::new(
        object(72),
        device(),
        ReadbackRequest::Buffer {
            label: Label::default(),
            src: buffer_with(BufferUsage::COPY_SRC, 64),
            range: BufferRange::new(0, 4),
        },
    );
    let clone = ticket.clone();

    clone.set_status(ReadbackStatus::Pending);
    assert_eq!(ticket.status(), ReadbackStatus::Pending);

    clone.publish(vec![9, 9], None);
    match ticket.try_read().expect("ready").expect("data").data() {
        ReadbackViewData::Buffer { bytes } => assert_eq!(bytes, &[9, 9]),
        ReadbackViewData::Texture { .. } => panic!("a buffer request returns buffer bytes"),
    }
}

#[test]
fn a_texture_ticket_reports_the_layout_alongside_the_bytes() {
    // Section 18.3: readback does not promise tightly packed texels, so the
    // layout travels with the bytes.
    let ticket = ReadbackTicket::new(
        object(73),
        device(),
        ReadbackRequest::Texture {
            label: Label::default(),
            src: texture_with(TextureDescriptor::new_2d(
                4,
                2,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COPY_SRC,
            )),
            subresource: TextureSubresourceLayers {
                aspect: TextureAspect::Color,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent: Extent3d::d2(4, 2),
        },
    );

    let layout = ReadbackTexelLayout {
        bytes_per_row: 256,
        rows_per_image: 2,
        total_size: 512,
    };
    ticket.publish(vec![7u8; 512], Some(layout));

    match ticket.try_read().expect("ready").expect("data").data() {
        ReadbackViewData::Texture {
            bytes,
            layout: reported,
        } => {
            assert_eq!(bytes.len(), 512);
            // The padded row pitch is reported rather than hidden: 256 is far
            // more than the 16 logical bytes of a row, and that is the point.
            assert_eq!(reported, layout);
            assert_eq!(reported.bytes_per_row, 256);
            assert_eq!(reported.total_size, 512);
        }
        ReadbackViewData::Buffer { .. } => panic!("a texture request returns texel layout"),
    }
}

#[test]
fn every_terminal_state_is_an_error_and_not_an_endless_none() {
    let make = || {
        ReadbackTicket::new(
            object(74),
            device(),
            ReadbackRequest::Buffer {
                label: Label::default(),
                src: buffer_with(BufferUsage::COPY_SRC, 64),
                range: BufferRange::new(0, 4),
            },
        )
    };

    let abandoned = make();
    abandoned.set_status(ReadbackStatus::Abandoned);
    match abandoned.try_read() {
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::InvalidUsage),
        Ok(_) => panic!("an abandoned request never produces data"),
    }

    let lost = make();
    lost.set_status(ReadbackStatus::DeviceLost);
    match lost.try_read() {
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::DeviceLost),
        Ok(_) => panic!("a lost device never produces data"),
    }

    let failed = make();
    failed.set_status(ReadbackStatus::Failed);
    match failed.try_read() {
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::BackendFailure),
        Ok(_) => panic!("a failed readback never produces data"),
    }
}

#[test]
fn a_ready_ticket_without_bytes_is_reported_as_a_backend_fault() {
    // `publish` is the only correct way to reach Ready, so this state is
    // unreachable through the crate's own writers — but the reader must not
    // panic or hand out an empty slice if a backend ever reaches it.
    let ticket = ReadbackTicket::new(
        object(75),
        device(),
        ReadbackRequest::Buffer {
            label: Label::default(),
            src: buffer_with(BufferUsage::COPY_SRC, 64),
            range: BufferRange::new(0, 4),
        },
    );
    ticket.set_status(ReadbackStatus::Ready);

    match ticket.try_read() {
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::BackendFailure),
        Ok(_) => panic!("Ready without published bytes is a fault"),
    }
}

#[test]
fn the_status_wire_values_round_trip_through_a_ticket() {
    // The status is stored as its wire value in a shared cell, so the mapping
    // in both directions is part of the contract. Every state is checked, not
    // just the ones the other tests happen to use.
    for status in [
        ReadbackStatus::NotSubmitted,
        ReadbackStatus::Pending,
        ReadbackStatus::Ready,
        ReadbackStatus::Abandoned,
        ReadbackStatus::DeviceLost,
        ReadbackStatus::Failed,
    ] {
        let ticket = ReadbackTicket::new(
            object(76),
            device(),
            ReadbackRequest::Buffer {
                label: Label::default(),
                src: buffer_with(BufferUsage::COPY_SRC, 64),
                range: BufferRange::new(0, 4),
            },
        );
        ticket.set_status(status);
        assert_eq!(ticket.status(), status);
    }
}

#[test]
fn a_retained_upload_payload_survives_the_descriptor_that_named_it() {
    // Section 17.2's retention rule: the job owns the bytes, so the only handle
    // the caller kept can go away without invalidating the job's payload.
    let bytes: std::sync::Arc<[u8]> = vec![0xAB].into();
    let descriptor = TextureUploadDescriptor {
        label: Label::default(),
        dst: texture_with(TextureDescriptor::new_2d(
            1,
            1,
            TextureFormat::R8Unorm,
            TextureUsage::COPY_DST,
        )),
        subresource: TextureSubresourceLayers {
            aspect: TextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        origin: Origin3d { x: 0, y: 0, z: 0 },
        extent: Extent3d::d2(1, 1),
        source_layout: HostTexelLayout {
            bytes_per_row: 1,
            rows_per_image: 1,
        },
        bytes: std::sync::Arc::clone(&bytes),
    };
    let job = UploadJob::new(object(77), device(), UploadDescriptor::Texture(descriptor));
    // The caller's own handle to the payload is gone; the job's is not.
    drop(bytes);

    match job.descriptor() {
        UploadDescriptor::Texture(texture) => {
            assert_eq!(&texture.bytes[..], &[0xAB]);
            assert_eq!(texture.extent, Extent3d::d2(1, 1));
            assert_eq!(std::sync::Arc::strong_count(&texture.bytes), 1);
        }
        UploadDescriptor::Buffer(_) => panic!("the job described a texture upload"),
    }
}

#[test]
fn a_buffer_support_query_carries_the_usage_it_asks_about() {
    let query = BufferSupportQuery::new(BufferUsage::STORAGE);
    assert_eq!(query.usage(), BufferUsage::STORAGE);
    assert_eq!(query, BufferSupportQuery::new(BufferUsage::STORAGE));
    assert_ne!(query, BufferSupportQuery::new(BufferUsage::VERTEX));
}

#[test]
fn a_supported_buffer_answer_reports_its_ceiling() {
    let supported = BufferSupport::Supported(BufferSupportLimits::new(1 << 28));
    assert!(supported.is_supported());
    assert_eq!(supported.limits().unwrap().max_size(), 1 << 28);

    assert!(!BufferSupport::Unsupported.is_supported());
    assert!(BufferSupport::Unsupported.limits().is_none());
}
