//! Positive, negative, and boundary conformance tests for asynchronous mapping.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use crate::api::error::RhiErrorKind;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId};
use crate::api::platform::{DeviceLossInfo, OptionalFeature};
use crate::api::resource::{BufferDescriptor, BufferRange, BufferUsage, MapMode};
use crate::api::tests::mock::{
    mapped_buffers_for_test, mapped_buffers_with_alignment_for_test,
    staging_mapped_buffers_for_test,
};

fn identity(value: u64) -> DeviceIdentity {
    DeviceIdentity::new(DeviceInstanceId::new(value))
}

fn mapped(usage: BufferUsage) -> (crate::api::platform::Device, crate::api::resource::Buffer) {
    let (device, _) = mapped_buffers_for_test(identity(91), true);
    let buffer = device
        .create_buffer(&BufferDescriptor::new(32, usage))
        .expect("mapping fixture buffer");
    (device, buffer)
}

fn ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("fixture promised an immediately ready map"),
    }
}

struct WakeCounter(AtomicUsize);

impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn immediate_ready_write_map_commits_on_drop_and_releases_exclusivity() {
    let (device, buffer) = mapped(BufferUsage::MAP_WRITE);
    {
        let mut mapping = ready(
            device
                .map_buffer(&buffer, MapMode::Write, BufferRange::new(0, 4))
                .expect("start write map"),
        )
        .expect("ready write map");
        mapping
            .bytes_mut()
            .expect("write bytes")
            .copy_from_slice(&[1, 2, 3, 4]);
        mapping.flush().expect("coherent flush is legal");
        assert_eq!(
            device
                .map_buffer(&buffer, MapMode::Write, BufferRange::new(4, 4))
                .unwrap_err()
                .kind(),
            RhiErrorKind::InvalidUsage
        );
    }
    let mapping = ready(
        device
            .map_buffer(&buffer, MapMode::Write, BufferRange::new(4, 4))
            .expect("start second write map"),
    )
    .expect("drop released lease");
    drop(mapping);
}

#[test]
fn pending_map_registers_a_waker_and_resolves_after_native_progress() {
    let (device, native) = mapped_buffers_for_test(identity(92), true);
    let buffer = device
        .create_buffer(&BufferDescriptor::new(16, BufferUsage::MAP_READ))
        .expect("buffer");
    native.hold_mapping();
    let future = device
        .map_buffer(&buffer, MapMode::Read, BufferRange::new(0, 4))
        .expect("start map");
    let counter = Arc::new(WakeCounter(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&counter));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(counter.0.load(Ordering::Relaxed), 0);
    native.release_mapping();
    assert_eq!(counter.0.load(Ordering::Relaxed), 1);
    let mapping = match future.as_mut().poll(&mut context) {
        Poll::Ready(Ok(mapping)) => mapping,
        _ => panic!("released mapping must become ready"),
    };
    drop(mapping);
}

#[test]
fn pending_map_wakes_and_returns_device_lost_without_publishing_bytes() {
    let (device, native) = mapped_buffers_for_test(identity(93), true);
    let buffer = device
        .create_buffer(&BufferDescriptor::new(16, BufferUsage::MAP_READ))
        .expect("buffer");
    native.hold_mapping();
    let future = device
        .map_buffer(&buffer, MapMode::Read, BufferRange::new(0, 4))
        .expect("start map");
    let counter = Arc::new(WakeCounter(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&counter));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    native.mark_lost(DeviceLossInfo::new("mapping loss".to_string()));
    assert_eq!(counter.0.load(Ordering::Relaxed), 1);
    assert_eq!(
        match future.as_mut().poll(&mut context) {
            Poll::Ready(Err(error)) => error.kind(),
            _ => panic!("loss must terminate a pending map"),
        },
        RhiErrorKind::DeviceLost
    );
}

#[test]
fn cancelling_a_pending_map_releases_its_exclusive_lease() {
    let (device, native) = mapped_buffers_for_test(identity(94), true);
    let buffer = device
        .create_buffer(&BufferDescriptor::new(16, BufferUsage::MAP_WRITE))
        .expect("buffer");
    native.hold_mapping();
    let future = device
        .map_buffer(&buffer, MapMode::Write, BufferRange::new(0, 4))
        .expect("start map");
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    // `pin!` yields a pinned borrow, so dropping that borrow would leave its
    // stack-owned future alive until the enclosing scope ends. A boxed pin is
    // the owned cancellation handle exercised by this test.
    let mut future = Box::pin(future);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    drop(future);
    native.release_mapping();
    let mapping = ready(
        device
            .map_buffer(&buffer, MapMode::Write, BufferRange::new(4, 4))
            .expect("cancellation released the lease"),
    )
    .expect("replacement map");
    drop(mapping);
}

#[test]
fn mapping_rejects_usage_empty_overflow_misalignment_and_accepts_exact_end() {
    let (device, buffer) = mapped(BufferUsage::COPY_DST);
    assert_eq!(
        device
            .map_buffer(&buffer, MapMode::Read, BufferRange::new(0, 4))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    let (device, buffer) = mapped(BufferUsage::MAP_READ);
    for range in [
        BufferRange::new(0, 0),
        BufferRange::new(30, 4),
        BufferRange::new(u64::MAX, 4),
        BufferRange::new(2, 4),
    ] {
        assert_eq!(
            device
                .map_buffer(&buffer, MapMode::Read, range)
                .unwrap_err()
                .kind(),
            RhiErrorKind::InvalidUsage
        );
    }
    let mapping = ready(
        device
            .map_buffer(&buffer, MapMode::Read, BufferRange::new(28, 4))
            .expect("exact end is in bounds"),
    )
    .expect("exact end mapping");
    drop(mapping);
}

/// A map range is two independently constrained quantities.  In particular,
/// this is the WebGPU shape (8-byte offset, 4-byte size), which one
/// `MapAlignment` value cannot represent without rejecting a legal range.
#[test]
fn mapping_validates_offset_and_size_against_distinct_reported_limits() {
    let (legacy_device, legacy_buffer) = mapped(BufferUsage::MAP_READ);
    // Old facts retain their one-alignment behavior during the migration.
    let legacy = ready(
        legacy_device
            .map_buffer(&legacy_buffer, MapMode::Read, BufferRange::new(4, 4))
            .expect("legacy common alignment accepts matching range"),
    );
    drop(legacy);

    let (device, _) = mapped_buffers_with_alignment_for_test(identity(92), 8, 4);
    let buffer = device
        .create_buffer(&crate::api::resource::BufferDescriptor::new(
            32,
            BufferUsage::MAP_READ,
        ))
        .expect("mapping buffer");

    let legal = ready(
        device
            .map_buffer(&buffer, MapMode::Read, BufferRange::new(8, 4))
            .expect("8-byte offset and 4-byte size are legal"),
    );
    drop(legal);
    for range in [BufferRange::new(4, 4), BufferRange::new(8, 2)] {
        assert_eq!(
            device
                .map_buffer(&buffer, MapMode::Read, range)
                .unwrap_err()
                .kind(),
            RhiErrorKind::InvalidUsage
        );
    }
}

#[test]
fn read_write_flush_invalidate_wrong_device_and_loss_are_structured() {
    let (device, buffer) = mapped(BufferUsage::MAP_READ.union(BufferUsage::MAP_WRITE));
    let mut read = ready(
        device
            .map_buffer(&buffer, MapMode::Read, BufferRange::new(0, 4))
            .expect("start read map"),
    )
    .expect("read map");
    assert_eq!(
        read.bytes_mut().unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
    read.invalidate().expect("coherent invalidate is legal");
    assert_eq!(read.flush().unwrap_err().kind(), RhiErrorKind::InvalidUsage);
    drop(read);

    let (other, _) = mapped_buffers_for_test(identity(95), true);
    assert_eq!(
        other
            .map_buffer(&buffer, MapMode::Read, BufferRange::new(0, 4))
            .unwrap_err()
            .kind(),
        RhiErrorKind::WrongDevice
    );

    let (lost_device, lost_native) = mapped_buffers_for_test(identity(96), true);
    let lost_buffer = lost_device
        .create_buffer(&BufferDescriptor::new(8, BufferUsage::MAP_READ))
        .expect("buffer before loss");
    lost_native.mark_lost(DeviceLossInfo::new("mapping loss".to_string()));
    assert_eq!(
        lost_device
            .map_buffer(&lost_buffer, MapMode::Read, BufferRange::new(0, 4))
            .unwrap_err()
            .kind(),
        RhiErrorKind::DeviceLost
    );
}

#[test]
fn ordinary_staging_mapping_does_not_require_mappable_primary_buffers() {
    let (device, _) = staging_mapped_buffers_for_test(identity(97));
    assert!(
        !device
            .capabilities()
            .supports_feature(OptionalFeature::MappablePrimaryBuffers),
        "the fixture models a staging-only mapping backend"
    );
    let buffer = device
        .create_buffer(&BufferDescriptor::new(
            8,
            BufferUsage::MAP_READ.union(BufferUsage::COPY_DST),
        ))
        .expect("ordinary staging buffer is creatable without the broader feature");
    let mapping = ready(
        device
            .map_buffer(&buffer, MapMode::Read, BufferRange::new(0, 4))
            .expect("ordinary staging map does not require MappablePrimaryBuffers"),
    )
    .expect("ordinary staging mapping succeeds");
    drop(mapping);

    assert_eq!(
        device
            .create_buffer(&BufferDescriptor::new(
                8,
                BufferUsage::MAP_READ.union(BufferUsage::VERTEX),
            ))
            .unwrap_err()
            .kind(),
        RhiErrorKind::Unsupported,
        "the absent feature is represented by the unsupported broader usage combination"
    );
}

#[test]
fn mappable_primary_buffers_authorizes_the_broader_usage_rows_not_map_itself() {
    let (device, _) = mapped_buffers_for_test(identity(98), true);
    assert!(
        device
            .capabilities()
            .supports_feature(OptionalFeature::MappablePrimaryBuffers)
    );
    let buffer = device
        .create_buffer(&BufferDescriptor::new(
            8,
            BufferUsage::MAP_WRITE.union(BufferUsage::VERTEX),
        ))
        .expect("primary map usage is admitted when its exact row and feature are present");
    let mapping = ready(
        device
            .map_buffer(&buffer, MapMode::Write, BufferRange::new(0, 4))
            .expect("map operation still relies on map usage, support, and lease state"),
    )
    .expect("primary mapping resolves");
    drop(mapping);
}
