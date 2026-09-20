//! Positive, negative, and boundary conformance tests for asynchronous mapping.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use crate::api::error::RhiErrorKind;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId};
use crate::api::platform::DeviceLossInfo;
use crate::api::resource::{BufferDescriptor, BufferRange, BufferUsage, MapMode};
use crate::api::tests::mock::mapped_buffers_for_test;

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
fn a_device_without_mapping_capability_refuses_before_native_mapping() {
    let (device, _) = crate::api::tests::mock::buffers_for_test(identity(97), 64);
    let buffer = device
        .create_buffer(&BufferDescriptor::new(8, BufferUsage::MAP_READ))
        .expect("buffer creation is distinct from map capability");
    assert_eq!(
        device
            .map_buffer(&buffer, MapMode::Read, BufferRange::new(0, 4))
            .unwrap_err()
            .kind(),
        RhiErrorKind::Unsupported
    );
}
