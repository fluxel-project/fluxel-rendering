//! WGL fixture for portable GL-family core conformance.
//!
//! This module owns the only native setup: a short-lived Host HWND and the
//! backend-private WGL owner worker.  Once the provider exists, all assertions
//! use the public `PlatformProvider`/`Device` and shared workloads.  No WGL
//! context, HDC, HGLRC, or `glow` dispatch table crosses into `tests/common`.
//!
//! It is deliberately a hardware fixture rather than a mock test.  A host
//! without a usable desktop GL 4.x driver reports a skipped fixture by returning
//! early; a route advertised by a successfully opened provider must pass the
//! shared logical-creation and empty-plan cases.

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};

use fluxel_host::{Window, WindowConfig};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::api::identity::DeviceInstanceId;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, PlatformProvider};
use crate::api::submission::{CompletionState, SubmissionPlanBuilder};
use crate::backend::gl::api::{ContextEpoch, ContextStamp, DeviceIdentity as GlDeviceIdentity};
use crate::backend::gl::native::wgl::{WglWorkerDescriptor, spawn_provider};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            // WGL device adoption is immediately ready today.  Retaining the
            // ordinary executor shape means this fixture remains valid if the
            // native worker later needs an asynchronous readiness handshake.
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn provider(window: &Window) -> Result<PlatformProvider, String> {
    let instance = DeviceInstanceId::new(0x474C_0001);
    let stamp = ContextStamp::new(
        GlDeviceIdentity::new(0x474C_0001).expect("non-zero GL fixture identity"),
        ContextEpoch::INITIAL,
    );
    let descriptor = WglWorkerDescriptor::new(
        stamp,
        window
            .window_handle()
            .map_err(|error| format!("Host WindowHandle unavailable: {error}"))?,
        window
            .display_handle()
            .map_err(|error| format!("Host DisplayHandle unavailable: {error}"))?,
        [32, 32],
    )
    .map_err(|error| format!("WGL descriptor rejected Host handles: {error:?}"))?;
    let native = spawn_provider(instance, descriptor)
        .map_err(|error| format!("WGL provider setup unavailable: {error}"))?;
    Ok(PlatformProvider::new(
        BackendKind::OpenGl,
        instance,
        Box::new(native),
    ))
}

#[test]
fn wgl_fixture_runs_public_core_creation_and_empty_submission_cases() {
    let mut window =
        match Window::new(WindowConfig::new("Fluxel RHI GL conformance", 32, 32).unwrap()) {
            Ok(window) => window,
            Err(error) => {
                eprintln!("SKIPPED WGL core fixture: Host window unavailable: {error}");
                return;
            }
        };
    let provider = match provider(&window) {
        Ok(provider) => provider,
        Err(reason) => {
            eprintln!("SKIPPED WGL core fixture: {reason}");
            return;
        }
    };
    let device = block_on(provider.request_device(DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new(),
    )))
    .unwrap_or_else(|error| {
        panic!("WGL provider opened but public device request failed: {error}")
    });

    crate::backend::conformance::cases::core_device::logical_creation(
        &device,
        "WGL public core logical creation",
    )
    .require_pass("WGL public core logical creation");

    let plan = SubmissionPlanBuilder::new(&device)
        .build()
        .unwrap_or_else(|error| panic!("WGL public empty-plan construction failed: {error}"));
    let receipt = block_on(device.submit(plan))
        .unwrap_or_else(|error| panic!("WGL public empty-plan submission failed: {error}"));
    assert!(matches!(
        device.completion_state(receipt.completion()).unwrap(),
        CompletionState::Complete
    ));

    // The provider and device must release their WGL worker before the Host
    // destroys its borrowed HWND.
    drop(device);
    drop(provider);
    window.close().expect("close WGL fixture Host window");
}
