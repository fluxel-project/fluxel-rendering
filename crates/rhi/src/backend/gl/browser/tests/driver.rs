//! Live browser coverage for the v13 WebGL owner-thread driver.
//!
//! These are deliberately browser tests rather than pure mocks: registration
//! and dispatch must prove that the registry reaches an actual WebGL2 context.
//! Loss itself is notified by the host bridge, so this test uses the same
//! notification entry point rather than pretending that a synthetic driver
//! exception exercises browser lifecycle delivery.

use wasm_bindgen_test::*;

use crate::api::error::RhiErrorKind;
use crate::backend::gl::browser::WebGl2ExecutionDriver;
use crate::backend::gl::platform::GlExecutionDriver;

#[wasm_bindgen_test]
fn registered_driver_dispatches_to_the_live_webgl_owner() {
    let driver = WebGl2ExecutionDriver::register(super::provider()).expect("register WebGL2");
    driver
        .dispatch("webgl2-live-driver-dispatch")
        .expect("dispatch reaches the live WebGL2 context");
    driver.unregister().expect("unregister WebGL2");
}

#[wasm_bindgen_test]
fn loss_notification_makes_later_dispatch_terminal_not_successful() {
    let driver = WebGl2ExecutionDriver::register(super::provider()).expect("register WebGL2");
    driver
        .notify_context_lost()
        .expect("host loss notification clears WebGL2 state");
    let error = driver
        .dispatch("webgl2-dispatch-after-loss")
        .expect_err("a lost WebGL2 context cannot accept work");
    assert_eq!(error.kind(), RhiErrorKind::DeviceLost);
    driver.unregister().expect("unregister lost WebGL2");
}
