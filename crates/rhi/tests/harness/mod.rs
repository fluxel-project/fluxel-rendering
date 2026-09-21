//! Shared execution policy for RHI hardware conformance.
//!
//! A test case records only portable RHI work. A backend fixture supplies a
//! private provider, shader artifact, compatible lane, and (for presentation)
//! a host-owned native target. This module supplies the portable terminal
//! semantics shared by every fixture. Browser and mobile runners may drive the
//! returned futures from their own event loop; no `Waker::noop()` or native
//! handle appears here.

mod outcome;

pub(crate) use outcome::CaseOutcome;

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;
use std::task::Wake;
use std::thread;
use std::time::Duration;

use crate::api::platform::Device;
use crate::api::submission::{CompletionPoint, CompletionState};

/// Small native-test executor used by fixtures which are already running on a
/// platform thread.  Unlike the old `Waker::noop()` polling helpers this waker
/// is connected to the current thread, so an asynchronous WebGPU/mobile-style
/// completion can wake the future instead of being mistaken for a synchronous
/// operation.  Browser runners do not use this function; they drive the same
/// futures from their event loop.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWaker(thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let current = thread::current();
    let waker = Waker::from(Arc::new(ThreadWaker(current)));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park_timeout(Duration::from_millis(1)),
        }
    }
}

/// Waits by the public completion Future and accepts only a successful terminal
/// result. Timeout and event-loop pumping belong to the caller's platform test
/// runner, rather than being faked as a portable busy loop here.
pub(crate) async fn require_complete(device: &Device, point: CompletionPoint, case: &str) {
    match device.wait_completion(point).await {
        Ok(CompletionState::Complete) => {}
        Ok(CompletionState::Pending) => {
            panic!("{case}: wait_completion returned Pending although its Future resolved")
        }
        Ok(terminal) => panic!("{case}: accepted GPU work reached {terminal:?}"),
        Err(error) => panic!("{case}: completion wait failed: {error}"),
    }
}
