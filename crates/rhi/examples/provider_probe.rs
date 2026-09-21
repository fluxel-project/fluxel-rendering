//! Minimal public provider composition example.
//!
//! This binary deliberately stops after adapter/device discovery.  Rendering
//! workloads live in the host-injected examples beside it; the important seam
//! demonstrated here is that an application obtains a portable provider from
//! the public platform entry point and never names a backend-native object.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread;
use std::time::Duration;

use fluxel_rhi::api::platform::PlatformProvider;
#[cfg(all(windows, feature = "dx12"))]
use fluxel_rhi::create_dx12_provider;
// Windows can intentionally build Vulkan without DX12.  Prefer DX12 only when
// both are available; do not accidentally turn that legitimate feature set
// into the "no provider" branch.
#[cfg(all(
    feature = "vulkan",
    not(target_arch = "wasm32"),
    not(all(windows, feature = "dx12"))
))]
use fluxel_rhi::create_vulkan_provider;

fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWaker(thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park_timeout(Duration::from_millis(1)),
        }
    }
}

fn probe(provider: PlatformProvider) {
    let adapters = match block_on(provider.enumerate_adapters()) {
        Ok(Some(adapters)) => adapters,
        Ok(None) => {
            println!("provider does not expose explicit adapter enumeration");
            return;
        }
        Err(error) => {
            eprintln!("adapter enumeration failed: {error}");
            return;
        }
    };
    println!("{} adapter(s) for {:?}", adapters.len(), provider.backend());
    for adapter in adapters {
        println!("- {} ({:?})", adapter.name(), adapter.backend());
    }
}

#[allow(unreachable_code)]
fn main() {
    #[cfg(all(windows, feature = "dx12"))]
    let provider = match create_dx12_provider() {
        Ok(provider) => provider,
        Err(error) => {
            eprintln!("DX12 provider unavailable: {error}");
            return;
        }
    };
    #[cfg(all(windows, feature = "dx12"))]
    probe(provider);
    return;
    #[cfg(all(
        feature = "vulkan",
        not(target_arch = "wasm32"),
        not(all(windows, feature = "dx12"))
    ))]
    let provider = match create_vulkan_provider() {
        Ok(provider) => provider,
        Err(error) => {
            eprintln!("Vulkan provider unavailable: {error}");
            return;
        }
    };
    #[cfg(all(
        feature = "vulkan",
        not(target_arch = "wasm32"),
        not(all(windows, feature = "dx12"))
    ))]
    probe(provider);
    return;
    #[cfg(not(any(
        all(windows, feature = "dx12"),
        all(
            feature = "vulkan",
            not(target_arch = "wasm32"),
            not(all(windows, feature = "dx12"))
        )
    )))]
    {
        eprintln!("no native provider feature is enabled for this target");
        return;
    }
}
