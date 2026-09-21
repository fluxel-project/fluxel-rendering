//! Minimal public provider composition example.
//!
//! This binary deliberately stops after adapter/device discovery.  Rendering
//! workloads live in the host-injected examples beside it; the important seam
//! demonstrated here is that an application obtains a portable provider from
//! the public platform entry point and never names a backend-native object.

#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;
#[cfg(not(target_arch = "wasm32"))]
use std::pin::pin;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::task::{Context, Poll, Wake, Waker};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use fluxel_rhi::api::platform::PlatformProvider;
#[cfg(all(windows, feature = "dx12"))]
use fluxel_rhi::create_dx12_provider;
// Windows can intentionally build Vulkan without DX12.  Prefer DX12 only when
// both are available; do not accidentally turn that legitimate feature set
// into the "no provider" branch.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use fluxel_rhi::create_metal_provider;
#[cfg(all(
    feature = "vulkan",
    not(target_arch = "wasm32"),
    not(all(windows, feature = "dx12"))
))]
use fluxel_rhi::create_vulkan_provider;
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use fluxel_rhi::create_webgpu_provider;

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    let provider = match create_metal_provider() {
        Ok(provider) => provider,
        Err(error) => {
            eprintln!("Metal provider unavailable: {error}");
            return;
        }
    };
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    probe(provider);
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    return;
    #[cfg(not(any(
        all(windows, feature = "dx12"),
        all(
            feature = "vulkan",
            not(target_arch = "wasm32"),
            not(all(windows, feature = "dx12"))
        ),
        all(feature = "metal", target_vendor = "apple")
    )))]
    {
        eprintln!("no native provider feature is enabled for this target");
        return;
    }
}

// Browser adapter enumeration necessarily yields to the JavaScript event loop;
// `thread::park` is neither available nor a valid Promise executor there. The
// same public composition seam is used, with browser scheduling kept in this
// binary rather than leaked into the RHI API.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn main() {
    wasm_bindgen_futures::spawn_local(async {
        let provider = match create_webgpu_provider() {
            Ok(provider) => provider,
            Err(error) => {
                web_sys::console::error_1(&format!("WebGPU provider unavailable: {error}").into());
                return;
            }
        };
        match provider.enumerate_adapters().await {
            Ok(Some(adapters)) => {
                web_sys::console::log_1(&format!("{} WebGPU adapter(s)", adapters.len()).into());
                for adapter in adapters {
                    web_sys::console::log_1(&format!("- {}", adapter.name()).into());
                }
            }
            Ok(None) => web_sys::console::log_1(&"WebGPU adapter enumeration unavailable".into()),
            Err(error) => {
                web_sys::console::error_1(&format!("adapter enumeration failed: {error}").into())
            }
        }
    });
}

#[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
fn main() {
    eprintln!("provider_probe requires the `webgpu` feature on wasm32");
}
