//! S0 copy and completion consumer for the rebuilt Fluxel Renderer.
//!
//! This executable intentionally has no renderer-private access. It creates
//! three immutable Asset snapshots for one logical identity, submits all three
//! before the first completion wait, then checks the complete GPU readbacks.

#![cfg_attr(not(windows), allow(dead_code))]

mod capabilities;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod capture;
mod compute;
mod demo;
mod demo_evidence;
mod diagnostics;
mod executable_cache;
mod raster;
mod read_set;
mod s2_shader;
mod s2_texture;
mod s3_material;
mod s5_forward;
mod s5_forward_surface;
mod s6_composed;
mod s7_deferred;
mod s7_device_isolation;
mod s7_surface;
mod s8_material;
mod shader_admission;
mod surface;

use std::{
    error::Error,
    fs,
    future::Future,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::Instant,
};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_renderer::{Renderer, RendererError};
use fluxel_rhi::{
    api::Device,
    backend::{ExecutionBackend, MappingBackend, ReadbackBackend, ResourceBackend},
};

use shader_admission::json_string;

mod main_copy;
mod main_demo;
mod main_native;
mod main_support;

#[cfg(all(windows, feature = "dx12"))]
use main_native::run_dx12;
#[cfg(all(windows, feature = "native-gl-wgl"))]
use main_native::run_wgl;
use main_support::{block_on, output_directory};

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize before native objects and balance only after they are gone;
    // capture and the surrounding renderer share this thread's apartment.
    #[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
    let _capture_runtime = capture::CaptureRuntime::new()?;
    // Pixel-sized RHI surfaces and desktop client captures must share one
    // coordinate space. This configures only this fixture process.
    #[cfg(windows)]
    unsafe {
        use windows::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)?;
    }
    let (output, workload) = output_directory()?;
    fs::create_dir_all(&output)?;
    #[cfg(all(windows, feature = "dx12", feature = "native-gl-wgl"))]
    return Err("select exactly one rendering integration backend feature".into());
    #[cfg(all(windows, feature = "dx12", not(feature = "native-gl-wgl")))]
    return run_dx12(&output, workload);
    #[cfg(all(windows, feature = "native-gl-wgl", not(feature = "dx12")))]
    return run_wgl(&output, workload);
    #[cfg(not(all(
        windows,
        any(feature = "dx12", feature = "native-gl-wgl"),
        not(all(feature = "dx12", feature = "native-gl-wgl"))
    )))]
    {
        let _ = (output, workload);
        Err("select --features dx12 or --features native-gl-wgl on Windows".into())
    }
}
