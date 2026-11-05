//! Native bootstrap shared by the persistent visible demo selectors.

use std::{error::Error, path::Path};

use super::{demo, main_support::Workload};

pub(super) fn route(workload: Workload) -> Option<demo::DemoRoute> {
    match workload {
        Workload::DemoForward => Some(demo::DemoRoute::Forward),
        Workload::DemoDeferred => Some(demo::DemoRoute::Deferred),
        _ => None,
    }
}

pub(super) fn output_name(route: demo::DemoRoute) -> &'static str {
    match route {
        demo::DemoRoute::Forward => "demo-forward",
        demo::DemoRoute::Deferred => "demo-deferred",
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(super) fn wgl_window_properties(workload: Workload) -> (&'static str, [u32; 2]) {
    if route(workload).is_some() {
        ("Fluxel Renderer Demo", [640, 480])
    } else {
        ("Fluxel S1 Surface", [320, 180])
    }
}

#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
fn event(event: fluxel_host::WindowEvent) -> Option<demo::DemoEvent> {
    match event {
        fluxel_host::WindowEvent::CloseRequested => Some(demo::DemoEvent::CloseRequested),
        fluxel_host::WindowEvent::Resized { width, height } => {
            Some(demo::DemoEvent::Resized([width, height]))
        }
        fluxel_host::WindowEvent::Minimized => Some(demo::DemoEvent::Minimized),
        fluxel_host::WindowEvent::Restored { width, height } => {
            Some(demo::DemoEvent::Restored([width, height]))
        }
        fluxel_host::WindowEvent::SurfaceDestroyed => Some(demo::DemoEvent::SurfaceDestroyed),
        fluxel_host::WindowEvent::SurfaceCreated => Some(demo::DemoEvent::SurfaceCreated),
        fluxel_host::WindowEvent::Suspended => Some(demo::DemoEvent::Suspended),
        fluxel_host::WindowEvent::Resumed => Some(demo::DemoEvent::Resumed),
        fluxel_host::WindowEvent::RedrawRequested => Some(demo::DemoEvent::RedrawRequested),
        _ => None,
    }
}

#[cfg(all(windows, feature = "dx12"))]
pub(super) fn run_dx12(
    device: fluxel_rhi::api::Device<fluxel_rhi::dx12::Dx12>,
    route: demo::DemoRoute,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    use fluxel_host::{Window, WindowConfig, WindowVisibility};
    use raw_window_handle::HasWindowHandle;

    let window = Window::new(
        WindowConfig::new("Fluxel Renderer Demo", 640, 480)?
            .with_visibility(WindowVisibility::Hidden),
    )?;
    let raw = window
        .window_handle()
        .map_err(|error| format!("demo DX12 window handle: {error:?}"))?
        .as_raw();
    // SAFETY: the Host window survives its presentation surface and all frames.
    let surface = unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
    let evidence = demo::evidence_enabled();
    let mut shown = false;
    demo::run(
        device,
        surface,
        route,
        [640, 480],
        output,
        || {
            Ok::<_, Box<dyn Error>>(
                window
                    .poll_events()?
                    .into_iter()
                    .filter_map(event)
                    .collect(),
            )
        },
        |frame, extent| {
            if !shown {
                window.show_background()?;
                shown = true;
            }
            if evidence && matches!(frame, 0 | 5 | 10) {
                crate::capture::snapshot_client(&window, output, frame, extent)?;
            }
            Ok(())
        },
    )
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(super) fn run_wgl(
    device: fluxel_rhi::api::Device<fluxel_rhi::gl::Gl>,
    window: &fluxel_host::Window,
    route: demo::DemoRoute,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let surface = fluxel_rhi::gl::Gl::surface(&device)?;
    // Some WGL drivers discard work submitted while this HWND is hidden. The
    // Host reveal is non-activating, so map before the first surface frame;
    // this callback remains idempotent for the demo lifecycle contract.
    window.show_background()?;
    let mut shown = true;
    let evidence = demo::evidence_enabled();
    demo::run(
        device,
        surface,
        route,
        [640, 480],
        output,
        || {
            Ok::<_, Box<dyn Error>>(
                window
                    .poll_events()?
                    .into_iter()
                    .filter_map(event)
                    .collect(),
            )
        },
        |frame, extent| {
            if !shown {
                window.show_background()?;
                shown = true;
            }
            if evidence && matches!(frame, 0 | 5 | 10) {
                crate::capture::snapshot_client(window, output, frame, extent)?;
            }
            Ok(())
        },
    )
}
