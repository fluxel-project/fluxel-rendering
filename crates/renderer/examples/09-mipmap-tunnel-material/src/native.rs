//! Native Surface adapter for the tunnel's visible acceptance mode.

use super::*;
use fluxel_host::{Window, WindowConfig, WindowVisibility};
use fluxel_rhi::{
    api::{
        ColorSpace, CompositeAlphaMode, DeviceRequest, PresentMode, Surface, SurfaceConfig,
        SurfaceUsage, TextureFormat,
    },
    backend::SurfaceBackend,
};
use renderer_09_mipmap_tunnel_material::oracle;

#[cfg(feature = "dx12")]
pub(super) fn run_dx12(options: Options) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::dx12::Dx12Instance;
    let capture = setup_visible_capture(&options)?;
    let native = Dx12Instance::new()?;
    let instance = native.instance()?;
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no DX12 adapter")?;
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    }))?;
    diagnostics::record_dx12_device_open(&options.output)?;
    match options.mode {
        Mode::Offscreen => run_offscreen(device, &options),
        Mode::Visible => {
            use raw_window_handle::HasWindowHandle;
            let window = window()?;
            let raw = window
                .window_handle()
                .map_err(|error| format!("visible DX12 window handle: {error:?}"))?
                .as_raw();
            let surface =
                unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
            visible(
                device,
                surface,
                &window,
                capture.as_ref().ok_or("visible mode requires capture")?,
                &options,
            )
        }
    }
}

#[cfg(feature = "native-gl-wgl")]
pub(super) fn run_wgl(options: Options) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::{
        api::{AdapterId, Instance, InstanceId},
        gl::{Gl, GlInstance},
    };
    let capture = setup_visible_capture(&options)?;
    let window = window()?;
    let native = unsafe { GlInstance::from_wgl_window(&window, scene::EXTENT) }?;
    let id = InstanceId::allocate()?;
    let instance = unsafe { Instance::<Gl>::from_native(id, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no WGL adapter")?;
    let device = block_on(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: id,
            ordinal: adapter.id.ordinal,
        },
        required_features: Vec::new(),
    }))?;
    let observed = device.clone();
    let mut observer = diagnostics::WglObserver::install(&observed)?;
    let result = match options.mode {
        Mode::Offscreen => run_offscreen(device, &options),
        Mode::Visible => visible(
            device.clone(),
            Gl::surface(&device)?,
            &window,
            capture.as_ref().ok_or("visible mode requires capture")?,
            &options,
        ),
    };
    result.and(observer.write(&observed, &options.output))
}

fn setup_visible_capture(
    options: &Options,
) -> Result<Option<integration_capture::CaptureRuntime>, Box<dyn Error>> {
    if matches!(options.mode, Mode::Offscreen) {
        return Ok(None);
    }
    unsafe {
        use windows::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)?;
    }
    fs::write(
        options.output.join("visible-dpi.json"),
        "{\"schemaVersion\":1,\"processDpiAwareness\":\"per-monitor-aware-v2\",\"scope\":\"before HWND, WinRT, and native GPU objects\"}\n",
    )?;
    Ok(Some(integration_capture::CaptureRuntime::new()?))
}

fn window() -> Result<Window, Box<dyn Error>> {
    Ok(Window::new(
        WindowConfig::new(
            "Fluxel mipmap tunnel Material",
            scene::EXTENT[0],
            scene::EXTENT[1],
        )?
        .with_visibility(WindowVisibility::Hidden),
    )?)
}

fn configure<B: SurfaceBackend>(surface: &mut Surface<B>) -> Result<(), Box<dyn Error>> {
    let caps = surface.capabilities()?;
    let config = [TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm]
        .into_iter()
        .filter(|format| caps.formats.contains(format))
        .map(|format| SurfaceConfig {
            extent: scene::EXTENT,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .find(|config| {
            matches!(
                surface.configuration_support(config),
                fluxel_rhi::api::Support::Supported(())
            )
        })
        .ok_or("Surface has no supported opaque FIFO RGBA8 configuration")?;
    block_on(surface.configure(&config))?;
    Ok(())
}

fn visible<B>(
    device: Device<B>,
    mut surface: Surface<B>,
    window: &Window,
    _capture: &integration_capture::CaptureRuntime,
    options: &Options,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend
        + SurfaceBackend,
{
    evidence::write_device(&options.output, &device)?;
    configure(&mut surface)?;
    window_oracle::write_contract(&options.output, scene::EXTENT)?;
    let input = scene::input(&options.model, &options.asset, options.anisotropy)?;
    evidence::write_inputs(&options.output, &input)?;
    let probes = oracle::reference(&input)?;
    oracle::write(&options.output, &probes)?;
    let mut renderer = Renderer::new(device)?;
    let bounds = RenderBounds::new([-64.; 3], [64.; 3])?;
    for (frame, (route, name)) in [(Route::Forward, "forward"), (Route::Deferred, "deferred")]
        .into_iter()
        .enumerate()
    {
        let shown = match route {
            Route::Forward => block_on(renderer.submit_forward_surface(
                &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
                input.snapshot.clone(),
                input.view,
                bounds,
                &input.assets,
                block_on(surface.acquire())?,
            ))?,
            Route::Deferred => block_on(renderer.submit_deferred_surface(
                &DeferredCpuPipeline::new(BoundsVisibility),
                input.snapshot.clone(),
                input.view,
                bounds,
                &input.assets,
                block_on(surface.acquire())?,
            ))?,
        };
        block_on(renderer.complete(&shown))?;
        window.show_background()?;
        let companion = match route {
            Route::Forward => block_on(renderer.submit_forward(
                &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
                input.snapshot.clone(),
                input.view,
                bounds,
                &input.assets,
            ))?,
            Route::Deferred => block_on(renderer.submit_deferred(
                &DeferredCpuPipeline::new(BoundsVisibility),
                input.snapshot.clone(),
                input.view,
                bounds,
                &input.assets,
            ))?,
        };
        let raw = block_on(renderer.readback(&companion))?;
        fs::write(
            options
                .output
                .join(format!("visible-{name}-companion-readback.rgba")),
            &raw,
        )?;
        if options.anisotropy == 1 {
            oracle::verify(&raw, &probes, matches!(route, Route::Deferred))?;
        }
        integration_capture::snapshot_client(window, &options.output, frame, scene::EXTENT)?;
        let bmp = fs::read(
            options
                .output
                .join(format!("surface-client-frame-{frame}.bmp")),
        )?;
        window_oracle::compare_bmp_rgb(&bmp, &raw, scene::EXTENT)?;
    }
    fs::write(
        options.output.join("visible-result.json"),
        if options.anisotropy == 1 {
            "{\"schemaVersion\":1,\"surface\":\"Forward and Deferred Renderer Surface leases\",\"comparison\":\"dense exact RGB except fixed 512-pixel DWM mask\",\"result\":\"passed with independent aniso1 oracle\"}\n"
        } else {
            "{\"schemaVersion\":1,\"surface\":\"Forward and Deferred Renderer Surface leases\",\"comparison\":\"dense exact RGB except fixed 512-pixel DWM mask\",\"result\":\"unverified aniso16 capability control\"}\n"
        },
    )?;
    if !options.exit_after_capture {
        wait_for_close(window)?;
    }
    Ok(())
}

fn wait_for_close(window: &Window) -> Result<(), Box<dyn Error>> {
    loop {
        if window
            .poll_events()?
            .into_iter()
            .any(|event| matches!(event, fluxel_host::WindowEvent::CloseRequested))
        {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}
