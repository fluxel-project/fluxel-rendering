//! Native DX12/WGL openings. Both routes run Renderer; no direct RHI draw exists.
use super::*;
use fluxel_host::{Window, WindowConfig, WindowVisibility};
use fluxel_rhi::{
    api::{
        ColorSpace, CompositeAlphaMode, DeviceRequest, PresentMode, Surface, SurfaceConfig,
        SurfaceUsage, TextureFormat,
    },
    backend::SurfaceBackend,
};

#[cfg(feature = "dx12")]
pub(super) fn run_dx12(options: Options) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::dx12::Dx12Instance;
    if matches!(options.mode, Mode::Visible) {
        enable_dpi(&options)?;
    }
    let capture = matches!(options.mode, Mode::Visible)
        .then(crate::integration_capture::CaptureRuntime::new)
        .transpose()?;
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
    crate::diagnostics::record_dx12_device_open(&options.output)?;
    match options.mode {
        Mode::Offscreen => run_offscreen(device, &options),
        Mode::Visible => {
            use raw_window_handle::HasWindowHandle;
            let host = hidden_window()?;
            let surface = unsafe {
                fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(
                    &device,
                    host.window_handle()
                        .map_err(|error| format!("Host window handle: {error:?}"))?
                        .as_raw(),
                )
            }?;
            visible(device, surface, &host, &options, capture.as_ref())
        }
    }
}

#[cfg(feature = "native-gl-wgl")]
pub(super) fn run_wgl(options: Options) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::{
        api::{AdapterId, Instance, InstanceId},
        gl::{Gl, GlInstance},
    };
    if matches!(options.mode, Mode::Visible) {
        enable_dpi(&options)?;
    }
    let capture = matches!(options.mode, Mode::Visible)
        .then(crate::integration_capture::CaptureRuntime::new)
        .transpose()?;
    let host = hidden_window()?;
    let native = unsafe { GlInstance::from_wgl_window(&host, EXTENT) }?;
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
    let mut observer = crate::diagnostics::WglObserver::install(&observed)?;
    let result = match options.mode {
        Mode::Offscreen => run_offscreen(device, &options),
        Mode::Visible => (|| {
            host.show_background()?;
            let surface = Gl::surface(&device)?;
            visible(device, surface, &host, &options, capture.as_ref())
        })(),
    };
    result.and(observer.write(&observed, &options.output))
}

fn hidden_window() -> Result<Window, Box<dyn Error>> {
    Ok(Window::new(
        WindowConfig::new("Fluxel Renderer 04 dynamic uniform", EXTENT[0], EXTENT[1])?
            .with_visibility(WindowVisibility::Hidden),
    )?)
}

fn enable_dpi(options: &Options) -> Result<(), Box<dyn Error>> {
    unsafe {
        use windows::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)?;
    }
    fs::write(
        options.output.join("visible-dpi.json"),
        "{\"schemaVersion\":1,\"processDpiAwareness\":\"per-monitor-aware-v2\",\"scope\":\"before HWND/native GPU\"}\n",
    )?;
    Ok(())
}

fn configure<B: SurfaceBackend>(surface: &mut Surface<B>) -> Result<(), Box<dyn Error>> {
    let caps = surface.capabilities()?;
    let config = [TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm]
        .into_iter()
        .filter(|format| caps.formats.contains(format))
        .map(|format| SurfaceConfig {
            extent: EXTENT,
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
    device: fluxel_rhi::api::Device<B>,
    mut surface: Surface<B>,
    host: &Window,
    options: &Options,
    capture: Option<&crate::integration_capture::CaptureRuntime>,
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
    let _capture = capture.ok_or("visible mode requires initialized Windows Graphics Capture")?;
    configure(&mut surface)?;
    let input = inputs::grid(options.time, options.reverse)?;
    write_inputs(&options.output, &input, options)?;
    let expected = oracle::grid_reference(EXTENT, input.camera, &input.transforms);
    let probes = runtime::stable_probes(&expected)?;
    fs::write(options.output.join("expected.rgba"), &expected.rgba)?;
    let mask: Vec<u8> = (0..EXTENT[0] * EXTENT[1])
        .map(|pixel| {
            let (x, y) = (pixel % EXTENT[0], pixel / EXTENT[0]);
            u8::from(y >= EXTENT[1] - 16 && (x < 16 || x >= EXTENT[0] - 16))
        })
        .collect();
    fs::write(options.output.join("visible-dense-mask.bin"), mask)?;
    fs::write(
        options.output.join("visible-oracle-contract.txt"),
        format!(
            "independent grid RGB tolerance=2 at predeclared probes={probes:?}; window vs companion exact RGB except fixed bottom 16x16 DWM corners; opaque alpha ignored\n"
        ),
    )?;
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal();
    let mut renderer = Renderer::new(device)?;
    let shown = block_on(renderer.submit_forward_surface(
        &pipeline,
        input.snapshot.clone(),
        input.view,
        input.bounds,
        &input.assets,
        block_on(surface.acquire())?,
    ))?;
    block_on(renderer.complete(&shown))?;
    host.show_background()?;
    let companion = block_on(renderer.submit_forward(
        &pipeline,
        input.snapshot,
        input.view,
        input.bounds,
        &input.assets,
    ))?;
    let raw = block_on(renderer.readback(&companion))?;
    fs::write(options.output.join("visible-companion-readback.rgba"), &raw)?;
    runtime::verify(&raw, &expected.rgba, &probes)?;
    crate::integration_capture::snapshot_client(host, &options.output, 0, EXTENT)?;
    let bmp = fs::read(options.output.join("surface-client-frame-0.bmp"))?;
    dense_rgb(&bmp, &raw)?;
    fs::write(
        options.output.join("visible-result.json"),
        "{\"schemaVersion\":1,\"surface\":\"depth-tested Forward acquired lease\",\"companion\":\"same-input offscreen Renderer submission\",\"comparison\":\"dense exact RGB except fixed DWM corners\"}\n",
    )?;
    if !options.exit_after_capture {
        loop {
            if host
                .poll_events()?
                .into_iter()
                .any(|event| matches!(event, fluxel_host::WindowEvent::CloseRequested))
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
    }
    Ok(())
}

fn dense_rgb(bmp: &[u8], raw: &[u8]) -> Result<(), Box<dyn Error>> {
    if bmp.len() != 54 + raw.len()
        || raw.len() != (EXTENT[0] * EXTENT[1] * 4) as usize
        || &bmp[..2] != b"BM"
    {
        return Err("WGC BMP shape differs from companion".into());
    }
    for pixel in 0..(EXTENT[0] * EXTENT[1]) as usize {
        let at = pixel * 4;
        let x = pixel as u32 % EXTENT[0];
        let y = pixel as u32 / EXTENT[0];
        let masked = y >= EXTENT[1] - 16 && (x < 16 || x >= EXTENT[0] - 16);
        if !masked
            && [bmp[54 + at + 2], bmp[54 + at + 1], bmp[54 + at]]
                != [raw[at], raw[at + 1], raw[at + 2]]
        {
            return Err(format!("window RGB differs from companion at {x},{y}").into());
        }
    }
    Ok(())
}
