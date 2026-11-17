//! Perspective Material tunnel using an externally supplied RHI-09 asset pair.

use std::{error::Error, fs, path::PathBuf};

use fluxel_renderer::{
    BoundsVisibility, DeferredCpuPipeline, ForwardCpuPipeline, RenderBounds, Renderer,
};
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use renderer_09_mipmap_tunnel_material::oracle;
use renderer_09_mipmap_tunnel_material::scene;
mod evidence;

#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[path = "../../rendering-integration/src/diagnostics.rs"]
mod diagnostics;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
#[path = "../../rendering-integration/src/capture.rs"]
mod integration_capture;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod native;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[path = "../../pbr-four-texture-material/src/window_oracle.rs"]
mod window_oracle;

#[derive(Clone, Copy)]
enum Route {
    Forward,
    Deferred,
}
#[derive(Clone, Copy)]
enum Mode {
    Offscreen,
    Visible,
}
struct Options {
    mode: Mode,
    model: PathBuf,
    asset: PathBuf,
    anisotropy: u16,
    exit_after_capture: bool,
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse()?;
    if options.output.exists() && fs::read_dir(&options.output)?.next().is_some() {
        return Err("output must be new or empty".into());
    }
    fs::create_dir_all(&options.output)?;
    #[cfg(all(windows, feature = "dx12"))]
    return native::run_dx12(options);
    #[cfg(all(windows, not(feature = "dx12"), feature = "native-gl-wgl"))]
    return native::run_wgl(options);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "native-gl-wgl"))))]
    Err("select a native backend on Windows".into())
}

fn parse() -> Result<Options, Box<dyn Error>> {
    let mut mode = None;
    let mut model = None;
    let mut asset = None;
    let mut anisotropy = 1;
    let mut exit_after_capture = false;
    let mut output = PathBuf::from("target/renderer-09-mipmap-tunnel-material");
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model = Some(args.next().ok_or("--model requires a path")?.into()),
            "--asset" => asset = Some(args.next().ok_or("--asset requires a path")?.into()),
            "--anisotropy16" => anisotropy = 16,
            "--offscreen" | "--visible" => {
                if mode.is_some() {
                    return Err("choose one rendering mode".into());
                }
                mode = Some(if arg == "--offscreen" {
                    Mode::Offscreen
                } else {
                    Mode::Visible
                });
            }
            "--exit-after-capture" => exit_after_capture = true,
            "--output" => output = args.next().ok_or("--output requires a directory")?.into(),
            _ => return Err(format!("unsupported argument: {}", arg.to_string_lossy()).into()),
        }
    }
    Ok(Options {
        mode: mode.ok_or("select --offscreen or --visible")?,
        model: model.ok_or("--model is required")?,
        asset: asset.ok_or("--asset is required")?,
        anisotropy,
        exit_after_capture,
        output,
    })
}

fn run_offscreen<B>(device: Device<B>, options: &Options) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    evidence::write_device(&options.output, &device)?;
    let input = scene::input(&options.model, &options.asset, options.anisotropy)?;
    evidence::write_inputs(&options.output, &input)?;
    let probes = oracle::reference(&input)?;
    oracle::write(&options.output, &probes)?;
    let mut renderer = Renderer::new(device)?;
    for (name, route) in [("forward", Route::Forward), ("deferred", Route::Deferred)] {
        let submission = match route {
            Route::Forward => block_on(renderer.submit_forward(
                &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
                input.snapshot.clone(),
                input.view,
                RenderBounds::new([-64.; 3], [64.; 3])?,
                &input.assets,
            ))?,
            Route::Deferred => block_on(renderer.submit_deferred(
                &DeferredCpuPipeline::new(BoundsVisibility),
                input.snapshot.clone(),
                input.view,
                RenderBounds::new([-64.; 3], [64.; 3])?,
                &input.assets,
            ))?,
        };
        let raw = block_on(renderer.readback(&submission))?;
        fs::write(options.output.join(format!("{name}-readback.rgba")), raw)?;
        let raw = fs::read(options.output.join(format!("{name}-readback.rgba")))?;
        if options.anisotropy == 1 {
            oracle::verify(&raw, &probes, matches!(route, Route::Deferred))?;
        }
        if renderer.pending() != 0 {
            return Err("completion did not retire tunnel owners".into());
        }
    }
    fs::write(
        options.output.join("result.json"),
        if options.anisotropy == 1 {
            "{\"result\":\"passed\",\"scope\":\"offscreen perspective tunnel aniso1 Material path with independent derivative mip oracle\"}\n"
        } else {
            "{\"result\":\"unverified\",\"scope\":\"aniso16 capability control; no scalar trilinear numerical claim\"}\n"
        },
    )?;
    Ok(())
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::{
        sync::Arc,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark()
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark()
        }
    }
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[allow(dead_code)]
fn json_string(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a string is infallible")
}
