//! Renderer-owned dynamic-uniform/depth acceptance consumer.

mod runtime;
use runtime::run_offscreen;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[path = "../../rendering-integration/src/diagnostics.rs"]
mod diagnostics;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
#[path = "../../rendering-integration/src/capture.rs"]
mod integration_capture;

use std::{error::Error, fs, path::PathBuf};

use fluxel_renderer::{BoundsVisibility, ForwardCpuPipeline, Renderer};
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use renderer_04_dynamic_uniform_material::{
    inputs::{self, EXTENT},
    oracle,
    scene::{CUBE_COUNT, encode_matrix},
};

#[derive(Clone, Copy)]
enum Mode {
    Offscreen,
    Visible,
}

struct Options {
    mode: Mode,
    output: PathBuf,
    reverse: bool,
    time: f32,
    exit_after_capture: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_options()?;
    fs::create_dir_all(&options.output)?;
    #[cfg(all(windows, feature = "dx12"))]
    return native::run_dx12(options);
    #[cfg(all(windows, not(feature = "dx12"), feature = "native-gl-wgl"))]
    return native::run_wgl(options);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "native-gl-wgl"))))]
    Err("select --features dx12 or native-gl-wgl on Windows".into())
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let mut mode = None;
    let mut output = PathBuf::from("target/renderer-04-dynamic-uniform-material");
    let mut reverse = false;
    let mut time: f32 = 0.0;
    let mut exit_after_capture = false;
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        match argument.to_string_lossy().as_ref() {
            "--offscreen" => mode = Some(Mode::Offscreen),
            "--visible" => mode = Some(Mode::Visible),
            "--reverse" => reverse = true,
            "--time" => {
                time = args
                    .next()
                    .ok_or("--time requires a finite f32")?
                    .to_string_lossy()
                    .parse()?
            }
            "--output" => output = args.next().ok_or("--output requires a directory")?.into(),
            "--exit-after-capture" => exit_after_capture = true,
            _ => {
                return Err(format!("unsupported argument: {}", argument.to_string_lossy()).into());
            }
        }
    }
    if !time.is_finite() {
        return Err("--time must be finite".into());
    }
    Ok(Options {
        mode: mode.ok_or("select exactly one of --offscreen or --visible")?,
        output,
        reverse,
        time,
        exit_after_capture,
    })
}

fn write_inputs(
    output: &std::path::Path,
    input: &inputs::GridInput,
    options: &Options,
) -> Result<(), Box<dyn Error>> {
    fs::write(output.join("source-main.rs"), include_bytes!("main.rs"))?;
    fs::write(output.join("source-inputs.rs"), include_bytes!("inputs.rs"))?;
    fs::write(output.join("source-scene.rs"), include_bytes!("scene.rs"))?;
    fs::write(output.join("source-oracle.rs"), include_bytes!("oracle.rs"))?;
    fs::write(output.join("source-native.rs"), include_bytes!("native.rs"))?;
    fs::write(
        output.join("source-runtime.rs"),
        include_bytes!("runtime.rs"),
    )?;
    fs::write(output.join("camera.bin"), encode_matrix(input.camera))?;
    fs::write(output.join("Cargo.toml"), include_bytes!("../Cargo.toml"))?;
    fs::write(output.join("Cargo.lock"), include_bytes!("../Cargo.lock"))?;
    fs::write(
        output.join("material.txt"),
        format!("{:#?}", input.assets.materials[0].value()),
    )?;
    fs::write(
        output.join("scene-records.txt"),
        input
            .snapshot
            .record_ids()
            .map(|id| format!("{:#?}\n", input.snapshot.record(id)))
            .collect::<String>(),
    )?;
    let matrices: Vec<u8> = input
        .transforms
        .iter()
        .flat_map(|matrix| encode_matrix(*matrix))
        .collect();
    fs::write(output.join("model-matrices.bin"), matrices)?;
    fs::write(
        output.join("geometry.bin"),
        &input.assets.geometry[0].value().streams[0],
    )?;
    fs::write(
        output.join("depth-contract.json"),
        format!(
            "{{\"schemaVersion\":1,\"source\":\"RHI 04 dynamic_uniform_buffer\",\"drawCount\":{},\"indexedSource\":{},\"rendererGeometry\":\"{}\",\"depth\":{{\"format\":\"Depth32Float\",\"clear\":1.0,\"compare\":\"LessEqual\",\"write\":true}},\"cull\":\"None\",\"time\":{},\"reverseDrawOrder\":{}}}\n",
            input.transforms.len(),
            input.transforms.len() == CUBE_COUNT as usize,
            if input.transforms.len() == CUBE_COUNT as usize {
                "36-index expansion"
            } else {
                "three-vertex overlap control"
            },
            options.time,
            options.reverse
        ),
    )?;
    Ok(())
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context, Poll, Wake, Waker},
    };
    struct ThreadWake {
        ready: AtomicBool,
        thread: std::thread::Thread,
    }
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.ready.store(true, Ordering::Release);
            self.thread.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.ready.store(true, Ordering::Release);
            self.thread.unpark();
        }
    }
    let wake = Arc::new(ThreadWake {
        ready: AtomicBool::new(true),
        thread: std::thread::current(),
    });
    let waker = Waker::from(Arc::clone(&wake));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                while !wake.ready.swap(false, Ordering::AcqRel) {
                    std::thread::park();
                }
            }
        }
    }
}

#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod native;

#[allow(dead_code)]
fn json_string(value: &str) -> String {
    format!("{:?}", value)
}
