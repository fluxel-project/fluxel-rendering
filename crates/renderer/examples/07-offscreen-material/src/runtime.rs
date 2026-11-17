//! Renderer-owned offscreen execution for the paired capture controls.

use std::{
    error::Error,
    future::Future,
    path::Path,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use fluxel_renderer::{BoundsVisibility, ForwardCpuPipeline, GeometryData, Renderer};
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use crate::{evidence, inputs, oracle};

/// Executes both producer-color controls through one same-Graph submission each.
pub fn run_offscreen<B>(
    device: Device<B>,
    dragon: GeometryData,
    plane: GeometryData,
    output: &Path,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let expected = oracle::build(&dragon, &plane)?;
    let primary = inputs::build_capture_scene(dragon.clone(), plane.clone())?;
    let control = inputs::build_capture_scene_with_albedo(
        dragon.clone(),
        plane.clone(),
        oracle::CONTROL_ALBEDO,
    )?;
    evidence::write_pre_submission(output, &dragon, &plane, &expected, &primary, &control)?;
    let mut renderer = Renderer::new(device)?;
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let primary_submission = block_on(renderer.submit_forward_capture(
        &pipeline,
        primary.capture_snapshot,
        primary.capture_view,
        primary.visibility,
        &primary.assets,
        primary.slot.clone(),
        primary.receiver,
    ))?;
    let primary_rgba = block_on(renderer.readback(&primary_submission))?;
    std::fs::write(output.join("primary-readback.rgba"), &primary_rgba)?;
    if renderer.pending() != 0 {
        return Err("primary capture readback did not retire its Graph owner".into());
    }
    let control_submission = block_on(renderer.submit_forward_capture(
        &pipeline,
        control.capture_snapshot,
        control.capture_view,
        control.visibility,
        &control.assets,
        control.slot.clone(),
        control.receiver,
    ))?;
    let control_rgba = block_on(renderer.readback(&control_submission))?;
    std::fs::write(output.join("control-readback.rgba"), &control_rgba)?;
    if renderer.pending() != 0 {
        return Err("control capture readback did not retire its Graph owner".into());
    }
    oracle::verify_pair(&primary_rgba, &control_rgba, &expected)?;
    std::fs::write(
        output.join("result.json"),
        "{\"schemaVersion\":1,\"result\":\"same-Graph capture receiver controls verified\",\"scope\":\"offscreen capture texture consumer; no PBR or refraction claim\"}\n",
    )?;
    Ok(())
}

/// Drives the short native-device futures used by this standalone example.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}
struct ThreadWake(std::thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
