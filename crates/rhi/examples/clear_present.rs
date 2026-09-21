//! Host-injected clear/present lifecycle workload.
//!
//! This is the minimal visible example: a host creates a presentation target;
//! the portable sequence configures it, acquires one non-cloneable frame, clears
//! it, presents after a plan point, waits for presentation, and re-acquires it.

use fluxel_rhi::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
    RecorderDescriptor, StoreOp,
};
use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};
use fluxel_rhi::api::presentation::{PresentationConfiguration, PresentationTarget};
use fluxel_rhi::api::shader::ShaderLocation;
use fluxel_rhi::api::submission::{SubmissionLaneId, SubmissionPlanBuilder};

/// Host-specific inputs for the portable presentation lifecycle.
pub trait ClearPresentFixture {
    /// Provider created by the platform integration.
    fn provider(&self) -> &PlatformProvider;
    /// Request which names the target as a required presentation route.
    fn device_request(&self) -> DeviceRequestDescriptor;
    /// The host-owned target (HWND/ANativeWindow/CAMetalLayer/canvas remain private).
    fn target(&self) -> &PresentationTarget;
    /// Configuration selected from this device's presentation capabilities.
    fn configuration(&self, device: &Device) -> RhiResult<PresentationConfiguration>;
    /// Lane which accepts raster work.
    fn raster_lane(&self, device: &Device) -> RhiResult<SubmissionLaneId>;
}

/// Runs one clear → present → wait → acquire/abandon cycle.
pub async fn run(
    fixture: &impl ClearPresentFixture,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let device = fixture
        .provider()
        .request_device(fixture.device_request())
        .await?;
    let configuration = fixture.configuration(&device)?;
    let lane = fixture.raster_lane(&device)?;
    let mut surface = device
        .configure_presentation(fixture.target(), &configuration)
        .await?;
    let frame = surface.acquire().await?;
    let scope = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Frame(frame.attachment()),
            load: LoadOp::Clear(ColorClearValue::Float([0.05, 0.2, 0.4, 1.0])),
            store: StoreOp::Store,
            resolve: None,
            depth_slice: None,
        },
    );
    let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
    recorder.begin_raster(&scope)?.end()?;
    let work = recorder.finish()?;
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan.add_batch(lane, vec![work])?;
    plan.present_after(frame, point)?;
    let receipt = device.submit(plan.build()?).await?;
    let present = receipt
        .presents()
        .first()
        .expect("a submitted present plan produces one present receipt");
    let _state = device.wait_present(present.id()).await?;

    // Reacquisition demonstrates that present retired the old frame ownership.
    let next = surface.acquire().await?;
    next.abandon().await?;
    Ok(())
}

fn main() {
    eprintln!(
        "clear_present is a host-injected source template; see examples/README.md for integration"
    );
}
