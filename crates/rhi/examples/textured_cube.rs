//! Textured-cube workload template.
//!
//! A cube combines indexed geometry, a sampled texture and a sampler.  The
//! fixture owns only shader-code-form compilation and vertex/index uploads;
//! the provider/device boundary remains entirely public RHI.

use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};

/// Host/platform inputs for the textured-cube workload.
pub trait TexturedCubeFixture {
    fn provider(&self) -> &PlatformProvider;
    fn request(&self) -> DeviceRequestDescriptor;
    fn render_textured_cube(&self, device: &Device) -> RhiResult<()>;
}

/// Acquire a device and run one textured-cube frame.
pub async fn run(fixture: &impl TexturedCubeFixture) -> RhiResult<()> {
    let device = fixture.provider().request_device(fixture.request()).await?;
    fixture.render_textured_cube(&device)
}

fn main() {
    eprintln!("textured_cube: provide a host fixture and call run().await");
}
