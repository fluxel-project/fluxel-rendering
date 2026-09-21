//! Storage-buffer compute example.

use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};

/// Compute fixture.  Shader modules and the output oracle are backend code
/// form, while dispatch/submission/readback are public RHI operations.
pub trait ComputeFixture {
    fn provider(&self) -> &PlatformProvider;
    fn request(&self) -> DeviceRequestDescriptor;
    fn dispatch_and_verify(&self, device: &Device) -> RhiResult<()>;
}

/// Run one storage-buffer compute round trip.
pub async fn run(fixture: &impl ComputeFixture) -> RhiResult<()> {
    let device = fixture.provider().request_device(fixture.request()).await?;
    fixture.dispatch_and_verify(&device)
}

fn main() {
    eprintln!("compute: provide a host fixture and call run().await");
}
