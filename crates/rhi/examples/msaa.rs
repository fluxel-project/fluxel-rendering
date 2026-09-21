//! Capability-gated MSAA/resolve example.

use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};

/// The fixture must select a published sample count and resolve route; it may
/// return `Unsupported` when the adapter does not expose one.
pub trait MsaaFixture {
    fn provider(&self) -> &PlatformProvider;
    fn request(&self) -> DeviceRequestDescriptor;
    fn render_and_resolve(&self, device: &Device) -> RhiResult<()>;
}

pub async fn run(fixture: &impl MsaaFixture) -> RhiResult<()> {
    let device = fixture.provider().request_device(fixture.request()).await?;
    fixture.render_and_resolve(&device)
}

fn main() {
    eprintln!("msaa: provide a host fixture and call run().await");
}
