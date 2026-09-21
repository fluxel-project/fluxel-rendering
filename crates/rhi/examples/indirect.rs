//! Indirect draw/dispatch example.

use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};

pub trait IndirectFixture {
    fn provider(&self) -> &PlatformProvider;
    fn request(&self) -> DeviceRequestDescriptor;
    fn execute_indirect(&self, device: &Device) -> RhiResult<()>;
}

pub async fn run(fixture: &impl IndirectFixture) -> RhiResult<()> {
    let device = fixture.provider().request_device(fixture.request()).await?;
    fixture.execute_indirect(&device)
}

fn main() {
    eprintln!("indirect: provide a host fixture and call run().await");
}
