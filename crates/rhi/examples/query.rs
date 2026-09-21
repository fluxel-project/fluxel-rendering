//! Occlusion-query resolve/readback example.

use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};

pub trait QueryFixture {
    fn provider(&self) -> &PlatformProvider;
    fn request(&self) -> DeviceRequestDescriptor;
    fn query_resolve_and_verify(&self, device: &Device) -> RhiResult<()>;
}

pub async fn run(fixture: &impl QueryFixture) -> RhiResult<()> {
    let device = fixture.provider().request_device(fixture.request()).await?;
    fixture.query_resolve_and_verify(&device)
}

fn main() {
    eprintln!("query: provide a host fixture and call run().await");
}
