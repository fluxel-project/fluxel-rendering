//! Public-API triangle example entry point.
//!
//! The shader artifact and host surface stay in the fixture, but provider and
//! device acquisition are the same portable calls used by an application.

use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};

/// Inputs supplied by a platform integration without exposing native handles.
pub trait TriangleFixture {
    fn provider(&self) -> &PlatformProvider;
    fn request(&self) -> DeviceRequestDescriptor;
    fn draw_triangle(&self, device: &Device) -> RhiResult<()>;
}

/// Acquire a device and execute the fixture's portable triangle recording.
pub async fn run(fixture: &impl TriangleFixture) -> RhiResult<()> {
    let device = fixture.provider().request_device(fixture.request()).await?;
    fixture.draw_triangle(&device)
}

fn main() {
    eprintln!("triangle: provide a host fixture and call triangle::run().await");
}
