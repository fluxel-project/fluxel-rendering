//! Host-injected off-screen triangle workload.
//!
//! This source demonstrates the portable raster/readback sequence. The fixture
//! supplies only shader-code-form-specific pipeline and vertex data; no native
//! provider, window, or graphics context becomes part of the RHI API.

use fluxel_rhi::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
    RecorderDescriptor, StoreOp,
};
use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::format::TextureFormat;
use fluxel_rhi::api::identity::Label;
use fluxel_rhi::api::pipeline::RasterPipeline;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};
use fluxel_rhi::api::resource::buffer::BufferBinding;
use fluxel_rhi::api::resource::subresource::{Origin3d, TextureAspect, TextureSubresourceLayers};
use fluxel_rhi::api::resource::texture::{Extent3d, TextureDescriptor, TextureUsage};
use fluxel_rhi::api::resource::transfer::{ReadbackRequest, ReadbackViewData};
use fluxel_rhi::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use fluxel_rhi::api::shader::ShaderLocation;
use fluxel_rhi::api::submission::{SubmissionLaneId, SubmissionPlanBuilder};

/// Portable objects whose creation depends on the fixture's shader code form.
pub struct TriangleResources {
    /// Raster pipeline targeting one `Rgba8Unorm` attachment at location zero.
    pub pipeline: RasterPipeline,
    /// Optional slot-zero vertex stream; vertex-index shaders use `None`.
    pub vertex: Option<BufferBinding>,
}

/// Platform fixture for the portable triangle sequence.
#[allow(
    async_fn_in_trait,
    reason = "this source-only template is called directly by a host integration, not trait-object erased"
)]
pub trait OffscreenTriangleFixture {
    /// Host-created provider.
    fn provider(&self) -> &PlatformProvider;
    /// Headless device request.
    fn device_request(&self) -> DeviceRequestDescriptor;
    /// Raster-capable lane selected from the device's published facts.
    fn raster_lane(&self, device: &Device) -> RhiResult<SubmissionLaneId>;
    /// Creates the pipeline and any optional vertex resource.
    async fn create_resources(&self, device: &Device) -> RhiResult<TriangleResources>;
    /// Verifies the center texel after row-pitch-aware decoding by the fixture.
    fn verify_center_pixel(&self, bytes: &[u8], bytes_per_row: u32) -> RhiResult<()>;
}

/// Renders a triangle to an 8×8 texture and passes its mapped bytes to the
/// fixture's portable-format oracle.
pub async fn run(fixture: &impl OffscreenTriangleFixture) -> RhiResult<()> {
    let device = fixture
        .provider()
        .request_device(fixture.device_request())
        .await?;
    let lane = fixture.raster_lane(&device)?;
    let resources = fixture.create_resources(&device).await?;
    let extent = Extent3d::d2(8, 8);
    let target = device.create_texture(&TextureDescriptor::new_2d(
        extent.width,
        extent.height,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
    ))?;
    let view = device.create_texture_view(
        &target,
        &TextureViewDescriptor::whole(&target, TextureViewDimension::D2)?,
    )?;
    let scope = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Texture(view),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
            resolve: None,
            depth_slice: None,
        },
    );
    let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
    {
        let mut raster = recorder.begin_raster(&scope)?;
        raster.set_pipeline(&resources.pipeline)?;
        if let Some(vertex) = resources.vertex.as_ref() {
            raster.set_vertex_buffer(0, vertex)?;
        }
        raster.draw(0..3, 0..1)?;
        raster.end()?;
    }
    let ticket = recorder.encode_readback(ReadbackRequest::Texture {
        label: Label(Some("offscreen_triangle pixels".into())),
        src: target,
        subresource: TextureSubresourceLayers {
            aspect: TextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        origin: Origin3d { x: 0, y: 0, z: 0 },
        extent,
    })?;
    let work = recorder.finish()?;
    let mut plan = SubmissionPlanBuilder::new(&device);
    plan.add_batch(lane, vec![work])?;
    let receipt = device.submit(plan.build()?).await?;
    let _completion = device.wait_completion(receipt.completion()).await?;
    let mapped = ticket.read().await?;
    let ReadbackViewData::Texture { bytes, layout, .. } = mapped.data() else {
        unreachable!("a texture readback request must expose texture bytes");
    };
    fixture.verify_center_pixel(bytes, layout.bytes_per_row)
}

fn main() {
    eprintln!(
        "offscreen_triangle is a host-injected source template; see examples/README.md for integration"
    );
}
