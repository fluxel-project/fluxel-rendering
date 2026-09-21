//! Host-injected storage-buffer compute workload.
//!
//! This is a compile-checked source template rather than an independently
//! runnable binary. Native provider construction belongs to Fluxel's platform
//! integration, so an application supplies a [`HeadlessComputeFixture`] whose
//! implementation owns its DX12/Vulkan/Metal/WebGPU/GL setup. The RHI command
//! sequence below is portable and unchanged across those implementations.

use fluxel_rhi::api::binding::{BindGroup, BindGroupIndex};
use fluxel_rhi::api::command::{ComputeScopeDescriptor, RecorderDescriptor};
use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::identity::Label;
use fluxel_rhi::api::pipeline::ComputePipeline;
use fluxel_rhi::api::platform::{Device, DeviceRequestDescriptor, PlatformProvider};
use fluxel_rhi::api::resource::buffer::{Buffer, BufferRange};
use fluxel_rhi::api::resource::transfer::{ReadbackRequest, ReadbackViewData};
use fluxel_rhi::api::submission::{SubmissionLaneId, SubmissionPlanBuilder};

/// Code-form-specific objects required by the portable compute sequence.
///
/// An implementation may create DXIL, SPIR-V, MSL, WGSL, or GLSL shader
/// artifacts. Those forms are intentionally outside this example: only the
/// resulting portable pipeline, bind group, and output buffer cross this seam.
pub struct ComputeResources {
    /// Pipeline whose entry point writes the output buffer.
    pub pipeline: ComputePipeline,
    /// Bind group accepted by `pipeline` at group zero.
    pub bind_group: BindGroup,
    /// Storage buffer written by the dispatch and subsequently read back.
    pub output: Buffer,
    /// Exact readable byte range of `output`.
    pub output_size: u64,
}

/// The small platform-specific part of the compute example.
///
/// A host integration retains every native object needed to create the
/// provider. It does not leak those handles through this trait; only portable
/// RHI types are returned to the common workload.
#[allow(
    async_fn_in_trait,
    reason = "this source-only template is called directly by a host integration, not trait-object erased"
)]
pub trait HeadlessComputeFixture {
    /// Provider constructed by the host/platform layer.
    fn provider(&self) -> &PlatformProvider;

    /// Device requirements and adapter selection for this workload.
    fn device_request(&self) -> DeviceRequestDescriptor;

    /// Selects a lane that accepts compute work on `device`.
    fn compute_lane(&self, device: &Device) -> RhiResult<SubmissionLaneId>;

    /// Creates code-form-specific shader resources, then exposes portable
    /// objects for the common sequence.
    async fn create_resources(&self, device: &Device) -> RhiResult<ComputeResources>;

    /// Checks the shader-specific output oracle.
    fn verify_output(&self, bytes: &[u8]) -> RhiResult<()>;
}

/// Executes one portable `set_pipeline -> set_bind_group -> dispatch ->
/// readback` round trip.
///
/// The await points are intentional: device and pipeline creation may wait for
/// native/browser completion; command recording itself remains synchronous;
/// readback returns an RAII mapping lease whose `Drop` closes any native map.
pub async fn run(fixture: &impl HeadlessComputeFixture) -> RhiResult<()> {
    let device = fixture
        .provider()
        .request_device(fixture.device_request())
        .await?;
    let lane = fixture.compute_lane(&device)?;
    let resources = fixture.create_resources(&device).await?;

    let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
    {
        let mut compute = recorder.begin_compute(&ComputeScopeDescriptor::new())?;
        compute.set_pipeline(&resources.pipeline)?;
        compute.set_bind_group(BindGroupIndex::new(0), &resources.bind_group, &[])?;
        compute.dispatch(1, 1, 1)?;
        compute.end()?;
    }
    let ticket = recorder.encode_readback(ReadbackRequest::Buffer {
        label: Label(Some("headless_compute output".into())),
        src: resources.output,
        range: BufferRange::new(0, resources.output_size),
    })?;
    let work = recorder.finish()?;

    let mut plan = SubmissionPlanBuilder::new(&device);
    plan.add_batch(lane, vec![work])?;
    let receipt = device.submit(plan.build()?).await?;
    // Submission acceptance and terminal GPU completion are distinct facts.
    let _completion = device.wait_completion(receipt.completion()).await?;

    let view = ticket.read().await?;
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        unreachable!("a buffer readback request must expose buffer bytes");
    };
    fixture.verify_output(bytes)
}

fn main() {
    eprintln!(
        "headless_compute is a host-injected source template; see examples/README.md for integration"
    );
}
