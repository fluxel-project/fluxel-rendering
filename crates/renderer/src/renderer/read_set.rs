use super::{PendingWork, Renderer, RendererError, Submission, validate_read_set_requirements};
use crate::{
    graph::{
        ReadSetRasterFrame,
        read_set_raster::{TEXEL_OFFSET, UPLOAD_BYTES},
    },
    residency::ReadSetRasterBuffers,
};
use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_rhi::{
    api::{BindGroupLayout, CompletionRequest, Error, MapMode, RasterPipeline},
    backend::{ExecutionBackend, MappingBackend, RasterBackend, ReadbackBackend, ResourceBackend},
};
use std::ops::Range;

const INPUT_BYTES: usize = 20;
const UNIFORM_BYTES: u64 = 16;
const ALIGNMENT: u64 = 256;
const EXTENT: [u32; 2] = [8, 8];

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Submits the fixed private dynamic-uniform plus sampled-texture fixture.
    /// The input is exactly four little-endian `f32` tint components followed
    /// by one RGBA8 texel; this deliberately does not establish a material API.
    pub async fn submit_read_set_raster<K>(
        &mut self,
        snapshot: AssetSnapshot<K, Vec<u8>>,
        pipeline: RasterPipeline<B>,
        layout: BindGroupLayout<B>,
        range: Range<u64>,
    ) -> Result<Submission, RendererError>
    where
        K: AssetKind,
        B: RasterBackend,
    {
        self.ensure_open()?;
        validate(
            snapshot.value(),
            pipeline.device(),
            layout.device(),
            self.device.id(),
            &range,
        )
        .map_err(|e| self.observe_rhi_error(e))?;
        if range.end > self.device.capabilities().limits.max_buffer_bytes
            || u32::try_from(range.start).is_err()
        {
            return Err(RendererError::Rhi(Error::InvalidRange));
        }
        validate_read_set_requirements(
            &self.device,
            range.end,
            range.start,
            UPLOAD_BYTES,
            256 * u64::from(EXTENT[1]),
        )
        .map_err(|error| self.observe_renderer_error(error))?;
        let frame =
            ReadSetRasterFrame::compile(range.clone()).map_err(|e| self.observe_rhi_error(e))?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let serial = self.next_serial()?;
        let plan = self
            .read_set_raster_plan_for(&frame)
            .map_err(|e| self.observe_renderer_error(e))?;
        let buffers = ReadSetRasterBuffers::allocate(&self.device, range.end, plan.row_pitch())
            .map_err(|e| self.observe_rhi_error(e))?;
        frame
            .validate_bindings(self.device.id(), &buffers)
            .map_err(|e| self.observe_renderer_error(e.into()))?;
        let bindings = plan
            .bindings(layout, &buffers, &frame)
            .map_err(|e| self.observe_rhi_error(e))?;
        self.bindings_validated();
        let mut map = unsafe {
            self.device
                .map_buffer(&buffers.upload, 0..UPLOAD_BYTES, MapMode::Write)
                .await
        }
        .map_err(|e| self.observe_rhi_error(e))?;
        let mapped = map.bytes_mut().map_err(|e| self.observe_rhi_error(e))?;
        mapped.fill(0);
        mapped[..16].copy_from_slice(&snapshot.value()[..16]);
        mapped[TEXEL_OFFSET as usize..TEXEL_OFFSET as usize + 4]
            .copy_from_slice(&snapshot.value()[16..20]);
        map.flush(0..UPLOAD_BYTES)
            .map_err(|e| self.observe_rhi_error(e))?;
        drop(map);
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|e| self.observe_rhi_error(e))?;
        let upload = execution.commands(plan.upload_step(), |e| {
            plan.record_upload(e, &buffers, &frame)
        });
        let raster = match upload {
            Ok(()) => execution.commands(plan.raster_step(), |e| {
                plan.record_raster(e, &buffers, &pipeline, &bindings, &frame)
            }),
            Err(e) => Err(e),
        };
        let mut ticket = None;
        let recorded = match raster {
            Ok(()) => execution.commands(plan.readback_step(), |e| {
                ticket = Some(plan.record_readback(e, &buffers, &frame)?);
                Ok(())
            }),
            Err(e) => Err(e),
        };
        let outcome = unsafe { execution.finish() };
        let work = PendingWork::ReadSetRaster {
            _frame: frame,
            _buffers: Box::new(buffers),
            _bindings: bindings,
            _pipeline: pipeline,
            plan,
            extent: EXTENT,
            row_pitch: 256,
        };
        self.settle_submission(slot, serial, snapshot, ticket, (outcome, recorded), work)
    }
}
pub(super) fn validate(
    input: &[u8],
    pipeline: fluxel_rhi::api::DeviceId,
    layout: fluxel_rhi::api::DeviceId,
    device: fluxel_rhi::api::DeviceId,
    range: &Range<u64>,
) -> Result<(), Error> {
    if pipeline != device || layout != device {
        return Err(Error::WrongDevice);
    };
    if input.len() != INPUT_BYTES
        || !range.start.is_multiple_of(ALIGNMENT)
        || range.end.checked_sub(range.start) != Some(ALIGNMENT)
        || range.end < UNIFORM_BYTES
    {
        return Err(Error::InvalidRange);
    };
    Ok(())
}
