use super::{PendingWork, Renderer, RendererError, Submission, validate_copy_requirements};
use crate::graph::CopyFrame;
use crate::residency::UploadBuffers;
use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_rhi::{
    api::{CompletionRequest, MapMode},
    backend::{ExecutionBackend, MappingBackend, ReadbackBackend, ResourceBackend},
};
use std::ops::Range;
impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Uploads one immutable byte generation into fresh RHI buffers.
    pub async fn submit<K: AssetKind>(
        &mut self,
        snapshot: AssetSnapshot<K, Vec<u8>>,
    ) -> Result<Submission, RendererError> {
        let len =
            u64::try_from(snapshot.value().len()).map_err(|_| RendererError::IdentityExhausted)?;
        self.submit_copy_range(snapshot, 0..len).await
    }

    /// Uploads an immutable generation and copies only the selected byte range.
    ///
    /// Source ranges and resource extents are validated independently for each
    /// submission while compatible copies reuse the same device execution plan.
    pub async fn submit_copy_range<K: AssetKind>(
        &mut self,
        snapshot: AssetSnapshot<K, Vec<u8>>,
        source_range: Range<u64>,
    ) -> Result<Submission, RendererError> {
        self.ensure_open()?;
        let len =
            u64::try_from(snapshot.value().len()).map_err(|_| RendererError::IdentityExhausted)?;
        if len == 0 {
            return Err(RendererError::EmptyAsset);
        }
        let frame = CopyFrame::compile_range(len, source_range.clone())?;
        validate_copy_requirements(&self.device, len, source_range.end - source_range.start)
            .map_err(|error| self.observe_renderer_error(error))?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let serial = self.next_serial()?;
        let plan = self
            .copy_plan_for(&frame)
            .map_err(|error| self.observe_renderer_error(error))?;
        let buffers =
            UploadBuffers::allocate(&self.device, len, source_range.end - source_range.start)
                .map_err(|error| self.observe_rhi_error(error))?;
        frame
            .validate_bindings(self.device.id(), &buffers)
            .map_err(RendererError::from)
            .map_err(|error| self.observe_renderer_error(error))?;
        self.bindings_validated();
        let mut map = unsafe {
            self.device
                .map_buffer(&buffers.upload, 0..len, MapMode::Write)
                .await
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        map.bytes_mut()
            .map_err(|error| self.observe_rhi_error(error))?
            .copy_from_slice(snapshot.value());
        map.flush(0..len)
            .map_err(|error| self.observe_rhi_error(error))?;
        drop(map);
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        let upload = execution.commands(plan.upload_step(), |e| {
            plan.record_upload(e, &buffers, &frame)
        });
        let mut ticket = None;
        let recorded = match upload {
            Ok(()) => execution.commands(plan.readback_step(), |e| {
                ticket = Some(plan.record_readback(e, &buffers, &frame)?);
                Ok(())
            }),
            Err(e) => Err(e),
        };
        let outcome = unsafe { execution.finish() };
        self.settle_submission(
            slot,
            serial,
            snapshot,
            ticket,
            (outcome, recorded),
            PendingWork::Copy {
                _buffers: buffers,
                _frame: frame,
                plan,
            },
        )
    }
}
