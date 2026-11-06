use super::{
    PendingInputs, PendingWork, Renderer, RendererError, Submission, validate_compute_input,
};
use crate::residency::ComputeBuffers;
use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_rhi::{
    api::{BindGroupLayout, CompletionRequest, ComputePipeline, MapMode},
    backend::{CommandBackend, ExecutionBackend, MappingBackend, ReadbackBackend, ResourceBackend},
};
impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Submits the fixed in-place `u32` compute fixture.
    pub async fn submit_compute<K>(
        &mut self,
        snapshot: AssetSnapshot<K, Vec<u8>>,
        pipeline: ComputePipeline<B>,
        layout: BindGroupLayout<B>,
    ) -> Result<Submission, RendererError>
    where
        K: AssetKind,
        B: CommandBackend,
    {
        self.ensure_open()?;
        let len = validate_compute_input(
            snapshot.value(),
            pipeline.device(),
            layout.device(),
            &self.device,
        )
        .map_err(|error| self.observe_renderer_error(error))?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let serial = self.next_serial()?;
        let plan = self
            .compute_plan_for(len)
            .map_err(|error| self.observe_renderer_error(error))?;
        let buffers = ComputeBuffers::allocate(&self.device, len)
            .map_err(|error| self.observe_rhi_error(error))?;
        let bindings = plan
            .fixture_bindings(layout, &buffers)
            .map_err(|error| self.observe_rhi_error(error))?;
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
        let upload = execution.commands(plan.upload_step(), |e| plan.record_upload(e, &buffers));
        let compute = match upload {
            Ok(()) => execution.commands(plan.compute_step(), |e| {
                plan.record_compute(e, &buffers, &pipeline, &bindings, plan.count()?)
            }),
            Err(e) => Err(e),
        };
        let mut ticket = None;
        let recorded = match compute {
            Ok(()) => execution.commands(plan.readback_step(), |e| {
                ticket = Some(plan.record_readback(e, &buffers)?);
                Ok(())
            }),
            Err(e) => Err(e),
        };
        let outcome = unsafe { execution.finish() };
        let work = PendingWork::Compute {
            _buffers: buffers,
            _bindings: bindings,
            _pipeline: pipeline,
            plan,
        };
        self.settle_submission(
            slot,
            serial,
            PendingInputs::Bytes {
                _bytes: snapshot.value_arc(),
            },
            ticket,
            (outcome, recorded),
            work,
        )
    }
}
