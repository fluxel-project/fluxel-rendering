use super::{
    PendingWork, Renderer, RendererError, Submission, validate_raster_input,
    validate_raster_requirements,
};
use crate::residency::{RasterBuffers, SurfaceRasterBuffers};
use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_rhi::{
    api::{CompletionRequest, Error, MapMode, PresentBinding, RasterPipeline, SurfaceImage},
    backend::{ExecutionBackend, MappingBackend, RasterBackend, ReadbackBackend, ResourceBackend},
};
impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Submits the fixed position-plus-color triangle fixture to an owned RGBA8 target.
    pub async fn submit_raster<K>(
        &mut self,
        snapshot: AssetSnapshot<K, Vec<u8>>,
        pipeline: RasterPipeline<B>,
        extent: [u32; 2],
    ) -> Result<Submission, RendererError>
    where
        K: AssetKind,
        B: RasterBackend,
    {
        self.ensure_open()?;
        let len = validate_raster_input(
            snapshot.value(),
            extent,
            pipeline.device(),
            self.device.id(),
        )
        .map_err(|error| self.observe_renderer_error(error))?;
        validate_raster_requirements(&self.device, len, extent, true, 0)
            .map_err(|error| self.observe_renderer_error(error))?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let serial = self.next_serial()?;
        let plan = self
            .raster_plan_for(len, extent)
            .map_err(|error| self.observe_renderer_error(error))?;
        let buffers = RasterBuffers::allocate(&self.device, len, extent, plan.row_pitch())
            .map_err(|error| self.observe_rhi_error(error))?;
        plan.validate_bindings(self.device.id(), &buffers)
            .map_err(|error| self.observe_renderer_error(error.into()))?;
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
        let raster = match upload {
            Ok(()) => execution.commands(plan.raster_step(), |e| {
                plan.record_raster(e, &buffers, &pipeline)
            }),
            Err(e) => Err(e),
        };
        let mut ticket = None;
        let recorded = match raster {
            Ok(()) => execution.commands(plan.readback_step(), |e| {
                ticket = Some(plan.record_readback(e, &buffers)?);
                Ok(())
            }),
            Err(e) => Err(e),
        };
        let outcome = unsafe { execution.finish() };
        let row_pitch = plan.row_pitch();
        let work = PendingWork::Raster {
            _buffers: Box::new(buffers),
            _pipeline: pipeline,
            plan,
            extent,
            row_pitch,
        };
        self.settle_submission(slot, serial, snapshot, ticket, (outcome, recorded), work)
    }

    /// Submits the fixed triangle fixture directly to one acquired surface image.
    ///
    /// The caller owns surface configuration and acquisition. This method takes
    /// the exclusive image lease and transfers it to the RHI present protocol;
    /// call [`Self::complete`] with the returned submission before replacing the
    /// surface configuration or releasing the host window.
    pub async fn submit_surface_raster<K>(
        &mut self,
        snapshot: AssetSnapshot<K, Vec<u8>>,
        pipeline: RasterPipeline<B>,
        image: SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<Submission, RendererError>
    where
        K: AssetKind,
        B: RasterBackend,
    {
        self.ensure_open()?;
        let len = validate_raster_input(
            snapshot.value(),
            extent,
            pipeline.device(),
            self.device.id(),
        )
        .map_err(|error| self.observe_renderer_error(error))?;
        validate_raster_requirements(&self.device, len, extent, false, 0)
            .map_err(|error| self.observe_renderer_error(error))?;
        if image.device() != self.device.id() {
            return Err(RendererError::Rhi(Error::WrongDevice));
        }
        if !image.is_current() {
            return Err(RendererError::Rhi(Error::SurfaceOutdated));
        }
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let serial = self.next_serial()?;
        let plan = self
            .surface_raster_plan_for(len, extent)
            .map_err(|error| self.observe_renderer_error(error))?;
        let buffers = SurfaceRasterBuffers::allocate(&self.device, len)
            .map_err(|error| self.observe_rhi_error(error))?;
        plan.validate_bindings(self.device.id(), &buffers)
            .map_err(|error| self.observe_renderer_error(error.into()))?;
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
        let upload = execution.commands(plan.upload_step(), |encoder| {
            plan.record_upload(encoder, &buffers)
        });
        let raster = match upload {
            Ok(()) => execution.commands(plan.raster_step(), |encoder| {
                plan.record_raster(encoder, &buffers, &image, &pipeline)
            }),
            Err(error) => Err(error),
        };
        let recorded = match raster {
            Ok(()) => execution
                .present(plan.present_step(), PresentBinding::new(image))
                .map_err(|failure| failure.error),
            Err(error) => Err(error),
        };
        let outcome = unsafe { execution.finish() };
        let work = PendingWork::SurfaceRaster {
            _buffers: buffers,
            _pipeline: pipeline,
            plan,
        };
        self.settle_submission(slot, serial, snapshot, None, (outcome, recorded), work)
    }
}
