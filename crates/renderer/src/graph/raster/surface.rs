//! Surface raster fixture plan and recording.
use super::*;

impl<B: Backend> SurfaceRasterPlan<B> {
    pub(crate) fn compile(
        device: &Device<B>,
        id: PlanId,
        len: u64,
        extent: [u32; 2],
    ) -> Result<Self> {
        if len == 0 || !len.is_multiple_of(VERTEX_STRIDE) || extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        if id.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let lane = graphics_lane(device)?;
        let mut graph = GraphBuilder::default();
        let source = graph
            .import_buffer_desc(surface_upload_desc(len), host_write_use())
            .map_err(|_| Error::InvalidRange)?;
        let vertex = graph
            .create_buffer_desc(surface_vertex_desc(len))
            .map_err(|_| Error::InvalidRange)?;
        let source_slot = BufferSlot::from(source);
        let vertex_slot = BufferSlot::from(vertex);
        let (upload, vertex) = graph
            .add_pass("surface-raster-upload", |p| {
                p.read_with_use(
                    source,
                    0..len,
                    copy_read_buffer_use(),
                    BufferUsage::COPY_SRC,
                )?;
                p.write_with_use(vertex, 0..len, copy_write_use(), BufferUsage::COPY_DST)
            })
            .map_err(|_| Error::InvalidState)?;
        let (raster, ()) = graph
            .add_pass("surface-raster-draw", |p| {
                p.read_with_use(vertex, 0..len, vertex_read_use(), BufferUsage::VERTEX)?;
                p.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        let execution = CompiledExecution::compile(&graph)?;
        if execution.logical().passes() != [upload, raster] {
            return Err(Error::InvalidState);
        }
        let upload_step = StepId { plan: id, index: 0 };
        let raster_step = StepId { plan: id, index: 1 };
        let present_step = StepId { plan: id, index: 2 };
        let mut steps = execution.steps(device, id, lane)?;
        if steps.len() != 2 {
            return Err(Error::InvalidState);
        }
        steps.push(StepDesc {
            lane,
            kind: StepKind::Present,
            waits: vec![StepWait {
                previous: raster_step,
                destination: AccessScope {
                    stages: StageMask::ALL_COMMANDS,
                    accesses: AccessMask::ALL,
                },
            }],
        });
        let submission = unsafe { SubmissionPlan::from_compiled(id, Arc::from(steps)) };
        submission.validate_for_execution(device)?;
        Ok(Self {
            id,
            len,
            extent,
            lane,
            submission,
            execution,
            upload_pass: upload,
            raster_pass: raster,
            upload_step,
            raster_step,
            present_step,
            source_slot,
            vertex_slot,
        })
    }
    pub(crate) fn id(&self) -> PlanId {
        self.id
    }
    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }
    pub(crate) fn upload_step(&self) -> StepId {
        self.upload_step
    }
    pub(crate) fn raster_step(&self) -> StepId {
        self.raster_step
    }
    pub(crate) fn present_step(&self) -> StepId {
        self.present_step
    }
    fn frame<'a>(
        &'a self,
        device: DeviceId,
        b: &'a SurfaceRasterBuffers<B>,
    ) -> std::result::Result<FrameBindings<'a, B>, BindingError>
    where
        B: ResourceBackend,
    {
        FrameBindings::new(
            &self.execution,
            device,
            vec![
                BoundBuffer::new(
                    self.source_slot,
                    b.upload_instance,
                    &b.upload,
                    &b.upload_desc,
                    host_write_use(),
                ),
                BoundBuffer::new(
                    self.vertex_slot,
                    b.vertex_instance,
                    &b.vertex,
                    &b.vertex_desc,
                    empty_buffer_use(),
                ),
            ],
            Vec::new(),
        )
    }
    pub(crate) fn validate_bindings(
        &self,
        d: DeviceId,
        b: &SurfaceRasterBuffers<B>,
    ) -> std::result::Result<(), BindingError>
    where
        B: ResourceBackend,
    {
        self.frame(d, b).map(|_| ())
    }
    pub(crate) fn record_upload(
        &self,
        e: &mut CommandEncoder<B>,
        b: &SurfaceRasterBuffers<B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        let f = self.frame(self.id.device, b).map_err(binding_error)?;
        let p = f.pass(self.upload_pass)?;
        let src = p.buffer(
            self.source_slot,
            0..self.len,
            AccessKind::Read,
            bc(copy_read_buffer_use(), BufferUsage::COPY_SRC),
        )?;
        let dst = p.buffer(
            self.vertex_slot,
            0..self.len,
            AccessKind::Write,
            bc(copy_write_use(), BufferUsage::COPY_DST),
        )?;
        p.transition(e)?;
        e.copy_buffer(src, 0, dst, 0, self.len)
    }
    pub(crate) fn record_raster(
        &self,
        e: &mut CommandEncoder<B>,
        b: &SurfaceRasterBuffers<B>,
        image: &SurfaceImage<B>,
        pipeline: &RasterPipeline<B>,
    ) -> Result<()>
    where
        B: RasterBackend + ResourceBackend,
    {
        self.check(image.device())?;
        self.check(pipeline.device())?;
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        let f = self.frame(self.id.device, b).map_err(binding_error)?;
        let p = f.pass(self.raster_pass)?;
        let vertex = p.buffer(
            self.vertex_slot,
            0..self.len,
            AccessKind::Read,
            bc(vertex_read_use(), BufferUsage::VERTEX),
        )?;
        p.transition(e)?;
        draw_surface(
            e,
            image,
            pipeline,
            vertex,
            self.extent,
            self.vertex_count()?,
        )
    }
    fn vertex_count(&self) -> Result<u32> {
        u32::try_from(self.len / VERTEX_STRIDE).map_err(|_| Error::InvalidRange)
    }
    fn check(&self, d: DeviceId) -> Result<()> {
        if d == self.id.device {
            Ok(())
        } else {
            Err(Error::WrongDevice)
        }
    }
}
impl<B: ExecutionBackend> SurfaceRasterPlan<B> {
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>> {
        let facts = outcome.facts();
        let expected = [self.upload_step, self.raster_step, self.present_step];
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(&expected) {
            return Err(Error::InvalidState);
        }
        if !facts
            .presents
            .is_some_and(|r| r.len() == 1 && r[0].accepted && r[0].step == Some(self.present_step))
        {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|p| {
                p.iter().find(|p| {
                    p.point.device() == self.id.device
                        && p.covers.lane_frontiers.iter().any(|&(lane, step)| {
                            lane == self.lane
                                && (step == self.raster_step || step == self.present_step)
                        })
                })
            })
            .map(|p| p.point.clone())
            .ok_or(Error::InvalidState)
    }
}
