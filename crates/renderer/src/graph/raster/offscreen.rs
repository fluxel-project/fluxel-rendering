//! Offscreen raster fixture plan and recording.
use super::*;

impl<B: Backend> RasterPlan<B> {
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
        let row_pitch = aligned_row_pitch(extent[0])?;
        let lane = graphics_lane(device)?;
        let mut graph = GraphBuilder::default();
        let source = graph
            .import_buffer_desc(raster_upload_desc(len), host_write_use())
            .map_err(|_| Error::InvalidRange)?;
        let vertex = graph
            .create_buffer_desc(raster_vertex_desc(len))
            .map_err(|_| Error::InvalidRange)?;
        let target = graph
            .create_texture(raster_target_desc(extent))
            .map_err(|_| Error::InvalidRange)?;
        let rb_len = u64::from(row_pitch)
            .checked_mul(u64::from(extent[1]))
            .ok_or(Error::InvalidRange)?;
        let readback_buffer = graph
            .create_buffer_desc(raster_readback_desc(rb_len))
            .map_err(|_| Error::InvalidRange)?;
        let source_slot = BufferSlot::from(source);
        let vertex_slot = BufferSlot::from(vertex);
        let target_slot = TextureSlot::from(target);
        let readback_slot = BufferSlot::from(readback_buffer);
        let whole = TextureRange::new(TextureAspects::COLOR, 0..1, 0..1);
        let (upload, vertex) = graph
            .add_pass("raster-upload", |p| {
                p.read_with_use(
                    source,
                    0..len,
                    copy_read_buffer_use(),
                    BufferUsage::COPY_SRC,
                )?;
                p.write_with_use(vertex, 0..len, copy_write_use(), BufferUsage::COPY_DST)
            })
            .map_err(|_| Error::InvalidState)?;
        let (raster, target) = graph
            .add_pass("raster-draw", |p| {
                p.read_with_use(vertex, 0..len, vertex_read_use(), BufferUsage::VERTEX)?;
                p.write_texture_with_use(
                    target,
                    whole.clone(),
                    color_write_use(),
                    ImageLayout::ColorAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )
            })
            .map_err(|_| Error::InvalidState)?;
        let (readback, ()) = graph
            .add_pass("raster-readback", |p| {
                p.read_texture_with_use(
                    target,
                    whole.clone(),
                    copy_read_texture_use(),
                    ImageLayout::CopySrc,
                    TextureUsage::COPY_SRC,
                )?;
                p.write_with_use(
                    readback_buffer,
                    0..rb_len,
                    copy_write_use(),
                    BufferUsage::COPY_DST,
                )?;
                p.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        let execution = CompiledExecution::compile(&graph)?;
        if execution.logical().passes() != [upload, raster, readback] {
            return Err(Error::InvalidState);
        }
        let submission = execution.submission(device, id, lane)?;
        Ok(Self {
            id,
            len,
            extent,
            row_pitch,
            lane,
            submission,
            execution,
            upload_pass: upload,
            raster_pass: raster,
            readback_pass: readback,
            upload_step: StepId { plan: id, index: 0 },
            raster_step: StepId { plan: id, index: 1 },
            readback_step: StepId { plan: id, index: 2 },
            source_slot,
            vertex_slot,
            readback_slot,
            target_slot,
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
    pub(crate) fn readback_step(&self) -> StepId {
        self.readback_step
    }
    pub(crate) fn row_pitch(&self) -> u32 {
        self.row_pitch
    }
    pub(crate) fn readback_len(&self) -> u64 {
        u64::from(self.row_pitch) * u64::from(self.extent[1])
    }
    fn frame<'a>(
        &'a self,
        d: DeviceId,
        b: &'a RasterBuffers<B>,
    ) -> std::result::Result<FrameBindings<'a, B>, BindingError>
    where
        B: ResourceBackend,
    {
        FrameBindings::new(
            &self.execution,
            d,
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
                BoundBuffer::new(
                    self.readback_slot,
                    b.readback_instance,
                    &b.readback,
                    &b.readback_desc,
                    empty_buffer_use(),
                ),
            ],
            vec![BoundTexture::new(
                self.target_slot,
                b.target_instance,
                &b.target,
                &b.target_desc,
                empty_texture_use(),
                ImageLayout::Undefined,
            )],
        )
    }
    pub(crate) fn validate_bindings(
        &self,
        d: DeviceId,
        b: &RasterBuffers<B>,
    ) -> std::result::Result<(), BindingError>
    where
        B: ResourceBackend,
    {
        self.frame(d, b).map(|_| ())
    }
    pub(crate) fn record_upload(
        &self,
        e: &mut CommandEncoder<B>,
        b: &RasterBuffers<B>,
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
        b: &RasterBuffers<B>,
        pipeline: &RasterPipeline<B>,
    ) -> Result<()>
    where
        B: RasterBackend + ResourceBackend,
    {
        self.check(b.view.device())?;
        self.check(pipeline.device())?;
        let f = self.frame(self.id.device, b).map_err(binding_error)?;
        let p = f.pass(self.raster_pass)?;
        let vertex = p.buffer(
            self.vertex_slot,
            0..self.len,
            AccessKind::Read,
            bc(vertex_read_use(), BufferUsage::VERTEX),
        )?;
        let _ = p.texture(
            self.target_slot,
            TextureRange::new(TextureAspects::COLOR, 0..1, 0..1),
            AccessKind::Write,
            tc(
                color_write_use(),
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            ),
        )?;
        p.transition(e)?;
        draw_texture(
            e,
            &b.view,
            pipeline,
            vertex,
            self.extent,
            self.vertex_count()?,
        )
    }
    pub(crate) fn record_readback(
        &self,
        e: &mut CommandEncoder<B>,
        b: &RasterBuffers<B>,
    ) -> Result<ReadbackTicket<B>>
    where
        B: ReadbackBackend + ResourceBackend,
    {
        let f = self.frame(self.id.device, b).map_err(binding_error)?;
        let p = f.pass(self.readback_pass)?;
        let tex = p.texture(
            self.target_slot,
            TextureRange::new(TextureAspects::COLOR, 0..1, 0..1),
            AccessKind::Read,
            tc(
                copy_read_texture_use(),
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            ),
        )?;
        let dst = p.buffer(
            self.readback_slot,
            0..self.readback_len(),
            AccessKind::Write,
            bc(copy_write_use(), BufferUsage::COPY_DST),
        )?;
        p.transition(e)?;
        e.texture_to_buffer(
            fluxel_rhi::api::TextureRegion {
                texture: tex,
                mip: 0,
                origin: [0, 0, 0],
                extent: [self.extent[0], self.extent[1], 1],
                aspect: TextureAspects::COLOR,
            },
            dst,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: self.row_pitch,
                rows_per_image: self.extent[1],
            },
        )?;
        e.readback_buffer(dst, 0..self.readback_len())
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
impl<B: ExecutionBackend> RasterPlan<B> {
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>> {
        let f = outcome.facts();
        let expected = [self.upload_step, self.raster_step, self.readback_step];
        if f.status != ExecutionStatus::Accepted || f.accepted != Some(&expected) {
            return Err(Error::InvalidState);
        }
        f.checkpoints
            .and_then(|p| {
                p.iter().find(|p| {
                    p.point.device() == self.id.device
                        && p.covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == self.readback_step)
                })
            })
            .map(|p| p.point.clone())
            .ok_or(Error::InvalidState)
    }
}
