//! Fixed raster fixtures lowered through shared graph execution contracts.
use super::{
    binding::{BindingError, BufferSlot, TextureSlot},
    definition::{
        AccessKind, BufferAccessContract, GraphBuilder, GraphPass, TextureAccessContract,
        TextureRange,
    },
    execution::{BoundBuffer, BoundTexture, CompiledExecution, FrameBindings, binding_error},
};
use crate::residency::{RasterBuffers, SurfaceRasterBuffers};
use fluxel_rhi::{
    api::{
        AccessMask, AccessScope, AttachmentView, BufferDesc, BufferUsage, BufferUses, ClearColor,
        ColorAttachment, CommandEncoder, CompletionPoint, Device, DeviceId, Error, ExecutionStatus,
        ImageLayout, LaneClass, LaneId, LoadOp, MemoryClass, PlanId, RasterPassDesc,
        RasterPipeline, ReadbackTicket, Rect, Result, StageMask, StepDesc, StepId, StepKind,
        StepWait, StoreOp, SubmissionPlan, SurfaceImage, TexelCopyBufferLayout, TextureAspects,
        TextureUsage, TextureUses, Viewport,
    },
    backend::{
        Backend, CommandBackend, ExecutionBackend, ExecutionMode, ExecutionOutcome, RasterBackend,
        ReadbackBackend, ResourceBackend,
    },
};
use std::sync::Arc;
type Outcome<B> = <<B as ExecutionBackend>::Mode as ExecutionMode<B>>::Outcome;
const VERTEX_STRIDE: u64 = 24;
pub(crate) struct RasterPlan<B: Backend> {
    id: PlanId,
    len: u64,
    extent: [u32; 2],
    row_pitch: u32,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    execution: CompiledExecution,
    upload_pass: GraphPass,
    raster_pass: GraphPass,
    readback_pass: GraphPass,
    upload_step: StepId,
    raster_step: StepId,
    readback_step: StepId,
    source_slot: BufferSlot,
    vertex_slot: BufferSlot,
    readback_slot: BufferSlot,
    target_slot: TextureSlot,
}
pub(crate) struct SurfaceRasterPlan<B: Backend> {
    id: PlanId,
    len: u64,
    extent: [u32; 2],
    lane: LaneId,
    submission: SubmissionPlan<B>,
    execution: CompiledExecution,
    upload_pass: GraphPass,
    raster_pass: GraphPass,
    upload_step: StepId,
    raster_step: StepId,
    present_step: StepId,
    source_slot: BufferSlot,
    vertex_slot: BufferSlot,
}
impl<B: Backend> SurfaceRasterPlan<B> {
    pub(crate) fn compile(
        device: &Device<B>,
        id: PlanId,
        len: u64,
        extent: [u32; 2],
    ) -> Result<Self> {
        if len == 0 || len % VERTEX_STRIDE != 0 || extent.contains(&0) {
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
impl<B: Backend> RasterPlan<B> {
    pub(crate) fn compile(
        device: &Device<B>,
        id: PlanId,
        len: u64,
        extent: [u32; 2],
    ) -> Result<Self> {
        if len == 0 || len % VERTEX_STRIDE != 0 || extent.contains(&0) {
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
fn graphics_lane<B: Backend>(d: &Device<B>) -> Result<LaneId> {
    d.lanes()
        .iter()
        .find(|x| x.class == LaneClass::Graphics)
        .map(|x| x.id)
        .ok_or(Error::WrongLane)
}
fn bc(uses: BufferUses, descriptor_usage: BufferUsage) -> BufferAccessContract {
    BufferAccessContract {
        uses,
        descriptor_usage,
    }
}
fn tc(
    uses: TextureUses,
    layout: ImageLayout,
    descriptor_usage: TextureUsage,
) -> TextureAccessContract {
    TextureAccessContract {
        uses,
        layout,
        descriptor_usage,
    }
}
fn draw_texture<B: RasterBackend>(
    e: &mut CommandEncoder<B>,
    view: &fluxel_rhi::api::TextureView<B>,
    pipe: &RasterPipeline<B>,
    v: &fluxel_rhi::api::Buffer<B>,
    extent: [u32; 2],
    count: u32,
) -> Result<()> {
    let c = [Some(ColorAttachment {
        view: AttachmentView::Texture(view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 0.])),
        store: StoreOp::Store,
    })];
    draw(e, "fluxel raster fixture", &c, pipe, v, extent, count)
}
fn draw_surface<B: RasterBackend>(
    e: &mut CommandEncoder<B>,
    image: &SurfaceImage<B>,
    pipe: &RasterPipeline<B>,
    v: &fluxel_rhi::api::Buffer<B>,
    extent: [u32; 2],
    count: u32,
) -> Result<()> {
    let c = [Some(ColorAttachment {
        view: AttachmentView::Surface(image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    draw(
        e,
        "fluxel surface raster fixture",
        &c,
        pipe,
        v,
        extent,
        count,
    )
}
fn draw<B: RasterBackend>(
    e: &mut CommandEncoder<B>,
    label: &'static str,
    c: &[Option<ColorAttachment<'_, B>>],
    pipe: &RasterPipeline<B>,
    v: &fluxel_rhi::api::Buffer<B>,
    x: [u32; 2],
    n: u32,
) -> Result<()> {
    let mut p = e.begin_raster(&RasterPassDesc {
        label: Some(label),
        colors: c,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: x[0],
            height: x[1],
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })?;
    p.viewport(Viewport {
        x: 0.,
        y: 0.,
        width: x[0] as f32,
        height: x[1] as f32,
        min_depth: 0.,
        max_depth: 1.,
    })?;
    p.scissor(Rect {
        x: 0,
        y: 0,
        width: x[0],
        height: x[1],
    })?;
    p.set_pipeline(pipe)?;
    p.set_vertex_buffer(0, v, 0..u64::from(n) * VERTEX_STRIDE)?;
    p.draw(0..n, 0..1)?;
    p.finish()
}
fn raster_target_desc(x: [u32; 2]) -> fluxel_rhi::api::TextureDesc {
    fluxel_rhi::api::TextureDesc {
        label: Some("fluxel raster target".to_owned()),
        dimension: fluxel_rhi::api::TextureDimension::D2,
        extent: [x[0], x[1], 1],
        array_layers: 1,
        format: fluxel_rhi::api::TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        view_formats: Vec::new(),
    }
}
fn raster_upload_desc(s: u64) -> BufferDesc {
    BufferDesc {
        label: Some("fluxel renderer raster upload".to_owned()),
        size: s,
        usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
        memory: MemoryClass::Upload,
    }
}
fn raster_vertex_desc(s: u64) -> BufferDesc {
    BufferDesc {
        label: Some("fluxel renderer raster vertices".to_owned()),
        size: s,
        usage: BufferUsage::COPY_DST | BufferUsage::VERTEX,
        memory: MemoryClass::Device,
    }
}
fn raster_readback_desc(s: u64) -> BufferDesc {
    BufferDesc {
        label: Some("fluxel renderer raster readback".to_owned()),
        size: s,
        usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        memory: MemoryClass::Readback,
    }
}
fn surface_upload_desc(s: u64) -> BufferDesc {
    BufferDesc {
        label: Some("fluxel renderer surface raster upload".to_owned()),
        size: s,
        usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
        memory: MemoryClass::Upload,
    }
}
fn surface_vertex_desc(s: u64) -> BufferDesc {
    BufferDesc {
        label: Some("fluxel renderer surface raster vertices".to_owned()),
        size: s,
        usage: BufferUsage::COPY_DST | BufferUsage::VERTEX,
        memory: MemoryClass::Device,
    }
}
fn aligned_row_pitch(w: u32) -> Result<u32> {
    w.checked_mul(4)
        .ok_or(Error::InvalidRange)?
        .checked_add(255)
        .ok_or(Error::InvalidRange)
        .map(|x| x / 256 * 256)
}
fn empty_buffer_use() -> BufferUses {
    BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}
fn host_write_use() -> BufferUses {
    BufferUses {
        stages: StageMask::HOST,
        accesses: AccessMask::HOST_WRITE,
    }
}
fn copy_write_use() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}
fn vertex_read_use() -> BufferUses {
    BufferUses {
        stages: StageMask::VERTEX_INPUT,
        accesses: AccessMask::VERTEX_READ,
    }
}
fn copy_read_buffer_use() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}
fn empty_texture_use() -> TextureUses {
    TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}
fn color_write_use() -> TextureUses {
    TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    }
}
fn copy_read_texture_use() -> TextureUses {
    TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}
