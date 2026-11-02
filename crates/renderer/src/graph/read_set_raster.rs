//! Fixed uniform-plus-texture fixture over the shared Graph execution core.

use super::{
    binding::{BindingError, BufferSlot, TextureSlot},
    compiler::CompiledGraph,
    definition::{
        AccessKind, BufferAccessContract, GraphBuilder, GraphPass, TextureAccessContract,
        TextureRange,
    },
    execution::{BoundBuffer, BoundTexture, CompiledExecution, FrameBindings, binding_error},
};
use crate::residency::ReadSetRasterBuffers;
use fluxel_rhi::{
    api::{
        AccessMask, AttachmentView, BindGroup, BindGroupDesc, BindGroupEntry, BindGroupLayout,
        BindingResource, BufferDesc, BufferUsage, BufferUses, ClearColor, ColorAttachment,
        CommandEncoder, CompletionPoint, Device, DeviceId, Error, ExecutionStatus, ImageLayout,
        LaneClass, LaneId, LoadOp, MemoryClass, PlanId, RasterPassDesc, RasterPipeline,
        ReadbackTicket, Rect, Result, StageMask, StepId, StoreOp, SubmissionPlan,
        TexelCopyBufferLayout, TextureAspects, TextureRegion, TextureUsage, TextureUses, Viewport,
    },
    backend::{
        Backend, ExecutionBackend, ExecutionMode, ExecutionOutcome, RasterBackend, ReadbackBackend,
        ResourceBackend,
    },
};
use std::ops::Range;

type Outcome<B> = <<B as ExecutionBackend>::Mode as ExecutionMode<B>>::Outcome;
pub(crate) const UPLOAD_BYTES: u64 = 768;
pub(crate) const TEXEL_OFFSET: u64 = 512;
pub(crate) const ROW_PITCH: u32 = 256;
pub(crate) const EXTENT: [u32; 2] = [8, 8];

/// Current physical accesses, separate from the reusable device topology.
pub(crate) struct ReadSetRasterFrame {
    execution: CompiledExecution,
    uniform_range: Range<u64>,
    upload: GraphPass,
    raster: GraphPass,
    readback: GraphPass,
    source_slot: BufferSlot,
    uniform_slot: BufferSlot,
    readback_slot: BufferSlot,
    texture_slot: TextureSlot,
    target_slot: TextureSlot,
}

impl ReadSetRasterFrame {
    pub(crate) fn compile(destination: Range<u64>) -> Result<Self> {
        if destination.start % 256 != 0
            || destination.end.checked_sub(destination.start) != Some(256)
            || u32::try_from(destination.start).is_err()
        {
            return Err(Error::InvalidRange);
        }
        let uniform_range = destination.start
            ..destination
                .start
                .checked_add(16)
                .ok_or(Error::InvalidRange)?;
        let mut graph = GraphBuilder::default();
        let upload = graph
            .import_buffer_desc(
                BufferDesc {
                    label: None,
                    size: UPLOAD_BYTES,
                    usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
                    memory: MemoryClass::Upload,
                },
                host(),
            )
            .map_err(|_| Error::InvalidState)?;
        let uniform = graph
            .create_buffer_desc(BufferDesc {
                label: None,
                size: destination.end,
                usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
                memory: MemoryClass::Device,
            })
            .map_err(|_| Error::InvalidState)?;
        let source = graph
            .create_texture(tex_desc(
                "read-set source",
                [1, 1],
                TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            ))
            .map_err(|_| Error::InvalidState)?;
        let target = graph
            .create_texture(tex_desc(
                "read-set target",
                EXTENT,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            ))
            .map_err(|_| Error::InvalidState)?;
        let readback_buffer = graph
            .create_buffer_desc(BufferDesc {
                label: None,
                size: u64::from(ROW_PITCH) * u64::from(EXTENT[1]),
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .map_err(|_| Error::InvalidState)?;
        let (upload_pass, (uniform, source)) = graph
            .add_pass("read-set-upload", |pass| {
                pass.read_with_use(upload, 0..16, copy_read(), BufferUsage::COPY_SRC)?;
                pass.read_with_use(
                    upload,
                    TEXEL_OFFSET..TEXEL_OFFSET + 4,
                    copy_read(),
                    BufferUsage::COPY_SRC,
                )?;
                let uniform = pass.write_with_use(
                    uniform,
                    uniform_range.clone(),
                    copy_write(),
                    BufferUsage::COPY_DST,
                )?;
                let source = pass.write_texture_with_use(
                    source,
                    whole(),
                    copy_tex_write(),
                    ImageLayout::CopyDst,
                    TextureUsage::COPY_DST,
                )?;
                Ok((uniform, source))
            })
            .map_err(|_| Error::InvalidState)?;
        let (raster, target) = graph
            .add_pass("read-set-raster", |pass| {
                pass.read_with_use(
                    uniform,
                    uniform_range.clone(),
                    shader_read(),
                    BufferUsage::UNIFORM,
                )?;
                pass.read_texture_with_use(
                    source,
                    whole(),
                    shader_tex_read(),
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
                pass.write_texture_with_use(
                    target,
                    whole(),
                    color_write(),
                    ImageLayout::ColorAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )
            })
            .map_err(|_| Error::InvalidState)?;
        let (readback, ()) = graph
            .add_pass("read-set-readback", |pass| {
                pass.read_texture_with_use(
                    target,
                    whole(),
                    copy_tex_read(),
                    ImageLayout::CopySrc,
                    TextureUsage::COPY_SRC,
                )?;
                pass.write_with_use(
                    readback_buffer,
                    0..u64::from(ROW_PITCH) * u64::from(EXTENT[1]),
                    copy_write(),
                    BufferUsage::COPY_DST,
                )?;
                pass.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        let execution = CompiledExecution::compile(&graph)?;
        if execution.logical().passes() != [upload_pass, raster, readback] {
            return Err(Error::InvalidState);
        }
        Ok(Self {
            execution,
            uniform_range,
            upload: upload_pass,
            raster,
            readback,
            source_slot: upload.into(),
            uniform_slot: uniform.into(),
            readback_slot: readback_buffer.into(),
            texture_slot: source.into(),
            target_slot: target.into(),
        })
    }

    fn bindings<'a, B: ResourceBackend>(
        &'a self,
        device: DeviceId,
        buffers: &'a ReadSetRasterBuffers<B>,
    ) -> std::result::Result<FrameBindings<'a, B>, BindingError> {
        FrameBindings::new(
            &self.execution,
            device,
            vec![
                BoundBuffer::new(
                    self.source_slot,
                    buffers.upload_instance,
                    &buffers.upload,
                    &buffers.upload_desc,
                    host(),
                ),
                BoundBuffer::new(
                    self.uniform_slot,
                    buffers.uniform_instance,
                    &buffers.uniform,
                    &buffers.uniform_desc,
                    empty(),
                ),
                BoundBuffer::new(
                    self.readback_slot,
                    buffers.readback_instance,
                    &buffers.readback,
                    &buffers.readback_desc,
                    empty(),
                ),
            ],
            vec![
                BoundTexture::new(
                    self.texture_slot,
                    buffers.source_instance,
                    &buffers.source,
                    &buffers.source_desc,
                    buffers.source_initial,
                    buffers.source_layout,
                ),
                BoundTexture::new(
                    self.target_slot,
                    buffers.target_instance,
                    &buffers.target,
                    &buffers.target_desc,
                    buffers.target_initial,
                    buffers.target_layout,
                ),
            ],
        )
    }

    pub(crate) fn validate_bindings<B: ResourceBackend>(
        &self,
        device: DeviceId,
        buffers: &ReadSetRasterBuffers<B>,
    ) -> std::result::Result<(), BindingError> {
        self.bindings(device, buffers).map(|_| ())
    }
}

/// The exact bind group and dynamic range admitted for one retained frame.
pub(crate) struct ReadSetRasterBindings<B: Backend> {
    plan: PlanId,
    uniform: super::ResourceInstanceId,
    source: super::ResourceInstanceId,
    uniform_range: Range<u64>,
    group: BindGroup<B>,
    _layout: BindGroupLayout<B>,
}

/// RHI topology reusable only after the candidate Graph contract is checked.
pub(crate) struct ReadSetRasterPlan<B: Backend> {
    device: Device<B>,
    id: PlanId,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    logical: CompiledGraph,
}

impl<B: Backend> ReadSetRasterPlan<B> {
    pub(crate) fn compile(
        device: &Device<B>,
        id: PlanId,
        frame: &ReadSetRasterFrame,
    ) -> Result<Self> {
        if id.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let lane = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .map(|lane| lane.id)
            .ok_or(Error::WrongLane)?;
        Ok(Self {
            device: device.clone(),
            id,
            lane,
            submission: frame.execution.submission(device, id, lane)?,
            logical: frame.execution.logical().clone(),
        })
    }
    pub(crate) fn validate_frame(&self, frame: &ReadSetRasterFrame) -> Result<()> {
        if self
            .logical
            .recompile_reason_for_dynamic_buffers(frame.execution.logical())
            .is_some()
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
    pub(crate) fn id(&self) -> PlanId {
        self.id
    }
    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }
    pub(crate) fn row_pitch(&self) -> u32 {
        ROW_PITCH
    }
    pub(crate) fn upload_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 0,
        }
    }
    pub(crate) fn raster_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 1,
        }
    }
    pub(crate) fn readback_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 2,
        }
    }

    pub(crate) fn bindings(
        &self,
        layout: BindGroupLayout<B>,
        buffers: &ReadSetRasterBuffers<B>,
        frame: &ReadSetRasterFrame,
    ) -> Result<ReadSetRasterBindings<B>>
    where
        B: ResourceBackend,
    {
        if layout.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        self.validate_frame(frame)?;
        frame
            .validate_bindings(self.id.device, buffers)
            .map_err(binding_error)?;
        let entries = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &buffers.uniform,
                    range: 0..16,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&buffers.source_view),
            },
        ];
        let group = self.device.create_bind_group(&BindGroupDesc {
            label: Some("fluxel renderer read-set"),
            layout: &layout,
            entries: &entries,
        })?;
        Ok(ReadSetRasterBindings {
            plan: self.id,
            uniform: buffers.uniform_instance,
            source: buffers.source_instance,
            uniform_range: frame.uniform_range.clone(),
            group,
            _layout: layout,
        })
    }

    pub(crate) fn record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        buffers: &ReadSetRasterBuffers<B>,
        frame: &ReadSetRasterFrame,
    ) -> Result<()>
    where
        B: ResourceBackend + fluxel_rhi::backend::CommandBackend,
    {
        self.validate_frame(frame)?;
        let bindings = frame
            .bindings(self.id.device, buffers)
            .map_err(binding_error)?;
        let pass = bindings.pass(frame.upload)?;
        let source = pass.buffer(
            frame.source_slot,
            0..16,
            AccessKind::Read,
            bc(copy_read(), BufferUsage::COPY_SRC),
        )?;
        pass.buffer(
            frame.source_slot,
            TEXEL_OFFSET..TEXEL_OFFSET + 4,
            AccessKind::Read,
            bc(copy_read(), BufferUsage::COPY_SRC),
        )?;
        let uniform = pass.buffer(
            frame.uniform_slot,
            frame.uniform_range.clone(),
            AccessKind::Write,
            bc(copy_write(), BufferUsage::COPY_DST),
        )?;
        let texture = pass.texture(
            frame.texture_slot,
            whole(),
            AccessKind::Write,
            tc(
                copy_tex_write(),
                ImageLayout::CopyDst,
                TextureUsage::COPY_DST,
            ),
        )?;
        pass.transition(encoder)?;
        encoder.copy_buffer(source, 0, uniform, frame.uniform_range.start, 16)?;
        encoder.buffer_to_texture(
            source,
            TexelCopyBufferLayout {
                offset: TEXEL_OFFSET,
                bytes_per_row: ROW_PITCH,
                rows_per_image: 1,
            },
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
        )
    }

    pub(crate) fn record_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        buffers: &ReadSetRasterBuffers<B>,
        pipeline: &RasterPipeline<B>,
        bindings: &ReadSetRasterBindings<B>,
        frame: &ReadSetRasterFrame,
    ) -> Result<()>
    where
        B: RasterBackend + ResourceBackend,
    {
        if pipeline.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if bindings.plan != self.id
            || bindings.uniform != buffers.uniform_instance
            || bindings.source != buffers.source_instance
            || bindings.uniform_range != frame.uniform_range
        {
            return Err(Error::InvalidState);
        }
        self.validate_frame(frame)?;
        let resources = frame
            .bindings(self.id.device, buffers)
            .map_err(binding_error)?;
        let pass = resources.pass(frame.raster)?;
        pass.buffer(
            frame.uniform_slot,
            frame.uniform_range.clone(),
            AccessKind::Read,
            bc(shader_read(), BufferUsage::UNIFORM),
        )?;
        pass.texture(
            frame.texture_slot,
            whole(),
            AccessKind::Read,
            tc(
                shader_tex_read(),
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            ),
        )?;
        pass.texture(
            frame.target_slot,
            whole(),
            AccessKind::Write,
            tc(
                color_write(),
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            ),
        )?;
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&buffers.target_view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.; 4])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel read-set fixture"),
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        raster.viewport(Viewport {
            x: 0.,
            y: 0.,
            width: 8.,
            height: 8.,
            min_depth: 0.,
            max_depth: 1.,
        })?;
        raster.scissor(Rect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        })?;
        raster.set_pipeline(pipeline)?;
        raster.set_bind_group(
            0,
            &bindings.group,
            &[u32::try_from(frame.uniform_range.start).map_err(|_| Error::InvalidRange)?],
        )?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }

    pub(crate) fn record_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        buffers: &ReadSetRasterBuffers<B>,
        frame: &ReadSetRasterFrame,
    ) -> Result<ReadbackTicket<B>>
    where
        B: ReadbackBackend + ResourceBackend,
    {
        self.validate_frame(frame)?;
        let resources = frame
            .bindings(self.id.device, buffers)
            .map_err(binding_error)?;
        let pass = resources.pass(frame.readback)?;
        let texture = pass.texture(
            frame.target_slot,
            whole(),
            AccessKind::Read,
            tc(
                copy_tex_read(),
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            ),
        )?;
        let readback = pass.buffer(
            frame.readback_slot,
            0..u64::from(ROW_PITCH) * 8,
            AccessKind::Write,
            bc(copy_write(), BufferUsage::COPY_DST),
        )?;
        pass.transition(encoder)?;
        encoder.texture_to_buffer(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: ROW_PITCH,
                rows_per_image: 8,
            },
        )?;
        encoder.readback_buffer(readback, 0..u64::from(ROW_PITCH) * 8)
    }
}

impl<B: ExecutionBackend> ReadSetRasterPlan<B> {
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>> {
        let facts = outcome.facts();
        let expected = [self.upload_step(), self.raster_step(), self.readback_step()];
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(&expected) {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points.iter().find(|point| {
                    point.point.device() == self.id.device
                        && point
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == self.readback_step())
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }
}

fn whole() -> TextureRange {
    TextureRange::new(TextureAspects::COLOR, 0..1, 0..1)
}
fn tex_desc(label: &str, extent: [u32; 2], usage: TextureUsage) -> fluxel_rhi::api::TextureDesc {
    fluxel_rhi::api::TextureDesc {
        label: Some(label.into()),
        dimension: fluxel_rhi::api::TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: fluxel_rhi::api::TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage,
        view_formats: vec![],
    }
}
fn empty() -> BufferUses {
    BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}
fn host() -> BufferUses {
    BufferUses {
        stages: StageMask::HOST,
        accesses: AccessMask::HOST_WRITE,
    }
}
fn copy_read() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}
fn copy_write() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}
fn shader_read() -> BufferUses {
    BufferUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::UNIFORM_READ,
    }
}
fn copy_tex_write() -> TextureUses {
    TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}
fn shader_tex_read() -> TextureUses {
    TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::SHADER_READ,
    }
}
fn color_write() -> TextureUses {
    TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    }
}
fn copy_tex_read() -> TextureUses {
    TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
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

#[cfg(test)]
#[path = "../tests/graph/read_set_raster.rs"]
mod tests;
