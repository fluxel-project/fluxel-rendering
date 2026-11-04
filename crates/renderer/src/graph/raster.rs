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
mod offscreen;
mod surface;

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
