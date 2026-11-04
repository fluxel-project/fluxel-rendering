//! Graph declaration and recording for the bounded Forward consumer.

mod access_contracts;
mod declaration;
mod offscreen_declaration;
mod offscreen_plan;
mod offscreen_record;
mod phase;
mod read_scope;
mod spec;
mod surface_plan;
mod surface_record;
mod validation;
use access_contracts::*;
pub(crate) use phase::{ClearForwardPass, ForwardPhaseDriver, OpaqueForwardPass};
pub(crate) use read_scope::ForwardReadScope;
use read_scope::ForwardReadState;
use spec::{forward_shape, surface_forward_shape, SurfaceForwardShape};
pub(crate) use spec::{
    BufferGraphFact, ForwardDrawShape, ForwardGraphDraw, ForwardGraphSpec,
    ForwardGraphTextureUpload, ForwardGraphUniformUpload, ForwardGraphVertexUpload, ForwardShape,
    TextureGraphFact,
};
use validation::*;

use fluxel_rhi::{
    api::{
        AccessMask, AccessScope, AttachmentView, BufferDesc, BufferUsage, BufferUses, ClearColor,
        ColorAttachment, CommandEncoder, CompletionPoint, Device, DeviceId, Error, ExecutionStatus,
        ImageLayout, LaneClass, LaneId, LoadOp, PlanId, RasterPassDesc, Rect, Result, StageMask,
        StepDesc, StepId, StepKind, StepWait, StoreOp, SubmissionPlan, SurfaceImage,
        TexelCopyBufferLayout, TextureAspects, TextureDesc, TextureFormat, TextureRegion,
        TextureUsage, TextureUses, Viewport,
    },
    backend::{
        CommandBackend, ExecutionBackend, ExecutionMode, ExecutionOutcome, RasterBackend,
        ReadbackBackend, ResourceBackend,
    },
};
use std::{marker::PhantomData, ops::Range, sync::Arc};

use crate::residency::{ForwardFrame, SurfaceForwardFrame};
use crate::{
    pipeline::{ForwardPhaseContext, PipelineError},
    preparation::{GeometryBufferOwner, GeometryDraw, GeometryStreamBinding, PreparedGeometry},
    FrameTopology,
};

use super::{
    binding::{BufferSlot, TextureSlot},
    definition::{
        AccessKind, BufferAccessContract, GraphBuilder, GraphPass, TextureAccessContract,
        TextureRange,
    },
    execution::{binding_error, BoundBuffer, BoundTexture, CompiledExecution, FrameBindings},
};
use declaration::{declare_surface_with, SurfaceForwardDeclaration};
use offscreen_declaration::{declare_offscreen_with, OffscreenForwardDeclaration};

type Outcome<B> = <<B as ExecutionBackend>::Mode as ExecutionMode<B>>::Outcome;

fn same_buffer_slots(left: &[BufferSlot], right: &[BufferSlot]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.resource == right.resource)
}

fn same_texture_slots(left: &[TextureSlot], right: &[TextureSlot]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.resource == right.resource)
}

pub(crate) enum ForwardPlanBuildError {
    Rhi(Error),
    Pipeline(PipelineError),
}

impl From<Error> for ForwardPlanBuildError {
    fn from(error: Error) -> Self {
        Self::Rhi(error)
    }
}

/// A compiled Forward graph for one concrete prepared topology.
pub(crate) struct ForwardPlan<B: ResourceBackend> {
    id: PlanId,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    execution: CompiledExecution,
    declaration: GraphBuilder,
    upload_pass: GraphPass,
    raster_pass: GraphPass,
    readback_pass: GraphPass,
    resident_upload_slot: BufferSlot,
    frame_upload_slot: BufferSlot,
    uniform_slot: BufferSlot,
    stream_slots: Vec<BufferSlot>,
    texture_slots: Vec<TextureSlot>,
    target_slot: TextureSlot,
    readback_slot: BufferSlot,
    draw_count: usize,
    stream_count: usize,
    texture_count: usize,
    extent: [u32; 2],
    shape: ForwardShape,
    uniform_alignment: u64,
    topology: FrameTopology,
}

/// Forward execution plan whose final raster attachment is an acquired surface
/// image. The private Forward graph still owns upload, draw-input and
/// completion dependency declarations; the surface lease is deliberately not
/// represented as a persistent Graph texture owner.
pub(crate) struct SurfaceForwardPlan<B: ResourceBackend> {
    id: PlanId,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    execution: CompiledExecution,
    declaration: GraphBuilder,
    upload_pass: GraphPass,
    raster_pass: GraphPass,
    present_step: StepId,
    resident_upload_slot: BufferSlot,
    frame_upload_slot: BufferSlot,
    uniform_slot: BufferSlot,
    stream_slots: Vec<BufferSlot>,
    texture_slots: Vec<TextureSlot>,
    extent: [u32; 2],
    draw_count: usize,
    stream_count: usize,
    texture_count: usize,
    color_format: TextureFormat,
    shape: SurfaceForwardShape,
    topology: FrameTopology,
    uniform_alignment: u64,
}

#[cfg(test)]
#[path = "../tests/graph/forward_surface.rs"]
mod tests;
