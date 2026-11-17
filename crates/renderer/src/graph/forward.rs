//! Graph declaration and recording for the bounded Forward consumer.

use super::validate_resident_texture_devices;

mod access_contracts;
mod capture_binding_collection;
mod capture_bindings;
mod capture_declaration;
mod capture_plan;
mod capture_record;
mod capture_specular_bindings;
mod capture_surface_declaration;
mod capture_surface_plan;
mod capture_surface_record;
mod capture_upload;
mod declaration;
mod offscreen_declaration;
mod offscreen_plan;
mod offscreen_record;
mod phase;
mod read_scope;
mod shared_record;
mod spec;
mod surface_plan;
mod surface_record;
mod validation;
use access_contracts::*;
pub(crate) use phase::{ClearForwardPass, ForwardPhaseDriver, OpaqueForwardPass};
pub(crate) use read_scope::ForwardReadScope;
use read_scope::ForwardReadState;
pub(crate) use spec::{
    BufferGraphFact, ForwardDrawShape, ForwardGraphDraw, ForwardGraphSpec,
    ForwardGraphTextureUpload, ForwardGraphUniformUpload, ForwardGraphVertexUpload, ForwardShape,
    TextureGraphFact,
};
use spec::{SurfaceForwardShape, forward_shape, surface_forward_shape};
use validation::*;

use fluxel_rhi::{
    api::{
        AccessMask, AccessScope, AttachmentView, BufferDesc, BufferUsage, BufferUses, ClearColor,
        ColorAttachment, CommandEncoder, CompletionPoint, DepthStencilAttachment, Device, DeviceId,
        Error, ExecutionStatus, ImageLayout, LaneClass, LaneId, LoadOp, PlanId, RasterPassDesc,
        Rect, Result, StageMask, StepDesc, StepId, StepKind, StepWait, StoreOp, SubmissionPlan,
        SurfaceImage, TexelCopyBufferLayout, TextureAspects, TextureDesc, TextureFormat,
        TextureRegion, TextureUsage, TextureUses, Viewport,
    },
    backend::{
        CommandBackend, ExecutionBackend, ExecutionMode, ExecutionOutcome, RasterBackend,
        ReadbackBackend, ResourceBackend,
    },
};
use std::{marker::PhantomData, ops::Range, sync::Arc};

use crate::residency::{ForwardFrame, SurfaceForwardFrame};
use crate::{
    FrameTopology,
    pipeline::{ForwardPhaseContext, PipelineError},
    preparation::{GeometryBufferOwner, GeometryDraw, GeometryStreamBinding, PreparedGeometry},
};

use super::{
    binding::{BufferSlot, TextureSlot},
    definition::{
        AccessKind, BufferAccessContract, GraphBuilder, GraphPass, TextureAccessContract,
        TextureRange,
    },
    execution::{
        BoundBuffer, BoundSurfaceTexture, BoundTexture, CompiledExecution, FrameBindings,
        binding_error,
    },
};
use capture_declaration::{CaptureForwardDeclaration, CaptureFrameSlots, declare_capture_forward};
pub(crate) use capture_plan::CaptureForwardPlan;
use capture_surface_declaration::{
    CaptureForwardSurfaceDeclaration, declare_capture_surface_forward,
};
pub(crate) use capture_surface_plan::CaptureForwardSurfacePlan;
use declaration::{SurfaceForwardDeclaration, declare_surface_with};
use offscreen_declaration::{OffscreenForwardDeclaration, declare_offscreen_with};

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

fn same_mip_passes(
    left: &[super::mip_generation::MipGenerationPass],
    right: &[super::mip_generation::MipGenerationPass],
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.pass.1 == right.pass.1
                && left.texture == right.texture
                && left.source_mip == right.source_mip
                && left.destination_mip == right.destination_mip
        })
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
    mip_passes: Vec<super::mip_generation::MipGenerationPass>,
    raster_pass: GraphPass,
    readback_pass: GraphPass,
    resident_upload_slot: BufferSlot,
    frame_upload_slot: BufferSlot,
    uniform_slot: BufferSlot,
    environment_upload_slot: Option<BufferSlot>,
    specular_upload_slot: Option<BufferSlot>,
    specular_view_uniform_slot: Option<BufferSlot>,
    stream_slots: Vec<BufferSlot>,
    texture_slots: Vec<TextureSlot>,
    environment_slot: Option<TextureSlot>,
    specular_radiance_slot: Option<TextureSlot>,
    specular_brdf_lut_slot: Option<TextureSlot>,
    depth_slot: Option<TextureSlot>,
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
/// lease represented by an RHI-managed Graph texture binding.
pub(crate) struct SurfaceForwardPlan<B: ResourceBackend> {
    id: PlanId,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    execution: CompiledExecution,
    declaration: GraphBuilder,
    upload_pass: GraphPass,
    mip_passes: Vec<super::mip_generation::MipGenerationPass>,
    raster_pass: GraphPass,
    present_step: StepId,
    resident_upload_slot: BufferSlot,
    frame_upload_slot: BufferSlot,
    uniform_slot: BufferSlot,
    environment_upload_slot: Option<BufferSlot>,
    specular_upload_slot: Option<BufferSlot>,
    specular_view_uniform_slot: Option<BufferSlot>,
    stream_slots: Vec<BufferSlot>,
    texture_slots: Vec<TextureSlot>,
    environment_slot: Option<TextureSlot>,
    specular_radiance_slot: Option<TextureSlot>,
    specular_brdf_lut_slot: Option<TextureSlot>,
    depth_slot: Option<TextureSlot>,
    surface_target_slot: TextureSlot,
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
