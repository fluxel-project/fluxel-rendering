//! Graph declaration and recording for the bounded Forward consumer.

use super::validate_resident_texture_devices;

mod access_contracts;
mod capture_binding_collection;
mod capture_bindings;
mod capture_declaration;
mod capture_depth_alias;
mod capture_oit_declaration;
use capture_oit_declaration::CaptureOitDeclaration;
mod capture_oit_bindings;
mod capture_oit_plan;
mod capture_oit_record;
mod capture_oit_stage_record;
mod capture_oit_surface_declaration;
mod capture_oit_terminal_record;
pub(crate) use capture_oit_plan::CaptureOitPlan;
mod capture_plan;
mod capture_receiver;
mod capture_record;
mod capture_specular_bindings;
mod capture_surface_declaration;
mod capture_surface_plan;
mod capture_surface_record;
mod capture_terminal;
mod capture_upload;
mod declaration;
pub(in crate::graph) mod hdr_output;
mod hdr_record;
mod offscreen_bindings;
mod offscreen_declaration;
mod offscreen_plan;
mod offscreen_prefix;
mod offscreen_record;
mod oit_declaration;
mod oit_draw_reads;
mod oit_read_context;
pub(crate) use oit_read_context::ForwardOitReadContext;
mod oit_plan;
mod oit_record;
mod oit_stages;
pub(in crate::graph) use capture_depth_alias::{same_buffer_owner_desc, same_texture_owner_desc};
mod overlay;
pub(in crate::graph) use capture_declaration::SharedResidentImports;
pub(in crate::graph) use capture_receiver::{
    ForwardCaptureReceiver, ForwardReceiverTerminal,
    declare_forward_capture_namespace_with_contribution, declare_receiver_after_deferred,
    declare_receiver_terminal,
};
pub(in crate::graph) use overlay::{
    OverlaySlots, declare_overlay_prefix, declare_overlay_prefix_with_shared,
};
mod phase;
mod read_scope;
mod shared_record;
mod spec;
mod surface_plan;
mod surface_record;
mod validation;
use access_contracts::*;
pub(crate) use phase::{ClearForwardPass, ForwardPhaseDriver, OpaqueForwardPass};
pub(crate) use read_scope::{DrawReads, ForwardReadScope, ForwardReadState};
pub(crate) use spec::{
    BufferGraphFact, ForwardDrawShape, ForwardGraphDraw, ForwardGraphSpec,
    ForwardGraphTextureUpload, ForwardGraphUniformUpload, ForwardGraphVertexUpload, ForwardShape,
    TextureGraphFact,
};
use spec::{SurfaceForwardShape, forward_shape, surface_forward_shape};
pub(crate) use validation::draw_accesses as forward_draw_accesses;
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
use std::{marker::PhantomData, num::NonZeroU32, ops::Range, sync::Arc};

use crate::residency::{ForwardFrame, OitOwners, OitTextureOwner, SurfaceForwardFrame};
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
use offscreen_declaration::{
    HdrForwardTerminal, OffscreenForwardDeclaration, declare_offscreen_hdr_with,
    declare_offscreen_with,
};

/// Exact offscreen resources and stable draw partition for one OIT frame.
/// This is Graph-private: the renderer retains all actual native programs and
/// sampled bind groups through completion separately.
#[derive(Clone, Copy)]
pub(crate) struct ForwardOitInput<'a, B: ResourceBackend> {
    pub(crate) owners: &'a OitOwners<B>,
    pub(crate) output: &'a OitTextureOwner<B>,
    pub(crate) opaque_count: usize,
    pub(crate) peel_count: usize,
    pub(crate) rounds: NonZeroU32,
}

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
    multisample_color_slot: Option<TextureSlot>,
    readback_slot: BufferSlot,
    draw_count: usize,
    stream_count: usize,
    texture_count: usize,
    extent: [u32; 2],
    shape: ForwardShape,
    uniform_alignment: u64,
    topology: FrameTopology,
    oit: Option<oit_plan::ForwardOitPlan>,
    hdr_terminal: Option<HdrForwardTerminal>,
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
