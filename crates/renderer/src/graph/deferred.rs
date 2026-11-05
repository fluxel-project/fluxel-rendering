//! Private two-pass Deferred graph for the first bounded offscreen probe.

mod declaration;
mod diagnostic;
mod phase;
mod plan;
mod record;
mod record_view;
mod surface;
mod surface_record;
mod surface_validate;

use fluxel_rhi::{
    api::{
        AccessMask, AttachmentView, BufferUsage, BufferUses, ClearColor, ColorAttachment,
        CommandEncoder, CompletionPoint, DepthStencilAttachment, Device, Error, ExecutionStatus,
        ImageLayout, LaneClass, LaneId, LoadOp, PlanId, RasterPassDesc, Rect, Result, StageMask,
        StepId, StoreOp, SubmissionPlan, TexelCopyBufferLayout, TextureAspects, TextureFormat,
        TextureRegion, TextureUsage, TextureUses, Viewport,
    },
    backend::{
        CommandBackend, ExecutionBackend, ExecutionMode, ExecutionOutcome, RasterBackend,
        ReadbackBackend, ResourceBackend,
    },
};
use std::ops::Range;

use crate::{
    preparation::{GeometryBufferOwner, GeometryDraw, GeometryStreamBinding, PreparedGeometry},
    residency::{DeferredDraw, DeferredFrame, DeferredSurfaceFrame, ForwardTextureUpload},
};

use super::{
    binding::{BufferSlot, TextureSlot},
    definition::{
        AccessKind, BufferAccessContract, GraphBuilder, GraphError, GraphPass,
        TextureAccessContract, TextureRange,
    },
    execution::{BoundBuffer, BoundTexture, CompiledExecution, FrameBindings, binding_error},
};

type Outcome<B> = <<B as ExecutionBackend>::Mode as ExecutionMode<B>>::Outcome;

/// A compiled four-pass graph: upload, G-buffer geometry, lighting, readback.
pub(crate) struct DeferredPlan<B: ResourceBackend> {
    device: Device<B>,
    id: PlanId,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    execution: CompiledExecution,
    declaration: GraphBuilder,
    upload_pass: GraphPass,
    deformation_pass: Option<GraphPass>,
    geometry_pass: GraphPass,
    lighting_pass: GraphPass,
    output: DeferredOutputPlan,
    resident_upload_slot: BufferSlot,
    frame_upload_slot: BufferSlot,
    uniform_slot: BufferSlot,
    stream_slots: Vec<BufferSlot>,
    texture_slot: TextureSlot,
    gbuffer_slots: [TextureSlot; 3],
    target_slot: Option<TextureSlot>,
    readback_slot: Option<BufferSlot>,
    deformation_slots: Option<[BufferSlot; 4]>,
    extent: [u32; 2],
    uniform_alignment: u64,
    transform_bytes: u64,
}

/// Fresh private Graph declaration used only to select a cached plan.
///
/// It contains no backend owner identity; the selected plan separately checks
/// every concrete owner immediately before recording.
pub(crate) struct DeferredDeclaration {
    graph: GraphBuilder,
    lane: LaneId,
    upload_pass: GraphPass,
    deformation_pass: Option<GraphPass>,
    geometry_pass: GraphPass,
    lighting_pass: GraphPass,
    output: DeferredOutputDeclaration,
    resident_upload_slot: BufferSlot,
    frame_upload_slot: BufferSlot,
    uniform_slot: BufferSlot,
    stream_slots: Vec<BufferSlot>,
    texture_slot: TextureSlot,
    gbuffer_slots: [TextureSlot; 3],
    target_slot: Option<TextureSlot>,
    readback_slot: Option<BufferSlot>,
    deformation_slots: Option<[BufferSlot; 4]>,
    extent: [u32; 2],
    uniform_alignment: u64,
    transform_bytes: u64,
}

/// The physical final-output owner is deliberately kept out of the common
/// Deferred body.  A surface lease has no persistent Graph binding.
#[derive(Clone, Copy)]
enum DeferredOutputPlan {
    Offscreen { readback_pass: GraphPass },
    Surface { present_step: StepId },
}

#[derive(Clone, Copy)]
enum DeferredOutputDeclaration {
    Offscreen { readback_pass: GraphPass },
    Surface,
}

use phase::DeferredDeformationGraph;
pub(crate) use phase::{
    DeferredDeformationPass, DeferredGeometryPass, DeferredLightingPass, DeferredPhaseDriver,
};

fn whole_color() -> TextureRange {
    TextureRange::new(TextureAspects::COLOR, 0..1, 0..1)
}

fn whole_depth() -> TextureRange {
    TextureRange::new(TextureAspects::DEPTH, 0..1, 0..1)
}

fn buffer_contract(uses: BufferUses, usage: BufferUsage) -> BufferAccessContract {
    BufferAccessContract {
        uses,
        descriptor_usage: usage,
    }
}

fn texture_contract(
    uses: TextureUses,
    layout: ImageLayout,
    usage: TextureUsage,
) -> TextureAccessContract {
    TextureAccessContract {
        uses,
        layout,
        descriptor_usage: usage,
    }
}

fn copy_read() -> BufferAccessContract {
    buffer_contract(
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        },
        BufferUsage::COPY_SRC,
    )
}

fn copy_write() -> BufferAccessContract {
    buffer_contract(
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        },
        BufferUsage::COPY_DST,
    )
}

fn vertex_read() -> BufferAccessContract {
    buffer_contract(
        BufferUses {
            stages: StageMask::VERTEX_INPUT,
            accesses: AccessMask::VERTEX_READ,
        },
        BufferUsage::VERTEX,
    )
}

fn uniform_read() -> BufferAccessContract {
    buffer_contract(
        BufferUses {
            stages: StageMask::VERTEX | StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        BufferUsage::UNIFORM,
    )
}

fn compute_read() -> BufferAccessContract {
    buffer_contract(
        BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_READ,
        },
        BufferUsage::STORAGE,
    )
}

fn compute_parameter_read() -> BufferAccessContract {
    buffer_contract(
        BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_READ,
        },
        BufferUsage::UNIFORM,
    )
}

fn compute_write() -> BufferAccessContract {
    // The selected storage binding is RHI `StorageAccess::ReadWrite`; its
    // transition contract remains read/write even though this WGSL body stores.
    buffer_contract(
        BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
        },
        BufferUsage::STORAGE,
    )
}

fn texture_copy_write() -> TextureAccessContract {
    texture_contract(
        TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        },
        ImageLayout::CopyDst,
        TextureUsage::COPY_DST,
    )
}

fn sampled_read() -> TextureAccessContract {
    texture_contract(
        TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        ImageLayout::ShaderReadOnly,
        TextureUsage::TEXTURE_BINDING,
    )
}

fn color_write() -> TextureAccessContract {
    texture_contract(
        TextureUses {
            stages: StageMask::COLOR_OUTPUT,
            accesses: AccessMask::COLOR_WRITE,
        },
        ImageLayout::ColorAttachment,
        TextureUsage::RENDER_ATTACHMENT,
    )
}

fn depth_write() -> TextureAccessContract {
    texture_contract(
        TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::DEPTH_WRITE,
        },
        ImageLayout::DepthStencilAttachment,
        TextureUsage::RENDER_ATTACHMENT,
    )
}

fn texture_copy_read() -> TextureAccessContract {
    texture_contract(
        TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        },
        ImageLayout::CopySrc,
        TextureUsage::COPY_SRC,
    )
}

fn texture_copy_end(copy: &ForwardTextureUpload) -> Result<u64> {
    copy.source_offset
        .checked_add(
            u64::from(copy.bytes_per_row)
                .checked_mul(u64::from(copy.rows_per_image))
                .ok_or(Error::InvalidRange)?,
        )
        .ok_or(Error::InvalidRange)
}

fn draw_accesses<B: ResourceBackend>(
    frame: &DeferredFrame<B>,
    draw: &DeferredDraw<B>,
) -> Result<Vec<Option<Range<u64>>>> {
    draw_accesses_parts(&frame.resident, frame.deformation.as_ref(), draw)
}

fn draw_accesses_parts<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    deformation: Option<&crate::residency::DeferredDeformation<B>>,
    draw: &DeferredDraw<B>,
) -> Result<Vec<Option<Range<u64>>>> {
    if draw.vertex_layouts.len() != draw.vertex_ranges.len()
        || (deformation.is_none() && draw.vertex_layouts.len() != resident.streams.len())
        || draw.vertices.start >= draw.vertices.end
    {
        return Err(Error::InvalidRange);
    }
    let bindings = if let Some(deformation) = &deformation {
        if draw.vertex_layouts.len() != 1 {
            return Err(Error::InvalidRange);
        }
        vec![GeometryStreamBinding {
            owner: GeometryBufferOwner::new(
                resident.device,
                deformation.product.desc.clone(),
                deformation.product.incoming,
            ),
            range: draw.vertex_ranges[0].clone(),
        }]
    } else {
        resident
            .streams
            .iter()
            .zip(&draw.vertex_ranges)
            .map(|(owner, range)| GeometryStreamBinding {
                owner: GeometryBufferOwner::new(
                    resident.device,
                    owner.desc.clone(),
                    owner.incoming,
                ),
                range: range.clone(),
            })
            .collect::<Vec<_>>()
    };
    PreparedGeometry {
        streams: draw.vertex_layouts.clone(),
    }
    .bind(
        resident.device,
        &bindings,
        GeometryDraw {
            vertices: draw.vertices.clone(),
            instances: 0..1,
        },
    )
    .map(|prepared| prepared.accessed_ranges().to_vec())
    .map_err(|_| Error::InvalidRange)
}

#[cfg(test)]
#[path = "../tests/graph/deferred.rs"]
mod tests;
