//! Private two-pass Deferred graph for the first bounded offscreen probe.

use super::validate_resident_texture_devices;

mod capture_aliases;
mod capture_bindings;
mod capture_declaration;
mod capture_namespace;
mod capture_owner_validation;
mod capture_plan;
mod capture_producer;
mod capture_record;
mod declaration;
mod declaration_validation;
mod diagnostic;
mod forward_capture_bindings;
mod forward_capture_declaration;
mod forward_capture_owners;
mod forward_capture_plan;
mod forward_capture_record;
mod hybrid_record;
mod imports;
mod oit_bindings;
mod oit_declaration;
mod oit_plan;
mod oit_record;
mod phase;
mod plan;
mod plan_bindings;
mod prefix;
mod raster;
mod record;
mod record_lighting;
mod record_view;
mod reverse_capture_bindings;
mod reverse_capture_declaration;
mod reverse_capture_owners;
mod reverse_capture_plan;
mod reverse_capture_record;
mod shared_imports;
mod surface;
mod surface_record;
mod surface_validate;
mod terminal;
mod transparent_bindings;
use oit_declaration::DeferredOitDeclaration;
pub(crate) use oit_plan::DeferredOitPlan;
mod transparent_declaration;
mod transparent_plan;
mod transparent_record;
pub(crate) use capture_plan::DeferredCapturePlan;
pub(crate) use capture_producer::DeferredCaptureOverlay;
pub(crate) use forward_capture_plan::DeferredForwardCapturePlan;
pub(crate) use reverse_capture_plan::ForwardDeferredCapturePlan;
use transparent_declaration::DeferredTransparentDeclaration;
pub(crate) use transparent_plan::DeferredTransparentPlan;
mod upload_record;
mod uploads;
mod validation;

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
    execution::{
        BoundBuffer, BoundSurfaceTexture, BoundTexture, CompiledExecution, FrameBindings,
        binding_error,
    },
};

type Outcome<B> = <<B as ExecutionBackend>::Mode as ExecutionMode<B>>::Outcome;

/// A compiled Deferred graph: upload, G-buffer geometry, lighting, composite,
/// and (for offscreen work) readback.
pub(crate) struct DeferredPlan<B: ResourceBackend> {
    device: Device<B>,
    id: PlanId,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    execution: CompiledExecution,
    declaration: GraphBuilder,
    upload_pass: GraphPass,
    mip_generation: Vec<super::mip_generation::MipGenerationPass>,
    deformation_pass: Option<GraphPass>,
    geometry_pass: GraphPass,
    lighting_pass: GraphPass,
    composition_passes: Vec<GraphPass>,
    composite_pass: Option<GraphPass>,
    output: DeferredOutputPlan,
    resident_upload_slot: BufferSlot,
    frame_upload_slot: BufferSlot,
    uniform_slot: BufferSlot,
    stream_slots: Vec<BufferSlot>,
    texture_slots: Vec<TextureSlot>,
    environment_upload_slot: Option<BufferSlot>,
    environment_slot: Option<TextureSlot>,
    specular_upload_slot: Option<BufferSlot>,
    specular_view_uniform_slot: Option<BufferSlot>,
    specular_radiance_slot: Option<TextureSlot>,
    specular_brdf_lut_slot: Option<TextureSlot>,
    gbuffer_slots: [TextureSlot; 3],
    effects_slot: Option<TextureSlot>,
    scene_color_slot: Option<TextureSlot>,
    surface_target_slot: Option<TextureSlot>,
    target_slot: Option<TextureSlot>,
    readback_slot: Option<BufferSlot>,
    deformation_slots: Option<[BufferSlot; 4]>,
    extent: [u32; 2],
    uniform_alignment: u64,
    transform_bytes: Vec<u64>,
}

/// Fresh private Graph declaration used only to select a cached plan.
///
/// It contains no backend owner identity; the selected plan separately checks
/// every concrete owner immediately before recording.
#[derive(Clone)]
pub(crate) struct DeferredDeclaration {
    graph: GraphBuilder,
    lane: LaneId,
    upload_pass: GraphPass,
    mip_generation: Vec<super::mip_generation::MipGenerationPass>,
    deformation_pass: Option<GraphPass>,
    geometry_pass: GraphPass,
    lighting_pass: GraphPass,
    composite_pass: Option<GraphPass>,
    output: DeferredOutputDeclaration,
    resident_upload_slot: BufferSlot,
    frame_upload_slot: BufferSlot,
    uniform_slot: BufferSlot,
    stream_slots: Vec<BufferSlot>,
    texture_slots: Vec<TextureSlot>,
    environment_upload_slot: Option<BufferSlot>,
    environment_slot: Option<TextureSlot>,
    specular_upload_slot: Option<BufferSlot>,
    specular_view_uniform_slot: Option<BufferSlot>,
    specular_radiance_slot: Option<TextureSlot>,
    specular_brdf_lut_slot: Option<TextureSlot>,
    gbuffer_slots: [TextureSlot; 3],
    effects_slot: Option<TextureSlot>,
    scene_color_slot: Option<TextureSlot>,
    surface_target_slot: Option<TextureSlot>,
    target_slot: Option<TextureSlot>,
    readback_slot: Option<BufferSlot>,
    deformation_slots: Option<[BufferSlot; 4]>,
    extent: [u32; 2],
    uniform_alignment: u64,
    transform_bytes: Vec<u64>,
}

/// The physical final-output owner is deliberately kept out of the common
/// Deferred body.  A surface lease has no persistent Graph binding.
#[derive(Clone, Copy)]
enum DeferredOutputPlan {
    /// A capture producer stops at its lighting product. It has no composite,
    /// readback, or presentation operation of its own.
    Capture,
    Offscreen {
        readback_pass: GraphPass,
    },
    Surface {
        present_step: StepId,
    },
}

#[derive(Clone, Copy)]
enum DeferredOutputDeclaration {
    Capture,
    Offscreen { readback_pass: GraphPass },
    Surface,
}

use phase::DeferredDeformationGraph;
pub(crate) use phase::{
    DeferredDeformationPass, DeferredGeometryPass, DeferredLightingPass, DeferredPhaseDriver,
};
pub(super) use raster::{configure_raster, render_area};

fn whole_color() -> TextureRange {
    TextureRange::new(TextureAspects::COLOR, 0..1, 0..1)
}

fn one_color_mip(mip: u32) -> TextureRange {
    TextureRange::new(TextureAspects::COLOR, mip..mip + 1, 0..1)
}

fn all_color_mips(mips: u32) -> TextureRange {
    TextureRange::new(TextureAspects::COLOR, 0..mips, 0..1)
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

fn lighting_uniform_read() -> BufferAccessContract {
    buffer_contract(
        BufferUses {
            stages: StageMask::FRAGMENT,
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
    if draw.vertex_layouts.is_empty()
        || draw.vertex_layouts.len() != draw.vertex_ranges.len()
        || draw.vertex_layouts.len() != draw.vertex_streams.len()
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
        draw.vertex_streams
            .iter()
            .zip(&draw.vertex_ranges)
            .map(|(&index, range)| {
                let owner = resident.streams.get(index).ok_or(Error::InvalidRange)?;
                Ok(GeometryStreamBinding {
                    owner: {
                        GeometryBufferOwner::new(
                            resident.device,
                            owner.desc.clone(),
                            owner.incoming,
                        )
                    },
                    range: range.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?
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
