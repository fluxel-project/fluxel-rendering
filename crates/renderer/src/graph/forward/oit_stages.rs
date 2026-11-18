//! Material stages appended to the caller's real Graph, before its terminal.

use super::*;
use super::{oit_draw_reads::OitDrawReads, oit_plan::ForwardOitPlan};
use crate::graph::{
    definition::TextureHandle,
    oit::{DualDepthPeelInputs, OitPeelSide, declare_dual_depth_peeling_rounds},
    oit_record::OitRecordSlots,
};
use std::ops::Range;

pub(super) struct OitStageDeclaration {
    pub(super) raster_pass: GraphPass,
    pub(super) opaque_color: TextureHandle,
    pub(super) opaque_depth: TextureHandle,
    pub(super) output: TextureHandle,
    pub(super) oit: ForwardOitPlan,
}

/// Appends opaque, dual-depth-peeling and excluded draws without sealing the
/// Graph with readback or presentation. The caller retains the final texture
/// version so a following capture consumer can alias that exact product.
#[allow(clippy::too_many_arguments)]
pub(super) fn declare_stages<B, C>(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    reads: &OitDrawReads<'_>,
    input: &ForwardOitInput<'_, B>,
    dependency: GraphPass,
    target: TextureHandle,
    depth: TextureHandle,
    contribute: &mut C,
) -> std::result::Result<OitStageDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    let peel_end = input
        .opaque_count
        .checked_add(input.peel_count)
        .ok_or(Error::InvalidRange)?;
    if input.peel_count == 0 || peel_end > spec.draws.len() {
        return Err(Error::InvalidRange.into());
    }
    let mut contribution_error = None;
    let opaque = graph.add_pass("forward-opaque", |pass| {
        pass.depends_on(dependency)?;
        reads.declare_with(
            pass,
            0..input.opaque_count,
            contribute,
            &mut contribution_error,
        )?;
        let color = pass.write_texture_with_use(
            target,
            whole(),
            color_write_uses().uses,
            ImageLayout::ColorAttachment,
            TextureUsage::RENDER_ATTACHMENT,
        )?;
        let depth = pass.write_texture_with_use(
            depth,
            whole_depth(),
            depth_write_uses().uses,
            ImageLayout::DepthStencilAttachment,
            TextureUsage::RENDER_ATTACHMENT,
        )?;
        Ok((color, depth))
    });
    if let Some(error) = contribution_error {
        return Err(ForwardPlanBuildError::Pipeline(error));
    }
    let (raster_pass, (opaque_color, opaque_depth)) = opaque.map_err(|_| Error::InvalidState)?;
    let (output, oit) = declare_peeling_stages(
        graph,
        spec,
        reads,
        input,
        opaque_color,
        opaque_depth,
        input.opaque_count..peel_end,
        contribute,
    )?;
    Ok(OitStageDeclaration {
        raster_pass,
        opaque_color,
        opaque_depth,
        output,
        oit,
    })
}

/// Declares the receiver against copied color and the producer's depth owner.
#[allow(clippy::too_many_arguments)]
pub(super) fn declare_projected_receiver<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    handles: &super::capture_declaration::FrameHandles,
    dependency: GraphPass,
    background_copy: GraphPass,
    input: Option<&ForwardOitInput<'_, B>>,
) -> std::result::Result<(GraphPass, TextureHandle, Option<ForwardOitPlan>), ForwardPlanBuildError>
{
    let Some(input) = input else {
        let writes_depth = spec
            .shape
            .draws
            .iter()
            .any(|draw| draw.compatibility.writes_depth());
        let (raster, color, _) = super::capture_declaration::declare_raster(
            graph,
            spec,
            FrameTopology::SortedForward,
            dependency,
            handles,
            |context: &mut ForwardPhaseContext<'_, '_, '_, B>| {
                context.sorted(|pass| {
                    pass.depends_on_uploads()?;
                    pass.read_vertex_streams()?;
                    pass.read_frame_uniforms()?;
                    pass.read_sampled_textures()?;
                    if writes_depth {
                        pass.write_selected_depth()?;
                    } else {
                        pass.read_selected_depth()?;
                    }
                    pass.read_write_main_color()
                })
            },
        )?;
        return Ok((raster, color, None));
    };

    let peel_end = input
        .opaque_count
        .checked_add(input.peel_count)
        .ok_or(Error::InvalidRange)?;
    if input.peel_count == 0 || peel_end > spec.shape.draws.len() {
        return Err(Error::InvalidRange.into());
    }
    if spec
        .shape
        .draws
        .iter()
        .enumerate()
        .any(|(index, draw)| draw.compatibility.writes_depth() != (index < input.opaque_count))
    {
        return Err(Error::InvalidState.into());
    }
    let color = handles.target;
    let depth = handles.depth.ok_or(Error::InvalidState)?;
    let reads = OitDrawReads::from_capture_handles(handles, spec);
    let mut contribute = |context: &mut ForwardOitReadContext<'_, '_>| context.all_reads();
    let (raster, (color, depth)) = if input.opaque_count == 0 {
        (background_copy, (color, depth))
    } else {
        // Load the copied background and the producer depth. Peeling and the
        // excluded suffix must consume the resulting versions.
        let mut contribution_error = None;
        let opaque = graph.add_pass("forward-capture-refraction-opaque", |pass| {
            pass.depends_on(dependency)?;
            reads.declare_with(
                pass,
                0..input.opaque_count,
                &mut contribute,
                &mut contribution_error,
            )?;
            let color = pass.write_texture_with_use(
                color,
                whole(),
                color_read_write_uses().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            let depth = pass.write_texture_with_use(
                depth,
                whole_depth(),
                depth_read_write_uses().uses,
                ImageLayout::DepthStencilAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            Ok((color, depth))
        });
        if let Some(error) = contribution_error {
            return Err(ForwardPlanBuildError::Pipeline(error));
        }
        opaque.map_err(|_| Error::InvalidState)?
    };
    let (output, oit) = declare_peeling_stages(
        graph,
        spec,
        &reads,
        input,
        color,
        depth,
        input.opaque_count..peel_end,
        &mut contribute,
    )?;
    Ok((raster, output, Some(oit)))
}

#[allow(clippy::too_many_arguments)]
fn declare_peeling_stages<B, C>(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    reads: &OitDrawReads<'_>,
    input: &ForwardOitInput<'_, B>,
    opaque_color: TextureHandle,
    opaque_depth: TextureHandle,
    peel_draws: Range<usize>,
    contribute: &mut C,
) -> std::result::Result<(TextureHandle, ForwardOitPlan), ForwardPlanBuildError>
where
    B: ResourceBackend,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    let ping_pong = [
        import_side(graph, &input.owners.ping_pong[0])?,
        import_side(graph, &input.owners.ping_pong[1])?,
    ];
    let handles = DualDepthPeelInputs {
        ping_pong,
        back_accumulator: import_owner(graph, &input.owners.back_accumulator)?,
        opaque_color,
        opaque_depth,
        output: import_owner(graph, input.output)?,
    };
    let slots = OitRecordSlots::from(handles);
    let mut contribution_error = None;
    let product = declare_dual_depth_peeling_rounds(graph, handles, input.rounds, |_, pass| {
        reads.declare_with(
            pass,
            peel_draws.clone(),
            contribute,
            &mut contribution_error,
        )
    });
    if let Some(error) = contribution_error {
        return Err(ForwardPlanBuildError::Pipeline(error));
    }
    let product = product.map_err(|_| Error::InvalidState)?;
    let mut output = product.output;
    let fallback = if peel_draws.end < spec.draws.len() {
        contribution_error = None;
        let fallback_result = graph.add_pass("oit-excluded-sorted", |pass| {
            reads.declare_with(
                pass,
                peel_draws.end..spec.draws.len(),
                contribute,
                &mut contribution_error,
            )?;
            pass.read_texture_with_use(
                opaque_depth,
                whole_depth(),
                depth_read().uses,
                ImageLayout::DepthStencilAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            let next = pass.write_texture_with_use(
                output,
                whole(),
                color_read_write_uses().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            Ok(next)
        });
        if let Some(error) = contribution_error {
            return Err(ForwardPlanBuildError::Pipeline(error));
        }
        let (pass, next) = fallback_result.map_err(|_| Error::InvalidState)?;
        output = next;
        Some(pass)
    } else {
        None
    };
    Ok((
        output,
        ForwardOitPlan {
            product,
            slots,
            opaque_count: input.opaque_count,
            peel_count: input.peel_count,
            rounds: input.rounds,
            fallback,
            surface: None,
            hdr_readback: None,
        },
    ))
}

fn import_owner<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    owner: &OitTextureOwner<B>,
) -> Result<TextureHandle> {
    graph
        .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
        .map_err(|_| Error::InvalidState)
}

fn import_side<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    side: &crate::residency::OitPingPongOwners<B>,
) -> Result<OitPeelSide> {
    Ok(OitPeelSide {
        min_max: import_owner(graph, &side.min_max)?,
        front: import_owner(graph, &side.front)?,
        back: import_owner(graph, &side.back)?,
    })
}

fn depth_read() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::DEPTH_READ,
        },
        layout: ImageLayout::DepthStencilAttachment,
        descriptor_usage: TextureUsage::RENDER_ATTACHMENT,
    }
}
