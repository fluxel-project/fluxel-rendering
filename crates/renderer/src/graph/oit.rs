//! Logical dual-depth-peeling Graph declaration.
//!
//! This module owns only the resource versions, dependencies, and portable RHI
//! contracts for OIT. Renderer code will later provide actual raster pipelines
//! and recording; this declaration deliberately does not become a second draw
//! scheduler or a submission plan.

use std::num::NonZeroU32;

use fluxel_rhi::api::{
    AccessMask, ImageLayout, StageMask, TextureFormat, TextureUsage, TextureUses,
};

use super::definition::{
    GraphBuilder, GraphError, GraphPass, PassBuilder, TextureAccessContract, TextureHandle,
};

/// The three attachments on one side of the OIT ping-pong pair.
///
/// All handles must already belong to the Graph being declared. The caller
/// imports the allocated OIT owners once, so this Graph only tracks versions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OitPeelSide {
    /// Two extrema encoded as `Rg32Float` or the admitted `Rg16Float` fallback.
    pub(crate) min_max: TextureHandle,
    /// Premultiplied front color accumulation.
    pub(crate) front: TextureHandle,
    /// Per-layer back contribution.
    pub(crate) back: TextureHandle,
}

/// OIT textures and opaque inputs already imported into one real Graph.
///
/// `output` is deliberately distinct from every sampled input: final composite
/// cannot sample and attach-write the same texture in one pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DualDepthPeelInputs {
    pub(crate) ping_pong: [OitPeelSide; 2],
    pub(crate) back_accumulator: TextureHandle,
    /// The already-defined opaque color product copied by a shader fullscreen pass.
    pub(crate) opaque_color: TextureHandle,
    /// The already-defined opaque depth attachment, tested read-only by initial peel.
    pub(crate) opaque_depth: TextureHandle,
    /// A separate final color attachment, never sampled by this declaration.
    pub(crate) output: TextureHandle,
}

/// A material-draw declaration stage that consumes the real `PassBuilder`.
///
/// The callback must declare its real vertex/uniform/sampled-texture reads on
/// that builder; no parallel manifest is accepted or retained here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OitDrawStage {
    InitialDepth,
    Peel { iteration: u32 },
}

/// One MRT clear of a ping-pong side's three attachments.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OitClearPass {
    pub(crate) pass: GraphPass,
}

/// One complete peel iteration: clear its destination, peel, then accumulate back.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OitPeelIteration {
    pub(crate) clear: OitClearPass,
    pub(crate) peel: GraphPass,
    pub(crate) back_accumulate: GraphPass,
}

/// All declared passes. These are Graph authorities for later compilation and recording.
#[derive(Clone, Debug)]
pub(crate) struct DualDepthPeelPasses {
    pub(crate) initial_clear: [OitClearPass; 2],
    pub(crate) initialize_back_accumulator: GraphPass,
    pub(crate) initial_depth: GraphPass,
    pub(crate) iterations: Vec<OitPeelIteration>,
    pub(crate) final_composite: GraphPass,
}

/// Final logical versions and pass identities produced by a peeling declaration.
#[derive(Clone, Debug)]
pub(crate) struct DualDepthPeelGraph {
    pub(crate) passes: DualDepthPeelPasses,
    pub(crate) final_front: TextureHandle,
    pub(crate) final_back_accumulator: TextureHandle,
    pub(crate) output: TextureHandle,
}

/// Declares a nonzero number of iterations. A round is one peel operation, not
/// one resolved transparent layer, so callers must not treat this count as an
/// exact layer budget.
///
/// An error is fatal to the containing Graph declaration. The caller must
/// discard this `GraphBuilder` and abort the whole declaration rather than
/// swallowing the error and recording its partial pass sequence.
pub(crate) fn declare_dual_depth_peeling_rounds<F>(
    graph: &mut GraphBuilder,
    inputs: DualDepthPeelInputs,
    rounds: NonZeroU32,
    mut declare_draw_reads: F,
) -> Result<DualDepthPeelGraph, GraphError>
where
    F: FnMut(OitDrawStage, &mut PassBuilder<'_>) -> Result<(), GraphError>,
{
    declare_dual_depth_peeling_rounds_inner(graph, inputs, rounds, &mut declare_draw_reads)
}

fn declare_dual_depth_peeling_rounds_inner<F>(
    graph: &mut GraphBuilder,
    inputs: DualDepthPeelInputs,
    rounds: NonZeroU32,
    declare_draw_reads: &mut F,
) -> Result<DualDepthPeelGraph, GraphError>
where
    F: FnMut(OitDrawStage, &mut PassBuilder<'_>) -> Result<(), GraphError>,
{
    validate_inputs(graph, inputs)?;
    // Every round adds three actual passes. Reject an unrepresentable pass
    // count before authoring or reserving storage; this is an ID bound, not a
    // quality or scene-layer limit.
    let added_passes = rounds
        .get()
        .checked_mul(3)
        .and_then(|count| count.checked_add(5))
        .ok_or(GraphError::InvalidRange)?;
    u32::try_from(graph.passes.len())
        .ok()
        .and_then(|count| count.checked_add(added_passes))
        .ok_or(GraphError::InvalidRange)?;
    let mut iterations = Vec::new();
    iterations
        .try_reserve_exact(rounds.get() as usize)
        .map_err(|_| GraphError::InvalidRange)?;

    // Load-clear is represented by an attachment write, not a logical read of
    // an undefined owner. Raster recording later supplies the concrete clear
    // values: ping0 extrema sentinel/front/back zero; ping1 extrema (0, 1)
    // and front/back zero.
    let (initial_clear_0, mut ping_0) =
        declare_clear_side(graph, inputs.ping_pong[0], "oit-clear-ping0")?;
    let (initial_clear_1, ping_1) =
        declare_clear_side(graph, inputs.ping_pong[1], "oit-clear-ping1")?;

    let (initialize_back_accumulator, mut back_accumulator) =
        declare_back_accumulator_init(graph, inputs.opaque_color, inputs.back_accumulator)?;
    let initial_depth = graph
        .add_pass("oit-initial-depth", |pass| {
            read_texture(pass, ping_1.min_max, shader_read())?;
            read_texture(pass, inputs.opaque_depth, depth_read())?;
            declare_draw_reads(OitDrawStage::InitialDepth, pass)?;
            ping_0.min_max = write_texture(pass, ping_0.min_max, color_read_write())?;
            Ok(())
        })?
        .0;

    let mut read_id = 0usize;
    let mut ping = [ping_0, ping_1];
    for iteration in 0..rounds.get() {
        let write_id = 1 - read_id;
        let (clear, cleared) = declare_clear_side(
            graph,
            ping[write_id],
            if write_id == 0 {
                "oit-clear-ping0"
            } else {
                "oit-clear-ping1"
            },
        )?;
        ping[write_id] = cleared;
        let read_side = ping[read_id];
        let write_side = ping[write_id];
        let (peel, next_side) = graph.add_pass("oit-peel", |pass| {
            read_texture(pass, read_side.min_max, shader_read())?;
            read_texture(pass, read_side.front, shader_read())?;
            declare_draw_reads(OitDrawStage::Peel { iteration }, pass)?;
            Ok(OitPeelSide {
                // MAX blending reads each attachment's clear/current value.
                min_max: write_texture(pass, write_side.min_max, color_read_write())?,
                front: write_texture(pass, write_side.front, color_read_write())?,
                back: write_texture(pass, write_side.back, color_read_write())?,
            })
        })?;
        ping[write_id] = next_side;
        let back = next_side.back;
        let (back_accumulate, next_accumulator) =
            graph.add_pass("oit-back-accumulate", |pass| {
                read_texture(pass, back, shader_read())?;
                // The RHI operation is attachment read/write, rather than sampling
                // `back_accumulator` while writing it in the same pass.
                write_texture(pass, back_accumulator, color_read_write())
            })?;
        back_accumulator = next_accumulator;
        iterations.push(OitPeelIteration {
            clear,
            peel,
            back_accumulate,
        });
        read_id = write_id;
    }

    let final_front = ping[read_id].front;
    let (final_composite, output) = graph.add_pass("oit-final-composite", |pass| {
        read_texture(pass, final_front, shader_read())?;
        read_texture(pass, back_accumulator, shader_read())?;
        write_texture(pass, inputs.output, color_write())
    })?;
    Ok(DualDepthPeelGraph {
        passes: DualDepthPeelPasses {
            initial_clear: [initial_clear_0, initial_clear_1],
            initialize_back_accumulator,
            initial_depth,
            iterations,
            final_composite,
        },
        final_front,
        final_back_accumulator: back_accumulator,
        output,
    })
}

fn declare_clear_side(
    graph: &mut GraphBuilder,
    side: OitPeelSide,
    label: &str,
) -> Result<(OitClearPass, OitPeelSide), GraphError> {
    let (pass, side) = graph.add_pass(label, |pass| {
        Ok(OitPeelSide {
            min_max: write_texture(pass, side.min_max, color_write())?,
            front: write_texture(pass, side.front, color_write())?,
            back: write_texture(pass, side.back, color_write())?,
        })
    })?;
    Ok((OitClearPass { pass }, side))
}

fn declare_back_accumulator_init(
    graph: &mut GraphBuilder,
    opaque_color: TextureHandle,
    accumulator: TextureHandle,
) -> Result<(GraphPass, TextureHandle), GraphError> {
    graph.add_pass("oit-init-back-accumulator", |pass| {
        read_texture(pass, opaque_color, shader_read())?;
        write_texture(pass, accumulator, color_write())
    })
}

fn read_texture(
    pass: &mut PassBuilder<'_>,
    texture: TextureHandle,
    contract: TextureAccessContract,
) -> Result<(), GraphError> {
    let range = pass.full_texture_range(texture)?;
    pass.read_texture_with_use(
        texture,
        range,
        contract.uses,
        contract.layout,
        contract.descriptor_usage,
    )
}

fn write_texture(
    pass: &mut PassBuilder<'_>,
    texture: TextureHandle,
    contract: TextureAccessContract,
) -> Result<TextureHandle, GraphError> {
    let range = pass.full_texture_range(texture)?;
    pass.write_texture_with_use(
        texture,
        range,
        contract.uses,
        contract.layout,
        contract.descriptor_usage,
    )
}

fn validate_inputs(graph: &GraphBuilder, inputs: DualDepthPeelInputs) -> Result<(), GraphError> {
    let all = [
        inputs.ping_pong[0].min_max,
        inputs.ping_pong[0].front,
        inputs.ping_pong[0].back,
        inputs.ping_pong[1].min_max,
        inputs.ping_pong[1].front,
        inputs.ping_pong[1].back,
        inputs.back_accumulator,
        inputs.opaque_color,
        inputs.opaque_depth,
        inputs.output,
    ];
    if all.iter().any(|handle| handle.graph != graph.graph) {
        return Err(GraphError::UnknownResource);
    }
    if all.iter().enumerate().any(|(index, handle)| {
        all[..index]
            .iter()
            .any(|prior| prior.resource == handle.resource)
    }) {
        return Err(GraphError::InvalidRange);
    }
    if texture_desc(graph, inputs.ping_pong[0].min_max)?.format
        != texture_desc(graph, inputs.ping_pong[1].min_max)?.format
    {
        // Both directions must use the same fragment-depth representation.
        return Err(GraphError::InvalidRange);
    }
    for side in inputs.ping_pong {
        validate_target(graph, side.min_max, true)?;
        validate_target(graph, side.front, false)?;
        validate_target(graph, side.back, false)?;
    }
    validate_target(graph, inputs.back_accumulator, false)?;
    validate_sampled_color(graph, inputs.opaque_color)?;
    validate_opaque_depth(graph, inputs.opaque_depth)?;
    validate_final_output(graph, inputs.output)?;
    let extent = texture_desc(graph, inputs.ping_pong[0].min_max)?.extent;
    if all
        .iter()
        .any(|texture| texture_desc(graph, *texture).map_or(true, |desc| desc.extent != extent))
    {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

fn validate_target(
    graph: &GraphBuilder,
    texture: TextureHandle,
    min_max: bool,
) -> Result<(), GraphError> {
    let desc = texture_desc(graph, texture)?;
    let expected_format = if min_max {
        matches!(
            desc.format,
            TextureFormat::Rg32Float | TextureFormat::Rg16Float
        )
    } else {
        desc.format == TextureFormat::Rgba16Float
    };
    if !expected_format
        || !color_texture(desc)
        || !desc.usage.contains(TextureUsage::TEXTURE_BINDING)
    {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

fn validate_sampled_color(graph: &GraphBuilder, texture: TextureHandle) -> Result<(), GraphError> {
    let desc = texture_desc(graph, texture)?;
    if !color_texture(desc) || !desc.usage.contains(TextureUsage::TEXTURE_BINDING) {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

fn validate_opaque_depth(graph: &GraphBuilder, texture: TextureHandle) -> Result<(), GraphError> {
    let desc = texture_desc(graph, texture)?;
    if desc.format != TextureFormat::Depth32Float
        || !single_2d(desc)
        || !desc.usage.contains(TextureUsage::RENDER_ATTACHMENT)
    {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

fn validate_final_output(graph: &GraphBuilder, texture: TextureHandle) -> Result<(), GraphError> {
    let desc = texture_desc(graph, texture)?;
    if !color_texture(desc) {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

fn texture_desc(
    graph: &GraphBuilder,
    texture: TextureHandle,
) -> Result<&fluxel_rhi::api::TextureDesc, GraphError> {
    graph.texture_version(texture)?;
    graph
        .textures
        .get(texture.resource)
        .map(|node| &node.desc)
        .ok_or(GraphError::UnknownResource)
}

fn single_2d(desc: &fluxel_rhi::api::TextureDesc) -> bool {
    desc.dimension == fluxel_rhi::api::TextureDimension::D2
        && desc.extent[0] != 0
        && desc.extent[1] != 0
        && desc.extent[2] == 1
        && desc.array_layers == 1
        && desc.mip_levels == 1
        && desc.samples == 1
}

fn color_texture(desc: &fluxel_rhi::api::TextureDesc) -> bool {
    single_2d(desc) && desc.usage.contains(TextureUsage::RENDER_ATTACHMENT)
}

fn shader_read() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        layout: ImageLayout::ShaderReadOnly,
        descriptor_usage: TextureUsage::TEXTURE_BINDING,
    }
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

fn color_write() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::COLOR_OUTPUT,
            accesses: AccessMask::COLOR_WRITE,
        },
        layout: ImageLayout::ColorAttachment,
        descriptor_usage: TextureUsage::RENDER_ATTACHMENT,
    }
}

fn color_read_write() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::COLOR_OUTPUT,
            accesses: AccessMask::COLOR_READ | AccessMask::COLOR_WRITE,
        },
        layout: ImageLayout::ColorAttachment,
        descriptor_usage: TextureUsage::RENDER_ATTACHMENT,
    }
}
