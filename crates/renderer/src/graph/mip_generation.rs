//! Graph-declared GPU mip generation for one ordinary Forward texture.
//!
//! Each blit receives its own pass because a middle mip is a CopyDst in one
//! blit and a CopySrc in the next.  That makes its state transition explicit
//! to the shared Graph state planner rather than relying on encoder-local
//! state tracking.
//!
//! Source and destination retain the immutable asset's exact format. For sRGB
//! images the selected RHI blit samples decoded linear RGB and encodes the
//! destination attachment again; alpha stays linear. A byte-space RGBA8
//! average would not preserve this contract. Descriptor support is checked at
//! allocation and an unsupported blit fails this Graph execution without
//! publishing a completed chain.

use fluxel_rhi::{
    api::{
        AccessMask, CommandEncoder, Error, FilterMode, ImageLayout, Result, StageMask,
        TextureAspects, TextureDesc, TextureDimension, TextureRegion, TextureUsage, TextureUses,
    },
    backend::{CommandBackend, ResourceBackend},
};

use crate::residency::ForwardTexture;

use super::{
    binding::TextureSlot,
    definition::{
        AccessKind, GraphBuilder, GraphError, GraphPass, TextureAccessContract, TextureHandle,
        TextureRange,
    },
    execution::FrameBindings,
};

/// One independently scheduled source-to-destination mip blit.
#[derive(Clone, Debug)]
pub(crate) struct MipGenerationPass {
    pub(crate) pass: GraphPass,
    /// Index into the caller's ordinary Forward texture owner/slot arrays.
    pub(crate) texture: usize,
    pub(crate) source_mip: u32,
    pub(crate) destination_mip: u32,
    desc: TextureDesc,
    seed_source: Option<TextureSlot>,
}

/// Declares all remaining levels after `texture` already has a defined mip 0.
///
/// The returned version has every mip defined. `texture_index` is only the
/// caller's physical-owner lookup key; the Graph derives all resource
/// dependency and range authority from `texture` itself.
pub(crate) fn declare_chain(
    graph: &mut GraphBuilder,
    texture: TextureHandle,
    texture_index: usize,
    levels: u32,
) -> std::result::Result<(TextureHandle, Vec<MipGenerationPass>), GraphError> {
    let desc = graph
        .textures
        .get(texture.resource)
        .filter(|_| texture.graph == graph.graph)
        .ok_or(GraphError::UnknownResource)?
        .desc
        .clone();
    // Check the nonce/current version before authoring any pass. A caller
    // cannot use a stale handle to manufacture a chain for an older version.
    graph.texture_version(texture)?;
    if levels == 0 || levels != desc.mip_levels || desc.dimension != TextureDimension::D2 {
        return Err(GraphError::InvalidRange);
    }

    let mut current = texture;
    let mut passes = Vec::with_capacity(levels.saturating_sub(1) as usize);
    for destination_mip in 1..levels {
        let source_mip = destination_mip - 1;
        let (pass, output) =
            graph.add_pass(format!("mip-generation-{destination_mip}"), |pass| {
                pass.read_texture_with_use(
                    current,
                    one_color_mip(source_mip),
                    copy_read().uses,
                    ImageLayout::CopySrc,
                    TextureUsage::COPY_SRC,
                )?;
                pass.write_texture_with_use(
                    current,
                    one_color_mip(destination_mip),
                    copy_write().uses,
                    ImageLayout::CopyDst,
                    TextureUsage::COPY_DST,
                )
            })?;
        current = output;
        passes.push(MipGenerationPass {
            pass,
            texture: texture_index,
            source_mip,
            destination_mip,
            desc: desc.clone(),
            seed_source: None,
        });
    }
    Ok((current, passes))
}

/// Copies a Graph-produced base level into a separate texture before downsampling.
pub(super) fn declare_capture_seed(
    graph: &mut GraphBuilder,
    source: TextureHandle,
    destination: TextureHandle,
    texture: usize,
) -> std::result::Result<(TextureHandle, MipGenerationPass), GraphError> {
    let desc = graph
        .textures
        .get(destination.resource)
        .filter(|_| destination.graph == graph.graph)
        .ok_or(GraphError::UnknownResource)?
        .desc
        .clone();
    if source.resource == destination.resource || desc.mip_levels < 2 {
        return Err(GraphError::InvalidRange);
    }
    let (pass, output) = graph.add_pass("capture-refraction-copy-base", |pass| {
        pass.read_texture_with_use(
            source,
            one_color_mip(0),
            copy_read().uses,
            ImageLayout::CopySrc,
            TextureUsage::COPY_SRC,
        )?;
        pass.write_texture_with_use(
            destination,
            one_color_mip(0),
            copy_write().uses,
            ImageLayout::CopyDst,
            TextureUsage::COPY_DST,
        )
    })?;
    Ok((
        output,
        MipGenerationPass {
            pass,
            texture,
            source_mip: 0,
            destination_mip: 0,
            desc,
            seed_source: Some(source.into()),
        },
    ))
}

/// Records one checked mip blit against the actual Forward texture owner.
///
/// The concrete texture used by the blit comes only from `CheckedPass`, after
/// Graph range/kind/contract validation. `actual` is an independent owner-fact
/// check; it never grants recording authority by itself.
pub(crate) fn record<B: ResourceBackend + CommandBackend>(
    generation: &MipGenerationPass,
    bindings: &FrameBindings<'_, B>,
    slot: TextureSlot,
    actual: &ForwardTexture<B>,
    encoder: &mut CommandEncoder<B>,
) -> Result<()> {
    validate_actual(generation, actual)?;
    let source_extent = mip_extent(&generation.desc, generation.source_mip)?;
    let destination_extent = mip_extent(&generation.desc, generation.destination_mip)?;
    let pass = bindings.pass(generation.pass)?;
    let source = pass.texture(
        generation.seed_source.unwrap_or(slot),
        one_color_mip(generation.source_mip),
        AccessKind::Read,
        copy_read(),
    )?;
    let destination = pass.texture(
        slot,
        one_color_mip(generation.destination_mip),
        AccessKind::Write,
        copy_write(),
    )?;
    pass.transition(encoder)?;
    let source = TextureRegion {
        texture: source,
        mip: generation.source_mip,
        origin: [0, 0, 0],
        extent: source_extent,
        aspect: TextureAspects::COLOR,
    };
    let destination = TextureRegion {
        texture: destination,
        mip: generation.destination_mip,
        origin: [0, 0, 0],
        extent: destination_extent,
        aspect: TextureAspects::COLOR,
    };
    if generation.seed_source.is_some() {
        encoder.copy_texture(source, destination)
    } else {
        encoder.blit(source, destination, FilterMode::Linear)
    }
}

fn validate_actual<B: ResourceBackend>(
    generation: &MipGenerationPass,
    actual: &ForwardTexture<B>,
) -> Result<()> {
    let desc = &generation.desc;
    if actual.texture.device() != actual.device
        || !same_texture_desc(&actual.desc, desc)
        || desc.dimension != TextureDimension::D2
        || desc.extent[2] != 1
        || desc.array_layers != 1
        || desc.samples != 1
        || (generation.seed_source.is_none() && generation.source_mip >= generation.destination_mip)
        || generation.destination_mip >= desc.mip_levels
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn same_texture_desc(left: &TextureDesc, right: &TextureDesc) -> bool {
    left.dimension == right.dimension
        && left.extent == right.extent
        && left.array_layers == right.array_layers
        && left.format == right.format
        && left.mip_levels == right.mip_levels
        && left.samples == right.samples
        && left.usage == right.usage
        && left.view_formats == right.view_formats
}

fn mip_extent(desc: &TextureDesc, mip: u32) -> Result<[u32; 3]> {
    if mip >= desc.mip_levels {
        return Err(Error::InvalidRange);
    }
    Ok([
        (desc.extent[0] >> mip).max(1),
        (desc.extent[1] >> mip).max(1),
        1,
    ])
}

fn one_color_mip(mip: u32) -> TextureRange {
    TextureRange::new(TextureAspects::COLOR, mip..mip + 1, 0..1)
}

fn copy_read() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        },
        layout: ImageLayout::CopySrc,
        descriptor_usage: TextureUsage::COPY_SRC,
    }
}

fn copy_write() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        },
        layout: ImageLayout::CopyDst,
        descriptor_usage: TextureUsage::COPY_DST,
    }
}

#[cfg(test)]
#[path = "../tests/graph/mip_generation.rs"]
mod tests;
