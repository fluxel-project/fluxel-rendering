//! The produced scene color becomes one sampled refraction mip chain per capture.

use super::*;
use crate::graph::definition::TextureHandle;
use crate::graph::mip_generation::{MipGenerationPass, declare_capture_seed, declare_chain};

/// One immutable producer link and its separately owned sampled receiver texture.
pub(super) struct CaptureRefractionMips {
    producer: TextureHandle,
    texture: usize,
    levels: u32,
}

pub(super) fn prepare_aliases(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    capture_slots: &[bool],
    producer: TextureHandle,
    projected: bool,
) -> Result<(Vec<Option<TextureHandle>>, Option<CaptureRefractionMips>)> {
    let mut aliases = Vec::with_capacity(capture_slots.len());
    let mut refraction = None;
    for (texture, &is_capture) in capture_slots.iter().enumerate() {
        if !is_capture {
            aliases.push(None);
            continue;
        }
        let desc = &spec.textures.get(texture).ok_or(Error::InvalidRange)?.desc;
        if desc.mip_levels == 1 {
            aliases.push(Some(producer));
            continue;
        }
        if !projected || refraction.is_some() {
            return Err(Error::InvalidState);
        }
        let destination = graph
            .create_texture(desc.clone())
            .map_err(|_| Error::InvalidState)?;
        aliases.push(Some(destination));
        refraction = Some(CaptureRefractionMips {
            producer,
            texture,
            levels: desc.mip_levels,
        });
    }
    Ok((aliases, refraction))
}

pub(super) fn append_chain(
    graph: &mut GraphBuilder,
    handles: &mut capture_declaration::FrameHandles,
    plan: Option<CaptureRefractionMips>,
    passes: &mut Vec<MipGenerationPass>,
) -> Result<()> {
    if let Some(plan) = plan {
        let destination = handles
            .textures
            .get_mut(plan.texture)
            .ok_or(Error::InvalidRange)?;
        let (seeded, copy) = declare_capture_seed(graph, plan.producer, *destination, plan.texture)
            .map_err(|_| Error::InvalidState)?;
        let (output, chain) = declare_chain(graph, seeded, plan.texture, plan.levels)
            .map_err(|_| Error::InvalidState)?;
        *destination = output;
        passes.push(copy);
        passes.extend(chain);
    }
    Ok(())
}
