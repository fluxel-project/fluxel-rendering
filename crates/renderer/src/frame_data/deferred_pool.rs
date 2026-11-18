//! Completion-gated reuse facts for Deferred frame and G-buffer allocations.
//!
//! The renderer coordinator establishes completion and successful readback before
//! retaining an entry. This module only preserves actual owner state and checks
//! saved allocation facts for a later compatible submission.

use fluxel_rhi::{
    api::{
        AccessMask, DeviceId, ImageLayout, StageMask, TextureDesc, TextureDimension, TextureFormat,
        TextureUsage, TextureUses,
    },
    backend::ResourceBackend,
};

use crate::{
    frame_data::pool,
    residency::{ForwardFrameBuffers, ForwardTexture},
};

/// A completed Deferred frame allocation eligible for a later compatible use.
pub(crate) struct DeferredFramePoolEntry<B: ResourceBackend> {
    /// The final-color, uniform, upload, and readback owners.
    pub(crate) frame: ForwardFrameBuffers<B>,
    /// Albedo, normal/roughness, and depth owners in Deferred attachment order.
    pub(crate) gbuffer: [ForwardTexture<B>; 3],
    pub(crate) effects: Option<ForwardTexture<B>>,
}

/// Records the actual post-readback states for a completed Deferred allocation.
///
/// The two color targets and depth target all leave the Deferred lighting pass
/// as sampled fragment inputs. Completion and readback success are intentionally
/// not inferred here; the renderer only calls this after it has established both.
pub(crate) fn recycle<B: ResourceBackend>(
    frame: ForwardFrameBuffers<B>,
    mut gbuffer: [ForwardTexture<B>; 3],
    mut effects: Option<ForwardTexture<B>>,
) -> DeferredFramePoolEntry<B> {
    for texture in &mut gbuffer {
        texture.incoming = sampled_fragment_uses();
        texture.layout = pooled_layout();
    }
    if let Some(texture) = &mut effects {
        texture.incoming = sampled_fragment_uses();
        texture.layout = pooled_layout();
    }
    DeferredFramePoolEntry {
        frame: pool::recycle(frame, true),
        gbuffer,
        effects,
    }
}

/// Checks actual saved owners before reusing a completed Deferred allocation.
///
/// The RHI currently exposes texture and view device provenance publicly, while
/// the owner retains the exact allocation descriptor. A texture-view's aspect
/// and subresource range are not queryable through the public RHI handle, so
/// this check relies only on the controlled owner's exact one-mip/one-layer
/// descriptor and never substitutes a Graph declaration for allocation facts.
pub(crate) fn compatible<B: ResourceBackend>(
    entry: &DeferredFramePoolEntry<B>,
    device: DeviceId,
    extent: [u32; 2],
    uniform_bytes: u64,
    row_pitch: u32,
    pbr: bool,
    effects: bool,
) -> bool {
    pool::compatible(&entry.frame, device, extent, uniform_bytes, row_pitch)
        && gbuffer_compatible(&entry.gbuffer, device, extent, pbr)
        && effect_compatible(&entry.effects, device, extent, effects)
}

fn effect_compatible<B: ResourceBackend>(
    effects: &Option<ForwardTexture<B>>,
    device: DeviceId,
    extent: [u32; 2],
    required: bool,
) -> bool {
    match (effects, required) {
        (None, false) => true,
        (Some(texture), true) => {
            texture.texture.device() == device
                && texture.view.device() == device
                && descriptor_compatible(&texture.desc, extent, TextureFormat::Rgba16Float)
                && texture.incoming == sampled_fragment_uses()
                && texture.layout == pooled_layout()
        }
        _ => false,
    }
}

fn gbuffer_compatible<B: ResourceBackend>(
    gbuffer: &[ForwardTexture<B>; 3],
    device: DeviceId,
    extent: [u32; 2],
    pbr: bool,
) -> bool {
    let formats = [
        crate::residency::deferred_albedo_format(pbr),
        TextureFormat::Rgba8Unorm,
        TextureFormat::Depth32Float,
    ];
    gbuffer.iter().zip(formats).all(|(texture, format)| {
        texture.texture.device() == device
            && texture.view.device() == device
            && descriptor_compatible(&texture.desc, extent, format)
            && texture.incoming == sampled_fragment_uses()
            && texture.layout == pooled_layout()
    })
}

fn descriptor_compatible(desc: &TextureDesc, extent: [u32; 2], format: TextureFormat) -> bool {
    desc.dimension == TextureDimension::D2
        && desc.extent == [extent[0], extent[1], 1]
        && desc.array_layers == 1
        && desc.format == format
        && desc.mip_levels == 1
        && desc.samples == 1
        && desc.usage == TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING
        && desc.view_formats.is_empty()
}

fn pooled_layout() -> ImageLayout {
    ImageLayout::ShaderReadOnly
}

fn sampled_fragment_uses() -> TextureUses {
    TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::SHADER_READ,
    }
}

#[cfg(test)]
#[path = "../tests/frame_data/deferred_pool.rs"]
mod tests;
