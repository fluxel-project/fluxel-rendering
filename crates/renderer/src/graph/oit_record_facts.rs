use crate::graph::{
    binding::{BufferSlot, ResourceInstanceId, TextureSlot},
    oit::DualDepthPeelInputs,
};
use crate::residency::{ForwardDepthAttachment, ForwardTexture, OitOwners};
pub(super) fn validate_instances<B: ResourceBackend>(
    owners: &OitOwners<B>,
    depth: &OitDepthTargetRef<'_, B>,
    opaque: &OitColorTargetRef<'_, B>,
    output: &OitColorTargetRef<'_, B>,
) -> Result<()> {
    let all = [
        opaque.instance,
        output.instance,
        depth.instance,
        owners.ping_pong[0].min_max.instance,
        owners.ping_pong[0].front.instance,
        owners.ping_pong[0].back.instance,
        owners.ping_pong[1].min_max.instance,
        owners.ping_pong[1].front.instance,
        owners.ping_pong[1].back.instance,
        owners.back_accumulator.instance,
    ];
    if all.iter().enumerate().any(|(i, x)| all[..i].contains(x)) {
        Err(Error::InvalidState)
    } else {
        Ok(())
    }
}
use fluxel_rhi::{
    api::{BindGroup, Error, RasterPipeline, Result, Texture, TextureDesc, TextureView},
    backend::ResourceBackend,
};

pub(super) fn validate_external<B: ResourceBackend>(
    device: fluxel_rhi::api::DeviceId,
    extent: [u32; 2],
    owner: &OitColorTargetRef<'_, B>,
) -> Result<()> {
    if owner.device != device
        || owner.texture.device() != device
        || owner.view.device() != device
        || owner.desc.extent != [extent[0], extent[1], 1]
    {
        Err(Error::InvalidState)
    } else {
        Ok(())
    }
}
pub(super) fn validate_depth<B: ResourceBackend>(
    device: fluxel_rhi::api::DeviceId,
    extent: [u32; 2],
    owner: &OitDepthTargetRef<'_, B>,
) -> Result<()> {
    if owner.device != device
        || owner.texture.device() != device
        || owner.view.device() != device
        || owner.desc.extent != [extent[0], extent[1], 1]
    {
        Err(Error::InvalidState)
    } else {
        Ok(())
    }
}
pub(crate) struct OitRecordSlots {
    pub(crate) ping_pong: [OitSideSlots; 2],
    pub(crate) back_accumulator: TextureSlot,
    pub(crate) opaque_color: TextureSlot,
    pub(crate) opaque_depth: TextureSlot,
    pub(crate) output: TextureSlot,
}
pub(crate) struct OitSideSlots {
    pub(crate) min_max: TextureSlot,
    pub(crate) front: TextureSlot,
    pub(crate) back: TextureSlot,
}
impl From<DualDepthPeelInputs> for OitRecordSlots {
    fn from(value: DualDepthPeelInputs) -> Self {
        Self {
            ping_pong: value.ping_pong.map(|side| OitSideSlots {
                min_max: side.min_max.into(),
                front: side.front.into(),
                back: side.back.into(),
            }),
            back_accumulator: value.back_accumulator.into(),
            opaque_color: value.opaque_color.into(),
            opaque_depth: value.opaque_depth.into(),
            output: value.output.into(),
        }
    }
}
pub(crate) struct OitDrawSlots<'a> {
    pub(crate) streams: &'a [BufferSlot],
    pub(crate) uniform: BufferSlot,
    pub(crate) textures: &'a [TextureSlot],
    pub(crate) environment: Option<TextureSlot>,
    pub(crate) specular_radiance: Option<TextureSlot>,
    pub(crate) specular_brdf_lut: Option<TextureSlot>,
    pub(crate) specular_view_uniform: Option<BufferSlot>,
}
pub(crate) struct OitSampledGroupRef<'a, B: ResourceBackend> {
    pub(crate) group: &'a BindGroup<B>,
    pub(crate) textures: &'a [ResourceInstanceId],
}
pub(crate) struct OitDrawPrograms<'a, B: ResourceBackend> {
    pub(crate) draw: usize,
    pub(crate) initial: &'a RasterPipeline<B>,
    pub(crate) peel: &'a RasterPipeline<B>,
    pub(crate) initial_group3: OitSampledGroupRef<'a, B>,
    pub(crate) peel_group3: [OitSampledGroupRef<'a, B>; 2],
    pub(crate) empty_group2: Option<&'a BindGroup<B>>,
}
pub(crate) struct OitFullscreenCommand<'a, B: ResourceBackend> {
    pub(crate) pipeline: &'a RasterPipeline<B>,
    pub(crate) bind_group: OitSampledGroupRef<'a, B>,
}
pub(crate) struct OitColorTargetRef<'a, B: ResourceBackend> {
    pub(crate) texture: &'a Texture<B>,
    pub(crate) view: &'a TextureView<B>,
    pub(crate) desc: &'a TextureDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) device: fluxel_rhi::api::DeviceId,
}

/// Borrowed identity of the opaque depth target used by an OIT product.
///
/// This deliberately borrows the actual texture and view held by its enclosing
/// frame.  OIT command validation is pointer-exact, so cloning native handles
/// here would make a distinct Rust owner and weaken the binding proof.
pub(crate) struct OitDepthTargetRef<'a, B: ResourceBackend> {
    pub(crate) texture: &'a Texture<B>,
    pub(crate) view: &'a TextureView<B>,
    pub(crate) desc: &'a TextureDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) device: fluxel_rhi::api::DeviceId,
}

impl<'a, B: ResourceBackend> From<&'a ForwardDepthAttachment<B>> for OitDepthTargetRef<'a, B> {
    fn from(value: &'a ForwardDepthAttachment<B>) -> Self {
        Self {
            texture: &value.texture,
            view: &value.view,
            desc: &value.desc,
            instance: value.instance,
            device: value.device,
        }
    }
}

impl<'a, B: ResourceBackend> From<&'a ForwardTexture<B>> for OitDepthTargetRef<'a, B> {
    fn from(value: &'a ForwardTexture<B>) -> Self {
        Self {
            texture: &value.texture,
            view: &value.view,
            desc: &value.desc,
            instance: value.instance,
            device: value.device,
        }
    }
}
pub(crate) struct OitRecordCommands<'a, B: ResourceBackend> {
    pub(crate) draws: &'a [OitDrawPrograms<'a, B>],
    pub(crate) init_back: OitFullscreenCommand<'a, B>,
    pub(crate) back_accumulate: [OitFullscreenCommand<'a, B>; 2],
    pub(crate) final_composite: [OitFullscreenCommand<'a, B>; 2],
    pub(crate) opaque_color: OitColorTargetRef<'a, B>,
    pub(crate) output: OitColorTargetRef<'a, B>,
}
