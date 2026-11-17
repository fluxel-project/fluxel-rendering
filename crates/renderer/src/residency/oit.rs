//! Concrete intermediate ownership for bounded dual depth peeling.
//!
//! This module owns only the seven color targets required by the Babylon-style
//! algorithm. The opaque depth attachment remains the Forward frame owner's
//! real attachment; OIT deliberately does not create a competing depth model.

use fluxel_rhi::{
    api::{
        AccessMask, Device, DeviceId, Error, ImageLayout, Result, StageMask, Support, Texture,
        TextureAspects, TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureUses,
        TextureView, TextureViewDesc, TextureViewDimension,
    },
    backend::ResourceBackend,
};

use crate::graph::ResourceInstanceId;

/// The min/max depth representation selected from actual device facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OitDepthPrecision {
    /// Preferred representation. It is read through an unfilterable-float
    /// `textureLoad` binding, never a filtering material sampler.
    Float32,
    /// Fallback only when the exact `Rg32Float` descriptor is unsupported.
    Float16,
}

/// Handle-free, fully admitted OIT allocation facts.
pub(crate) struct OitPlan {
    device: DeviceId,
    pub(crate) extent: [u32; 2],
    pub(crate) depth_precision: OitDepthPrecision,
    pub(crate) min_max_desc: TextureDesc,
    pub(crate) color_desc: TextureDesc,
}

/// One actual texture, its view, and the facts Graph checked binding consumes.
pub(crate) struct OitTextureOwner<B: ResourceBackend> {
    pub(crate) texture: Texture<B>,
    pub(crate) view: TextureView<B>,
    pub(crate) view_desc: TextureViewDesc,
    pub(crate) desc: TextureDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
    pub(crate) device: DeviceId,
}

/// One ping-pong side: min/max depth plus front and back color products.
pub(crate) struct OitPingPongOwners<B: ResourceBackend> {
    pub(crate) min_max: OitTextureOwner<B>,
    pub(crate) front: OitTextureOwner<B>,
    pub(crate) back: OitTextureOwner<B>,
}

/// Seven frame-owned targets retained through the submission completion.
pub(crate) struct OitOwners<B: ResourceBackend> {
    pub(crate) ping_pong: [OitPingPongOwners<B>; 2],
    pub(crate) back_accumulator: OitTextureOwner<B>,
    pub(crate) extent: [u32; 2],
    pub(crate) depth_precision: OitDepthPrecision,
    pub(crate) device: DeviceId,
}

impl OitPlan {
    /// Admits all exact target descriptors before any OIT texture is created.
    pub(crate) fn admit<B: ResourceBackend>(device: &Device<B>, extent: [u32; 2]) -> Result<Self> {
        if extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        if device.capabilities().limits.max_color_attachments < 3 {
            return Err(Error::Unsupported("dual-depth-peeling MRT admission"));
        }
        if device.capabilities().limits.max_bind_groups < 4
            || device.capabilities().limits.max_bindings_per_group < 2
        {
            return Err(Error::Unsupported("dual-depth-peeling binding admission"));
        }
        let color_desc = color_desc(extent);
        require_oit_target(device, &color_desc)?;
        let (depth_precision, min_max_desc) =
            match require_oit_target(device, &min_max_desc(extent, TextureFormat::Rg32Float)) {
                Ok(()) => (
                    OitDepthPrecision::Float32,
                    min_max_desc(extent, TextureFormat::Rg32Float),
                ),
                Err(_) => {
                    let fallback = min_max_desc(extent, TextureFormat::Rg16Float);
                    require_oit_target(device, &fallback)?;
                    (OitDepthPrecision::Float16, fallback)
                }
            };
        Ok(Self {
            device: device.id(),
            extent,
            depth_precision,
            min_max_desc,
            color_desc,
        })
    }
}

impl<B: ResourceBackend> OitOwners<B> {
    /// Creates exactly the admitted seven targets. Any post-admission failure is
    /// pre-submission and drops the partial native owners normally.
    pub(crate) fn allocate(device: &Device<B>, plan: OitPlan) -> Result<Self> {
        if plan.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let first = OitPingPongOwners {
            min_max: allocate_target(device, plan.min_max_desc.clone())?,
            front: allocate_target(device, plan.color_desc.clone())?,
            back: allocate_target(device, plan.color_desc.clone())?,
        };
        let second = OitPingPongOwners {
            min_max: allocate_target(device, plan.min_max_desc.clone())?,
            front: allocate_target(device, plan.color_desc.clone())?,
            back: allocate_target(device, plan.color_desc.clone())?,
        };
        let back_accumulator = allocate_target(device, plan.color_desc.clone())?;
        Ok(Self {
            ping_pong: [first, second],
            back_accumulator,
            extent: plan.extent,
            depth_precision: plan.depth_precision,
            device: device.id(),
        })
    }

    /// Rechecks saved actual facts before Graph imports or recording bindings.
    pub(crate) fn validate(&self, device: DeviceId) -> Result<()> {
        if self.device != device {
            return Err(Error::WrongDevice);
        }
        if self.extent.contains(&0) {
            return Err(Error::InvalidState);
        }
        let instances = [
            self.ping_pong[0].min_max.instance,
            self.ping_pong[0].front.instance,
            self.ping_pong[0].back.instance,
            self.ping_pong[1].min_max.instance,
            self.ping_pong[1].front.instance,
            self.ping_pong[1].back.instance,
            self.back_accumulator.instance,
        ];
        for (index, instance) in instances.iter().enumerate() {
            if instances[..index].contains(instance) {
                return Err(Error::InvalidState);
            }
        }
        let min_max_format = match self.depth_precision {
            OitDepthPrecision::Float32 => TextureFormat::Rg32Float,
            OitDepthPrecision::Float16 => TextureFormat::Rg16Float,
        };
        for side in &self.ping_pong {
            valid_owner(&side.min_max, device, self.extent, min_max_format)?;
            valid_owner(&side.front, device, self.extent, TextureFormat::Rgba16Float)?;
            valid_owner(&side.back, device, self.extent, TextureFormat::Rgba16Float)?;
        }
        valid_owner(
            &self.back_accumulator,
            device,
            self.extent,
            TextureFormat::Rgba16Float,
        )
    }
}

fn min_max_desc(extent: [u32; 2], format: TextureFormat) -> TextureDesc {
    target_desc("fluxel OIT minmax", extent, format)
}

fn color_desc(extent: [u32; 2]) -> TextureDesc {
    target_desc("fluxel OIT color", extent, TextureFormat::Rgba16Float)
}

fn target_desc(label: &str, extent: [u32; 2], format: TextureFormat) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn require_oit_target<B: ResourceBackend>(device: &Device<B>, desc: &TextureDesc) -> Result<()> {
    match device.texture_support(desc) {
        Support::Supported(support)
            if support.blendable
                && support.max_extent[0] >= desc.extent[0]
                && support.max_extent[1] >= desc.extent[1]
                && support.max_extent[2] >= desc.extent[2]
                && support.max_array_layers >= desc.array_layers
                && support.max_mips >= desc.mip_levels
                && support.sample_counts.contains(&desc.samples) =>
        {
            Ok(())
        }
        Support::Supported(_) => Err(Error::Unsupported("dual-depth-peeling target support")),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

fn allocate_target<B: ResourceBackend>(
    device: &Device<B>,
    desc: TextureDesc,
) -> Result<OitTextureOwner<B>> {
    let texture = device.create_texture(&desc)?;
    let view_desc = TextureViewDesc {
        format: desc.format,
        dimension: TextureViewDimension::D2,
        aspects: TextureAspects::COLOR,
        mips: 0..1,
        layers: 0..1,
    };
    let view = device.create_view(&texture, &view_desc)?;
    Ok(OitTextureOwner {
        texture,
        view,
        view_desc,
        desc,
        instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
        incoming: empty_uses(),
        layout: ImageLayout::Undefined,
        device: device.id(),
    })
}

fn valid_owner<B: ResourceBackend>(
    owner: &OitTextureOwner<B>,
    device: DeviceId,
    extent: [u32; 2],
    format: TextureFormat,
) -> Result<()> {
    if owner.device != device
        || owner.texture.device() != device
        || owner.view.device() != device
        || owner.desc.dimension != TextureDimension::D2
        || owner.desc.extent != [extent[0], extent[1], 1]
        || owner.desc.array_layers != 1
        || owner.desc.format != format
        || owner.desc.mip_levels != 1
        || owner.desc.samples != 1
        || owner.desc.usage != (TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING)
        || !owner.desc.view_formats.is_empty()
        || owner.view_desc.format != format
        || owner.view_desc.dimension != TextureViewDimension::D2
        || owner.view_desc.aspects != TextureAspects::COLOR
        || owner.view_desc.mips != (0..1)
        || owner.view_desc.layers != (0..1)
        || owner.incoming != empty_uses()
        || owner.layout != ImageLayout::Undefined
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn empty_uses() -> TextureUses {
    TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}
