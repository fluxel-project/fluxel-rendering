//! Completed device-local Forward products keyed by immutable asset generations.
//!
//! Uploading frames retain their own resident owners until completion. This cache
//! receives only clones of products after the coordinator has observed that
//! completion, so a failed or pending upload can never become reusable here.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Weak},
};

use fluxel_assets::{AssetId, AssetSnapshot};
use fluxel_rhi::{
    api::{AccessMask, BufferUses, DeviceId, StageMask, TextureUsage},
    backend::ResourceBackend,
};

use crate::graph::material_read_contract;
use crate::preparation::{
    ColorTextureArrayAsset, ColorTextureArrayData, ColorTextureAsset, ColorTextureData,
    ColorTextureVolumeAsset, ColorTextureVolumeData, DepthTextureArrayAsset, DepthTextureArrayData,
    DepthTextureAsset, DepthTextureData, ForwardAssets, ForwardDrawInput, GeometryAsset,
    GeometryData, RefractionRadianceAsset, RefractionRadianceData, ResolvedForwardTexture,
};

use super::{ForwardTexture, ForwardVertexStream, LayeredTextureView};

/// Cache identity for one completed geometry realization on one RHI device.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ForwardGeometryCacheKey {
    asset: AssetId<GeometryAsset>,
    generation: u64,
    device: DeviceId,
}

impl ForwardGeometryCacheKey {
    /// Builds a key from the exact immutable snapshot selected by a draw.
    pub(crate) fn from_snapshot(
        snapshot: &AssetSnapshot<GeometryAsset, GeometryData>,
        device: DeviceId,
    ) -> Self {
        Self {
            asset: snapshot.id(),
            generation: snapshot.generation().get(),
            device,
        }
    }
}

/// Cache identity for one completed texture realization on one RHI device.
/// Asset kind and binding usage distinguish physical owners of the same generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ForwardTextureCacheKey {
    Color {
        asset: AssetId<ColorTextureAsset>,
        generation: u64,
        device: DeviceId,
    },
    StorageColor {
        asset: AssetId<ColorTextureAsset>,
        generation: u64,
        device: DeviceId,
    },
    ColorArray {
        asset: AssetId<ColorTextureArrayAsset>,
        generation: u64,
        device: DeviceId,
        generated_mips: Option<u32>,
        view: LayeredTextureView,
    },
    ColorVolume {
        asset: AssetId<ColorTextureVolumeAsset>,
        generation: u64,
        device: DeviceId,
        generated_mips: Option<u32>,
    },
    Depth {
        asset: AssetId<DepthTextureAsset>,
        generation: u64,
        device: DeviceId,
    },
    DepthArray {
        asset: AssetId<DepthTextureArrayAsset>,
        generation: u64,
        device: DeviceId,
    },
    RefractionRadiance {
        asset: AssetId<RefractionRadianceAsset>,
        generation: u64,
        device: DeviceId,
    },
}

impl ForwardTextureCacheKey {
    /// Builds a key from the exact immutable snapshot selected by a draw.
    pub(crate) fn from_snapshot(
        snapshot: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
        device: DeviceId,
    ) -> Self {
        Self::Color {
            asset: snapshot.id(),
            generation: snapshot.generation().get(),
            device,
        }
    }

    /// Builds a key from one immutable prefiltered radiance generation.
    pub(crate) fn from_refraction_radiance_snapshot(
        snapshot: &AssetSnapshot<RefractionRadianceAsset, RefractionRadianceData>,
        device: DeviceId,
    ) -> Self {
        Self::RefractionRadiance {
            asset: snapshot.id(),
            generation: snapshot.generation().get(),
            device,
        }
    }

    fn device(self) -> DeviceId {
        match self {
            Self::Color { device, .. }
            | Self::StorageColor { device, .. }
            | Self::ColorArray { device, .. }
            | Self::ColorVolume { device, .. }
            | Self::Depth { device, .. }
            | Self::DepthArray { device, .. }
            | Self::RefractionRadiance { device, .. } => device,
        }
    }
}

/// Reusable Forward products whose uploads have completed successfully.
///
/// This owner never tracks pending work. The Renderer submission owner retains
/// pending `ForwardFrame` allocations independently, including when cache
/// retention drops an obsolete generation.
pub(crate) struct ForwardResidentCache<B: ResourceBackend> {
    geometry: HashMap<ForwardGeometryCacheKey, Vec<ForwardVertexStream<B>>>,
    textures: HashMap<ForwardTextureCacheKey, ForwardTexture<B>>,
    /// Weak source-payload witnesses for independently selected radiance maps.
    /// A completed GPU product can stay reusable only while an owner still
    /// retains the exact immutable CPU generation.
    refraction_radiance: HashMap<ForwardTextureCacheKey, Weak<RefractionRadianceData>>,
    color_arrays: HashMap<ForwardTextureCacheKey, Weak<ColorTextureArrayData>>,
    color_volumes: HashMap<ForwardTextureCacheKey, Weak<ColorTextureVolumeData>>,
    depth_textures: HashMap<ForwardTextureCacheKey, Weak<DepthTextureData>>,
    depth_arrays: HashMap<ForwardTextureCacheKey, Weak<DepthTextureArrayData>>,
}

impl<B: ResourceBackend> Default for ForwardResidentCache<B> {
    fn default() -> Self {
        Self {
            geometry: HashMap::new(),
            textures: HashMap::new(),
            refraction_radiance: HashMap::new(),
            color_arrays: HashMap::new(),
            color_volumes: HashMap::new(),
            depth_textures: HashMap::new(),
            depth_arrays: HashMap::new(),
        }
    }
}

impl<B: ResourceBackend> ForwardResidentCache<B> {
    /// Derives the completed geometry cache key used by one prepared draw.
    pub(crate) fn geometry_key(
        draw: &ForwardDrawInput,
        device: DeviceId,
    ) -> ForwardGeometryCacheKey {
        ForwardGeometryCacheKey::from_snapshot(draw.geometry(), device)
    }

    /// Derives the completed texture cache key used by one prepared draw.
    #[cfg(test)]
    pub(crate) fn texture_key(
        draw: &ForwardDrawInput,
        texture_index: usize,
        device: DeviceId,
    ) -> Option<ForwardTextureCacheKey> {
        draw.asset_texture(texture_index)
            .map(|texture| ForwardTextureCacheKey::from_snapshot(texture, device))
    }

    /// Derives the cache key for an immutable material texture.
    pub(crate) fn resolved_texture_key(
        &mut self,
        draw: &ForwardDrawInput,
        texture_index: usize,
        device: DeviceId,
    ) -> Option<ForwardTextureCacheKey> {
        match draw.resolved_textures().get(texture_index)? {
            ResolvedForwardTexture::StorageAsset(texture) => {
                Some(ForwardTextureCacheKey::StorageColor {
                    asset: texture.id(),
                    generation: texture.generation().get(),
                    device,
                })
            }
            ResolvedForwardTexture::Asset(texture) => {
                Some(ForwardTextureCacheKey::from_snapshot(texture, device))
            }
            ResolvedForwardTexture::ArrayAsset(texture) => {
                Some(self.layered_texture_key(texture, device, None, LayeredTextureView::Array))
            }
            ResolvedForwardTexture::GeneratedArray { asset, mip_levels } => {
                Some(self.layered_texture_key(
                    asset,
                    device,
                    Some(*mip_levels),
                    LayeredTextureView::Array,
                ))
            }
            ResolvedForwardTexture::CubeAsset {
                asset,
                generated_mips,
            } => Some(self.layered_texture_key(
                asset,
                device,
                *generated_mips,
                LayeredTextureView::Cube,
            )),
            ResolvedForwardTexture::VolumeAsset {
                asset,
                generated_mips,
            } => {
                let key = ForwardTextureCacheKey::ColorVolume {
                    asset: asset.id(),
                    generation: asset.generation().get(),
                    device,
                    generated_mips: *generated_mips,
                };
                self.color_volumes
                    .insert(key, Arc::downgrade(&asset.value_arc()));
                Some(key)
            }
            ResolvedForwardTexture::DepthAsset(texture) => {
                let key = ForwardTextureCacheKey::Depth {
                    asset: texture.id(),
                    generation: texture.generation().get(),
                    device,
                };
                self.depth_textures
                    .insert(key, Arc::downgrade(&texture.value_arc()));
                Some(key)
            }
            ResolvedForwardTexture::DepthArrayAsset(texture) => {
                let key = ForwardTextureCacheKey::DepthArray {
                    asset: texture.id(),
                    generation: texture.generation().get(),
                    device,
                };
                self.depth_arrays
                    .insert(key, Arc::downgrade(&texture.value_arc()));
                Some(key)
            }
            ResolvedForwardTexture::RefractionRadiance(texture) => {
                let key =
                    ForwardTextureCacheKey::from_refraction_radiance_snapshot(texture, device);
                self.refraction_radiance
                    .insert(key, Arc::downgrade(&texture.value_arc()));
                Some(key)
            }
            ResolvedForwardTexture::CaptureSlot(_) => None,
        }
    }

    fn layered_texture_key(
        &mut self,
        texture: &AssetSnapshot<ColorTextureArrayAsset, ColorTextureArrayData>,
        device: DeviceId,
        generated_mips: Option<u32>,
        view: LayeredTextureView,
    ) -> ForwardTextureCacheKey {
        let key = ForwardTextureCacheKey::ColorArray {
            asset: texture.id(),
            generation: texture.generation().get(),
            device,
            generated_mips,
            view,
        };
        self.color_arrays
            .insert(key, Arc::downgrade(&texture.value_arc()));
        key
    }

    /// Returns reusable vertex streams for a completed exact geometry generation.
    pub(crate) fn geometry(
        &self,
        key: ForwardGeometryCacheKey,
    ) -> Option<Vec<ForwardVertexStream<B>>> {
        self.geometry.get(&key).cloned()
    }

    /// Returns a reusable sampled texture for a completed exact texture generation.
    pub(crate) fn texture(&self, key: ForwardTextureCacheKey) -> Option<ForwardTexture<B>> {
        self.textures.get(&key).cloned()
    }

    /// Publishes completed vertex streams under an exact immutable asset key.
    ///
    /// `streams` must be taken from a frame whose upload completion has already
    /// been observed by the coordinator. The cache normalizes their outgoing
    /// state to vertex reads for the next Graph import. Empty products are not
    /// published.
    pub(crate) fn publish_geometry(
        &mut self,
        key: ForwardGeometryCacheKey,
        streams: &[&ForwardVertexStream<B>],
    ) -> bool {
        if streams.is_empty()
            || streams
                .iter()
                .any(|stream| stream.buffer.device() != key.device)
            || self.geometry.contains_key(&key)
        {
            return false;
        }
        let streams = streams
            .iter()
            .map(|stream| ForwardVertexStream {
                buffer: stream.buffer.clone(),
                desc: stream.desc.clone(),
                instance: stream.instance,
                incoming: vertex_read_uses(),
            })
            .collect();
        self.geometry.insert(key, streams);
        true
    }

    /// Publishes one completed texture under its exact immutable asset key.
    ///
    /// `texture` must come from a frame whose upload completion has already
    /// been observed by the coordinator. The cache records the shader-read
    /// outgoing state and layout for the next Graph import.
    pub(crate) fn publish_texture(
        &mut self,
        key: ForwardTextureCacheKey,
        texture: &ForwardTexture<B>,
    ) -> bool {
        if texture.texture.device() != key.device()
            || texture.device != key.device()
            || self.textures.contains_key(&key)
        {
            return false;
        }
        let Some(contract) = material_read_contract(&texture.desc) else {
            return false;
        };
        if matches!(key, ForwardTextureCacheKey::StorageColor { .. })
            != (contract.descriptor_usage == TextureUsage::STORAGE_BINDING)
        {
            return false;
        }
        self.textures.insert(
            key,
            ForwardTexture {
                texture: texture.texture.clone(),
                view: texture.view.clone(),
                view_dimension: texture.view_dimension,
                desc: texture.desc.clone(),
                instance: texture.instance,
                incoming: contract.uses,
                layout: contract.layout,
                device: texture.device,
                filterable: texture.filterable,
            },
        );
        true
    }

    /// Retains only exact generations still supplied by the current asset set.
    ///
    /// Pending submission owners are deliberately outside this cache and remain
    /// retained by the submission lifecycle even when this removes an obsolete
    /// completed generation.
    pub(crate) fn retain_assets(&mut self, assets: &ForwardAssets) {
        let geometry = assets
            .geometry
            .iter()
            .map(|snapshot| (snapshot.id(), snapshot.generation().get()))
            .collect::<HashSet<_>>();
        let textures = assets
            .textures
            .iter()
            .map(|snapshot| (snapshot.id(), snapshot.generation().get()))
            .collect::<HashSet<_>>();
        self.geometry
            .retain(|key, _| geometry.contains(&(key.asset, key.generation)));
        self.refraction_radiance
            .retain(|_, source| source.strong_count() != 0);
        self.color_arrays
            .retain(|_, source| source.strong_count() != 0);
        self.color_volumes
            .retain(|_, source| source.strong_count() != 0);
        self.depth_textures
            .retain(|_, source| source.strong_count() != 0);
        self.depth_arrays
            .retain(|_, source| source.strong_count() != 0);
        self.textures.retain(|key, _| match key {
            ForwardTextureCacheKey::Color {
                asset, generation, ..
            }
            | ForwardTextureCacheKey::StorageColor {
                asset, generation, ..
            } => textures.contains(&(*asset, *generation)),
            ForwardTextureCacheKey::ColorArray { .. } => self.color_arrays.contains_key(key),
            ForwardTextureCacheKey::ColorVolume { .. } => self.color_volumes.contains_key(key),
            ForwardTextureCacheKey::Depth { .. } => self.depth_textures.contains_key(key),
            ForwardTextureCacheKey::DepthArray { .. } => self.depth_arrays.contains_key(key),
            ForwardTextureCacheKey::RefractionRadiance { .. } => {
                self.refraction_radiance.contains_key(key)
            }
        });
    }

    /// Drops completed products belonging to one replaced or lost device.
    pub(crate) fn remove_device(&mut self, device: DeviceId) {
        self.geometry.retain(|key, _| key.device != device);
        self.textures.retain(|key, _| key.device() != device);
        self.refraction_radiance
            .retain(|key, _| key.device() != device);
        self.color_arrays.retain(|key, _| key.device() != device);
        self.color_volumes.retain(|key, _| key.device() != device);
        self.depth_textures.retain(|key, _| key.device() != device);
        self.depth_arrays.retain(|key, _| key.device() != device);
    }

    /// Removes every completed reusable product.
    pub(crate) fn clear(&mut self) {
        self.geometry.clear();
        self.textures.clear();
        self.refraction_radiance.clear();
        self.color_arrays.clear();
        self.color_volumes.clear();
        self.depth_textures.clear();
        self.depth_arrays.clear();
    }
}

impl<B: ResourceBackend> Clone for ForwardVertexStream<B> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            desc: self.desc.clone(),
            instance: self.instance,
            incoming: self.incoming,
        }
    }
}

impl<B: ResourceBackend> Clone for ForwardTexture<B> {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            view: self.view.clone(),
            view_dimension: self.view_dimension,
            desc: self.desc.clone(),
            instance: self.instance,
            incoming: self.incoming,
            layout: self.layout,
            device: self.device,
            filterable: self.filterable,
        }
    }
}

fn vertex_read_uses() -> BufferUses {
    BufferUses {
        stages: StageMask::VERTEX_INPUT,
        accesses: AccessMask::VERTEX_READ,
    }
}
