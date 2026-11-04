//! Completed device-local Forward products keyed by immutable asset generations.
//!
//! Uploading frames retain their own resident owners until completion. This cache
//! receives only clones of products after the coordinator has observed that
//! completion, so a failed or pending upload can never become reusable here.

use std::collections::{HashMap, HashSet};

use fluxel_assets::{AssetId, AssetSnapshot};
use fluxel_rhi::{
    api::{AccessMask, BufferUses, DeviceId, ImageLayout, StageMask, TextureUses},
    backend::ResourceBackend,
};

use crate::preparation::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardDrawInput, GeometryAsset,
    GeometryData,
};

use super::{ForwardTexture, ForwardVertexStream};

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

/// Cache identity for one completed sampled-texture realization on one RHI device.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ForwardTextureCacheKey {
    asset: AssetId<ColorTextureAsset>,
    generation: u64,
    device: DeviceId,
}

impl ForwardTextureCacheKey {
    /// Builds a key from the exact immutable snapshot selected by a draw.
    pub(crate) fn from_snapshot(
        snapshot: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
        device: DeviceId,
    ) -> Self {
        Self {
            asset: snapshot.id(),
            generation: snapshot.generation().get(),
            device,
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
}

impl<B: ResourceBackend> Default for ForwardResidentCache<B> {
    fn default() -> Self {
        Self {
            geometry: HashMap::new(),
            textures: HashMap::new(),
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
    pub(crate) fn texture_key(draw: &ForwardDrawInput, device: DeviceId) -> ForwardTextureCacheKey {
        ForwardTextureCacheKey::from_snapshot(draw.texture(), device)
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

    /// Publishes one completed sampled texture under its exact immutable asset key.
    ///
    /// `texture` must come from a frame whose upload completion has already
    /// been observed by the coordinator. The cache records the shader-read
    /// outgoing state and layout for the next Graph import.
    pub(crate) fn publish_texture(
        &mut self,
        key: ForwardTextureCacheKey,
        texture: &ForwardTexture<B>,
    ) -> bool {
        if texture.texture.device() != key.device || self.textures.contains_key(&key) {
            return false;
        }
        self.textures.insert(
            key,
            ForwardTexture {
                texture: texture.texture.clone(),
                view: texture.view.clone(),
                desc: texture.desc.clone(),
                instance: texture.instance,
                incoming: shader_read_uses(),
                layout: ImageLayout::ShaderReadOnly,
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
        self.textures
            .retain(|key, _| textures.contains(&(key.asset, key.generation)));
    }

    /// Drops completed products belonging to one replaced or lost device.
    pub(crate) fn remove_device(&mut self, device: DeviceId) {
        self.geometry.retain(|key, _| key.device != device);
        self.textures.retain(|key, _| key.device != device);
    }

    /// Removes every completed reusable product.
    pub(crate) fn clear(&mut self) {
        self.geometry.clear();
        self.textures.clear();
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
            desc: self.desc.clone(),
            instance: self.instance,
            incoming: self.incoming,
            layout: self.layout,
        }
    }
}

fn vertex_read_uses() -> BufferUses {
    BufferUses {
        stages: StageMask::VERTEX_INPUT,
        accesses: AccessMask::VERTEX_READ,
    }
}

fn shader_read_uses() -> TextureUses {
    TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::SHADER_READ,
    }
}
