//! Renderer-local executable Deferred program cache.
//!
//! The cache keeps the selected frame's G-buffer set and one lighting variant.
//! Submitted frames retain cloned RHI pipeline and binding handles, so replacing
//! a cache entry does not release the executable handles held by pending work.

use std::{collections::HashMap, sync::Arc};

use fluxel_rhi::{
    api::{DeviceId, TextureFormat},
    backend::ResourceBackend,
};

use crate::{PreparedMaterial, preparation::VertexLayoutKey};

use super::{GbufferProgram, LightingProgram};

/// Complete fixed-function facts for one realized G-buffer program.
///
/// The artifact identity is the full composition request identity rather than
/// a source hash, and device provenance is retained even though the cache is
/// renderer-local. That makes the key suitable for explicit loss clearing and
/// prevents accidental reuse if cache ownership is ever moved.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct GbufferProgramKey {
    device: DeviceId,
    artifact: fluxel_shader::ShaderArtifactKey,
    vertex_layouts: Vec<VertexLayoutKey>,
    transform_uniform_bytes: u64,
    color_formats: [TextureFormat; 2],
    depth_format: TextureFormat,
    multisample_count: u32,
    pbr: bool,
    effects: bool,
    surface: crate::renderer::RasterSurfaceState,
}

impl GbufferProgramKey {
    pub(super) fn for_material(
        device: DeviceId,
        prepared: &PreparedMaterial,
        transform_uniform_bytes: u64,
        pbr: bool,
        effects: bool,
        surface: crate::renderer::RasterSurfaceState,
    ) -> Self {
        Self {
            device,
            artifact: prepared.shader().cache_key().clone(),
            vertex_layouts: prepared
                .geometry()
                .streams
                .iter()
                .map(crate::preparation::vertex_layout_key)
                .collect(),
            transform_uniform_bytes,
            color_formats: [
                crate::residency::deferred_albedo_format(pbr),
                crate::residency::deferred_normal_format(pbr),
            ],
            depth_format: TextureFormat::Depth32Float,
            multisample_count: 1,
            pbr,
            effects,
            surface,
        }
    }
}

/// Complete fixed-function facts for one realized Deferred lighting program.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct LightingProgramKey {
    device: DeviceId,
    artifact: fluxel_shader::ShaderArtifactKey,
    sampled_formats: [TextureFormat; 3],
    color_format: TextureFormat,
    multisample_count: u32,
    pbr: bool,
    effects: bool,
}

impl LightingProgramKey {
    pub(super) fn for_artifact(
        device: DeviceId,
        artifact: fluxel_shader::ShaderArtifactKey,
        pbr: bool,
        effects: bool,
        color_format: TextureFormat,
    ) -> Self {
        Self {
            device,
            artifact,
            sampled_formats: [
                crate::residency::deferred_albedo_format(pbr),
                crate::residency::deferred_normal_format(pbr),
                TextureFormat::Depth32Float,
            ],
            color_format,
            multisample_count: 1,
            pbr,
            effects,
        }
    }
}

/// Bounded cache for the current Deferred program pair of one renderer.
///
/// A deferred frame owns cloned RHI executable handles through completion. The cache
/// intentionally retains at most one program for each pass so material source
/// variants cannot accumulate indefinitely in a long-lived renderer.
pub(crate) struct DeferredProgramCache<B: ResourceBackend> {
    gbuffer: HashMap<GbufferProgramKey, Arc<GbufferProgram<B>>>,
    lighting: HashMap<LightingProgramKey, Arc<LightingProgram<B>>>,
}

impl<B: ResourceBackend> Default for DeferredProgramCache<B> {
    fn default() -> Self {
        Self {
            gbuffer: HashMap::new(),
            lighting: HashMap::new(),
        }
    }
}

impl<B: ResourceBackend> DeferredProgramCache<B> {
    pub(super) fn get_gbuffer(&self, key: &GbufferProgramKey) -> Option<Arc<GbufferProgram<B>>> {
        self.gbuffer.get(key).cloned()
    }

    pub(super) fn get_lighting(&self, key: &LightingProgramKey) -> Option<Arc<LightingProgram<B>>> {
        self.lighting.get(key).cloned()
    }

    pub(super) fn retain_gbuffer(&mut self, active: &std::collections::HashSet<GbufferProgramKey>) {
        self.gbuffer.retain(|key, _| active.contains(key));
    }

    /// Publishes a fully realized executable in the selected frame's set.
    pub(super) fn publish_gbuffer(
        &mut self,
        key: GbufferProgramKey,
        program: Arc<GbufferProgram<B>>,
    ) {
        self.gbuffer.insert(key, program);
    }

    /// Publishes only a fully realized lighting executable as the active one.
    pub(super) fn publish_lighting(
        &mut self,
        key: LightingProgramKey,
        program: Arc<LightingProgram<B>>,
    ) {
        self.lighting.clear();
        self.lighting.insert(key, program);
    }

    /// Removes all renderer-local executable cache entries.
    ///
    /// Existing submitted frames keep their cloned RHI executable handles.
    pub(crate) fn clear(&mut self) {
        self.gbuffer.clear();
        self.lighting.clear();
    }
}
