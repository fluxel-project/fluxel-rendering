//! Logical CPU preparation for the first built-in Forward consumer.
//!
//! This module resolves exact scene asset generations, validates CPU payloads,
//! composes a Material with a bounded Forward Shader template, and retains
//! the resulting immutable inputs. It does not allocate GPU objects, declare
//! Graph access, or submit work.

use std::ops::Range;

use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_material::{
    DynamicValue, MaterialAsset, MaterialDefinition, MaterialInstance, SurfaceAlphaMode,
    SurfaceSideOrientation,
};
use fluxel_shader::ShaderComposer;

use crate::{
    FeatureHandle, MeshVisibility, PreparedFrame, RenderId, RenderTransform, RenderView,
    SceneAssetRef,
};

use super::ForwardDrawSource;
use super::{
    GeometryLayout, MaterialPreparationError, PreparedMaterial, RefractionRadianceAsset,
    RefractionRadianceData, prepare_material,
};

#[path = "forward_array.rs"]
mod forward_array;
#[path = "forward_depth.rs"]
mod forward_depth;
#[path = "forward_depth_array.rs"]
mod forward_depth_array;
#[path = "forward_oit.rs"]
mod forward_oit;
#[path = "forward_pbr.rs"]
mod forward_pbr;
#[path = "forward_transmission.rs"]
mod forward_transmission;
#[path = "forward_volume.rs"]
mod forward_volume;
pub use forward_array::{ColorTextureArrayAsset, ColorTextureArrayData};
pub use forward_depth::{DepthTextureAsset, DepthTextureData};
pub use forward_depth_array::{DepthTextureArrayAsset, DepthTextureArrayData};
pub(crate) use forward_oit::{OitDrawPartition, partition_oit_draws, prepare_oit_material};
pub use forward_volume::{ColorTextureVolumeAsset, ColorTextureVolumeData};
#[path = "forward_validation.rs"]
mod forward_validation;
use forward_pbr::normal_from_local;
use forward_validation::{
    color_uv0_template, finite_transform, forward_template, multiply_columns, resolve_asset,
    validate_cpu_geometry, validate_material_schema,
};

/// Marker for immutable geometry assets consumed by the initial Forward path.
pub struct GeometryAsset;

impl AssetKind for GeometryAsset {}

/// CPU geometry streams and their non-indexed vertex range.
#[derive(Clone, Debug)]
pub struct GeometryData {
    /// Runtime stream and attribute descriptions.
    pub layout: GeometryLayout,
    /// CPU bytes for each layout stream in matching order.
    pub streams: Vec<Vec<u8>>,
    /// Vertex IDs consumed by this geometry's non-indexed draw.
    pub vertices: Range<u32>,
}

/// Marker for immutable RGBA8 texture assets consumed by the initial Forward path.
pub struct ColorTextureAsset;

impl AssetKind for ColorTextureAsset {}

/// CPU payload for one RGBA8 color texture.
#[derive(Clone, Debug)]
pub struct ColorTextureData {
    /// Nonzero texture width and height.
    pub extent: [u32; 2],
    /// Transfer encoding of the supplied RGBA8 payload.
    pub encoding: TextureEncoding,
    /// Total mip levels requested for the immutable GPU texture. `rgba8`
    /// always contains only the base level; levels above zero are generated
    /// by the renderer after upload.
    pub mip_levels: u32,
    /// Exactly four bytes per texel in row-major order.
    pub rgba8: Vec<u8>,
}

/// Transfer encoding carried by an RGBA8 color-texture asset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureEncoding {
    /// Bytes are already linear RGBA values.
    Linear,
    /// RGB bytes are sRGB values decoded by the sampled texture format; alpha
    /// remains linear as required by the RGBA8 sRGB format contract.
    Srgb,
    /// RGB stores sRGB-encoded bytes sampled raw through an RGBA8 UNorm view.
    /// The material can decode the filtered sample in its shader.
    SrgbEncodedUnorm,
}

impl TextureEncoding {
    /// The exact RHI format that preserves this payload's sampling semantics.
    #[must_use]
    pub fn rgba8_format(self) -> fluxel_rhi::api::TextureFormat {
        match self {
            Self::Linear | Self::SrgbEncodedUnorm => fluxel_rhi::api::TextureFormat::Rgba8Unorm,
            Self::Srgb => fluxel_rhi::api::TextureFormat::Rgba8UnormSrgb,
        }
    }
}

#[path = "forward_capture.rs"]
mod forward_capture;
#[path = "forward_material_instance.rs"]
mod forward_material_instance;
#[path = "forward_material_request.rs"]
mod forward_material_request;
pub(crate) use forward_capture::ResolvedForwardTexture;
pub use forward_capture::{
    ForwardCaptureResources, ForwardTextureSource, PreparedCaptureReceiver,
    prepare_forward_capture_receiver, prepare_forward_capture_receiver_with_instance,
    prepare_forward_scene_refraction_receiver,
    prepare_forward_scene_refraction_receiver_with_sampler,
};
use forward_material_instance::{resolve_resource_aliases, resolve_uniform_values};
use forward_material_request::material_request;

/// Logical asset references and dynamic tint owned by one Forward scene record.
#[derive(Clone, Debug)]
pub struct ForwardResources {
    /// Exact immutable geometry generation.
    pub geometry: SceneAssetRef<GeometryAsset>,
    /// Exact immutable Material generation.
    pub material: SceneAssetRef<MaterialAsset>,
    /// Named material texture sources. Names must exactly match the material
    /// contribution's texture resource names.
    pub textures: Vec<ForwardTextureBinding>,
    /// Named sampler states for this draw's material bindings.
    ///
    /// Image assets deliberately do not own sampler state: one immutable image
    /// generation may be sampled by different draws with different policies.
    /// Omitted ordinary samplers retain the built-in default policy; comparison
    /// samplers always require an explicit matching descriptor.
    pub samplers: Vec<ForwardSamplerBinding>,
    /// Per-record vec4 value uploaded to the schema's material uniform binding.
    pub tint: [f32; 4],
}

/// One named texture source supplied to a Forward material.
#[derive(Clone, Debug)]
pub struct ForwardTextureBinding {
    /// Exact resource name declared by the material contribution.
    pub resource: String,
    /// Source resolved for this material resource.
    pub source: ForwardTextureSource,
}

/// One named sampler binding supplied by a Forward draw.
///
/// The descriptor is immutable CPU input. Native sampler ownership is created
/// by the selected renderer backend and retained with the submitted frame.
#[derive(Clone, Debug)]
pub struct ForwardSamplerBinding {
    /// Exact material resource name declared for the sampler.
    pub resource: String,
    /// Requested sampler state. The declared material binding determines
    /// whether this must carry a comparison function.
    pub descriptor: fluxel_rhi::api::SamplerDesc,
}

impl ForwardSamplerBinding {
    /// Binds `descriptor` to the named material sampler resource.
    #[must_use]
    pub fn new(resource: impl Into<String>, descriptor: fluxel_rhi::api::SamplerDesc) -> Self {
        Self {
            resource: resource.into(),
            descriptor,
        }
    }
}

impl ForwardTextureBinding {
    /// Binds an immutable color-texture asset to `resource`.
    #[must_use]
    pub fn asset(resource: impl Into<String>, asset: SceneAssetRef<ColorTextureAsset>) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::Asset(asset),
        }
    }

    /// Binds an immutable layered color-texture asset to `resource`.
    #[must_use]
    pub fn array_asset(
        resource: impl Into<String>,
        asset: AssetSnapshot<ColorTextureArrayAsset, ColorTextureArrayData>,
    ) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::ArrayAsset(asset),
        }
    }

    /// Binds one base-only array asset whose remaining mip levels are generated
    /// by the selected renderer path.
    #[must_use]
    pub fn array_asset_with_generated_mips(
        resource: impl Into<String>,
        asset: AssetSnapshot<ColorTextureArrayAsset, ColorTextureArrayData>,
        mip_levels: u32,
    ) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::ArrayAssetWithGeneratedMips { asset, mip_levels },
        }
    }

    /// Binds a packed cube asset to `resource`.
    ///
    /// The asset layers are physical faces in `+X, -X, +Y, -Y, +Z, -Z`
    /// order and its authored mip payloads remain exact.
    #[must_use]
    pub fn cube_asset(
        resource: impl Into<String>,
        asset: AssetSnapshot<ColorTextureArrayAsset, ColorTextureArrayData>,
    ) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::CubeAsset {
                asset,
                generated_mips: None,
            },
        }
    }

    /// Binds a base-only packed cube asset whose remaining mips are generated
    /// by the selected renderer path.
    #[must_use]
    pub fn cube_asset_with_generated_mips(
        resource: impl Into<String>,
        asset: AssetSnapshot<ColorTextureArrayAsset, ColorTextureArrayData>,
        generated_mips: u32,
    ) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::CubeAsset {
                asset,
                generated_mips: Some(generated_mips),
            },
        }
    }

    /// Binds one immutable prefiltered HDR radiance generation to a linear material texture.
    #[must_use]
    pub fn refraction_radiance(
        resource: impl Into<String>,
        asset: AssetSnapshot<RefractionRadianceAsset, RefractionRadianceData>,
    ) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::RefractionRadiance(asset),
        }
    }

    /// Binds a same-frame capture slot to `resource`.
    #[must_use]
    pub fn capture(resource: impl Into<String>, slot: impl Into<String>) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::CaptureSlot(slot.into()),
        }
    }
}

/// A Forward draw supplied directly by one typed render feature.
#[derive(Clone, Debug)]
pub struct ForwardDrawRequest {
    /// Identity unique within the requesting feature for this frame.
    pub local_id: u64,
    /// Explicit cross-source ordering value.
    pub order: i64,
    /// Exact immutable assets and dynamic tint for the draw.
    pub resources: ForwardResources,
    /// World-from-local transform for the draw.
    pub transform: RenderTransform,
}

/// Asset snapshots visible to the Forward preparation consumer.
///
/// The caller retains normal asset-store ownership; this structure only supplies
/// immutable snapshots from which a scene record selects one exact generation.
#[derive(Clone, Default)]
pub struct ForwardAssets {
    /// Available geometry snapshots.
    pub geometry: Vec<AssetSnapshot<GeometryAsset, GeometryData>>,
    /// Available Material snapshots.
    pub materials: Vec<AssetSnapshot<MaterialAsset, MaterialDefinition>>,
    /// Available color-texture snapshots.
    pub textures: Vec<AssetSnapshot<ColorTextureAsset, ColorTextureData>>,
}

/// One fully resolved, still CPU-only Forward draw input.
#[derive(Clone)]
pub struct ForwardDrawInput {
    source: ForwardDrawSource,
    order: i64,
    geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    textures: Vec<ResolvedForwardTexture>,
    prepared: PreparedMaterial,
    samplers: Vec<(u32, fluxel_rhi::api::SamplerDesc)>,
    tint: [f32; 4],
    uniform_values: Vec<DynamicValue>,
    mesh_side_orientation: SurfaceSideOrientation,
    mesh_visibility: MeshVisibility,
    transform: RenderTransform,
    view: RenderView,
    clip_from_local: [[f32; 4]; 4],
    normal_from_local: Option<[[f32; 4]; 4]>,
    instance_transforms: Option<super::forward_instances::InstanceTransforms>,
    particle: Option<ParticleDrawPayload>,
    // The selected material resources that implement projected scene-color
    // transmission. This stays with the retained draw so stage-specific OIT
    // material recompilation cannot silently fall back to environment-only
    // transmission.
    scene_refraction: Option<forward_capture::SceneRefractionBinding>,
}

/// Renderer-private state that selects a GPU-produced particle instance stream.
///
/// The draw retains no native buffer. Renderer assembly supplies the exact
/// completion-owned particle product to the reflected instance stream.
#[derive(Clone)]
pub(crate) struct ParticleDrawPayload {
    count: u32,
    transform: [u8; super::particle_material::PARTICLE_TRANSFORM_BYTES as usize],
}

impl ParticleDrawPayload {
    /// Returns the number of particle instances consumed by this draw.
    pub(crate) const fn count(&self) -> u32 {
        self.count
    }

    /// Returns the fully packed particle transform uniform record.
    pub(crate) const fn transform(
        &self,
    ) -> &[u8; super::particle_material::PARTICLE_TRANSFORM_BYTES as usize] {
        &self.transform
    }
}

impl ForwardDrawInput {
    /// Returns the concrete instance indices consumed by this draw.
    pub fn instance_range(&self) -> Range<u32> {
        0..self.particle.as_ref().map_or_else(
            || {
                self.instance_transforms
                    .map_or(1, |instances| instances.count)
            },
            ParticleDrawPayload::count,
        )
    }

    pub(crate) fn has_instance_transforms(&self) -> bool {
        self.instance_transforms.is_some()
    }

    pub(crate) fn instance_reverses_winding(&self) -> bool {
        self.instance_transforms
            .is_some_and(|instances| instances.reverses_winding)
    }

    /// Returns the retained particle specialization, when this is a particle draw.
    pub(crate) fn particle(&self) -> Option<&ParticleDrawPayload> {
        self.particle.as_ref()
    }

    /// Returns the source that requested this draw.
    pub const fn source(&self) -> ForwardDrawSource {
        self.source
    }

    /// Returns this draw's explicit cross-source ordering value.
    pub const fn order(&self) -> i64 {
        self.order
    }

    /// Returns the scene record that requested this draw, when it has one.
    ///
    /// Feature-owned draws have no manufactured scene identity and return
    /// `None`.
    pub fn record(&self) -> Option<RenderId> {
        match self.source {
            ForwardDrawSource::Scene(record) => Some(record),
            ForwardDrawSource::Feature { .. } => None,
        }
    }

    /// Returns the retained immutable geometry generation.
    pub fn geometry(&self) -> &AssetSnapshot<GeometryAsset, GeometryData> {
        &self.geometry
    }

    /// Returns the retained immutable Material generation.
    pub fn material(&self) -> &AssetSnapshot<MaterialAsset, MaterialDefinition> {
        &self.material
    }

    /// Returns all resolved material texture sources in schema binding order.
    pub(crate) fn resolved_textures(&self) -> &[ResolvedForwardTexture] {
        &self.textures
    }

    /// Returns the asset texture at `index`, if its source is an immutable
    /// color-texture asset.
    #[cfg(test)]
    pub(crate) fn asset_texture(
        &self,
        index: usize,
    ) -> Option<&AssetSnapshot<ColorTextureAsset, ColorTextureData>> {
        match self.textures.get(index)? {
            ResolvedForwardTexture::Asset(texture) => Some(texture),
            _ => None,
        }
    }

    /// Returns the capture slot at `index`, if that is its resolved source.
    pub(crate) fn capture_texture_slot(&self, index: usize) -> Option<&str> {
        match self.textures.get(index)? {
            ResolvedForwardTexture::CaptureSlot(slot) => Some(slot),
            _ => None,
        }
    }

    /// Returns the prepared logical Material, Shader, and geometry products.
    pub fn prepared(&self) -> &PreparedMaterial {
        &self.prepared
    }

    /// Returns the validated explicit ordinary-sampler descriptor for `binding`.
    ///
    /// `None` names the legacy renderer default rather than an absent material
    /// sampler binding.
    pub(crate) fn sampler_descriptor(&self, binding: u32) -> Option<&fluxel_rhi::api::SamplerDesc> {
        self.samplers
            .iter()
            .find(|(actual, _)| *actual == binding)
            .map(|(_, descriptor)| descriptor)
    }

    /// Returns the validated dynamic tint.
    pub fn tint(&self) -> [f32; 4] {
        self.tint
    }

    /// Returns the record's resolved transform.
    pub fn transform(&self) -> RenderTransform {
        self.transform
    }

    /// Returns the per-draw instance values in reflected uniform binding order.
    pub fn uniform_values(&self) -> &[DynamicValue] {
        &self.uniform_values
    }

    pub(crate) const fn mesh_side_orientation(&self) -> SurfaceSideOrientation {
        self.mesh_side_orientation
    }

    /// Returns the prepared mesh opacity multiplier.
    pub const fn mesh_visibility(&self) -> MeshVisibility {
        self.mesh_visibility
    }

    pub(crate) fn effective_alpha_mode(&self) -> SurfaceAlphaMode {
        effective_alpha_mode(
            self.prepared.material().surface_policy().alpha_mode,
            self.mesh_visibility,
        )
    }

    /// Returns the validated scene view retained from the same snapshot.
    pub fn view(&self) -> &RenderView {
        &self.view
    }

    /// Returns `view.transform * record.transform` in column-major order.
    ///
    /// The initial Forward convention treats the retained view transform as
    /// clip-from-world and the record transform as world-from-local.
    pub fn clip_from_local(&self) -> [[f32; 4]; 4] {
        self.clip_from_local
    }

    /// Returns the PBR local-normal transform, when this draw consumes normals.
    pub fn normal_from_local(&self) -> Option<[[f32; 4]; 4]> {
        self.normal_from_local
    }

    pub(crate) fn scene_refraction_names(&self) -> Option<(&str, &str)> {
        self.scene_refraction
            .as_ref()
            .map(forward_capture::SceneRefractionBinding::names)
    }

    /// Returns the exact transform-uniform payload size selected for this draw.
    pub const fn transform_uniform_bytes(&self) -> u64 {
        if self.particle.is_some() {
            super::particle_material::PARTICLE_TRANSFORM_BYTES
        } else if self.normal_from_local.is_some() {
            if self.prepared.uses_scene_refraction() {
                crate::frame_data::forward::PBR_SCENE_REFRACTION_TRANSFORM_BYTES
            } else {
                crate::frame_data::forward::PBR_TRANSFORM_BYTES
            }
        } else {
            crate::frame_data::forward::UNLIT_FORWARD_TRANSFORM_BYTES
        }
    }

    /// Returns the output extent of the retained view.
    pub fn extent(&self) -> [u32; 2] {
        self.view.extent
    }
}

pub(super) fn effective_alpha_mode(
    mode: SurfaceAlphaMode,
    visibility: MeshVisibility,
) -> SurfaceAlphaMode {
    match (mode, visibility.get() < 1.0) {
        (SurfaceAlphaMode::Opaque, true) => SurfaceAlphaMode::Blend,
        (SurfaceAlphaMode::Mask, true) => SurfaceAlphaMode::MaskBlend,
        _ => mode,
    }
}

#[path = "forward_errors.rs"]
mod forward_errors;
pub use forward_errors::ForwardPreparationError;

#[path = "forward_lighting.rs"]
mod lighting;
#[cfg(test)]
pub(crate) use lighting::prepare_feature_draws;
pub use lighting::prepare_forward;
pub(crate) use lighting::{prepare_feature_draws_with_lighting, prepare_forward_with_lighting};

#[path = "forward_draw.rs"]
mod forward_draw;
use forward_draw::prepare_draw;

#[path = "forward_particles.rs"]
mod particles;
pub(crate) use particles::prepare_particle_draw;

#[path = "forward_textures.rs"]
mod forward_textures;
use forward_textures::{ordered_texture_sources, resolve_asset_textures};

#[path = "forward_batch.rs"]
mod forward_batch;
pub(crate) use forward_batch::{ForwardBatchStats, ForwardCompatibilityKey, batch_forward_draws};

#[cfg(test)]
#[path = "../tests/preparation/feature_draws.rs"]
mod feature_draws;

#[cfg(test)]
#[path = "../tests/preparation/forward_pbr.rs"]
mod forward_pbr_tests;

#[cfg(test)]
#[path = "../tests/preparation/color_uv0.rs"]
mod color_uv0_tests;

#[cfg(test)]
#[path = "../tests/preparation/forward_texture_validation.rs"]
mod forward_texture_validation_tests;

#[cfg(test)]
#[path = "../tests/preparation/forward_sampler.rs"]
mod forward_sampler_tests;
