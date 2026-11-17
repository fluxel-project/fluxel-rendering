//! Logical CPU preparation for the first built-in Forward consumer.
//!
//! This module resolves exact scene asset generations, validates CPU payloads,
//! composes a Material with a bounded Forward Shader template, and retains
//! the resulting immutable inputs. It does not allocate GPU objects, declare
//! Graph access, or submit work.

use std::{collections::HashMap, ops::Range};

use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_shader::ShaderComposer;

use crate::{FeatureHandle, PreparedFrame, RenderId, RenderTransform, RenderView, SceneAssetRef};

use super::ForwardDrawSource;
use super::{
    GeometryLayout, MaterialPreparationError, PreparedMaterial, VertexLayoutKey, prepare_material,
    vertex_layout_key,
};

#[path = "forward_pbr.rs"]
mod forward_pbr;
#[path = "forward_validation.rs"]
mod forward_validation;
use forward_pbr::normal_from_local;
use forward_validation::{
    color_uv0_template, finite_transform, forward_template, multiply_columns, resolve_asset,
    validate_cpu_geometry, validate_material_schema, validate_texture,
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
}

impl TextureEncoding {
    /// The exact RHI format that preserves this payload's sampling semantics.
    #[must_use]
    pub fn rgba8_format(self) -> fluxel_rhi::api::TextureFormat {
        match self {
            Self::Linear => fluxel_rhi::api::TextureFormat::Rgba8Unorm,
            Self::Srgb => fluxel_rhi::api::TextureFormat::Rgba8UnormSrgb,
        }
    }
}

#[path = "forward_capture.rs"]
mod forward_capture;
#[path = "forward_material_request.rs"]
mod forward_material_request;
pub(crate) use forward_capture::ResolvedForwardTexture;
pub use forward_capture::{
    ForwardCaptureResources, ForwardTextureSource, PreparedCaptureReceiver,
    prepare_forward_capture_receiver,
};
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
    /// Optional named ordinary sampler state for this draw's material binding.
    ///
    /// Image assets deliberately do not own sampler state: one immutable image
    /// generation may be sampled by different draws with different policies.
    /// When omitted, the built-in consumer retains its existing default policy.
    pub sampler: Option<ForwardSamplerBinding>,
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

/// One named ordinary sampler binding supplied by a Forward draw.
///
/// The descriptor is immutable CPU input. Native sampler ownership is created
/// by the selected renderer backend and retained with the submitted frame.
#[derive(Clone, Debug)]
pub struct ForwardSamplerBinding {
    /// Exact material resource name declared for the ordinary sampler.
    pub resource: String,
    /// Requested ordinary sampler state.
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
    sampler: Option<fluxel_rhi::api::SamplerDesc>,
    tint: [f32; 4],
    transform: RenderTransform,
    view: RenderView,
    clip_from_local: [[f32; 4]; 4],
    normal_from_local: Option<[[f32; 4]; 4]>,
}

/// One exact device-program compatibility class for opaque Forward draws.
///
/// The key deliberately describes only facts which select a raster program:
/// the complete composed Shader request and every resolved vertex-buffer
/// layout. Dynamic uniform values, transform, texture identity and source
/// bytes remain per-draw binding facts, so they do not split a program batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ForwardCompatibilityKey {
    shader: fluxel_shader::ShaderArtifactKey,
    layouts: Vec<VertexLayoutKey>,
}

impl ForwardCompatibilityKey {
    pub(crate) fn from_draw(draw: &ForwardDrawInput) -> Self {
        Self {
            shader: draw.prepared.shader().cache_key().clone(),
            layouts: draw
                .prepared
                .geometry()
                .streams
                .iter()
                .map(vertex_layout_key)
                .collect(),
        }
    }
}

/// A contiguous compatible range in [`ForwardPreparedBatches::draws`].
///
/// Every draw in a range may use the same raster pipeline. They still retain
/// separate dynamic bindings and are recorded as separate draws by the first
/// Forward product.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ForwardDrawBatch {
    pub(crate) key: ForwardCompatibilityKey,
    pub(crate) draws: Range<usize>,
}

/// Deterministic Forward opaque ordering and its program-compatible ranges.
///
/// Batches only merge adjacent compatible draws. The initial Forward product
/// can run without depth, so moving a later compatible draw ahead of an
/// intervening incompatible draw could change overlapping geometry. Preserve
/// the pipeline's selected global order regardless of depth selection.
#[derive(Clone)]
pub(crate) struct ForwardPreparedBatches {
    draws: Vec<ForwardDrawInput>,
    batches: Vec<ForwardDrawBatch>,
}

impl ForwardPreparedBatches {
    /// Returns counts suitable for renderer workload reporting.
    pub(crate) fn stats(&self) -> ForwardBatchStats {
        ForwardBatchStats {
            candidates: self.draws.len(),
            batches: self.batches.len(),
        }
    }

    /// Consumes the container into ordered draws and their contiguous ranges.
    pub(crate) fn into_draws_and_batches(self) -> (Vec<ForwardDrawInput>, Vec<ForwardDrawBatch>) {
        (self.draws, self.batches)
    }
}

/// Counts derived from one Forward opaque preparation result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ForwardBatchStats {
    /// Number of prepared opaque candidates.
    pub(crate) candidates: usize,
    /// Number of contiguous raster-program compatibility classes.
    pub(crate) batches: usize,
}

impl ForwardDrawInput {
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
    pub(crate) fn asset_texture(
        &self,
        index: usize,
    ) -> Option<&AssetSnapshot<ColorTextureAsset, ColorTextureData>> {
        match self.textures.get(index)? {
            ResolvedForwardTexture::Asset(texture) => Some(texture),
            ResolvedForwardTexture::CaptureSlot(_) => None,
        }
    }

    /// Returns the capture slot at `index`, if that is its resolved source.
    pub(crate) fn capture_texture_slot(&self, index: usize) -> Option<&str> {
        match self.textures.get(index)? {
            ResolvedForwardTexture::Asset(_) => None,
            ResolvedForwardTexture::CaptureSlot(slot) => Some(slot),
        }
    }

    /// Returns the prepared logical Material, Shader, and geometry products.
    pub fn prepared(&self) -> &PreparedMaterial {
        &self.prepared
    }

    /// Returns this draw's validated explicit ordinary-sampler descriptor.
    ///
    /// `None` names the legacy renderer default rather than an absent material
    /// sampler binding.
    pub(crate) fn sampler_descriptor(&self) -> Option<&fluxel_rhi::api::SamplerDesc> {
        self.sampler.as_ref()
    }

    /// Returns the validated dynamic tint.
    pub fn tint(&self) -> [f32; 4] {
        self.tint
    }

    /// Returns the record's resolved transform.
    pub fn transform(&self) -> RenderTransform {
        self.transform
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

    /// Returns the exact transform-uniform payload size selected for this draw.
    pub const fn transform_uniform_bytes(&self) -> u64 {
        if self.normal_from_local.is_some() {
            192
        } else {
            64
        }
    }

    /// Returns the output extent of the retained view.
    pub fn extent(&self) -> [u32; 2] {
        self.view.extent
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

fn prepare_draw(
    view: &RenderView,
    source: ForwardDrawSource,
    order: i64,
    resources: &ForwardResources,
    transform: RenderTransform,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
    environment: bool,
    specular: bool,
) -> Result<ForwardDrawInput, ForwardPreparationError> {
    if !finite_transform(transform) {
        return Err(ForwardPreparationError::InvalidTransform { source });
    }
    if !resources.tint.iter().all(|value| value.is_finite()) {
        return Err(ForwardPreparationError::InvalidTint { source });
    }
    let geometry = resolve_asset(&assets.geometry, resources.geometry, "geometry", source)?;
    let material = resolve_asset(&assets.materials, resources.material, "material", source)?;
    let request = material_request(
        material.value(),
        &geometry.value().layout,
        view,
        source,
        environment,
        specular,
    )?;
    let prepared = prepare_material(&material, &geometry.value().layout, composer, request)
        .map_err(ForwardPreparationError::Material)?;
    let schema = validate_material_schema(&prepared, source)?;
    let textures = resolve_asset_textures(resources, &schema, assets, source)?;
    let sampler = super::forward_sampler::resolve_sampler_binding(
        resources.sampler.as_ref(),
        &schema,
        source,
    )?;
    let prepared = prepared.with_binding_schema(schema);
    validate_cpu_geometry(geometry.value(), prepared.geometry(), source)?;
    let normal_from_local = normal_from_local(
        geometry.value(),
        prepared.geometry(),
        transform,
        crate::pbr::is_pbr(material.value().output()),
        matches!(
            material.value().output(),
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. }
        ),
        source,
    )?;
    let clip_from_local = multiply_columns(view.transform, transform);
    if !clip_from_local
        .iter()
        .flatten()
        .all(|value| value.is_finite())
    {
        return Err(ForwardPreparationError::InvalidTransform { source });
    }
    Ok(ForwardDrawInput {
        source,
        order,
        geometry,
        material,
        textures,
        prepared,
        sampler,
        tint: resources.tint,
        transform,
        view: view.clone(),
        clip_from_local,
        normal_from_local,
    })
}

pub(crate) fn ordered_texture_sources<'a>(
    provided: &'a [ForwardTextureBinding],
    expected: &[super::MaterialTextureBinding],
    source: ForwardDrawSource,
) -> Result<Vec<&'a ForwardTextureBinding>, ForwardPreparationError> {
    let mut by_name = HashMap::with_capacity(provided.len());
    for texture in provided {
        if by_name.insert(texture.resource.as_str(), texture).is_some() {
            return Err(ForwardPreparationError::DuplicateTextureResource {
                source,
                resource: texture.resource.clone(),
            });
        }
    }
    let mut ordered = Vec::with_capacity(expected.len());
    for texture in expected {
        let Some(provided) = by_name.get(texture.name.as_str()) else {
            return Err(ForwardPreparationError::MissingTextureResource {
                source,
                resource: texture.name.clone(),
            });
        };
        ordered.push(*provided);
    }
    for texture in provided {
        if !expected
            .iter()
            .any(|expected| expected.name == texture.resource)
        {
            return Err(ForwardPreparationError::UnexpectedTextureResource {
                source,
                resource: texture.resource.clone(),
            });
        }
    }
    Ok(ordered)
}

fn resolve_asset_textures(
    resources: &ForwardResources,
    schema: &super::MaterialBindingSchema,
    assets: &ForwardAssets,
    source: ForwardDrawSource,
) -> Result<Vec<ResolvedForwardTexture>, ForwardPreparationError> {
    ordered_texture_sources(&resources.textures, &schema.textures, source)?
        .into_iter()
        .map(|texture| match &texture.source {
            ForwardTextureSource::Asset(reference) => {
                let asset = resolve_asset(&assets.textures, *reference, "texture", source)?;
                validate_texture(asset.value(), source)?;
                Ok(ResolvedForwardTexture::Asset(asset))
            }
            ForwardTextureSource::CaptureSlot(_) => {
                Err(ForwardPreparationError::CaptureTextureOutsideReceiver {
                    source,
                    resource: texture.resource.clone(),
                })
            }
        })
        .collect()
}

/// Orders opaque Forward draws into contiguous raster-program batches.
///
/// This runs after all asset, material and geometry validation. It owns the
/// returned draws so a later device coordinator cannot accidentally apply the
/// ranges to an un-ordered input vector. It preserves candidate order and only
/// joins neighboring draws that select identical raster-program facts.
#[path = "forward_batch.rs"]
mod forward_batch;
pub(crate) use forward_batch::batch_forward_draws;
