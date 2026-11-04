//! Logical CPU preparation for the first built-in Forward consumer.
//!
//! This module resolves exact scene asset generations, validates CPU payloads,
//! composes a Material with one of two Forward Shader templates, and retains
//! the resulting immutable inputs. It does not allocate GPU objects, declare
//! Graph access, or submit work.

use std::{fmt, ops::Range};

use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_shader::ShaderComposer;

use crate::{FeatureHandle, PreparedFrame, RenderId, RenderTransform, RenderView, SceneAssetRef};

use super::ForwardDrawSource;
use super::{
    GeometryLayout, MaterialPreparationError, PreparedMaterial, VertexLayoutKey, prepare_material,
    vertex_layout_key,
};

#[path = "forward_validation.rs"]
mod forward_validation;
use forward_validation::{
    finite_transform, forward_template, multiply_columns, resolve_asset, validate_cpu_geometry,
    validate_material_schema, validate_texture,
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
    /// Exactly four bytes per texel in row-major order.
    pub rgba8: Vec<u8>,
}

/// Logical asset references and dynamic tint owned by one Forward scene record.
#[derive(Clone, Debug)]
pub struct ForwardResources {
    /// Exact immutable geometry generation.
    pub geometry: SceneAssetRef<GeometryAsset>,
    /// Exact immutable Material generation.
    pub material: SceneAssetRef<MaterialAsset>,
    /// Exact immutable color-texture generation.
    pub texture: SceneAssetRef<ColorTextureAsset>,
    /// Per-record material tint uploaded through group 0 binding 0 later.
    pub tint: [f32; 4],
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
    texture: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    prepared: PreparedMaterial,
    tint: [f32; 4],
    transform: RenderTransform,
    view: RenderView,
    clip_from_local: [[f32; 4]; 4],
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
/// has no depth attachment, so moving a later compatible draw ahead of an
/// intervening incompatible draw could change the result for overlapping
/// geometry. Preserving prepared candidate order is the safe opaque baseline
/// until a depth-tested product has independent evidence.
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

    /// Returns the retained immutable color-texture generation.
    pub fn texture(&self) -> &AssetSnapshot<ColorTextureAsset, ColorTextureData> {
        &self.texture
    }

    /// Returns the prepared logical Material, Shader, and geometry products.
    pub fn prepared(&self) -> &PreparedMaterial {
        &self.prepared
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

    /// Returns the output extent of the retained view.
    pub fn extent(&self) -> [u32; 2] {
        self.view.extent
    }
}

/// A structured refusal while preparing one built-in Forward draw.
#[derive(Debug)]
pub enum ForwardPreparationError {
    /// A feature submitted the same local draw identity more than once.
    DuplicateFeatureLocalId {
        /// Feature whose request list was invalid.
        feature: FeatureHandle,
        /// Duplicate identity within that feature's request list.
        local_id: u64,
    },
    /// An exact asset generation referenced by a scene record was unavailable.
    MissingAsset {
        /// Asset category.
        asset: &'static str,
        /// Draw requesting the asset.
        source: ForwardDrawSource,
    },
    /// The supplied asset list contained one exact identity more than once.
    DuplicateAsset {
        /// Asset category.
        asset: &'static str,
        /// Draw requesting the asset.
        source: ForwardDrawSource,
    },
    /// The per-record tint has a non-finite component.
    InvalidTint {
        /// Draw requesting the tint.
        source: ForwardDrawSource,
    },
    /// A record transform has a non-finite component.
    InvalidTransform {
        /// Draw requesting the transform.
        source: ForwardDrawSource,
    },
    /// A retained view has a non-finite component or empty extent.
    InvalidView,
    /// Geometry stream count differs from its declared runtime layout.
    GeometryStreamCount {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
        /// Declared stream count.
        expected: usize,
        /// Actual CPU stream count.
        actual: usize,
    },
    /// Geometry declared an inverted vertex range.
    InvalidVertexRange {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// The initial Forward path does not yet consume instanced streams.
    InstanceStreamUnsupported {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
        /// Stream index.
        stream: usize,
    },
    /// A CPU stream cannot satisfy its selected draw's physical attribute footprint.
    GeometryPayloadTooShort {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
        /// Stream index.
        stream: usize,
    },
    /// Geometry does not provide the required `POSITION` semantic.
    MissingPosition {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// `UV` and `COLOR` must occur together for the first optional-attribute template.
    IncompleteOptionalAttributes {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// A Material contribution is outside the initial Forward binding schema.
    MaterialBindingSchema {
        /// Draw requesting the Material.
        source: ForwardDrawSource,
    },
    /// Material, Shader, or geometry logical validation failed.
    Material(MaterialPreparationError),
    /// Texture dimensions or payload length are invalid.
    InvalidTexture {
        /// Draw requesting the texture.
        source: ForwardDrawSource,
    },
}

impl fmt::Display for ForwardPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Forward preparation refused: {self:?}")
    }
}

impl std::error::Error for ForwardPreparationError {}

/// Resolves every selected candidate into a retained logical Forward draw input.
///
/// The supplied [`PreparedFrame`] already binds candidate identities to one
/// immutable scene version. This function resolves the exact asset generations
/// named by those records and keeps cloned snapshots in its output so later GPU
/// preparation cannot accidentally switch to newer content.
pub fn prepare_forward(
    frame: &PreparedFrame<'_, ForwardResources>,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
) -> Result<Vec<ForwardDrawInput>, ForwardPreparationError> {
    let view = frame.view_record();
    if view.extent.contains(&0) || !finite_transform(view.transform) {
        return Err(ForwardPreparationError::InvalidView);
    }
    let mut inputs = Vec::with_capacity(frame.candidates().len());
    for &record_id in frame.candidates() {
        let record =
            frame
                .record(record_id)
                .map_err(|_| ForwardPreparationError::InvalidTransform {
                    source: ForwardDrawSource::Scene(record_id),
                })?;
        inputs.push(prepare_draw(
            view,
            ForwardDrawSource::Scene(record_id),
            0,
            &record.resources,
            record.transform,
            assets,
            composer,
        )?);
    }
    Ok(inputs)
}

/// Resolves feature-owned Forward requests against the selected frame view.
///
/// The requests retain exact asset generations and are validated through the
/// same path as scene-owned records. Their local identities are checked before
/// any asset lookup or shader work so invalid feature payloads have no GPU-side
/// effect.
pub(crate) fn prepare_feature_draws(
    frame: &PreparedFrame<'_, ForwardResources>,
    feature: FeatureHandle,
    requests: &[ForwardDrawRequest],
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
) -> Result<Vec<ForwardDrawInput>, ForwardPreparationError> {
    let view = frame.view_record();
    if view.extent.contains(&0) || !finite_transform(view.transform) {
        return Err(ForwardPreparationError::InvalidView);
    }
    let mut local_ids = std::collections::HashSet::with_capacity(requests.len());
    for request in requests {
        if !local_ids.insert(request.local_id) {
            return Err(ForwardPreparationError::DuplicateFeatureLocalId {
                feature,
                local_id: request.local_id,
            });
        }
    }
    requests
        .iter()
        .map(|request| {
            prepare_draw(
                view,
                ForwardDrawSource::Feature {
                    feature,
                    local_id: request.local_id,
                },
                request.order,
                &request.resources,
                request.transform,
                assets,
                composer,
            )
        })
        .collect()
}

fn prepare_draw(
    view: &RenderView,
    source: ForwardDrawSource,
    order: i64,
    resources: &ForwardResources,
    transform: RenderTransform,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
) -> Result<ForwardDrawInput, ForwardPreparationError> {
    if !finite_transform(transform) {
        return Err(ForwardPreparationError::InvalidTransform { source });
    }
    if !resources.tint.iter().all(|value| value.is_finite()) {
        return Err(ForwardPreparationError::InvalidTint { source });
    }
    let geometry = resolve_asset(&assets.geometry, resources.geometry, "geometry", source)?;
    let material = resolve_asset(&assets.materials, resources.material, "material", source)?;
    let texture = resolve_asset(&assets.textures, resources.texture, "texture", source)?;
    validate_texture(texture.value(), source)?;
    let template = forward_template(&geometry.value().layout, source)?;
    let prepared = prepare_material(
        &material,
        &geometry.value().layout,
        composer,
        template.request(),
    )
    .map_err(ForwardPreparationError::Material)?;
    validate_material_schema(&prepared, source)?;
    validate_cpu_geometry(geometry.value(), prepared.geometry(), source)?;
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
        texture,
        prepared,
        tint: resources.tint,
        transform,
        view: view.clone(),
        clip_from_local,
    })
}

/// Orders opaque Forward draws into contiguous raster-program batches.
///
/// This runs after all asset, material and geometry validation. It owns the
/// returned draws so a later device coordinator cannot accidentally apply the
/// ranges to an un-ordered input vector. It preserves candidate order and only
/// joins neighboring draws that select identical raster-program facts.
pub(crate) fn batch_forward_draws(draws: Vec<ForwardDrawInput>) -> ForwardPreparedBatches {
    let runs = coalesce_adjacent(draws, ForwardCompatibilityKey::from_draw);
    let mut ordered = Vec::new();
    let mut batches = Vec::with_capacity(runs.len());
    for (key, members) in runs {
        let start = ordered.len();
        ordered.extend(members);
        batches.push(ForwardDrawBatch {
            key,
            draws: start..ordered.len(),
        });
    }
    ForwardPreparedBatches {
        draws: ordered,
        batches,
    }
}

fn coalesce_adjacent<T, K>(items: Vec<T>, key_for: impl Fn(&T) -> K) -> Vec<(K, Vec<T>)>
where
    K: Eq,
{
    let mut runs: Vec<(K, Vec<T>)> = Vec::new();
    for item in items {
        let key = key_for(&item);
        match runs.last_mut() {
            Some((last_key, members)) if *last_key == key => members.push(item),
            _ => runs.push((key, vec![item])),
        }
    }
    runs
}

#[cfg(test)]
#[path = "../tests/preparation/forward_batches.rs"]
mod forward_batches;

#[cfg(test)]
#[path = "../tests/preparation/feature_draws.rs"]
mod feature_draws;
