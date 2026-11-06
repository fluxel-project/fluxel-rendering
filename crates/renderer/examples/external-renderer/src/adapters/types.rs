//! Adapter-owned scene and feature data types.

use std::collections::BTreeMap;

#[path = "implementation.rs"]
mod implementation;
#[path = "support.rs"]
mod support;
use fluxel_assets::AssetKind;
use fluxel_material::MaterialAsset;
use fluxel_renderer::{
    CpuFeature, CpuPhase, CpuPhaseContributions, DirtyRange, FeatureContext, FeatureError,
    FeatureHandle, FrameSnapshot, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, SceneAssetRef, TypedFeatureSnapshot, validate_context,
};
pub(crate) use implementation::{EcsStyleAdapter, ObjectAdapter, ReferenceAdapter};
use support::*;

struct TriangleGeometryAsset;
struct QuadGeometryAsset;

impl AssetKind for TriangleGeometryAsset {}
impl AssetKind for QuadGeometryAsset {}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ExternalResources {
    Triangle {
        geometry: SceneAssetRef<TriangleGeometryAsset>,
        material: SceneAssetRef<MaterialAsset>,
    },
    Quad {
        geometry: SceneAssetRef<QuadGeometryAsset>,
        material: SceneAssetRef<MaterialAsset>,
    },
}

impl ExternalResources {
    fn material(&self) -> MaterialReference {
        match self {
            Self::Triangle { material, .. } | Self::Quad { material, .. } => *material,
        }
    }

    fn with_material(&self, material: MaterialReference) -> Self {
        match self {
            Self::Triangle { geometry, .. } => Self::Triangle {
                geometry: *geometry,
                material,
            },
            Self::Quad { geometry, .. } => Self::Quad {
                geometry: *geometry,
                material,
            },
        }
    }
}

type ExternalScene = RenderScene<ExternalResources>;
type ExternalSnapshot = FrameSnapshot<ExternalResources>;
type ExternalRecord = RenderRecord<ExternalResources>;
type MaterialReference = SceneAssetRef<MaterialAsset>;

/// The observable result produced by either application-side adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReferenceResult {
    pub(crate) initial: SceneObservation,
    pub(crate) updated: SceneObservation,
    pub(crate) retained: SceneObservation,
    pub(crate) features: FeatureObservation,
}

/// Public-scene facts consumed by the reference without exposing Scene internals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SceneObservation {
    pub(crate) version: u64,
    pub(crate) record_material_slots: Vec<u32>,
    pub(crate) record_material_generations: Vec<u64>,
    pub(crate) view_extents: Vec<[u32; 2]>,
    pub(crate) record_dirty: Vec<DirtyRange>,
    pub(crate) view_dirty: Vec<DirtyRange>,
}

/// Two differently shaped application-owned Feature payloads tied to one frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FeatureObservation {
    pub(crate) initial_version: u64,
    pub(crate) updated_version: u64,
    pub(crate) initial_color_count: usize,
    pub(crate) updated_color: [u8; 4],
    pub(crate) initial_label_count: usize,
    pub(crate) updated_label: String,
    pub(crate) prepared_color_count: usize,
    pub(crate) prepared_view_count: usize,
    pub(crate) version_mismatch: FeatureError,
    pub(crate) handle_mismatch: FeatureError,
    pub(crate) cross_scene_mismatch: FeatureError,
    pub(crate) foreign_issuer_mismatch: FeatureError,
    pub(crate) cross_issuer_mismatch: FeatureError,
    pub(crate) storage_issuer_mismatch: FeatureError,
    pub(crate) consumed_contributions: usize,
    pub(crate) rejected_contribution: FeatureError,
    pub(crate) storage_context_mismatch: FeatureError,
    pub(crate) mixed_version_contribution: FeatureError,
    pub(crate) replayed_contribution: FeatureError,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EvidenceFacts {
    pub(crate) initial_version: u64,
    pub(crate) updated_version: u64,
    pub(crate) retained_version: u64,
    pub(crate) initial_material_generations: Vec<u64>,
    pub(crate) updated_material_generations: Vec<u64>,
    pub(crate) updated_extent: Vec<[u32; 2]>,
    pub(crate) updated_color: [u8; 4],
    pub(crate) updated_label: String,
    pub(crate) feature_initial_version: u64,
    pub(crate) feature_updated_version: u64,
    pub(crate) version_mismatch: FeatureError,
    pub(crate) handle_mismatch: FeatureError,
    pub(crate) cross_scene_mismatch: FeatureError,
    pub(crate) foreign_issuer_mismatch: FeatureError,
    pub(crate) cross_issuer_mismatch: FeatureError,
    pub(crate) storage_issuer_mismatch: FeatureError,
    pub(crate) consumed_contributions: usize,
    pub(crate) rejected_contribution: FeatureError,
    pub(crate) storage_context_mismatch: FeatureError,
    pub(crate) mixed_version_contribution: FeatureError,
    pub(crate) replayed_contribution: FeatureError,
}

pub(crate) struct EvidenceAttempt {
    directory: std::path::PathBuf,
    expected: EvidenceFacts,
}

struct DenseColorFeature {
    handle: FeatureHandle,
}

struct ViewLabelFeature {
    handle: FeatureHandle,
}

struct DenseColorPrepared {
    count: usize,
}

struct ViewLabelPrepared {
    count: usize,
}

impl CpuFeature<ExternalResources> for DenseColorFeature {
    type Input = Vec<[u8; 4]>;
    type Prepared = DenseColorPrepared;

    fn prepare(
        &self,
        context: &FeatureContext<'_, ExternalResources>,
        input: &TypedFeatureSnapshot<Self::Input>,
    ) -> Result<Self::Prepared, FeatureError> {
        validate_context(context, input, self.handle)?;
        if input.data().len() != context.snapshot().record_ids().count() {
            return Err(FeatureError::InvalidPreparedData {
                feature: "dense-colors",
                detail: "color count does not match scene records",
            });
        }
        Ok(DenseColorPrepared {
            count: input.data().len(),
        })
    }
}

impl CpuFeature<ExternalResources> for ViewLabelFeature {
    type Input = BTreeMap<u32, String>;
    type Prepared = ViewLabelPrepared;

    fn prepare(
        &self,
        context: &FeatureContext<'_, ExternalResources>,
        input: &TypedFeatureSnapshot<Self::Input>,
    ) -> Result<Self::Prepared, FeatureError> {
        validate_context(context, input, self.handle)?;
        if input.data().len() != context.snapshot().view_ids().count() {
            return Err(FeatureError::InvalidPreparedData {
                feature: "view-labels",
                detail: "label count does not match scene views",
            });
        }
        Ok(ViewLabelPrepared {
            count: input.data().len(),
        })
    }
}
