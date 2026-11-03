//! Application-owned adapters for the public RenderScene update boundary.

use std::{collections::BTreeMap, fs, path::Path};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentCall, MaterialAsset, MaterialCompiler, MaterialDefinition,
    MaterialFragment, MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType,
    StaticValue, SurfaceOutput,
};
use fluxel_renderer::{
    CpuFeature, CpuPhase, CpuPhaseContributions, DirtyRange, FeatureContext, FeatureError,
    FeatureHandle, FrameSnapshot, RenderBounds, RenderChange, RenderChangeSet, RenderRecord,
    RenderScene, RenderTransform, RenderView, SceneAssetRef, SceneDirty, SceneError,
    TypedFeatureSnapshot, validate_context,
};

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

/// A direct application object map that owns external identity mapping.
pub(crate) struct ObjectAdapter;

/// An ECS-shaped adapter whose component columns remain application-owned.
pub(crate) struct EcsStyleAdapter;

/// Shared operations intentionally use only public Renderer scene types.
pub(crate) struct ReferenceAdapter;

impl ObjectAdapter {
    pub(crate) fn reference_scene() -> Result<ReferenceResult, Box<dyn std::error::Error>> {
        let mut objects = BTreeMap::new();
        let assets = asset_references()?;
        objects.insert(
            10_u64,
            sample_record(ExternalResources::Triangle {
                geometry: assets.triangle_geometry,
                material: assets.material_initial,
            })?,
        );
        objects.insert(
            20_u64,
            sample_record(ExternalResources::Quad {
                geometry: assets.quad_geometry,
                material: assets.material_initial,
            })?,
        );
        ReferenceAdapter::apply(
            objects.into_values().collect(),
            vec![RenderView::new(RenderTransform::identity(), [640, 360])?],
            assets.material_replacement,
        )
    }
}

impl EcsStyleAdapter {
    pub(crate) fn reference_scene() -> Result<ReferenceResult, Box<dyn std::error::Error>> {
        let assets = asset_references()?;
        let entities = [10_u64, 20_u64];
        let resources = [
            ExternalResources::Triangle {
                geometry: assets.triangle_geometry,
                material: assets.material_initial,
            },
            ExternalResources::Quad {
                geometry: assets.quad_geometry,
                material: assets.material_initial,
            },
        ];
        let transforms = [RenderTransform::identity(), RenderTransform::identity()];
        let bounds = [unit_bounds()?, unit_bounds()?];
        let records = entities
            .iter()
            .enumerate()
            .map(|(index, _)| RenderRecord {
                resources: resources[index].clone(),
                transform: transforms[index],
                bounds: bounds[index],
            })
            .collect();
        ReferenceAdapter::apply(
            records,
            vec![RenderView::new(RenderTransform::identity(), [640, 360])?],
            assets.material_replacement,
        )
    }
}

impl ReferenceAdapter {
    /// Saves source data and independent expectations before scene evaluation.
    pub(crate) fn begin_evidence(
        root: &Path,
        attempt: &str,
        input: &str,
        expected: EvidenceFacts,
    ) -> Result<EvidenceAttempt, Box<dyn std::error::Error>> {
        let directory = root.join(attempt);
        fs::create_dir_all(&directory)?;
        fs::write(directory.join("input.txt"), input)?;
        fs::write(directory.join("expected.txt"), format!("{expected:#?}\n"))?;
        fs::write(
            directory.join("actual.txt"),
            "actual unavailable before scene outcome\n",
        )?;
        fs::write(directory.join("error.txt"), "pending scene evaluation\n")?;
        Ok(EvidenceAttempt {
            directory,
            expected,
        })
    }

    /// Saves actual facts before executing the evidence oracle.
    pub(crate) fn complete_evidence(
        attempt: &EvidenceAttempt,
        actual: &ReferenceResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let observed = evidence_facts(actual);
        fs::write(
            attempt.directory.join("actual.txt"),
            format!("{observed:#?}\n"),
        )?;
        if observed != attempt.expected {
            fs::write(
                attempt.directory.join("error.txt"),
                format!(
                    "oracle mismatch\nexpected: {:#?}\nactual: {observed:#?}\n",
                    attempt.expected
                ),
            )?;
            return Err("external-renderer evidence oracle mismatch".into());
        }
        fs::write(attempt.directory.join("error.txt"), "no error\n")?;
        Ok(())
    }

    /// Records a construction error after input and expected facts are durable.
    pub(crate) fn fail_evidence(
        attempt: &EvidenceAttempt,
        error: &dyn std::fmt::Display,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(attempt.directory.join("error.txt"), format!("{error}\n"))?;
        Ok(())
    }

    pub(crate) fn expected_evidence() -> EvidenceFacts {
        EvidenceFacts {
            initial_version: 1,
            updated_version: 2,
            retained_version: 1,
            initial_material_generations: vec![1, 1],
            updated_material_generations: vec![2, 1],
            updated_extent: vec![[800, 450]],
            updated_color: [0, 0, 255, 255],
            updated_label: String::from("resized-main"),
            feature_initial_version: 1,
            feature_updated_version: 2,
            version_mismatch: FeatureError::SnapshotVersionMismatch {
                feature: "dense-colors",
                scene_version: 2,
                feature_version: 1,
            },
            handle_mismatch: FeatureError::FeatureHandleMismatch {
                expected: "view-labels",
                actual: "dense-colors",
            },
            cross_scene_mismatch: FeatureError::ForeignScene,
            foreign_issuer_mismatch: FeatureError::FeatureHandleMismatch {
                expected: "scene-issued feature",
                actual: "dense-colors",
            },
            cross_issuer_mismatch: FeatureError::FeatureHandleMismatch {
                expected: "scene-issued feature",
                actual: "dense-colors",
            },
            storage_issuer_mismatch: FeatureError::FeatureHandleMismatch {
                expected: "storage issuer",
                actual: "dense-colors",
            },
            consumed_contributions: 2,
            rejected_contribution: FeatureError::FeatureHandleMismatch {
                expected: "view-labels",
                actual: "dense-colors",
            },
            storage_context_mismatch: FeatureError::ForeignIssuer,
            mixed_version_contribution: FeatureError::FeatureHandleMismatch {
                expected: "view-labels",
                actual: "view-labels",
            },
            replayed_contribution: FeatureError::FeatureHandleMismatch {
                expected: "dense-colors",
                actual: "dense-colors",
            },
        }
    }

    pub(crate) fn assert_equivalent(
        direct: &ReferenceResult,
        ecs: &ReferenceResult,
    ) -> Result<(), String> {
        if direct == ecs {
            Ok(())
        } else {
            Err(format!(
                "adapter results differ: direct={direct:?}, ecs={ecs:?}"
            ))
        }
    }

    fn apply(
        records: Vec<ExternalRecord>,
        views: Vec<RenderView>,
        replacement: MaterialReference,
    ) -> Result<ReferenceResult, Box<dyn std::error::Error>> {
        let mut scene = ExternalScene::new()?;
        let issuer = scene.feature_issuer()?;
        let dense_handle = issuer.issue("dense-colors")?;
        let labels_handle = issuer.issue("view-labels")?;
        let dense_feature = DenseColorFeature {
            handle: dense_handle,
        };
        let labels_feature = ViewLabelFeature {
            handle: labels_handle,
        };
        let mut changes = RenderChangeSet::new();
        for record in records {
            changes.push(RenderChange::Insert(record));
        }
        for view in views {
            changes.push(RenderChange::InsertView(view));
        }
        let receipt = scene.apply(&changes)?;
        let initial_snapshot = scene.snapshot();
        let mut color_storage = issuer.storage::<Vec<[u8; 4]>>();
        let mut label_storage = issuer.storage::<BTreeMap<u32, String>>();
        let initial_dense_context =
            issuer.context(&initial_snapshot, CpuPhase::Reference, dense_handle)?;
        let initial_labels_context =
            issuer.context(&initial_snapshot, CpuPhase::Reference, labels_handle)?;
        let initial_colors = color_storage.publish(
            &initial_dense_context,
            vec![[255, 0, 0, 255], [0, 255, 0, 255]],
        )?;
        let initial_labels = label_storage.publish(
            &initial_labels_context,
            BTreeMap::from([(0_u32, String::from("main"))]),
        )?;

        let first = *receipt
            .inserted()
            .first()
            .ok_or_else(|| std::io::Error::other("reference input inserted no record"))?;
        let mut update = RenderChangeSet::new();
        update.push(RenderChange::Update(
            first,
            RenderRecord {
                resources: initial_snapshot
                    .record(first)?
                    .resources
                    .with_material(replacement),
                transform: RenderTransform::identity(),
                bounds: unit_bounds()?,
            },
        ));
        let view = *receipt
            .inserted_views()
            .first()
            .ok_or_else(|| std::io::Error::other("reference input inserted no view"))?;
        update.push(RenderChange::UpdateView(
            view,
            RenderView::new(RenderTransform::identity(), [800, 450])?,
        ));
        scene.apply(&update)?;
        let updated_snapshot = scene.snapshot();
        let dense_context = issuer.context(&updated_snapshot, CpuPhase::Reference, dense_handle)?;
        let labels_context =
            issuer.context(&updated_snapshot, CpuPhase::Reference, labels_handle)?;
        let updated_colors =
            color_storage.publish(&dense_context, vec![[0, 0, 255, 255], [0, 255, 0, 255]])?;
        let updated_labels = label_storage.publish(
            &labels_context,
            BTreeMap::from([(0_u32, String::from("resized-main"))]),
        )?;
        let dense_colors = dense_feature.prepare(
            &dense_context,
            color_storage
                .current()
                .ok_or_else(|| std::io::Error::other("color storage lost its publication"))?,
        )?;
        let view_labels = labels_feature.prepare(
            &labels_context,
            label_storage
                .current()
                .ok_or_else(|| std::io::Error::other("label storage lost its publication"))?,
        )?;
        let version_mismatch = match dense_feature.prepare(&dense_context, &initial_colors) {
            Err(error) => error,
            Ok(_) => {
                return Err("older typed data unexpectedly prepared for a later snapshot".into());
            }
        };
        let handle_mismatch = match labels_feature.prepare(&dense_context, &updated_labels) {
            Err(error) => error,
            Ok(_) => return Err("wrong feature handle unexpectedly prepared a feature".into()),
        };
        let (cross_scene_mismatch, foreign_issuer_mismatch) = feature_boundary_errors()?;
        let second_issuer = scene.feature_issuer()?;
        let cross_issuer_mismatch =
            match second_issuer.context(&updated_snapshot, CpuPhase::Reference, dense_handle) {
                Err(error) => error,
                Ok(_) => return Err("another issuer accepted a foreign same-scene handle".into()),
            };
        let second_dense_handle = second_issuer.issue("dense-colors")?;
        let second_dense_context =
            second_issuer.context(&updated_snapshot, CpuPhase::Reference, second_dense_handle)?;
        let second_dense_feature = DenseColorFeature {
            handle: second_dense_handle,
        };
        let storage_context_mismatch = match second_dense_feature.prepare(
            &second_dense_context,
            color_storage
                .current()
                .ok_or_else(|| std::io::Error::other("color storage lost its publication"))?,
        ) {
            Err(error) => error,
            Ok(_) => return Err("B context consumed A storage payload".into()),
        };
        let mut foreign_storage = second_issuer.storage::<Vec<[u8; 4]>>();
        let storage_issuer_mismatch = match foreign_storage
            .publish(&dense_context, vec![[17, 17, 17, 255], [19, 19, 19, 255]])
        {
            Err(error) => error,
            Ok(_) => return Err("another issuer storage accepted a foreign context".into()),
        };
        let mut contributions = CpuPhaseContributions::default();
        contributions.accept(&dense_context, dense_context.contribute())?;
        contributions.accept(&labels_context, labels_context.contribute())?;
        let rejected_contribution =
            match contributions.accept(&labels_context, dense_context.contribute()) {
                Err(error) => error,
                Ok(_) => return Err("phase collector accepted another feature contribution".into()),
            };
        let replayed_contribution =
            match contributions.accept(&dense_context, dense_context.contribute()) {
                Err(error) => error,
                Ok(_) => return Err("phase collector accepted replayed contribution".into()),
            };
        let mut mixed_versions = CpuPhaseContributions::default();
        mixed_versions.accept(&initial_dense_context, initial_dense_context.contribute())?;
        let mixed_version_contribution =
            match mixed_versions.accept(&labels_context, labels_context.contribute()) {
                Err(error) => error,
                Ok(_) => return Err("phase collector accepted mixed scene versions".into()),
            };

        Ok(ReferenceResult {
            initial: observe(&initial_snapshot)?,
            updated: observe(&updated_snapshot)?,
            retained: observe(&initial_snapshot)?,
            features: FeatureObservation {
                initial_version: initial_colors.scene_version(),
                updated_version: updated_colors.scene_version(),
                initial_color_count: initial_colors.data().len(),
                updated_color: updated_colors.data()[0],
                initial_label_count: initial_labels.data().len(),
                updated_label: updated_labels
                    .data()
                    .get(&0)
                    .ok_or_else(|| std::io::Error::other("updated label was absent"))?
                    .clone(),
                prepared_color_count: dense_colors.count,
                prepared_view_count: view_labels.count,
                version_mismatch,
                handle_mismatch,
                cross_scene_mismatch,
                foreign_issuer_mismatch,
                cross_issuer_mismatch,
                storage_issuer_mismatch,
                consumed_contributions: contributions.len(),
                rejected_contribution,
                storage_context_mismatch,
                mixed_version_contribution,
                replayed_contribution,
            },
        })
    }
}

fn evidence_facts(actual: &ReferenceResult) -> EvidenceFacts {
    EvidenceFacts {
        initial_version: actual.initial.version,
        updated_version: actual.updated.version,
        retained_version: actual.retained.version,
        initial_material_generations: actual.initial.record_material_generations.clone(),
        updated_material_generations: actual.updated.record_material_generations.clone(),
        updated_extent: actual.updated.view_extents.clone(),
        updated_color: actual.features.updated_color,
        updated_label: actual.features.updated_label.clone(),
        feature_initial_version: actual.features.initial_version,
        feature_updated_version: actual.features.updated_version,
        version_mismatch: actual.features.version_mismatch.clone(),
        handle_mismatch: actual.features.handle_mismatch.clone(),
        cross_scene_mismatch: actual.features.cross_scene_mismatch.clone(),
        foreign_issuer_mismatch: actual.features.foreign_issuer_mismatch.clone(),
        cross_issuer_mismatch: actual.features.cross_issuer_mismatch.clone(),
        storage_issuer_mismatch: actual.features.storage_issuer_mismatch.clone(),
        consumed_contributions: actual.features.consumed_contributions,
        rejected_contribution: actual.features.rejected_contribution.clone(),
        storage_context_mismatch: actual.features.storage_context_mismatch.clone(),
        mixed_version_contribution: actual.features.mixed_version_contribution.clone(),
        replayed_contribution: actual.features.replayed_contribution.clone(),
    }
}

fn feature_boundary_errors() -> Result<(FeatureError, FeatureError), Box<dyn std::error::Error>> {
    let first = ExternalScene::new()?;
    let second = ExternalScene::new()?;
    let first_snapshot = first.snapshot();
    let second_snapshot = second.snapshot();
    let first_issuer = first.feature_issuer()?;
    let second_issuer = second.feature_issuer()?;
    let first_handle = first_issuer.issue("dense-colors")?;
    let second_handle = second_issuer.issue("dense-colors")?;
    let first_context = first_issuer.context(&first_snapshot, CpuPhase::Reference, first_handle)?;
    let mut first_storage = first_issuer.storage::<Vec<[u8; 4]>>();
    let input = first_storage.publish(&first_context, Vec::<[u8; 4]>::new())?;
    let feature = DenseColorFeature {
        handle: second_handle,
    };
    let cross_scene_mismatch = match feature.prepare(
        &second_issuer.context(&second_snapshot, CpuPhase::Reference, second_handle)?,
        &input,
    ) {
        Err(error) => error,
        Ok(_) => return Err("a matching version from another scene unexpectedly prepared".into()),
    };
    let foreign_issuer_mismatch =
        match second_issuer.context(&second_snapshot, CpuPhase::Reference, first_handle) {
            Err(error) => error,
            Ok(_) => return Err("a foreign issuer handle unexpectedly created context".into()),
        };
    Ok((cross_scene_mismatch, foreign_issuer_mismatch))
}

fn observe(snapshot: &ExternalSnapshot) -> Result<SceneObservation, SceneError> {
    let record_material_slots = snapshot
        .record_ids()
        .map(|id| {
            snapshot
                .record(id)
                .map(|record| record.resources.material().id().slot())
        })
        .collect::<Result<_, _>>()?;
    let record_material_generations = snapshot
        .record_ids()
        .map(|id| {
            snapshot
                .record(id)
                .map(|record| record.resources.material().generation().get())
        })
        .collect::<Result<_, _>>()?;
    let view_extents = snapshot
        .view_ids()
        .map(|id| snapshot.view(id).map(|view| view.extent))
        .collect::<Result<_, _>>()?;
    let SceneDirty { records, views } = snapshot.dirty().clone();
    Ok(SceneObservation {
        version: snapshot.version(),
        record_material_slots,
        record_material_generations,
        view_extents,
        record_dirty: records,
        view_dirty: views,
    })
}

fn sample_record(resources: ExternalResources) -> Result<ExternalRecord, SceneError> {
    Ok(ExternalRecord {
        resources,
        transform: RenderTransform::identity(),
        bounds: unit_bounds()?,
    })
}

fn unit_bounds() -> Result<RenderBounds, SceneError> {
    RenderBounds::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
}

#[derive(Clone, Copy)]
struct AssetReferences {
    triangle_geometry: SceneAssetRef<TriangleGeometryAsset>,
    quad_geometry: SceneAssetRef<QuadGeometryAsset>,
    material_initial: MaterialReference,
    material_replacement: MaterialReference,
}

fn asset_references() -> Result<AssetReferences, Box<dyn std::error::Error>> {
    let triangle_store = AssetStore::<TriangleGeometryAsset, Vec<u8>, String>::new();
    let triangle_handle = triangle_store.create()?;
    let triangle = initial(&triangle_store, &triangle_handle, vec![1, 2, 3], 3)?;
    let quad_store = AssetStore::<QuadGeometryAsset, Vec<u8>, String>::new();
    let quad_handle = quad_store.create()?;
    let quad = initial(&quad_store, &quad_handle, vec![4, 5, 6, 7], 4)?;

    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let material_handle = material_store.create()?;
    let material = initial(
        &material_store,
        &material_handle,
        material_definition(true),
        1,
    )?;
    let replacement = replace(
        &material_store,
        &material_handle,
        material_definition(false),
        1,
    )?;
    if material.id() != replacement.id() || material.generation() >= replacement.generation() {
        return Err("material replacement must retain identity and advance generation".into());
    }
    let initial_variant = MaterialCompiler::compile_for(&material)?;
    let replacement_variant = MaterialCompiler::compile_for(&replacement)?;
    if initial_variant.material()
        != Some(fluxel_material::MaterialAssetRef::from_snapshot(&material))
        || replacement_variant.material()
            != Some(fluxel_material::MaterialAssetRef::from_snapshot(
                &replacement,
            ))
        || initial_variant.key() == replacement_variant.key()
    {
        return Err("S3 material compilation did not preserve exact asset generations".into());
    }
    Ok(AssetReferences {
        triangle_geometry: SceneAssetRef::from_snapshot(&triangle),
        quad_geometry: SceneAssetRef::from_snapshot(&quad),
        material_initial: SceneAssetRef::from_snapshot(&material),
        material_replacement: SceneAssetRef::from_snapshot(&replacement),
    })
}

fn initial<K: AssetKind, V: Send + Sync + 'static>(
    store: &AssetStore<K, V, String>,
    handle: &fluxel_assets::AssetHandle<K>,
    value: V,
    bytes: u64,
) -> Result<AssetSnapshot<K, V>, Box<dyn std::error::Error>> {
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(bytes))?),
        Acquire::Ready(_) | Acquire::Waiter(_) | Acquire::Failed(_) => {
            Err("new asset unexpectedly did not nominate a producer".into())
        }
    }
}

fn replace<K: AssetKind, V: Send + Sync + 'static>(
    store: &AssetStore<K, V, String>,
    handle: &fluxel_assets::AssetHandle<K>,
    value: V,
    bytes: u64,
) -> Result<AssetSnapshot<K, V>, Box<dyn std::error::Error>> {
    match store.request_replacement(handle)? {
        Production::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(bytes))?),
        Production::Waiter(_) => Err("asset replacement unexpectedly waited for production".into()),
    }
}

fn material_definition(enabled: bool) -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: String::from("sample_color"),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: String::from(
                    "fn sample_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0); }",
                ),
            },
            MaterialFragment {
                name: String::from("custom_tint"),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: vec![FragmentCall {
                    fragment: String::from("sample_color"),
                    arguments: Vec::new(),
                }],
                wgsl: String::from(
                    "fn custom_tint() -> vec4<f32> { return sample_color() * tint; }",
                ),
            },
        ],
        "custom_tint",
        vec![MaterialParameter {
            name: String::from("tint"),
            ty: MaterialType::Vec4,
            resource: String::from("material_tint"),
        }],
        vec![
            MaterialResource {
                name: String::from("material_tint"),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation {
                    group: 0,
                    binding: 0,
                },
            },
            MaterialResource {
                name: String::from("material_texture"),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation {
                    group: 0,
                    binding: 1,
                },
            },
        ],
        std::collections::BTreeMap::from([(
            String::from("MATERIAL_ENABLED"),
            StaticValue::Bool(enabled),
        )]),
        SurfaceOutput::Color,
    )
}
