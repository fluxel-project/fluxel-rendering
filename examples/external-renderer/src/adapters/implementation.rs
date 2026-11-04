//! Public adapter entry points and the shared reference execution.

use super::*;
use std::{fs, path::Path};

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
