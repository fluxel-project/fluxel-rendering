use super::adapters::ReferenceAdapter;
use fluxel_renderer::{DirtyRange, FeatureError};
use std::{fs, path::PathBuf, process::Command};

#[test]
fn external_preparation_matches_independent_candidates() {
    super::preparation::run().expect("external preparation consumer");
}

#[test]
fn material_geometry_preparation_uses_real_compilers() {
    super::material_preparation::run().expect("material-backed geometry preparation");
}

#[test]
fn opaque_groups_preserve_layout_material_and_binding_compatibility() {
    use fluxel_renderer::{RenderBounds, RenderChange, RenderRecord, RenderScene, RenderTransform};
    let mut scene = RenderScene::new().unwrap();
    let changes = (0..5u8)
        .map(|value| {
            RenderChange::Insert(RenderRecord {
                resources: value,
                transform: RenderTransform::identity(),
                bounds: RenderBounds::new([0.0; 3], [1.0; 3]).unwrap(),
            })
        })
        .collect();
    let receipt = scene.apply(&changes).unwrap();
    let ids = receipt.inserted();
    // Only records zero and four have the same layout, material and binding.
    let batches = super::opaque::group([
        ((0, 0, 0), ids[4]),
        ((1, 0, 0), ids[1]),
        ((0, 1, 0), ids[2]),
        ((0, 0, 1), ids[3]),
        ((0, 0, 0), ids[0]),
    ]);
    assert_eq!(
        batches,
        vec![
            ((0, 0, 0), vec![ids[0], ids[4]]),
            ((0, 0, 1), vec![ids[3]]),
            ((0, 1, 0), vec![ids[2]]),
            ((1, 0, 0), vec![ids[1]]),
        ]
    );
    assert!(super::opaque::group::<(u8, u8, u8)>([]).is_empty());
}

#[test]
fn ordinary_and_ecs_style_adapters_produce_the_same_public_scene_result() {
    let evidence = test_evidence_root();
    let (direct, ecs) = super::run(&evidence, ReferenceAdapter::expected_evidence())
        .expect("real executable path must validate");

    ReferenceAdapter::assert_equivalent(&direct, &ecs).expect("adapters must agree");
    assert_eq!(direct.initial.version, 1);
    assert_eq!(direct.updated.version, 2);
    assert_eq!(direct.initial.record_material_slots.len(), 2);
    assert_eq!(
        direct.updated.record_material_slots,
        direct.initial.record_material_slots
    );
    assert_eq!(direct.initial.record_material_generations, [1, 1]);
    assert_eq!(direct.updated.record_material_generations, [2, 1]);
    assert_eq!(direct.retained, direct.initial);
    assert_eq!(direct.initial.view_extents, [[640, 360]]);
    assert_eq!(direct.updated.view_extents, [[800, 450]]);
    assert_eq!(
        direct.updated.record_dirty,
        [DirtyRange { first: 0, last: 0 }]
    );
    assert_eq!(
        direct.updated.view_dirty,
        [DirtyRange { first: 0, last: 0 }]
    );
    assert_eq!(direct.features.initial_version, direct.initial.version);
    assert_eq!(direct.features.updated_version, direct.updated.version);
    assert_eq!(direct.features.initial_color_count, 2);
    assert_eq!(direct.features.updated_color, [0, 0, 255, 255]);
    assert_eq!(direct.features.initial_label_count, 1);
    assert_eq!(direct.features.updated_label, "resized-main");
    assert_eq!(direct.features.prepared_color_count, 2);
    assert_eq!(direct.features.prepared_view_count, 1);
    assert_eq!(
        direct.features.version_mismatch,
        FeatureError::SnapshotVersionMismatch {
            feature: "dense-colors",
            scene_version: 2,
            feature_version: 1,
        }
    );
    assert_eq!(
        direct.features.handle_mismatch,
        FeatureError::FeatureHandleMismatch {
            expected: "view-labels",
            actual: "dense-colors",
        }
    );
    assert_eq!(
        direct.features.cross_scene_mismatch,
        FeatureError::ForeignScene
    );
    assert_eq!(
        direct.features.foreign_issuer_mismatch,
        FeatureError::FeatureHandleMismatch {
            expected: "scene-issued feature",
            actual: "dense-colors",
        }
    );
    assert_eq!(
        direct.features.cross_issuer_mismatch,
        FeatureError::FeatureHandleMismatch {
            expected: "scene-issued feature",
            actual: "dense-colors",
        }
    );
    assert_eq!(
        direct.features.storage_issuer_mismatch,
        FeatureError::FeatureHandleMismatch {
            expected: "storage issuer",
            actual: "dense-colors",
        }
    );
    assert_eq!(direct.features.consumed_contributions, 2);
    assert_eq!(
        direct.features.rejected_contribution,
        FeatureError::FeatureHandleMismatch {
            expected: "view-labels",
            actual: "dense-colors",
        }
    );
    assert_eq!(
        direct.features.storage_context_mismatch,
        FeatureError::ForeignIssuer
    );
    assert_eq!(
        direct.features.mixed_version_contribution,
        FeatureError::FeatureHandleMismatch {
            expected: "view-labels",
            actual: "view-labels",
        }
    );
    assert_eq!(
        direct.features.replayed_contribution,
        FeatureError::FeatureHandleMismatch {
            expected: "dense-colors",
            actual: "dense-colors",
        }
    );

    let mut incorrect = ReferenceAdapter::expected_evidence();
    incorrect.updated_version = 99;
    let failed = evidence.join("real-entry-oracle-failure");
    assert!(super::run(&failed, incorrect).is_err());
    for file in ["input.txt", "expected.txt", "actual.txt", "error.txt"] {
        assert!(
            failed.join("object-adapter").join(file).is_file(),
            "failed oracle must retain {file}"
        );
    }
    let error = fs::read_to_string(failed.join("object-adapter/error.txt"))
        .expect("failure record must be readable");
    assert!(error.contains("oracle mismatch"));
}

fn test_evidence_root() -> PathBuf {
    std::env::temp_dir().join(format!("external-renderer-evidence-{}", std::process::id()))
}

#[test]
fn external_code_cannot_access_renderer_private_graph_state() {
    assert_compile_fails("private-graph", "E0603", "module `graph` is private");
}

#[test]
fn external_code_cannot_construct_a_phase_context_or_submit_work() {
    assert_compile_fails("private-phase-context", "E0624", "associated function `new` is private");
}

fn assert_compile_fails(case: &str, code: &str, message: &str) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("compile-fail")
        .join(case)
        .join("Cargo.toml");
    let target = std::env::temp_dir().join(format!(
        "external-renderer-private-graph-{}",
        std::process::id()
    ));
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args([
            "check",
            "--manifest-path",
            manifest.to_str().expect("fixture manifest path is Unicode"),
            "--locked",
            "--target-dir",
            target.to_str().expect("temporary target path is Unicode"),
        ])
        .output()
        .expect("compile-fail fixture must launch cargo");
    assert!(!output.status.success(), "private module fixture compiled");
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostics.contains(code) && diagnostics.contains(message),
        "fixture failed for an unrelated reason:\n{diagnostics}"
    );
}
