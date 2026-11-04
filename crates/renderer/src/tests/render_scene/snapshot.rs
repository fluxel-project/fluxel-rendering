//! Snapshot persistence tests.
use super::{
    RenderBounds, RenderChange, RenderRecord, RenderResources, RenderScene, RenderTransform,
    SceneAssetRef,
};
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};

struct TestGeometry;
impl fluxel_assets::AssetKind for TestGeometry {}

struct TestMaterial;
impl fluxel_assets::AssetKind for TestMaterial {}

type TestResources = RenderResources<TestGeometry, TestMaterial>;
type TestScene = RenderScene<TestResources>;
type TestRecord = RenderRecord<TestResources>;

fn record(geometry: u64) -> TestRecord {
    RenderRecord {
        resources: RenderResources {
            geometry: geometry_reference(geometry),
            material: material_reference(geometry + 100),
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.0; 3], [1.0; 3]).expect("valid bounds"),
    }
}

fn geometry_reference(value: u64) -> SceneAssetRef<TestGeometry> {
    let store = AssetStore::<TestGeometry, Vec<u8>, ()>::new();
    let handle = store.create().expect("geometry handle");
    let snapshot = snapshot(&store, &handle, value);
    SceneAssetRef::from_snapshot(&snapshot)
}

fn material_reference(value: u64) -> SceneAssetRef<TestMaterial> {
    let store = AssetStore::<TestMaterial, Vec<u8>, ()>::new();
    let handle = store.create().expect("material handle");
    let snapshot = snapshot(&store, &handle, value);
    SceneAssetRef::from_snapshot(&snapshot)
}

fn snapshot<K: fluxel_assets::AssetKind>(
    store: &AssetStore<K, Vec<u8>, ()>,
    handle: &fluxel_assets::AssetHandle<K>,
    value: u64,
) -> AssetSnapshot<K, Vec<u8>> {
    match store.acquire(handle).expect("asset producer") {
        Acquire::Producer(producer) => producer
            .commit(value.to_le_bytes().to_vec(), ResidentBytes::new(8))
            .expect("asset snapshot"),
        _ => unreachable!("new asset must produce"),
    }
}

#[test]
fn snapshot_reuses_unaffected_chunks() {
    let mut scene = TestScene::new().expect("scene");
    let changes = (0..193)
        .map(|index| RenderChange::Insert(record(index)))
        .collect();
    let receipt = scene.apply(&changes).expect("insert");
    let first = scene.snapshot();
    let changed = receipt.inserted()[192];

    let updated = record(1000);
    let original = first.record(changed).expect("old").clone();
    scene
        .apply(
            &[RenderChange::Update(changed, updated.clone())]
                .into_iter()
                .collect(),
        )
        .expect("update");
    let second = scene.snapshot();

    assert!(std::sync::Arc::ptr_eq(
        super::snapshot_chunk(&first.records, 0).expect("first unchanged chunk"),
        super::snapshot_chunk(&second.records, 0).expect("second unchanged chunk")
    ));
    assert!(!std::sync::Arc::ptr_eq(
        super::snapshot_chunk(&first.records, 3).expect("first changed chunk"),
        super::snapshot_chunk(&second.records, 3).expect("second changed chunk")
    ));
    assert_eq!(first.record(changed).expect("old"), &original);
    assert_eq!(second.record(changed).expect("new"), &updated);
}

#[test]
fn persistent_snapshot_directory_copies_only_the_changed_path() {
    let empty_chunk = || std::sync::Arc::new(Vec::<Option<super::SnapshotEntry<()>>>::new());
    let first = Some(super::replace_snapshot_chunk(None, 2, 0, empty_chunk()));
    let sparse = Some(super::replace_snapshot_chunk(
        first.as_ref(),
        2,
        2,
        empty_chunk(),
    ));
    let updated = Some(super::replace_snapshot_chunk(
        sparse.as_ref(),
        2,
        0,
        empty_chunk(),
    ));

    let (old_right, new_right) = match (sparse.as_deref(), updated.as_deref()) {
        (
            Some(super::SnapshotTree::Branch {
                right: old_right, ..
            }),
            Some(super::SnapshotTree::Branch {
                right: new_right, ..
            }),
        ) => (
            old_right.as_ref().expect("sparse right branch"),
            new_right.as_ref().expect("retained right branch"),
        ),
        _ => panic!("persistent root branches"),
    };
    assert!(std::sync::Arc::ptr_eq(old_right, new_right));
}

#[test]
fn retained_snapshot_is_concurrently_readable_while_scene_changes() {
    let mut scene = TestScene::new().expect("scene");
    let original = record(7);
    let id = scene
        .apply(
            &[RenderChange::Insert(original.clone())]
                .into_iter()
                .collect(),
        )
        .expect("insert")
        .inserted()[0];
    let old = scene.snapshot();

    std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            for _ in 0..1_000 {
                assert_eq!(old.record(id).expect("retained value"), &original);
            }
        });
        for geometry in 8..108 {
            scene
                .apply(
                    &[RenderChange::Update(id, record(geometry))]
                        .into_iter()
                        .collect(),
                )
                .expect("update");
        }
        reader.join().expect("reader must not panic");
    });
    assert_eq!(old.record(id).expect("retained value"), &original);
}
