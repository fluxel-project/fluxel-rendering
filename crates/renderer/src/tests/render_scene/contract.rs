use super::{
    RenderBounds, RenderChange, RenderRecord, RenderResources, RenderScene, RenderTransform,
    RenderView, SceneAssetRef, SceneCapacity, SceneError,
};
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, Production, ResidentBytes};

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

fn replacement<K: fluxel_assets::AssetKind>(
    store: &AssetStore<K, Vec<u8>, ()>,
    handle: &fluxel_assets::AssetHandle<K>,
    value: u64,
) -> AssetSnapshot<K, Vec<u8>> {
    match store
        .request_replacement(handle)
        .expect("replacement producer")
    {
        Production::Producer(producer) => producer
            .commit(value.to_le_bytes().to_vec(), ResidentBytes::new(8))
            .expect("replacement snapshot"),
        Production::Waiter(_) => unreachable!("single test caller must produce"),
    }
}

fn view(extent: [u32; 2]) -> RenderView {
    RenderView::new(RenderTransform::identity(), extent).expect("valid view")
}

#[test]
fn batch_adds_multiple_records_and_views() {
    let mut scene = TestScene::new().expect("scene");
    let second_record = record(11);
    let receipt = scene
        .apply(
            &[
                RenderChange::Insert(record(10)),
                RenderChange::Insert(second_record.clone()),
                RenderChange::InsertView(view([320, 180])),
                RenderChange::InsertView(view([640, 360])),
            ]
            .into_iter()
            .collect(),
        )
        .expect("batch");
    assert_eq!(scene.len(), 2);
    assert_eq!(scene.view_len(), 2);
    assert_eq!(receipt.inserted().len(), 2);
    assert_eq!(receipt.inserted_views().len(), 2);
    assert_eq!(
        scene.record(receipt.inserted()[1]).expect("record"),
        &second_record
    );
    assert_eq!(
        scene
            .view(receipt.inserted_views()[0])
            .expect("view")
            .extent,
        [320, 180]
    );
}

#[test]
fn empty_batch_is_a_noop() {
    let mut scene = TestScene::new().expect("scene");
    let receipt = scene.apply(&Default::default()).expect("empty batch");
    assert!(receipt.inserted().is_empty());
    assert!(scene.is_empty());
    assert_eq!(scene.view_len(), 0);
}

#[test]
fn stale_identity_cannot_address_a_reused_slot() {
    let mut scene = TestScene::new().expect("scene");
    let original = scene
        .apply(&[RenderChange::Insert(record(1))].into_iter().collect())
        .expect("insert")
        .inserted()[0];
    scene
        .apply(&[RenderChange::Remove(original)].into_iter().collect())
        .expect("remove");
    let replacement = scene
        .apply(&[RenderChange::Insert(record(2))].into_iter().collect())
        .expect("replacement")
        .inserted()[0];
    assert_eq!(original.slot(), replacement.slot());
    assert_ne!(original.generation(), replacement.generation());
    assert_eq!(
        scene.record(original),
        Err(SceneError::StaleId { operation: 0 })
    );
}

#[test]
fn removing_a_record_compacts_dense_storage_without_changing_stable_ids() {
    let mut scene = TestScene::new().expect("scene");
    let records = [record(1), record(2), record(3)];
    let changes = records.iter().cloned().map(RenderChange::Insert).collect();
    let receipt = scene.apply(&changes).expect("insert");
    let old = scene.snapshot();
    let removed = receipt.inserted()[1];
    let moved = receipt.inserted()[2];

    scene
        .apply(&[RenderChange::Remove(removed)].into_iter().collect())
        .expect("remove");

    assert_eq!(scene.record_values.len(), 2);
    assert_eq!(scene.record_slots.len(), 2);
    assert_eq!(scene.records[moved.slot() as usize].value, Some(1));
    assert_eq!(scene.record(moved).expect("moved record"), &records[2]);
    assert_eq!(old.record(moved).expect("old snapshot record"), &records[2]);
    assert_eq!(
        scene.snapshot().dirty().records,
        vec![super::DirtyRange {
            first: removed.slot(),
            last: moved.slot(),
        }]
    );
}

#[test]
fn dense_remove_of_tail_marks_only_the_deleted_slot() {
    let mut scene = TestScene::new().expect("scene");
    let receipt = scene
        .apply(
            &[
                RenderChange::Insert(record(1)),
                RenderChange::Insert(record(2)),
            ]
            .into_iter()
            .collect(),
        )
        .expect("insert");
    let tail = receipt.inserted()[1];
    scene
        .apply(&[RenderChange::Remove(tail)].into_iter().collect())
        .expect("remove tail");
    assert_eq!(
        scene.snapshot().dirty().records,
        vec![super::DirtyRange {
            first: tail.slot(),
            last: tail.slot(),
        }]
    );
}

#[test]
fn dense_remove_then_update_of_moved_record_is_atomic_and_marks_both_slots() {
    let mut scene = TestScene::new().expect("scene");
    let receipt = scene
        .apply(
            &[
                RenderChange::Insert(record(1)),
                RenderChange::Insert(record(2)),
            ]
            .into_iter()
            .collect(),
        )
        .expect("insert");
    let deleted = receipt.inserted()[0];
    let moved = receipt.inserted()[1];
    let updated = record(20);
    scene
        .apply(
            &[
                RenderChange::Remove(deleted),
                RenderChange::Update(moved, updated.clone()),
            ]
            .into_iter()
            .collect(),
        )
        .expect("batch");
    assert_eq!(scene.record(moved).expect("updated moved record"), &updated);
    assert_eq!(
        scene.snapshot().dirty().records,
        vec![super::DirtyRange { first: 0, last: 1 }]
    );
}

#[test]
fn cross_chunk_compaction_marks_deleted_and_moved_slots_without_mutating_old_snapshot() {
    let mut scene = TestScene::new().expect("scene");
    let changes = (0..65)
        .map(|value| RenderChange::Insert(record(value)))
        .collect();
    let receipt = scene.apply(&changes).expect("insert");
    let deleted = receipt.inserted()[0];
    let moved = receipt.inserted()[64];
    let old = scene.snapshot();

    scene
        .apply(&[RenderChange::Remove(deleted)].into_iter().collect())
        .expect("remove");
    let current = scene.snapshot();

    assert_eq!(
        scene.record(moved).expect("moved"),
        old.record(moved).expect("old moved")
    );
    assert_eq!(old.record_ids().count(), 65);
    assert_eq!(current.record_ids().count(), 64);
    assert_eq!(
        current.dirty().records,
        vec![
            super::DirtyRange { first: 0, last: 0 },
            super::DirtyRange {
                first: 64,
                last: 64,
            },
        ]
    );
}

#[test]
fn foreign_or_conflicting_batch_is_atomic() {
    let mut first = TestScene::new().expect("first");
    let foreign = first
        .apply(&[RenderChange::Insert(record(1))].into_iter().collect())
        .expect("insert")
        .inserted()[0];
    let mut scene = TestScene::new().expect("second");
    let original = record(2);
    let id = scene
        .apply(
            &[RenderChange::Insert(original.clone())]
                .into_iter()
                .collect(),
        )
        .expect("insert")
        .inserted()[0];
    let result = scene.apply(
        &[
            RenderChange::Update(id, record(3)),
            RenderChange::Remove(foreign),
        ]
        .into_iter()
        .collect(),
    );
    assert_eq!(result, Err(SceneError::ForeignId { operation: 1 }));
    assert_eq!(scene.record(id).expect("unchanged"), &original);

    let result = scene.apply(
        &[
            RenderChange::Update(id, record(3)),
            RenderChange::Remove(id),
        ]
        .into_iter()
        .collect(),
    );
    assert_eq!(result, Err(SceneError::ConflictingChange { operation: 1 }));
    assert_eq!(scene.record(id).expect("unchanged"), &original);

    let result = scene.apply(
        &[
            RenderChange::Remove(id),
            RenderChange::Update(id, record(4)),
        ]
        .into_iter()
        .collect(),
    );
    assert_eq!(result, Err(SceneError::ConflictingChange { operation: 1 }));
    assert_eq!(scene.record(id).expect("unchanged"), &original);
}

#[test]
fn capacities_reject_a_whole_batch() {
    let mut scene = TestScene::with_capacity(SceneCapacity::new(1, 1)).expect("scene");
    let result = scene.apply(
        &[
            RenderChange::Insert(record(1)),
            RenderChange::Insert(record(2)),
        ]
        .into_iter()
        .collect(),
    );
    assert_eq!(
        result,
        Err(SceneError::RecordCapacityExceeded { operation: 1 })
    );
    assert!(scene.is_empty());
    let result = scene.apply(
        &[
            RenderChange::InsertView(view([1, 1])),
            RenderChange::InsertView(view([2, 2])),
        ]
        .into_iter()
        .collect(),
    );
    assert_eq!(
        result,
        Err(SceneError::ViewCapacityExceeded { operation: 1 })
    );
    assert_eq!(scene.view_len(), 0);
}

#[test]
fn invalid_adapter_values_are_refused() {
    assert_eq!(
        RenderTransform::from_columns([[f32::NAN, 0.0, 0.0, 0.0]; 4]),
        Err(SceneError::InvalidTransform)
    );
    assert_eq!(
        RenderBounds::new([1.0, 0.0, 0.0], [0.0, 1.0, 1.0]),
        Err(SceneError::InvalidBounds)
    );
    assert_eq!(
        RenderView::new(RenderTransform::identity(), [0, 1]),
        Err(SceneError::InvalidView)
    );

    let mut scene = TestScene::new().expect("scene");
    assert_eq!(
        scene.apply(
            &[RenderChange::InsertView(RenderView {
                transform: RenderTransform::identity(),
                extent: [0, 1],
            })]
            .into_iter()
            .collect(),
        ),
        Err(SceneError::InvalidView)
    );
    assert_eq!(scene.view_len(), 0);
}

#[test]
fn resources_pin_logical_asset_identity_and_content_generation() {
    let geometry_store = AssetStore::<TestGeometry, Vec<u8>, ()>::new();
    let geometry_handle = geometry_store.create().expect("geometry handle");
    let geometry_first = snapshot(&geometry_store, &geometry_handle, 1);
    let geometry_second = replacement(&geometry_store, &geometry_handle, 2);
    let material_store = AssetStore::<TestMaterial, Vec<u8>, ()>::new();
    let material_handle = material_store.create().expect("material handle");
    let material = snapshot(&material_store, &material_handle, 3);

    let first = SceneAssetRef::from_snapshot(&geometry_first);
    let second = SceneAssetRef::from_snapshot(&geometry_second);
    let material = SceneAssetRef::from_snapshot(&material);
    assert_eq!(first.id(), second.id());
    assert!(first.generation() < second.generation());

    let mut scene = TestScene::new().expect("scene");
    let id = scene
        .apply(
            &[RenderChange::Insert(RenderRecord {
                resources: RenderResources {
                    geometry: first,
                    material,
                },
                transform: RenderTransform::identity(),
                bounds: RenderBounds::new([-1.0; 3], [1.0; 3]).expect("bounds"),
            })]
            .into_iter()
            .collect(),
        )
        .expect("insert")
        .inserted()[0];
    let old = scene.snapshot();
    scene
        .apply(
            &[RenderChange::Update(
                id,
                RenderRecord {
                    resources: RenderResources {
                        geometry: second,
                        material,
                    },
                    transform: RenderTransform::identity(),
                    bounds: RenderBounds::new([-1.0; 3], [1.0; 3]).expect("bounds"),
                },
            )]
            .into_iter()
            .collect(),
        )
        .expect("update");

    assert_eq!(
        old.record(id).expect("old record").resources.geometry,
        first
    );
    assert_eq!(
        scene.record(id).expect("new record").resources.geometry,
        second
    );
}

#[test]
fn generation_exhaustion_does_not_wrap() {
    let mut scene = TestScene::new().expect("scene");
    scene.records.push(super::Slot {
        generation: u32::MAX,
        value: None,
    });
    scene.record_free.insert(0);
    let result = scene.apply(&[RenderChange::Insert(record(1))].into_iter().collect());
    assert_eq!(result, Err(SceneError::IdentityExhausted { operation: 0 }));
    assert!(scene.is_empty());
}

#[test]
fn snapshot_version_exhaustion_rejects_the_whole_batch() {
    let mut scene = TestScene::new().expect("scene");
    scene.version = u64::MAX;
    assert_eq!(
        scene.apply(&[RenderChange::Insert(record(1))].into_iter().collect()),
        Err(SceneError::SnapshotVersionExhausted)
    );
    assert!(scene.is_empty());
}

#[test]
fn snapshot_retains_old_values_and_publishes_precise_dirty_ranges() {
    let mut scene = TestScene::new().expect("scene");
    let original = record(2);
    let receipt = scene
        .apply(
            &[
                RenderChange::Insert(record(1)),
                RenderChange::Insert(original.clone()),
            ]
            .into_iter()
            .collect(),
        )
        .expect("insert");
    let first = scene.snapshot();
    let second_id = receipt.inserted()[1];
    let updated = record(20);
    scene
        .apply(
            &[RenderChange::Update(second_id, updated.clone())]
                .into_iter()
                .collect(),
        )
        .expect("update");
    let second = scene.snapshot();

    assert_eq!(first.version(), 1);
    assert_eq!(second.version(), 2);
    assert_eq!(first.record(second_id).expect("old value"), &original);
    assert_eq!(second.record(second_id).expect("new value"), &updated);
    assert_eq!(
        second.dirty().records,
        vec![super::DirtyRange { first: 1, last: 1 }]
    );
    assert!(second.dirty().views.is_empty());
    assert_eq!(second.record_ids().collect::<Vec<_>>(), receipt.inserted());
}

#[test]
fn view_conflicts_are_order_independent_and_do_not_change_snapshots() {
    let mut scene = TestScene::new().expect("scene");
    let view_id = scene
        .apply(
            &[RenderChange::InsertView(view([40, 30]))]
                .into_iter()
                .collect(),
        )
        .expect("insert")
        .inserted_views()[0];
    let before = scene.snapshot();
    let result = scene.apply(
        &[
            RenderChange::RemoveView(view_id),
            RenderChange::UpdateView(view_id, view([80, 60])),
        ]
        .into_iter()
        .collect(),
    );

    assert_eq!(result, Err(SceneError::ConflictingChange { operation: 1 }));
    assert_eq!(scene.snapshot().version(), before.version());
    assert_eq!(before.view(view_id).expect("old view").extent, [40, 30]);
}
