//! Render-only records and transactional scene changes.
//!
//! The scene owns no gameplay identity, ECS query, GPU object, or Graph work.

use std::{
    collections::{BTreeSet, HashSet},
    fmt,
    ops::Bound::{Excluded, Unbounded},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use fluxel_assets::{AssetId, AssetKind, AssetSnapshot, ContentGeneration};

static NEXT_SCENE: AtomicU64 = AtomicU64::new(1);
const SNAPSHOT_CHUNK_SIZE: usize = 64;
const SNAPSHOT_TREE_DEPTH: u8 = 26;

/// An identity issued by one scene for one render record.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RenderId {
    scene: u64,
    slot: u32,
    generation: u32,
}

impl RenderId {
    /// Returns the record slot.
    pub fn slot(self) -> u32 {
        self.slot
    }
    /// Returns the slot reuse generation.
    pub fn generation(self) -> u32 {
        self.generation
    }
}

/// An identity issued by one scene for one view.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RenderViewId {
    scene: u64,
    slot: u32,
    generation: u32,
}

impl RenderViewId {
    /// Returns the view slot.
    pub fn slot(self) -> u32 {
        self.slot
    }
    /// Returns the slot reuse generation.
    pub fn generation(self) -> u32 {
        self.generation
    }
}

/// A finite, resolved transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderTransform([[f32; 4]; 4]);

impl RenderTransform {
    /// Creates a transform from a column-major matrix.
    pub fn from_columns(columns: [[f32; 4]; 4]) -> Result<Self, SceneError> {
        if columns.iter().flatten().all(|value| value.is_finite()) {
            Ok(Self(columns))
        } else {
            Err(SceneError::InvalidTransform)
        }
    }
    /// Returns an identity transform.
    pub fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
    /// Returns the stored columns.
    pub fn columns(self) -> [[f32; 4]; 4] {
        self.0
    }
}

/// A finite axis-aligned bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderBounds {
    minimum: [f32; 3],
    maximum: [f32; 3],
}

impl RenderBounds {
    /// Creates non-inverted finite bounds.
    pub fn new(minimum: [f32; 3], maximum: [f32; 3]) -> Result<Self, SceneError> {
        if minimum
            .iter()
            .chain(maximum.iter())
            .all(|value| value.is_finite())
            && minimum
                .iter()
                .zip(maximum)
                .all(|(minimum, maximum)| minimum <= &maximum)
        {
            Ok(Self { minimum, maximum })
        } else {
            Err(SceneError::InvalidBounds)
        }
    }
    /// Returns the lower corner.
    pub fn minimum(self) -> [f32; 3] {
        self.minimum
    }
    /// Returns the upper corner.
    pub fn maximum(self) -> [f32; 3] {
        self.maximum
    }
}

/// A pinned logical asset identity and immutable content generation.
pub struct SceneAssetRef<K: AssetKind> {
    id: AssetId<K>,
    generation: ContentGeneration,
}

impl<K: AssetKind> Copy for SceneAssetRef<K> {}

impl<K: AssetKind> Clone for SceneAssetRef<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: AssetKind> fmt::Debug for SceneAssetRef<K> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SceneAssetRef")
            .field("id", &self.id)
            .field("generation", &self.generation)
            .finish()
    }
}

impl<K: AssetKind> PartialEq for SceneAssetRef<K> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.generation == other.generation
    }
}

impl<K: AssetKind> Eq for SceneAssetRef<K> {}

impl<K: AssetKind> SceneAssetRef<K> {
    /// Pins the logical identity and generation of an asset snapshot.
    pub fn from_snapshot<V>(snapshot: &AssetSnapshot<K, V>) -> Self {
        Self {
            id: snapshot.id(),
            generation: snapshot.generation(),
        }
    }

    /// Returns the logical asset identity.
    pub const fn id(self) -> AssetId<K> {
        self.id
    }

    /// Returns the exact asset content generation.
    pub const fn generation(self) -> ContentGeneration {
        self.generation
    }
}

/// A convenience resource bundle with pinned geometry and material references.
///
/// Consumers that need additional asset kinds may use their own `Clone` resource
/// bundle for `RenderRecord` and `RenderScene` instead.
pub struct RenderResources<G: AssetKind, M: AssetKind> {
    /// The exact geometry generation used by the record.
    pub geometry: SceneAssetRef<G>,
    /// The exact material generation used by the record.
    pub material: SceneAssetRef<M>,
}

impl<G: AssetKind, M: AssetKind> Copy for RenderResources<G, M> {}

impl<G: AssetKind, M: AssetKind> Clone for RenderResources<G, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: AssetKind, M: AssetKind> fmt::Debug for RenderResources<G, M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RenderResources")
            .field("geometry", &self.geometry)
            .field("material", &self.material)
            .finish()
    }
}

impl<G: AssetKind, M: AssetKind> PartialEq for RenderResources<G, M> {
    fn eq(&self, other: &Self) -> bool {
        self.geometry == other.geometry && self.material == other.material
    }
}

impl<G: AssetKind, M: AssetKind> Eq for RenderResources<G, M> {}

/// One render-only object record with a consumer-owned resource bundle.
pub struct RenderRecord<R> {
    /// Logical inputs resolved later by preparation.
    pub resources: R,
    /// The resolved transform.
    pub transform: RenderTransform,
    /// The resolved bound.
    pub bounds: RenderBounds,
}

impl<R: Clone> Clone for RenderRecord<R> {
    fn clone(&self) -> Self {
        Self {
            resources: self.resources.clone(),
            transform: self.transform,
            bounds: self.bounds,
        }
    }
}

impl<R: fmt::Debug> fmt::Debug for RenderRecord<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RenderRecord")
            .field("resources", &self.resources)
            .field("transform", &self.transform)
            .field("bounds", &self.bounds)
            .finish()
    }
}

impl<R: PartialEq> PartialEq for RenderRecord<R> {
    fn eq(&self, other: &Self) -> bool {
        self.resources == other.resources
            && self.transform == other.transform
            && self.bounds == other.bounds
    }
}

/// One adapter-owned camera or view record.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderView {
    /// The view transform.
    pub transform: RenderTransform,
    /// The nonempty output extent.
    pub extent: [u32; 2],
}

impl RenderView {
    /// Creates a view with a nonempty extent.
    pub fn new(transform: RenderTransform, extent: [u32; 2]) -> Result<Self, SceneError> {
        if extent.contains(&0) {
            Err(SceneError::InvalidView)
        } else {
            Ok(Self { transform, extent })
        }
    }
}

/// Live scene capacity limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneCapacity {
    /// Maximum live records.
    pub records: usize,
    /// Maximum live views.
    pub views: usize,
}

impl SceneCapacity {
    /// Creates explicit record and view limits.
    pub const fn new(records: usize, views: usize) -> Self {
        Self { records, views }
    }
}

impl Default for SceneCapacity {
    fn default() -> Self {
        Self::new(usize::MAX, usize::MAX)
    }
}

/// One atomic batch operation.
#[derive(Clone, Debug, PartialEq)]
pub enum RenderChange<R> {
    /// Inserts a record.
    Insert(RenderRecord<R>),
    /// Replaces a live record.
    Update(RenderId, RenderRecord<R>),
    /// Removes a live record.
    Remove(RenderId),
    /// Inserts a view.
    InsertView(RenderView),
    /// Replaces a live view.
    UpdateView(RenderViewId, RenderView),
    /// Removes a live view.
    RemoveView(RenderViewId),
}

/// A submission-ordered batch of scene operations.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderChangeSet<R>(Vec<RenderChange<R>>);

impl<R> Default for RenderChangeSet<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> RenderChangeSet<R> {
    /// Creates an empty batch.
    pub const fn new() -> Self {
        Self(Vec::new())
    }
    /// Adds an operation.
    pub fn push(&mut self, change: RenderChange<R>) {
        self.0.push(change);
    }
    /// Returns the ordered operations.
    pub fn changes(&self) -> &[RenderChange<R>] {
        &self.0
    }
    /// Returns whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<R> FromIterator<RenderChange<R>> for RenderChangeSet<R> {
    fn from_iter<T: IntoIterator<Item = RenderChange<R>>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// Identities allocated by one successful batch.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChangeReceipt {
    inserted: Vec<RenderId>,
    views: Vec<RenderViewId>,
}

impl ChangeReceipt {
    /// Returns record IDs in insertion order.
    pub fn inserted(&self) -> &[RenderId] {
        &self.inserted
    }
    /// Returns view IDs in insertion order.
    pub fn inserted_views(&self) -> &[RenderViewId] {
        &self.views
    }
}

/// One inclusive range of slots changed by a scene version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirtyRange {
    /// First changed slot.
    pub first: u32,
    /// Last changed slot, inclusive.
    pub last: u32,
}

/// Exact slot ranges changed by one published scene version.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SceneDirty {
    /// Render-record slot ranges changed in this version.
    pub records: Vec<DirtyRange>,
    /// View slot ranges changed in this version.
    pub views: Vec<DirtyRange>,
}

#[derive(Clone)]
struct SnapshotEntry<T> {
    slot: u32,
    generation: u32,
    value: T,
}

type SnapshotChunk<T> = Arc<Vec<Option<SnapshotEntry<T>>>>;
type SnapshotRoot<T> = Option<Arc<SnapshotTree<T>>>;

enum SnapshotTree<T> {
    Branch {
        left: SnapshotRoot<T>,
        right: SnapshotRoot<T>,
    },
    Leaf(SnapshotChunk<T>),
}

/// An immutable render-scene view published after a successful batch.
#[derive(Clone)]
pub struct FrameSnapshot<R> {
    scene: u64,
    version: u64,
    records: SnapshotRoot<RenderRecord<R>>,
    views: SnapshotRoot<RenderView>,
    dirty: SceneDirty,
}

impl<R> FrameSnapshot<R> {
    pub(crate) const fn provenance(&self) -> u64 {
        self.scene
    }
    /// Returns this snapshot's monotonically increasing version.
    pub fn version(&self) -> u64 {
        self.version
    }
    /// Returns exact slots changed while publishing this version.
    pub fn dirty(&self) -> &SceneDirty {
        &self.dirty
    }
    /// Returns live record identities in deterministic slot order.
    pub fn record_ids(&self) -> impl Iterator<Item = RenderId> + '_ {
        snapshot_ids(&self.records, self.scene, RenderId::from_entry).into_iter()
    }
    /// Returns live view identities in deterministic slot order.
    pub fn view_ids(&self) -> impl Iterator<Item = RenderViewId> + '_ {
        snapshot_ids(&self.views, self.scene, RenderViewId::from_entry).into_iter()
    }
    /// Looks up a record without observing later scene changes.
    pub fn record(&self, id: RenderId) -> Result<&RenderRecord<R>, SceneError> {
        self.lookup(&self.records, id.scene, id.slot, id.generation)
    }
    /// Looks up a view without observing later scene changes.
    pub fn view(&self, id: RenderViewId) -> Result<&RenderView, SceneError> {
        self.lookup(&self.views, id.scene, id.slot, id.generation)
    }
    fn lookup<'a, T>(
        &self,
        root: &'a SnapshotRoot<T>,
        scene: u64,
        slot: u32,
        generation: u32,
    ) -> Result<&'a T, SceneError> {
        if scene != self.scene {
            return Err(SceneError::ForeignId { operation: 0 });
        }
        let index = usize::try_from(slot).expect("u32 fits usize");
        let Some(entry) = snapshot_chunk(root, index / SNAPSHOT_CHUNK_SIZE)
            .and_then(|chunk| chunk.get(index % SNAPSHOT_CHUNK_SIZE))
            .and_then(Option::as_ref)
            .filter(|entry| entry.generation == generation)
        else {
            return Err(SceneError::StaleId { operation: 0 });
        };
        Ok(&entry.value)
    }
}

impl RenderId {
    fn from_entry<T>(scene: u64, entry: &SnapshotEntry<T>) -> Self {
        Self {
            scene,
            slot: entry.slot,
            generation: entry.generation,
        }
    }
}

impl RenderViewId {
    fn from_entry<T>(scene: u64, entry: &SnapshotEntry<T>) -> Self {
        Self {
            scene,
            slot: entry.slot,
            generation: entry.generation,
        }
    }
}

/// A structured scene validation refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SceneError {
    /// A transform contains a non-finite component.
    InvalidTransform,
    /// Bounds are non-finite or inverted.
    InvalidBounds,
    /// A view has an empty extent.
    InvalidView,
    /// An ID came from another scene.
    ForeignId {
        /// Index of the rejected operation in the submitted batch.
        operation: usize,
    },
    /// An ID was removed, reused, or was never issued.
    StaleId {
        /// Index of the rejected operation in the submitted batch.
        operation: usize,
    },
    /// A later operation writes an ID already written by the batch.
    ConflictingChange {
        /// Index of the later conflicting operation in the submitted batch.
        operation: usize,
    },
    /// A batch exceeds record capacity.
    RecordCapacityExceeded {
        /// Index of the operation that would exceed capacity.
        operation: usize,
    },
    /// A batch exceeds view capacity.
    ViewCapacityExceeded {
        /// Index of the operation that would exceed capacity.
        operation: usize,
    },
    /// A generation cannot advance without wrapping.
    IdentityExhausted {
        /// Index of the operation that would require generation wraparound.
        operation: usize,
    },
    /// The process-local scene identity source is exhausted.
    SceneIdentityExhausted,
    /// The scene cannot issue another feature issuer token.
    FeatureIssuerExhausted,
    /// A nonempty batch cannot be assigned a later snapshot version.
    SnapshotVersionExhausted,
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransform => f.write_str("render transform contains a non-finite value"),
            Self::InvalidBounds => f.write_str("render bounds are non-finite or inverted"),
            Self::InvalidView => f.write_str("render view has an empty extent"),
            Self::ForeignId { operation } => {
                write!(f, "change {operation} names an ID from another scene")
            }
            Self::StaleId { operation } => write!(f, "change {operation} names a stale ID"),
            Self::ConflictingChange { operation } => {
                write!(f, "change {operation} conflicts with an earlier change")
            }
            Self::RecordCapacityExceeded { operation } => {
                write!(f, "change {operation} exceeds record capacity")
            }
            Self::ViewCapacityExceeded { operation } => {
                write!(f, "change {operation} exceeds view capacity")
            }
            Self::IdentityExhausted { operation } => {
                write!(f, "change {operation} exhausts an identity generation")
            }
            Self::SceneIdentityExhausted => {
                f.write_str("render scene identity source is exhausted")
            }
            Self::FeatureIssuerExhausted => {
                f.write_str("render scene feature issuer source is exhausted")
            }
            Self::SnapshotVersionExhausted => {
                f.write_str("render scene snapshot versions are exhausted")
            }
        }
    }
}

impl std::error::Error for SceneError {}

#[derive(Clone)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

impl<T> Slot<T> {
    fn vacant() -> Self {
        Self {
            generation: 0,
            value: None,
        }
    }
}

/// Mutable scene state with transactional batch updates.
///
/// `R` is one static resource bundle per record, not an individual asset kind.
pub struct RenderScene<R: Clone> {
    scene: u64,
    next_feature_issuer: AtomicU64,
    capacity: SceneCapacity,
    records: Vec<Slot<usize>>,
    record_values: Vec<RenderRecord<R>>,
    record_slots: Vec<usize>,
    views: Vec<Slot<RenderView>>,
    record_free: BTreeSet<usize>,
    view_free: BTreeSet<usize>,
    record_count: usize,
    view_count: usize,
    version: u64,
    snapshot_records: SnapshotRoot<RenderRecord<R>>,
    snapshot_views: SnapshotRoot<RenderView>,
    snapshot_dirty: SceneDirty,
}

impl<R: Clone> RenderScene<R> {
    /// Returns the controlled issuer for CPU-only feature handles.
    pub fn feature_issuer(&self) -> Result<crate::FeatureIssuer, SceneError> {
        let issuer = self
            .next_feature_issuer
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| SceneError::FeatureIssuerExhausted)?;
        Ok(crate::FeatureIssuer::new(self.scene, issuer))
    }
    /// Creates a scene with no practical capacity limit.
    pub fn new() -> Result<Self, SceneError> {
        Self::with_capacity(SceneCapacity::default())
    }
    /// Creates a scene with explicit capacity limits.
    pub fn with_capacity(capacity: SceneCapacity) -> Result<Self, SceneError> {
        let scene = NEXT_SCENE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| SceneError::SceneIdentityExhausted)?;
        Ok(Self {
            scene,
            next_feature_issuer: AtomicU64::new(1),
            capacity,
            records: Vec::new(),
            record_values: Vec::new(),
            record_slots: Vec::new(),
            views: Vec::new(),
            record_free: BTreeSet::new(),
            view_free: BTreeSet::new(),
            record_count: 0,
            view_count: 0,
            version: 0,
            snapshot_records: None,
            snapshot_views: None,
            snapshot_dirty: SceneDirty::default(),
        })
    }
    /// Returns the number of live records.
    pub fn len(&self) -> usize {
        self.record_count
    }
    /// Returns whether no record is live.
    pub fn is_empty(&self) -> bool {
        self.record_count == 0
    }
    /// Returns the number of live views.
    pub fn view_len(&self) -> usize {
        self.view_count
    }
    /// Publishes an immutable view that remains stable across later updates.
    pub fn snapshot(&self) -> FrameSnapshot<R> {
        FrameSnapshot {
            scene: self.scene,
            version: self.version,
            records: self.snapshot_records.clone(),
            views: self.snapshot_views.clone(),
            dirty: self.snapshot_dirty.clone(),
        }
    }
    /// Looks up a live record.
    pub fn record(&self, id: RenderId) -> Result<&RenderRecord<R>, SceneError> {
        let dense = *self.records[self.record_index(id, 0)?]
            .value
            .as_ref()
            .expect("validated record");
        Ok(&self.record_values[dense])
    }
    /// Looks up a live view.
    pub fn view(&self, id: RenderViewId) -> Result<&RenderView, SceneError> {
        Ok(self.views[self.view_index(id, 0)?]
            .value
            .as_ref()
            .expect("validated view"))
    }
    /// Applies all changes or leaves the scene untouched.
    pub fn apply(&mut self, changes: &RenderChangeSet<R>) -> Result<ChangeReceipt, SceneError> {
        self.validate_changes(changes)?;
        if changes.is_empty() {
            return Ok(ChangeReceipt::default());
        }
        let mut record_dirty = BTreeSet::new();
        let mut view_dirty = BTreeSet::new();
        let mut receipt = ChangeReceipt::default();
        for (operation, change) in changes.0.iter().enumerate() {
            match change {
                RenderChange::Insert(record) => {
                    let id = self
                        .insert_record(record.clone(), operation)
                        .expect("prevalidated insert");
                    record_dirty.insert(usize::try_from(id.slot).expect("u32 fits usize"));
                    receipt.inserted.push(id);
                }
                RenderChange::Update(id, record) => {
                    let index = self
                        .record_index(*id, operation)
                        .expect("prevalidated record");
                    let dense = self.records[index].value.expect("prevalidated record");
                    self.record_values[dense] = record.clone();
                    record_dirty.insert(index);
                }
                RenderChange::Remove(id) => {
                    let index = self
                        .record_index(*id, operation)
                        .expect("prevalidated record");
                    let dense = self.records[index]
                        .value
                        .take()
                        .expect("prevalidated record");
                    self.record_values.swap_remove(dense);
                    self.record_slots.swap_remove(dense);
                    if let Some(&moved_slot) = self.record_slots.get(dense) {
                        self.records[moved_slot].value = Some(dense);
                        record_dirty.insert(moved_slot);
                    }
                    self.record_count -= 1;
                    self.record_free.insert(index);
                    record_dirty.insert(index);
                }
                RenderChange::InsertView(view) => {
                    let id = self
                        .insert_view(view.clone(), operation)
                        .expect("prevalidated insert");
                    view_dirty.insert(usize::try_from(id.slot).expect("u32 fits usize"));
                    receipt.views.push(id);
                }
                RenderChange::UpdateView(id, view) => {
                    let index = self.view_index(*id, operation).expect("prevalidated view");
                    self.views[index].value = Some(view.clone());
                    view_dirty.insert(index);
                }
                RenderChange::RemoveView(id) => {
                    let index = self.view_index(*id, operation).expect("prevalidated view");
                    self.views[index].value = None;
                    self.view_count -= 1;
                    self.view_free.insert(index);
                    view_dirty.insert(index);
                }
            }
        }
        self.version += 1;
        self.snapshot_dirty = SceneDirty {
            records: slot_ranges(&record_dirty),
            views: slot_ranges(&view_dirty),
        };
        self.publish_snapshot(&record_dirty, &view_dirty);
        Ok(receipt)
    }
    fn validate_changes(&self, changes: &RenderChangeSet<R>) -> Result<(), SceneError> {
        if !changes.is_empty() && self.version == u64::MAX {
            return Err(SceneError::SnapshotVersionExhausted);
        }
        let mut record_writes = HashSet::new();
        let mut view_writes = HashSet::new();
        let mut records = self.record_count;
        let mut views = self.view_count;
        let mut record_preview = AllocationPreview::new(&self.records, &self.record_free);
        let mut view_preview = AllocationPreview::new(&self.views, &self.view_free);
        for (operation, change) in changes.0.iter().enumerate() {
            match change {
                RenderChange::Insert(_) => {
                    if records == self.capacity.records {
                        return Err(SceneError::RecordCapacityExceeded { operation });
                    }
                    record_preview.allocate(operation)?;
                    records += 1;
                }
                RenderChange::Update(id, _) | RenderChange::Remove(id) => {
                    let index = self.record_index(*id, operation)?;
                    if !record_writes.insert(index) {
                        return Err(SceneError::ConflictingChange { operation });
                    }
                    if matches!(change, RenderChange::Remove(_)) {
                        record_preview.release(index);
                        records -= 1;
                    }
                }
                RenderChange::InsertView(view) => {
                    if view.extent.contains(&0) {
                        return Err(SceneError::InvalidView);
                    }
                    if views == self.capacity.views {
                        return Err(SceneError::ViewCapacityExceeded { operation });
                    }
                    view_preview.allocate(operation)?;
                    views += 1;
                }
                RenderChange::UpdateView(id, view) => {
                    if view.extent.contains(&0) {
                        return Err(SceneError::InvalidView);
                    }
                    let index = self.view_index(*id, operation)?;
                    if !view_writes.insert(index) {
                        return Err(SceneError::ConflictingChange { operation });
                    }
                }
                RenderChange::RemoveView(id) => {
                    let index = self.view_index(*id, operation)?;
                    if !view_writes.insert(index) {
                        return Err(SceneError::ConflictingChange { operation });
                    }
                    view_preview.release(index);
                    views -= 1;
                }
            }
        }
        Ok(())
    }
    fn publish_snapshot(&mut self, records: &BTreeSet<usize>, views: &BTreeSet<usize>) {
        publish_record_chunks(
            &self.records,
            &self.record_values,
            records,
            &mut self.snapshot_records,
        );
        publish_view_chunks(&self.views, views, &mut self.snapshot_views);
    }
    fn record_index(&self, id: RenderId, operation: usize) -> Result<usize, SceneError> {
        validate_slot(
            self.scene,
            &self.records,
            id.scene,
            id.slot,
            id.generation,
            operation,
        )
    }
    fn view_index(&self, id: RenderViewId, operation: usize) -> Result<usize, SceneError> {
        validate_slot(
            self.scene,
            &self.views,
            id.scene,
            id.slot,
            id.generation,
            operation,
        )
    }
    fn insert_record(
        &mut self,
        value: RenderRecord<R>,
        operation: usize,
    ) -> Result<RenderId, SceneError> {
        let index = allocate(&mut self.records, &mut self.record_free, operation)?;
        let slot = &mut self.records[index];
        let dense = self.record_values.len();
        slot.value = Some(dense);
        self.record_values.push(value);
        self.record_slots.push(index);
        self.record_count += 1;
        Ok(RenderId {
            scene: self.scene,
            slot: u32::try_from(index).map_err(|_| SceneError::IdentityExhausted { operation })?,
            generation: slot.generation,
        })
    }
    fn insert_view(
        &mut self,
        value: RenderView,
        operation: usize,
    ) -> Result<RenderViewId, SceneError> {
        let index = allocate(&mut self.views, &mut self.view_free, operation)?;
        let slot = &mut self.views[index];
        slot.value = Some(value);
        self.view_count += 1;
        Ok(RenderViewId {
            scene: self.scene,
            slot: u32::try_from(index).map_err(|_| SceneError::IdentityExhausted { operation })?,
            generation: slot.generation,
        })
    }
}

struct AllocationPreview<'a, T> {
    slots: &'a [Slot<T>],
    free: &'a BTreeSet<usize>,
    next_free: Option<usize>,
    released: BTreeSet<usize>,
    next_new: usize,
}

impl<'a, T> AllocationPreview<'a, T> {
    fn new(slots: &'a [Slot<T>], free: &'a BTreeSet<usize>) -> Self {
        Self {
            slots,
            free,
            next_free: free.first().copied(),
            released: BTreeSet::new(),
            next_new: slots.len(),
        }
    }
    fn release(&mut self, slot: usize) {
        self.released.insert(slot);
    }
    fn allocate(&mut self, operation: usize) -> Result<(), SceneError> {
        let existing = self.next_free;
        let released = self.released.first().copied();
        let index = match (existing, released) {
            (Some(existing), Some(released)) if existing < released => {
                self.consume_existing(existing);
                existing
            }
            (Some(_), Some(released)) => {
                self.released.remove(&released);
                released
            }
            (Some(existing), None) => {
                self.consume_existing(existing);
                existing
            }
            (None, Some(released)) => {
                self.released.remove(&released);
                released
            }
            (None, None) => {
                u32::try_from(self.next_new)
                    .map_err(|_| SceneError::IdentityExhausted { operation })?;
                let index = self.next_new;
                self.next_new = self
                    .next_new
                    .checked_add(1)
                    .ok_or(SceneError::IdentityExhausted { operation })?;
                index
            }
        };
        self.slots
            .get(index)
            .map_or(0, |slot| slot.generation)
            .checked_add(1)
            .filter(|generation| *generation != 0)
            .ok_or(SceneError::IdentityExhausted { operation })?;
        Ok(())
    }

    fn consume_existing(&mut self, existing: usize) {
        self.next_free = self
            .free
            .range((Excluded(existing), Unbounded))
            .next()
            .copied();
    }
}

fn slot_ranges(slots: &BTreeSet<usize>) -> Vec<DirtyRange> {
    let mut ranges = Vec::new();
    let mut start = None;
    let mut previous: u32 = 0;
    for &slot in slots {
        let slot = u32::try_from(slot).expect("scene slot fits u32");
        match start {
            Some(_) if previous.checked_add(1) == Some(slot) => previous = slot,
            Some(first) => {
                ranges.push(DirtyRange {
                    first,
                    last: previous,
                });
                start = Some(slot);
                previous = slot;
            }
            None => {
                start = Some(slot);
                previous = slot;
            }
        }
    }
    if let Some(first) = start {
        ranges.push(DirtyRange {
            first,
            last: previous,
        });
    }
    ranges
}

fn replace_snapshot_chunk<T>(
    root: Option<&Arc<SnapshotTree<T>>>,
    depth: u8,
    index: usize,
    chunk: SnapshotChunk<T>,
) -> Arc<SnapshotTree<T>> {
    if depth == 0 {
        return Arc::new(SnapshotTree::Leaf(chunk));
    }
    let (old_left, old_right) = match root.map(Arc::as_ref) {
        Some(SnapshotTree::Branch { left, right }) => (left.clone(), right.clone()),
        None => (None, None),
        Some(SnapshotTree::Leaf(_)) => unreachable!("snapshot tree has a fixed depth"),
    };
    let bit = (index >> (depth - 1)) & 1;
    let (left, right) = if bit == 0 {
        (
            Some(replace_snapshot_chunk(
                old_left.as_ref(),
                depth - 1,
                index,
                chunk,
            )),
            old_right,
        )
    } else {
        (
            old_left,
            Some(replace_snapshot_chunk(
                old_right.as_ref(),
                depth - 1,
                index,
                chunk,
            )),
        )
    };
    Arc::new(SnapshotTree::Branch { left, right })
}

fn snapshot_chunk<T>(root: &SnapshotRoot<T>, index: usize) -> Option<&SnapshotChunk<T>> {
    let mut node = root.as_deref()?;
    for depth in (1..=SNAPSHOT_TREE_DEPTH).rev() {
        node = match node {
            SnapshotTree::Branch { left, right } => {
                if ((index >> (depth - 1)) & 1) == 0 {
                    left.as_deref()?
                } else {
                    right.as_deref()?
                }
            }
            SnapshotTree::Leaf(_) => return None,
        };
    }
    match node {
        SnapshotTree::Leaf(chunk) => Some(chunk),
        SnapshotTree::Branch { .. } => None,
    }
}

fn snapshot_ids<T, I>(
    root: &SnapshotRoot<T>,
    scene: u64,
    from_entry: fn(u64, &SnapshotEntry<T>) -> I,
) -> Vec<I> {
    let mut ids = Vec::new();
    visit_snapshot_chunks(root.as_deref(), &mut |chunk| {
        ids.extend(
            chunk
                .iter()
                .filter_map(Option::as_ref)
                .map(|entry| from_entry(scene, entry)),
        );
    });
    ids
}

fn visit_snapshot_chunks<T>(
    node: Option<&SnapshotTree<T>>,
    visit: &mut impl FnMut(&SnapshotChunk<T>),
) {
    match node {
        Some(SnapshotTree::Branch { left, right }) => {
            visit_snapshot_chunks(left.as_deref(), visit);
            visit_snapshot_chunks(right.as_deref(), visit);
        }
        Some(SnapshotTree::Leaf(chunk)) => visit(chunk),
        None => {}
    }
}

fn publish_record_chunks<R: Clone>(
    slots: &[Slot<usize>],
    values: &[RenderRecord<R>],
    dirty: &BTreeSet<usize>,
    output: &mut SnapshotRoot<RenderRecord<R>>,
) {
    for &chunk_index in dirty
        .iter()
        .map(|slot| slot / SNAPSHOT_CHUNK_SIZE)
        .collect::<BTreeSet<_>>()
        .iter()
    {
        let start = chunk_index * SNAPSHOT_CHUNK_SIZE;
        let end = (start + SNAPSHOT_CHUNK_SIZE).min(slots.len());
        let chunk = Arc::new(
            (start..end)
                .map(|slot| {
                    slots[slot].value.map(|dense| SnapshotEntry {
                        slot: u32::try_from(slot).expect("scene slot fits u32"),
                        generation: slots[slot].generation,
                        value: values[dense].clone(),
                    })
                })
                .collect(),
        );
        *output = Some(replace_snapshot_chunk(
            output.as_ref(),
            SNAPSHOT_TREE_DEPTH,
            chunk_index,
            chunk,
        ));
    }
}

fn publish_view_chunks(
    slots: &[Slot<RenderView>],
    dirty: &BTreeSet<usize>,
    output: &mut SnapshotRoot<RenderView>,
) {
    for &chunk_index in dirty
        .iter()
        .map(|slot| slot / SNAPSHOT_CHUNK_SIZE)
        .collect::<BTreeSet<_>>()
        .iter()
    {
        let start = chunk_index * SNAPSHOT_CHUNK_SIZE;
        let end = (start + SNAPSHOT_CHUNK_SIZE).min(slots.len());
        let chunk = Arc::new(
            (start..end)
                .map(|slot| {
                    slots[slot].value.as_ref().map(|value| SnapshotEntry {
                        slot: u32::try_from(slot).expect("scene slot fits u32"),
                        generation: slots[slot].generation,
                        value: value.clone(),
                    })
                })
                .collect(),
        );
        *output = Some(replace_snapshot_chunk(
            output.as_ref(),
            SNAPSHOT_TREE_DEPTH,
            chunk_index,
            chunk,
        ));
    }
}

fn allocate<T>(
    slots: &mut Vec<Slot<T>>,
    free: &mut BTreeSet<usize>,
    operation: usize,
) -> Result<usize, SceneError> {
    let index = match free.pop_first() {
        Some(index) => index,
        None => {
            u32::try_from(slots.len()).map_err(|_| SceneError::IdentityExhausted { operation })?;
            slots.push(Slot::vacant());
            slots.len() - 1
        }
    };
    let slot = &mut slots[index];
    slot.generation = slot
        .generation
        .checked_add(1)
        .filter(|value| *value != 0)
        .ok_or(SceneError::IdentityExhausted { operation })?;
    Ok(index)
}

fn validate_slot<T>(
    scene: u64,
    slots: &[Slot<T>],
    foreign: u64,
    index: u32,
    generation: u32,
    operation: usize,
) -> Result<usize, SceneError> {
    if scene != foreign {
        return Err(SceneError::ForeignId { operation });
    }
    let index = usize::try_from(index).expect("u32 fits usize");
    match slots.get(index) {
        Some(slot) if slot.generation == generation && slot.value.is_some() => Ok(index),
        _ => Err(SceneError::StaleId { operation }),
    }
}

#[cfg(test)]
#[path = "../tests/render_scene/contract.rs"]
mod tests;
