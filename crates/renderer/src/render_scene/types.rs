//! Render-scene value types and immutable snapshot handles.

use fluxel_assets::{AssetId, AssetKind, AssetSnapshot, ContentGeneration};
use std::{fmt, sync::Arc};

#[path = "error.rs"]
mod error;
#[path = "scene.rs"]
mod scene;
#[path = "snapshot.rs"]
mod snapshot;
pub use error::SceneError;
pub use scene::RenderScene;
use scene::Slot;
use snapshot::{
    SNAPSHOT_CHUNK_SIZE, allocate, publish_record_chunks, publish_view_chunks, slot_ranges,
    snapshot_chunk, snapshot_ids, validate_slot,
};

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
    /// Explicit eye information for lighting routes that require world-space view direction.
    pub shading_camera: Option<super::ShadingCamera>,
    /// The view's directional light shared by Forward and Deferred PBR.
    pub directional_light: super::DirectionalLight,
    /// Whether this view is rendering a reflected camera.
    pub mirrored: bool,
    /// Whether PBR surface emissive contribution is enabled for this view.
    pub emissive_enabled: bool,
}

impl RenderView {
    /// Associates validated world-space eye information with this view.
    pub fn with_shading_camera(mut self, camera: super::ShadingCamera) -> Self {
        self.shading_camera = Some(camera);
        self
    }

    /// Associates one directional light with this view snapshot.
    pub fn with_directional_light(mut self, light: super::DirectionalLight) -> Self {
        self.directional_light = light;
        self
    }

    /// Selects reflected-camera raster and two-sided PBR behavior for this view.
    ///
    /// Supply its existing Fluxel clip transform and reflected shading camera
    /// independently. This flag does not construct or modify either transform.
    #[must_use]
    pub fn with_mirrored_view(mut self) -> Self {
        self.mirrored = true;
        self
    }

    /// Selects whether PBR surface emissive contribution is evaluated for this view.
    ///
    /// This only selects the renderer's PBR specialization. It does not modify
    /// material-owned emissive parameters or the view transform.
    #[must_use]
    pub fn with_emissive_enabled(mut self, enabled: bool) -> Self {
        self.emissive_enabled = enabled;
        self
    }

    /// Creates a view with a nonempty extent and the default directional light.
    pub fn new(transform: RenderTransform, extent: [u32; 2]) -> Result<Self, SceneError> {
        if extent.contains(&0) {
            Err(SceneError::InvalidView)
        } else {
            Ok(Self {
                transform,
                extent,
                shading_camera: None,
                directional_light: super::DirectionalLight::default(),
                mirrored: false,
                emissive_enabled: true,
            })
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
