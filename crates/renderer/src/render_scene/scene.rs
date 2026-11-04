//! Transactional mutable RenderScene state and change validation.

#[cfg(test)]
use super::snapshot::replace_snapshot_chunk;
use super::*;
use std::{
    collections::{BTreeSet, HashSet},
    ops::Bound::{Excluded, Unbounded},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_SCENE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub(super) struct Slot<T> {
    pub(super) generation: u32,
    pub(super) value: Option<T>,
}

impl<T> Slot<T> {
    pub(super) fn vacant() -> Self {
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

#[cfg(test)]
#[path = "../tests/render_scene/contract.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/render_scene/snapshot.rs"]
mod snapshot_tests;
