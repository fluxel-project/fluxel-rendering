//! Persistent snapshot-tree updates and slot allocation helpers.

use super::*;
use std::{collections::BTreeSet, sync::Arc};

pub(super) const SNAPSHOT_CHUNK_SIZE: usize = 64;
const SNAPSHOT_TREE_DEPTH: u8 = 26;

pub(super) fn slot_ranges(slots: &BTreeSet<usize>) -> Vec<DirtyRange> {
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

pub(super) fn replace_snapshot_chunk<T>(
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

pub(super) fn snapshot_chunk<T>(root: &SnapshotRoot<T>, index: usize) -> Option<&SnapshotChunk<T>> {
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

pub(super) fn snapshot_ids<T, I>(
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

pub(super) fn visit_snapshot_chunks<T>(
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

pub(super) fn publish_record_chunks<R: Clone>(
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

pub(super) fn publish_view_chunks(
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

pub(super) fn allocate<T>(
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

pub(super) fn validate_slot<T>(
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
