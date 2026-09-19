//! Completion-safe retirement: when an RHI object's backing may be released.
//!
//! This module owns rhi-design 02 section 18.6 and 05 section 41.6. Its single
//! responsibility is deciding when an object's backing may be released, and
//! holding that backing until then.
//!
//! # Why a dropped handle is not a release
//!
//! The contract is explicit that three different facts are not the same one:
//!
//! ```text
//! public handle drop
//!     !=
//! logical object is no longer referenced
//!     !=
//! native backing may be released immediately
//! ```
//!
//! A public handle is an `Arc` clone of a backend object, so without this
//! registry the last handle dropping *is* the release — and that is precisely
//! what section 18.6 forbids, because submitted GPU work may still be reading
//! the object. The registry therefore holds one reference of its own, which is
//! what keeps the backing alive past the caller's last clone, and drops it only
//! when both of the contract's conditions hold:
//!
//! ```text
//! final CPU logical owner no longer references it
//! +
//! every accepted GPU work item referencing it is terminal
//! ```
//!
//! Terminal is `Complete`, `DeviceLost`, or `Failed` — the same three states
//! [`CompletionState::is_terminal`] already names, which is why this module asks
//! that method rather than spelling the list a second time.
//!
//! # The token is a completion point, not a plan point
//!
//! The contract states the condition in terms of "the last batch that actually
//! referenced the object", and the plan knows which batches those are. But what
//! a device later *observes* is a [`CompletionPoint`]. A plan point is a
//! position inside one plan and is not what any device API answers with, so a
//! plan-point-keyed registry could only ever be told about terminality in a
//! vocabulary that does not exist: it would hold every submitted object forever.
//! Keying on the observable token is what makes release reachable at all.
//!
//! # How a backing reaches this registry
//!
//! Each public handle exposes a `pub(crate) fn backing()` returning an owned
//! `Arc` clone of its backend object, and the device façade — which is the only
//! thing that calls it — registers what it just created. That accessor opens
//! nothing up: a `*Backend` trait has no verb its public handle does not already
//! expose, and a spare clone can only *defer* a reclamation, never cause one.
//!
//! # Retirement is internal, so there is no new public verb
//!
//! Section 41.6 says "RHI internal retirement may use completion of the last
//! batch that actually referenced the object". Nothing here is reachable from a
//! public type: the observable surface is the `*_reclaimed` counters and the
//! tooling `ObjectReclaimed` event, both of which this registry feeds.
//!
//! # What this module does not own
//!
//! It does not talk to a backend and it never calls one. A completion state is
//! *reported to* it by the device that observed it, rather than asked for, so
//! that no sweep can re-enter the device from under its own lock. It also does
//! not decide when work is done and does not order submissions — it consumes
//! those answers, and it never sends one.

use std::sync::Arc;

use super::binding::{BindGroupBackend, BindGroupLayoutBackend};
use super::pipeline::{
    ComputePipelineBackend, PipelineInterfaceBackend, RasterPipelineBackend,
};
use super::platform::ObjectId;
use super::resource::{
    BufferBackend, SamplerBackend, TextureBackend, TextureViewBackend,
};
use super::shader::ShaderModuleBackend;
use super::statistics::ObjectKind;
use super::submission::CompletionPoint;

/// One object's backing, held past the caller's last handle clone.
///
/// The variants are the ten logical object kinds the freeze checklist marks
/// "GPU-safe retirement". Upload jobs, readback tickets, acquired frames and
/// configured presentations are deliberately absent: they are transient or
/// device-scoped rather than inventoried objects, and each has its own terminal
/// path.
pub(crate) enum Retained {
    /// A buffer.
    Buffer(Arc<dyn BufferBackend>),
    /// A texture.
    Texture(Arc<dyn TextureBackend>),
    /// A texture view.
    TextureView(Arc<dyn TextureViewBackend>),
    /// A sampler.
    Sampler(Arc<dyn SamplerBackend>),
    /// A shader module.
    ShaderModule(Arc<dyn ShaderModuleBackend>),
    /// A bind group layout.
    BindGroupLayout(Arc<dyn BindGroupLayoutBackend>),
    /// A bind group.
    BindGroup(Arc<dyn BindGroupBackend>),
    /// A pipeline interface.
    PipelineInterface(Arc<dyn PipelineInterfaceBackend>),
    /// A raster pipeline.
    RasterPipeline(Arc<dyn RasterPipelineBackend>),
    /// A compute pipeline.
    ComputePipeline(Arc<dyn ComputePipelineBackend>),
}

impl Retained {
    /// This backing's object id.
    pub(crate) fn id(&self) -> ObjectId {
        match self {
            Self::Buffer(inner) => inner.id(),
            Self::Texture(inner) => inner.id(),
            Self::TextureView(inner) => inner.id(),
            Self::Sampler(inner) => inner.id(),
            Self::ShaderModule(inner) => inner.id(),
            Self::BindGroupLayout(inner) => inner.id(),
            Self::BindGroup(inner) => inner.id(),
            Self::PipelineInterface(inner) => inner.id(),
            Self::RasterPipeline(inner) => inner.id(),
            Self::ComputePipeline(inner) => inner.id(),
        }
    }

    /// The statistics inventory kind this backing enters as.
    pub(crate) fn kind(&self) -> ObjectKind {
        match self {
            Self::Buffer(_) => ObjectKind::Buffer,
            Self::Texture(_) => ObjectKind::Texture,
            Self::TextureView(_) => ObjectKind::TextureView,
            Self::Sampler(_) => ObjectKind::Sampler,
            Self::ShaderModule(_) => ObjectKind::ShaderModule,
            Self::BindGroupLayout(_) => ObjectKind::BindGroupLayout,
            Self::BindGroup(_) => ObjectKind::BindGroup,
            Self::PipelineInterface(_) => ObjectKind::PipelineInterface,
            Self::RasterPipeline(_) => ObjectKind::RasterPipeline,
            Self::ComputePipeline(_) => ObjectKind::ComputePipeline,
        }
    }

    /// How many owners outside this registry still hold the object.
    ///
    /// The count is a snapshot, and it is deliberately used in the direction
    /// where being wrong is cheap. Observing a stale non-zero owner only defers
    /// a reclamation that a later sweep performs; the opposite mistake — reading
    /// zero while an owner exists — is not reachable here, because an external
    /// owner is itself what a caller would clone from, and a caller with no
    /// handle left has nothing to clone.
    fn external_owners(&self) -> usize {
        let strong = match self {
            Self::Buffer(inner) => Arc::strong_count(inner),
            Self::Texture(inner) => Arc::strong_count(inner),
            Self::TextureView(inner) => Arc::strong_count(inner),
            Self::Sampler(inner) => Arc::strong_count(inner),
            Self::ShaderModule(inner) => Arc::strong_count(inner),
            Self::BindGroupLayout(inner) => Arc::strong_count(inner),
            Self::BindGroup(inner) => Arc::strong_count(inner),
            Self::PipelineInterface(inner) => Arc::strong_count(inner),
            Self::RasterPipeline(inner) => Arc::strong_count(inner),
            Self::ComputePipeline(inner) => Arc::strong_count(inner),
        };
        // The registry's own reference is not an owner in the contract's sense.
        strong.saturating_sub(1)
    }
}

/// One inventoried object and the last point known to reference it.
struct Entry {
    retained: Retained,
    last_use: Option<CompletionPoint>,
}

impl Entry {
    /// Whether section 18.6's two conditions both hold.
    fn is_reclaimable(&self, terminal: &[CompletionPoint]) -> bool {
        // With no accepted work referencing the object, the second condition is
        // vacuously satisfied — there is no GPU reader to wait for.
        let work_is_terminal = self
            .last_use
            .is_none_or(|point| terminal.contains(&point));
        work_is_terminal && self.retained.external_owners() == 0
    }
}

/// The device's live inventory and its reclamation decisions.
#[derive(Default)]
pub(crate) struct RetirementRegistry {
    state: std::sync::Mutex<State>,
}

#[derive(Default)]
struct State {
    entries: Vec<Entry>,
    terminal: Vec<CompletionPoint>,
    all_terminal: bool,
}

/// One object whose backing this registry released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reclaimed {
    /// The released object.
    pub(crate) id: ObjectId,
    /// What kind of object it was.
    pub(crate) kind: ObjectKind,
}

impl RetirementRegistry {
    /// An empty registry.
    pub(crate) fn new() -> Self {
        Self {
            state: std::sync::Mutex::new(State::default()),
        }
    }

    /// Takes charge of one newly created object.
    ///
    /// The backing is held from here until [`Self::sweep`] releases it, which is
    /// what makes a dropped public handle stop being a release.
    ///
    /// Registering does not sweep. The device that owns this registry sweeps at
    /// a frame boundary, because a sweep per registration would make creating
    /// `n` objects cost `O(n^2)`.
    pub(crate) fn register(&self, retained: Retained) {
        let entry = Entry {
            retained,
            last_use: None,
        };
        self.lock().entries.push(entry);
    }

    /// Records that `point` references the object `id`.
    ///
    /// The latest point wins, because an object is free only once the *last*
    /// work item referencing it is terminal: a later pending use must keep it
    /// alive even after an earlier use has completed. [`CompletionPoint`] is
    /// ordered, so "latest" is a maximum rather than "most recently reported" —
    /// a submission that lands out of order must not lower the bar.
    ///
    /// This never releases: recording a use can only extend how long a backing
    /// is held, so a caller cannot lose a reclamation by forgetting anything
    /// here.
    pub(crate) fn note_use(&self, id: ObjectId, point: CompletionPoint) {
        let mut state = self.lock();
        let Some(entry) = state.entries.iter_mut().find(|entry| entry.retained.id() == id) else {
            // A use of an object this device never inventoried — a foreign
            // device's object that a portable check should already have
            // refused. Ignoring it keeps a bookkeeping miss from becoming a
            // panic; the missing refusal is the bug, and it is not this
            // module's to report.
            return;
        };
        entry.last_use = Some(match entry.last_use {
            Some(previous) if previous > point => previous,
            _ => point,
        });
    }

    /// Reports that one completion point reached a terminal state.
    pub(crate) fn note_terminal(&self, point: CompletionPoint) {
        let mut state = self.lock();
        if !state.terminal.contains(&point) {
            state.terminal.push(point);
        }
    }

    /// Reports device loss, which makes every point terminal at once.
    ///
    /// Loss is not "each outstanding point failed in turn": there is no later
    /// observation that could ever report a point from a lost device, so a
    /// registry that waited for per-point reports would hold every object from a
    /// lost device forever.
    pub(crate) fn note_device_lost(&self) {
        self.lock().all_terminal = true;
    }

    /// Releases every object whose contract conditions now hold.
    ///
    /// Returns what was released so the caller can report it; a released object
    /// is gone from the registry, so a second sweep reports it again only if it
    /// was registered again.
    pub(crate) fn sweep(&self) -> Vec<Reclaimed> {
        let mut state = self.lock();
        if state.all_terminal {
            let reclaimed = state
                .entries
                .iter()
                .map(|entry| Reclaimed {
                    id: entry.retained.id(),
                    kind: entry.retained.kind(),
                })
                .collect();
            state.entries.clear();
            return reclaimed;
        }

        let mut reclaimed = Vec::new();
        let terminal = std::mem::take(&mut state.terminal);
        let mut index = 0;
        while index < state.entries.len() {
            if state.entries[index].is_reclaimable(&terminal) {
                let entry = state.entries.remove(index);
                reclaimed.push(Reclaimed {
                    id: entry.retained.id(),
                    kind: entry.retained.kind(),
                });
            } else {
                index += 1;
            }
        }
        state.terminal = terminal;
        reclaimed
    }

    /// How many objects this registry currently holds.
    pub(crate) fn len(&self) -> usize {
        self.lock().entries.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic inside a sweep would otherwise poison every later
        // registration, turning one bookkeeping bug into a device that can no
        // longer create anything. The state is a plain inventory, so recovering
        // it is sound: the worst case is a stale entry, which a later sweep
        // reclaims.
        self.state.lock().unwrap_or_else(|poison| poison.into_inner())
    }
}

#[cfg(test)]
mod tests;
