//! Layer 2: the GL-family state machine that owns what the driver already holds.
//!
//! This layer exists for one reason: Layer 1 accepts every call and emits every
//! call, and a renderer that re-installs an unchanged pipeline, re-binds an
//! unchanged texture, or rebuilds a vertex array it built last frame pays the
//! full cost of the driver call for no change in driver state.  Layer 2 keeps a
//! mirror of what each domain of driver state currently holds, compares the
//! desired state against it, and emits only the calls that would move it.
//!
//! # Layout
//!
//! - [`knowledge`] -- the domains, the dirty mask, and the one type that makes a
//!   skip sound: a value is either *known* to be in the driver or *unknown*, and
//!   only a known value that compares equal to the desired one may be skipped.
//! - [`backend`] -- the Layer 1 traits this layer is written against, split into
//!   the ones every executable backend implements and the optional command
//!   domains reached only through their own bounds.
//! - [`event`] -- the invalidations the mirror must act on, with the plan's
//!   matrix recorded in one table.
//! - [`error`], [`counters`] -- what a refused transition carries out, and what
//!   an accepted one is accountable for.
//! - [`cache`] -- the derived-state machinery: structural keys, budgets,
//!   deterministic eviction, leases, and reverse-dependency invalidation.
//!
//! One module per state domain follows, each owning its own `desired`/`applied`
//! pair and its own `reconcile`/`invalidate` pair, so that a domain's
//! correctness is reviewable without reading another domain.  A domain that
//! overrides state already established by another domain does not exist here:
//! where the plan and the state-machine handoff disagreed about which domain
//! owns a value, [`knowledge`]'s own documentation records the resolution.
//!
//! # What this layer does not do
//!
//! It does not validate the caller's descriptors.  Layer 1 already rejects an
//! invalid framebuffer, an out-of-range texture unit, a buffer without the role
//! being bound, and a draw that leaves its index buffer -- and it does so on
//! every path including the ones this layer skips.  Re-checking would mean two
//! definitions of validity that can drift, and the cheaper one would win.
//!
//! It does not own objects.  Resources, programs, framebuffers and their
//! lifetimes belong to Layer 1's tables and to the caller's residency; this
//! layer owns only the *bindings* and the *state* those objects are installed
//! into, plus the derived objects it created itself and must therefore destroy
//! itself.
//!
//! It does not decide policy.  Which passes run, which resources are live, and
//! when a frame presents belong to the Renderer and the RenderGraph above it.

#![allow(
    unused_imports,
    reason = "Layer 2 declarations land before the domain modules that consume them"
)]

pub(super) mod backend;
pub(super) mod cache;
pub(super) mod counters;
pub(super) mod error;
pub(super) mod event;
pub(super) mod knowledge;

pub(crate) use backend::{GlOptionalComputeBackend, GlOptionalIndirectBackend, GlStateBackend};
pub(crate) use cache::{
    CacheBudget, CacheMode, DEFAULT_BUDGET, DependencySet, ResourceRef, StructuralCache,
};
pub(crate) use counters::{
    CacheCounters, DomainCounters, LifecycleCounters, StateCounters, SubmissionCounters,
};
pub(crate) use error::{PartialApplication, StateError};
pub(crate) use event::{ScopedRawAccess, StateEvent};
pub(crate) use knowledge::{DirtyDomains, DriverKnowledge, ExecutionMode, StateDomain};
