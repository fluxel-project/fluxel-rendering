//! Conservative GL-family driver-state mirror.
//!
//! The GL, GLES and WebGL adapters all run on a mutable context.  This module
//! owns no GL objects: it remembers only values successfully installed through
//! its context-private authority. `Unknown` is intentionally the initial state; assuming
//! GL defaults would make a raw host-context call silently invalidate a skip.
//!
//! The module is backend-private.  It is shared by desktop GL, GLES and WebGL2
//! lowering, while profile/extension admission remains in the capability probe.

mod cache;
mod context;
mod event;
mod knowledge;

#[allow(
    unused_imports,
    reason = "execution owners adopt these private state types incrementally"
)]
pub(crate) use cache::{
    CacheBudget, CacheCounters, CacheMutation, DependencySet, ResourceRef, StructuralCache,
};
#[allow(
    unused_imports,
    reason = "browser/native owner wiring follows the shared model"
)]
pub(crate) use context::{
    BindingFlush, BoundGroupPacket, CanonicalBlockId, ContextState, ContextStateError,
    DerivedCacheKey, DerivedCacheKind, PassPacket, PixelTransferPacket, RasterPipelineBlocks,
    RasterPipelineDiff, RasterPipelinePacket,
};
#[allow(
    unused_imports,
    reason = "event authority is consumed by future owner wiring"
)]
pub(crate) use event::{ScopedRawAccess, StateEvent};
pub(crate) use knowledge::{DirtyDomains, DriverKnowledge, ExecutionMode, StateDomain};
