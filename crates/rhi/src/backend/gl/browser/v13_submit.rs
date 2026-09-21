//! WebGL2's v13 submission split.
//!
//! A WebGL command stream cannot be rewound after its first JavaScript call.
//! Consequently this module deliberately has two separate phases:
//!
//! * phase A asks the browser driver's object resolver to turn every recorded
//!   payload into an *owned*, executable browser action.  It performs no GL or
//!   JavaScript call.  Missing native backing, an unsupported command family,
//!   and an invalid resource relationship must all fail here.
//! * phase B consumes that action list on the owner thread.  An error from this
//!   phase is post-commit: the caller must publish a failed/lost completion,
//!   never return it as `submit(Err)`.
//!
//! The resolver is intentionally browser-private.  `GlObjectName` is only the
//! device adapter's opaque object namespace; the browser owns the generation
//! safe `BufferId`, `TextureId`, `ProgramId`, framebuffer and binding metadata
//! needed by the typed executor.  Keeping that translation here prevents those
//! WebGL identities leaking into the common RHI API.

use crate::api::command::record::RecordedPayload;
use crate::api::error::RhiResult;
use crate::backend::gl::platform::{GlSubmissionBatch, GlSubmissionPlan};

use super::driver::BrowserDriverState;

/// Opaque owner-table key for a readback result sink.  A Phase-B action must
/// never retain a portable `ReadbackTicket`: the ticket is registered during
/// Phase A and this key is the only capability the action carries back to the
/// browser owner when its bytes are produced.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct BrowserReadbackSinkId(pub(super) u64);

/// Fully resolved transfer work.  These packets deliberately use only the GL
/// typed namespace and copied CPU bytes, so Phase B cannot resurrect a dropped
/// portable resource or accidentally cross a WebGL context boundary.
pub(super) enum BrowserV13CopyPacket {
    Buffer {
        source: crate::backend::gl::api::GlBufferRange,
        destination: crate::backend::gl::api::GlBufferRange,
    },
    Texture {
        source: crate::backend::gl::api::GlTextureRegion,
        destination: crate::backend::gl::api::GlTextureRegion,
    },
}

pub(super) enum BrowserV13UploadPacket {
    Buffer {
        destination: crate::backend::gl::api::GlBufferRange,
        bytes: std::sync::Arc<[u8]>,
    },
    Texture {
        destination: crate::backend::gl::api::GlTextureRegion,
        layout: crate::backend::gl::api::GlPixelLayout,
        bytes: std::sync::Arc<[u8]>,
    },
}

pub(super) enum BrowserV13ReadbackPacket {
    Buffer {
        source: crate::backend::gl::api::GlBufferRange,
        sink: BrowserReadbackSinkId,
    },
    Texture {
        source: crate::backend::gl::api::GlTextureRegion,
        layout: crate::backend::gl::api::GlPixelLayout,
        sink: BrowserReadbackSinkId,
    },
}

/// Query operations are separate packets because WebGL2 has distinct core
/// occlusion and extension-timer command routes. `Resolve` is intentionally
/// absent: WebGL2 has no query-to-buffer resolve operation and must reject it
/// in Phase A instead of synchronously reading a result.
pub(super) enum BrowserV13QueryPacket {
    BeginOcclusion(crate::backend::gl::api::QueryId),
    EndOcclusion,
    BeginElapsed(crate::backend::gl::api::QueryId),
    EndElapsed,
    Timestamp(crate::backend::gl::api::QueryId),
}

/// The non-raster packet vocabulary dispatched by the browser owner.  This is
/// the explicit hand-off between resolver-side object validation and executor
/// calls; driver code may not inspect a `RecordedPayload` in Phase B.
pub(super) enum BrowserV13ActionPacket {
    Copy(BrowserV13CopyPacket),
    Upload(BrowserV13UploadPacket),
    Readback(BrowserV13ReadbackPacket),
    Query(BrowserV13QueryPacket),
}

/// One adapter which turns a resolved packet into a Phase-B action without
/// exposing browser object identities outside this backend module. The driver
/// owns the actual dispatch method because it owns the readback sink table and
/// completion bookkeeping.
pub(super) struct BrowserV13PacketAction(pub(super) BrowserV13ActionPacket);

impl BrowserV13PhaseBAction for BrowserV13PacketAction {
    fn execute(self: Box<Self>, owner: &mut BrowserDriverState) -> RhiResult<()> {
        owner.execute_v13_packet(self.0)
    }
}

/// Browser-private Phase-B action.  Implementations contain only resolved,
/// generation-safe browser object ids and copied scalar/payload data; they may
/// not retain a `RecordedPayload` or a public RHI handle.
pub(super) trait BrowserV13PhaseBAction {
    /// Executes one already-admitted action on the browser owner thread.
    ///
    /// A failure here is necessarily post-commit once an earlier action ran.
    /// The driver therefore turns it into terminal completion/device state,
    /// rather than exposing a false `submit` rejection.
    fn execute(self: Box<Self>, owner: &mut BrowserDriverState) -> RhiResult<()>;
}

/// Converts portable recording data to owned WebGL2 executor actions.
///
/// The implementation is the sole bridge from the `GlObjectName` namespace to
/// the browser's typed tables.  It is also where a raster pipeline's program,
/// VAO and fixed state, bind-group bindings, view/framebuffer metadata, and
/// copy/query routes are looked up.  Returning success without creating a
/// Phase-B action is forbidden: every recorded payload must be consumed.
pub(super) trait BrowserV13ObjectResolver {
    /// Called when a later Phase-A command rejects the submission after this
    /// resolver registered owner-side provisional state (such as a readback
    /// sink).  It must restore every such ticket to a terminal state because
    /// no browser command was entered and no completion will arrive.
    fn phase_a_aborted(&self) {}
    /// Raster begin resolves texture/frame views into a concrete framebuffer
    /// description.  It must not create that framebuffer yet.
    fn raster_begin(
        &self,
        _: &crate::api::command::record::RasterBegin,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("raster")
    }
    fn raster_draw(
        &self,
        _: &crate::api::command::record::RasterDraw,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("raster")
    }
    fn raster_end(&self) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("raster")
    }
    /// Copy, clear, upload and readback records are all checked against the
    /// browser resource table before their action can exist.
    fn copy(
        &self,
        _: &crate::api::command::record::CopyRecord,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("copy")
    }
    fn upload(
        &self,
        _: &crate::api::resource::transfer::UploadJob,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("upload")
    }
    fn readback(
        &self,
        _: &crate::api::resource::transfer::ReadbackTicket,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("readback")
    }
    fn query_begin(
        &self,
        _: &crate::api::query::QuerySet,
        _: u32,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("query")
    }
    fn query_end(
        &self,
        _: &crate::api::query::QuerySet,
        _: u32,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("query")
    }
    fn timestamp_write(
        &self,
        _: &crate::api::query::QuerySet,
        _: u32,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("timestamp query")
    }
    fn query_resolve(
        &self,
        _: &crate::api::command::record::QueryResolve,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("query resolve")
    }
    fn debug_push(
        &self,
        _: &crate::api::identity::Label,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("debug marker")
    }
    fn debug_pop(&self) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("debug marker")
    }
    fn debug_marker(
        &self,
        _: &crate::api::identity::Label,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        unsupported("debug marker")
    }
}

/// An owned, fully admitted WebGL2 batch.
pub(super) struct BrowserV13Batch {
    pub(super) point: crate::api::submission::PlanPoint,
    actions: Vec<Box<dyn BrowserV13PhaseBAction>>,
}

/// The complete Phase-A product for one v13 submission plan.
///
/// It owns no GL/JS handles directly; those remain in the owner-thread
/// discovery table, addressed by the generation-safe IDs in each action.
pub(super) struct BrowserV13Submission {
    batches: Vec<BrowserV13Batch>,
}

impl BrowserV13Submission {
    /// Builds all actions before the first WebGL call.
    pub(super) fn phase_a(
        plan: GlSubmissionPlan<'_>,
        resolver: &dyn BrowserV13ObjectResolver,
    ) -> RhiResult<Self> {
        let mut batches = Vec::with_capacity(plan.batches.len());
        for batch in plan.batches {
            let point = batch.point;
            let actions = match lower_batch(batch, resolver) {
                Ok(actions) => actions,
                Err(error) => {
                    resolver.phase_a_aborted();
                    return Err(error);
                }
            };
            batches.push(BrowserV13Batch { point, actions });
        }
        Ok(Self { batches })
    }

    /// Runs the admitted stream.  Callers must treat an error as post-commit.
    /// The returned point is the last batch whose command stream was entered;
    /// it gives the driver enough context to fail the correct completion token.
    pub(super) fn phase_b(
        self,
        owner: &mut BrowserDriverState,
    ) -> Result<Vec<crate::api::submission::PlanPoint>, BrowserV13PostCommitFailure> {
        let mut entered = Vec::with_capacity(self.batches.len());
        for batch in self.batches {
            entered.push(batch.point);
            for action in batch.actions {
                if let Err(error) = action.execute(owner) {
                    return Err(BrowserV13PostCommitFailure {
                        entered_batches: entered,
                        error,
                    });
                }
            }
        }
        Ok(entered)
    }

    #[cfg(test)]
    fn batch_count(&self) -> usize {
        self.batches.len()
    }
}

fn lower_batch(
    batch: GlSubmissionBatch<'_>,
    resolver: &dyn BrowserV13ObjectResolver,
) -> RhiResult<Vec<Box<dyn BrowserV13PhaseBAction>>> {
    let mut actions = Vec::with_capacity(batch.commands.len());
    for command in batch.commands {
        // ResourceUse is the synchronization input, not merely capture
        // metadata.  WebGL2 command order makes ordinary raster/copy writes
        // visible to later commands, but it has no `glMemoryBarrier` route for
        // shader-storage visibility.  Refuse that exact dependency during
        // phase A rather than accepting a stream whose required barrier cannot
        // be emitted.  The WebGL2 binding lowerer likewise has no storage-image
        // route, so this is intentionally a defensive second gate.
        if command.uses.iter().any(requires_unavailable_barrier) {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::Unsupported,
                "WebGL2 has no memory-barrier route for shader-storage writes",
            )
            .at("WebGL2::submit phase A resource visibility"));
        }
        // This is intentionally the only call made during phase A.  Resolver
        // implementations are data-table lookups/conversions, never JS calls.
        actions.push(lower_payload(&command.payload, resolver)?);
    }
    Ok(actions)
}

fn requires_unavailable_barrier(use_: &crate::api::command::ResourceUse) -> bool {
    use crate::api::command::AccessMask;
    use crate::api::command::ResourceUse;
    match use_ {
        ResourceUse::Buffer(value) => value.access.contains(AccessMask::SHADER_WRITE),
        ResourceUse::Texture(value) => value.access.contains(AccessMask::SHADER_WRITE),
        ResourceUse::Frame(_) | ResourceUse::AccelerationStructure(_) | ResourceUse::Query(_) => {
            false
        }
    }
}

/// Exhaustive phase-A classification.  WebGL2 has no compute, mesh, ray or
/// acceleration-structure execution route, so those are rejected before the
/// browser is touched.  The remaining families are delegated only after their
/// concrete object metadata has been resolved by the owner-thread table.
fn lower_payload(
    payload: &RecordedPayload,
    resolver: &dyn BrowserV13ObjectResolver,
) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
    use crate::api::command::record::RecordedPayload as P;
    match payload {
        P::RasterBegin(value) => resolver.raster_begin(value),
        P::RasterDraw(value) => resolver.raster_draw(value),
        P::RasterEnd => resolver.raster_end(),
        P::Copy(value) => resolver.copy(value),
        P::Upload(value) => resolver.upload(value),
        P::Readback(value) => resolver.readback(value),
        P::QueryBegin { set, index } => resolver.query_begin(set, *index),
        P::QueryEnd { set, index } => resolver.query_end(set, *index),
        P::TimestampWrite { set, index } => resolver.timestamp_write(set, *index),
        P::QueryResolve(value) => resolver.query_resolve(value),
        P::DebugPush(label) => resolver.debug_push(label),
        P::DebugPop => resolver.debug_pop(),
        P::DebugMarker(label) => resolver.debug_marker(label),
        P::MeshDispatch(_)
        | P::MeshIndirect(_)
        | P::RayTracingBegin(_)
        | P::RayTracingDispatch(_)
        | P::RayTracingEnd
        | P::AccelerationStructure(_) => {
            unsupported("mesh, ray-tracing, and acceleration structures")
        }
        P::ComputeBegin(_) | P::ComputeDispatch(_) | P::ComputeEnd | P::ComputeIndirect(_) => {
            unsupported("compute commands")
        }
        P::RasterIndirect(_) => unsupported("indirect raster commands"),
    }
}

fn unsupported(family: &'static str) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
    Err(crate::api::error::RhiError::new(
        crate::api::error::RhiErrorKind::Unsupported,
        format!("WebGL2 has no {family} lowering route"),
    )
    .at("WebGL2::submit phase A"))
}

/// A Phase-B failure after browser work may have begun.
pub(super) struct BrowserV13PostCommitFailure {
    pub(super) entered_batches: Vec<crate::api::submission::PlanPoint>,
    pub(super) error: crate::api::error::RhiError,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use crate::api::command::ResourceUse;
    use crate::api::command::record::{RecordedCommand, RecordedPayload};
    use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
    use crate::api::identity::Label;
    use crate::api::identity::{DeviceIdentity, DeviceInstanceId};
    use crate::api::submission::{PlanPoint, SubmissionBatchId, SubmissionPlanId};
    use crate::backend::gl::platform::{GlSubmissionBatch, GlSubmissionPlan};

    use super::{BrowserV13ObjectResolver, BrowserV13PhaseBAction, BrowserV13Submission};

    struct Reject;
    impl BrowserV13ObjectResolver for Reject {
        fn debug_push(&self, _: &Label) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
            Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "WebGL2 command route is absent",
            ))
        }
    }

    struct Count(&'static AtomicU32);
    impl BrowserV13PhaseBAction for Count {
        fn execute(self: Box<Self>, _: &mut super::BrowserDriverState) -> RhiResult<()> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct CountingResolver(&'static AtomicU32);
    impl BrowserV13ObjectResolver for CountingResolver {
        fn debug_push(&self, _: &Label) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
            Ok(Box::new(Count(self.0)))
        }
    }

    fn command() -> RecordedCommand {
        RecordedCommand {
            payload: RecordedPayload::DebugPush(Label(Some("phase-a".into()))),
            uses: Vec::<ResourceUse>::new(),
        }
    }

    fn point() -> PlanPoint {
        PlanPoint::new(
            SubmissionPlanId::new(DeviceIdentity::new(DeviceInstanceId::new(7)), 1),
            SubmissionBatchId::new(0),
        )
    }

    #[test]
    fn unsupported_payload_is_rejected_before_phase_b() {
        let command = command();
        let plan = GlSubmissionPlan {
            batches: vec![GlSubmissionBatch {
                point: point(),
                commands: vec![&command],
            }],
        };
        assert!(BrowserV13Submission::phase_a(plan, &Reject).is_err());
    }

    #[test]
    fn phase_a_owns_one_action_for_each_recorded_payload() {
        let command = command();
        static COUNT: AtomicU32 = AtomicU32::new(0);
        COUNT.store(0, Ordering::Relaxed);
        let plan = GlSubmissionPlan {
            batches: vec![GlSubmissionBatch {
                point: point(),
                commands: vec![&command],
            }],
        };
        let submission = BrowserV13Submission::phase_a(plan, &CountingResolver(&COUNT)).unwrap();
        // Lowering must not execute browser work.
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
        assert_eq!(submission.batch_count(), 1);
    }
}
