//! Browser fence and query object execution.
//!
//! Fence observations are accepted-unknown: a parameter answer that is not a
//! usable number reports `Unknown` instead of guessing, so leases stay live and
//! completion-safe release keeps waiting.
//!
//! This module owns query object lifetime and the occlusion domain. Timer
//! measurements are classified by `exec_timer.rs`, which reads them through the
//! acquired timer commands and withholds a disrupted interval; the result path
//! here only decides which of the two domains owns a recorded target.

use web_sys::WebGl2RenderingContext as Gl;

use super::super::api::{
    GlError, GlFamilyApi as _, GlFenceLease, GlFenceStatus, GlMemoryBarrier, GlOcclusionQueryApi,
    GlQueryObjectsApi, GlQueryResult, GlSyncApi, GlWaitBound, QueryId, SyncId,
};
use super::discovery::WebGl2BrowserDiscovery;

/// `clientWaitSync` timeout ceiling: exact f64 integers end at 2^53-1, and a
/// larger bounded wait is indistinguishable from "effectively infinite".
const MAX_EXACT_TIMEOUT_NS: u64 = 9_007_199_254_740_991;

impl GlSyncApi for WebGl2BrowserDiscovery {
    fn create_fence(&mut self) -> Result<GlFenceLease, GlError> {
        const OP: &str = "create-fence";
        self.assert_provider_ready(OP)?;
        let raw = self
            .raw
            .fence_sync(Gl::SYNC_GPU_COMMANDS_COMPLETE, 0)
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        let slot = Self::allocate_slot(&mut self.next_sync_slot, OP)?;
        let id = SyncId::new(self.context_stamp(), slot, 0);
        self.syncs.insert(
            slot,
            super::objects::BrowserSync {
                generation: id.generation,
                raw,
            },
        );
        match self.fences.issue(id) {
            Ok(lease) => Ok(lease),
            Err(error) => {
                if let Some(entry) = self.syncs.remove(&slot) {
                    self.raw.delete_sync(Some(&entry.raw));
                }
                Err(error)
            }
        }
    }

    fn destroy_fence(&mut self, fence: GlFenceLease) -> Result<(), GlError> {
        const OP: &str = "destroy-fence";
        self.assert_provider_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let entry = self.sync(OP, fence.fence)?;
        self.raw.delete_sync(Some(&entry.raw));
        self.syncs.remove(&fence.fence.slot);
        self.fences.revoke(fence);
        self.driver_error(OP)
    }

    fn poll_fence(&mut self, fence: GlFenceLease) -> Result<GlFenceStatus, GlError> {
        const OP: &str = "poll-fence";
        self.assert_provider_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let entry = self.sync(OP, fence.fence)?;
        let status = self.sync_status(&entry.raw);
        Ok(status)
    }

    fn wait_fence(
        &mut self,
        fence: GlFenceLease,
        bound: GlWaitBound,
    ) -> Result<GlFenceStatus, GlError> {
        const OP: &str = "wait-fence";
        self.assert_provider_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let entry = self.sync(OP, fence.fence)?;
        // A zero bound is a poll: no flush flag, no wait.
        let flags = if bound.nanoseconds == 0 {
            0
        } else {
            Gl::SYNC_FLUSH_COMMANDS_BIT
        };
        let timeout = bound.nanoseconds.min(MAX_EXACT_TIMEOUT_NS) as f64;
        let result = self
            .raw
            .client_wait_sync_with_f64(&entry.raw, flags, timeout);
        Ok(match result {
            Gl::ALREADY_SIGNALED | Gl::CONDITION_SATISFIED => GlFenceStatus::Complete,
            Gl::TIMEOUT_EXPIRED => GlFenceStatus::Pending,
            // WAIT_FAILED and any other answer stay Unknown so accepted work
            // keeps its leases until a terminal observation exists.
            _ => GlFenceStatus::Unknown,
        })
    }

    fn flush(&mut self) -> Result<(), GlError> {
        const OP: &str = "flush";
        self.assert_provider_ready(OP)?;
        self.raw.flush();
        self.driver_error(OP)
    }

    fn memory_barrier(&mut self, barriers: GlMemoryBarrier) -> Result<(), GlError> {
        barriers.validate_nonempty()?;
        // WebGL2 deliberately exposes neither `glMemoryBarrier` nor a compatible
        // extension route. Returning Unsupported is essential: a fence orders
        // completion but cannot manufacture shader/cache visibility semantics.
        Err(GlError::Unsupported {
            operation: "memory_barrier",
            reason: "WebGL2 has no glMemoryBarrier route",
        })
    }

    fn memory_barrier_by_region(&mut self, barriers: GlMemoryBarrier) -> Result<(), GlError> {
        barriers.validate_by_region()?;
        Err(GlError::Unsupported {
            operation: "memory_barrier_by_region",
            reason: "WebGL2 has no glMemoryBarrierByRegion route",
        })
    }

    fn texture_barrier(&mut self) -> Result<(), GlError> {
        Err(GlError::Unsupported {
            operation: "texture_barrier",
            reason: "WebGL2 has no texture-feedback barrier route",
        })
    }
}

/// One measurement answer, if it is an exact non-negative integer count.
///
/// Every query domain shares this reading, so a counter that is fractional,
/// negative, non-finite, or absent is reported as no answer rather than being
/// coerced into a count that was never measured.
pub(super) fn measured_value(value: f64) -> Option<u64> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0).then_some(value as u64)
}

impl WebGl2BrowserDiscovery {
    /// Reads one fence status with accepted-unknown semantics.
    fn sync_status(&self, raw: &web_sys::WebGlSync) -> GlFenceStatus {
        let value = self.raw.get_sync_parameter(raw, Gl::SYNC_STATUS);
        match value.as_f64() {
            Some(value) if value == f64::from(Gl::SIGNALED) => GlFenceStatus::Complete,
            Some(value) if value == f64::from(Gl::UNSIGNALED) => GlFenceStatus::Pending,
            _ => GlFenceStatus::Unknown,
        }
    }
}

impl GlQueryObjectsApi for WebGl2BrowserDiscovery {
    fn create_query(&mut self) -> Result<QueryId, GlError> {
        const OP: &str = "create-query";
        self.assert_provider_ready(OP)?;
        let raw = self
            .raw
            .create_query()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        let slot = Self::allocate_slot(&mut self.next_query_slot, OP)?;
        let id = QueryId::new(self.context_stamp(), slot, 0);
        self.queries.insert(
            slot,
            super::objects::BrowserQuery {
                generation: id.generation,
                raw,
                target: None,
            },
        );
        Ok(id)
    }

    fn destroy_query(&mut self, query: QueryId) -> Result<(), GlError> {
        const OP: &str = "destroy-query";
        self.query(OP, query)?;
        let entry = self
            .queries
            .remove(&query.slot)
            .ok_or_else(|| Self::validation(OP, "query disappeared"))?;
        if self.active_query == Some(query.slot) {
            self.active_query = None;
        }
        self.raw.delete_query(Some(&entry.raw));
        self.driver_error(OP)
    }

    fn query_result(&mut self, query: QueryId) -> Result<GlQueryResult, GlError> {
        const OP: &str = "query-result";
        self.assert_provider_ready(OP)?;
        self.query(OP, query)?;
        let (raw, target, active) = match self.queries.get(&query.slot) {
            Some(entry) => (
                entry.raw.clone(),
                entry.target,
                self.active_query == Some(query.slot),
            ),
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        if target.is_none() {
            return Err(Self::validation(OP, "query has no recorded measurement"));
        }
        if active {
            return Ok(GlQueryResult::Pending);
        }
        // A timer target is read through the extension's own accessors, which
        // also decide whether the interval may be reported at all.
        if target.is_some_and(super::exec_timer::is_timer_target) {
            return self.timer_result(OP, &raw);
        }
        let available = self
            .raw
            .get_query_parameter(&raw, Gl::QUERY_RESULT_AVAILABLE)
            .as_bool();
        let Some(available) = available else {
            return Ok(GlQueryResult::Unknown);
        };
        if !available {
            return Ok(GlQueryResult::Pending);
        }
        let value = self
            .raw
            .get_query_parameter(&raw, Gl::QUERY_RESULT)
            .as_f64();
        Ok(match value.and_then(measured_value) {
            Some(value) => GlQueryResult::Available(value),
            None => GlQueryResult::Unknown,
        })
    }
}

impl GlOcclusionQueryApi for WebGl2BrowserDiscovery {
    fn begin_occlusion_query(&mut self, query: QueryId) -> Result<(), GlError> {
        const OP: &str = "begin-occlusion-query";
        self.assert_provider_ready(OP)?;
        self.query(OP, query)?;
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = self
            .queries
            .get(&query.slot)
            .map(|entry| entry.raw.clone())
            .ok_or_else(|| Self::validation(OP, "query disappeared"))?;
        self.raw
            .begin_query(super::format_map::SAMPLES_PASSED, &raw);
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query.slot) {
            entry.target = Some(super::format_map::SAMPLES_PASSED);
        }
        self.active_query = Some(query.slot);
        Ok(())
    }

    fn end_occlusion_query(&mut self) -> Result<(), GlError> {
        const OP: &str = "end-occlusion-query";
        self.assert_provider_ready(OP)?;
        let Some(active) = self.active_query.take() else {
            return Err(Self::validation(OP, "no occlusion query is active"));
        };
        self.raw.end_query(super::format_map::SAMPLES_PASSED);
        // `endQuery` is the command that changes the browser's active-query
        // state.  Do not commit the local transition until its error has been
        // observed: on a recoverable API failure the same query still owns the
        // context slot and a later end must be able to close it.
        if let Err(error) = self.driver_error(OP) {
            self.active_query = Some(active);
            return Err(error);
        }
        Ok(())
    }
}
