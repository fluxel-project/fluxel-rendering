//! Native fence, query, and timer execution.
//!
//! Fence observations are accepted-unknown: a `WAIT_FAILED` or any answer that
//! is not a usable status reports `Unknown` instead of guessing, so leases
//! stay live and completion-safe release keeps waiting. Timer queries are the
//! core GL 3.3/GLES 3.0 `TIME_ELAPSED`/`TIMESTAMP` domains; the desktop and
//! embedded families have no disjoint concept (that instrumentation belongs
//! to the WebGL disjoint timer extension), so `GlQueryResult::Disjoint` is
//! never produced here and a nonzero counter width is real evidence.

use super::super::{
    GlElapsedQueryApi, GlError, GlFamilyApi as _, GlFenceLease, GlFenceStatus, GlOcclusionQueryApi,
    GlQueryObjectsApi, GlQueryResult, GlSyncApi, GlTimestampQueryApi, GlWaitBound, QueryId, SyncId,
};
use super::provider::{NativeGlProvider, QueryTarget};

/// `client_waitSync` accepts a 64-bit timeout, but glow 0.18 binds it as
/// `i32` nanoseconds. A larger bounded wait is clamped to that ceiling
/// (about two seconds); a longer effective wait stays expressible through
/// repeated polls, and zero remains a pure poll.
const MAX_WAIT_TIMEOUT_NS: i32 = i32::MAX;

impl GlSyncApi for NativeGlProvider<'_> {
    fn create_fence(&mut self) -> Result<GlFenceLease, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-fence";
        self.assert_ready(OP)?;
        // SAFETY: current-context contract; the fence is deleted if the Fluxel
        // lease book rejects the issue.
        let raw = unsafe { self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) }.map_err(
            |message| GlError::Driver {
                operation: OP,
                message,
            },
        )?;
        let slot = self.slot(OP)?;
        let id = SyncId::new(self.context_stamp(), slot, 0);
        self.syncs.insert(id, raw);
        match self.fences.issue(id) {
            Ok(lease) => Ok(lease),
            Err(error) => {
                if let Some(entry) = self.syncs.remove(&id) {
                    // SAFETY: current-context contract; rollback of this
                    // transaction's own object.
                    unsafe { self.gl.delete_sync(entry) };
                }
                Err(error)
            }
        }
    }

    fn destroy_fence(&mut self, fence: GlFenceLease) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-fence";
        self.assert_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let raw = self.sync(OP, fence.fence)?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_sync(raw) };
        self.syncs.remove(&fence.fence);
        self.fences.revoke(fence);
        self.driver_error(OP)
    }

    fn poll_fence(&mut self, fence: GlFenceLease) -> Result<GlFenceStatus, GlError> {
        const OP: &str = "poll-fence";
        self.assert_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let raw = self.sync(OP, fence.fence)?;
        Ok(self.sync_status(raw))
    }

    fn wait_fence(
        &mut self,
        fence: GlFenceLease,
        bound: GlWaitBound,
    ) -> Result<GlFenceStatus, GlError> {
        use glow::HasContext as _;
        const OP: &str = "wait-fence";
        self.assert_ready(OP)?;
        self.validate_object_context(OP, fence.fence.context)?;
        self.fences.validate(fence)?;
        let raw = self.sync(OP, fence.fence)?;
        // A zero bound is a poll: no flush flag, no wait.
        let flags = if bound.nanoseconds == 0 {
            0
        } else {
            glow::SYNC_FLUSH_COMMANDS_BIT
        };
        let timeout = if bound.nanoseconds == 0 {
            0
        } else {
            u64::try_from(MAX_WAIT_TIMEOUT_NS).unwrap_or(i32::MAX as u64) as i32
        };
        // SAFETY: current-context contract; the fence is live.
        let result = unsafe { self.gl.client_wait_sync(raw, flags, timeout) };
        Ok(match result {
            glow::ALREADY_SIGNALED | glow::CONDITION_SATISFIED => GlFenceStatus::Complete,
            glow::TIMEOUT_EXPIRED => GlFenceStatus::Pending,
            // WAIT_FAILED and any other answer stay Unknown so accepted work
            // keeps its leases until a terminal observation exists.
            _ => GlFenceStatus::Unknown,
        })
    }

    fn flush(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "flush";
        self.assert_ready(OP)?;
        // SAFETY: current-context contract.
        unsafe { self.gl.flush() };
        self.driver_error(OP)
    }
}

impl NativeGlProvider<'_> {
    /// Reads one fence status with accepted-unknown semantics.
    fn sync_status(&self, raw: glow::NativeFence) -> GlFenceStatus {
        use glow::HasContext as _;
        // SAFETY: current-context contract; the fence is live.
        let value = unsafe { self.gl.get_sync_parameter_i32(raw, glow::SYNC_STATUS) };
        match value as u32 {
            glow::SIGNALED => GlFenceStatus::Complete,
            glow::UNSIGNALED => GlFenceStatus::Pending,
            _ => GlFenceStatus::Unknown,
        }
    }

    fn sync(&self, operation: &'static str, id: SyncId) -> Result<glow::NativeFence, GlError> {
        self.validate_object_context(operation, id.context)?;
        // Map keys are full identities, so a hit implies the same generation.
        self.syncs
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "fence is not live"))
    }
}

impl GlQueryObjectsApi for NativeGlProvider<'_> {
    fn create_query(&mut self) -> Result<QueryId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-query";
        self.assert_ready(OP)?;
        // SAFETY: current-context contract.
        let raw = unsafe { self.gl.create_query() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        let slot = self.slot(OP)?;
        let id = QueryId::new(self.context_stamp(), slot, 0);
        self.queries.insert(
            id,
            super::provider::NativeQuery {
                generation: id.generation,
                raw,
                target: None,
            },
        );
        Ok(id)
    }

    fn destroy_query(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-query";
        self.assert_ready(OP)?;
        self.query(OP, query)?;
        let entry = self
            .queries
            .remove(&query)
            .ok_or_else(|| Self::validation(OP, "query disappeared"))?;
        if self.active_query == Some(query) {
            self.active_query = None;
        }
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_query(entry.raw) };
        self.driver_error(OP)
    }

    fn query_result(&mut self, query: QueryId) -> Result<GlQueryResult, GlError> {
        use glow::HasContext as _;
        const OP: &str = "query-result";
        self.assert_ready(OP)?;
        self.query(OP, query)?;
        let (raw, target, active) = match self.queries.get(&query) {
            Some(entry) => (entry.raw, entry.target, self.active_query == Some(query)),
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        if target.is_none() {
            return Err(Self::validation(OP, "query has no recorded measurement"));
        }
        if active {
            return Ok(GlQueryResult::Pending);
        }
        // SAFETY: current-context contract; the query is live and inactive.
        let available = unsafe {
            self.gl
                .get_query_parameter_u32(raw, glow::QUERY_RESULT_AVAILABLE)
        };
        // A driver answer that is not exactly zero or one is indeterminate;
        // accepted-unknown keeps the observation honest.
        if available == 0 {
            return Ok(GlQueryResult::Pending);
        }
        if available != 1 {
            return Ok(GlQueryResult::Unknown);
        }
        let value = unsafe { self.gl.get_query_parameter_u64(raw, glow::QUERY_RESULT) };
        Ok(GlQueryResult::Available(value))
    }
}

impl GlOcclusionQueryApi for NativeGlProvider<'_> {
    fn begin_occlusion_query(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "begin-occlusion-query";
        self.assert_ready(OP)?;
        self.query(OP, query)?;
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = match self.queries.get(&query) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        // SAFETY: current-context contract; the query is live and idle.
        unsafe { self.gl.begin_query(glow::SAMPLES_PASSED, raw) };
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query) {
            entry.target = Some(QueryTarget::SamplesPassed);
        }
        self.active_query = Some(query);
        Ok(())
    }

    fn end_occlusion_query(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "end-occlusion-query";
        self.assert_ready(OP)?;
        let Some(active) = self.active_query.take() else {
            return Err(Self::validation(OP, "no occlusion query is active"));
        };
        let Some(entry) = self.queries.get(&active) else {
            return Err(Self::validation(OP, "query disappeared"));
        };
        if entry.target != Some(QueryTarget::SamplesPassed) {
            self.active_query = Some(active);
            return Err(Self::validation(
                OP,
                "the active query is not an occlusion query",
            ));
        }
        // SAFETY: current-context contract.
        unsafe { self.gl.end_query(glow::SAMPLES_PASSED) };
        self.driver_error(OP)
    }
}

impl GlElapsedQueryApi for NativeGlProvider<'_> {
    fn begin_elapsed_query(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "begin-elapsed-query";
        self.assert_ready(OP)?;
        self.require_timer_capability(OP)?;
        self.query(OP, query)?;
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = match self.queries.get(&query) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        // SAFETY: current-context contract; the query is live and idle, and
        // the timer capability was proved by discovery.
        unsafe { self.gl.begin_query(glow::TIME_ELAPSED, raw) };
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query) {
            entry.target = Some(QueryTarget::TimeElapsed);
        }
        self.active_query = Some(query);
        Ok(())
    }

    fn end_elapsed_query(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "end-elapsed-query";
        self.assert_ready(OP)?;
        self.require_timer_capability(OP)?;
        let Some(active) = self.active_query.take() else {
            return Err(Self::validation(OP, "no elapsed query is active"));
        };
        let Some(entry) = self.queries.get(&active) else {
            return Err(Self::validation(OP, "query disappeared"));
        };
        if entry.target != Some(QueryTarget::TimeElapsed) {
            self.active_query = Some(active);
            return Err(Self::validation(
                OP,
                "the active query is not an elapsed query",
            ));
        }
        // SAFETY: current-context contract.
        unsafe { self.gl.end_query(glow::TIME_ELAPSED) };
        self.driver_error(OP)
    }
}

impl GlTimestampQueryApi for NativeGlProvider<'_> {
    fn query_timestamp(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "query-timestamp";
        self.assert_ready(OP)?;
        self.require_timer_capability(OP)?;
        self.query(OP, query)?;
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = match self.queries.get(&query) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        // SAFETY: current-context contract; a timestamp write never becomes an
        // active begin/end pair and the counter width was probed.
        unsafe { self.gl.query_counter(raw, glow::TIMESTAMP) };
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query) {
            entry.target = Some(QueryTarget::Timestamp);
        }
        Ok(())
    }
}

impl NativeGlProvider<'_> {
    /// Timer domains execute only on the proved `TimerQuery` capability:
    /// resolved core evidence plus a nonzero `GL_QUERY_COUNTER_BITS` answer.
    fn require_timer_capability(&self, operation: &'static str) -> Result<(), GlError> {
        if self
            .discovery
            .capabilities()
            .supports(super::super::GlCapability::TimerQuery)
        {
            Ok(())
        } else {
            Err(GlError::Unsupported {
                operation,
                reason: "this context did not prove a nonzero timer-query counter width",
            })
        }
    }
}
