//! Mock sync, presentation and query domains.
//!
//! Everything that observes or orders work rather than producing it.

use super::*;

impl GlSyncApi for MockGlFamilyApi {
    fn create_fence(&mut self) -> Result<GlFenceLease, GlError> {
        self.ready("create-fence")?;
        let id = SyncId::new(self.stamp, self.slot()?, 0);
        self.syncs.insert(id);
        let lease = self.fences.issue(id)?;
        self.calls.push(MockCall::CreateFence(lease));
        Ok(lease)
    }
    fn destroy_fence(&mut self, l: GlFenceLease) -> Result<(), GlError> {
        self.ready("destroy-fence")?;
        self.stamp("destroy-fence", l.fence.context)?;
        self.fences.validate(l)?;
        self.syncs.remove(&l.fence);
        self.fences.revoke(l);
        self.calls.push(MockCall::DestroyFence(l));
        Ok(())
    }
    fn poll_fence(&mut self, l: GlFenceLease) -> Result<GlFenceStatus, GlError> {
        self.ready("poll-fence")?;
        self.stamp("poll-fence", l.fence.context)?;
        self.fences.validate(l)?;
        self.calls.push(MockCall::PollFence(l));
        Ok(GlFenceStatus::Pending)
    }
    fn wait_fence(&mut self, l: GlFenceLease, _: GlWaitBound) -> Result<GlFenceStatus, GlError> {
        self.ready("wait-fence")?;
        self.stamp("wait-fence", l.fence.context)?;
        self.fences.validate(l)?;
        self.calls.push(MockCall::WaitFence(l));
        Ok(GlFenceStatus::Pending)
    }
    fn flush(&mut self) -> Result<(), GlError> {
        self.ready("flush")?;
        self.calls.push(MockCall::Flush);
        Ok(())
    }
}
impl GlSurfacePresentationApi for MockGlFamilyApi {
    fn acquire_surface_image(&mut self) -> Result<GlSurfaceAcquire, GlError> {
        self.ready("acquire-surface-image")?;
        if self.surface_suspended || self.surface_size.is_zero() {
            return Ok(GlSurfaceAcquire::Suspended);
        }
        let image = SurfaceImageId::new(self.stamp, self.slot()?, 0);
        let lease = self.surface.acquire(image, self.surface_size)?;
        self.calls.push(MockCall::AcquireSurface(lease));
        Ok(GlSurfaceAcquire::Lease(lease))
    }
    fn resize_surface(&mut self, size: GlSurfaceSize) -> Result<(), GlError> {
        self.ready("resize-surface")?;
        self.surface.invalidate_generation()?;
        self.surface_size = size;
        self.calls.push(MockCall::ResizeSurface(size));
        Ok(())
    }
    fn suspend_surface(&mut self) -> Result<(), GlError> {
        self.ready("suspend-surface")?;
        self.surface.invalidate_generation()?;
        self.surface_suspended = true;
        self.calls.push(MockCall::SuspendSurface);
        Ok(())
    }
    fn resume_surface(&mut self) -> Result<(), GlError> {
        self.ready("resume-surface")?;
        self.surface.invalidate_generation()?;
        self.surface_suspended = false;
        self.calls.push(MockCall::ResumeSurface);
        Ok(())
    }
    fn present_surface(&mut self, l: GlSurfaceLease) -> Result<(), GlError> {
        self.ready("present-surface")?;
        self.stamp("present-surface", l.image.context)?;
        self.surface.consume(l)?;
        self.calls.push(MockCall::PresentSurface(l));
        Ok(())
    }
}
impl GlQueryObjectsApi for MockGlFamilyApi {
    fn create_query(&mut self) -> Result<QueryId, GlError> {
        self.ready("create-query")?;
        let id = QueryId::new(self.stamp, self.slot()?, 0);
        self.queries.insert(id);
        self.calls.push(MockCall::CreateQuery(id));
        Ok(id)
    }
    fn destroy_query(&mut self, id: QueryId) -> Result<(), GlError> {
        self.ready("destroy-query")?;
        self.live("destroy-query", id, |this| this.queries.contains(&id))?;
        self.queries.remove(&id);
        self.calls.push(MockCall::DestroyQuery(id));
        Ok(())
    }
    fn query_result(&mut self, id: QueryId) -> Result<GlQueryResult, GlError> {
        self.ready("query-result")?;
        self.live("query-result", id, |this| this.queries.contains(&id))?;
        self.calls.push(MockCall::QueryResult(id));
        // Injected answers model completion for differential tests; without
        // injection the oracle stays honest about not knowing.
        Ok(self
            .query_results
            .get(&id)
            .copied()
            .unwrap_or(GlQueryResult::Pending))
    }
}
impl GlOcclusionQueryApi for MockGlFamilyApi {
    fn begin_occlusion_query(&mut self, id: QueryId) -> Result<(), GlError> {
        self.ready("begin-occlusion-query")?;
        self.live("begin-occlusion-query", id, |this| {
            this.queries.contains(&id)
        })?;
        self.calls.push(MockCall::BeginOcclusion(id));
        Ok(())
    }
    fn end_occlusion_query(&mut self) -> Result<(), GlError> {
        self.ready("end-occlusion-query")?;
        self.calls.push(MockCall::EndOcclusion);
        Ok(())
    }
}
impl GlElapsedQueryApi for MockGlFamilyApi {
    fn begin_elapsed_query(&mut self, id: QueryId) -> Result<(), GlError> {
        self.ready("begin-elapsed-query")?;
        self.live("begin-elapsed-query", id, |this| this.queries.contains(&id))?;
        self.calls.push(MockCall::BeginElapsed(id));
        Ok(())
    }
    fn end_elapsed_query(&mut self) -> Result<(), GlError> {
        self.ready("end-elapsed-query")?;
        self.calls.push(MockCall::EndElapsed);
        Ok(())
    }
}
impl GlTimestampQueryApi for MockGlFamilyApi {
    fn query_timestamp(&mut self, id: QueryId) -> Result<(), GlError> {
        self.ready("query-timestamp")?;
        self.live("query-timestamp", id, |this| this.queries.contains(&id))?;
        self.calls.push(MockCall::QueryTimestamp(id));
        Ok(())
    }
}
