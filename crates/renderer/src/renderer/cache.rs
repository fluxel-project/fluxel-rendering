use super::{NEXT_PLAN_ORDINAL, Renderer, RendererError};
use crate::graph::{
    ComputePlan, CopyFrame, CopyPlan, RasterPlan, ReadSetRasterFrame, ReadSetRasterPlan,
    SurfaceRasterPlan,
};
use fluxel_rhi::{
    api::PlanId,
    backend::{ExecutionBackend, MappingBackend, ReadbackBackend, ResourceBackend},
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum PlanKey {
    Copy,
    Raster { bytes: u64, extent: [u32; 2] },
    SurfaceRaster { bytes: u64, extent: [u32; 2] },
    Compute { bytes: u64 },
    ReadSetRaster,
}

pub(super) enum CachedPlan<B: ExecutionBackend> {
    Copy(Arc<CopyPlan<B>>),
    Raster(Arc<RasterPlan<B>>),
    SurfaceRaster(Arc<SurfaceRasterPlan<B>>),
    Compute(Arc<ComputePlan<B>>),
    ReadSetRaster(Arc<ReadSetRasterPlan<B>>),
}
impl<B: ExecutionBackend> CachedPlan<B> {
    pub(super) fn id(&self) -> PlanId {
        match self {
            Self::Copy(v) => v.id(),
            Self::Raster(v) => v.id(),
            Self::SurfaceRaster(v) => v.id(),
            Self::Compute(v) => v.id(),
            Self::ReadSetRaster(v) => v.id(),
        }
    }
}
impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    fn ordinal(&self) -> Result<u64, RendererError> {
        crate::atomic::claim_next(&NEXT_PLAN_ORDINAL).ok_or(RendererError::IdentityExhausted)
    }
    pub(super) fn copy_plan_for(
        &mut self,
        frame: &CopyFrame,
    ) -> Result<Arc<CopyPlan<B>>, RendererError> {
        let key = PlanKey::Copy;
        if let Some(CachedPlan::Copy(v)) = self.plans.get(&key) {
            v.validate_frame(frame)?;
            return Ok(Arc::clone(v));
        }
        let plan = Arc::new(CopyPlan::compile(
            &self.device,
            PlanId {
                device: self.device.id(),
                ordinal: self.ordinal()?,
            },
            frame,
        )?);
        self.plans.insert(key, CachedPlan::Copy(Arc::clone(&plan)));
        Ok(plan)
    }
    pub(super) fn raster_plan_for(
        &mut self,
        len: u64,
        extent: [u32; 2],
    ) -> Result<Arc<RasterPlan<B>>, RendererError> {
        let key = PlanKey::Raster { bytes: len, extent };
        if let Some(CachedPlan::Raster(v)) = self.plans.get(&key) {
            return Ok(Arc::clone(v));
        }
        let plan = Arc::new(RasterPlan::compile(
            &self.device,
            PlanId {
                device: self.device.id(),
                ordinal: self.ordinal()?,
            },
            len,
            extent,
        )?);
        self.plans
            .insert(key, CachedPlan::Raster(Arc::clone(&plan)));
        Ok(plan)
    }
    pub(super) fn surface_raster_plan_for(
        &mut self,
        len: u64,
        extent: [u32; 2],
    ) -> Result<Arc<SurfaceRasterPlan<B>>, RendererError> {
        let key = PlanKey::SurfaceRaster { bytes: len, extent };
        if let Some(CachedPlan::SurfaceRaster(v)) = self.plans.get(&key) {
            return Ok(Arc::clone(v));
        }
        let plan = Arc::new(SurfaceRasterPlan::compile(
            &self.device,
            PlanId {
                device: self.device.id(),
                ordinal: self.ordinal()?,
            },
            len,
            extent,
        )?);
        self.plans
            .insert(key, CachedPlan::SurfaceRaster(Arc::clone(&plan)));
        Ok(plan)
    }
    pub(super) fn compute_plan_for(
        &mut self,
        len: u64,
    ) -> Result<Arc<ComputePlan<B>>, RendererError> {
        let key = PlanKey::Compute { bytes: len };
        if let Some(CachedPlan::Compute(v)) = self.plans.get(&key) {
            return Ok(Arc::clone(v));
        }
        let plan = Arc::new(ComputePlan::compile(
            &self.device,
            PlanId {
                device: self.device.id(),
                ordinal: self.ordinal()?,
            },
            len,
        )?);
        self.plans
            .insert(key, CachedPlan::Compute(Arc::clone(&plan)));
        Ok(plan)
    }
    pub(super) fn read_set_raster_plan_for(
        &mut self,
        frame: &ReadSetRasterFrame,
    ) -> Result<Arc<ReadSetRasterPlan<B>>, RendererError> {
        let key = PlanKey::ReadSetRaster;
        if let Some(CachedPlan::ReadSetRaster(v)) = self.plans.get(&key) {
            v.validate_frame(frame)?;
            return Ok(Arc::clone(v));
        }
        let plan = Arc::new(ReadSetRasterPlan::compile(
            &self.device,
            PlanId {
                device: self.device.id(),
                ordinal: self.ordinal()?,
            },
            frame,
        )?);
        self.plans
            .insert(key, CachedPlan::ReadSetRaster(Arc::clone(&plan)));
        Ok(plan)
    }
}
