//! Narrow compute domain for the Deferred deformation witness only.

use super::*;

impl ComputeOps<WitnessBackend> for WitnessCompute {
    const AVAILABLE: bool = true;

    type PipelineFuture<'a>
        = Ready<Result<()>>
    where
        WitnessBackend: 'a;

    fn create_pipeline<'a>(
        _: &'a Arc<Native>,
        _: &'a ComputePipelineDesc<WitnessBackend>,
    ) -> Self::PipelineFuture<'a> {
        ready(Ok(()))
    }

    fn begin(_: &mut ()) -> Result<()> {
        Ok(())
    }

    fn set_pipeline(_: &mut (), _: &()) -> Result<()> {
        Ok(())
    }

    fn set_bind_group(_: &mut (), _: u32, _: &Arc<Native>, _: &[u32]) -> Result<()> {
        Ok(())
    }

    fn dispatch(_: &mut (), _: [u32; 3]) -> Result<()> {
        Ok(())
    }

    fn dispatch_indirect(_: &mut (), _: &Arc<NativeBuffer>, _: u64) -> Result<()> {
        Ok(())
    }

    fn end(_: &mut ()) -> Result<()> {
        Ok(())
    }

    fn abort(_: &mut ()) {}
}
