//! Private logical resources, versions, dependencies, validation, and execution.
//!
//! Graph will lower directly to portable RHI recording and completion contracts.
//! It does not expose a public scheduler or duplicate backend GPU semantics.

mod access;
mod binding;
pub(crate) mod compiler;
mod compute;
mod copy;
mod deferred;
pub(crate) use deferred::{
    DeferredDeformationPass, DeferredGeometryPass, DeferredLightingPass, DeferredPhaseDriver,
    DeferredPlan,
};
mod definition;
mod execution;
mod forward;
#[allow(unused_imports)]
// Consumed by the coordinator once it lowers resident/frame planning facts.
pub(crate) use forward::{
    BufferGraphFact, ClearForwardPass, ForwardDrawShape, ForwardGraphDraw, ForwardGraphSpec,
    ForwardGraphTextureUpload, ForwardGraphUniformUpload, ForwardGraphVertexUpload,
    ForwardPhaseDriver, ForwardPlan, ForwardPlanBuildError, ForwardReadScope, ForwardShape,
    OpaqueForwardPass, SurfaceForwardPlan, TextureGraphFact,
};
mod raster;
pub(crate) mod read_set_raster;
mod state;

pub(crate) use binding::{BindingError, ResourceInstanceId};
pub(crate) use compute::{ComputeFixtureBindings, ComputePlan};
pub(crate) use copy::{CopyFrame, CopyPlan};
pub(crate) use raster::{RasterPlan, SurfaceRasterPlan};
pub(crate) use read_set_raster::{ReadSetRasterBindings, ReadSetRasterFrame, ReadSetRasterPlan};

#[cfg(test)]
#[path = "../tests/graph/graph_cost_diagnostic.rs"]
mod cost_diagnostic;
