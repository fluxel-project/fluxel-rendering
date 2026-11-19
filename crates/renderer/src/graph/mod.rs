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
    DeferredCaptureOverlay, DeferredCapturePlan, DeferredDeformationPass,
    DeferredForwardCapturePlan, DeferredGeometryPass, DeferredLightingPass, DeferredOitPlan,
    DeferredPhaseDriver, DeferredPlan, DeferredTransparentPlan, ForwardDeferredCapturePlan,
};
mod definition;
mod execution;
mod forward;
pub(crate) mod indirect;
mod mip_generation;
pub(crate) mod oit;
pub(crate) mod oit_record;
mod owner_validation;
pub(crate) mod particle_update;
#[allow(unused_imports)]
// Consumed by the coordinator once it lowers resident/frame planning facts.
pub(crate) use forward::{
    BufferGraphFact, CaptureForwardPlan, CaptureForwardSurfacePlan, CaptureOitPlan,
    ClearForwardPass, DrawReads, ForwardDrawShape, ForwardGraphDraw, ForwardGraphSpec,
    ForwardGraphTextureUpload, ForwardGraphUniformUpload, ForwardGraphVertexUpload,
    ForwardPhaseDriver, ForwardPlan, ForwardPlanBuildError, ForwardReadScope, ForwardReadState,
    ForwardShape, OpaqueForwardPass, SurfaceForwardPlan, TextureGraphFact, material_read_contract,
};
mod raster;
pub(crate) mod read_set_raster;
mod state;

pub(crate) use binding::{BindingError, ResourceInstanceId};
pub(crate) use compute::{ComputeFixtureBindings, ComputePlan};
pub(crate) use copy::{CopyFrame, CopyPlan};
use definition::{TextureRange, texture_format_aspects};
use owner_validation::validate_resident_texture_devices;
pub(crate) use raster::{RasterPlan, SurfaceRasterPlan};
pub(crate) use read_set_raster::{ReadSetRasterBindings, ReadSetRasterFrame, ReadSetRasterPlan};

#[cfg(test)]
#[path = "../tests/graph/graph_cost_diagnostic.rs"]
mod cost_diagnostic;

pub(crate) use forward::{ForwardOitInput, ForwardOitReadContext};

fn material_aspect(desc: &fluxel_rhi::api::TextureDesc) -> fluxel_rhi::api::TextureAspects {
    texture_format_aspects(desc.format)
}

fn material_mips_layers(desc: &fluxel_rhi::api::TextureDesc) -> TextureRange {
    TextureRange::new(
        material_aspect(desc),
        0..desc.mip_levels,
        0..desc.array_layers,
    )
}

fn material_mip_layer(desc: &fluxel_rhi::api::TextureDesc, mip: u32, layer: u32) -> TextureRange {
    TextureRange::new(material_aspect(desc), mip..mip + 1, layer..layer + 1)
}
