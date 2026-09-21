//! One authoritative feature-route registry for the GL family.
//!
//! A feature is admitted only by the route recorded here: core version or an
//! exact extension spelling, plus every command token its lowering needs.
//! `facts` owns mutable per-context observations; it deliberately does not
//! own a second requirement table.

mod compute;
mod debug;
mod draw;
mod query;
mod sync;
mod texture;

use std::collections::BTreeSet;

use super::{
    api::GlKnownExtension,
    profile::{GlProfile, GlVersion},
};

pub(crate) use super::api::GlKnownExtension as GlExtension;

/// Native functions or browser extension-object methods a lowering needs.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlFunction {
    DispatchCompute,
    DispatchComputeIndirect,
    ShaderStorageBlockBinding,
    BindImageTexture,
    MemoryBarrier,
    MemoryBarrierByRegion,
    TextureBarrier,
    TextureBarrierNv,
    DrawArraysIndirect,
    DrawElementsIndirect,
    MultiDrawArrays,
    MultiDrawElements,
    MultiDrawArraysIndirect,
    MultiDrawElementsIndirect,
    MultiDrawArraysIndirectCount,
    MultiDrawElementsIndirectCount,
    QueryCounter,
    GetQueryObjectUi64,
    QueryCounterExt,
    GetQueryObjectExt,
    BeginQuery,
    EndQuery,
    GetQueryObject,
    TexParameterAnisotropy,
    CompressedTexImage2d,
    CompressedTexSubImage2d,
    FramebufferTextureMultiview,
    PushDebugGroup,
    PopDebugGroup,
    ObjectLabel,
    FenceSync,
    ClientWaitSync,
    DeleteSync,
    WebGlMultiDrawArrays,
    WebGlMultiDrawElements,
}

impl GlFunction {
    /// Exact native token for this route. Browser extension-object methods have
    /// no native spelling and are intentionally never admitted by WGL/EGL.
    pub(crate) const fn native_symbol(self) -> Option<&'static str> {
        Some(match self {
            Self::DispatchCompute => "glDispatchCompute",
            Self::DispatchComputeIndirect => "glDispatchComputeIndirect",
            Self::ShaderStorageBlockBinding => "glShaderStorageBlockBinding",
            Self::BindImageTexture => "glBindImageTexture",
            Self::MemoryBarrier => "glMemoryBarrier",
            Self::MemoryBarrierByRegion => "glMemoryBarrierByRegion",
            Self::TextureBarrier => "glTextureBarrier",
            Self::TextureBarrierNv => "glTextureBarrierNV",
            Self::DrawArraysIndirect => "glDrawArraysIndirect",
            Self::DrawElementsIndirect => "glDrawElementsIndirect",
            Self::MultiDrawArrays => "glMultiDrawArrays",
            Self::MultiDrawElements => "glMultiDrawElements",
            Self::MultiDrawArraysIndirect => "glMultiDrawArraysIndirect",
            Self::MultiDrawElementsIndirect => "glMultiDrawElementsIndirect",
            Self::MultiDrawArraysIndirectCount => "glMultiDrawArraysIndirectCount",
            Self::MultiDrawElementsIndirectCount => "glMultiDrawElementsIndirectCount",
            Self::QueryCounter => "glQueryCounter",
            Self::GetQueryObjectUi64 => "glGetQueryObjectui64v",
            Self::BeginQuery => "glBeginQuery",
            Self::EndQuery => "glEndQuery",
            Self::GetQueryObject => "glGetQueryObjectuiv",
            Self::TexParameterAnisotropy => "glTexParameterf",
            Self::CompressedTexImage2d => "glCompressedTexImage2D",
            Self::CompressedTexSubImage2d => "glCompressedTexSubImage2D",
            Self::FramebufferTextureMultiview => "glFramebufferTextureMultiviewOVR",
            Self::PushDebugGroup => "glPushDebugGroup",
            Self::PopDebugGroup => "glPopDebugGroup",
            Self::ObjectLabel => "glObjectLabel",
            Self::FenceSync => "glFenceSync",
            Self::ClientWaitSync => "glClientWaitSync",
            Self::DeleteSync => "glDeleteSync",
            Self::QueryCounterExt
            | Self::GetQueryObjectExt
            | Self::WebGlMultiDrawArrays
            | Self::WebGlMultiDrawElements => return None,
        })
    }
}

/// Private evidence vocabulary. Public capabilities are published only after
/// the selected route is callable by the corresponding lowering.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlFeature {
    Compute,
    IndirectDispatch,
    ShaderStorageBuffers,
    ImageLoadStore,
    IndirectDraw,
    MultiDrawIndirect,
    MultiDraw,
    IndirectCount,
    TimerQuery,
    OcclusionQuery,
    SamplerAnisotropy,
    CompressionBcS3tc,
    CompressionBcRgtc,
    CompressionBcBptc,
    CompressionBc,
    CompressionEtc2Eac,
    CompressionAstcLdr,
    CompressionAstcHdr,
    Multiview,
    DebugMarkers,
    Sync,
    MemoryBarrier,
    MemoryBarrierByRegion,
    TextureBarrier,
}

/// The route actually selected for one context.  Keeping this evidence lets a
/// caller distinguish GL 4.5 core from ARB and NV texture-barrier lowering;
/// it must not guess an entry token from the portable feature name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SelectedRoute {
    Core,
    Extension(GlExtension),
}

/// Selects a legal version/extension route without considering whether its
/// command tokens loaded. `facts` owns those observations; route policy lives
/// entirely here.
pub(crate) fn selected_route(
    feature: GlFeature,
    profile: GlProfile,
    extensions: &BTreeSet<GlExtension>,
) -> Option<SelectedRoute> {
    if feature == GlFeature::CompressionBc {
        return [
            GlFeature::CompressionBcS3tc,
            GlFeature::CompressionBcRgtc,
            GlFeature::CompressionBcBptc,
        ]
        .into_iter()
        .all(|part| selected_route(part, profile, extensions).is_some())
        .then_some(SelectedRoute::Core);
    }
    match feature {
        GlFeature::MultiDraw => {
            if matches!(profile, GlProfile::Desktop { .. }) && profile.has_core(GL14, NEVER) {
                return Some(SelectedRoute::Core);
            }
            if extensions.contains(&GlExtension::ExtMultiDrawArrays) {
                return Some(SelectedRoute::Extension(GlExtension::ExtMultiDrawArrays));
            }
            return extensions
                .contains(&GlExtension::WebglMultiDraw)
                .then_some(SelectedRoute::Extension(GlExtension::WebglMultiDraw));
        }
        GlFeature::TimerQuery => {
            if profile.has_core(GL33, NEVER) {
                return Some(SelectedRoute::Core);
            }
            if extensions.contains(&GlExtension::ArbTimerQuery) {
                return Some(SelectedRoute::Extension(GlExtension::ArbTimerQuery));
            }
            return extensions
                .contains(&GlExtension::ExtDisjointTimerQuery)
                .then_some(SelectedRoute::Extension(GlExtension::ExtDisjointTimerQuery));
        }
        GlFeature::TextureBarrier => {
            if matches!(profile, GlProfile::Desktop { .. }) && profile.has_core(GL45, NEVER) {
                return Some(SelectedRoute::Core);
            }
            if extensions.contains(&GlExtension::ArbTextureBarrier) {
                return Some(SelectedRoute::Extension(GlExtension::ArbTextureBarrier));
            }
            return extensions
                .contains(&GlExtension::NvTextureBarrier)
                .then_some(SelectedRoute::Extension(GlExtension::NvTextureBarrier));
        }
        _ => {}
    }
    let requirement = requirement(feature)?;
    if profile.has_core(requirement.desktop_core, requirement.gles_core) {
        Some(SelectedRoute::Core)
    } else {
        requirement
            .extensions
            .iter()
            .copied()
            .find(|extension| extensions.contains(extension))
            .map(SelectedRoute::Extension)
    }
}

/// Exact command group for a selected route, including incompatible extension
/// object/native carriers.
pub(crate) fn route_functions(feature: GlFeature, route: SelectedRoute) -> &'static [GlFunction] {
    match (feature, route) {
        (GlFeature::MultiDraw, SelectedRoute::Extension(GlExtension::WebglMultiDraw)) => &[
            GlFunction::WebGlMultiDrawArrays,
            GlFunction::WebGlMultiDrawElements,
        ],
        (GlFeature::MultiDraw, _) => &[GlFunction::MultiDrawArrays, GlFunction::MultiDrawElements],
        (GlFeature::TimerQuery, SelectedRoute::Extension(GlExtension::ExtDisjointTimerQuery)) => {
            &[GlFunction::QueryCounterExt, GlFunction::GetQueryObjectExt]
        }
        (GlFeature::TimerQuery, _) => &[GlFunction::QueryCounter, GlFunction::GetQueryObjectUi64],
        (GlFeature::TextureBarrier, SelectedRoute::Extension(GlExtension::NvTextureBarrier)) => {
            &[GlFunction::TextureBarrierNv]
        }
        (GlFeature::TextureBarrier, _) => &[GlFunction::TextureBarrier],
        (GlFeature::CompressionBc, _) => &[],
        _ => {
            requirement(feature)
                .expect("ordinary feature has a route")
                .functions
        }
    }
}

pub(crate) fn supports(
    feature: GlFeature,
    profile: GlProfile,
    extensions: &BTreeSet<GlExtension>,
    functions: &BTreeSet<GlFunction>,
) -> bool {
    if feature == GlFeature::CompressionBc {
        return [
            GlFeature::CompressionBcS3tc,
            GlFeature::CompressionBcRgtc,
            GlFeature::CompressionBcBptc,
        ]
        .into_iter()
        .all(|part| supports(part, profile, extensions, functions));
    }
    selected_route(feature, profile, extensions).is_some_and(|route| {
        route_functions(feature, route)
            .iter()
            .all(|function| functions.contains(function))
    })
}

pub(crate) fn missing_function(
    feature: GlFeature,
    profile: GlProfile,
    extensions: &BTreeSet<GlExtension>,
    functions: &BTreeSet<GlFunction>,
) -> Option<GlFunction> {
    if feature == GlFeature::CompressionBc {
        return [
            GlFeature::CompressionBcS3tc,
            GlFeature::CompressionBcRgtc,
            GlFeature::CompressionBcBptc,
        ]
        .into_iter()
        .find_map(|part| missing_function(part, profile, extensions, functions));
    }
    let route = selected_route(feature, profile, extensions)?;
    route_functions(feature, route)
        .iter()
        .copied()
        .find(|function| !functions.contains(function))
}

/// A declarative core-or-extension route. Extensions are alternatives, never
/// inferred merely because another profile exposes a similarly named symbol.
#[derive(Clone, Copy)]
pub(crate) struct Requirement {
    pub(crate) desktop_core: GlVersion,
    pub(crate) gles_core: GlVersion,
    pub(crate) extensions: &'static [GlKnownExtension],
    pub(crate) functions: &'static [GlFunction],
}

pub(crate) const NEVER: GlVersion = GlVersion::new(u8::MAX, u8::MAX);
pub(crate) const GL15: GlVersion = GlVersion::new(1, 5);
pub(crate) const GL14: GlVersion = GlVersion::new(1, 4);
pub(crate) const GL30: GlVersion = GlVersion::new(3, 0);
pub(crate) const GL32: GlVersion = GlVersion::new(3, 2);
pub(crate) const GL33: GlVersion = GlVersion::new(3, 3);
pub(crate) const GL40: GlVersion = GlVersion::new(4, 0);
pub(crate) const GL42: GlVersion = GlVersion::new(4, 2);
pub(crate) const GL43: GlVersion = GlVersion::new(4, 3);
pub(crate) const GL45: GlVersion = GlVersion::new(4, 5);
pub(crate) const GL46: GlVersion = GlVersion::new(4, 6);
pub(crate) const GLES30: GlVersion = GlVersion::new(3, 0);
pub(crate) const GLES31: GlVersion = GlVersion::new(3, 1);
pub(crate) const GLES32: GlVersion = GlVersion::new(3, 2);

/// Returns the unique declarative route for ordinary feature families.
/// Composite features (`CompressionBc`, carrier-dependent multi-draw and
/// timer-query) are resolved explicitly by `GlFeatureProbe` because their
/// alternatives require different command tokens.
pub(crate) fn requirement(feature: GlFeature) -> Option<Requirement> {
    compute::requirement(feature)
        .or_else(|| sync::requirement(feature))
        .or_else(|| draw::requirement(feature))
        .or_else(|| query::requirement(feature))
        .or_else(|| texture::requirement(feature))
        .or_else(|| debug::requirement(feature))
}

/// The exact callable token group for a semantic feature.  The three
/// carrier-dependent families are centralized here instead of leaving native
/// entry-point code to recreate their function lists.
pub(crate) fn required_functions(feature: GlFeature) -> &'static [GlFunction] {
    match feature {
        GlFeature::MultiDraw => &[GlFunction::MultiDrawArrays, GlFunction::MultiDrawElements],
        GlFeature::TimerQuery => &[GlFunction::QueryCounter, GlFunction::GetQueryObjectUi64],
        GlFeature::CompressionBc => &[],
        _ => {
            requirement(feature)
                .expect("ordinary feature has a route")
                .functions
        }
    }
}
