//! The Layer 1 surface Layer 2 is written against, as three bounds.
//!
//! Layer 2 is generic over its backend rather than holding a trait object.  The
//! reason is not speed: it is that the *presence* of an optional command domain
//! is a fact about the backend type, and a bound makes "this profile has no
//! compute domain" a compile-time property of the code path instead of a
//! runtime branch that a missing method would have to answer with a stub.  The
//! plan and the state-machine handoff both require that a profile without the
//! optional domains has no empty compute state group, and an `impl` block
//! bounded on the trait is exactly that statement.
//!
//! # The three bounds
//!
//! - [`GlStateBackend`] is what every command backend implements: the three
//!   providers that execute commands (`native`, `browser`, `mock`) all do.  It
//!   is the bound on the machine itself and on every required state domain.
//! - [`GlOptionalComputeBackend`] and [`GlOptionalIndirectBackend`] are not.
//!   `native` and `mock` implement them and the browser provider does not, so
//!   the bounded `impl` blocks are the only place those domains can be reached.
//! - What the *trait* implements and what the *resolved snapshot* supports are
//!   different questions, and both are asked.  A backend that implements the
//!   compute trait on a context whose capability row is unresolved must still
//!   refuse before any side effect, which is why the optional-domain entry
//!   points read the discovery snapshot as well as the bound.
//!
//! # What is deliberately not in any bound
//!
//! The context providers -- the WGL and EGL surface owners -- implement only
//! the family trait and presentation, because they own a context and a
//! drawable rather than a command vocabulary.  They are therefore not
//! `GlStateBackend`s, and a state machine cannot be built directly on one: the
//! executable backend for such a context is the native provider built over it.
//! Writing this down matters because "the provider that has the GL context"
//! and "the provider that executes commands" are easy to conflate.

use crate::webgl2::api::{
    GlBindingApi, GlComputeDispatchApi, GlCopyDomainApi, GlDispatchIndirectApi, GlDrawIndirectApi,
    GlElapsedQueryApi, GlFamilyApi, GlFramebufferApi, GlMultiDrawApi, GlOcclusionQueryApi,
    GlQueryObjectsApi, GlRasterCommandApi, GlResourceApi, GlSamplerApi, GlShaderApi,
    GlStorageBufferApi, GlStorageImageApi, GlSurfacePresentationApi, GlSyncApi,
    GlTimestampQueryApi, GlVertexApi,
};

/// Every command domain the three executable backends implement.
///
/// A domain belongs here when `native`, `browser` and `mock` all implement its
/// trait; that is the test this list was built from, and it is why the traits
/// for occlusion, elapsed and timestamp queries are here despite the
/// capabilities they serve being optional on a real profile.  The trait's
/// presence means the layer can *express* the request, not that the profile can
/// serve it -- capability resolution is the snapshot's answer, and an
/// unsupported capability is refused before any side effect.
pub(crate) trait GlStateBackend:
    GlFamilyApi
    + GlResourceApi
    + GlSamplerApi
    + GlCopyDomainApi
    + GlFramebufferApi
    + GlRasterCommandApi
    + GlShaderApi
    + GlBindingApi
    + GlVertexApi
    + GlSyncApi
    + GlQueryObjectsApi
    + GlOcclusionQueryApi
    + GlElapsedQueryApi
    + GlTimestampQueryApi
    + GlMultiDrawApi
    + GlSurfacePresentationApi
{
}

impl<T> GlStateBackend for T where
    T: GlFamilyApi
        + GlResourceApi
        + GlSamplerApi
        + GlCopyDomainApi
        + GlFramebufferApi
        + GlRasterCommandApi
        + GlShaderApi
        + GlBindingApi
        + GlVertexApi
        + GlSyncApi
        + GlQueryObjectsApi
        + GlOcclusionQueryApi
        + GlElapsedQueryApi
        + GlTimestampQueryApi
        + GlMultiDrawApi
        + GlSurfacePresentationApi
{
}

/// A backend with the optional compute, storage and storage-image domains.
///
/// The three travel together because they are the same capability decision on
/// every shipped profile: a context that resolves one of them resolves the
/// others, and `native` gates all three on the same desktop floor.
pub(crate) trait GlOptionalComputeBackend:
    GlStateBackend + GlComputeDispatchApi + GlStorageBufferApi + GlStorageImageApi
{
}

impl<T> GlOptionalComputeBackend for T where
    T: GlStateBackend + GlComputeDispatchApi + GlStorageBufferApi + GlStorageImageApi
{
}

/// A backend with the optional indirect draw and dispatch domains.
///
/// Separate from the compute bound because they are separable in fact: the
/// single indirect commands are core on the versions that have them while the
/// browser provider has no indirect route at all, and a future profile could
/// have one without the other.
pub(crate) trait GlOptionalIndirectBackend:
    GlStateBackend + GlDrawIndirectApi + GlDispatchIndirectApi
{
}

impl<T> GlOptionalIndirectBackend for T where
    T: GlStateBackend + GlDrawIndirectApi + GlDispatchIndirectApi
{
}
