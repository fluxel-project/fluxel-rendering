//! Resources, routes, upload, and readback (specification sections 9 through 18).
//!
//! This module is the composition entry for the resource chapter. It owns the
//! tree's shape and the names developers type; the rules themselves live in the
//! submodules:
//!
//! ```text
//! buffer       usage bits, placement preference, buffer creation   (11.1 - 12.4)
//! texture      texture usage, dimension, descriptor, creation       (13.1 - 13.4)
//! subresource  aspects, subresource ranges, origin, host layout     (14.1 - 14.5)
//! view         texture views                                       (15.1 - 15.3)
//! sampler      samplers                                            (16.1 - 16.2)
//! route        whether a portable copy/blit route exists            (9.1 - 9.4)
//! transfer     upload jobs and readback tickets                    (17.1 - 18.8)
//! ```
//!
//! Format facts are not here: [`crate::api::format`] owns them, and both
//! modules answer in the same vocabulary on purpose — a caller compares what a
//! format can do against what a descriptor may be created as without learning
//! two shapes.
//!
//! # The device verbs this chapter assigns to `Device`, and why they are not
//! # written yet
//!
//! Sections 12 through 18 declare their creation verbs in `impl Device` blocks,
//! because that is where a caller looks for them. Rust allows that inherent impl
//! to live in another module of the same crate, so each verb belongs next to the
//! descriptor it consumes — which is here.
//!
//! None of them can be written yet. `impl Device` requires `crate::api::platform`,
//! which is written but not declared (see [`crate::api`]), and every one of them
//! must reach a backend. The verbs are therefore listed here by their exact
//! specification signatures, so that the debt is named rather than implied:
//!
//! ```text
//! Device::create_buffer(&self, desc: &BufferDescriptor) -> RhiResult<Buffer>
//! Device::create_texture(&self, desc: &TextureDescriptor) -> RhiResult<Texture>
//! Device::create_texture_view(&self, texture: &Texture, desc: &TextureViewDescriptor)
//!     -> RhiResult<TextureView>
//! Device::create_sampler(&self, desc: &SamplerDescriptor) -> RhiResult<Sampler>
//! Device::create_buffer_upload(&self, desc: BufferUploadDescriptor) -> RhiResult<UploadJob>
//! Device::create_texture_upload(&self, desc: TextureUploadDescriptor) -> RhiResult<UploadJob>
//! ```
//!
//! Three further verbs belong to this chapter but to other owners, and are
//! recorded here only so that a reader looking for them in this tree is not left
//! guessing:
//!
//! ```text
//! CommandRecorder::encode_upload(&mut self, upload: &UploadJob) -> RhiResult<()>
//! CommandRecorder::encode_readback(&mut self, request: ReadbackRequest)
//!     -> RhiResult<ReadbackTicket>
//!     -- section 18.5; owner is api::command (adjudication A7, 0.16-plan.md)
//!
//! EnabledCapabilities::texture_view_format_compatible(&self, base, view) -> bool
//!     -- section 8.5; the receiver is owned by module 01
//! ```
//!
//! What *is* written here is everything a descriptor can be asked about without a
//! device: the descriptor types and their builders, the opaque resource tokens
//! and their readback accessors, the usage bitsets, and the portable validation
//! rules. The rules are `pub(crate) fn validate_*` functions in the module that
//! owns the descriptor they check, so that the device façade calls one rule
//! rather than restating it, and so that the contract tests in
//! `crate::api::tests` can drive every accept and reject path without a GPU.
//!
//! One consequence of section 3.1 is visible in those signatures: a validator
//! takes the *capability answer* it must respect (`&BufferSupport`,
//! `&TextureSupport`, alignments) as a parameter instead of reading a device.
//! That keeps the rule portable and testable, and leaves the device responsible
//! only for producing the facts.

pub mod buffer;
pub mod route;
pub mod sampler;
pub mod subresource;
pub mod texture;
pub mod transfer;
pub mod view;

pub use buffer::{
    Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferSupport, BufferSupportLimits,
    BufferSupportQuery, BufferUsage, ResourceMemoryPreference,
};
pub use route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport, TexelCopyLayoutLimits,
};
pub use sampler::{AddressMode, CompareFunction, FilterMode, Sampler, SamplerDescriptor};
pub use subresource::{
    HostTexelLayout, Origin3d, TextureAspect, TextureAspects, TextureSubresourceLayers,
    TextureSubresourceRange,
};
pub use texture::{
    Extent3d, Texture, TextureDescriptor, TextureDimension, TextureUsage, TextureViewCompatibility,
};
pub use transfer::{
    BufferUploadDescriptor, ReadbackData, ReadbackRequest, ReadbackStatus, ReadbackTexelLayout,
    ReadbackTicket, TextureUploadDescriptor, UploadDescriptor, UploadJob,
};
pub use view::{TextureView, TextureViewDescriptor, TextureViewDimension};
