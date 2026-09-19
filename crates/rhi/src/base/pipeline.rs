//! The pipeline chapter's seam: the native pipeline state object behind one
//! [`ComputePipeline`](crate::api::pipeline::ComputePipeline).
//!
//! A separate module from [`crate::base::shader`] for the reason that one is
//! separate from [`crate::base::resource`]: a pipeline is reached from its own
//! handle and grows for its own reason. The shader seam carries what a producer's
//! compiler produced; this one carries what the *driver* built out of it, which is
//! the first object in this crate whose creation the driver can refuse for a reason
//! about the program rather than about the descriptor.
//!
//! # One trait, and why it is named for compute
//!
//! The raster pipeline is a different native object with a different descriptor —
//! fixed-function state, a render-target signature, a vertex layout — and section
//! 27 and section 28 are two chapters. A single `PipelineBackend` would have to
//! either name both descriptors or name neither, so it is not written; the raster
//! seam lands with the raster lowering, and the name of this one says which
//! chapter's object it carries.
//!
//! # What this seam deliberately does not do
//!
//! It carries no `create`-shaped method, for the reason
//! [`crate::base::shader`] gives. The creation call lives on
//! [`DeviceBackend`](crate::base::platform::DeviceBackend), next to
//! `create_buffer`, because every portable check section 28 lists sits *before*
//! it — the capability gate, the two device-identity comparisons, the interface
//! and merged-requirement checks, the workgroup shape and its five limits. A
//! method here that took a descriptor would be a second place a pipeline could be
//! created, and the second place is where those checks get skipped.
//!
//! It carries no "is this pipeline usable" verb. A pipeline that a device built is
//! one the device can dispatch; whether the *dispatch* is legal is section 33's
//! question and is answered from the interface, which the portable layer still
//! holds.

use std::any::Any;

/// The native pipeline state object behind one
/// [`ComputePipeline`](crate::api::pipeline::ComputePipeline).
///
/// Implemented by a backend, held by the portable handle, and never reachable from
/// outside the crate. Like [`crate::base::resource::BufferBackend`] it carries the
/// object and not the operations: a dispatch is lowered by *the device's* backend,
/// which downcasts the pipeline and every bound group in one place, so a method
/// here would put one pipeline's binding operation behind an arbitrary receiver.
pub(crate) trait ComputePipelineBackend: Send + Sync + 'static {
    /// This pipeline as an opaque native object.
    ///
    /// The downcast's callers are the backend's own command lowering, which reaches
    /// the native pipeline state from the bound pipeline to hand it to the native
    /// bind verb, and the backend's own test set.
    fn as_any(&self) -> &dyn Any;
}
