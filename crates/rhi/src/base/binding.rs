//! The binding chapter's seam: the native descriptor packet behind one
//! [`BindGroup`](crate::api::binding::BindGroup).
//!
//! A separate module from [`crate::base::resource`] for the reason that one is
//! separate from [`crate::base::platform`]: a bind group is reached from its own
//! handle and grows for its own reason. The resource seam carries the allocations a
//! caller asked for; this one carries the *packet that points at them*, which is
//! assembled at bind-group creation and read when a command binds it.
//!
//! # Why there is one trait here and not two
//!
//! [`crate::api::binding::BindGroupLayout`] has no type in this module, and its
//! absence is a decision rather than an omission. Direct3D 12 has no
//! descriptor-set-layout object: the analogue is the *root signature*, and a root
//! signature is a property of a pipeline rather than of one layout — one PSO has
//! exactly one, and it is built from the whole ordered group sequence. So the
//! native work of `Device::create_bind_group_layout` is zero on this backend, and
//! the lowering that does need the layouts reads their canonical descriptors
//! through [`crate::api::pipeline::PipelineInterface`].
//!
//! Vulkan and WebGPU do have the object — `VkDescriptorSetLayout`,
//! `GPUBindGroupLayout` — and they will need a trait beside this one. It is not
//! declared now: `CLAUDE.md` section 8 lands a seam when a consumer needs it, and
//! a trait whose only implementation would be an empty `as_any` is a trait with no
//! consumer at all.
//!
//! # What this seam deliberately does not do
//!
//! It carries no `create`-shaped method, for the reason
//! [`crate::base::shader`] gives: the creation call lives on
//! [`DeviceBackend`](crate::base::platform::DeviceBackend), next to
//! `create_buffer`, because everything section 22.3 checks — the layout match, the
//! range rules, the device's binding limits — sits *before* it. A method here that
//! took a descriptor would be a second place a group could be created, and the
//! second place is where those checks get skipped.
//!
//! It carries no update verb either, and that one is section 22.2's: a P0
//! [`BindGroup`](crate::api::binding::BindGroup) is immutable after creation, so a
//! change is a new group rather than a write into an existing packet.

use std::any::Any;

/// The native descriptor packet behind one
/// [`BindGroup`](crate::api::binding::BindGroup).
///
/// Implemented by a backend, held by the portable handle, and never reachable from
/// outside the crate. Like [`crate::base::resource::BufferBackend`] it carries the
/// object and not the operations: a command is lowered by *the device's* backend,
/// which downcasts this and every other group in one place, so a method here would
/// put one group's binding operation behind an arbitrary receiver.
pub(crate) trait BindGroupBackend: Send + Sync + 'static {
    /// This packet as an opaque native object.
    ///
    /// The downcast's callers are the backend's own command lowering, which reaches
    /// a native descriptor from each bound group to hand it to the native bind
    /// verb, and the backend's own test set.
    fn as_any(&self) -> &dyn Any;
}
