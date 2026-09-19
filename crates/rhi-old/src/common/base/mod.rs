//! The base: the part of the RHI that does not depend on what hardware can do.
//!
//! ```text
//! identity / resources / lifetime / submission / capability query
//! ```
//!
//! The base is deliberately tiny, and the reason is the design's principle:
//! **unify semantics and ownership, not hardware capability.** Everything here is
//! true of every backend *because of what Fluxel must guarantee*, not because some
//! hardware happens to provide it.
//!
//! # What is not in the base, and why
//!
//! | Not in the base | Where it belongs | Why |
//! | --- | --- | --- |
//! | `draw`, `dispatch` | the graphics and compute families | a base method would force every backend to answer for a capability it may not have |
//! | `storage_texture` | the storage-texture family | same, and the capability varies per adapter, not only per backend |
//! | `graphics_queue`, `compute_queue`, `transfer_queue` | the queue-shape rows | DX12 and Vulkan expose several queues, Metal's model differs, WebGPU has its own constraints and WebGL2 is not the same thing at all |
//!
//! The membership test is therefore: **a base item is something all five backends
//! must agree on for Fluxel's own semantics to hold.** Identity, lifetime,
//! submission and capability *query* pass that test. A queue layout is a fact
//! about the hardware, so it fails it, however convenient it would be.
//!
//! # What the base owns today
//!
//! [`stamp::DeviceStamp`] is the identity primitive: the device a backend object
//! belongs to, plus the generation of that device. Two existing implementations
//! already need it -- the native path stamps owned resources with
//! `fluxel_rendergraph::PhysicalResourceIdentity`, and the GL family stamps
//! buffer and texture ids with a `ContextStamp` of device identity plus context
//! epoch -- so promoting one stamp is extraction, not invention.
//!
//! Still owed by the base, in this order: resource identity and its staleness
//! rule, lifetime and lease-before-completion, submission and quarantine, and the
//! capability query that negotiates families.

pub(crate) mod lifetime;
pub(crate) mod resource;
pub(crate) mod stamp;
pub(crate) mod submission;
