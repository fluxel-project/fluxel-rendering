//! The RHI's three layers, and the rule that separates them.
//!
//! ```text
//!                     RHI Base
//!      identity / resources / lifetime / submission
//!                          |
//!                Capability Families
//!    Graphics / Compute / StorageBuffer / StorageTexture /
//!    Indirect / AsyncCompute / TransferQueue / ...
//!                          |
//!              Backend Implementations
//!          DX12 / Vulkan / Metal / WebGPU / GL family
//! ```
//!
//! # The design principle
//!
//! **Capability-oriented RHI: unify semantics and ownership, not hardware
//! capability.**
//!
//! The base is small on purpose. It unifies what every backend must agree on
//! regardless of what the hardware can do -- resource identity, lifetime,
//! dependency and submission -- and it does not model a queue layout, a command
//! vocabulary or a feature set that only some hardware has. A queue is therefore
//! not `graphics_queue()` / `compute_queue()` / `transfer_queue()` on the base,
//! because that shape forces every backend to pretend it has all three: it is a
//! capability family instead.
//!
//! # What a capability family is, and what it is not
//!
//! A family is a **vocabulary**, not a permission. `ComputeApi` says how compute
//! is asked for; it does not say that this device can compute. The distinction is
//! not pedantry, because capability varies below the backend:
//!
//! ```text
//! Vulkan backend
//!   |- GPU A: storage texture, every format
//!   |- GPU B: storage texture, some formats
//!   \- GPU C: no storage texture
//! ```
//!
//! Equating "the trait is implemented" with "the hardware supports it" would
//! therefore be wrong in the ordinary case, not just in a corner. What actually
//! happens is negotiation: a caller asks the device for a family, the device
//! answers from its own discovery, and a proven answer yields a handle whose use
//! needs no further checks.
//!
//! ```text
//! pass requirement        -- declarations, in capability terms
//!        |
//! device capabilities     -- discovery, per adapter
//!        |
//! satisfied -> compile execution plan
//! unmet     -> UnsupportedCapability
//! ```
//!
//! The requirement is stated in capability terms and never in backend terms: a
//! node requires `Graphics + StorageBuffer + Indirect`, not `Vulkan`. That is
//! what lets one compiled graph run on whatever device can serve it, and what
//! lets a RenderGraph emit a cross-queue schedule where an asynchronous-compute
//! family exists and a single-queue schedule where it does not -- instead of the
//! RHI compressing every backend to the weakest one's shape from the start.
//!
//! # The two refusals stay distinct
//!
//! 1. **The backend cannot serve the family at all.** The vocabulary is absent,
//!    so no call site could have been written.
//! 2. **The backend serves the family and this device did not prove it.** The
//!    vocabulary exists, the row is unproved, and [`family::Requirement`] reports
//!    which of the ledger's three conditions failed.
//!
//! Collapsing these into one "unsupported" answer is what forces a lowest common
//! denominator. The first is answered by a trait bound; the second by a value.

pub(crate) mod families;
pub(crate) mod family;
pub(crate) mod graphics;
pub(crate) mod handle;
pub(crate) mod negotiate;
