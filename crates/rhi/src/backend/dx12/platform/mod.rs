//! The Direct3D 12 platform chapter: adapters, the logical device, and the
//! request that hands one over.
//!
//! Three responsibilities, three files, in the order the portable layer reaches
//! them:
//!
//! - [`provider`] — the DXGI factory, adapter selection and `D3D12CreateDevice`.
//! - [`device`] — the native device, its liveness cell, and the lowering verbs
//!   behind [`crate::base::platform::DeviceBackend`].
//! - [`request`] — the one-shot handover of that device to the portable layer.
//!
//! What this chapter does not own: the capability tables themselves
//! ([`facts`] derives them from a live device and its doc says from what), and
//! the classification of a native failure ([`crate::backend::dx12::ffi`]).
//!
//! # Reachability, and the shape the expectation takes
//!
//! Nothing outside this chapter's tests reaches anything here: section 59 keeps
//! the provider off the public surface, and section 5.1 puts the host integration
//! that would open one in another crate. One `#![cfg_attr(not(test),
//! expect(dead_code, ..))]` carries that fact for the whole chapter, and it opens
//! [`provider`] rather than this file.
//!
//! The placement is deliberate. An expectation on a module makes that module's
//! items live roots, so an expectation on *this* file would make every file
//! beneath it a root as well and would then sit unfulfilled — the chapter's items
//! would all be "reachable" from an attribute rather than from the crate, which
//! is the opposite of what the attribute is asserting. On [`provider`] it says
//! the true thing: the entry point into this chapter is the provider, nothing
//! outside the chapter names it, and everything the chapter contains is reachable
//! only from it and from these tests.
//!
//! It changes what the lint reports *elsewhere*, which is invisible from inside
//! any single file and was measured rather than reasoned about. A module-scope
//! expectation makes references out of that module count as live for the modules
//! they point at. So the portable items the provider calls — `AdapterId::new`,
//! `AdapterId::serial`, `AdapterInfo::new`, `AvailableCapabilities::from_facts`,
//! `ObjectId::new`, `DeviceRequirements::is_empty` — are *not* dead whenever that
//! module is compiled, and *are* dead whenever it is not. `dx12` is compiled as a
//! unit, so in practice they are either all live together or all dead together.
//!
//! Their expectations are therefore gated on the backend feature, not on
//! `not(test)`. The per-item `not(test)` form was written first and does not
//! hold: with `dx12` on, the provider's reference makes each callee live, so a
//! `not(test)` expectation on the callee sits unfulfilled and fails the gate.
//!
//! The alternative repair — conditioning those expectations on `feature = "dx12"`
//! — was rejected for putting a backend's name into the portable layer six times
//! over, and for needing six edits per backend from here on. One expectation at
//! this chapter's root keeps the direction of the dependency intact and stays
//! true: this chapter is unreached, so its callers in the portable layer are
//! unreached too, in every feature combination.
//!
//! The same propagation is why [`crate::backend::dx12::ffi`] carries no
//! expectation on the two helpers the provider calls. Its one remaining
//! expectation is on an item with no caller in any configuration, which is a
//! different case and is fulfilled everywhere.

mod device;
mod facts;
mod provider;
mod request;

#[cfg(test)]
mod tests;
