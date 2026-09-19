//! The transition barriers one step of a recording needs, released together.
//!
//! # Why this is a type and not four lines at each call site
//!
//! `D3D12_RESOURCE_BARRIER`'s `pResource` is a `ManuallyDrop`, so a barrier built
//! the obvious way takes one reference to the resource and dropping the barrier
//! does **not** release it. `ResourceBarrier` copies the struct into the command
//! stream rather than taking ownership of it, so a caller that lets the barrier
//! fall out of scope leaks exactly one reference per barrier — a leak that grows
//! with every frame and never shows up as an error.
//!
//! Reclaiming it is a `ManuallyDrop::drop` on a union member, which is `unsafe`
//! and easy to get wrong in the direction of a double release. Putting it in a
//! `Drop` impl means it is written once, with one rationale, and cannot be
//! forgotten at a call site. Note that `D3D12_RESOURCE_BARRIER`'s own `Clone` is
//! a `transmute_copy` that does *not* add a reference, so a clone of a barrier is
//! a second owner of one reference — this type never clones one.
//!
//! # What this module does not own
//!
//! Which transition a step needs, and in which order relative to the command it
//! brackets. That is the invariant
//! [`crate::backend::dx12::command`] documents — every list leaves every buffer
//! it touched in `D3D12_RESOURCE_STATE_COMMON` — and it belongs to the lowering
//! that knows what the step is, not to the barrier holder.

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, ID3D12GraphicsCommandList,
    ID3D12Resource,
};

/// The transition barriers one step of a recording needs, released together.
#[derive(Default)]
pub(super) struct Transitions {
    barriers: Vec<D3D12_RESOURCE_BARRIER>,
}

impl Transitions {
    /// Adds one transition from `before` to `after` on `resource`.
    pub(super) fn push(
        &mut self,
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        self.barriers.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // The one reference this barrier owns, released in `Drop`.
                    pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                    // Every resource this backend copies through is a buffer, and
                    // a buffer has one subresource. The constant is Direct3D 12's
                    // own "all of them", which is the honest value for a resource
                    // that has exactly one.
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        });
    }

    /// Appends these transitions to `list` in the order they were pushed.
    ///
    /// Takes `&self` rather than consuming, which is what lets the caller keep
    /// the value alive until it drops and releases the references above.
    pub(super) fn record(&self, list: &ID3D12GraphicsCommandList) {
        if self.barriers.is_empty() {
            return;
        }
        // SAFETY: `ResourceBarrier` reads the slice it is given and copies each
        // barrier into the command stream; the vector outlives the call, and the
        // resources it names are kept alive by the caller for as long as the
        // recorded list can execute.
        unsafe { list.ResourceBarrier(&self.barriers) };
    }
}

impl Drop for Transitions {
    fn drop(&mut self) {
        for barrier in &mut self.barriers {
            // SAFETY: every barrier in this vector was built by `push` above,
            // which is the only constructor, so `Transition` is the variant that
            // is live and reading it is not reading an inactive union member.
            // `pResource` is a `ManuallyDrop` because the generated struct has no
            // `Drop` of its own; releasing it here is the one release of the one
            // reference `push` took, and nothing else touches this vector.
            unsafe {
                let transition = &mut barrier.Anonymous.Transition;
                std::mem::ManuallyDrop::drop(&mut transition.pResource);
            }
        }
    }
}
