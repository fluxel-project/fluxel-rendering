//! The group state domain: applying a batch of binding updates as one
//! transition, and what happens when one of them fails partway.
//!
//! Responsibility: expand a group into the per-domain updates it contains, in a
//! fixed order, and mark every domain it touched unknown when the application
//! cannot complete.
//!
//! Not owned here: the individual bindings (the domains above) and the decision
//! of which groups exist (the Renderer).
//!
//! Nothing is implemented here yet, and the reportable half of a partial
//! application is not this domain's.  It belongs to the domain that failed
//! rather than to the group: a domain stops at the first refused call, leaves
//! its own applied state unknown, reports how far it got in
//! [`super::error::StateError::Backend`]'s `PartialApplication`, and counts the
//! refusal in `lifecycle.driver_errors`.  The group half -- the fixed expansion
//! order above, and the short-circuit that would skip a whole expansion when the
//! group applied is the one already applied -- is a decision above these domains
//! rather than a mirror in this one; [`super::knowledge::StateDomain::Groups`]
//! records that, and why.
//! `LifecycleCounters::poisoned_groups` is therefore never incremented, so a
//! report that reads it as zero cannot distinguish "no group ever failed" from
//! "a group failed partway" and must not be read that way.
