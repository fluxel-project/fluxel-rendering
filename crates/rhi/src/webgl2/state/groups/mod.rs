//! The group state domain: applying a batch of binding updates as one
//! transition, and what happens when one of them fails partway.
//!
//! Responsibility: expand a group into the per-domain updates it contains, in a
//! fixed order, and mark every domain it touched unknown when the application
//! cannot complete.
//!
//! Not owned here: the individual bindings (the domains above) and the decision
//! of which groups exist (the Renderer).  This domain is the only place a
//! partial application is turned into a poisoned group, and
//! [`super::counters::LifecycleCounters::poisoned_groups`] is its counter.
