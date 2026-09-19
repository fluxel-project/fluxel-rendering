//! The sync state domain: memory barriers, client waits, and the fences a
//! frame boundary is built from.
//!
//! Responsibility: emit a barrier at most once per category set that actually
//! needs one, and make a wait observable exactly when the caller asked for it.
//!
//! Not owned here: the commands a barrier orders (the domains that emit them)
//! and the residency policy that decides when a frame may retire (the
//! Renderer).  This domain has no derived cache, because a barrier is not an
//! object and a redundant barrier is a redundancy the mirror can prove without
//! one.
