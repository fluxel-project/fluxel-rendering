//! The compute state domain: which compute program is installed, the storage
//! bindings it reads and writes, and the dispatch that consumes them.
//!
//! Responsibility: make the backend's compute program, storage-buffer bindings
//! and storage-image bindings agree with what the caller asked for.
//!
//! Not owned here: the graphics pipeline (pipeline) and the program objects
//! themselves; this domain shares the program cache with the pipeline domain
//! rather than linking a second program for the same stages.  It is reachable
//! only through [`super::GlOptionalComputeBackend`], because a profile without
//! the compute trait must have no empty compute state group.
