//! The geometry state domain: which vertex-array object is bound, and the
//! vertex-array objects derived from a complete vertex layout.
//!
//! Responsibility: make the backend's bound vertex array agree with what the
//! caller asked for, and reuse one derived vertex-array object for one layout.
//!
//! Not owned here: the buffer contents and the buffer bindings a vertex array
//! records (buffers), and the pipeline's attribute declarations (pipeline).
//! The derivation is keyed by the layout value the caller supplies, and the
//! buffers it names are the entry's dependencies.
