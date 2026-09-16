//! The buffer state domain: which buffer is bound in each binding role, and
//! what range of it is bound.
//!
//! Responsibility: make the backend's buffer binding points agree with what the
//! caller asked for, per role, and leave no scratch binding behind.
//!
//! Not owned here: the buffer objects and their storage (Layer 1's resource
//! tables), and the vertex-array objects that record their own bindings
//! (geometry).  This domain has no derived cache: a binding is not an object.
