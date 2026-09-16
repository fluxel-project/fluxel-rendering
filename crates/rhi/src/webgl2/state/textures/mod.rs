//! The texture state domain: the active texture unit, the texture bound to
//! each unit and target, and the sampler bound to each unit.
//!
//! Responsibility: make the backend's texture-unit and sampler bindings agree
//! with what the caller asked for.
//!
//! Not owned here: the texture images and sampler objects (Layer 1's resource
//! tables) and the framebuffers built from texture views (session).  Layer 1
//! facts this domain exists for: `bind_texture` silently moves the active unit,
//! and several paths leave a unit bound after they finish.
