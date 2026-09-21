//! Metal lowering for the frozen Fluxel v13 execution contract.
//!
//! All Objective-C objects remain below this module.  The implementation uses
//! one retained ownership domain per logical device, tracked Metal hazards and
//! a transactional two-phase submission path.  Capability facts are published
//! only for native operations whose complete lowering exists here.

mod binding;
mod command;
mod device;
mod facts;
mod format;
mod logic;
mod pipeline;
mod presentation;
mod provider;
mod query;
mod resolve;
mod resource;
mod shader;

pub(crate) use provider::MetalProvider;

#[cfg(test)]
mod tests;
