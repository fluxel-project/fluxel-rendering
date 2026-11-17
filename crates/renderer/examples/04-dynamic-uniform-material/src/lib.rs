//! Immutable CPU inputs and independent math for the Renderer adaptation of RHI 04.

pub mod inputs;
pub mod oracle;

#[cfg(test)]
#[path = "tests/oracle.rs"]
mod oracle_tests;
pub mod scene;
