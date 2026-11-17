//! Typed material composition and generic WGSL contributions.
//!
//! Material owns authoring validation, static variants, instance values, and a
//! private normalized representation. It emits only generic WGSL source and
//! interface declarations; a Shader consumer supplies pass templates and entry
//! points before composing a complete program.

#![deny(missing_docs)]

mod authoring;
mod compiler;
mod diagnostic;
mod runtime;

pub use authoring::*;
pub use compiler::{
    MaterialBindingDeclaration, MaterialCompiler, MaterialParameterBinding,
    MaterialShaderContribution,
};
pub use diagnostic::MaterialError;
pub use runtime::*;

#[cfg(test)]
mod tests;
