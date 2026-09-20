//! Shader entry points on Direct3D 12.
//!
//! One file today — [`module`] — and it is the smallest chapter in this backend
//! for a reason the native API is responsible for: D3D12 has no shader-module
//! object, so there is nothing here to create, check or compile ahead of pipeline
//! creation. Read [`module`]'s doc before treating a successful `create_shader`
//! as evidence that a shader compiled.
//!
//! The re-export is the chapter's inside face, so callers say `shader::create_shader`
//! rather than naming the file.

mod module;

#[cfg(test)]
pub(crate) use module::Dx12ShaderModule;
pub(crate) use module::create_shader;
