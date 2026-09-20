//! Shader entry points on Direct3D 12.
//!
//! What this file owns is small, and the reason is a property of the native API
//! rather than a gap in this lowering:
//!
//! # Direct3D 12 has no shader-module object
//!
//! D3D12 has no `ID3D12Shader`, no `CreateShaderModule`, and no
//! `CheckFeatureSupport` question that could ask whether a piece of bytecode is
//! acceptable. `D3D12_SHADER_BYTECODE` is a pointer and a length. The compile unit
//! is the *pipeline state object*: `CreateComputePipelineState` and
//! `CreateGraphicsPipelineState` are where the driver's shader compiler runs and
//! where a malformed or unsupported program is refused.
//!
//! **So a successful `create_shader` does not prove the driver accepted the
//! bytecode**, and this module must not be read as if it did. What
//! [`Dx12ShaderModule`] holds is the artifact's validated bytes, kept alive for the
//! pipeline lowering that will hand them to `D3D12_SHADER_BYTECODE`; the acceptance
//! verdict that *has* already happened is
//! [`crate::api::capability::EnabledCapabilities::shader_acceptance`]'s, which is a
//! statement about this device's recorded facts and not about the driver's opinion
//! of the program.
//!
//! Writing this down is not pedantry: the natural reading of "the module was
//! created" is "the shader compiled", and a later change that treats this call as
//! proof of compilation would move a real compile error to a place where it can no
//! longer be reported as section 19.10 requires — through
//! [`crate::api::RhiError`] with a diagnostic.
//!
//! # What a refusal would look like here, and why there is none
//!
//! There is nothing left to refuse. Section 19.10's portable rules ran in
//! `Device::create_shader`, the acceptance verdict ran there too, and copying bytes
//! has no failure mode. An `Unsupported` returned from here would have to be
//! manufactured, and a manufactured refusal is worse than an absence: it would tell
//! a caller its artifact was unacceptable when nothing had looked at it — discipline
//! 3's silent substitute read backwards.

use std::any::Any;
use std::sync::Arc;

use crate::api::shader::ShaderArtifact;
use crate::api::shader::backend::ShaderModuleBackend;

/// The validated entry point behind a portable
/// [`ShaderModule`](crate::api::shader::ShaderModule).
///
/// Holds the artifact rather than a copy of its bytes, because the artifact is
/// already the owner of those bytes ([`ShaderCode`](crate::api::shader::ShaderCode)
/// keeps them behind an `Arc`) and because the pipeline lowering needs more from it
/// than the bytes: section 28.1 requires a pipeline to be re-describable from the
/// artifacts it was built from, so the stage, the entry point and the interface
/// have to travel with the code.
pub(crate) struct Dx12ShaderModule {
    /// The artifact this module stands for.
    ///
    /// Read by [`Self::dxil`] to reach the bytes, and it is why the portable
    /// handle's `native` field is load-bearing: dropping this drops the artifact,
    /// and for a backend that speaks a runtime-compiled form this is where the
    /// compiled program would live. The field is deliberately the artifact rather
    /// than a copy of its bytes — [`ShaderCode`](crate::api::shader::ShaderCode)
    /// already keeps them behind an `Arc`, so a second copy would duplicate a
    /// shader for no reason.
    artifact: Arc<ShaderArtifact>,
}

impl Dx12ShaderModule {
    /// The DXIL bytes the driver will be handed, or `None` for another form.
    ///
    /// `Option` rather than a byte slice because this is where the code form
    /// becomes the driver's problem: a `ShaderArtifact` may legitimately carry a
    /// form this backend does not consume, and the portable acceptance verdict is
    /// what refuses that case before this backend is reached. A lowering that
    /// silently treated a non-DXIL artifact as empty bytecode would hand the driver
    /// a zero-length program, which `CreateComputePipelineState` accepts as a
    /// *library-less* state object on some drivers and refuses on others — a
    /// difference invisible to the caller. Returning `None` is the honest answer and
    /// forces the pipeline lowering to say what it does about it.
    ///
    /// The only accessor here. A sibling returning the whole artifact was written
    /// and deleted rather than kept "for the pipeline lowering": that lowering needs
    /// the *bytes*, which is what `D3D12_SHADER_BYTECODE` takes, and everything else
    /// it might want — the stage, the entry-point name, the interface — is already
    /// readable from the portable handle. A second reader would be a second answer
    /// to the same question, which is the duplication this crate deletes rather than
    /// tolerates.
    pub(crate) fn dxil(&self) -> Option<&[u8]> {
        match &self.artifact.code {
            crate::api::shader::ShaderCode::Dxil(bytes) => Some(bytes),
            _ => None,
        }
    }
}

impl ShaderModuleBackend for Dx12ShaderModule {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Prepares one entry point for later use by a pipeline state.
///
/// Takes the artifact by reference and wraps a clone in an `Arc`: the portable
/// handle already holds one copy and the pipeline lowering will want another, and a
/// second deep copy of a shader's bytes for no reason is the kind of cost that shows
/// up only in a profile.
///
/// Infallible, and it stays that way rather than returning an `RhiResult` with a
/// `Ok` in every arm. The seam's method returns a `Result` because a backend that
/// speaks a runtime-compiled form genuinely fails here; this one dispatches at the
/// pipeline instead, so a `Result` would be a shape with one inhabitant and would
/// invite a later edit to put a refusal in it that nothing can justify.
pub(crate) fn create_shader(artifact: &ShaderArtifact) -> Dx12ShaderModule {
    Dx12ShaderModule {
        artifact: Arc::new(artifact.clone()),
    }
}
