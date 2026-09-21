//! Compute dispatch dimensions and explicit memory visibility barriers.

use super::{GlError, GlFamilyApi, ProgramId};

/// Immutable per-context compute limits required to validate dispatch groups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlComputeLimits {
    pub max_group_count: [u32; 3],
    pub max_group_size: [u32; 3],
    pub max_group_invocations: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDispatchGroups(pub [u32; 3]);
impl GlDispatchGroups {
    pub(crate) fn validate(self, limits: GlComputeLimits) -> Result<(), GlError> {
        if self
            .0
            .iter()
            .zip(limits.max_group_count)
            .any(|(&got, max)| got == 0 || got > max)
        {
            return Err(GlError::Validation {
                operation: "dispatch",
                message: "workgroup count is zero or exceeds a discovered axis limit".into(),
            });
        }
        Ok(())
    }
}

/// Compute-only domain; WebGL2 providers do not implement it.
pub(crate) trait GlComputeDispatchApi: GlFamilyApi {
    /// Installs the linked compute program dispatch work executes.
    ///
    /// This is the fixed compute pipeline's selection word: a provider only
    /// accepts a program whose discovery snapshot proved the compute
    /// capability, and dispatch rejects while no program is installed.
    fn set_compute_program(&mut self, program: ProgramId) -> Result<(), GlError>;
    fn dispatch(&mut self, groups: GlDispatchGroups) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{GlComputeLimits, GlDispatchGroups};
    use crate::backend::gl::api::GlMemoryBarrier;
    #[test]
    fn rejects_bad_preflight_values() {
        let l = GlComputeLimits {
            max_group_count: [1; 3],
            max_group_size: [1; 3],
            max_group_invocations: 1,
        };
        assert!(GlDispatchGroups([0, 1, 1]).validate(l).is_err());
        assert!(GlMemoryBarrier(0).validate_nonempty().is_err());
    }
}
