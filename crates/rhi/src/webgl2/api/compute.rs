//! Compute dispatch dimensions and explicit memory visibility barriers.

use super::{GlError, GlFamilyApi};

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlMemoryBarrier(pub u32);
impl GlMemoryBarrier {
    pub const SHADER_STORAGE: Self = Self(1 << 0);
    pub const SHADER_IMAGE_ACCESS: Self = Self(1 << 1);
    pub const TEXTURE_FETCH: Self = Self(1 << 2);
    pub const VERTEX_ATTRIB_ARRAY: Self = Self(1 << 3);
    pub const COMMAND: Self = Self(1 << 4);
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub(crate) fn validate_nonempty(self) -> Result<(), GlError> {
        (self.0 != 0)
            .then_some(())
            .ok_or_else(|| GlError::Validation {
                operation: "memory_barrier",
                message: "at least one barrier class is required".into(),
            })
    }
}

/// Compute-only domain; WebGL2 providers do not implement it.
pub(crate) trait GlComputeDispatchApi: GlFamilyApi {
    fn dispatch(&mut self, groups: GlDispatchGroups) -> Result<(), GlError>;
    fn memory_barrier(&mut self, barriers: GlMemoryBarrier) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{GlComputeLimits, GlDispatchGroups, GlMemoryBarrier};
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
