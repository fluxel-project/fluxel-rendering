//! Indirect ABI layouts and ranges; single, multi, and count paths are distinct.

use super::{GlBufferRange, GlError, GlFamilyApi};

/// ABI of one `DrawArraysIndirect` record (four u32 words).
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDrawArraysIndirectAbi {
    pub count: u32,
    pub instance_count: u32,
    pub first: u32,
    pub base_instance: u32,
}
/// ABI of one `DrawElementsIndirect` record (four u32 words plus signed base vertex).
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDrawElementsIndirectAbi {
    pub count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
}

fn validate_indirect_range(range: GlBufferRange, operation: &'static str) -> Result<(), GlError> {
    if range.size == 0 || range.offset % 4 != 0 || range.size % 4 != 0 {
        Err(GlError::Validation {
            operation,
            message: "indirect buffer ranges must be nonempty and 4-byte aligned".into(),
        })
    } else {
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlIndirectAbi {
    NonIndexed,
    Indexed,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlIndirectCommandRange {
    pub range: GlBufferRange,
    pub command_offset: u64,
    pub draw_count: u32,
    pub stride: u32,
    pub abi: GlIndirectAbi,
}
impl GlIndirectCommandRange {
    pub(crate) fn validate(self, operation: &'static str) -> Result<(), GlError> {
        validate_indirect_range(self.range, operation)?;
        let record = match self.abi {
            GlIndirectAbi::NonIndexed => 16,
            GlIndirectAbi::Indexed => 20,
        };
        if self.draw_count == 0
            || self.command_offset % 4 != 0
            || self.stride != 0 && (self.stride < record || self.stride % 4 != 0)
        {
            return Err(GlError::Validation {
                operation,
                message: "indirect offset, count, or stride violates the selected ABI".into(),
            });
        }
        let stride = if self.stride == 0 {
            record
        } else {
            self.stride
        } as u64;
        let needed = self
            .command_offset
            .checked_add((u64::from(self.draw_count) - 1).saturating_mul(stride))
            .and_then(|v| v.checked_add(u64::from(record)));
        if needed.is_none_or(|v| v > self.range.size) {
            return Err(GlError::Validation {
                operation,
                message: "indirect commands exceed their buffer range".into(),
            });
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlIndirectCountRange {
    pub range: GlBufferRange,
    pub count_offset: u64,
    pub max_draw_count: u32,
}
impl GlIndirectCountRange {
    pub(crate) fn validate(self) -> Result<(), GlError> {
        validate_indirect_range(self.range, "multi_draw_indirect_count")?;
        if self.max_draw_count == 0
            || self.count_offset % 4 != 0
            || self
                .count_offset
                .checked_add(4)
                .is_none_or(|v| v > self.range.size)
        {
            return Err(GlError::Validation {
                operation: "multi_draw_indirect_count",
                message: "count range is not a 4-byte in-range value".into(),
            });
        }
        Ok(())
    }
}

pub(crate) trait GlDrawIndirectApi: GlFamilyApi {
    fn draw_indirect(&mut self, command: GlIndirectCommandRange) -> Result<(), GlError>;
}
pub(crate) trait GlDispatchIndirectApi: GlFamilyApi {
    fn dispatch_indirect(&mut self, command: GlBufferRange, offset: u64) -> Result<(), GlError>;
}
pub(crate) trait GlMultiDrawIndirectApi: GlFamilyApi {
    fn multi_draw_indirect(&mut self, commands: GlIndirectCommandRange) -> Result<(), GlError>;
}
pub(crate) trait GlMultiDrawCountApi: GlFamilyApi {
    fn multi_draw_indirect_count(
        &mut self,
        commands: GlIndirectCommandRange,
        count: GlIndirectCountRange,
    ) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{GlIndirectAbi, GlIndirectCommandRange};
    use crate::webgl2::api::{BufferId, ContextEpoch, ContextStamp, DeviceIdentity, GlBufferRange};
    #[test]
    fn rejects_misaligned_indirect_offset() {
        let s = ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL);
        let c = GlIndirectCommandRange {
            range: GlBufferRange {
                buffer: BufferId::new(s, 1, 1),
                offset: 0,
                size: 16,
            },
            command_offset: 2,
            draw_count: 1,
            stride: 0,
            abi: GlIndirectAbi::NonIndexed,
        };
        assert!(c.validate("draw_indirect").is_err());
    }
}
