//! Indirect command-buffer record layouts and their validated ranges.
//!
//! The records are the WebGPU-aligned indirect argument layouts this contract
//! accepts, so a caller that uploads WebGPU-shaped arguments needs no repacking.
//! The single, multi, and count paths stay distinct types because they have
//! distinct record shapes, distinct buffer roles, and distinct capability
//! evidence; they are never interchangeable. This module owns the layouts and
//! the range arithmetic only: issuing a command, binding its buffer, and
//! checking the context's capability belong to the provider.

use super::{GlBufferRange, GlError, GlFamilyApi};

/// ABI of the non-indexed draw record: four u32 words.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDrawArraysIndirectAbi {
    pub count: u32,
    pub instance_count: u32,
    pub first: u32,
    pub base_instance: u32,
}
/// ABI of the indexed draw record: four u32 words plus a signed base vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDrawElementsIndirectAbi {
    pub count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
}
/// ABI of the dispatch record: three u32 work-group counts.
///
/// This is exactly the WebGPU dispatch-indirect argument layout. A wider
/// platform record that carries an extra word is deliberately unreachable: one
/// command has one record shape here, so a stale wider record cannot silently
/// have its fourth word reinterpreted as a work-group count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDispatchIndirectAbi {
    pub workgroup_count: [u32; 3],
}

impl GlDispatchIndirectAbi {
    /// Exact record size in bytes, taken from the layout instead of restated.
    pub(crate) const SIZE: u64 = std::mem::size_of::<Self>() as u64;
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
/// One dispatch record selected inside a command buffer range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDispatchIndirectCommand {
    pub range: GlBufferRange,
    pub command_offset: u64,
}
impl GlDispatchIndirectCommand {
    /// Rejects a record that is misaligned or not wholly inside its range.
    ///
    /// The one-record path has no stride and no count: a dispatch record is
    /// always read whole, so the only failure modes are alignment and extent.
    pub(crate) fn validate(self, operation: &'static str) -> Result<(), GlError> {
        validate_indirect_range(self.range, operation)?;
        if self.command_offset % 4 != 0
            || self
                .command_offset
                .checked_add(GlDispatchIndirectAbi::SIZE)
                .is_none_or(|end| end > self.range.size)
        {
            return Err(GlError::Validation {
                operation,
                message: "indirect dispatch record leaves its buffer range".into(),
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
    /// Issues one work-group triple read from an indirect command buffer.
    fn dispatch_indirect(&mut self, command: GlDispatchIndirectCommand) -> Result<(), GlError>;
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
    use super::{
        GlDispatchIndirectAbi, GlDispatchIndirectCommand, GlIndirectAbi, GlIndirectCommandRange,
    };
    use crate::webgl2::api::{
        BufferId, ContextEpoch, ContextStamp, DeviceIdentity, GlBufferRange, GlError,
    };

    fn stamp() -> ContextStamp {
        ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
    }
    fn range(offset: u64, size: u64) -> GlBufferRange {
        GlBufferRange {
            buffer: BufferId::new(stamp(), 1, 1),
            offset,
            size,
        }
    }

    #[test]
    fn rejects_misaligned_indirect_offset() {
        let s = stamp();
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

    /// The dispatch record is the WebGPU dispatch argument layout: three u32
    /// work-group counts, twelve bytes, no padding word.
    #[test]
    fn dispatch_record_is_three_work_group_counts() {
        assert_eq!(GlDispatchIndirectAbi::SIZE, 12);
        assert_eq!(std::mem::align_of::<GlDispatchIndirectAbi>(), 4);
        let record = GlDispatchIndirectAbi {
            workgroup_count: [2, 3, 4],
        };
        assert_eq!(record.workgroup_count, [2, 3, 4]);
    }

    #[test]
    fn accepts_a_whole_dispatch_record_inside_its_range() {
        let whole = GlDispatchIndirectCommand {
            range: range(0, 12),
            command_offset: 0,
        };
        assert_eq!(whole.validate("dispatch-indirect"), Ok(()));
        let trailing = GlDispatchIndirectCommand {
            range: range(16, 24),
            command_offset: 8,
        };
        assert_eq!(trailing.validate("dispatch-indirect"), Ok(()));
    }

    #[test]
    fn rejects_dispatch_records_that_do_not_fit_their_range() {
        let missing_word = GlDispatchIndirectCommand {
            range: range(0, 8),
            command_offset: 0,
        };
        assert!(matches!(
            missing_word.validate("dispatch-indirect"),
            Err(GlError::Validation { .. })
        ));
        let past_the_end = GlDispatchIndirectCommand {
            range: range(0, 24),
            command_offset: 16,
        };
        assert!(past_the_end.validate("dispatch-indirect").is_err());
        let misaligned = GlDispatchIndirectCommand {
            range: range(0, 24),
            command_offset: 6,
        };
        assert!(misaligned.validate("dispatch-indirect").is_err());
    }

    #[test]
    fn rejects_dispatch_commands_outside_the_buffer_role() {
        let misaligned_range = GlDispatchIndirectCommand {
            range: range(2, 12),
            command_offset: 0,
        };
        assert!(misaligned_range.validate("dispatch-indirect").is_err());
        let empty = GlDispatchIndirectCommand {
            range: range(0, 0),
            command_offset: 0,
        };
        assert!(empty.validate("dispatch-indirect").is_err());
    }

    /// A record reaching past `u32`/`i32` arithmetic must fail closed instead of
    /// wrapping into a plausible in-range offset.
    #[test]
    fn overflowing_dispatch_offsets_are_rejected_not_wrapped() {
        let overflowing = GlDispatchIndirectCommand {
            range: range(0, u64::MAX - 3),
            command_offset: u64::MAX - 3,
        };
        assert!(overflowing.validate("dispatch-indirect").is_err());
    }
}
