//! W2's indirect-dispatch family, pure half: the command buffer one
//! `vkCmdDispatchIndirect` reads its workgroup counts from.
//!
//! The command carries no counts of its own. `Vulkan` reads a
//! `VkDispatchIndirectCommand` -- three `u32` counts, [`COMMAND_SIZE`] bytes -- out
//! of `buffer` at `offset`, so the only rules this family owns before the driver
//! are the ones the specification attaches to that read:
//!
//! - the offset must be aligned to [`COMMAND_ALIGNMENT`];
//! - the whole command must fit inside the buffer the caller declared, with the end
//!   computed in checked arithmetic so a `u64::MAX` offset is an overrun rather than
//!   a wrapped range that happens to pass.
//!
//! # The counts are deliberately not inspected
//!
//! [`super::compute::dispatch_groups`] refuses a zero workgroup dimension for the
//! *direct* form, because there the counts are the caller's own arguments and a
//! zero one would make a mistake look like a dispatch that legitimately did nothing.
//! Here the counts are data the graph wrote through some earlier command, so this
//! layer cannot see them at record time and must not pretend to: a zero axis inside
//! the buffer is the driver's own legal no-op, and the graph's compiler is what
//! established the buffer's contents. The one fact this family *can* check is the
//! one it can see -- the buffer's declared usage, which the owning half reads from
//! the device's table.
//!
//! Kept pure, like [`super::compute`] and [`super::draw`], so the refusals are
//! provable without a device: the encoder that records the command cannot be built
//! without a real command pool.

/// The size of one `VkDispatchIndirectCommand`: three `u32` workgroup counts.
///
/// A test in this module's own tests pins it against the driver's own struct, because
/// the value is a contract between the specification and this constant and a change
/// to either side would otherwise move the other silently.
pub(crate) const COMMAND_SIZE: u64 = 12;

/// The alignment `Vulkan` requires of an indirect command's offset.
///
/// The specification states a multiple of four, which is also the alignment of the
/// command's first field; a test pins the two together.
pub(crate) const COMMAND_ALIGNMENT: u64 = 4;

/// Why an indirect dispatch's command-buffer read is not one `Vulkan` defines.
///
/// Every variant is a value returned before the driver is reached, and the three
/// stay distinct because they are different mistakes to fix: an offset that is off
/// by a few bytes, a command that leaves the buffer, and an offset so large that its
/// end does not exist.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IndirectRangeError {
    /// The offset is not a multiple of [`COMMAND_ALIGNMENT`].
    Misaligned,
    /// The command's end overflows the `u64` byte space.
    ///
    /// Separate from [`Self::OutOfBounds`] because the two read the same on the
    /// cause and differ on the fix: no buffer is large enough for this offset, while
    /// an out-of-bounds read is fixed by a larger buffer or a smaller offset.
    Overflow,
    /// The command leaves the buffer the caller declared.
    OutOfBounds,
}

/// Lowers the offset one indirect dispatch reads at, or refuses the read.
///
/// The returned offset is the value the command takes, so a caller that has called
/// this has already had both rules checked; there is no separate "validated" wrapper
/// a caller could forget to go through.
pub(crate) fn dispatch_range(offset: u64, buffer_size: u64) -> Result<u64, IndirectRangeError> {
    if offset % COMMAND_ALIGNMENT != 0 {
        return Err(IndirectRangeError::Misaligned);
    }
    let end = offset
        .checked_add(COMMAND_SIZE)
        .ok_or(IndirectRangeError::Overflow)?;
    if end > buffer_size {
        return Err(IndirectRangeError::OutOfBounds);
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_that_fits_is_recorded_at_its_own_offset() {
        // The offset is returned unchanged: this backend has no packing or
        // translation to apply, and inventing one would be a second truth about a
        // command whose whole parameter set is that number.
        assert_eq!(dispatch_range(0, COMMAND_SIZE), Ok(0));
        assert_eq!(dispatch_range(12, 24), Ok(12));
        assert_eq!(
            dispatch_range(4096, u64::MAX),
            Ok(4096),
            "the driver's own maximum is the only ceiling, and it is not this rule's"
        );
    }

    #[test]
    fn the_command_size_is_the_drivers_own_struct() {
        // The constant is the specification's layout of three `u32` counts, and the
        // driver's struct is the other spelling of the same fact. Pinning them
        // together is what makes a change to either one visible here rather than at
        // a validation error inside the driver.
        assert_eq!(
            COMMAND_SIZE,
            core::mem::size_of::<ash::vk::DispatchIndirectCommand>() as u64
        );
        assert_eq!(
            COMMAND_ALIGNMENT,
            core::mem::align_of::<ash::vk::DispatchIndirectCommand>() as u64
        );
    }

    #[test]
    fn an_unaligned_offset_is_refused() {
        // The command's first field is a `u32`, so an offset that is not a multiple
        // of four would place the read inside a count rather than at its start.
        assert_eq!(
            dispatch_range(1, 256),
            Err(IndirectRangeError::Misaligned)
        );
        assert_eq!(
            dispatch_range(2, 256),
            Err(IndirectRangeError::Misaligned)
        );
        assert_eq!(
            dispatch_range(3, 256),
            Err(IndirectRangeError::Misaligned)
        );
        assert_eq!(
            dispatch_range(u64::MAX - 4, u64::MAX),
            Err(IndirectRangeError::Misaligned),
            "the offset is checked before the end, so an unaligned huge one names the low bits"
        );
    }

    #[test]
    fn a_command_past_the_buffer_is_refused() {
        // A buffer of exactly the command's size holds exactly one command at zero.
        assert_eq!(
            dispatch_range(COMMAND_SIZE, COMMAND_SIZE),
            Err(IndirectRangeError::OutOfBounds),
            "the command starts one byte past the end"
        );
        assert_eq!(
            dispatch_range(0, COMMAND_SIZE - 1),
            Err(IndirectRangeError::OutOfBounds),
            "a buffer one byte short does not hold the command"
        );
        assert_eq!(dispatch_range(0, 0), Err(IndirectRangeError::OutOfBounds));
    }

    #[test]
    fn an_offset_whose_end_overflows_is_refused() {
        // `u64::MAX - 3` is a multiple of four, so the overflow is the only rule it
        // can fail; the two shapes are asserted separately because a check that ran
        // the alignment test first would otherwise report the wrong sentence.
        assert_eq!(
            dispatch_range(u64::MAX - 3, u64::MAX),
            Err(IndirectRangeError::Overflow)
        );
        assert_eq!(
            dispatch_range(COMMAND_ALIGNMENT, u64::MAX),
            Ok(COMMAND_ALIGNMENT),
            "a normal offset against the largest buffer still records"
        );
    }
}
