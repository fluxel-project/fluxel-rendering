//! W2's compute family, pure half: the workgroup counts one dispatch records.
//!
//! `vkCmdDispatch` takes three group counts and nothing else, so the only rule this
//! family owns before the driver is the one the common contract states: every group
//! dimension must be non-zero. A zero group count is a legal driver no-op rather than
//! an error, so recording it would make a caller's mistake look like a dispatch that
//! legitimately did nothing -- and the run-time check the contract promises moves to
//! the one place that can still refuse it.
//!
//! The counts are returned unchanged. `Vulkan` 1.0 takes three unsigned counts, so
//! there is no packing, range or alignment to lower; inventing a value here would be
//! a second truth about a command whose whole parameter set is those three numbers.
//!
//! Kept pure, like [`super::draw`], so the refusal is provable without a device: the
//! encoder that records the command cannot be built without a real command pool.

/// Why a dispatch's workgroup counts are not ones this backend records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DispatchError {
    /// At least one group dimension is zero.
    ///
    /// The whole triple is carried rather than the offending axis, because a
    /// dispatch is described by its three counts together and a caller fixing one
    /// dimension wants to see the request that arrived.
    ZeroGroups([u32; 3]),
}

/// Lowers the portable workgroup counts onto the three the command takes.
///
/// Every dimension must be non-zero: a zero group count is a legal driver no-op, so
/// one that arrives here is a caller's mistake rather than a request to do nothing.
/// The tuple is returned by value, which is the whole lowering.
pub(crate) fn dispatch_groups(groups: [u32; 3]) -> Result<[u32; 3], DispatchError> {
    if groups.contains(&0) {
        return Err(DispatchError::ZeroGroups(groups));
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_zero_triple_is_recorded_unchanged() {
        // One group per axis is the retained artifact's shape, and the counts are
        // carried rather than recomputed: `Vulkan` takes exactly these three numbers.
        assert_eq!(dispatch_groups([1, 1, 1]), Ok([1, 1, 1]));
        assert_eq!(dispatch_groups([64, 32, 8]), Ok([64, 32, 8]));
        assert_eq!(
            dispatch_groups([u32::MAX, 1, 1]),
            Ok([u32::MAX, 1, 1]),
            "the driver's own maximum is the only ceiling, and it is not this rule's"
        );
    }

    #[test]
    fn a_zero_on_any_axis_is_refused_and_the_triple_is_carried() {
        // Each axis alone, because a check on only the first would accept the other
        // two and produce exactly the silent no-op the contract refuses.
        assert_eq!(
            dispatch_groups([0, 1, 1]),
            Err(DispatchError::ZeroGroups([0, 1, 1]))
        );
        assert_eq!(
            dispatch_groups([1, 0, 1]),
            Err(DispatchError::ZeroGroups([1, 0, 1]))
        );
        assert_eq!(
            dispatch_groups([1, 1, 0]),
            Err(DispatchError::ZeroGroups([1, 1, 0]))
        );
        assert_eq!(
            dispatch_groups([0, 0, 0]),
            Err(DispatchError::ZeroGroups([0, 0, 0]))
        );
    }
}
