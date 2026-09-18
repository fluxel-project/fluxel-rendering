//! W2's storage-buffer family, pure half: the validated range one storage binding
//! names.
//!
//! # `Vulkan` has no storage-buffer object
//!
//! A storage buffer is a `VkDescriptorBufferInfo` written into a descriptor set, so
//! what this backend can build before a set exists is the *fact* the write needs: a
//! range that is non-empty and fits the buffer it names. The end is computed in
//! checked arithmetic, so a `u64::MAX` offset is a refusal rather than a wrapped
//! range that passes the bound.
//!
//! [`super::bind_group::buffer_info`] obeys exactly that rule where a binding number
//! and a driver handle exist -- the boundary repetition step 8 records for the copy
//! ranges -- and it calls [`check_range`] rather than keeping a second spelling of
//! the arithmetic. There is one implementation and the driver path inherits it.
//!
//! # Why the binding is a value rather than a handle
//!
//! `StorageBuffer` is a *resource role*: it gates how a binding is built and records
//! nothing, so the handle that builds one owns no encoder. [`StorageBufferBinding`]
//! therefore carries base resource identity and the two numbers, and
//! [`StorageBufferBinding::at`] is how it reaches a
//! [`BindGroupEntry`](crate::common::binding::BindGroupEntry): the binding number is
//! the layout's business, and a value that carried one before a layout named it would
//! be inventing a fact.
//!
//! # What is deliberately not here
//!
//! No alignment rule. `Vulkan` places no offset alignment on a storage buffer beyond
//! the minimum it reports for the descriptor type, and that minimum is a device limit
//! the bind-group step reads where it can actually enforce it; a second, invented
//! alignment here would refuse ranges the driver accepts.

use crate::common::base::resource::BufferId;
use crate::common::binding::{BindGroupEntry, BindingResource};

/// Why a storage-buffer range cannot be bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RangeError {
    /// The range is zero bytes, so it binds nothing.
    Zero,
    /// The range does not fit inside the buffer, or its end would wrap.
    OutOfBounds,
}

/// Checks the one rule every storage-buffer range obeys.
///
/// The size is checked before the end, so an empty range is its own sentence rather
/// than an arithmetic result, and the end is computed with `checked_add`, so
/// `offset = u64::MAX` is an overrun instead of a wrapped range that satisfies
/// `end <= buffer_size`.
pub(crate) fn check_range(offset: u64, size: u64, buffer_size: u64) -> Result<(), RangeError> {
    if size == 0 {
        return Err(RangeError::Zero);
    }
    let fits = offset
        .checked_add(size)
        .is_some_and(|end| end <= buffer_size);
    if !fits {
        return Err(RangeError::OutOfBounds);
    }
    Ok(())
}

/// A storage-buffer range this device's table accepted.
///
/// It is produced only by the storage-buffer family's verb, which is what makes "a
/// storage binding names a buffer this device owns, for a range that fits it" a fact
/// of the value rather than a check the caller has to remember.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageBufferBinding {
    /// The buffer the binding reads or writes.
    buffer: BufferId,
    /// The byte offset the bound range begins at.
    offset: u64,
    /// The size of the bound range in bytes.
    size: u64,
}

impl StorageBufferBinding {
    /// Wraps a range already checked against the buffer's own declared size.
    pub(crate) const fn new(buffer: BufferId, offset: u64, size: u64) -> Self {
        Self {
            buffer,
            offset,
            size,
        }
    }

    /// Returns the buffer this binding names.
    pub(crate) const fn buffer(&self) -> BufferId {
        self.buffer
    }

    /// Returns the byte offset the bound range begins at.
    pub(crate) const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the bound range's size in bytes.
    pub(crate) const fn size(&self) -> u64 {
        self.size
    }

    /// Places this binding at `binding` of a bind group.
    ///
    /// The number is supplied here rather than stored, because the layout is what
    /// declares which numbers exist and the family verb that built this value never
    /// saw one.
    pub(crate) const fn at(&self, binding: u32) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::Buffer {
                buffer: self.buffer,
                offset: self.offset,
                size: self.size,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::base::stamp::DeviceStamp;
    use fluxel_rendergraph::{DeviceIdentity, PhysicalResourceIdentity};

    fn buffer() -> BufferId {
        BufferId::new(
            DeviceStamp::initial(DeviceIdentity::new(1)),
            PhysicalResourceIdentity::new(7),
        )
    }

    #[test]
    fn a_range_that_fits_the_buffer_is_accepted() {
        assert_eq!(check_range(0, 256, 256), Ok(()));
        assert_eq!(check_range(256, 256, 512), Ok(()));
        // The end exactly at the buffer's own end is inside it.
        assert_eq!(check_range(128, 128, 256), Ok(()));
    }

    #[test]
    fn a_zero_sized_range_is_refused_by_name() {
        // `Vulkan` would accept the descriptor, so the refusal has to happen here:
        // a binding that names no bytes is a description mistake, not a valid range.
        assert_eq!(check_range(0, 0, 256), Err(RangeError::Zero));
        assert_eq!(check_range(256, 0, 256), Err(RangeError::Zero));
    }

    #[test]
    fn a_range_past_the_buffer_is_refused() {
        assert_eq!(check_range(128, 256, 256), Err(RangeError::OutOfBounds));
        assert_eq!(check_range(257, 1, 256), Err(RangeError::OutOfBounds));
    }

    #[test]
    fn a_wrapping_end_is_an_overrun_rather_than_a_range_that_passes() {
        // `u64::MAX - 3` is the largest size whose end still wraps, and `u64::MAX`
        // itself is the offset the borrowed path's unchecked addition would wrap.
        assert_eq!(
            check_range(u64::MAX - 3, 8, u64::MAX),
            Err(RangeError::OutOfBounds)
        );
        assert_eq!(
            check_range(u64::MAX, 1, u64::MAX),
            Err(RangeError::OutOfBounds)
        );
    }

    #[test]
    fn a_binding_carries_its_buffer_and_its_range() {
        let binding = StorageBufferBinding::new(buffer(), 64, 128);
        assert_eq!(binding.buffer(), buffer());
        assert_eq!(binding.offset(), 64);
        assert_eq!(binding.size(), 128);
    }

    #[test]
    fn a_binding_reaches_a_bind_group_entry_at_the_number_it_is_given() {
        // The layout declares the number, so the same value can fill any number
        // without the family having guessed one.
        let binding = StorageBufferBinding::new(buffer(), 16, 80);
        let entry = binding.at(3);
        assert_eq!(entry.binding, 3);
        assert_eq!(
            entry.resource,
            BindingResource::Buffer {
                buffer: buffer(),
                offset: 16,
                size: 80,
            }
        );
        assert_ne!(binding.at(0), binding.at(1));
    }

    #[test]
    fn two_bindings_differ_when_any_of_their_three_facts_do() {
        let base = StorageBufferBinding::new(buffer(), 0, 256);
        assert_eq!(base, StorageBufferBinding::new(buffer(), 0, 256));
        assert_ne!(base, StorageBufferBinding::new(buffer(), 4, 256));
        assert_ne!(base, StorageBufferBinding::new(buffer(), 0, 128));
        assert_ne!(
            base,
            StorageBufferBinding::new(
                BufferId::new(buffer().stamp(), PhysicalResourceIdentity::new(8)),
                0,
                256,
            )
        );
    }
}
