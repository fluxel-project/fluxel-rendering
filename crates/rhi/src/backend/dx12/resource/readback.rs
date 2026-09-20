//! CPU readback mapping for completed Direct3D 12 staging buffers.
//!
//! This stays in the resource chapter because it describes a `READBACK` heap
//! allocation, not command ordering or completion. The command spine owns when
//! a completed batch invokes it and publishes the resulting bytes to the portable
//! `ReadbackTicket`.

use crate::backend::dx12::ffi;
use crate::backend::dx12::resource::Dx12Buffer;

/// Copies `size` completed bytes from a DX12 readback allocation.
///
/// The returned `Vec` deliberately owns the bytes. The portable readback view is
/// then an RAII guard over ticket-owned CPU data; it is not allowed to borrow a
/// mapped `ID3D12Resource` after this method unmaps it.
pub(crate) fn readback_bytes(staging: &Dx12Buffer, size: u64) -> Result<Vec<u8>, ffi::NativeError> {
    let length = usize::try_from(size).map_err(|_| {
        ffi::NativeError::driver_contract_violation(
            "the completed readback range does not fit this process address space",
            "Dx12 readback mapping",
        )
    })?;

    let mut pointer: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: `staging` is a READBACK heap allocation retained until completion;
    // Map writes one CPU pointer into `pointer`, which stays valid through the
    // matching Unmap below.
    unsafe {
        staging
            .resource()
            .Map(0, None, Some(&mut pointer))
            .map_err(|error| ffi::NativeError::new(&error, "Dx12 readback mapping"))?;
    }
    let Some(pointer) = core::ptr::NonNull::new(pointer.cast::<u8>()) else {
        // SAFETY: a successful Map above has one matching Unmap, even when a
        // non-conforming driver supplied a null mapping pointer.
        unsafe { staging.resource().Unmap(0, None) };
        return Err(ffi::NativeError::driver_contract_violation(
            "Map reported success without producing a pointer",
            "Dx12 readback mapping",
        ));
    };

    // SAFETY: the mapping covers this staging resource; `size` was the checked
    // copy extent used to create it. The copy completes before the matching
    // Unmap, so the returned bytes have no native mapping lifetime.
    let bytes = unsafe {
        let bytes = core::slice::from_raw_parts(pointer.as_ptr(), length).to_vec();
        staging.resource().Unmap(0, None);
        bytes
    };
    Ok(bytes)
}
