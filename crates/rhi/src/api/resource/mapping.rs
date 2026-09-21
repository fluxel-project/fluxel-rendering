//! General host mapping for buffers explicitly created with map usage.
//!
//! Upload/readback tickets remain asynchronous transfer facilities. A mapping is
//! a direct host lease over a resource whose descriptor and capability answer
//! explicitly permit it. The caller is responsible for ordering GPU work before
//! mapping; the returned future waits for prior GPU use to retire before it
//! grants host access. This module enforces object ownership, map usage, range
//! alignment, and one active lease per buffer.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::Device;
use crate::api::platform::requirements::LimitKey;
use crate::api::resource::backend::{MappedBufferBackend, MappingRequestBackend};
use crate::api::resource::buffer::{
    Buffer, BufferRange, BufferUsage, validate_buffer_ownership, validate_buffer_range,
};

/// Direction of one buffer mapping lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapMode {
    /// CPU reads bytes produced by the device.
    Read,
    /// CPU writes bytes consumed by the device.
    Write,
}

/// A pending asynchronous mapping request returned by [`Device::map_buffer`].
///
/// The request keeps the buffer's exclusive mapping lease until it resolves or
/// is dropped.  Cancellation is therefore safe: dropping a pending future
/// releases the lease and cancels the backend request without mapping bytes.
pub struct MapBufferFuture<'a> {
    device: &'a Device,
    buffer: &'a Buffer,
    mode: MapMode,
    request: Option<Box<dyn MappingRequestBackend>>,
    /// True until this future either releases its reservation or transfers it
    /// to the resulting [`MappedRange`].
    owns_lease: bool,
    // Raw mutable pointers are neither `Send` nor `Sync`; PhantomData stores no
    // pointer and introduces no reference count. Mapping remains thread-affine
    // without adding an ownership mechanism to the public object model.
    _thread_affine: PhantomData<*mut ()>,
}

impl core::fmt::Debug for MapBufferFuture<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MapBufferFuture")
            .field("mode", &self.mode)
            .field("buffer", &self.buffer.id())
            .finish_non_exhaustive()
    }
}

impl<'a> Future for MapBufferFuture<'a> {
    type Output = RhiResult<MappedRange<'a>>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        // A loss can occur after the backend registered the waker but before it
        // reports readiness.  Check it before accepting a native lease so no
        // bytes are exposed from a dead execution domain.
        if let Err(error) = this.device.require_active() {
            this.request.take();
            this.release_lease();
            return Poll::Ready(Err(error));
        }
        let Some(request) = this.request.as_mut() else {
            return Poll::Ready(Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a mapping future was polled after it completed",
            )));
        };
        match request.poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(backend)) => {
                this.request.take();
                // The ready range now owns the sole portable reservation. Do
                // not release it from this future's subsequent Drop.
                this.owns_lease = false;
                Poll::Ready(Ok(MappedRange::new(
                    this.buffer,
                    backend,
                    matches!(this.mode, MapMode::Write),
                )))
            }
            Poll::Ready(Err(error)) => {
                this.request.take();
                this.release_lease();
                Poll::Ready(Err(error))
            }
        }
    }
}

impl Drop for MapBufferFuture<'_> {
    fn drop(&mut self) {
        // Drop the native request first: it may own a native reservation whose
        // teardown must finish before another portable map can be admitted.
        self.request.take();
        self.release_lease();
    }
}

impl MapBufferFuture<'_> {
    fn release_lease(&mut self) {
        if self.owns_lease {
            self.buffer.end_map();
            self.owns_lease = false;
        }
    }
}

/// A ready mapping lease returned by awaiting [`MapBufferFuture`].
///
/// Drop always unmaps the native allocation and releases portable exclusivity.
/// `flush` and `invalidate` are explicit visibility operations for non-coherent
/// memory; coherent backends implement them as no-ops.
pub struct MappedRange<'a> {
    buffer: &'a Buffer,
    backend: Option<Box<dyn MappedBufferBackend>>,
    writable: bool,
    // See MapBufferFuture: this is a zero-sized thread-affinity marker, not a
    // shared owner or reference-counted token.
    _thread_affine: PhantomData<*mut ()>,
}

impl core::fmt::Debug for MappedRange<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MappedRange")
            .field("mode", &self.mode())
            .field("length", &self.bytes().len())
            .finish()
    }
}

impl<'a> MappedRange<'a> {
    fn new(buffer: &'a Buffer, backend: Box<dyn MappedBufferBackend>, writable: bool) -> Self {
        Self {
            buffer,
            backend: Some(backend),
            writable,
            _thread_affine: PhantomData,
        }
    }

    /// The mapped byte range.
    pub fn bytes(&self) -> &[u8] {
        self.backend
            .as_deref()
            .expect("live mapping backend")
            .bytes()
    }

    /// Returns mutable bytes for a write mapping.
    pub fn bytes_mut(&mut self) -> RhiResult<&mut [u8]> {
        if !self.writable {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a read mapping cannot be written",
            ));
        }
        self.backend
            .as_deref_mut()
            .expect("live mapping backend")
            .bytes_mut()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "backend returned a non-writable write mapping",
                )
            })
    }

    /// Makes writes visible to the device on non-coherent memory.
    pub fn flush(&mut self) -> RhiResult<()> {
        if !self.writable {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "flush requires a write mapping",
            ));
        }
        self.backend
            .as_deref_mut()
            .expect("live mapping backend")
            .flush()
    }

    /// Makes device writes visible to the host on non-coherent memory.
    pub fn invalidate(&mut self) -> RhiResult<()> {
        if self.writable {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "invalidate requires a read mapping",
            ));
        }
        self.backend
            .as_deref_mut()
            .expect("live mapping backend")
            .invalidate()
    }

    /// The mapping direction.
    pub fn mode(&self) -> MapMode {
        if self.writable {
            MapMode::Write
        } else {
            MapMode::Read
        }
    }
}

impl Drop for MappedRange<'_> {
    fn drop(&mut self) {
        // Native unmap precedes making the portable lease available again.
        self.backend.take();
        self.buffer.end_map();
    }
}

/// Mutable mapping spelling retained for code that wants to name write leases.
pub type MappedRangeMut<'a> = MappedRange<'a>;

impl Device {
    /// Starts mapping an explicitly mappable range of `buffer`.
    ///
    /// The returned future is pending while native GPU use retires.  Awaiting it
    /// yields a borrowing RAII lease; dropping either the pending future or the
    /// ready lease releases the buffer's one portable mapping lease.
    pub fn map_buffer<'a>(
        &'a self,
        buffer: &'a Buffer,
        mode: MapMode,
        range: BufferRange,
    ) -> RhiResult<MapBufferFuture<'a>> {
        self.require_active()?;
        validate_buffer_ownership(buffer, self.identity())?;
        validate_buffer_range(range, buffer.descriptor().size)?;
        let required_usage = match mode {
            MapMode::Read => BufferUsage::MAP_READ,
            MapMode::Write => BufferUsage::MAP_WRITE,
        };
        if !buffer.descriptor().usage.contains(required_usage) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "buffer was not created with the required map usage",
            )
            .with_object(buffer.id()));
        }
        // Mapping a staging buffer is not an optional feature. The descriptor's
        // exact buffer-support answer already says whether this map usage can
        // exist on this backend. `MappablePrimaryBuffers` only describes the
        // *additional* ability to combine MAP_* with broader primary GPU uses;
        // it must not reject MAP_READ|COPY_DST or MAP_WRITE|COPY_SRC here.
        if !self
            .capabilities()
            .buffer_support(&crate::api::resource::BufferSupportQuery::new(
                buffer.descriptor().usage,
            ))
            .is_supported()
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this buffer's declared usage is no longer mappable on this device",
            )
            .with_object(buffer.id()));
        }
        // Offset and size are independent native facts.  Do not collapse them
        // to their maximum: that would reject valid WebGPU ranges (offset: 8,
        // size: 4), while collapsing to their minimum would defer a predictable
        // validation failure to the backend.  `MapAlignment` is only the
        // compatibility fallback for facts emitted before this distinction was
        // added.
        let legacy_alignment = self.capabilities().limit(LimitKey::MapAlignment);
        let offset_alignment = self
            .capabilities()
            .limit(LimitKey::MapOffsetAlignment)
            .or(legacy_alignment);
        let size_alignment = self
            .capabilities()
            .limit(LimitKey::MapSizeAlignment)
            .or(legacy_alignment);
        if let Some(alignment) = offset_alignment
            && alignment != 0
            && !range.offset.is_multiple_of(alignment)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "mapped range offset must be a multiple of map offset alignment {alignment}"
                ),
            ));
        }
        if let Some(alignment) = size_alignment
            && alignment != 0
            && !range.size.is_multiple_of(alignment)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("mapped range size must be a multiple of map size alignment {alignment}"),
            ));
        }
        buffer.begin_map()?;
        match self.native().map_buffer(buffer, mode, range) {
            Ok(request) => Ok(MapBufferFuture {
                device: self,
                buffer,
                mode,
                request: Some(request),
                owns_lease: true,
                _thread_affine: PhantomData,
            }),
            Err(error) => {
                buffer.end_map();
                Err(error)
            }
        }
    }
}
