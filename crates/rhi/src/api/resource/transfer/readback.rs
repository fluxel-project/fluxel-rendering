//! Section 18: readback tickets — resource bytes back to the caller.

use super::validate_texture_region;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::resource::buffer::{
    Buffer, BufferRange, BufferUsage, validate_buffer_ownership, validate_buffer_range,
};
use crate::api::resource::subresource::{Origin3d, TextureSubresourceLayers};
use crate::api::resource::texture::{Extent3d, Texture, TextureUsage, validate_texture_ownership};
use crate::api::submission::CompletionPoint;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::task::{Context, Poll, Waker};

/// A readback request: one buffer range, or one texture region.
///
/// The label lives inside each variant rather than on the enum because the two
/// requests describe different things, and a single label would have to mean
/// "the buffer or the texture" to a capture tool that needs to say which.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ReadbackRequest {
    /// Read a byte range of a buffer.
    Buffer {
        /// Diagnostic label. Excluded from every canonical hash (section 19.8).
        label: Label,
        /// The buffer to read from.
        src: Buffer,
        /// The range to read.
        range: BufferRange,
    },

    /// Read a texel region of a texture.
    Texture {
        /// Diagnostic label. Excluded from every canonical hash (section 19.8).
        label: Label,

        /// The texture to read from.
        src: Texture,

        /// Which mip level and array layers are read.
        subresource: TextureSubresourceLayers,
        /// Where in the level the region starts.
        origin: Origin3d,
        /// Size of the region in texels.
        extent: Extent3d,
    },
}
/// Where a readback request is in its life.
///
/// Section 18.2's state machine, transcribed:
///
/// ```text
/// NotSubmitted
///     |- submit accepted -> Pending
///     |- work/plan drop  -> Abandoned
///     \- device loss     -> DeviceLost
///
/// Pending
///     |- success         -> Ready
///     |- device loss     -> DeviceLost
///     \- backend failure -> Failed
/// ```
///
/// `NotSubmitted` and `Pending` are the only two non-terminal states, and
/// section 18.2 requires that a ticket never rests in either permanently. That
/// is a contract on the device, not on this enum: it is what makes polling a way
/// to make progress rather than a way to wait.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadbackStatus {
    /// Encoded into RecordedWork/Plan not yet successfully submitted.
    NotSubmitted,

    /// Successfully submitted, awaiting terminal GPU completion.
    Pending,

    /// CPU data is readable.
    Ready,

    /// Corresponding RecordedWork / SubmissionPlan was discarded before successful submit.
    Abandoned,

    /// The device was lost before the data was readable.
    DeviceLost,

    /// The GPU work reached a terminal failure.
    Failed,
}
impl ReadbackStatus {
    /// The wire value stored in a ticket's shared state.
    ///
    /// Private rather than `pub`, and defined by an exhaustive match rather than
    /// by `as u8`, so that adding a status is a compile error here instead of a
    /// silently wrong number in a shared cell.
    fn to_raw(self) -> u8 {
        match self {
            Self::NotSubmitted => 0,
            Self::Pending => 1,
            Self::Ready => 2,
            Self::Abandoned => 3,
            Self::DeviceLost => 4,
            Self::Failed => 5,
        }
    }

    /// The inverse of [`Self::to_raw`].
    ///
    /// The catch-all is unreachable for every value this crate writes, and it
    /// reports `Failed` rather than panicking: a state cell that cannot be
    /// interpreted is a backend fault, and section 18.2 makes `Failed` exactly
    /// the terminal state for one.
    fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::NotSubmitted,
            1 => Self::Pending,
            2 => Self::Ready,
            3 => Self::Abandoned,
            4 => Self::DeviceLost,
            _ => Self::Failed,
        }
    }
}
/// The byte layout of readback texels.
///
/// Readback does not promise tightly packed bytes, and section 18.3 gives three
/// reasons: a D3D12 copy footprint distinguishes an unpadded row size from an
/// aligned row pitch, backend staging layouts differ, and forcing a second CPU
/// repack inside the RHI would cost more than it saves. Returning the layout
/// instead makes the repack the caller's decision — and for a capture artifact
/// that wants a canonical blob, the capture layer's decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadbackTexelLayout {
    /// Byte distance between starts of adjacent valid rows.
    pub bytes_per_row: u32,

    /// Number of rows between starts of adjacent images/layers/depth slices.
    pub rows_per_image: u32,

    /// Total length of returned byte slice.
    pub total_size: u64,
}
/// CPU-readable bytes from a completed readback.
///
/// Borrowed from the ticket, because the bytes stay owned by the ticket's shared
/// state: a copy would double the peak memory of the one operation whose whole
/// purpose is moving bytes to the CPU.
///
/// This type has no `Debug`, `Clone`, or `Copy` impl, because section 18.3 shows
/// it with `#[non_exhaustive]` alone. A byte buffer that large should not be
/// cloned implicitly, and its `Debug` form would be a multi-megabyte log line.
#[non_exhaustive]
pub enum ReadbackViewData<'a> {
    /// A buffer range's bytes, tightly packed by definition.
    Buffer {
        /// The bytes read.
        bytes: &'a [u8],
    },

    /// A texture region's bytes, in the layout the backend produced.
    Texture {
        /// The bytes read, valid within `layout`.
        bytes: &'a [u8],
        /// Where the valid rows are inside `bytes`.
        layout: ReadbackTexelLayout,
    },
}

/// A scoped CPU mapping lease for a completed readback.
///
/// The view deliberately owns neither the bytes nor the ticket.  It borrows the
/// ticket's shared state, and its `Drop` is therefore the one place where a
/// backend that maps native staging memory may end that mapping lease.  The
/// current portable payload is already CPU-owned, so no backend action is needed
/// on drop; keeping the guard as a distinct type freezes the required lifetime
/// before a mapped backend is introduced.
pub struct ReadbackView<'a> {
    payload: &'a ReadbackPayload,
    _ticket: core::marker::PhantomData<&'a ReadbackTicket>,
}

impl<'a> ReadbackView<'a> {
    fn new(payload: &'a ReadbackPayload) -> Self {
        Self {
            payload,
            _ticket: core::marker::PhantomData,
        }
    }

    /// The ready bytes and, for a texture, the backend-produced texel layout.
    pub fn data(&self) -> ReadbackViewData<'_> {
        match self.payload.layout {
            Some(layout) => ReadbackViewData::Texture {
                bytes: &self.payload.bytes,
                layout,
            },
            None => ReadbackViewData::Buffer {
                bytes: &self.payload.bytes,
            },
        }
    }
}

impl Drop for ReadbackView<'_> {
    fn drop(&mut self) {
        // The portable payload has no native mapping to close. Backends that map
        // a staging allocation attach their unmap lease to this guard.
    }
}
/// The CPU bytes a completed readback published.
struct ReadbackPayload {
    bytes: Vec<u8>,
    /// Present exactly for a texture readback, which is the case whose layout is
    /// not implied by the request.
    layout: Option<ReadbackTexelLayout>,
}
/// Device-scoped state shared by every clone of one ticket.
///
/// All of this state is shared rather than copied because a ticket's clones are
/// one ticket: a cloned ticket that snapshotted its status would disagree with
/// the original the moment the device advanced, which is precisely the bug
/// section 18.4's "shared device-scoped state" note exists to prevent.
struct ReadbackTicketInner {
    id: ObjectId,
    device: DeviceIdentity,
    request: ReadbackRequest,
    /// The current [`ReadbackStatus`], as its wire value.
    ///
    /// Atomic because the device advances it from wherever completion is
    /// observed while the caller reads it from its own thread; the ticket
    /// imposes no locking on either side.
    status: AtomicU8,

    /// The published bytes, written once and never replaced.
    ///
    /// A `OnceLock` rather than a mutex: [`ReadbackTicket::try_read`] must hand
    /// out a borrow that lives as long as the ticket, and a lock guard cannot do
    /// that.
    data: std::sync::OnceLock<ReadbackPayload>,

    /// The point the ticket's work was accepted under, written once by the
    /// submit path.
    ///
    /// A `OnceLock` for the same reason as the payload beside it: the point is
    /// set once and never replaced, and readers take it by value rather than
    /// through a lock guard. It lives in this shared state rather than on the
    /// ticket because the device holds one clone and hands another to the
    /// caller — a clone that kept its own copy of the point would report `None`
    /// for work the device had already accepted.
    completion: std::sync::OnceLock<CompletionPoint>,
    /// Wakers registered by [`ReadbackTicket::read`].  Completion publishes a
    /// terminal state and wakes them, so async read never polls `Device::poll()`
    /// in a loop.
    waiters: std::sync::Mutex<Vec<Waker>>,
}
/// A handle to a pending, then completed, readback.
///
/// Cloning is sharing, not duplicating: every clone observes the same status,
/// the same completion point, and the same bytes. The ticket binds its own
/// [`DeviceIdentity`] and its own
/// state, so section 18.4's note applies — the caller never passes the device
/// again to ask about it.
#[derive(Clone)]
pub struct ReadbackTicket {
    inner: Arc<ReadbackTicketInner>,
}
impl ReadbackTicket {
    /// Assembles a ticket for a freshly encoded request.
    ///
    /// Crate-private: a ticket comes from `CommandRecorder::encode_readback`,
    /// which is the command chapter's verb, so nothing else may produce the pair
    /// of identity and shared state.
    pub(crate) fn new(id: ObjectId, device: DeviceIdentity, request: ReadbackRequest) -> Self {
        Self {
            inner: Arc::new(ReadbackTicketInner {
                id,
                device,
                request,
                status: AtomicU8::new(ReadbackStatus::NotSubmitted.to_raw()),
                data: std::sync::OnceLock::new(),
                completion: std::sync::OnceLock::new(),
                waiters: std::sync::Mutex::new(Vec::new()),
            }),
        }
    }

    /// This ticket's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }

    /// The device that owns this ticket's state.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }

    /// Original portable request.
    pub fn request(&self) -> &ReadbackRequest {
        &self.inner.request
    }

    /// The ticket's current state.
    ///
    /// Device need not be passed again;
    /// ticket already binds its own DeviceIdentity / internal state.
    pub fn status(&self) -> ReadbackStatus {
        ReadbackStatus::from_raw(self.inner.status.load(Ordering::Acquire))
    }

    /// The completion point the ticket's work was accepted under.
    ///
    /// `None` until a successful submission records one. Section 41.5 binds a
    /// ticket to the corresponding point of the plan that carries its readback,
    /// and that binding is created by the submit path; a ticket cannot mint a
    /// point for work the device has not accepted, so the unsubmitted case has
    /// nothing to report rather than a placeholder.
    ///
    /// The point is a token to wait on, not a result: `Some` says this work has
    /// a completion to observe, not that the bytes are readable — that stays
    /// [`Self::status`]'s and [`Self::try_read`]'s answer. It is also not a
    /// native fence value (section 41.1); it is named to
    /// [`DeviceIdentity`] so a foreign device's token is refusable.
    pub fn completion(&self) -> Option<CompletionPoint> {
        self.inner.completion.get().copied()
    }

    /// The bytes, once the request is `Ready`.
    ///
    /// ```text
    /// Ok(None)    not readable yet: NotSubmitted or Pending
    /// Ok(Some(_)) the data, while the state is Ready
    /// Err(_)      a terminal state that will never produce data
    /// ```
    ///
    /// Section 18.4 does not state which kind each terminal state reports, so
    /// this maps them by section 4's vocabulary: `DeviceLost` is
    /// [`RhiErrorKind::DeviceLost`], a backend failure is
    /// [`RhiErrorKind::BackendFailure`], and an abandoned request — dropped
    /// before a successful submit, so the data can never exist — reports
    /// [`RhiErrorKind::InvalidUsage`], because the caller is asking about work it
    /// discarded itself. None of the three returns `Ok(None)`: that would mean
    /// "keep polling", and section 18.2 makes all three terminal.
    ///
    /// The borrowing is why this is not `Result<Option<Vec<u8>>>`: the bytes live
    /// in the ticket's shared state, and copying them out would double the peak
    /// memory of the one operation whose purpose is moving bytes to the CPU.
    pub fn try_read<'a>(&'a self) -> RhiResult<Option<ReadbackView<'a>>> {
        match self.status() {
            ReadbackStatus::Ready => {
                let payload = self.inner.data.get().ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "readback is Ready but no bytes were published; the device must \
                         publish data before it reports Ready",
                    )
                    .with_object(self.inner.id)
                })?;
                Ok(Some(ReadbackView::new(payload)))
            }
            ReadbackStatus::NotSubmitted | ReadbackStatus::Pending => Ok(None),
            ReadbackStatus::Abandoned => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the recorded work carrying this readback was discarded before it was \
                 submitted, so it will never produce data",
            )
            .with_object(self.inner.id)),
            ReadbackStatus::DeviceLost => Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "the device was lost before this readback completed",
            )
            .with_object(self.inner.id)),
            ReadbackStatus::Failed => Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "the readback's GPU work failed",
            )
            .with_object(self.inner.id)),
        }
    }

    /// Waits for a terminal readback result without driving the device itself.
    ///
    /// The completion path wakes this future after it publishes data or a
    /// terminal failure. A caller must continue to run the platform event loop;
    /// this method does not turn it into a busy `Device::poll()` loop.
    pub async fn read(&self) -> RhiResult<ReadbackView<'_>> {
        std::future::poll_fn(|context| self.poll_read(context)).await
    }

    fn poll_read<'a>(&'a self, context: &mut Context<'_>) -> Poll<RhiResult<ReadbackView<'a>>> {
        match self.try_read() {
            Ok(Some(view)) => return Poll::Ready(Ok(view)),
            Err(error) => return Poll::Ready(Err(error)),
            Ok(None) => {}
        }

        let mut waiters = self
            .inner
            .waiters
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // Recheck while holding the waiter lock. This closes the interval where
        // completion could otherwise publish immediately before our waker is
        // registered and leave the future asleep forever.
        match self.try_read() {
            Ok(Some(view)) => Poll::Ready(Ok(view)),
            Err(error) => Poll::Ready(Err(error)),
            Ok(None) => {
                if !waiters.iter().any(|waker| waker.will_wake(context.waker())) {
                    waiters.push(context.waker().clone());
                }
                Poll::Pending
            }
        }
    }

    /// Advances the ticket's state.
    ///
    /// Crate-private, and reached by the device when it observes a submit, a
    /// discarded plan, or a loss. [`Self::publish`] is the only correct way to
    /// reach [`ReadbackStatus::Ready`]: a state without bytes is not readable,
    /// and `try_read` reports it as a backend fault.
    pub(crate) fn set_status(&self, status: ReadbackStatus) {
        if self.transition_to(status)
            && matches!(
                status,
                ReadbackStatus::Abandoned | ReadbackStatus::DeviceLost | ReadbackStatus::Failed
            )
        {
            self.wake_waiters();
        }
    }

    /// Performs the monotone half of the readback state machine.
    ///
    /// Native completion and device-loss observation may race on different
    /// threads. The first terminal transition wins permanently: a late portable
    /// `Pending` handoff cannot erase `Ready`, and a late mapping result cannot
    /// turn `DeviceLost` back into readable data.
    fn transition_to(&self, next: ReadbackStatus) -> bool {
        let mut current = self.inner.status.load(Ordering::Acquire);
        loop {
            let status = ReadbackStatus::from_raw(current);
            if status == next {
                return false;
            }
            if !matches!(
                status,
                ReadbackStatus::NotSubmitted | ReadbackStatus::Pending
            ) {
                return false;
            }
            if next == ReadbackStatus::Pending && status != ReadbackStatus::NotSubmitted {
                return false;
            }
            match self.inner.status.compare_exchange_weak(
                current,
                next.to_raw(),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    /// Records the completion point the ticket was accepted under.
    ///
    /// Crate-private, and reached by the device when a submit succeeds. Only the
    /// first call takes effect: section 41.5 binds a ticket to the point of the
    /// submission that carried it, so a later call would describe a second
    /// submission, and overwriting the point would move a caller's wait onto
    /// work it never encoded.
    ///
    /// The order is a contract, as in [`Self::publish`]: record the point before
    /// advancing the status to `Pending`, so a caller that observes the submit
    /// also observes the point that covers it.
    /// The caller that arrived is `Device::submit`'s readback walk, which runs in
    /// every build because it is portable code — a backend hands back serials, and
    /// turning a serial into the token a caller holds is this layer's work. The
    /// expectation that named "the backend port" as the future caller was
    /// therefore wrong about *which* layer would call it, which is why the
    /// attribute is gone rather than gated.
    pub(crate) fn set_completion(&self, point: CompletionPoint) {
        // The only failure is a second call, which is ignored rather than
        // overwritten: the point a caller already observed stays true.
        let _ = self.inner.completion.set(point);
    }

    /// Publishes the bytes and marks the ticket `Ready`.
    ///
    /// Crate-private. The order is the contract: the payload is stored first and
    /// the status is released afterwards, while readers acquire the status
    /// before touching the payload. A reader that observes `Ready` therefore
    /// also observes the bytes, which is what makes `try_read`'s `Ready` branch
    /// infallible in practice.
    /// The publisher that arrived is the DX12 command spine, which maps the
    /// readback staging allocation once the fence reports the batch finished and
    /// hands the bytes over here. It is still gated on the backend feature list
    /// rather than ungated, because a build with no backend compiled has no
    /// device that could complete GPU work in the first place.
    #[cfg_attr(
        not(any(
            test,
            // The backend features that actually compile a lowering. A feature
            // that selects nothing must not appear here: it would remove this
            // expectation in a configuration where the item really is dead, and
            // the gate would then be silent about it. When Vulkan lands and starts
            // calling this, its feature joins the list — which is rule 4.6's
            // "the matrix gets the row" applied to the attribute itself.
            feature = "dx12",
            feature = "vulkan"
        )),
        expect(
            dead_code,
            reason = "a backend publishes this when the readback's GPU work completes; a build \
                      with no backend compiled has no device that could complete any"
        )
    )]
    pub(crate) fn publish(&self, bytes: Vec<u8>, layout: Option<ReadbackTexelLayout>) {
        // The only failure is a second publish, which would silently discard the
        // first one's bytes; keeping the first is the conservative choice,
        // because the status a caller already acted on stays true.
        let _ = self.inner.data.set(ReadbackPayload { bytes, layout });
        if self.transition_to(ReadbackStatus::Ready) {
            self.wake_waiters();
        }
    }

    fn wake_waiters(&self) {
        let waiters = {
            let mut waiters = self
                .inner
                .waiters
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            core::mem::take(&mut *waiters)
        };
        for waker in waiters {
            waker.wake();
        }
    }
}
/// Checks a buffer readback.
///
/// Section 18.1's buffer list, portable half: `COPY_SRC` usage and a valid range.
/// The third item on the list — the buffer-to-buffer route's alignment — is the
/// device's answer, and it is deliberately *not* taken here. A validator that
/// took the limits would be deciding a device question inside a portable one, and
/// the caller would lose the difference between "you described this wrongly" and
/// "this device cannot do it": an empty device has no limits to pass, so the
/// alignment branch would have to answer `InvalidUsage` for a request whose
/// actual problem is that the route does not exist.
/// [`CommandRecorder::encode_readback`](crate::api::command::CommandRecorder::encode_readback)
/// asks the route first and then applies the alignment through the same
/// [`BufferCopyLayoutLimits`](crate::api::resource::route::BufferCopyLayoutLimits)
/// an upload uses.
pub(crate) fn validate_buffer_readback(
    src: &Buffer,
    range: BufferRange,
    target: DeviceIdentity,
) -> RhiResult<()> {
    validate_buffer_ownership(src, target)?;
    if !src.descriptor().usage.contains(BufferUsage::COPY_SRC) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the source buffer was not created with COPY_SRC usage",
        )
        .with_object(src.id()));
    }
    validate_buffer_range(range, src.descriptor().size)
}
/// Checks a texture readback.
///
/// Section 18.1's texture list: `COPY_SRC` usage, a single-sampled source, a
/// valid subresource/origin/extent, and a supported texture-to-buffer route. The
/// route is the device's question, asked by
/// `CommandRecorder::encode_readback`.
pub(crate) fn validate_texture_readback(
    src: &Texture,
    subresource: TextureSubresourceLayers,
    origin: Origin3d,
    extent: Extent3d,
    target: DeviceIdentity,
) -> RhiResult<()> {
    validate_texture_ownership(src, target)?;
    let base = src.descriptor();
    if !base.usage.contains(TextureUsage::COPY_SRC) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the source texture was not created with COPY_SRC usage",
        )
        .with_object(src.id()));
    }
    if base.sample_count != 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a texture readback requires a single-sampled source, not {} samples",
                base.sample_count
            ),
        )
        .with_object(src.id()));
    }
    validate_texture_region(base, subresource, origin, extent)
}
