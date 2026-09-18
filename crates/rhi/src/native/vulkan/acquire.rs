//! Step 10's acquire half, pure: what `vkAcquireNextImageKHR`'s answer means, and
//! the one quarantine rule an unpresented acquire enforces.
//!
//! `Vulkan` answers an acquire with an image index and a suboptimal flag, or with a
//! result that means one of four different things. This module is the half that turns
//! that answer into a value and names each refusal, because the interesting cases --
//! a timeout is not a lost surface, and a swapchain that needs reconfiguring is
//! neither -- cannot be produced on demand against a real driver. It creates nothing
//! and owns nothing: the semaphore, the driver call and the lease that retains both
//! are [`super::swapchain`]'s.
//!
//! # A timeout is not a failure, and "not ready" is not a timeout
//!
//! `vkAcquireNextImageKHR` can answer `VK_TIMEOUT` when the caller's own timeout
//! elapsed. That is *no image became available yet* and the surface is fine, so it is
//! [`AcquireError::Timeout`] and the caller may ask again. `VK_NOT_READY` is the
//! answer to a zero-timeout poll: it says the same thing about availability but the
//! caller asked not to wait, so it is its own sentence. Collapsing either into a
//! driver failure would turn an ordinary paced frame into a lost surface.
//!
//! `VK_ERROR_OUT_OF_DATE_KHR` is a different fact again -- the swapchain no longer
//! matches the surface and must be reconfigured -- and `VK_ERROR_SURFACE_LOST_KHR`
//! says the window system took the surface away. They need different fixes, so they
//! are different variants rather than one "surface problem".
//!
//! # The unpresented-acquire quarantine
//!
//! [`AcquiredImage`] is only the driver's answer. The lease that owns it is
//! [`super::swapchain::AcquireLease`], and its `Drop` is where plan section 4's
//! preserved semantic lives: an acquired image that is neither presented nor
//! discarded leaves a semaphore the presentation engine may still signal, so the
//! surface is poisoned and that semaphore is retained rather than destroyed or
//! reused. [`AcquireError::Poisoned`] is what a later acquire on such a surface
//! answers, and [`AcquireError::IndexOutOfRange`] is the same quarantine for the one
//! driver bug that can arrive on the *success* path.

use ash::vk;

/// The image a driver answered an acquire with.
///
/// The index addresses [`super::swapchain::Swapchain::images`]; whether it is in
/// range is checked where the image is looked up, because this value is the driver's
/// answer and nothing here owns the list it addresses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AcquiredImage {
    /// The index of the acquired image in the swapchain's own image list.
    pub(crate) index: u32,
    /// Whether the driver reported the swapchain as suboptimal for its surface.
    ///
    /// `VK_SUBOPTIMAL_KHR` is a successful acquire, not a refusal: the image is
    /// presentable and the reconfigure decision belongs to the step that owns
    /// `old_swapchain`.
    pub(crate) suboptimal: bool,
}

/// Why no image was acquired, or why a surface can no longer be acquired from.
///
/// Every variant is a distinct sentence because every one needs a different fix:
/// wait and retry, reconfigure, give up on the surface, report a driver result, or
/// refuse a quarantined surface for good.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AcquireError {
    /// The driver refused to create the acquire semaphore, so the presentation
    /// engine was never reached.
    Semaphore(vk::Result),
    /// An acquired image was neither presented nor discarded, so the acquire
    /// semaphore it used cannot be proven reusable and this surface is quarantined.
    ///
    /// This is a permanent state rather than a transient one: nothing in this
    /// backend can establish when the presentation engine is done with that
    /// semaphore, which is exactly why the lease retains it (plan section 4).
    Poisoned,
    /// The timeout elapsed before an image became available. The surface is fine.
    Timeout,
    /// No image was available and the call was asked not to wait.
    ///
    /// Distinct from [`Self::Timeout`]: the same fact about availability, reached by
    /// a caller that chose a zero timeout rather than by one whose wait ran out.
    NotReady,
    /// The swapchain no longer matches the surface and must be reconfigured.
    OutOfDate,
    /// The window system lost the surface.
    SurfaceLost,
    /// The driver refused for a reason with no more specific sentence here.
    Driver(vk::Result),
    /// The driver reported an image index the swapchain does not have.
    ///
    /// `Vulkan` promises an index into the swapchain's own image list, so this is a
    /// driver that contradicted its contract. The acquire itself succeeded, so the
    /// semaphore may be signaled at a time this backend cannot know: the surface is
    /// poisoned and the semaphore is retained, exactly as an unpresented drop does.
    IndexOutOfRange {
        /// The index the driver reported.
        index: u32,
        /// How many images the swapchain actually owns.
        images: usize,
    },
}

/// Lowers one `vkAcquireNextImageKHR` answer.
///
/// `Ok` carries the index and the suboptimal flag as they arrived; each error result
/// becomes its own variant, and anything else is reported as the driver's own value
/// rather than folded into one of the named cases.
pub(crate) fn outcome(
    result: Result<(u32, bool), vk::Result>,
) -> Result<AcquiredImage, AcquireError> {
    match result {
        Ok((index, suboptimal)) => Ok(AcquiredImage { index, suboptimal }),
        Err(vk::Result::TIMEOUT) => Err(AcquireError::Timeout),
        Err(vk::Result::NOT_READY) => Err(AcquireError::NotReady),
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(AcquireError::OutOfDate),
        Err(vk::Result::ERROR_SURFACE_LOST_KHR) => Err(AcquireError::SurfaceLost),
        Err(result) => Err(AcquireError::Driver(result)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_successful_acquire_carries_its_index_and_its_suboptimal_flag() {
        // The flag is information, not a refusal: a suboptimal acquire still hands
        // the caller a presentable image, and dropping the flag would hide the only
        // signal that a reconfigure is due.
        assert_eq!(
            outcome(Ok((2, false))),
            Ok(AcquiredImage {
                index: 2,
                suboptimal: false
            })
        );
        assert_eq!(
            outcome(Ok((0, true))),
            Ok(AcquiredImage {
                index: 0,
                suboptimal: true
            })
        );
    }

    #[test]
    fn a_timeout_and_a_poll_that_did_not_wait_are_different_sentences() {
        // Both say "no image yet", but they arrive for different reasons and a caller
        // acts on them differently.
        assert_eq!(outcome(Err(vk::Result::TIMEOUT)), Err(AcquireError::Timeout));
        assert_eq!(
            outcome(Err(vk::Result::NOT_READY)),
            Err(AcquireError::NotReady)
        );
    }

    #[test]
    fn a_stale_swapchain_and_a_lost_surface_are_different_sentences() {
        // One is fixed by reconfiguring the swapchain, the other by giving up on the
        // window system's surface; a single "surface problem" would erase that.
        assert_eq!(
            outcome(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)),
            Err(AcquireError::OutOfDate)
        );
        assert_eq!(
            outcome(Err(vk::Result::ERROR_SURFACE_LOST_KHR)),
            Err(AcquireError::SurfaceLost)
        );
    }

    #[test]
    fn every_refusal_is_distinct_and_a_driver_result_is_not_folded_into_one() {
        // The mapping must not invent a named case for a result it has not been
        // taught: an unrecognised driver result is reported as the driver's own
        // value, which is what a diagnostic needs.
        let named = [
            outcome(Err(vk::Result::TIMEOUT)),
            outcome(Err(vk::Result::NOT_READY)),
            outcome(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)),
            outcome(Err(vk::Result::ERROR_SURFACE_LOST_KHR)),
        ];
        for (index, error) in named.iter().enumerate() {
            for other in &named[index + 1..] {
                assert_ne!(error, other, "each acquire refusal is its own sentence");
            }
        }
        assert_eq!(
            outcome(Err(vk::Result::ERROR_DEVICE_LOST)),
            Err(AcquireError::Driver(vk::Result::ERROR_DEVICE_LOST))
        );
    }

    #[test]
    fn the_semaphore_and_quarantine_refusals_are_not_driver_results() {
        // These two are this backend's own facts, not driver results, so they can
        // never be produced by the lowering and stay distinct from it.
        assert_ne!(AcquireError::Poisoned, AcquireError::NotReady);
        assert_ne!(
            AcquireError::IndexOutOfRange { index: 3, images: 3 },
            AcquireError::Poisoned
        );
        assert_ne!(
            AcquireError::Semaphore(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            AcquireError::Driver(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        );
    }
}
