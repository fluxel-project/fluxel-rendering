//! Step 10's present half, pure: what `vkQueuePresentKHR`'s answer means.
//!
//! `Vulkan` answers a present with success, with success plus a suboptimal flag, or
//! with a result that means the swapchain must be reconfigured or the surface is
//! gone. This module is the half that names each of those, because none of them can
//! be produced on demand against a real driver -- the same reason step 10's acquire
//! lowering and step 9's fence answers are split this way. It creates nothing and
//! owns nothing: the driver call, the lease it consumes, and the wait semaphore it
//! retains are [`super::swapchain`]'s.
//!
//! # A suboptimal present is a present
//!
//! `VK_SUBOPTIMAL_KHR` says the swapchain still matches the surface well enough to
//! present, but a reconfigure is due. `ash` already separates it from success by
//! mapping it to `Ok(true)`, so it is [`PresentOutcome::Suboptimal`] rather than a
//! refusal: the image reached the presentation engine, and the decision to rebuild
//! the swapchain belongs to the step that owns `old_swapchain`.
//!
//! # Out of date is not surface lost
//!
//! `VK_ERROR_OUT_OF_DATE_KHR` is fixed by reconfiguring the swapchain, and
//! `VK_ERROR_SURFACE_LOST_KHR` is fixed by giving up on the window system's surface.
//! They need different callers, so they are different variants rather than one
//! "surface problem".

use ash::vk;

/// What a driver answered a present with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentOutcome {
    /// The image was presented.
    Presented,
    /// The image was presented, and the swapchain is now suboptimal for the surface.
    Suboptimal,
}

impl PresentOutcome {
    /// Whether the driver reported the swapchain as suboptimal for its surface.
    ///
    /// It is reported rather than acted on, exactly as the acquire's suboptimal flag
    /// is: the present succeeded, and reconfigure is a separate step with its own
    /// `old_swapchain` ownership.
    pub(crate) const fn is_suboptimal(self) -> bool {
        matches!(self, Self::Suboptimal)
    }
}

/// Why a present did not happen.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentError {
    /// The swapchain no longer matches the surface and must be reconfigured.
    OutOfDate,
    /// The window system lost the surface.
    SurfaceLost,
    /// The driver refused for a reason with no more specific sentence here.
    ///
    /// It keeps the driver's own value rather than being folded into a named case
    /// this backend has not been taught, which is what a diagnostic needs.
    Driver(vk::Result),
}

/// Lowers one `vkQueuePresentKHR` answer.
///
/// `Ok(false)` is `VK_SUCCESS` and `Ok(true)` is `VK_SUBOPTIMAL_KHR`; `ash` has
/// already made that distinction, so it is carried through rather than re-derived.
/// Each named error result becomes its own variant and anything else is reported as
/// the driver's own value.
pub(crate) fn outcome(result: Result<bool, vk::Result>) -> Result<PresentOutcome, PresentError> {
    match result {
        Ok(false) => Ok(PresentOutcome::Presented),
        Ok(true) => Ok(PresentOutcome::Suboptimal),
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(PresentError::OutOfDate),
        Err(vk::Result::ERROR_SURFACE_LOST_KHR) => Err(PresentError::SurfaceLost),
        Err(result) => Err(PresentError::Driver(result)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_successful_present_and_a_suboptimal_one_are_different_sentences() {
        // Suboptimal is information, not a refusal: the image was presented, and
        // dropping the flag would hide the only signal that a reconfigure is due.
        assert_eq!(outcome(Ok(false)), Ok(PresentOutcome::Presented));
        assert_eq!(outcome(Ok(true)), Ok(PresentOutcome::Suboptimal));
        assert!(!PresentOutcome::Presented.is_suboptimal());
        assert!(PresentOutcome::Suboptimal.is_suboptimal());
    }

    #[test]
    fn a_stale_swapchain_and_a_lost_surface_are_different_sentences() {
        // One is fixed by reconfiguring the swapchain, the other by giving up on the
        // window system's surface; a single "surface problem" would erase that.
        assert_eq!(
            outcome(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)),
            Err(PresentError::OutOfDate)
        );
        assert_eq!(
            outcome(Err(vk::Result::ERROR_SURFACE_LOST_KHR)),
            Err(PresentError::SurfaceLost)
        );
    }

    #[test]
    fn every_refusal_is_distinct_and_a_driver_result_is_not_folded_into_one() {
        // The mapping must not invent a named case for a result it has not been
        // taught: an unrecognised driver result is reported as the driver's own
        // value.
        let named = [
            outcome(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)),
            outcome(Err(vk::Result::ERROR_SURFACE_LOST_KHR)),
        ];
        for (index, error) in named.iter().enumerate() {
            for other in &named[index + 1..] {
                assert_ne!(error, other, "each present refusal is its own sentence");
            }
        }
        assert_eq!(
            outcome(Err(vk::Result::ERROR_DEVICE_LOST)),
            Err(PresentError::Driver(vk::Result::ERROR_DEVICE_LOST))
        );
        assert_eq!(
            outcome(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)),
            Err(PresentError::Driver(vk::Result::ERROR_OUT_OF_HOST_MEMORY))
        );
    }
}
