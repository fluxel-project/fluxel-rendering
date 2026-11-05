use super::{Renderer, RendererError};
use fluxel_rhi::{
    api::Error,
    backend::{ExecutionBackend, MappingBackend, ReadbackBackend, ResourceBackend},
};

/// Counts the renderer-owned values released by a successful shutdown.
///
/// A report with zero counts is returned when shutdown is called again after a
/// prior successful shutdown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShutdownReport {
    /// Pending submissions whose retained owners were released after idle.
    pub retired_submissions: usize,
    /// Private compiled fixture plans released after idle.
    pub released_cached_plans: usize,
}

/// Whether this renderer may create more device-specific work.
pub(super) enum Lifecycle {
    Active,
    Closing,
    Closed,
    Unavailable(Error),
}

impl Lifecycle {
    pub(super) fn ensure_open(&self) -> Result<(), RendererError> {
        match self {
            Self::Active => Ok(()),
            Self::Closing | Self::Closed | Self::Unavailable(_) => {
                Err(RendererError::RendererClosed)
            }
        }
    }
}

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Waits for the device to become idle, then releases all renderer-owned
    /// submission owners and cached fixture plans.
    ///
    /// Only a successful RHI idle wait is retirement evidence. If it fails,
    /// including with device loss, all pending owners and snapshots remain
    /// retained and this renderer rejects new work. Create a renderer for a
    /// replacement device instead of reusing this instance.
    pub async fn shutdown(&mut self) -> Result<ShutdownReport, RendererError> {
        match &self.lifecycle {
            Lifecycle::Closed => return Ok(ShutdownReport::default()),
            Lifecycle::Unavailable(error) => return Err(RendererError::Rhi(error.clone())),
            Lifecycle::Active | Lifecycle::Closing => {}
        }

        self.lifecycle = Lifecycle::Closing;

        if let Err(error) = self.device.wait_idle().await {
            if matches!(error, Error::DeviceLost) {
                self.plans.clear();
                self.clear_deferred_caches();
                self.forward_programs.clear();
                self.forward_residency.clear();
                self.forward_plans.clear();
                self.forward_surface_plans.clear();
                self.forward_frames = std::array::from_fn(|_| None);
                self.lifecycle = Lifecycle::Unavailable(error.clone());
            }
            return Err(RendererError::Rhi(error));
        }

        let retired_submissions = self.slots.iter_mut().filter_map(Option::take).count();
        let released_cached_plans = self.plans.len()
            + self.forward_plans.len()
            + self.forward_surface_plans.len()
            + self.deferred_plans.len()
            + self.deferred_surface_plans.len();
        self.plans.clear();
        self.clear_deferred_caches();
        self.forward_programs.clear();
        self.forward_residency.clear();
        self.forward_plans.clear();
        self.forward_surface_plans.clear();
        self.forward_frames = std::array::from_fn(|_| None);
        self.lifecycle = Lifecycle::Closed;
        Ok(ShutdownReport {
            retired_submissions,
            released_cached_plans,
        })
    }

    pub(super) fn ensure_device_available(&self) -> Result<(), RendererError> {
        match &self.lifecycle {
            Lifecycle::Unavailable(error) => Err(RendererError::Rhi(error.clone())),
            _ => Ok(()),
        }
    }

    /// Quarantines this device domain after RHI reports terminal loss.
    ///
    /// Pending owners remain in their slots because loss is not completion
    /// evidence. Cached fixture plans have no remaining executable device.
    pub(super) fn note_device_loss(&mut self, error: &Error) {
        if matches!(error, Error::DeviceLost) {
            self.plans.clear();
            self.clear_deferred_caches();
            self.forward_programs.clear();
            self.forward_residency.remove_device(self.device.id());
            self.forward_plans.clear();
            self.forward_surface_plans.clear();
            self.forward_frames = std::array::from_fn(|_| None);
            self.lifecycle = Lifecycle::Unavailable(error.clone());
        }
    }
}
