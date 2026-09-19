//! Synchronous public lifecycle queries and controls.

use super::{
    WebGpuAdapterInfo, WebGpuLossReason, WebGpuSession, WebGpuSessionError, WebGpuSessionState,
};

impl WebGpuSession {
    /// Returns configured canvas epoch; epochs never imply completion.
    pub fn canvas_epoch(&self) -> u64 {
        self.canvas_epoch.get()
    }
    /// Returns copied adapter facts suitable for browser evidence manifests.
    pub fn adapter_info(&self) -> WebGpuAdapterInfo {
        self.adapter_info.borrow().clone()
    }
    /// Returns the most recently observed closed loss reason.
    pub fn loss_reason(&self) -> Option<WebGpuLossReason> {
        self.shared.borrow().loss_reason
    }
    /// Stops producer work without claiming GPU completion.
    pub fn suspend(&mut self) -> Result<(), WebGpuSessionError> {
        self.require_nonterminal()?;
        if matches!(
            self.state(),
            WebGpuSessionState::Active | WebGpuSessionState::Suspended
        ) {
            self.shared.borrow_mut().state = WebGpuSessionState::Suspended;
        }
        Ok(())
    }
    /// Reconfigures only when a drawable extent and active device exist.
    pub fn resume(&mut self) -> Result<WebGpuSessionState, WebGpuSessionError> {
        self.require_nonterminal()?;
        if matches!(
            self.state(),
            WebGpuSessionState::Active | WebGpuSessionState::Suspended
        ) && self.desired_extent.get()[0] != 0
            && self.desired_extent.get()[1] != 0
            && self.objects.borrow().is_some()
        {
            self.configure()?;
        }
        Ok(self.state())
    }
}
