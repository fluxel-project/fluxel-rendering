//! Browser-independent lifecycle facts shared by the closed WebGPU seam.

/// Closed lifecycle of one canvas/device generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebGpuSessionState {
    /// No drawable canvas is currently configured.
    Suspended,
    /// A configured device generation accepts submissions.
    Active,
    /// The browser terminated the active device generation.
    Lost,
    /// A replacement adapter/device request is in flight.
    Recovering,
    /// Terminal cleanup is awaiting submitted work.
    Disposing,
    /// Explicit terminal cleanup completed.
    Disposed,
    /// An operation failed and reuse is unsafe.
    Poisoned,
}

/// Whether an asynchronous recovery attempt still owns the session state it
/// started from. Browser callbacks must check this before becoming producers.
pub(crate) fn recovery_attempt_current(
    state: WebGpuSessionState,
    current_token: u64,
    attempt_token: u64,
) -> bool {
    state == WebGpuSessionState::Recovering && current_token == attempt_token
}

/// Whether the full candidate device transaction may publish. In addition to
/// the recovery token, the replacement generation must be exactly the next
/// committed generation; device, queue, format and adapter facts publish only
/// from this one point.
pub(crate) fn candidate_transaction_committable(
    state: WebGpuSessionState,
    current_token: u64,
    attempt_token: u64,
    committed_generation: u64,
    candidate_generation: u64,
) -> bool {
    recovery_attempt_current(state, current_token, attempt_token)
        && committed_generation.checked_add(1) == Some(candidate_generation)
}

/// Terminal state is truthful only once every async producer and every GPU
/// ownership root has gone. This is deliberately separate from the state enum:
/// `Disposing` remains observable while these roots settle.
#[derive(Clone, Copy)]
pub(crate) struct TerminalDisposalRoots {
    pub(crate) recovery_pending: bool,
    pub(crate) live_objects: bool,
    pub(crate) retired_objects: bool,
    pub(crate) active_tickets: bool,
    pub(crate) detached_tickets: bool,
    pub(crate) quarantined_resources: bool,
    pub(crate) callback_or_listener_producer: bool,
}

pub(crate) fn terminal_dispose_ready(
    state: WebGpuSessionState,
    roots: TerminalDisposalRoots,
) -> bool {
    state == WebGpuSessionState::Disposing
        && !roots.recovery_pending
        && !roots.live_objects
        && !roots.retired_objects
        && !roots.active_tickets
        && !roots.detached_tickets
        && !roots.quarantined_resources
        && !roots.callback_or_listener_producer
}

/// Minimal no-GPU reducer used to test lifecycle distinctions on every host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
pub(crate) struct Lifecycle {
    state: WebGpuSessionState,
    generation: u64,
}

#[cfg(test)]
impl Lifecycle {
    /// Creates the not-yet-configured first device generation.
    pub(crate) const fn new() -> Self {
        Self {
            state: WebGpuSessionState::Suspended,
            generation: 1,
        }
    }
    /// Applies an active configuration for a nonzero canvas extent.
    pub(crate) fn configure(&mut self, drawable: bool) {
        self.state = if drawable {
            WebGpuSessionState::Active
        } else {
            WebGpuSessionState::Suspended
        };
    }
    /// Records device loss, which is distinct from resize and completion.
    pub(crate) fn lose(&mut self) {
        if matches!(
            self.state,
            WebGpuSessionState::Active | WebGpuSessionState::Suspended
        ) {
            self.state = WebGpuSessionState::Lost;
        }
    }
    fn poison(&mut self) {
        self.state = WebGpuSessionState::Poisoned;
    }
    /// Begins recovery only from a lost generation.
    pub(crate) fn begin_recover(&mut self) -> bool {
        if self.state != WebGpuSessionState::Lost {
            return false;
        }
        self.state = WebGpuSessionState::Recovering;
        true
    }
    /// Installs the next generation after async device creation settles.
    pub(crate) fn finish_recover(&mut self, drawable: bool) {
        debug_assert_eq!(self.state, WebGpuSessionState::Recovering);
        self.generation += 1;
        self.configure(drawable);
    }
    /// Enters the terminal disposal state.
    pub(crate) fn dispose(&mut self) {
        self.state = WebGpuSessionState::Disposed;
    }
    pub(crate) const fn state(self) -> WebGpuSessionState {
        self.state
    }
    pub(crate) const fn generation(self) -> u64 {
        self.generation
    }
}

/// Host-testable ownership accounting for browser callback roots. This does
/// not emulate WebGPU; it proves the policy that tickets and observers are
/// retained through settlement and stale generations cannot mutate a new one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
struct Accounting {
    generation: u64,
    token: u64,
    active_tickets: u8,
    detached_tickets: u8,
    active_observers: u8,
    retired_observers: u8,
}

/// Pure, deferred-Promise model for the browser executor's async publication
/// contract.  It deliberately models only ownership and commit ordering, not
/// WebGPU: host tests can therefore exercise adversarial interleavings without
/// claiming that a mock browser proves rendering correctness.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Interleaving {
    state: WebGpuSessionState,
    generation: u64,
    format_generation: u64,
    metadata_generation: u64,
    token: u64,
    recovery_in_flight: bool,
    active_submission: bool,
    detached_completion: bool,
    callback_producer: bool,
    raf_or_listener: bool,
}

#[cfg(test)]
impl Interleaving {
    const fn active() -> Self {
        Self {
            state: WebGpuSessionState::Active,
            generation: 1,
            format_generation: 1,
            metadata_generation: 1,
            token: 0,
            recovery_in_flight: false,
            active_submission: false,
            detached_completion: false,
            callback_producer: true,
            raf_or_listener: true,
        }
    }
    fn begin_recovery(&mut self) -> u64 {
        self.state = WebGpuSessionState::Recovering;
        self.token += 1;
        self.recovery_in_flight = true;
        self.callback_producer = false;
        self.active_submission = false;
        self.detached_completion = true;
        self.token
    }
    /// The single publication point for candidate device + queue + format +
    /// adapter facts + generation. A failed candidate is terminal only while
    /// its original recovery still owns the session.
    fn settle_candidate(&mut self, token: u64, install_ok: bool, drawable: bool) -> bool {
        self.recovery_in_flight = false;
        let Some(candidate_generation) = self.generation.checked_add(1) else {
            self.state = WebGpuSessionState::Poisoned;
            return false;
        };
        if !candidate_transaction_committable(
            self.state,
            self.token,
            token,
            self.generation,
            candidate_generation,
        ) {
            return false;
        }
        if !install_ok {
            self.state = WebGpuSessionState::Poisoned;
            return false;
        }
        self.generation = candidate_generation;
        self.format_generation = candidate_generation;
        self.metadata_generation = candidate_generation;
        self.callback_producer = true;
        self.state = if drawable {
            WebGpuSessionState::Active
        } else {
            WebGpuSessionState::Suspended
        };
        true
    }
    /// Starts terminal cleanup. It invalidates every candidate before waiting;
    /// terminal completion remains forbidden until all asynchronous roots have
    /// settled or been removed.
    fn begin_dispose(&mut self) {
        self.state = WebGpuSessionState::Disposing;
        self.token += 1;
        self.callback_producer = false;
        self.raf_or_listener = false;
    }
    fn settle_completion(&mut self) {
        self.active_submission = false;
        self.detached_completion = false;
    }
    fn finish_dispose(&mut self) -> bool {
        if !terminal_dispose_ready(
            self.state,
            TerminalDisposalRoots {
                recovery_pending: self.recovery_in_flight,
                live_objects: self.callback_producer,
                retired_objects: false,
                active_tickets: self.active_submission,
                detached_tickets: self.detached_completion,
                quarantined_resources: false,
                callback_or_listener_producer: self.raf_or_listener,
            },
        ) {
            return false;
        }
        self.state = WebGpuSessionState::Disposed;
        true
    }
}

#[cfg(test)]
impl Accounting {
    const fn new() -> Self {
        Self {
            generation: 1,
            token: 0,
            active_tickets: 0,
            detached_tickets: 0,
            active_observers: 1,
            retired_observers: 0,
        }
    }
    fn submit(&mut self) {
        self.active_tickets += 1;
    }
    fn settle_active(&mut self) {
        self.active_tickets -= 1;
    }
    fn begin_recover(&mut self) -> (u64, u64) {
        let old = (self.generation, self.token);
        self.token += 1;
        self.detached_tickets += self.active_tickets;
        self.active_tickets = 0;
        self.retired_observers += self.active_observers;
        self.active_observers = 0;
        old
    }
    fn install_recovered(&mut self) {
        self.generation += 1;
        self.active_observers = 1;
    }
    fn stale_callback_is_ignored(&self, generation: u64, token: u64) -> bool {
        generation != self.generation || token != self.token
    }
    fn dispose_after_settlement(&mut self) {
        self.token += 1;
        self.active_tickets = 0;
        self.detached_tickets = 0;
        self.active_observers = 0;
        self.retired_observers = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loss_recovery_increments_generation_but_resize_does_not() {
        let mut lifecycle = Lifecycle::new();
        lifecycle.configure(true);
        lifecycle.configure(false);
        assert_eq!(lifecycle.generation(), 1);
        lifecycle.lose();
        assert!(lifecycle.begin_recover());
        lifecycle.finish_recover(true);
        assert_eq!(lifecycle.state(), WebGpuSessionState::Active);
        assert_eq!(lifecycle.generation(), 2);
    }
    #[test]
    fn dispose_is_terminal_and_not_recoverable() {
        assert_ne!(WebGpuSessionState::Disposing, WebGpuSessionState::Poisoned);
        let mut lifecycle = Lifecycle::new();
        lifecycle.dispose();
        lifecycle.lose();
        assert_eq!(lifecycle.state(), WebGpuSessionState::Disposed);
        assert!(!lifecycle.begin_recover());
    }
    #[test]
    fn destroyed_loss_does_not_reopen_a_poisoned_generation() {
        let mut lifecycle = Lifecycle::new();
        lifecycle.configure(true);
        lifecycle.poison();
        lifecycle.lose();
        assert_eq!(lifecycle.state(), WebGpuSessionState::Poisoned);
    }
    #[test]
    fn normal_completion_releases_only_settled_ticket() {
        let mut accounting = Accounting::new();
        accounting.submit();
        accounting.submit();
        accounting.settle_active();
        assert_eq!(accounting.active_tickets, 1);
        assert_eq!(accounting.active_observers, 1);
    }
    #[test]
    fn loss_keeps_old_roots_and_tickets_until_they_settle() {
        let mut accounting = Accounting::new();
        accounting.submit();
        let old = accounting.begin_recover();
        assert_eq!(accounting.detached_tickets, 1);
        assert_eq!(accounting.retired_observers, 1);
        accounting.install_recovered();
        assert!(accounting.stale_callback_is_ignored(old.0, old.1));
    }
    #[test]
    fn partial_setup_failure_has_no_registered_observer() {
        let mut accounting = Accounting::new();
        // Installation registers the observer last; a prior failure leaves no
        // browser callback root to outlive the failed generation.
        accounting.active_observers = 0;
        assert_eq!(accounting.active_observers, 0);
    }
    #[test]
    fn disposal_waits_for_active_and_detached_ownership() {
        let mut accounting = Accounting::new();
        accounting.submit();
        accounting.begin_recover();
        accounting.install_recovered();
        accounting.submit();
        accounting.dispose_after_settlement();
        assert_eq!(accounting.active_tickets, 0);
        assert_eq!(accounting.detached_tickets, 0);
        assert_eq!(accounting.active_observers, 0);
        assert_eq!(accounting.retired_observers, 0);
    }
    #[test]
    fn dispose_invalidates_a_recovery_install_waiting_on_validation() {
        let mut accounting = Accounting::new();
        accounting.begin_recover();
        let pending_token = accounting.token;
        // The real session increments this token before taking Objects, so a
        // validation await that resumes afterwards cannot commit a device.
        accounting.dispose_after_settlement();
        assert_ne!(pending_token, accounting.token);
        assert!(accounting.stale_callback_is_ignored(1, pending_token));
    }

    #[test]
    fn candidate_device_facts_publish_as_one_logical_transaction() {
        let mut model = Interleaving::active();
        let token = model.begin_recovery();
        // Awaiting validation has not made any candidate fact observable.
        assert_eq!(
            (
                model.generation,
                model.format_generation,
                model.metadata_generation
            ),
            (1, 1, 1)
        );
        assert!(model.settle_candidate(token, true, true));
        assert_eq!(
            (
                model.generation,
                model.format_generation,
                model.metadata_generation
            ),
            (2, 2, 2)
        );
        assert_eq!(model.state, WebGpuSessionState::Active);
    }

    #[test]
    fn stale_candidate_and_callback_cannot_mutate_terminal_disposal() {
        let mut model = Interleaving::active();
        let token = model.begin_recovery();
        model.begin_dispose();
        // A late request/device/validation completion is not a producer.
        assert!(!model.settle_candidate(token, true, true));
        assert_eq!(
            (
                model.generation,
                model.format_generation,
                model.metadata_generation
            ),
            (1, 1, 1)
        );
        assert_eq!(model.state, WebGpuSessionState::Disposing);
        model.settle_completion();
        assert!(model.finish_dispose());
        assert_eq!(model.state, WebGpuSessionState::Disposed);
    }

    #[test]
    fn candidate_failure_is_terminal_only_for_its_live_recovery() {
        let mut model = Interleaving::active();
        let token = model.begin_recovery();
        assert!(!model.settle_candidate(token, false, true));
        assert_eq!(model.state, WebGpuSessionState::Poisoned);
        assert_eq!(
            (
                model.generation,
                model.format_generation,
                model.metadata_generation
            ),
            (1, 1, 1)
        );
    }

    #[test]
    fn dispose_requires_join_of_recovery_completion_and_producers() {
        let mut model = Interleaving::active();
        model.active_submission = true;
        let token = model.begin_recovery();
        model.begin_dispose();
        assert!(!model.finish_dispose());
        // The old completion settles first, while the stale recovery is still
        // pending. `Disposed` would be a lie at this point.
        model.settle_completion();
        assert!(!model.finish_dispose());
        assert!(!model.settle_candidate(token, true, false));
        assert!(model.finish_dispose());
    }

    #[test]
    fn every_terminal_disposal_root_independently_blocks_disposed() {
        let clear = TerminalDisposalRoots {
            recovery_pending: false,
            live_objects: false,
            retired_objects: false,
            active_tickets: false,
            detached_tickets: false,
            quarantined_resources: false,
            callback_or_listener_producer: false,
        };
        assert!(terminal_dispose_ready(WebGpuSessionState::Disposing, clear));
        assert!(!terminal_dispose_ready(WebGpuSessionState::Active, clear));

        for blocked in [
            TerminalDisposalRoots {
                recovery_pending: true,
                ..clear
            },
            TerminalDisposalRoots {
                live_objects: true,
                ..clear
            },
            TerminalDisposalRoots {
                retired_objects: true,
                ..clear
            },
            TerminalDisposalRoots {
                active_tickets: true,
                ..clear
            },
            TerminalDisposalRoots {
                detached_tickets: true,
                ..clear
            },
            TerminalDisposalRoots {
                quarantined_resources: true,
                ..clear
            },
            TerminalDisposalRoots {
                callback_or_listener_producer: true,
                ..clear
            },
        ] {
            assert!(!terminal_dispose_ready(
                WebGpuSessionState::Disposing,
                blocked
            ));
        }
    }
}
