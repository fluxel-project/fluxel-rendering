//! A portable, GPU-free backend that satisfies the API v1 contracts.
//!
//! # What it is
//!
//! [`Mock`] implements the API v1 backend traits end to end: device, resources,
//! pipelines, recording, frames, and presentation. It executes nothing. What it
//! does instead is *refuse* everything a real backend must refuse, so the
//! recorder's and the submission plan's own contracts can be exercised on a
//! machine with no GPU, no driver, and no window.
//!
//! Three properties make it a contract fixture rather than a stub:
//!
//! - It runs the real portable validators. `Mock::raster_pipeline` and
//!   `Mock::compute_pipeline` call the same descriptor validation a hardware
//!   backend runs before it lowers anything, so an illegal pipeline is refused
//!   by exactly the code a real device would use.
//! - It writes a journal. Every call a backend receives is appended to
//!   [`Mock::journal`], so a test can prove the recorder really reached the
//!   backend instead of only proving the recorder's own bookkeeping.
//! - It fails on demand. [`Mock::fail_next_record`] and its siblings make the
//!   next backend call return a structured failure, which is the only way to
//!   reach the "a backend failure poisons the recorder" branch of the recording
//!   contract.
//!
//! # What it deliberately is not
//!
//! There is no GPU, no driver, no validation layer, and no concurrency. A test
//! that passes against the mock says nothing about whether a driver accepts the
//! same commands, and no mock result may be recorded as hardware evidence.
//!
//! The mock declares [`BackendKind::Vulkan`] because a capability database must
//! name a backend family and shader acceptance is keyed by one. It does not
//! speak Vulkan and contains nothing Vulkan-specific.
//!
//! [`BackendKind::Vulkan`]: crate::rhi::BackendKind::Vulkan

mod frames;
mod pipelines;
mod platform;
mod recording;
mod resources;
#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex, OnceLock};

use super::binding::BindGroupLayoutDescriptor;
use super::capability::{CapabilityData, EnabledCapabilities};
use super::command::{CommandRecorder, RecorderDescriptor};
use super::diagnostics::{DiagnosticEvent, DiagnosticLog};
use super::format::TextureFormat;
use super::platform::{Device, DeviceIdentity, RhiError, RhiResult};
use super::resource::{Buffer, BufferDescriptor, BufferUsage, Texture, TextureUsage, TextureView};

use platform::{build_capabilities, MockDeviceBackend};

/// One injected backend failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Injected {
    /// The next recorded command fails inside the backend.
    Record,
    /// The next raster-scope close fails inside the backend.
    EndRaster,
    /// The next compute-scope close fails inside the backend.
    EndCompute,
    /// The next recording finalization fails inside the backend.
    Finish,
    /// The next frame acquisition fails inside the backend.
    Acquire,
}

/// Which failures are armed, one shot each.
///
/// One-shot is deliberate: a permanently failing backend would make it
/// impossible to observe that the *first* failure is what poisoned the
/// recorder, which is exactly the rule under test.
#[derive(Default)]
struct Control {
    record: bool,
    end_raster: bool,
    end_compute: bool,
    finish: bool,
    acquire: bool,
}

/// Device-scoped interning of layouts and pipeline interfaces.
///
/// Correctness never reads a hash, so interning compares canonical values
/// themselves and the id only ever identifies an interned value.
#[derive(Default)]
struct Interning {
    next: u64,
    layouts: Vec<(BindGroupLayoutDescriptor, u64)>,
    interfaces: Vec<(Vec<u64>, u64)>,
}

/// Everything the mock's backend halves share.
///
/// A backend receives `&self` for operations that mutate, which is what a real
/// backend needs to keep its bookkeeping behind interior mutability. The mock
/// does the same, for the same reason: the API's shape must not be softened
/// just because this implementation happens to have nothing to protect.
pub(crate) struct MockState {
    journal: Mutex<Vec<String>>,
    control: Mutex<Control>,
    interning: Mutex<Interning>,
    frames: Mutex<u64>,
    /// The device's diagnostic log, shared with every backend half that reports
    /// into it.
    ///
    /// It lives here rather than only in the device backend because a failure
    /// is reported by whichever backend half produced it — a recorder, a
    /// pipeline builder — and all of them must land in the one log the device
    /// drains. A second log would let a backend failure go unobserved.
    diagnostics: Arc<DiagnosticLog>,
}

impl MockState {
    fn new() -> Self {
        Self {
            journal: Mutex::new(Vec::new()),
            control: Mutex::new(Control::default()),
            interning: Mutex::new(Interning::default()),
            frames: Mutex::new(1),
            diagnostics: Arc::new(DiagnosticLog::new(MOCK_DIAGNOSTIC_CAPACITY)),
        }
    }

    /// The device's diagnostic log.
    fn diagnostic_log(&self) -> &Arc<DiagnosticLog> {
        &self.diagnostics
    }

    /// Locks a mutex, recovering from poisoning.
    ///
    /// A panic in one test's backend must not turn every later observation into
    /// a second panic, which would hide the first one.
    fn lock<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Appends one journal entry.
    fn note(&self, entry: impl Into<String>) {
        Self::lock(&self.journal).push(entry.into());
    }

    /// The journal, oldest first.
    fn journal(&self) -> Vec<String> {
        Self::lock(&self.journal).clone()
    }

    /// Arms one failure.
    fn arm(&self, which: Injected) {
        let mut control = Self::lock(&self.control);
        *slot(&mut control, which) = true;
    }

    /// Consumes one armed failure, returning whether it was armed.
    fn consume(&self, which: Injected) -> bool {
        let mut control = Self::lock(&self.control);
        core::mem::take(slot(&mut control, which))
    }

    /// The failure to return for `which`, having consumed an armed one.
    ///
    /// An injected failure is reported into the device's diagnostic log as an
    /// `Error` event, which is what a real backend does with a failure it
    /// detected itself: the portable `RhiError` is the control-flow channel and
    /// the diagnostic is the record. Without this the log would be a channel
    /// nothing ever writes to.
    fn take_failure(&self, which: Injected, operation: &'static str) -> Option<RhiError> {
        if !self.consume(which) {
            return None;
        }
        let error = RhiError::backend_failure(format!(
            "the mock backend was told to fail {operation}"
        ))
        .at(operation);
        self.diagnostics.push(DiagnosticEvent::from_error(&error));
        Some(error)
    }

    /// Mints the next frame serial.
    fn next_frame_serial(&self) -> u64 {
        let mut serial = Self::lock(&self.frames);
        let value = *serial;
        *serial = serial.wrapping_add(1);
        value
    }

    /// Interns a canonical bind group layout descriptor.
    fn intern_layout(&self, descriptor: &BindGroupLayoutDescriptor) -> u64 {
        let mut interning = Self::lock(&self.interning);
        let canonical = descriptor.canonicalized();
        if let Some((_, id)) = interning
            .layouts
            .iter()
            .find(|(existing, _)| *existing == canonical)
        {
            return *id;
        }
        let id = interning.next;
        interning.next = interning.next.wrapping_add(1);
        interning.layouts.push((canonical, id));
        id
    }

    /// Interns a pipeline interface's ordered group-layout identities.
    fn intern_interface(&self, groups: &[u64]) -> u64 {
        let mut interning = Self::lock(&self.interning);
        if let Some((_, id)) = interning
            .interfaces
            .iter()
            .find(|(existing, _)| existing.as_slice() == groups)
        {
            return *id;
        }
        let id = interning.next;
        interning.next = interning.next.wrapping_add(1);
        interning.interfaces.push((groups.to_vec(), id));
        id
    }
}

/// The number of diagnostic events the mock's device log retains.
const MOCK_DIAGNOSTIC_CAPACITY: usize = 64;

/// The failure slot behind one [`Injected`] value.
fn slot(control: &mut Control, which: Injected) -> &mut bool {
    match which {
        Injected::Record => &mut control.record,
        Injected::EndRaster => &mut control.end_raster,
        Injected::EndCompute => &mut control.end_compute,
        Injected::Finish => &mut control.finish,
        Injected::Acquire => &mut control.acquire,
    }
}

/// A complete GPU-free device, with the façades a caller records through.
///
/// Every `Mock` owns a fresh [`DeviceIdentity`], so resources from two mocks
/// are two devices and the cross-device refusals are reachable in a test.
pub(crate) struct Mock {
    device: Device,
    state: Arc<MockState>,
    /// The single presentation lease, created on first use.
    presentation_state: OnceLock<Arc<frames::MockConfiguredPresentationBackend>>,
}

impl Mock {
    /// A mock device with the compute feature enabled.
    pub(crate) fn new() -> Self {
        let state = Arc::new(MockState::new());
        let capabilities = Self::shared_capabilities();
        let backend = Arc::new(MockDeviceBackend::new(capabilities, Arc::clone(&state)));
        let device = Device::new(backend as Arc<dyn super::platform::DeviceBackend>);
        Self {
            device,
            state,
            presentation_state: OnceLock::new(),
        }
    }

    /// The capability database every mock shares.
    ///
    /// Two devices of the same kind have the same capability contract, so
    /// sharing one database is what a real platform does; it is also what makes
    /// `compatibility_id` agree between two mocks.
    fn shared_capabilities() -> Arc<CapabilityData> {
        static SHARED: OnceLock<Arc<CapabilityData>> = OnceLock::new();
        Arc::clone(SHARED.get_or_init(|| {
            build_capabilities().expect("the mock's capability database is internally consistent")
        }))
    }

    /// The device this mock owns.
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// This mock's device identity.
    pub(crate) fn identity(&self) -> DeviceIdentity {
        self.device.identity()
    }

    /// The facts this mock's device enabled.
    pub(crate) fn capabilities(&self) -> &EnabledCapabilities {
        self.device.capabilities()
    }

    /// Everything the backend halves were asked to do, oldest first.
    pub(crate) fn journal(&self) -> Vec<String> {
        self.state.journal()
    }

    /// Whether the journal contains `entry`.
    pub(crate) fn saw(&self, entry: &str) -> bool {
        self.journal().iter().any(|item| item == entry)
    }

    /// Moves every pending diagnostic out of the device.
    pub(crate) fn drain_diagnostics(&self) -> Vec<DiagnosticEvent> {
        let mut events = Vec::new();
        self.device.drain_diagnostics(&mut events);
        events
    }

    /// Makes the next recorded command fail inside the backend.
    pub(crate) fn fail_next_record(&self) {
        self.state.arm(Injected::Record);
    }

    /// Makes the next raster-scope close fail inside the backend.
    pub(crate) fn fail_next_end_raster(&self) {
        self.state.arm(Injected::EndRaster);
    }

    /// Makes the next compute-scope close fail inside the backend.
    pub(crate) fn fail_next_end_compute(&self) {
        self.state.arm(Injected::EndCompute);
    }

    /// Makes the next recording finalization fail inside the backend.
    pub(crate) fn fail_next_finish(&self) {
        self.state.arm(Injected::Finish);
    }

    /// Makes the next frame acquisition fail inside the backend.
    pub(crate) fn fail_next_acquire(&self) {
        self.state.arm(Injected::Acquire);
    }

    /// A fresh recorder over this mock's device.
    ///
    /// It is created through the device façade, so the recorder validates
    /// against the limits the backend reported rather than against limits the
    /// fixture invented.
    pub(crate) fn recorder(&self) -> CommandRecorder {
        self.try_recorder()
            .expect("the mock device creates a recorder")
    }

    /// A fresh recorder, with the device's refusal intact.
    pub(crate) fn try_recorder(&self) -> RhiResult<CommandRecorder> {
        self.device.create_recorder(&RecorderDescriptor::new())
    }

    /// A buffer, created through the device façade.
    ///
    /// It goes through the façade on purpose: a fixture that built resources
    /// behind the device's back would let a test assert something about a
    /// resource the device never accepted.
    pub(crate) fn buffer(&self, size: u64, usage: BufferUsage) -> Buffer {
        self.device
            .create_buffer(&BufferDescriptor::new(size, usage))
            .expect("the mock device creates the requested buffer")
    }

    /// A texture, created through the device façade.
    pub(crate) fn texture(
        &self,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Texture {
        self.device
            .create_texture(&super::resource::TextureDescriptor::new_2d(
                width, height, format, usage,
            ))
            .expect("the mock device creates the requested texture")
    }

    /// A whole 2D view over `texture`.
    pub(crate) fn view(&self, texture: &Texture) -> TextureView {
        let descriptor = super::resource::TextureViewDescriptor::whole(
            texture,
            super::resource::TextureViewDimension::D2,
        )
        .expect("the mock's textures have a viewable aspect");
        self.device
            .create_texture_view(texture, &descriptor)
            .expect("the mock device creates the requested view")
    }
}
