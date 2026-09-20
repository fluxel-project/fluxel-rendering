//! `VK_KHR_android_surface` / `VK_KHR_swapchain` presentation implementation.
//!
//! This is a correctness-first WSI owner.  It deliberately does not expose a
//! swapchain image as `Texture`/`TextureView`: only `VulkanFrameAttachment`
//! carries it and raster lowering may downcast that private backing.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Poll, Waker};
use std::time::Duration;

use ash::vk;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::platform::DeviceLossInfo;
use crate::api::presentation::backend::{
    AcquiredSurfaceFrame, ConfiguredPresentationBackend, FrameAttachmentBackend,
    PresentationBackend,
};
use crate::api::presentation::{
    AcquireError, AcquireErrorKind, AcquiredFrameId, Extent2d, PresentMode, PresentReceiptId,
    PresentState, PresentationConfiguration, PresentationExtent, PresentationExtentControl,
    PresentationTarget, PresentationTargetCapabilities,
};
use crate::backend::vulkan::platform::device::VulkanShared;

#[link(name = "android")]
unsafe extern "C" {
    fn ANativeWindow_acquire(window: *mut core::ffi::c_void);
    fn ANativeWindow_release(window: *mut core::ffi::c_void);
}

/// Command-spine supplied synchronization for presenting one acquired image.
///
/// `acquire_wait` is waited by the first submission which writes the frame;
/// `render_finished` is signalled by its final submission and waited by
/// `vkQueuePresentKHR`.  Both are binary semaphores owned by the accepted-work
/// retention domain, never by the portable frame handle.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VulkanPresentSync {
    pub(crate) acquire_wait: vk::Semaphore,
    pub(crate) render_finished: vk::Semaphore,
}

pub(crate) struct AndroidTarget {
    // Raw Android handles are not `Send` on all Rust targets. Store their opaque
    // numeric representation under the registry mutex and reconstruct them at
    // the Vulkan FFI boundary, exactly where they become native again.
    window: usize,
    // The registry acquires exactly one ANativeWindow reference.  Surfaces and
    // configured presentations retain this Arc, so unregistering a target can
    // make it terminal without invalidating a Vulkan surface still being
    // dismantled on the callback thread.
    alive: AtomicBool,
}

impl AndroidTarget {
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
}

impl Drop for AndroidTarget {
    fn drop(&mut self) {
        // Paired with `ANativeWindow_acquire` in `register`.  The registry and
        // all Vulkan surface owners use Arc ownership, so this is the one and
        // only release for the reference owned by Fluxel.
        unsafe { ANativeWindow_release(self.window as *mut core::ffi::c_void) };
    }
}

pub(crate) struct VulkanTargetRegistry {
    targets: Mutex<HashMap<ObjectId, Arc<AndroidTarget>>>,
}

/// Host-lifecycle ownership for one registry entry. This remains wholly below
/// the portable presentation API: it exists so Android's destroy callback can
/// release the native window reference at the precise framework boundary.
pub(crate) struct AndroidTargetRegistration {
    registry: Arc<VulkanTargetRegistry>,
    id: ObjectId,
}

impl Drop for AndroidTargetRegistration {
    fn drop(&mut self) {
        self.registry.unregister(self.id);
    }
}

impl VulkanTargetRegistry {
    pub(crate) fn new() -> Self {
        Self {
            targets: Mutex::new(HashMap::new()),
        }
    }

    /// Registers one host-owned Android native window.  This is backend-private:
    /// callers receive only the portable target identity.
    pub(crate) fn register(&self, window: *mut core::ffi::c_void) -> RhiResult<PresentationTarget> {
        if window.is_null() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cannot register a null ANativeWindow",
            ));
        }
        // NativeActivity only lends its callback pointer.  Hold our own native
        // reference until target unregister and all dependent surfaces retire.
        unsafe { ANativeWindow_acquire(window) };
        let id = ObjectId::next();
        self.targets
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(
                id,
                Arc::new(AndroidTarget {
                    window: window as usize,
                    alive: AtomicBool::new(true),
                }),
            );
        Ok(PresentationTarget::new(id))
    }

    /// Makes a target terminal and releases the registry's reference. Existing
    /// surface owners retain the acquired native reference only long enough to
    /// destroy their Vulkan objects safely.
    pub(crate) fn unregister(&self, id: ObjectId) {
        if let Some(target) = self
            .targets
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&id)
        {
            target.alive.store(false, Ordering::Release);
        }
    }

    pub(crate) fn registration(self: &Arc<Self>, id: ObjectId) -> AndroidTargetRegistration {
        AndroidTargetRegistration {
            registry: Arc::clone(self),
            id,
        }
    }

    fn target(&self, id: ObjectId) -> RhiResult<Arc<AndroidTarget>> {
        self.targets
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&id)
            .cloned()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::TargetLost,
                    "the Android target is not registered with this Vulkan provider",
                )
            })
    }

    pub(crate) fn create_surface(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
        id: ObjectId,
    ) -> RhiResult<(vk::SurfaceKHR, Arc<AndroidTarget>)> {
        let target = self.target(id)?;
        if !target.is_alive() {
            return Err(RhiError::new(
                RhiErrorKind::TargetLost,
                "the Android native window was destroyed",
            ));
        }
        let loader = ash::khr::android_surface::Instance::new(entry, instance);
        let create = vk::AndroidSurfaceCreateInfoKHR::default()
            .window(target.window as *mut core::ffi::c_void);
        unsafe { loader.create_android_surface(&create, None) }
            .map(|surface| (surface, target))
            .map_err(|result| native_error(result, "vkCreateAndroidSurfaceKHR"))
    }
}

struct SurfaceOwner {
    loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    _target: Arc<AndroidTarget>,
}

impl Drop for SurfaceOwner {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.surface, None) };
    }
}

impl SurfaceOwner {
    fn is_target_alive(&self) -> bool {
        self._target.is_alive()
    }
}

struct SwapchainOwner {
    // Strong ownership is intentional: this generation must retire its native
    // children into VulkanShared before the final VkDevice teardown can begin.
    shared: Arc<VulkanShared>,
    loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    // A present wait semaphore cannot be destroyed when vkQueuePresentKHR
    // returns: presentation may still be waiting on it.  The same image coming
    // back through acquire proves that wait is complete, which is the earliest
    // no-stall retirement point available to this baseline.
    retired_semaphores: Mutex<Vec<Vec<vk::Semaphore>>>,
    _surface: Arc<SurfaceOwner>,
}

/// Native WSI objects whose safe lifetime cannot be proven by queue idle.
///
/// Without present fences (`VK_EXT_swapchain_maintenance1`) or present IDs,
/// only reacquiring the same image proves that an individual present wait has
/// released its semaphore. Old generations are never reacquired, so this
/// record is intentionally retained by `VulkanShared` until device teardown.
pub(crate) struct VulkanSwapchainRetirement {
    loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    semaphores: Vec<vk::Semaphore>,
    _surface: Arc<SurfaceOwner>,
}

impl VulkanSwapchainRetirement {
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        for semaphore in self.semaphores {
            if semaphore != vk::Semaphore::null() {
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
        }
        unsafe { self.loader.destroy_swapchain(self.swapchain, None) };
    }
}

impl SwapchainOwner {
    fn retire_after_present(&self, image: u32, semaphores: [vk::Semaphore; 2]) {
        self.retired_semaphores
            .lock()
            .unwrap_or_else(|p| p.into_inner())[image as usize]
            .extend(semaphores);
    }

    fn reclaim_image_semaphores(&self, shared: &VulkanShared, image: u32) {
        let retired = std::mem::take(
            &mut self
                .retired_semaphores
                .lock()
                .unwrap_or_else(|p| p.into_inner())[image as usize],
        );
        for semaphore in retired {
            if semaphore != vk::Semaphore::null() {
                unsafe { shared.device.destroy_semaphore(semaphore, None) };
            }
        }
    }
}

impl Drop for SwapchainOwner {
    fn drop(&mut self) {
        let semaphores = self
            .retired_semaphores
            .get_mut()
            .unwrap_or_else(|p| p.into_inner())
            .iter_mut()
            .flat_map(|semaphores| semaphores.drain(..))
            .collect();
        let retirement = VulkanSwapchainRetirement {
            loader: self.loader.clone(),
            swapchain: self.swapchain,
            semaphores,
            _surface: Arc::clone(&self._surface),
        };
        self.shared.retire_swapchain(retirement);
    }
}

struct SwapchainState {
    current: Arc<SwapchainOwner>,
    acquired: Option<AcquiredImage>,
}

struct AcquiredImage {
    image_index: u32,
    swapchain: Arc<SwapchainOwner>,
    acquire_wait: vk::Semaphore,
    return_state: Arc<AtomicU8>,
}

struct AcquireWake {
    active: std::sync::atomic::AtomicBool,
    waker: Mutex<Option<std::task::Waker>>,
}

/// Device presentation facet. `entry`/`instance` outlive this object; callers
/// construct it only after enabling `VK_KHR_surface` and `VK_KHR_android_surface`.
pub(crate) struct VulkanPresentation {
    shared: Arc<VulkanShared>,
    surface: ash::khr::surface::Instance,
    android_surface: ash::khr::android_surface::Instance,
    swapchain: ash::khr::swapchain::Device,
    targets: Arc<VulkanTargetRegistry>,
    leased: Arc<Mutex<HashSet<ObjectId>>>,
    // TODO(vulkan-performance): retire observed terminal receipts once the
    // backend has a bounded observation epoch. Correctness currently prefers
    // stable answers for the full DeviceIdentity lifetime over eager pruning.
    presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
}

impl VulkanPresentation {
    /// The integration boundary deliberately takes raw Android handles. Host glue
    /// owns window creation; no Android/window type leaks into Fluxel's API.
    pub(crate) fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        shared: Arc<VulkanShared>,
        targets: Arc<VulkanTargetRegistry>,
    ) -> Self {
        Self {
            surface: ash::khr::surface::Instance::new(entry, instance),
            android_surface: ash::khr::android_surface::Instance::new(entry, instance),
            swapchain: ash::khr::swapchain::Device::new(instance, &shared.device),
            shared,
            targets,
            leased: Arc::new(Mutex::new(HashSet::new())),
            presents: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn create_surface(&self, target: ObjectId) -> RhiResult<(vk::SurfaceKHR, Arc<AndroidTarget>)> {
        let target = self.targets.target(target)?;
        if !target.is_alive() {
            return Err(RhiError::new(
                RhiErrorKind::TargetLost,
                "the Android native window was destroyed",
            ));
        }
        let create = vk::AndroidSurfaceCreateInfoKHR::default()
            .window(target.window as *mut core::ffi::c_void);
        unsafe { self.android_surface.create_android_surface(&create, None) }
            .map(|surface| (surface, target))
            .map_err(|result| native_error(result, "vkCreateAndroidSurfaceKHR"))
    }

    fn facts(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        if self.shared.loss_info().is_some() {
            return Err(lost_error());
        }
        let (surface, _target) = self.create_surface(target)?;
        let answer = surface_capabilities(
            &self.surface,
            self.shared.physical_device,
            self.shared.graphics_family,
            surface,
        );
        unsafe { self.surface.destroy_surface(surface, None) };
        answer.map_err(|error| observe(&self.shared, error, "Vulkan surface capability query"))
    }
}

impl PresentationBackend for VulkanPresentation {
    fn capabilities(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        self.facts(target)
    }

    fn configure(
        &self,
        device: DeviceIdentity,
        target: ObjectId,
        config: &PresentationConfiguration,
    ) -> RhiResult<Box<dyn ConfiguredPresentationBackend>> {
        if self.shared.loss_info().is_some() {
            return Err(lost_error());
        }
        let mut leased = self.leased.lock().unwrap_or_else(|p| p.into_inner());
        if !leased.insert(target) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "this presentation target already has an active configuration lease",
            ));
        }
        let surface = match self.create_surface(target) {
            Ok((surface, target_ref)) => Arc::new(SurfaceOwner {
                loader: self.surface.clone(),
                surface,
                _target: target_ref,
            }),
            Err(error) => {
                leased.remove(&target);
                return Err(error);
            }
        };
        let current = create_swapchain(
            &self.shared,
            &self.surface,
            &self.swapchain,
            Arc::clone(&surface),
            config,
        )
        .map_err(|error| observe(&self.shared, error, "vkCreateSwapchainKHR"));
        match current {
            Ok(current) => Ok(Box::new(VulkanConfiguredPresentation {
                shared: Arc::clone(&self.shared),
                surface_loader: self.surface.clone(),
                swapchain_loader: self.swapchain.clone(),
                surface,
                device,
                target,
                state: Arc::new(Mutex::new(SwapchainState {
                    current,
                    acquired: None,
                })),
                serial: AtomicU64::new(1),
                leased: Arc::clone(&self.leased),
                presents: Arc::clone(&self.presents),
                acquire_wake: Arc::new(AcquireWake {
                    active: std::sync::atomic::AtomicBool::new(false),
                    waker: Mutex::new(None),
                }),
            })),
            Err(error) => {
                // `SurfaceOwner` destroys the just-created surface.
                leased.remove(&target);
                Err(error)
            }
        }
    }

    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        self.presents
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&receipt)
            .cloned()
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::InvalidUsage, "unknown Vulkan present receipt")
            })
    }

    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        _: &std::task::Waker,
    ) -> RhiResult<PresentState> {
        // `vkQueuePresentKHR` is called only once its final queue submission has
        // been accepted, and this baseline records its terminal response directly.
        self.present_state(receipt)
    }
}

/// Backend-private acquired swapchain image. Raster lowering reads `image()` and
/// `sync()` after downcast; callers never observe either Vulkan object.
pub(crate) struct VulkanFrameAttachment {
    image: vk::Image,
    image_index: u32,
    sync: VulkanPresentSync,
    state: Arc<Mutex<SwapchainState>>,
    swapchain: Arc<SwapchainOwner>,
    swapchain_loader: ash::khr::swapchain::Device,
    queue: vk::Queue,
    shared: Arc<VulkanShared>,
    presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
    return_state: Arc<AtomicU8>,
}

impl VulkanFrameAttachment {
    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }
    pub(crate) fn sync(&self) -> VulkanPresentSync {
        self.sync
    }
}

impl FrameAttachmentBackend for VulkanFrameAttachment {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn present(&self, receipt: PresentReceiptId) {
        let state = if let Some(info) = self.shared.loss_info() {
            PresentState::DeviceLost(info)
        } else {
            let chains = [self.swapchain.swapchain];
            let indices = [self.image_index];
            let waits = [self.sync.render_finished];
            let info = vk::PresentInfoKHR::default()
                .wait_semaphores(&waits)
                .swapchains(&chains)
                .image_indices(&indices);
            let present_result = {
                let _queue = self.shared.queue_guard();
                unsafe { self.swapchain_loader.queue_present(self.queue, &info) }
            };
            match present_result {
                Ok(true) => PresentState::Outdated,
                Ok(false) => PresentState::Accepted,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => PresentState::Outdated,
                Err(vk::Result::ERROR_SURFACE_LOST_KHR) => PresentState::TargetLost,
                Err(result) => match observe(
                    &self.shared,
                    native_error(result, "vkQueuePresentKHR"),
                    "vkQueuePresentKHR",
                )
                .kind()
                {
                    RhiErrorKind::DeviceLost => self
                        .shared
                        .loss_info()
                        .map(PresentState::DeviceLost)
                        .unwrap_or_else(|| {
                            PresentState::Failed(crate::api::presentation::PresentFailure::new(
                                "Vulkan device loss could not be diagnosed",
                            ))
                        }),
                    _ => PresentState::Failed(crate::api::presentation::PresentFailure::new(
                        format!("vkQueuePresentKHR failed: {result:?}"),
                    )),
                },
            }
        };
        self.presents
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(receipt, state);
        self.return_state.store(1, Ordering::Release);
        self.swapchain.retire_after_present(
            self.image_index,
            [self.sync.acquire_wait, self.sync.render_finished],
        );
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .acquired = None;
    }

    fn terminate_present(&self, receipt: PresentReceiptId, state: PresentState) {
        self.presents
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(receipt, state);
        // No native present can be issued after terminal device loss. Retain
        // both semaphores with the swapchain generation until teardown: a
        // prior acquire signal or accepted queue wait may still own them.
        self.return_state.store(1, Ordering::Release);
        self.swapchain.retire_after_present(
            self.image_index,
            [self.sync.acquire_wait, self.sync.render_finished],
        );
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .acquired = None;
    }
}

impl Drop for VulkanFrameAttachment {
    fn drop(&mut self) {
        match self.return_state.load(Ordering::Acquire) {
            0 => unsafe {
                self.shared
                    .device
                    .destroy_semaphore(self.sync.acquire_wait, None);
                self.shared
                    .device
                    .destroy_semaphore(self.sync.render_finished, None);
            },
            2 => unsafe {
                self.shared
                    .device
                    .destroy_semaphore(self.sync.render_finished, None)
            },
            _ => {}
        }
    }
}

struct VulkanConfiguredPresentation {
    shared: Arc<VulkanShared>,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: Arc<SurfaceOwner>,
    device: DeviceIdentity,
    target: ObjectId,
    state: Arc<Mutex<SwapchainState>>,
    serial: AtomicU64,
    leased: Arc<Mutex<HashSet<ObjectId>>>,
    presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
    acquire_wake: Arc<AcquireWake>,
}

impl ConfiguredPresentationBackend for VulkanConfiguredPresentation {
    fn capabilities(&self) -> RhiResult<PresentationTargetCapabilities> {
        if self.shared.loss_info().is_some() {
            return Err(lost_error());
        }
        if !self.surface.is_target_alive() {
            return Err(RhiError::new(
                RhiErrorKind::TargetLost,
                "the Android native window was destroyed",
            ));
        }
        surface_capabilities(
            &self.surface_loader,
            self.shared.physical_device,
            self.shared.graphics_family,
            self.surface.surface,
        )
        .map_err(|error| observe(&self.shared, error, "Vulkan surface capability query"))
    }
    fn reconfigure_or_register_waker(
        &self,
        config: &PresentationConfiguration,
        _: &Waker,
    ) -> Poll<RhiResult<()>> {
        Poll::Ready((|| {
            if self.shared.loss_info().is_some() {
                return Err(lost_error());
            }
            if !self.surface.is_target_alive() {
                return Err(RhiError::new(
                    RhiErrorKind::TargetLost,
                    "the Android native window was destroyed",
                ));
            }
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.acquired.is_some() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "cannot recreate a Vulkan swapchain while a frame is acquired",
                ));
            }
            let old = Arc::clone(&state.current);
            let new = create_swapchain_with_old(
                &self.shared,
                &self.surface_loader,
                &self.swapchain_loader,
                Arc::clone(&self.surface),
                config,
                old.swapchain,
            )
            .map_err(|error| observe(&self.shared, error, "vkCreateSwapchainKHR"))?;
            state.current = new;
            Ok(())
        })())
    }
    fn try_acquire(
        &self,
        device: DeviceIdentity,
    ) -> Result<Option<AcquiredSurfaceFrame>, AcquireError> {
        if self.shared.loss_info().is_some() {
            return Err(AcquireError::new(
                AcquireErrorKind::DeviceLost,
                "the Vulkan device was lost; this presentation lease is terminal",
            ));
        }
        if !self.surface.is_target_alive() {
            return Err(AcquireError::new(
                AcquireErrorKind::TargetLost,
                "the Android native window was destroyed",
            ));
        }
        if device != self.device {
            return Err(AcquireError::new(
                AcquireErrorKind::DeviceLost,
                "the presentation lease belongs to another device",
            ));
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.acquired.is_some() {
            return Err(AcquireError::new(
                AcquireErrorKind::FrameOutstanding,
                "a Vulkan swapchain image remains acquired",
            ));
        }
        let current = Arc::clone(&state.current);
        if current.extent.width == 0 || current.extent.height == 0 {
            return Err(AcquireError::new(
                AcquireErrorKind::ZeroSizeOrSuspended,
                "the Vulkan target has a zero drawable extent",
            ));
        }
        let semaphore = unsafe {
            self.shared
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        }
        .map_err(|result| acquire_native(&self.shared, result, "vkCreateSemaphore for acquire"))?;
        let render_finished = match unsafe {
            self.shared
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        } {
            Ok(semaphore) => semaphore,
            Err(result) => {
                unsafe { self.shared.device.destroy_semaphore(semaphore, None) };
                return Err(acquire_native(
                    &self.shared,
                    result,
                    "vkCreateSemaphore for present",
                ));
            }
        };
        match unsafe {
            self.swapchain_loader.acquire_next_image(
                current.swapchain,
                0,
                semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((index, _suboptimal)) => {
                current.reclaim_image_semaphores(&self.shared, index);
                let serial = self.serial.fetch_add(1, Ordering::Relaxed);
                let return_state = Arc::new(AtomicU8::new(0));
                state.acquired = Some(AcquiredImage {
                    image_index: index,
                    swapchain: Arc::clone(&current),
                    acquire_wait: semaphore,
                    return_state: Arc::clone(&return_state),
                });
                Ok(Some(AcquiredSurfaceFrame {
                    serial,
                    extent: Extent2d {
                        width: current.extent.width,
                        height: current.extent.height,
                    },
                    attachment: Box::new(VulkanFrameAttachment {
                        image: current.images[index as usize],
                        image_index: index,
                        sync: VulkanPresentSync {
                            acquire_wait: semaphore,
                            render_finished,
                        },
                        state: Arc::clone(&self.state),
                        swapchain: current,
                        swapchain_loader: self.swapchain_loader.clone(),
                        queue: self.shared.graphics_queue,
                        shared: Arc::clone(&self.shared),
                        presents: Arc::clone(&self.presents),
                        return_state,
                    }),
                }))
            }
            Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
                unsafe { self.shared.device.destroy_semaphore(semaphore, None) };
                unsafe { self.shared.device.destroy_semaphore(render_finished, None) };
                Ok(None)
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                unsafe { self.shared.device.destroy_semaphore(semaphore, None) };
                unsafe { self.shared.device.destroy_semaphore(render_finished, None) };
                Err(AcquireError::new(
                    AcquireErrorKind::Outdated,
                    "Vulkan swapchain is out of date",
                ))
            }
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                unsafe { self.shared.device.destroy_semaphore(semaphore, None) };
                unsafe { self.shared.device.destroy_semaphore(render_finished, None) };
                Err(AcquireError::new(
                    AcquireErrorKind::TargetLost,
                    "Vulkan surface was lost",
                ))
            }
            Err(result) => {
                unsafe { self.shared.device.destroy_semaphore(semaphore, None) };
                unsafe { self.shared.device.destroy_semaphore(render_finished, None) };
                Err(acquire_native(
                    &self.shared,
                    result,
                    "vkAcquireNextImageKHR",
                ))
            }
        }
    }
    fn acquire_or_register_waker(
        &self,
        device: DeviceIdentity,
        waker: &std::task::Waker,
    ) -> std::task::Poll<Result<AcquiredSurfaceFrame, AcquireError>> {
        match self.try_acquire(device) {
            Ok(Some(frame)) => {
                self.shared.unregister_loss_waker(self.target.as_u64());
                std::task::Poll::Ready(Ok(frame))
            }
            Ok(None) => {
                // WSI offers no portable event object. A tiny backend worker is
                // therefore the correctness baseline: it re-wakes this future
                // while the surface owns all images, without asking callers to
                // poll `Device` or spinning an executor thread. The worker holds
                // only a Weak wake cell, so dropping the lease stops it.
                *self
                    .acquire_wake
                    .waker
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()) = Some(waker.clone());
                if let Err(info) = self.shared.register_loss_waker(self.target.as_u64(), waker) {
                    return std::task::Poll::Ready(Err(AcquireError::new(
                        AcquireErrorKind::DeviceLost,
                        info.message().to_owned(),
                    )));
                }
                if !self.acquire_wake.active.swap(true, Ordering::AcqRel) {
                    let weak = Arc::downgrade(&self.acquire_wake);
                    let shared = Arc::clone(&self.shared);
                    let loss_slot = self.target.as_u64();
                    if std::thread::Builder::new()
                        .name("fluxel-vulkan-acquire".into())
                        .spawn(move || wake_acquire_until_repolled(weak, shared, loss_slot))
                        .is_err()
                    {
                        self.acquire_wake.active.store(false, Ordering::Release);
                        return std::task::Poll::Ready(Err(AcquireError::new(
                            AcquireErrorKind::OutOfMemory,
                            "could not start the Vulkan acquire wake worker",
                        )));
                    }
                }
                std::task::Poll::Pending
            }
            Err(error) => {
                self.shared.unregister_loss_waker(self.target.as_u64());
                std::task::Poll::Ready(Err(error))
            }
        }
    }
    fn abandon(&self, _: AcquiredFrameId) -> RhiResult<()> {
        let acquired = self
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .acquired
            .take();
        if let Some(acquired) = acquired {
            let chains = [acquired.swapchain.swapchain];
            let indices = [acquired.image_index];
            let waits = [acquired.acquire_wait];
            let info = vk::PresentInfoKHR::default()
                .wait_semaphores(&waits)
                .swapchains(&chains)
                .image_indices(&indices);
            let result = {
                let _queue = self.shared.queue_guard();
                unsafe {
                    self.swapchain_loader
                        .queue_present(self.shared.graphics_queue, &info)
                }
            };
            acquired.return_state.store(2, Ordering::Release);
            acquired.swapchain.retire_after_present(
                acquired.image_index,
                [acquired.acquire_wait, vk::Semaphore::null()],
            );
            result.map_err(|result| {
                observe(
                    &self.shared,
                    native_error(result, "vkQueuePresentKHR for abandon"),
                    "vkQueuePresentKHR for abandon",
                )
            })?;
        }
        Ok(())
    }
    fn abandon_no_throw(&self, _: AcquiredFrameId) {
        // `Drop` cannot report an error. The API contract permits the backend to
        // mark recovery-needed here; a failed no-op present is therefore left to
        // swapchain teardown instead of panicking during unwinding.
        let _ = self.abandon(AcquiredFrameId::new(self.device, 0));
    }
    fn release(&self) {
        self.abandon_no_throw(AcquiredFrameId::new(self.device, 0));
        self.leased
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.target);
    }
}

impl Drop for VulkanConfiguredPresentation {
    fn drop(&mut self) {
        self.acquire_wake.active.store(false, Ordering::Release);
        self.shared.unregister_loss_waker(self.target.as_u64());
    }
}

fn wake_acquire_until_repolled(wake: Weak<AcquireWake>, shared: Arc<VulkanShared>, loss_slot: u64) {
    loop {
        std::thread::sleep(Duration::from_millis(8));
        let Some(wake) = wake.upgrade() else { return };
        if !wake.active.load(Ordering::Acquire) {
            return;
        }
        let waker = wake.waker.lock().unwrap_or_else(|p| p.into_inner()).take();
        // An executor may poll synchronously from `wake`. Publish this worker
        // as idle first, so a still-not-ready poll can arm its successor rather
        // than observe a worker that is already about to exit.
        wake.active.store(false, Ordering::Release);
        shared.unregister_loss_waker(loss_slot);
        if let Some(waker) = waker {
            waker.wake();
        }
        return;
    }
}

fn create_swapchain(
    shared: &Arc<VulkanShared>,
    surface: &ash::khr::surface::Instance,
    loader: &ash::khr::swapchain::Device,
    target: Arc<SurfaceOwner>,
    config: &PresentationConfiguration,
) -> Result<Arc<SwapchainOwner>, RhiError> {
    create_swapchain_with_old(
        shared,
        surface,
        loader,
        target,
        config,
        vk::SwapchainKHR::null(),
    )
}
fn create_swapchain_with_old(
    shared: &Arc<VulkanShared>,
    surface: &ash::khr::surface::Instance,
    loader: &ash::khr::swapchain::Device,
    target: Arc<SurfaceOwner>,
    config: &PresentationConfiguration,
    old: vk::SwapchainKHR,
) -> Result<Arc<SwapchainOwner>, RhiError> {
    let caps = unsafe {
        surface.get_physical_device_surface_capabilities(shared.physical_device, target.surface)
    }
    .map_err(|r| native_error(r, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?;
    let queue_supported = unsafe {
        surface.get_physical_device_surface_support(
            shared.physical_device,
            shared.graphics_family,
            target.surface,
        )
    }
    .map_err(|r| native_error(r, "vkGetPhysicalDeviceSurfaceSupportKHR"))?;
    if !queue_supported {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the selected Vulkan graphics queue cannot present to this Android surface",
        ));
    }
    if !caps
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the Vulkan surface does not allow swapchain images as color attachments",
        ));
    }
    let formats = unsafe {
        surface.get_physical_device_surface_formats(shared.physical_device, target.surface)
    }
    .map_err(|r| native_error(r, "vkGetPhysicalDeviceSurfaceFormatsKHR"))?;
    let format = choose_format(&formats, config.format()).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "requested presentation format is not offered by this Vulkan surface",
        )
    })?;
    let extent = choose_extent(caps, config.extent())?;
    let mode = choose_mode(
        unsafe {
            surface
                .get_physical_device_surface_present_modes(shared.physical_device, target.surface)
        }
        .map_err(|r| native_error(r, "vkGetPhysicalDeviceSurfacePresentModesKHR"))?,
        config.present_mode(),
    )?;
    let count = caps
        .min_image_count
        .saturating_add(1)
        .min(if caps.max_image_count == 0 {
            u32::MAX
        } else {
            caps.max_image_count
        });
    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(target.surface)
        .min_image_count(count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(choose_composite_alpha(caps.supported_composite_alpha)?)
        .present_mode(mode)
        .clipped(true)
        .old_swapchain(old);
    let swapchain = unsafe { loader.create_swapchain(&info, None) }
        .map_err(|r| native_error(r, "vkCreateSwapchainKHR"))?;
    let images = match unsafe { loader.get_swapchain_images(swapchain) } {
        Ok(images) => images,
        Err(result) => {
            unsafe { loader.destroy_swapchain(swapchain, None) };
            return Err(native_error(result, "vkGetSwapchainImagesKHR"));
        }
    };
    Ok(Arc::new(SwapchainOwner {
        shared: Arc::clone(shared),
        loader: loader.clone(),
        swapchain,
        extent,
        retired_semaphores: Mutex::new((0..images.len()).map(|_| Vec::new()).collect()),
        images,
        _surface: target,
    }))
}

fn surface_capabilities(
    loader: &ash::khr::surface::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
    surface: vk::SurfaceKHR,
) -> RhiResult<PresentationTargetCapabilities> {
    let queue_supported =
        unsafe { loader.get_physical_device_surface_support(physical, queue_family, surface) }
            .map_err(|r| native_error(r, "vkGetPhysicalDeviceSurfaceSupportKHR"))?;
    if !queue_supported {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the selected Vulkan graphics queue cannot present to this surface",
        ));
    }
    let formats = unsafe { loader.get_physical_device_surface_formats(physical, surface) }
        .map_err(|r| native_error(r, "vkGetPhysicalDeviceSurfaceFormatsKHR"))?;
    let modes = unsafe { loader.get_physical_device_surface_present_modes(physical, surface) }
        .map_err(|r| native_error(r, "vkGetPhysicalDeviceSurfacePresentModesKHR"))?;
    let caps = unsafe { loader.get_physical_device_surface_capabilities(physical, surface) }
        .map_err(|r| native_error(r, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?;
    let formats = if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
        ]
    } else {
        formats
            .into_iter()
            .filter_map(|f| portable_format(f.format))
            .collect()
    };
    let modes = modes.into_iter().filter_map(portable_mode).collect();
    let extent = if caps.current_extent.width == u32::MAX {
        PresentationExtentControl::Configurable {
            min: Extent2d {
                width: caps.min_image_extent.width,
                height: caps.min_image_extent.height,
            },
            max: Extent2d {
                width: caps.max_image_extent.width,
                height: caps.max_image_extent.height,
            },
        }
    } else {
        PresentationExtentControl::HostManaged {
            current: Some(Extent2d {
                width: caps.current_extent.width,
                height: caps.current_extent.height,
            }),
        }
    };
    Ok(PresentationTargetCapabilities::new(formats, modes, extent))
}

fn choose_format(
    formats: &[vk::SurfaceFormatKHR],
    wanted: TextureFormat,
) -> Option<vk::SurfaceFormatKHR> {
    if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        return native_format(wanted).map(|format| vk::SurfaceFormatKHR {
            format,
            color_space: formats[0].color_space,
        });
    }
    formats
        .iter()
        .copied()
        .find(|value| portable_format(value.format) == Some(wanted))
}

fn choose_composite_alpha(
    supported: vk::CompositeAlphaFlagsKHR,
) -> RhiResult<vk::CompositeAlphaFlagsKHR> {
    [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ]
    .into_iter()
    .find(|mode| supported.contains(*mode))
    .ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "the Vulkan surface exposes no supported composite-alpha mode",
        )
    })
}

fn native_format(format: TextureFormat) -> Option<vk::Format> {
    match format {
        TextureFormat::Bgra8Unorm => Some(vk::Format::B8G8R8A8_UNORM),
        TextureFormat::Bgra8UnormSrgb => Some(vk::Format::B8G8R8A8_SRGB),
        TextureFormat::Rgba8Unorm => Some(vk::Format::R8G8B8A8_UNORM),
        TextureFormat::Rgba8UnormSrgb => Some(vk::Format::R8G8B8A8_SRGB),
        _ => None,
    }
}
fn choose_extent(
    caps: vk::SurfaceCapabilitiesKHR,
    requested: PresentationExtent,
) -> RhiResult<vk::Extent2D> {
    if caps.current_extent.width != u32::MAX {
        return Ok(caps.current_extent);
    }
    let extent = match requested {
        PresentationExtent::Exact(e) => vk::Extent2D {
            width: e.width,
            height: e.height,
        },
        PresentationExtent::HostManaged => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a Vulkan target with configurable extent requires PresentationExtent::Exact",
            ));
        }
    };
    Ok(vk::Extent2D {
        width: extent
            .width
            .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
        height: extent
            .height
            .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
    })
}
fn choose_mode(
    modes: Vec<vk::PresentModeKHR>,
    requested: PresentMode,
) -> RhiResult<vk::PresentModeKHR> {
    let desired = match requested {
        PresentMode::Automatic | PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
    };
    if modes.contains(&desired) {
        Ok(desired)
    } else {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "requested Vulkan present mode is not offered by this surface",
        ))
    }
}
fn portable_format(format: vk::Format) -> Option<TextureFormat> {
    match format {
        vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
        vk::Format::B8G8R8A8_SRGB => Some(TextureFormat::Bgra8UnormSrgb),
        vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
        vk::Format::R8G8B8A8_SRGB => Some(TextureFormat::Rgba8UnormSrgb),
        _ => None,
    }
}
fn portable_mode(mode: vk::PresentModeKHR) -> Option<PresentMode> {
    match mode {
        vk::PresentModeKHR::FIFO => Some(PresentMode::Fifo),
        vk::PresentModeKHR::MAILBOX => Some(PresentMode::Mailbox),
        vk::PresentModeKHR::IMMEDIATE => Some(PresentMode::Immediate),
        _ => None,
    }
}
fn native_error(result: vk::Result, op: &'static str) -> RhiError {
    crate::backend::vulkan::ffi::to_rhi(result, op)
}
fn observe(shared: &VulkanShared, error: RhiError, _: &'static str) -> RhiError {
    if error.kind() == RhiErrorKind::DeviceLost {
        shared.mark_lost(DeviceLossInfo::new(error.message().to_owned()));
    }
    error
}
fn acquire_native(shared: &VulkanShared, result: vk::Result, op: &'static str) -> AcquireError {
    let error = observe(shared, native_error(result, op), op);
    let kind = match error.kind() {
        RhiErrorKind::DeviceLost => AcquireErrorKind::DeviceLost,
        RhiErrorKind::OutOfMemory => AcquireErrorKind::OutOfMemory,
        _ => AcquireErrorKind::TargetLost,
    };
    AcquireError::new(kind, error.message().to_owned())
}
fn lost_error() -> RhiError {
    RhiError::new(
        RhiErrorKind::DeviceLost,
        "the Vulkan device was lost; this presentation lease is terminal",
    )
}
