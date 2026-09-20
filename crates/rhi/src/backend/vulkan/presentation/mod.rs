//! Vulkan WSI lowering, kept entirely below Fluxel's presentation seam.
//!
//! A `PresentationTarget` is only an `ObjectId`; its Win32 handles, `VkSurfaceKHR`,
//! `VkSwapchainKHR`, acquired image index and binary semaphores are deliberately
//! private here.  This module is intentionally independent of command lowering:
//! the command spine supplies the acquire-wait and render-finished semaphores at
//! its integration boundary, rather than teaching the public `FrameAttachment`
//! about Vulkan synchronization.

#[cfg(windows)]
pub(crate) mod win32;

#[cfg(target_os = "android")]
pub(crate) mod android;

#[cfg(windows)]
pub(crate) use win32::{
    VulkanFrameAttachment, VulkanPresentSync, VulkanPresentation, VulkanSwapchainRetirement,
    VulkanTargetRegistry,
};

#[cfg(target_os = "android")]
pub(crate) use android::{
    VulkanFrameAttachment, VulkanPresentSync, VulkanPresentation, VulkanSwapchainRetirement,
    VulkanTargetRegistry,
};
