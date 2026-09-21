//! Vulkan provider and device ownership.
//!
//! Instance discovery, physical-device selection, logical-device creation, and
//! terminal-loss propagation live here. Dedicated resources and submission
//! mechanics are sibling chapters: owning a `VkDevice` is never used as proof
//! that their portable capability exists, and each fact is published only when
//! the corresponding path is closed end to end.

pub(crate) mod device;
mod facts;
mod provider;
mod request;

pub(crate) use provider::VulkanProvider;

#[cfg(test)]
mod tests;
