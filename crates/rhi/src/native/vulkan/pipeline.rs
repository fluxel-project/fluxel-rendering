//! Step 5's pipeline half: the pipeline layout and the compute pipeline.
//!
//! # The pipeline owns the layout it was created with
//!
//! In `Vulkan` a `VkPipeline` refers to its `VkPipelineLayout` when it is bound, so
//! destroying the layout before the pipeline is invalid. The layout is therefore an
//! owned field of [`ComputePipeline`] rather than a borrow a caller has to remember
//! to keep alive, and the two are destroyed together in the order the dependency
//! requires: the pipeline first, then the layout it refers to.
//!
//! # The shader module is transient
//!
//! `Vulkan` reads a module when a pipeline is created and never again, so a module
//! may be destroyed as soon as its pipeline exists. [`create_compute`] creates one,
//! records it into the create-info, and lets it drop on the way out of the call --
//! after `vkCreateComputePipelines` returned and not before. That order is a scope
//! rather than a comment, because [`shader::Module`] destroys its handle in `Drop`.
//!
//! # What the layout carries today, and what it will carry
//!
//! No descriptor set layouts and no push constant ranges yet. The set-layout slice
//! is already a parameter, so the descriptor increment fills it rather than changing
//! this call, and an empty slice is the honest description of a pipeline whose
//! shader declares no bindings. A push constant range would claim a mechanism no
//! retained recipe uses: the borrowed path being replaced passes an immediate-data
//! size of zero, which is "no push constants" here.
//!
//! # A failed creation is not partially usable
//!
//! `vkCreateComputePipelines` reports failure with the contents of its output array
//! **undefined**, and the specification directs an application not to use them. The
//! partially filled vector `ash` returns on that path is therefore dropped without
//! being read. That is deliberate: destroying a handle the driver did not promise to
//! have created would be worse than not destroying it.

use core::ffi::CStr;

use ash::vk;

use crate::shader_contract::ShaderStage;

use super::shader::{self, Module, ShaderError};

/// Why a pipeline layout or a pipeline could not be created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PipelineError {
    /// The driver refused to create the pipeline layout.
    Layout(vk::Result),
    /// The shader module could not be described or created.
    Shader(ShaderError),
    /// The driver refused to create the pipeline.
    Creation(vk::Result),
}

/// A `VkPipelineLayout` this backend owns exactly once.
pub(crate) struct PipelineLayout {
    device: ash::Device,
    handle: vk::PipelineLayout,
}

impl PipelineLayout {
    /// Returns the driver handle a pipeline is created against.
    pub(crate) const fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        // SAFETY: the handle was created by this device and is destroyed once here.
        // The device outlives this layout because the caller that created it holds
        // both, and every pipeline created against this layout is destroyed first.
        unsafe { self.device.destroy_pipeline_layout(self.handle, None) };
    }
}

impl core::fmt::Debug for PipelineLayout {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PipelineLayout")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

/// A `vk::Pipeline` for a compute kernel, with the layout it refers to.
pub(crate) struct ComputePipeline {
    device: ash::Device,
    handle: vk::Pipeline,
    /// The layout this pipeline refers to at bind time.
    ///
    /// An owned field, not a borrow: a pipeline that outlives its layout is invalid,
    /// so the only safe shape is for the longer-lived object to contain the shorter.
    layout: PipelineLayout,
}

impl ComputePipeline {
    /// Returns the driver handle the compute recorder binds.
    pub(crate) const fn handle(&self) -> vk::Pipeline {
        self.handle
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        // SAFETY: the handle was created by this device and is destroyed once here.
        // The layout field is released by field order *after* this body, which is the
        // order Vulkan requires: the pipeline refers to the layout, not the reverse.
        unsafe { self.device.destroy_pipeline(self.handle, None) };
    }
}

impl core::fmt::Debug for ComputePipeline {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ComputePipeline")
            .field("handle", &self.handle)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// Creates a pipeline layout over `set_layouts`.
///
/// An empty slice is legal and is what the retained compute artifacts use today:
/// their bindings are not expressible until the binding vocabulary lands. The slice
/// is borrowed only for the call, because `Vulkan` copies the handles into the
/// layout it creates -- so the descriptor set layouts themselves must still outlive
/// every pipeline created against this layout.
pub(crate) fn create_layout(
    device: &ash::Device,
    set_layouts: &[vk::DescriptorSetLayout],
) -> Result<PipelineLayout, PipelineError> {
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
    // SAFETY: the device is live; every set layout is a live handle this device
    // created, and the slice outlives the call.
    let handle = unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(PipelineError::Layout)?;
    Ok(PipelineLayout {
        device: device.clone(),
        handle,
    })
}

/// Creates a compute pipeline from `words`, entering `entry_point`, over `layout`.
///
/// `layout` is consumed rather than borrowed because the resulting pipeline keeps it
/// alive; see the module docs. `words` is the same SPIR-V payload
/// [`shader::create_module`] describes, and is validated before the driver sees it.
pub(crate) fn create_compute(
    device: &ash::Device,
    layout: PipelineLayout,
    words: &[u32],
    entry_point: &CStr,
) -> Result<ComputePipeline, PipelineError> {
    // The module lives exactly as long as it is needed: the driver reads it during
    // creation, and it is destroyed by its own drop on the way out of this call.
    let module: Module =
        shader::create_module(device, words).map_err(PipelineError::Shader)?;
    let stage = shader::stage(ShaderStage::Compute, module.handle(), entry_point);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout.handle());
    // The create-info slice must outlive the call, so it is a binding rather than an
    // inline array literal.
    let infos = [info];
    // SAFETY: the device is live; the layout and module are live handles this device
    // created; the create-info borrows only locals that outlive the call, and a null
    // pipeline cache is the valid "no cache" value.
    let created = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
    };
    let handle = match created {
        // One create-info yields exactly one pipeline.
        Ok(mut pipelines) => pipelines
            .pop()
            .expect("one create-info yields one pipeline"),
        // The output array is undefined on failure and is deliberately not read;
        // see the module docs.
        Err((_undefined, error)) => return Err(PipelineError::Creation(error)),
    };
    Ok(ComputePipeline {
        device: device.clone(),
        handle,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::native::vulkan::open;
    use crate::native::vulkan::shader::MINIMAL_COMPUTE_SPIRV;

    #[test]
    fn a_real_compute_pipeline_is_created_from_a_spirv_module_on_this_machine() {
        // Step 5's compute half against the real driver: the module is read during
        // creation, the pipeline refers to the layout, and dropping the pipeline
        // destroys both in the order Vulkan requires. Skips where no adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let layout =
            create_layout(opened.device.device(), &[]).expect("an empty pipeline layout");
        let pipeline = create_compute(
            opened.device.device(),
            layout,
            &MINIMAL_COMPUTE_SPIRV,
            c"main",
        )
        .expect("a compute pipeline whose module is valid SPIR-V");
        assert_ne!(pipeline.handle(), vk::Pipeline::null());
        // The drop destroys the pipeline and then its layout; nothing else releases
        // either handle.
        drop(pipeline);
    }

    #[test]
    fn a_payload_that_is_not_spirv_is_refused_before_the_driver_is_reached() {
        // The refusal happens inside `create_compute`, before a pipeline exists, and
        // the layout it consumed is released with it rather than leaked.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let layout =
            create_layout(opened.device.device(), &[]).expect("an empty pipeline layout");
        let refused = create_compute(opened.device.device(), layout, &[0x0000_0001], c"main");
        assert_eq!(
            refused.err(),
            Some(PipelineError::Shader(ShaderError::NotSpirV { found: 1 }))
        );
    }
}
