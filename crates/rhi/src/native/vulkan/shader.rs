//! Step 5's shader half: a SPIR-V module and the pipeline stage that names it.
//!
//! # The module owns the handle, not the words
//!
//! `Vulkan` copies the code when a `VkShaderModule` is created, so the words a
//! caller passes are borrowed only for the duration of the call. The owned type
//! here therefore holds the handle and the device that must destroy it -- not the
//! SPIR-V: keeping the words alive after the driver has copied them would be a
//! second owner of a fact the module no longer reads.
//!
//! A module is destroyed **after** the pipeline that consumed it exists, and
//! `Vulkan` permits destroying it immediately afterwards. That order is a scope
//! rather than a comment: [`create_module`] returns a value whose `Drop` destroys
//! the handle, and [`super::pipeline`] creates one inside the call that builds its
//! pipeline, so the drop happens on the way out of that call and not before.
//!
//! # What is refused here, and why only one thing can be
//!
//! `VkShaderModuleCreateInfo` requires `codeSize` to be a non-zero multiple of four
//! and `pCode` to point at four-byte-aligned SPIR-V. Taking `&[u32]` states the last
//! two facts in the type, so they cannot be got wrong at the call site and are not
//! re-checked. What remains is the payload itself:
//!
//! - an empty slice -- byte size zero, which the specification calls invalid;
//! - a first word that is not the SPIR-V magic number.
//!
//! Both are refused with a sentence before the driver is reached, for the reason
//! every refusal in this backend exists: a driver validation error is a worse answer
//! than a reason the caller can act on. Nothing here decodes the instruction stream.
//! That belongs to the step that *produces* the words (step 6, Naga `spv-out`), and
//! a second partial parser here would be a second truth about the same bytes.
//!
//! # One stage, one entry point, no specialization
//!
//! [`stage`] lowers one portable [`ShaderStage`] and one module into the
//! `VkPipelineShaderStageCreateInfo` a pipeline binds. The entry point is the
//! caller's `&CStr`, never a default: which function in a module a stage enters is a
//! property of the artifact, and a backend that assumed `main` would enter the wrong
//! function for an artifact whose entry point is something else.
//!
//! Specialization info is left null, which is the value that declares no
//! specialization constants at all. No retained artifact declares one, so filling
//! the field from a default would be a capability claim rather than a lowering.
//!
//! The stage mapping is exhaustive on purpose, without a wildcard: [`ShaderStage`]
//! is this crate's own closed enum, so a stage added later must be taught to this
//! match rather than silently lowered to no stage flag. That is the opposite shape
//! from a mapping over a `#[non_exhaustive]` portable enum, where a wildcard would
//! invent a value and the mapping therefore returns `Option`.

use core::ffi::CStr;

use ash::vk;

use crate::shader_contract::ShaderStage;

/// The first word of every SPIR-V module.
///
/// `0x07230203`, little-endian, as the specification's magic number.
pub(crate) const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Why a SPIR-V module could not be described or created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShaderError {
    /// The payload was empty, which is not a module `Vulkan` can create.
    Empty,
    /// The first word is not the SPIR-V magic number.
    ///
    /// The offending word is carried rather than a boolean, because "the caller
    /// handed this backend a different container" and "the payload is truncated"
    /// produce different first words, and the diagnostic should say which arrived.
    NotSpirV {
        /// The first word that was found.
        found: u32,
    },
    /// The driver refused to create the module.
    Create(vk::Result),
}

/// A `VkShaderModule` this backend owns and destroys exactly once.
pub(crate) struct Module {
    device: ash::Device,
    handle: vk::ShaderModule,
}

impl Module {
    /// Returns the driver handle a pipeline stage binds.
    pub(crate) const fn handle(&self) -> vk::ShaderModule {
        self.handle
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: the handle was created by this device and is destroyed once here.
        // The device outlives this module because the caller that created it holds
        // both, and Vulkan permits destroying a module as soon as the pipeline that
        // read it exists.
        unsafe { self.device.destroy_shader_module(self.handle, None) };
    }
}

impl core::fmt::Debug for Module {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Module")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

/// The create-info for a module built from `words`.
///
/// `words` is borrowed for as long as the returned value lives, which is exactly
/// what `ash`'s builder records: `p_code` points at the slice rather than copying
/// it. A caller therefore cannot free the words before the driver has read them.
pub(crate) fn module_create_info(
    words: &[u32],
) -> Result<vk::ShaderModuleCreateInfo<'_>, ShaderError> {
    let Some(&first) = words.first() else {
        return Err(ShaderError::Empty);
    };
    if first != SPIRV_MAGIC {
        return Err(ShaderError::NotSpirV { found: first });
    }
    Ok(vk::ShaderModuleCreateInfo::default().code(words))
}

/// Creates the module `words` describes, or reports why not.
///
/// The returned value owns the handle and destroys it on drop; see the module docs
/// for why the handle and not the words is what needs an owner.
pub(crate) fn create_module(device: &ash::Device, words: &[u32]) -> Result<Module, ShaderError> {
    let info = module_create_info(words)?;
    // SAFETY: the device is live; the create-info borrows `words`, which outlives
    // this call, and its code size and pointer alignment are stated by `&[u32]`.
    let handle =
        unsafe { device.create_shader_module(&info, None) }.map_err(ShaderError::Create)?;
    Ok(Module {
        device: device.clone(),
        handle,
    })
}

/// The `Vulkan` stage flag for one portable stage.
///
/// Exhaustive rather than `Option`-returning, because [`ShaderStage`] is a closed
/// enum this crate defines: a new variant must be taught to this match at compile
/// time instead of falling through to no stage flag at run time.
pub(crate) const fn stage_flags(stage: ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
    }
}

/// The pipeline stage that enters `entry_point` of `module`.
///
/// The entry point is borrowed rather than copied, so the returned value cannot
/// outlive the name the artifact owns. Nothing else is filled: a null
/// `p_specialization_info` is the declaration that the stage uses no specialization
/// constants.
pub(crate) fn stage(
    stage: ShaderStage,
    module: vk::ShaderModule,
    entry_point: &CStr,
) -> vk::PipelineShaderStageCreateInfo<'_> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage_flags(stage))
        .module(module)
        .name(entry_point)
}

/// A minimal, valid compute SPIR-V module, for the real-driver tests.
///
/// Assembled once with `spirv-as --target-env spv1.0` from exactly this source and
/// accepted by `spirv-val`:
///
/// ```text
/// OpCapability Shader
/// OpMemoryModel Logical GLSL450
/// OpEntryPoint GLCompute %main "main"
/// OpExecutionMode %main LocalSize 1 1 1
/// %void = OpTypeVoid
/// %fn = OpTypeFunction %void
/// %main = OpFunction %void None %fn
/// %entry = OpLabel
/// OpReturn
/// OpFunctionEnd
/// ```
///
/// It stands in for the retained artifacts until step 6 lowers their WGSL with Naga;
/// only tests read it, which is why it is gated rather than shipped as vocabulary.
#[cfg(test)]
pub(crate) const MINIMAL_COMPUTE_SPIRV: [u32; 35] = [
    0x0723_0203, 0x0001_0000, 0x0007_0000, 0x0000_0005, 0x0000_0000, 0x0002_0011, 0x0000_0001,
    0x0003_000e, 0x0000_0000, 0x0000_0001, 0x0005_000f, 0x0000_0005, 0x0000_0001, 0x6e69_616d,
    0x0000_0000, 0x0006_0010, 0x0000_0001, 0x0000_0011, 0x0000_0001, 0x0000_0001, 0x0000_0001,
    0x0002_0013, 0x0000_0002, 0x0003_0021, 0x0000_0003, 0x0000_0002, 0x0005_0036, 0x0000_0002,
    0x0000_0001, 0x0000_0000, 0x0000_0003, 0x0002_00f8, 0x0000_0004, 0x0001_00fd, 0x0001_0038,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::native::vulkan::open;

    #[test]
    fn an_empty_payload_is_refused_before_the_driver_is_reached() {
        // A zero code size is invalid rather than an empty module, so there is
        // nothing to create and the refusal is a value.
        assert_eq!(module_create_info(&[]).err(), Some(ShaderError::Empty));
    }

    #[test]
    fn a_payload_that_is_not_spirv_is_refused_and_names_its_first_word() {
        let words = [0x4342_5844u32, 0x0000_0000];
        assert_eq!(
            module_create_info(&words).err(),
            Some(ShaderError::NotSpirV { found: 0x4342_5844 })
        );
    }

    #[test]
    fn the_create_info_carries_the_words_their_byte_size_and_no_flags() {
        let info = module_create_info(&MINIMAL_COMPUTE_SPIRV).expect("a SPIR-V module");
        assert_eq!(info.code_size, MINIMAL_COMPUTE_SPIRV.len() * 4);
        assert_eq!(info.p_code, MINIMAL_COMPUTE_SPIRV.as_ptr());
        assert_eq!(info.flags, vk::ShaderModuleCreateFlags::empty());
        assert_eq!(info.s_type, vk::StructureType::SHADER_MODULE_CREATE_INFO);
        assert!(info.p_next.is_null(), "no extension chain is installed");
    }

    #[test]
    fn the_three_stages_map_to_their_own_single_bit() {
        // Distinct single bits, so a copy-paste in the match shows up as two stages
        // sharing one flag rather than as a silently wrong pipeline.
        let mapped = [
            stage_flags(ShaderStage::Vertex),
            stage_flags(ShaderStage::Fragment),
            stage_flags(ShaderStage::Compute),
        ];
        assert_eq!(mapped[0], vk::ShaderStageFlags::VERTEX);
        assert_eq!(mapped[1], vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(mapped[2], vk::ShaderStageFlags::COMPUTE);
        for (index, flag) in mapped.iter().enumerate() {
            for (other_index, other) in mapped.iter().enumerate() {
                if index != other_index {
                    assert!(
                        !flag.intersects(*other),
                        "{flag:?} and {other:?} share a stage bit"
                    );
                }
            }
            assert!(!flag.is_empty(), "a stage flag is never empty");
        }
        // The raw bit each portable stage claims, read from the constants rather
        // than from a second hand-written number: vertex and compute share no bit.
        assert_eq!(
            vk::ShaderStageFlags::VERTEX.as_raw() & vk::ShaderStageFlags::COMPUTE.as_raw(),
            0
        );
    }

    #[test]
    fn the_stage_carries_its_flag_its_module_and_its_entry_point() {
        use ash::vk::Handle;

        let module = vk::ShaderModule::from_raw(0x1234);
        let entry_point = c"main";
        let stage = stage(ShaderStage::Fragment, module, entry_point);
        assert_eq!(stage.stage, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(stage.module, module);
        assert_eq!(stage.p_name, entry_point.as_ptr());
        assert!(stage.p_next.is_null(), "no extension chain is installed");
        assert!(
            stage.p_specialization_info.is_null(),
            "the fixed artifacts declare no specialization constants"
        );
        assert_eq!(stage.flags, vk::PipelineShaderStageCreateFlags::empty());
    }

    #[test]
    fn a_real_spirv_module_is_created_and_destroyed_on_this_machine() {
        // Step 5's module half against the real driver: the words are validated by
        // this layer, the handle is the driver's, and dropping the owner destroys it.
        // Skips where no adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let module = create_module(opened.device.device(), &MINIMAL_COMPUTE_SPIRV)
            .expect("a module described by valid SPIR-V");
        assert_ne!(module.handle(), vk::ShaderModule::null());
        // The drop is the destruction; nothing else releases the handle.
        drop(module);
    }
}
