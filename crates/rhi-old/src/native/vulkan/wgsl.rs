//! Step 6: the retained WGSL artifact lowered to SPIR-V, before the driver sees
//! it.
//!
//! Decision F fixes the two shader routes this backend has: caller-supplied SPIR-V
//! is **passthrough**, and WGSL is lowered here with Naga's `wgsl-in` frontend and
//! `spv-out` backend. The two routes meet in one place -- [`create_module`], which
//! lowers when it has to and then hands the words to
//! [`super::shader::create_module`] -- so the driver is only ever reached with SPIR-V.
//!
//! # The dialect and profile check happens before driver creation
//!
//! Three refusals are values returned by [`lower`], and all three happen **before**
//! a `VkShaderModule` exists, which is what makes a bad artifact a sentence rather
//! than a validation-layer message:
//!
//! - **dialect.** The source is fed to the WGSL frontend. A GLSL or HLSL payload is
//!   not accepted because it is not this dialect; the failure names the parse, with
//!   the source span. (The other route, SPIR-V passthrough, is checked for the
//!   container by [`super::shader::module_create_info`] instead.)
//! - **entry point and stage.** A pipeline stage names one entry point of a module,
//!   and the module must have that entry point declared for that stage. A missing
//!   name and a name declared for another stage are different sentences, and both
//!   are decided here rather than by the driver, which would report neither.
//! - **profile.** The writer targets SPIR-V 1.0, the version a Vulkan 1.0 device
//!   accepts, and this backend's instance requests exactly `VK_API_VERSION_1_0`
//!   ([`super::instance`]). A module that needs a newer version is refused by the
//!   writer rather than handed to a driver that cannot load it.
//!
//! # Why the writer options are written field by field
//!
//! `naga::back::spv::Options::default()` is **not** the value that claims nothing:
//! it sets `ADJUST_COORDINATE_SPACE`, which flips the Y coordinate of
//! `BuiltIn::Position`. The borrowed Vulkan path being replaced does not set that
//! flag (the vendored `crates/wgpu-hal`, `vulkan/adapter.rs`), so inheriting the
//! default would silently flip every retained recipe's geometry against the frozen
//! oracle. Every field is therefore stated here, and a field Naga adds later is a
//! compile error at this literal rather than an inherited claim. The decisions:
//!
//! - `lang_version` is `(1, 0)`; see the profile note above.
//! - `flags` carries `FORCE_POINT_SIZE` and nothing else. A vertex module is lowered
//!   before the topology it will be paired with is known, `Points` is in the
//!   pipeline vocabulary, and the borrowed path always emits the built-in, so
//!   leaving it out would make point rendering a topology-dependent accident.
//!   `ADJUST_COORDINATE_SPACE`, `CLAMP_FRAG_DEPTH`, `LABEL_VARYINGS` and `DEBUG`
//!   are absent: the first changes geometry, the second clamps an output no retained
//!   fragment writes, the third writes decorations the specification does not
//!   require, and the fourth would make the emitted words depend on the build
//!   profile rather than on the artifact a cache hashes.
//! - `capabilities` is an explicit set, not `None`. `None` means "all capabilities
//!   are permitted", which would let a shader use `Float64` or `MultiView` on a
//!   device that enables no feature at all. The set is the one the borrowed path
//!   treats as always available without a `VkPhysicalDeviceFeature` or an extension.
//! - `use_storage_input_output_16` is `false`, because this device enables no f16
//!   feature and the capability is exactly what that field switches on.
//! - `fake_missing_bindings` is `true`, and the name is misleading. The binding map
//!   is deliberately empty -- this backend keeps each artifact's own `@group` /
//!   `@binding` numbers -- and an empty map plus this flag is the identity mapping.
//!   With `false` and an empty map, **every** resource would be refused as a missing
//!   binding; the flag here is the fallback that emits the binding the artifact
//!   declared, not a permission to invent one.
//! - the bounds policies are `Restrict` for index, buffer and image loads, because
//!   the device enables no robust-access feature, and `Unchecked` for binding arrays,
//!   which this layer's vocabulary does not contain.
//!
//! # Validation is Naga's, and it is fail-closed
//!
//! [`Validator::validate`] runs with `ValidationFlags::all()` and
//! `valid::Capabilities::empty()`, so an artifact using a feature this device has
//! not proved is refused at validation, before the writer runs. The two capability
//! layers are not duplicates: validation knows the portable feature set, while the
//! writer's set decides which SPIR-V capabilities may appear.
//!
//! [`Validator::validate`]: naga::valid::Validator::validate

use naga::back::spv;

use crate::shader_contract::ShaderStage;

use super::shader::{self, Module, ShaderError};

/// Why a WGSL artifact could not be lowered to SPIR-V.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WgslError {
    /// The source is not WGSL this frontend accepts.
    Parse {
        /// The frontend's diagnostic, rendered against the source.
        message: String,
    },
    /// The module parsed but is not a valid module.
    Validate {
        /// The validator's diagnostic.
        message: String,
    },
    /// No entry point of this name exists in the module.
    EntryPoint {
        /// The name that was looked for.
        name: String,
    },
    /// The named entry point exists but is declared for another stage.
    ///
    /// The stage that was found is Naga's, not the portable stage: the entry point
    /// may be a task, mesh or ray-tracing entry point, which this crate's closed
    /// [`ShaderStage`] has no name for, and saying so is more useful than calling it
    /// absent.
    Stage {
        /// The entry point's name.
        name: String,
        /// The stage the caller asked for.
        declared: ShaderStage,
        /// The stage the module declares for that name.
        found: naga::ShaderStage,
    },
    /// The writer refused the module for the target profile.
    Lower {
        /// The writer's diagnostic.
        message: String,
    },
}

/// Why a WGSL artifact did not become a `VkShaderModule`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ModuleError {
    /// The dialect, entry-point, stage or profile check refused, before the driver
    /// was reached.
    Lower(WgslError),
    /// The driver refused to create the module the lowering produced.
    Create(ShaderError),
}

/// The Naga stage a portable stage names.
///
/// Exhaustive without a wildcard, because [`ShaderStage`] is this crate's own closed
/// enum: a stage added later must be taught to this match rather than silently
/// lowered to no stage at all. The opposite direction -- Naga's stage to the
/// portable one -- would have to return `Option`, because Naga declares stages this
/// vocabulary deliberately does not.
pub(crate) const fn naga_stage(stage: ShaderStage) -> naga::ShaderStage {
    match stage {
        ShaderStage::Vertex => naga::ShaderStage::Vertex,
        ShaderStage::Fragment => naga::ShaderStage::Fragment,
        ShaderStage::Compute => naga::ShaderStage::Compute,
    }
}

/// The SPIR-V capabilities a device with no enabled feature can serve.
///
/// These are the capabilities the borrowed Vulkan path being replaced treats as
/// always available without a feature or an extension; every other capability is
/// conditional on something this backend's device does not enable, so permitting it
/// here would be a claim the ledger cannot back.
fn base_capabilities() -> naga::FastHashSet<spv::Capability> {
    [
        spv::Capability::Shader,
        spv::Capability::Matrix,
        spv::Capability::Sampled1D,
        spv::Capability::Image1D,
        spv::Capability::ImageQuery,
        spv::Capability::DerivativeControl,
        spv::Capability::StorageImageExtendedFormats,
    ]
    .into_iter()
    .collect()
}

/// The one writer configuration this backend lowers WGSL with.
///
/// The module docs state why each field has the value it has; the short version is
/// that nothing here is inherited from `Options::default()`, because that default
/// flips the Y coordinate of every position.
pub(crate) fn options() -> spv::Options<'static> {
    spv::Options {
        lang_version: (1, 0),
        flags: spv::WriterFlags::FORCE_POINT_SIZE,
        fake_missing_bindings: true,
        binding_map: spv::BindingMap::default(),
        capabilities: Some(base_capabilities()),
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Restrict,
            buffer: naga::proc::BoundsCheckPolicy::Restrict,
            image_load: naga::proc::BoundsCheckPolicy::Restrict,
            // Binding arrays are not in this layer's vocabulary, so there is no
            // index that could be checked.
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
        force_loop_bounding: true,
        ray_query_initialization_tracking: true,
        trace_ray_argument_validation: true,
        use_storage_input_output_16: false,
        debug_info: None,
        task_dispatch_limits: None,
        mesh_shader_primitive_indices_clamp: false,
        emit_int_div_checks: true,
    }
}

/// The entry point and stage check [`lower`] performs before the writer runs.
///
/// Split out so the two refusals are testable without a source that parses, and so
/// the name lookup is written once.
fn check_entry_point(
    module: &naga::Module,
    stage: ShaderStage,
    entry_point: &str,
) -> Result<(), WgslError> {
    let wanted = naga_stage(stage);
    match module
        .entry_points
        .iter()
        .find(|entry| entry.name == entry_point)
    {
        None => Err(WgslError::EntryPoint {
            name: entry_point.to_owned(),
        }),
        Some(entry) if entry.stage != wanted => Err(WgslError::Stage {
            name: entry_point.to_owned(),
            declared: stage,
            found: entry.stage,
        }),
        Some(_) => Ok(()),
    }
}

/// Lowers one WGSL artifact into the SPIR-V words a shader module is created from.
///
/// The words are this backend's own allocation and outlive the call; `Vulkan` copies
/// them when the module is created, so they are not retained by the module.
pub(crate) fn lower(
    source: &str,
    stage: ShaderStage,
    entry_point: &str,
) -> Result<Vec<u32>, WgslError> {
    let module = naga::front::wgsl::Frontend::new()
        .parse(source)
        .map_err(|error| WgslError::Parse {
            message: error.emit_to_string(source),
        })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| WgslError::Validate {
        message: format!("{error}"),
    })?;
    check_entry_point(&module, stage, entry_point)?;

    let pipeline = spv::PipelineOptions {
        shader_stage: naga_stage(stage),
        entry_point: entry_point.to_owned(),
    };
    spv::write_vec(&module, &info, &options(), Some(&pipeline)).map_err(|error| {
        WgslError::Lower {
            message: format!("{error}"),
        }
    })
}

/// Lowers `source` and creates the `VkShaderModule` it describes.
///
/// Everything that can refuse without a device refuses first, so a driver error
/// here is the driver's own answer about a module it was actually handed.
pub(crate) fn create_module(
    device: &ash::Device,
    source: &str,
    stage: ShaderStage,
    entry_point: &str,
) -> Result<Module, ModuleError> {
    let words = lower(source, stage, entry_point).map_err(ModuleError::Lower)?;
    shader::create_module(device, &words).map_err(ModuleError::Create)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::native::vulkan::open;
    use crate::resource::{ComputeKernel, RasterKernel};

    /// Whether `name` appears as a literal string in the emitted words.
    ///
    /// SPIR-V packs a literal string into words little-endian. Without the writer's
    /// debug flag the only name that reaches the module is the selected entry
    /// point's, so this proves the requested entry point is the one that was
    /// written.
    fn names_the_entry_point(words: &[u32], name: &str) -> bool {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        bytes
            .windows(name.len())
            .any(|window| window == name.as_bytes())
    }

    #[test]
    fn the_retained_artifacts_lower_to_their_own_entry_points() {
        // The retained triangle is one module with two stages, so it proves the
        // stage selection rather than only the parse.
        let triangle = RasterKernel::Triangle.wgsl_source();
        let vertex = lower(
            triangle,
            ShaderStage::Vertex,
            RasterKernel::Triangle.vertex_entry_point(),
        )
        .expect("the retained triangle vertex lowers");
        assert_eq!(vertex.first().copied(), Some(shader::SPIRV_MAGIC));
        assert!(names_the_entry_point(&vertex, "triangle_vertex"));

        let fragment = lower(
            triangle,
            ShaderStage::Fragment,
            RasterKernel::Triangle.fragment_entry_point(),
        )
        .expect("the retained triangle fragment lowers");
        assert_eq!(fragment.first().copied(), Some(shader::SPIRV_MAGIC));
        assert!(names_the_entry_point(&fragment, "color_fragment"));
        assert_ne!(
            vertex, fragment,
            "one module lowered for two stages is not the same words twice"
        );

        let compute = ComputeKernel::TextureStoreRgba8.wgsl_source();
        let words = lower(compute, ShaderStage::Compute, "store_rgba8")
            .expect("the retained storage-texture compute artifact lowers");
        assert_eq!(words.first().copied(), Some(shader::SPIRV_MAGIC));
        assert!(names_the_entry_point(&words, "store_rgba8"));
    }

    #[test]
    fn an_entry_point_declared_for_another_stage_is_refused_by_name() {
        let error = lower(
            RasterKernel::Triangle.wgsl_source(),
            ShaderStage::Fragment,
            "triangle_vertex",
        )
        .expect_err("a vertex entry point is not a fragment entry point");
        assert_eq!(
            error,
            WgslError::Stage {
                name: "triangle_vertex".to_owned(),
                declared: ShaderStage::Fragment,
                found: naga::ShaderStage::Vertex,
            }
        );
    }

    #[test]
    fn an_unknown_entry_point_is_refused_before_the_writer_runs() {
        let error = lower(
            RasterKernel::Triangle.wgsl_source(),
            ShaderStage::Vertex,
            "no_such_entry",
        )
        .expect_err("the module declares no such entry point");
        assert_eq!(
            error,
            WgslError::EntryPoint {
                name: "no_such_entry".to_owned(),
            }
        );
    }

    #[test]
    fn a_source_that_is_not_the_wgsl_dialect_is_refused_at_parse() {
        // GLSL is the dialect the GL family accepts natively; this backend lowers
        // only WGSL, so the payload is refused as a dialect rather than translated
        // by a route this backend does not implement.
        let error = lower("void main() { }", ShaderStage::Vertex, "main")
            .expect_err("GLSL is not WGSL");
        assert!(
            matches!(error, WgslError::Parse { .. }),
            "expected a parse refusal, got {error:?}"
        );
    }

    #[test]
    fn a_module_that_parses_but_does_not_validate_is_refused_before_the_writer() {
        // A zero workgroup dimension is legal syntax and illegal module: the frontend
        // parses the attribute and only the validator knows the size is out of range.
        let error = lower(
            "@compute @workgroup_size(0) fn main() {}",
            ShaderStage::Compute,
            "main",
        )
        .expect_err("a zero workgroup dimension is out of range");
        assert!(
            matches!(error, WgslError::Validate { .. }),
            "expected a validation refusal, got {error:?}"
        );
    }

    #[test]
    fn the_writer_targets_the_profile_a_vulkan_1_0_device_accepts() {
        let options = options();
        // SPIR-V 1.0 is the version the instance this backend opens requests.
        assert_eq!(options.lang_version, (1, 0));
        // The borrowed Vulkan path being replaced never flips Position Y, so
        // inheriting `Options::default()` here would move the frozen oracle.
        assert!(
            !options
                .flags
                .contains(spv::WriterFlags::ADJUST_COORDINATE_SPACE),
            "the Y coordinate is not adjusted"
        );
        assert!(options.flags.contains(spv::WriterFlags::FORCE_POINT_SIZE));
        assert!(!options.flags.contains(spv::WriterFlags::DEBUG));
        // The device enables no f16 feature, so the 16-bit storage I/O capability
        // this field switches on is not claimable.
        assert!(!options.use_storage_input_output_16);
        // An empty map with this fallback is the identity binding, not an invention:
        // with `false` every resource would be refused as a missing binding.
        assert!(options.fake_missing_bindings);
        assert!(options.binding_map.is_empty());
        assert_eq!(
            options.zero_initialize_workgroup_memory,
            spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
            "the device proves no native zero-init feature"
        );

        let capabilities = options.capabilities.expect("an explicit capability set");
        assert!(capabilities.contains(&spv::Capability::Shader));
        assert!(capabilities.contains(&spv::Capability::ImageQuery));
        assert!(capabilities.contains(&spv::Capability::Matrix));
        // A device with no enabled feature cannot serve these.
        assert!(!capabilities.contains(&spv::Capability::Float64));
        assert!(!capabilities.contains(&spv::Capability::Float16));
        assert!(!capabilities.contains(&spv::Capability::MultiView));
        assert!(!capabilities.contains(&spv::Capability::DrawParameters));
    }

    #[test]
    fn a_real_wgsl_vertex_module_is_created_and_destroyed_on_this_machine() {
        // Step 6 end to end against the real driver: the retained artifact's WGSL is
        // lowered by Naga, the words are validated by this layer, and the handle the
        // driver returns is owned and destroyed here. Skips where no adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let kernel = RasterKernel::IndexedPositionColor;
        let module = create_module(
            opened.device.device(),
            kernel.wgsl_source(),
            ShaderStage::Vertex,
            kernel.vertex_entry_point(),
        )
        .expect("the retained vertex lowers to a module the driver accepts");
        assert_ne!(module.handle(), ash::vk::ShaderModule::null());
        drop(module);
    }

    #[test]
    fn a_real_wgsl_compute_module_is_created_and_destroyed_on_this_machine() {
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let kernel = ComputeKernel::TextureStoreRgba8;
        let module = create_module(
            opened.device.device(),
            kernel.wgsl_source(),
            ShaderStage::Compute,
            kernel.entry_point(),
        )
        .expect("the retained compute artifact lowers to a module the driver accepts");
        assert_ne!(module.handle(), ash::vk::ShaderModule::null());
        drop(module);
    }

    #[test]
    fn the_driver_is_not_reached_when_the_lowering_refuses() {
        // The subject is the order: a bad entry point must not create a module, so
        // the refusal is the lowering's and never a driver error.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let error = create_module(
            opened.device.device(),
            RasterKernel::Triangle.wgsl_source(),
            ShaderStage::Vertex,
            "color_fragment",
        )
        .expect_err("a fragment entry point is not a vertex entry point");
        assert!(matches!(error, ModuleError::Lower(WgslError::Stage { .. })));
    }
}
