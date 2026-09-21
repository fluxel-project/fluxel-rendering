//! Conservative capability projection for Metal.
//!
//! Metal does not expose a single `CheckFormatSupport` equivalent. A valid
//! answer is instead the intersection of the selected `MTLDevice` family,
//! OS availability, and the lowering that this backend has actually wired.
//! `MetalCapabilityLimits` is that already-probed intersection; it deliberately
//! contains no optimistic defaults.

use crate::api::binding::vocabulary::{BindableKind, TextureSampleType};
use crate::api::binding::{
    BindingLimitClass, BindingSupport, BufferBindingAccess, SamplerKind, StorageAccess,
};
use crate::api::capability::{BindingSupportKey, CapabilityFacts};
use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery, format_aspects,
};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
use crate::api::resource::route::{
    RouteCapabilities, RouteQuery, RouteSupport, TexelCopyLayoutLimits,
};
use crate::api::resource::subresource::TextureAspect;
use crate::api::resource::texture::{
    Extent3d, TextureDimension, TextureUsage, TextureViewCompatibility,
};
use crate::api::resource::transient::{TransientAllocationSupport, TransientCapabilities};
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::ShaderStages;
use crate::api::shader::vocabulary::AcceptedCodeForm;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::sel;
use objc2_metal::{MTLDevice, MTLGPUFamily, MTLReadWriteTextureTier};

use super::format::{CompressedFamily, compressed_family, metal_format};

/// The device-specific facts collected by the provider before projecting
/// portable capabilities. `false` is intentionally the default for every
/// optional class: a native enum spelling is not proof that the selected GPU
/// family accepts it.
#[derive(Clone, Copy, Debug)]
pub(super) struct MetalCapabilityLimits {
    pub(super) max_buffer_size: u64,
    pub(super) max_texture_1d: u32,
    pub(super) max_texture_2d: u32,
    pub(super) max_texture_3d: u32,
    pub(super) max_array_layers: u32,
    pub(super) max_mip_levels: u32,
    pub(super) max_sampler_anisotropy: Option<u32>,
    pub(super) bc: bool,
    pub(super) etc2_eac: bool,
    pub(super) astc_ldr: bool,
    pub(super) astc_hdr: bool,
    /// Volume ASTC is a narrower capability than ASTC LDR: Apple documents it
    /// from Apple GPU family 3 onward.
    pub(super) astc_3d: bool,
    /// One-dimensional textures are not present on every Apple Metal profile.
    pub(super) texture_1d: bool,
    /// Cube-array views have a narrower feature-family floor than cube views.
    pub(super) texture_cube_array: bool,
    /// `Depth16Unorm` is OS-availability gated despite having an SDK spelling.
    pub(super) depth16_unorm: bool,
    /// Full RGBA32Float color capabilities (including blending) are a desktop
    /// Metal guarantee; mobile Apple families only guarantee attachment use.
    pub(super) rgba32_float_blend: bool,
    /// Visibility-result buffers and their resolve copy are implemented.
    pub(super) occlusion_queries: bool,
    /// Ordinary indirect draw/dispatch selectors are available. Count-buffer
    /// multi-draw is a separate capability and is never implied by this bit.
    pub(super) indirect_commands: bool,
    /// Direct draw selectors carrying base vertex/base instance are available
    /// on this family/runtime.  This is separate from ordinary indirect work:
    /// an old encoder may still draw, but cannot faithfully lower a non-zero
    /// portable base value.
    pub(super) base_vertex_instance: bool,
    /// The render encoder accepts `MTLDepthClipMode::Clamp`.
    pub(super) depth_clip_control: bool,
    /// Metal read/write texture tier, normalized to 0/1/2 after a guarded
    /// selector probe.
    pub(super) read_write_texture_tier: u8,
    /// Vertex/fragment direct texture bindings may use storage access.
    pub(super) raster_storage_textures: bool,
    /// Exact device-reported multisample counts, encoded as `1 << count`.
    /// Count one is always recorded when the texture path is live.
    pub(super) sample_count_mask: u32,
    /// Largest vertex-amplification count accepted by this exact device.
    pub(super) max_vertex_amplification_count: u32,
    /// Metal comparison sampler state is a distinct portable sampler contract.
    pub(super) comparison_samplers: bool,
    /// `ClampToBorderColor` is not universal on Metal hardware.
    pub(super) sampler_clamp_to_border: bool,
    /// Set only after resource creation and direct binding both handle storage
    /// textures. Exact ReadOnly/WriteOnly/ReadWrite rows are still emitted by
    /// `storage_accesses`; the latter is constrained by the selected device's
    /// `MTLReadWriteTextureTier` and documented per-tier format matrix.
    pub(super) storage_textures: bool,
    /// Set only after dispatch recording, bind-group application, and
    /// compute-pipeline lowering are all live. Creating an MTL compute-pipeline
    /// state is intentionally not enough: a pipeline which cannot be submitted
    /// is not a portable `Compute` capability.
    pub(super) compute_lowering: bool,
    /// Set with `compute_lowering` only after the ABI applies every published
    /// direct binding class, group, element and dynamic offset to a compute
    /// encoder. Raster visibility remains a separate future fact.
    pub(super) compute_binding_lowering: bool,
    /// Raster pipeline construction, attachment encoding, and direct draw
    /// replay are live. Kept separate from compute because neither command
    /// encoder implies the other in Metal.
    pub(super) raster_lowering: bool,
    /// Raster bind groups and immediate data are applied to both vertex and
    /// fragment encoder namespaces.
    pub(super) raster_binding_lowering: bool,
    /// Set only after the resource creator accepts the corresponding 1x texture
    /// descriptor family. Texture creation and command routes are separate
    /// facts: the latter are recorded below only where a blit encoder really
    /// consumes the resource.
    pub(super) texture_creation: bool,
    /// Set only after the exact native `MTLBlitCommandEncoder` copy lowerer is
    /// live.  This is deliberately distinct from texture creation: a resource
    /// that Metal can allocate is not automatically a direct portable route.
    pub(super) texture_copy_lowering: bool,
    /// Buffer/texture copy requires an independent layout proof: block-aware
    /// public validation plus the native per-image stride lowerer.
    pub(super) buffer_texture_copy_lowering: bool,
    /// Metal's device-family copy-buffer offset alignment. Row pitch itself is
    /// a four-byte native requirement and is recorded at the route below.
    pub(super) texel_copy_buffer_offset_alignment: u64,
    pub(super) max_color_attachments: u32,
    pub(super) max_color_attachment_bytes_per_sample: u32,
    pub(super) max_inter_stage_shader_variables: u32,
}

/// Projects the Metal device snapshot into v13 facts.
///
/// Every published texture/sample row has a matching resource and command
/// lowering. Unsupported sample/usage combinations remain absent even when the
/// SDK exposes a corresponding enum value.
pub(super) fn probe(limits: MetalCapabilityLimits) -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    facts.record_code_form(AcceptedCodeForm::Msl);
    facts.record_code_form(AcceptedCodeForm::Metallib);
    facts.record_transient_capabilities(TransientCapabilities {
        buffers: TransientAllocationSupport::Dedicated,
        textures: TransientAllocationSupport::Dedicated,
        mixed_resource_aliasing: false,
    });

    // Metal has a native compute command encoder, but it becomes an RHI fact
    // only after the dispatch lowerer is available.
    if limits.compute_lowering {
        facts.record_feature(OptionalFeature::Compute);
    }
    if limits.compute_binding_lowering {
        record_compute_binding_support(&mut facts, limits);
    }
    if limits.raster_binding_lowering {
        record_raster_binding_support(&mut facts, limits);
    }
    if limits.raster_lowering {
        // These are direct encoder calls in the current lowerer. Point fill
        // mode and depth-bias clamp remain absent until their native paths are
        // capability-closed; the independently probed optional rows below
        // cover depth clipping, multisampling, multiview and indirect work.
        facts.record_feature(OptionalFeature::PolygonModeLine);
        facts.record_feature(OptionalFeature::IndependentBlend);
        if limits.base_vertex_instance {
            facts.record_feature(OptionalFeature::BaseVertex);
            facts.record_feature(OptionalFeature::BaseInstance);
        }
        facts.record_limit(
            LimitKey::MaxColorAttachments,
            u64::from(limits.max_color_attachments),
        );
        facts.record_limit(
            LimitKey::MaxColorAttachmentBytesPerSample,
            u64::from(limits.max_color_attachment_bytes_per_sample),
        );
        // Direct Metal vertex fetch and direct argument buffers use the same
        // vertex-stage buffer namespace.  Reserving 15 vertex streams plus
        // at most 7 uniform, 7 storage and one immediate buffer is a strict
        // 30-slot portable budget under Metal's native 31-slot ceiling.
        facts.record_limit(LimitKey::MaxVertexBuffers, 15);
        facts.record_limit(LimitKey::MaxVertexAttributes, 31);
        facts.record_limit(LimitKey::MaxVertexBufferArrayStride, 2_048);
        facts.record_limit(
            LimitKey::MaxInterStageShaderVariables,
            u64::from(limits.max_inter_stage_shader_variables),
        );
        if limits.depth_clip_control {
            facts.record_feature(OptionalFeature::DepthClipControl);
        }
        if limits.indirect_commands && limits.base_vertex_instance {
            facts.record_feature(OptionalFeature::IndirectDraw);
            facts.record_feature(OptionalFeature::MultiDrawIndirect);
            facts.record_feature(OptionalFeature::IndirectFirstInstance);
        }
        if [2, 4, 8, 16]
            .into_iter()
            .any(|count| supports_sample_count(limits, count))
        {
            // Metal's render pipeline and attachment paths preserve per-sample
            // interpolation. Arbitrary fixed-function sample masks remain a
            // separate capability and are intentionally not published.
            facts.record_feature(OptionalFeature::MultisampledShading);
        }
        if limits.max_vertex_amplification_count > 1 {
            facts.record_feature(OptionalFeature::Multiview);
            // The Metal lowering supplies an explicit mapping for every active
            // bit, so sparse masks are native rather than emulated.
            facts.record_feature(OptionalFeature::SelectiveMultiview);
            facts.record_limit(
                LimitKey::MaxMultiviewViewCount,
                u64::from(limits.max_vertex_amplification_count.min(32)),
            );
        }
    }
    if limits.compute_lowering && limits.indirect_commands {
        facts.record_feature(OptionalFeature::IndirectDispatch);
    }
    if limits.occlusion_queries {
        facts.record_feature(OptionalFeature::OcclusionQuery);
        facts.record_feature(OptionalFeature::QueryResolve);
        facts.record_limit(LimitKey::MaxQueriesPerQuerySet, 65_536);
        facts.record_limit(LimitKey::QueryResolveBufferAlignment, 8);
    }
    if limits.compute_binding_lowering && limits.raster_binding_lowering {
        // `set*Bytes` accepts a maximum 4 KiB payload. The direct ABI uploads
        // the complete declared immediate address space in one call, so the
        // public maximum must be this native per-stage maximum, not a larger
        // sum across individual portable writes.
        facts.record_feature(OptionalFeature::Immediates);
        facts.record_limit(LimitKey::MaxImmediateSize, 4_096);
        facts.record_limit(LimitKey::ImmediateDataAlignment, 4);
    }
    if let Some(max) = limits.max_sampler_anisotropy.filter(|max| *max > 1) {
        facts.record_feature(OptionalFeature::SamplerAnisotropy);
        facts.record_limit(LimitKey::MaxSamplerAnisotropy, u64::from(max));
    }
    if limits.comparison_samplers {
        facts.record_feature(OptionalFeature::ComparisonSamplers);
    }
    if limits.sampler_clamp_to_border {
        facts.record_feature(OptionalFeature::SamplerClampToBorder);
    }

    // Metal shared-storage buffers back the exact MAP_* rows below, including
    // combinations with normal primary usages. Mapping waits for accepted use
    // in `resource::MetalMapRequest`, while shared storage has no explicit
    // CPU-cache flush/invalidate operation. A mapping lease is still not a
    // persistent mapping contract, so that stronger feature stays closed.
    facts.record_feature(OptionalFeature::MappablePrimaryBuffers);
    facts.record_feature(OptionalFeature::CoherentMapping);
    facts.record_limit(LimitKey::MapOffsetAlignment, 1);
    facts.record_limit(LimitKey::MapSizeAlignment, 1);

    facts.record_limit(LimitKey::MaxBufferSize, limits.max_buffer_size);
    // Metal's direct buffer arguments bind a range of an MTLBuffer; the exact
    // range ceiling is therefore the selected device's max buffer length, not
    // an invented uniform/storage sub-limit.
    facts.record_limit(
        LimitKey::MaxUniformBufferBindingSize,
        limits.max_buffer_size,
    );
    facts.record_limit(
        LimitKey::MaxStorageBufferBindingSize,
        limits.max_buffer_size,
    );
    facts.record_limit(
        LimitKey::MaxTexture1dDimension,
        u64::from(limits.max_texture_1d),
    );
    facts.record_limit(
        LimitKey::MaxTexture2dDimension,
        u64::from(limits.max_texture_2d),
    );
    facts.record_limit(
        LimitKey::MaxTexture3dDimension,
        u64::from(limits.max_texture_3d),
    );
    facts.record_limit(
        LimitKey::MaxTextureArrayLayers,
        u64::from(limits.max_array_layers),
    );

    record_buffer_support(&mut facts, limits.max_buffer_size);
    record_formats(&mut facts, limits);
    if limits.texture_creation {
        record_texture_support(&mut facts, limits);
    }
    if limits.texture_copy_lowering {
        record_copy_routes(&mut facts, limits);
    }
    // ClearTexture is deliberately coupled to the buffer-to-texture route:
    // color and compressed formats clear through an explicitly zeroed staging
    // footprint, while depth/stencil uses the separately validated render-pass
    // route. Do not publish the feature on a device where only texture-to-
    // texture copies happened to probe successfully.
    if limits.texture_creation && limits.buffer_texture_copy_lowering {
        facts.record_feature(OptionalFeature::ClearTexture);
    }
    facts
}

/// Builds the adapter snapshot from the native `MTLDevice`.
///
/// The two direct calls are intentional probes rather than guessed platform
/// constants: `maxBufferLength` is device-specific, and querying sample count
/// one verifies the native texture path before any texture fact is published.
/// Family predicates gate compressed formats because their enum values exist in
/// the SDK even when the selected GPU cannot create them. BC gets a dedicated
/// selector probe (rather than the deprecated `Mac1` family); ETC/EAC, ASTC LDR,
/// ASTC HDR, and volume ASTC each have their documented family gate. The copy
/// route is published independently, so a format never acquires an upload path
/// merely by being a native pixel-format spelling.
pub(super) fn baseline(device: &ProtocolObject<dyn MTLDevice>) -> CapabilityFacts {
    let apple2 = supports_family(device, MTLGPUFamily::Apple2);
    let apple3 = supports_family(device, MTLGPUFamily::Apple3);
    let apple6 = supports_family(device, MTLGPUFamily::Apple6);
    let apple7 = supports_family(device, MTLGPUFamily::Apple7);
    let mac2 = supports_family(device, MTLGPUFamily::Mac2);
    let texture_path = device.supportsTextureSampleCount(1);
    let max_2d = if supports_family(device, MTLGPUFamily::Apple10) {
        32_768
    } else if apple2 || mac2 {
        16_384
    } else {
        8_192
    };

    probe(MetalCapabilityLimits {
        max_buffer_size: device.maxBufferLength() as u64,
        max_texture_1d: max_2d,
        max_texture_2d: max_2d,
        max_texture_3d: 2_048,
        max_array_layers: 2_048,
        max_mip_levels: 15,
        // Metal's sampler descriptor accepts the standard 1..=16 interval.
        // A future backend policy that requests less must change this probe,
        // not silently clamp after advertising 16.
        max_sampler_anisotropy: Some(16),
        bc: supports_bc_texture_compression(device),
        // ETC2/EAC is part of the mobile Apple platform profile. macOS exposes
        // it only on Apple-family GPUs new enough to guarantee it; Intel/AMD
        // Macs must not inherit support merely because the SDK has the enums.
        etc2_eac: !cfg!(target_os = "macos") || apple7,
        astc_ldr: apple2,
        astc_hdr: apple6,
        astc_3d: apple3,
        // Keep the mobile profiles closed until their exact deployment-target
        // availability is represented in the probe rather than inferred from
        // an SDK enum value.
        texture_1d: cfg!(target_os = "macos"),
        texture_cube_array: cfg!(target_os = "macos")
            || supports_family(device, MTLGPUFamily::Apple4),
        depth16_unorm: cfg!(target_os = "macos"),
        rgba32_float_blend: cfg!(target_os = "macos"),
        occlusion_queries: true,
        indirect_commands: mac2 || apple3,
        base_vertex_instance: supports_base_vertex_instance(device),
        // Depth clipping is a baseline macOS capability and starts at Apple4
        // on mobile-family GPUs.
        depth_clip_control: supports_depth_clip_control(device),
        read_write_texture_tier: read_write_texture_tier(device),
        // Direct vertex/fragment texture arguments exist throughout the Metal
        // deployment floor used here. Read-write access remains independently
        // restricted by `read_write_texture_tier` and exact format rows.
        raster_storage_textures: true,
        sample_count_mask: [1_u32, 2, 4, 8, 16]
            .into_iter()
            .filter(|count| device.supportsTextureSampleCount(*count as usize))
            .fold(0_u32, |mask, count| mask | (1_u32 << count)),
        max_vertex_amplification_count: max_vertex_amplification_count(device),
        comparison_samplers: true,
        // Apple documents this through older feature sets. Mac2 is the
        // non-deprecated family predicate exposed by this SDK and is a
        // conservative subset of the documented desktop support.
        sampler_clamp_to_border: cfg!(target_os = "macos") && mac2,
        storage_textures: texture_path,
        // Pipeline objects, direct binding packets, and both compute/raster
        // command encoders are implemented by the submission lowerer.
        compute_lowering: true,
        compute_binding_lowering: true,
        raster_lowering: true,
        raster_binding_lowering: true,
        texture_creation: texture_path,
        texture_copy_lowering: texture_path,
        buffer_texture_copy_lowering: texture_path,
        // Apple's feature-set tables give 256 bytes on macOS and visionOS;
        // Apple3 has 16 bytes, while earlier supported Apple families require
        // 64. This is a route layout constraint, not a resource alignment.
        texel_copy_buffer_offset_alignment: if cfg!(any(
            target_os = "macos",
            target_os = "visionos"
        )) {
            256
        } else if apple3 {
            16
        } else {
            64
        },
        // The older supported family floor is four render targets; later
        // devices can expose eight, but a fact must be a lower bound proven
        // for this selected family, not a hopeful maximum.
        max_color_attachments: if apple2 || mac2 { 8 } else { 4 },
        // 16 bytes/sample is the documented oldest Metal family floor. The
        // higher family-specific values are an optimization opportunity, not
        // necessary to make the raster contract correct.
        max_color_attachment_bytes_per_sample: 16,
        max_inter_stage_shader_variables: if supports_family(device, MTLGPUFamily::Apple4) {
            31
        } else if cfg!(target_os = "macos") {
            30
        } else {
            15
        },
    })
}

/// Direct Metal arguments have independent buffer, texture, and sampler index
/// spaces, but Fluxel's portable limits are per class.  The ABI has no argument
/// buffer implementation yet, so this baseline reserves one index in each of
/// the two shared namespaces: 15 uniform + 15 storage buffers cannot exceed
/// Metal's 31 buffer arguments, and the analogous texture split cannot exceed
/// the portable direct-argument floor of 31.  The unused slot is intentional;
/// publishing 31 for each portable class would permit a mixed interface that
/// aliases native indices.
fn record_compute_binding_support(facts: &mut CapabilityFacts, limits: MetalCapabilityLimits) {
    const DIRECT_BUFFER_CLASS_LIMIT: u32 = 15;
    const DIRECT_TEXTURE_CLASS_LIMIT: u32 = 15;
    const DIRECT_SAMPLER_LIMIT: u32 = 16;

    facts.record_feature(OptionalFeature::BindingArrays);
    facts.record_limit(
        LimitKey::MaxBindingArrayElementsPerShaderStage,
        u64::from(DIRECT_TEXTURE_CLASS_LIMIT),
    );
    facts.record_limit(
        LimitKey::MaxBindingArraySamplerElementsPerShaderStage,
        u64::from(DIRECT_SAMPLER_LIMIT),
    );
    facts.record_binding_limit(
        crate::api::shader::ShaderStage::Compute,
        BindingLimitClass::UniformBuffers,
        DIRECT_BUFFER_CLASS_LIMIT,
    );
    facts.record_binding_limit(
        crate::api::shader::ShaderStage::Compute,
        BindingLimitClass::StorageBuffers,
        DIRECT_BUFFER_CLASS_LIMIT,
    );
    facts.record_binding_limit(
        crate::api::shader::ShaderStage::Compute,
        BindingLimitClass::SampledTextures,
        DIRECT_TEXTURE_CLASS_LIMIT,
    );
    facts.record_binding_limit(
        crate::api::shader::ShaderStage::Compute,
        BindingLimitClass::StorageTextures,
        DIRECT_TEXTURE_CLASS_LIMIT,
    );
    facts.record_binding_limit(
        crate::api::shader::ShaderStage::Compute,
        BindingLimitClass::Samplers,
        DIRECT_SAMPLER_LIMIT,
    );
    facts.record_limit(
        LimitKey::MaxDynamicUniformBuffersPerPipelineLayout,
        u64::from(DIRECT_BUFFER_CLASS_LIMIT),
    );
    facts.record_limit(
        LimitKey::MaxDynamicStorageBuffersPerPipelineLayout,
        u64::from(DIRECT_BUFFER_CLASS_LIMIT),
    );

    // Arrays are fixed spans in the direct ABI. Runtime-sized arrays,
    // acceleration structures, and external textures deliberately get no
    // positive row, which makes their public query fail closed.
    for array in [false, true] {
        for dynamic_offset in [false, true] {
            record_compute_binding(facts, BindableKind::UniformBuffer, array, dynamic_offset);
            for access in [
                BufferBindingAccess::ReadOnly,
                BufferBindingAccess::ReadWrite,
            ] {
                record_compute_binding(
                    facts,
                    BindableKind::StorageBuffer { access },
                    array,
                    dynamic_offset,
                );
            }
        }
        for kind in [
            SamplerKind::Filtering,
            SamplerKind::NonFiltering,
            SamplerKind::Comparison,
        ] {
            record_compute_binding(facts, BindableKind::Sampler { kind }, array, false);
        }
        for dimension in [
            TextureViewDimension::D1,
            TextureViewDimension::D2,
            TextureViewDimension::D2Array,
            TextureViewDimension::Cube,
            TextureViewDimension::CubeArray,
            TextureViewDimension::D3,
        ] {
            if !view_dimension_is_published(dimension, limits) {
                continue;
            }
            for sample_type in [
                TextureSampleType::Float,
                TextureSampleType::UnfilterableFloat,
                TextureSampleType::Sint,
                TextureSampleType::Uint,
                TextureSampleType::Depth,
            ] {
                record_compute_binding(
                    facts,
                    BindableKind::SampledTexture {
                        dimension,
                        sample_type,
                        multisampled: false,
                    },
                    array,
                    false,
                );
            }
        }
        for format in TextureFormat::all() {
            if !format_is_published(format, limits) || !storage_capable(format) {
                continue;
            }
            for dimension in [
                TextureViewDimension::D1,
                TextureViewDimension::D2,
                TextureViewDimension::D2Array,
                TextureViewDimension::D3,
            ] {
                for access in storage_accesses(format, limits) {
                    record_compute_binding(
                        facts,
                        BindableKind::StorageTexture {
                            dimension,
                            format,
                            access,
                        },
                        array,
                        false,
                    );
                }
            }
        }
    }
}

/// Publishes raster-stage bindings only after the render encoder applies the
/// exact same direct ABI as compute. Vertex and fragment have independent
/// native namespaces. The vertex buffer input consumes its buffer namespace,
/// hence the deliberately smaller vertex buffer-class budgets documented at
/// the raster limits above.
fn record_raster_binding_support(facts: &mut CapabilityFacts, limits: MetalCapabilityLimits) {
    const VERTEX_BUFFER_CLASS_LIMIT: u32 = 7;
    const FRAGMENT_BUFFER_CLASS_LIMIT: u32 = 15;
    const DIRECT_TEXTURE_CLASS_LIMIT: u32 = 15;
    const DIRECT_SAMPLER_LIMIT: u32 = 16;

    facts.record_feature(OptionalFeature::BindingArrays);
    // This device-wide key cannot vary by stage. Choose the vertex-safe
    // bound: a fixed array admitted at every declared visibility never makes
    // the vertex stream plus direct resource ABI exceed its 31 buffer slots.
    facts.record_limit(LimitKey::MaxBindingArrayElementsPerShaderStage, 7);
    facts.record_limit(
        LimitKey::MaxBindingArraySamplerElementsPerShaderStage,
        u64::from(DIRECT_SAMPLER_LIMIT),
    );
    for (stage, buffer_limit) in [
        (
            crate::api::shader::ShaderStage::Vertex,
            VERTEX_BUFFER_CLASS_LIMIT,
        ),
        (
            crate::api::shader::ShaderStage::Fragment,
            FRAGMENT_BUFFER_CLASS_LIMIT,
        ),
    ] {
        facts.record_binding_limit(stage, BindingLimitClass::UniformBuffers, buffer_limit);
        facts.record_binding_limit(stage, BindingLimitClass::StorageBuffers, buffer_limit);
        facts.record_binding_limit(
            stage,
            BindingLimitClass::SampledTextures,
            DIRECT_TEXTURE_CLASS_LIMIT,
        );
        facts.record_binding_limit(
            stage,
            BindingLimitClass::StorageTextures,
            if limits.raster_storage_textures {
                DIRECT_TEXTURE_CLASS_LIMIT
            } else {
                0
            },
        );
        facts.record_binding_limit(stage, BindingLimitClass::Samplers, DIRECT_SAMPLER_LIMIT);
    }
    // Dynamic-offset limits are pipeline-wide rather than stage-specific;
    // retain the vertex-safe limit for an interface visible to both stages.
    facts.record_limit(LimitKey::MaxDynamicUniformBuffersPerPipelineLayout, 7);
    facts.record_limit(LimitKey::MaxDynamicStorageBuffersPerPipelineLayout, 7);

    for visibility in [
        ShaderStages::VERTEX,
        ShaderStages::FRAGMENT,
        ShaderStages::VERTEX.union(ShaderStages::FRAGMENT),
    ] {
        record_direct_binding_support(facts, limits, visibility, limits.raster_storage_textures);
    }
}

fn record_direct_binding_support(
    facts: &mut CapabilityFacts,
    limits: MetalCapabilityLimits,
    visibility: ShaderStages,
    include_storage_textures: bool,
) {
    for array in [false, true] {
        for dynamic_offset in [false, true] {
            record_direct_binding(
                facts,
                visibility,
                BindableKind::UniformBuffer,
                array,
                dynamic_offset,
            );
            for access in [
                BufferBindingAccess::ReadOnly,
                BufferBindingAccess::ReadWrite,
            ] {
                record_direct_binding(
                    facts,
                    visibility,
                    BindableKind::StorageBuffer { access },
                    array,
                    dynamic_offset,
                );
            }
        }
        for kind in [
            SamplerKind::Filtering,
            SamplerKind::NonFiltering,
            SamplerKind::Comparison,
        ] {
            record_direct_binding(
                facts,
                visibility,
                BindableKind::Sampler { kind },
                array,
                false,
            );
        }
        for dimension in [
            TextureViewDimension::D1,
            TextureViewDimension::D2,
            TextureViewDimension::D2Array,
            TextureViewDimension::Cube,
            TextureViewDimension::CubeArray,
            TextureViewDimension::D3,
        ] {
            if !view_dimension_is_published(dimension, limits) {
                continue;
            }
            for sample_type in [
                TextureSampleType::Float,
                TextureSampleType::UnfilterableFloat,
                TextureSampleType::Sint,
                TextureSampleType::Uint,
                TextureSampleType::Depth,
            ] {
                record_direct_binding(
                    facts,
                    visibility,
                    BindableKind::SampledTexture {
                        dimension,
                        sample_type,
                        multisampled: false,
                    },
                    array,
                    false,
                );
            }
        }
        if include_storage_textures {
            for format in TextureFormat::all() {
                if !format_is_published(format, limits) || !storage_capable(format) {
                    continue;
                }
                for dimension in [
                    TextureViewDimension::D1,
                    TextureViewDimension::D2,
                    TextureViewDimension::D2Array,
                    TextureViewDimension::D3,
                ] {
                    for access in storage_accesses(format, limits) {
                        record_direct_binding(
                            facts,
                            visibility,
                            BindableKind::StorageTexture {
                                dimension,
                                format,
                                access,
                            },
                            array,
                            false,
                        );
                    }
                }
            }
        }
    }
}

fn record_direct_binding(
    facts: &mut CapabilityFacts,
    visibility: ShaderStages,
    kind: BindableKind,
    array: bool,
    dynamic_offset: bool,
) {
    facts.record_binding_support(
        BindingSupportKey {
            visibility,
            kind,
            array,
            runtime_sized: false,
            dynamic_offset,
        },
        BindingSupport::Supported,
    );
}

fn record_compute_binding(
    facts: &mut CapabilityFacts,
    kind: BindableKind,
    array: bool,
    dynamic_offset: bool,
) {
    facts.record_binding_support(
        BindingSupportKey {
            visibility: ShaderStages::COMPUTE,
            kind,
            array,
            runtime_sized: false,
            dynamic_offset,
        },
        BindingSupport::Supported,
    );
}

/// `supportsBCTextureCompression` was added later than the baseline Metal
/// device protocol. A selector check keeps an old OS or proxy capture device
/// from receiving an unknown Objective-C message; only then is its truthful
/// device-specific answer used. macOS's documented desktop baseline supports
/// BC, so it remains a positive answer when that optional selector is absent.
fn supports_bc_texture_compression(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    if cfg!(target_os = "macos") {
        return true;
    }
    let responds =
        AnyObject::class(device.as_ref()).responds_to(sel!(supportsBCTextureCompression));
    responds && device.supportsBCTextureCompression()
}

fn read_write_texture_tier(device: &ProtocolObject<dyn MTLDevice>) -> u8 {
    let responds = AnyObject::class(device.as_ref()).responds_to(sel!(readWriteTextureSupport));
    if !responds {
        return 0;
    }
    match device.readWriteTextureSupport() {
        MTLReadWriteTextureTier::Tier1 => 1,
        MTLReadWriteTextureTier::Tier2 => 2,
        _ => 0,
    }
}

pub(super) fn supports_depth_clip_control(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    // Metal documents depth-clip mode on every macOS GPU family and Apple4+
    // mobile GPUs. Keeping this predicate shared with pipeline construction is
    // what makes resetting Clamp back to Clip safe on older runtimes.
    cfg!(target_os = "macos") || supports_family(device, MTLGPUFamily::Apple4)
}

pub(super) fn supports_base_vertex_instance(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    // Keep the extended direct-draw selectors behind the same conservative
    // family floor used by the portable capability projection.  The ordinary
    // selectors remain a correct zero-base fallback below this floor.
    supports_family(device, MTLGPUFamily::Mac2) || supports_family(device, MTLGPUFamily::Apple3)
}

fn supports_family(device: &ProtocolObject<dyn MTLDevice>, family: MTLGPUFamily) -> bool {
    // `supportsFamily:` is newer than the oldest Apple deployment targets that
    // can compile this backend. SDK enum availability is not runtime method
    // availability, so every family answer goes through this one guarded seam.
    AnyObject::class(device.as_ref()).responds_to(sel!(supportsFamily:))
        && device.supportsFamily(family)
}

fn max_vertex_amplification_count(device: &ProtocolObject<dyn MTLDevice>) -> u32 {
    // Metal documents powers-of-two family limits. Query every next boundary
    // instead of inferring a family table, and cap at the public 32-bit mask.
    if !AnyObject::class(device.as_ref()).responds_to(sel!(supportsVertexAmplificationCount:)) {
        return 1;
    }
    let mut count = 1_u32;
    while count < 32 && device.supportsVertexAmplificationCount((count * 2) as usize) {
        count *= 2;
    }
    count
}

fn record_buffer_support(facts: &mut CapabilityFacts, max_size: u64) {
    for usage in BufferUsage::all() {
        // Ray-tracing allocation is a separate unfinished lowering, not a
        // synonym for normal storage. Keep every one of its usage bits closed.
        let unsupported = usage.is_empty()
            || usage.contains(BufferUsage::BLAS_INPUT)
            || usage.contains(BufferUsage::TLAS_INPUT)
            || usage.contains(BufferUsage::ACCELERATION_STRUCTURE_SCRATCH);
        facts.record_buffer_support(
            usage,
            if unsupported || max_size == 0 {
                BufferSupport::Unsupported
            } else {
                BufferSupport::Supported(BufferSupportLimits::new(max_size))
            },
        );
    }
}

fn record_formats(facts: &mut CapabilityFacts, limits: MetalCapabilityLimits) {
    for format in TextureFormat::all() {
        if !format_is_published(format, limits) {
            continue;
        }
        let depth = matches!(
            format,
            TextureFormat::Depth16Unorm
                | TextureFormat::Depth32Float
                | TextureFormat::Depth32FloatStencil8
        );
        let stencil = matches!(
            format,
            TextureFormat::Stencil8 | TextureFormat::Depth32FloatStencil8
        );
        let color = color_attachment_capable(format, limits);
        let storage = limits.storage_textures && storage_capable(format);
        let storage_read_write = storage
            && storage_accesses(format, limits).any(|access| access == StorageAccess::ReadWrite);
        facts.record_format(
            format,
            FormatFacts::new(
                format,
                StorageAccessSupport::new(storage, storage, storage_read_write),
                color,
                depth,
                stencil,
                blendable(format, limits),
            )
            .with_sampling_and_atomic(filterable(format), false),
        );
    }
}

fn record_texture_support(facts: &mut CapabilityFacts, limits: MetalCapabilityLimits) {
    for format in TextureFormat::all() {
        if !format_is_published(format, limits) {
            continue;
        }
        for dimension in [
            TextureDimension::D1,
            TextureDimension::D2,
            TextureDimension::D3,
        ] {
            if !texture_dimension_is_published(dimension, limits)
                || !compressed_dimension_is_lowered(format, dimension, limits)
            {
                continue;
            }
            for usage in TextureUsage::all() {
                for compatibility in [
                    TextureViewCompatibility::NONE,
                    TextureViewCompatibility::CUBE,
                ] {
                    for sample_count in [1_u32, 2, 4, 8, 16] {
                        if !supports_sample_count(limits, sample_count)
                            || !texture_key_is_lowered(
                                format,
                                dimension,
                                usage,
                                compatibility,
                                limits,
                            )
                            || (sample_count > 1
                                && (dimension != TextureDimension::D2
                                    || !multisample_usage_is_lowered(format, usage, limits)
                                    || compatibility != TextureViewCompatibility::NONE))
                        {
                            continue;
                        }
                        facts.record_texture_support(
                            &TextureSupportQuery::new(dimension, format, usage, sample_count)
                                .with_view_compatibility(compatibility),
                            TextureSupport::Supported(TextureSupportLimits::new(
                                extent_limit(dimension, limits),
                                if sample_count == 1 {
                                    limits.max_mip_levels
                                } else {
                                    1
                                },
                                if dimension == TextureDimension::D3 || sample_count > 1 {
                                    1
                                } else {
                                    limits.max_array_layers
                                },
                            )),
                        );
                    }
                }
            }
        }
    }
}

fn supports_sample_count(limits: MetalCapabilityLimits, count: u32) -> bool {
    count < u32::BITS && limits.sample_count_mask & (1_u32 << count) != 0
}

fn multisample_usage_is_lowered(
    format: TextureFormat,
    usage: TextureUsage,
    limits: MetalCapabilityLimits,
) -> bool {
    if metal_msaa_attachment_capable(format, limits) {
        let allowed = if metal_compute_resolve_capable(format, limits) {
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC)
        } else {
            TextureUsage::COLOR_ATTACHMENT
        };
        return !usage.is_empty() && allowed.contains(usage);
    }
    matches!(
        format,
        TextureFormat::Depth16Unorm
            | TextureFormat::Depth32Float
            | TextureFormat::Stencil8
            | TextureFormat::Depth32FloatStencil8
    ) && usage == TextureUsage::DEPTH_STENCIL_ATTACHMENT
}

fn metal_compute_resolve_capable(format: TextureFormat, limits: MetalCapabilityLimits) -> bool {
    metal_msaa_attachment_capable(format, limits)
        && matches!(
            format,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba16Float
        )
}

/// Formats for which this baseline can honor the complete public color-MSAA
/// contract. Integer formats are deliberately excluded: neither the native
/// attachment resolve nor the standalone compute fallback has a portable
/// integer-average semantic.
fn metal_msaa_attachment_capable(format: TextureFormat, limits: MetalCapabilityLimits) -> bool {
    color_attachment_capable(format, limits)
        && !integer_format(format)
        && !matches!(
            format,
            TextureFormat::R8Snorm
                | TextureFormat::Rg8Snorm
                | TextureFormat::Rgba8Snorm
                | TextureFormat::R16Snorm
                | TextureFormat::Rg16Snorm
                | TextureFormat::Rgba16Snorm
                | TextureFormat::R32Float
                | TextureFormat::Rg32Float
                | TextureFormat::Rgba32Float
                | TextureFormat::Rgb9e5Ufloat
        )
}

/// Records only copy routes the current command spine actually encodes.
///
/// The encoder lowers `CopyRecord::Buffer` and `CopyRecord::Texture`, including
/// per-array-image buffer strides. It does not select a depth/stencil plane, so
/// those aspects remain absent rather than silently copying the wrong bytes.
/// `MTLBlitCommandEncoder::copyFromTexture` is used only for same-format,
/// same-dimension, single-sampled colour textures. Standalone resolve uses a
/// private compute kernel rather than Metal's attachment-only resolve action;
/// filtered blit remains absent because it has no lowering.
fn record_copy_routes(facts: &mut CapabilityFacts, limits: MetalCapabilityLimits) {
    // Metal buffer-to-buffer copies take byte offsets and sizes. The backend
    // validation/lowering uses no stricter alignment than one byte.
    facts.record_route(
        RouteQuery::BufferToBuffer,
        RouteSupport::Supported(RouteCapabilities::new(None, None)),
    );

    for format in TextureFormat::all() {
        if !format_is_published(format, limits)
            || !format_aspects(format).contains(crate::api::resource::TextureAspects::COLOR)
        {
            continue;
        }
        // The compute resolver handles every region, destination mip and array
        // layer accepted by the public validator. Keep the exact same format /
        // sample gate as MSAA texture creation so a route is never more
        // optimistic than native allocation.
        if metal_compute_resolve_capable(format, limits) {
            for sample_count in [2_u32, 4, 8, 16] {
                if supports_sample_count(limits, sample_count) {
                    facts.record_route(
                        RouteQuery::Resolve {
                            format,
                            src_sample_count: sample_count,
                        },
                        RouteSupport::Supported(RouteCapabilities::new(None, None)),
                    );
                }
            }
        }
        for dimension in [
            TextureDimension::D1,
            TextureDimension::D2,
            TextureDimension::D3,
        ] {
            if !texture_dimension_is_published(dimension, limits)
                || !compressed_dimension_is_lowered(format, dimension, limits)
            {
                continue;
            }
            facts.record_route(
                RouteQuery::TextureToTexture {
                    src_dimension: dimension,
                    src_format: format,
                    src_aspect: TextureAspect::Color,
                    src_sample_count: 1,
                    dst_dimension: dimension,
                    dst_format: format,
                    dst_aspect: TextureAspect::Color,
                    dst_sample_count: 1,
                },
                RouteSupport::Supported(RouteCapabilities::new(None, None)),
            );
            if limits.buffer_texture_copy_lowering {
                // The portable validator establishes the block geometry and
                // the complete buffer footprint.  The native lowerer advances
                // this exact `bytes_per_row * rows_per_image` stride for every
                // array image, while a 3D copy is one native footprint whose
                // depth is carried by MTLSize. No extra image-stride alignment
                // or tightly-packed-3D rule is imposed by this path.
                let layout = Some(
                    TexelCopyLayoutLimits::new(limits.texel_copy_buffer_offset_alignment, 4)
                        .with_image_layout(1, false),
                );
                facts.record_route(
                    RouteQuery::BufferToTexture {
                        dimension,
                        format,
                        aspect: TextureAspect::Color,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(None, layout)),
                );
                facts.record_route(
                    RouteQuery::TextureToBuffer {
                        dimension,
                        format,
                        aspect: TextureAspect::Color,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(None, layout)),
                );
            }
        }
    }
}

fn format_is_published(format: TextureFormat, limits: MetalCapabilityLimits) -> bool {
    if metal_format(format).is_none() {
        return false;
    }
    if format == TextureFormat::Depth16Unorm && !limits.depth16_unorm {
        return false;
    }
    if matches!(
        format,
        TextureFormat::R16Unorm
            | TextureFormat::R16Snorm
            | TextureFormat::Rg16Unorm
            | TextureFormat::Rg16Snorm
            | TextureFormat::Rgba16Unorm
            | TextureFormat::Rgba16Snorm
    ) && !cfg!(target_os = "macos")
    {
        return false;
    }
    match compressed_family(format) {
        None => true,
        Some(CompressedFamily::Bc) => limits.bc,
        Some(CompressedFamily::Etc2Eac) => limits.etc2_eac,
        Some(CompressedFamily::AstcLdr) => limits.astc_ldr,
        Some(CompressedFamily::AstcHdr) => limits.astc_hdr,
    }
}

fn texture_dimension_is_published(
    dimension: TextureDimension,
    limits: MetalCapabilityLimits,
) -> bool {
    dimension != TextureDimension::D1 || limits.texture_1d
}

fn view_dimension_is_published(
    dimension: TextureViewDimension,
    limits: MetalCapabilityLimits,
) -> bool {
    (dimension != TextureViewDimension::D1 || limits.texture_1d)
        && (dimension != TextureViewDimension::CubeArray || limits.texture_cube_array)
}

fn color_attachment_capable(format: TextureFormat, _limits: MetalCapabilityLimits) -> bool {
    if compressed_family(format).is_some()
        || matches!(
            format,
            TextureFormat::Depth16Unorm
                | TextureFormat::Depth32Float
                | TextureFormat::Depth32FloatStencil8
                | TextureFormat::Stencil8
        )
    {
        return false;
    }
    // Desktop Metal only guarantees RGB9E5 sampling. Mobile Apple families
    // can render to it, but that route needs a family/OS probe before exposure.
    format != TextureFormat::Rgb9e5Ufloat || !cfg!(target_os = "macos")
}

fn blendable(format: TextureFormat, limits: MetalCapabilityLimits) -> bool {
    color_attachment_capable(format, limits)
        && !integer_format(format)
        && (format != TextureFormat::Rgba32Float || limits.rgba32_float_blend)
}

/// Metal's compressed 3D textures are ASTC-specific and start at Apple3.
/// Keeping BC and ETC/EAC at D1/D2 avoids treating a pixel-format availability
/// fact as proof of a volume-texture upload/copy contract.
fn compressed_dimension_is_lowered(
    format: TextureFormat,
    dimension: TextureDimension,
    limits: MetalCapabilityLimits,
) -> bool {
    if dimension != TextureDimension::D3 {
        return true;
    }
    matches!(
        compressed_family(format),
        Some(CompressedFamily::AstcLdr | CompressedFamily::AstcHdr)
    ) && limits.astc_3d
}

fn texture_key_is_lowered(
    format: TextureFormat,
    dimension: TextureDimension,
    usage: TextureUsage,
    compatibility: TextureViewCompatibility,
    limits: MetalCapabilityLimits,
) -> bool {
    if usage.is_empty()
        || (compatibility == TextureViewCompatibility::CUBE && dimension != TextureDimension::D2)
    {
        return false;
    }
    if usage.contains(TextureUsage::COLOR_ATTACHMENT) && !color_attachment_capable(format, limits) {
        return false;
    }
    let depth_or_stencil = matches!(
        format,
        TextureFormat::Depth16Unorm
            | TextureFormat::Depth32Float
            | TextureFormat::Depth32FloatStencil8
            | TextureFormat::Stencil8
    );
    // Metal render attachments are 2D (or a 2D-array slice). Keeping depth
    // and stencil formats out of D1/D3 facts prevents the clear and raster
    // routes from advertising a shape their native attachment descriptors
    // cannot encode.
    if depth_or_stencil && dimension != TextureDimension::D2 {
        return false;
    }
    if depth_or_stencil
        && (usage.contains(TextureUsage::COLOR_ATTACHMENT) || usage.contains(TextureUsage::STORAGE))
    {
        return false;
    }
    if !depth_or_stencil && usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        return false;
    }
    if compressed_family(format).is_some()
        && (usage.contains(TextureUsage::COLOR_ATTACHMENT)
            || usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
            || usage.contains(TextureUsage::STORAGE))
    {
        return false;
    }
    !usage.contains(TextureUsage::STORAGE) || (limits.storage_textures && storage_capable(format))
}

fn extent_limit(dimension: TextureDimension, limits: MetalCapabilityLimits) -> Extent3d {
    match dimension {
        TextureDimension::D1 => Extent3d::d1(limits.max_texture_1d),
        TextureDimension::D2 => Extent3d::d2(limits.max_texture_2d, limits.max_texture_2d),
        TextureDimension::D3 => Extent3d::d3(
            limits.max_texture_3d,
            limits.max_texture_3d,
            limits.max_texture_3d,
        ),
    }
}

fn integer_format(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::R8Uint
            | TextureFormat::R8Sint
            | TextureFormat::Rg8Uint
            | TextureFormat::Rg8Sint
            | TextureFormat::Rgba8Uint
            | TextureFormat::Rgba8Sint
            | TextureFormat::R16Uint
            | TextureFormat::R16Sint
            | TextureFormat::Rg16Uint
            | TextureFormat::Rg16Sint
            | TextureFormat::Rgba16Uint
            | TextureFormat::Rgba16Sint
            | TextureFormat::R32Uint
            | TextureFormat::R32Sint
            | TextureFormat::Rg32Uint
            | TextureFormat::Rg32Sint
            | TextureFormat::Rgba32Uint
            | TextureFormat::Rgba32Sint
            | TextureFormat::Rgb10a2Uint
    )
}

fn filterable(format: TextureFormat) -> bool {
    !integer_format(format)
        && !matches!(
            format,
            TextureFormat::R32Float
                | TextureFormat::Rg32Float
                | TextureFormat::Rgba32Float
                | TextureFormat::Depth16Unorm
                | TextureFormat::Depth32Float
                | TextureFormat::Depth32FloatStencil8
                | TextureFormat::Stencil8
        )
}

// This is intentionally a smaller set than the formats Metal can potentially
// expose as write textures. A format enters only once the MSL type, binding
// validation, and command encoder use all have a conformance case.
fn storage_capable(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::R32Uint
            | TextureFormat::R32Sint
            | TextureFormat::R32Float
            | TextureFormat::Rg32Uint
            | TextureFormat::Rg32Sint
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba8Uint
            | TextureFormat::Rgba8Sint
            | TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba16Uint
            | TextureFormat::Rgba16Sint
            | TextureFormat::Rgba16Float
            | TextureFormat::Rgba32Uint
            | TextureFormat::Rgba32Sint
            | TextureFormat::Rgba32Float
    )
}

fn storage_accesses(
    format: TextureFormat,
    limits: MetalCapabilityLimits,
) -> impl Iterator<Item = StorageAccess> {
    let read_write = match limits.read_write_texture_tier {
        0 => false,
        1 => matches!(
            format,
            TextureFormat::R32Uint | TextureFormat::R32Sint | TextureFormat::R32Float
        ),
        _ => matches!(
            format,
            TextureFormat::R32Uint
                | TextureFormat::R32Sint
                | TextureFormat::R32Float
                | TextureFormat::Rgba8Uint
                | TextureFormat::Rgba8Sint
                | TextureFormat::Rgba8Unorm
                | TextureFormat::Rgba16Uint
                | TextureFormat::Rgba16Sint
                | TextureFormat::Rgba16Float
                | TextureFormat::Rgba32Uint
                | TextureFormat::Rgba32Sint
                | TextureFormat::Rgba32Float
        ),
    };
    [
        Some(StorageAccess::ReadOnly),
        Some(StorageAccess::WriteOnly),
        read_write.then_some(StorageAccess::ReadWrite),
    ]
    .into_iter()
    .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_limits() -> MetalCapabilityLimits {
        MetalCapabilityLimits {
            max_buffer_size: 1,
            max_texture_1d: 1,
            max_texture_2d: 64,
            max_texture_3d: 1,
            max_array_layers: 1,
            max_mip_levels: 1,
            max_sampler_anisotropy: None,
            bc: false,
            etc2_eac: false,
            astc_ldr: false,
            astc_hdr: false,
            astc_3d: false,
            texture_1d: false,
            texture_cube_array: false,
            depth16_unorm: false,
            rgba32_float_blend: false,
            occlusion_queries: false,
            indirect_commands: false,
            base_vertex_instance: false,
            depth_clip_control: false,
            read_write_texture_tier: 0,
            raster_storage_textures: false,
            sample_count_mask: 1 << 1,
            max_vertex_amplification_count: 1,
            comparison_samplers: false,
            sampler_clamp_to_border: false,
            storage_textures: false,
            compute_lowering: false,
            compute_binding_lowering: false,
            raster_lowering: false,
            raster_binding_lowering: false,
            texture_creation: false,
            texture_copy_lowering: false,
            buffer_texture_copy_lowering: false,
            texel_copy_buffer_offset_alignment: 1,
            max_color_attachments: 1,
            max_color_attachment_bytes_per_sample: 4,
            max_inter_stage_shader_variables: 1,
        }
    }

    #[test]
    fn ordinary_feature_facts_open_only_after_their_exact_lowering_gates() {
        let closed =
            crate::api::capability::AvailableCapabilities::from_facts(probe(minimal_limits()));
        for feature in [
            OptionalFeature::OcclusionQuery,
            OptionalFeature::QueryResolve,
            OptionalFeature::IndirectDraw,
            OptionalFeature::IndirectDispatch,
            OptionalFeature::DepthClipControl,
            OptionalFeature::Multiview,
            OptionalFeature::SelectiveMultiview,
            OptionalFeature::MultisampledShading,
            OptionalFeature::ClearTexture,
        ] {
            assert!(!closed.supports_feature(feature));
        }

        let mut limits = minimal_limits();
        limits.occlusion_queries = true;
        limits.indirect_commands = true;
        limits.base_vertex_instance = true;
        limits.depth_clip_control = true;
        limits.compute_lowering = true;
        limits.raster_lowering = true;
        limits.texture_creation = true;
        limits.texture_copy_lowering = true;
        limits.buffer_texture_copy_lowering = true;
        limits.sample_count_mask |= 1 << 4;
        limits.max_vertex_amplification_count = 4;
        let open = crate::api::capability::AvailableCapabilities::from_facts(probe(limits));
        for feature in [
            OptionalFeature::OcclusionQuery,
            OptionalFeature::QueryResolve,
            OptionalFeature::IndirectDraw,
            OptionalFeature::IndirectDispatch,
            OptionalFeature::DepthClipControl,
            OptionalFeature::Multiview,
            OptionalFeature::SelectiveMultiview,
            OptionalFeature::MultisampledShading,
            OptionalFeature::ClearTexture,
        ] {
            assert!(open.supports_feature(feature), "missing {feature:?}");
        }
        assert_eq!(open.limit(LimitKey::MaxMultiviewViewCount), Some(4));
        assert!(
            open.texture_support(&TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT,
                4,
            ))
            .is_supported()
        );
        assert!(
            !open
                .texture_support(&TextureSupportQuery::new(
                    TextureDimension::D2,
                    TextureFormat::Rgba8Uint,
                    TextureUsage::COLOR_ATTACHMENT,
                    4,
                ))
                .is_supported(),
            "integer MSAA must stay closed because Metal cannot resolve it"
        );
        assert!(
            !open
                .texture_support(&TextureSupportQuery::new(
                    TextureDimension::D2,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::SAMPLED,
                    4,
                ))
                .is_supported(),
            "multisampled sampling has no published binding row yet"
        );
        assert!(
            open.route(&RouteQuery::Resolve {
                format: TextureFormat::Rgba8Unorm,
                src_sample_count: 4,
            })
            .is_supported()
        );
        assert!(
            !open
                .route(&RouteQuery::Resolve {
                    format: TextureFormat::Rgba8UnormSrgb,
                    src_sample_count: 4,
                })
                .is_supported(),
            "the compute resolver must not publish a non-writable sRGB target"
        );
    }

    #[test]
    fn storage_texture_access_is_tier_and_stage_gated() {
        let mut limits = minimal_limits();
        limits.storage_textures = true;
        limits.compute_binding_lowering = true;
        limits.raster_binding_lowering = true;
        limits.raster_storage_textures = true;
        limits.read_write_texture_tier = 1;
        let tier1 = crate::api::capability::AvailableCapabilities::from_facts(probe(limits));
        let query = |format, visibility| crate::api::binding::BindingSupportQuery {
            visibility,
            kind: crate::api::binding::BindingKind::StorageTexture {
                dimension: TextureViewDimension::D2,
                format,
                access: StorageAccess::ReadWrite,
            },
            count: crate::api::binding::BindingCount::One,
            dynamic_offset: false,
        };
        assert_eq!(
            tier1.binding_support(&query(TextureFormat::R32Uint, ShaderStages::COMPUTE)),
            BindingSupport::Supported
        );
        assert_eq!(
            tier1.binding_support(&query(TextureFormat::Rgba8Unorm, ShaderStages::FRAGMENT)),
            BindingSupport::Unsupported
        );

        limits.read_write_texture_tier = 2;
        let tier2 = crate::api::capability::AvailableCapabilities::from_facts(probe(limits));
        assert_eq!(
            tier2.binding_support(&query(TextureFormat::Rgba8Unorm, ShaderStages::FRAGMENT)),
            BindingSupport::Supported
        );
    }

    #[test]
    fn no_lowering_means_no_texture_creation_claim() {
        let facts = crate::api::capability::AvailableCapabilities::from_facts(probe(
            MetalCapabilityLimits {
                max_buffer_size: 1,
                max_texture_1d: 1,
                max_texture_2d: 1,
                max_texture_3d: 1,
                max_array_layers: 1,
                max_mip_levels: 1,
                max_sampler_anisotropy: None,
                bc: false,
                etc2_eac: false,
                astc_ldr: false,
                astc_hdr: false,
                astc_3d: false,
                texture_1d: false,
                texture_cube_array: false,
                depth16_unorm: false,
                rgba32_float_blend: false,
                occlusion_queries: false,
                indirect_commands: false,
                base_vertex_instance: false,
                depth_clip_control: false,
                read_write_texture_tier: 0,
                raster_storage_textures: false,
                sample_count_mask: 1 << 1,
                max_vertex_amplification_count: 1,
                comparison_samplers: false,
                sampler_clamp_to_border: false,
                storage_textures: false,
                compute_lowering: false,
                compute_binding_lowering: false,
                raster_lowering: false,
                raster_binding_lowering: false,
                texture_creation: false,
                texture_copy_lowering: false,
                buffer_texture_copy_lowering: false,
                texel_copy_buffer_offset_alignment: 1,
                max_color_attachments: 1,
                max_color_attachment_bytes_per_sample: 1,
                max_inter_stage_shader_variables: 1,
            },
        ));
        assert!(
            !facts
                .texture_support(&TextureSupportQuery::new(
                    TextureDimension::D2,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::SAMPLED,
                    1,
                ))
                .is_supported()
        );
    }

    #[test]
    fn compressed_family_is_explicitly_gated() {
        let mut limits = MetalCapabilityLimits {
            max_buffer_size: 1,
            max_texture_1d: 1,
            max_texture_2d: 1,
            max_texture_3d: 1,
            max_array_layers: 1,
            max_mip_levels: 1,
            max_sampler_anisotropy: None,
            bc: false,
            etc2_eac: false,
            astc_ldr: false,
            astc_hdr: false,
            astc_3d: false,
            texture_1d: false,
            texture_cube_array: false,
            depth16_unorm: false,
            rgba32_float_blend: false,
            occlusion_queries: false,
            indirect_commands: false,
            base_vertex_instance: false,
            depth_clip_control: false,
            read_write_texture_tier: 0,
            raster_storage_textures: false,
            sample_count_mask: 1 << 1,
            max_vertex_amplification_count: 1,
            comparison_samplers: false,
            sampler_clamp_to_border: false,
            storage_textures: false,
            compute_lowering: false,
            compute_binding_lowering: false,
            raster_lowering: false,
            raster_binding_lowering: false,
            texture_creation: true,
            texture_copy_lowering: true,
            buffer_texture_copy_lowering: false,
            texel_copy_buffer_offset_alignment: 1,
            max_color_attachments: 1,
            max_color_attachment_bytes_per_sample: 1,
            max_inter_stage_shader_variables: 1,
        };
        assert!(!format_is_published(TextureFormat::Bc1RgbaUnorm, limits));
        limits.bc = true;
        assert!(format_is_published(TextureFormat::Bc1RgbaUnorm, limits));

        // SDK enum spellings are not capability evidence for narrower Apple
        // profiles. Each gate has both a negative and positive boundary here.
        assert!(!format_is_published(TextureFormat::Depth16Unorm, limits));
        limits.depth16_unorm = true;
        assert!(format_is_published(TextureFormat::Depth16Unorm, limits));

        assert!(!texture_dimension_is_published(
            TextureDimension::D1,
            limits
        ));
        limits.texture_1d = true;
        assert!(texture_dimension_is_published(TextureDimension::D1, limits));

        assert!(!view_dimension_is_published(
            TextureViewDimension::CubeArray,
            limits
        ));
        limits.texture_cube_array = true;
        assert!(view_dimension_is_published(
            TextureViewDimension::CubeArray,
            limits
        ));

        assert!(!blendable(TextureFormat::Rgba32Float, limits));
        limits.rgba32_float_blend = true;
        assert!(blendable(TextureFormat::Rgba32Float, limits));
    }

    #[test]
    fn copy_routes_cover_only_the_lowered_colour_subset() {
        let facts = crate::api::capability::AvailableCapabilities::from_facts(probe(
            MetalCapabilityLimits {
                max_buffer_size: 1,
                max_texture_1d: 1,
                max_texture_2d: 1,
                max_texture_3d: 1,
                max_array_layers: 1,
                max_mip_levels: 1,
                max_sampler_anisotropy: None,
                bc: false,
                etc2_eac: false,
                astc_ldr: false,
                astc_hdr: false,
                astc_3d: false,
                texture_1d: false,
                texture_cube_array: false,
                depth16_unorm: false,
                rgba32_float_blend: false,
                occlusion_queries: false,
                indirect_commands: false,
                base_vertex_instance: false,
                depth_clip_control: false,
                read_write_texture_tier: 0,
                raster_storage_textures: false,
                sample_count_mask: 1 << 1,
                max_vertex_amplification_count: 1,
                comparison_samplers: false,
                sampler_clamp_to_border: false,
                storage_textures: false,
                compute_lowering: false,
                compute_binding_lowering: false,
                raster_lowering: false,
                raster_binding_lowering: false,
                texture_creation: true,
                texture_copy_lowering: true,
                buffer_texture_copy_lowering: true,
                texel_copy_buffer_offset_alignment: 256,
                max_color_attachments: 1,
                max_color_attachment_bytes_per_sample: 1,
                max_inter_stage_shader_variables: 1,
            },
        ));
        assert!(facts.route(&RouteQuery::BufferToBuffer).is_supported());
        assert!(
            facts
                .route(&RouteQuery::TextureToTexture {
                    src_dimension: TextureDimension::D2,
                    src_format: TextureFormat::Rgba8Unorm,
                    src_aspect: TextureAspect::Color,
                    src_sample_count: 1,
                    dst_dimension: TextureDimension::D2,
                    dst_format: TextureFormat::Rgba8Unorm,
                    dst_aspect: TextureAspect::Color,
                    dst_sample_count: 1,
                })
                .is_supported()
        );
        assert!(
            !facts
                .route(&RouteQuery::TextureToTexture {
                    src_dimension: TextureDimension::D2,
                    src_format: TextureFormat::Rgba8Unorm,
                    src_aspect: TextureAspect::Color,
                    src_sample_count: 1,
                    dst_dimension: TextureDimension::D2,
                    dst_format: TextureFormat::Rgba8Unorm,
                    dst_aspect: TextureAspect::Color,
                    dst_sample_count: 4,
                })
                .is_supported()
        );
        assert!(
            facts
                .route(&RouteQuery::TextureToBuffer {
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    aspect: TextureAspect::Color,
                })
                .is_supported()
        );
        assert!(
            !facts
                .route(&RouteQuery::TextureToBuffer {
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Depth32Float,
                    aspect: TextureAspect::Depth,
                })
                .is_supported()
        );
    }

    #[test]
    fn direct_compute_binding_facts_do_not_leak_into_raster_visibility() {
        let facts = crate::api::capability::AvailableCapabilities::from_facts(probe(
            MetalCapabilityLimits {
                max_buffer_size: 1,
                max_texture_1d: 1,
                max_texture_2d: 1,
                max_texture_3d: 1,
                max_array_layers: 1,
                max_mip_levels: 1,
                max_sampler_anisotropy: None,
                bc: false,
                etc2_eac: false,
                astc_ldr: false,
                astc_hdr: false,
                astc_3d: false,
                texture_1d: false,
                texture_cube_array: false,
                depth16_unorm: false,
                rgba32_float_blend: false,
                occlusion_queries: false,
                indirect_commands: false,
                base_vertex_instance: false,
                depth_clip_control: false,
                read_write_texture_tier: 0,
                raster_storage_textures: false,
                sample_count_mask: 1 << 1,
                max_vertex_amplification_count: 1,
                comparison_samplers: false,
                sampler_clamp_to_border: false,
                storage_textures: false,
                compute_lowering: true,
                compute_binding_lowering: true,
                raster_lowering: false,
                raster_binding_lowering: false,
                texture_creation: true,
                texture_copy_lowering: false,
                buffer_texture_copy_lowering: false,
                texel_copy_buffer_offset_alignment: 1,
                max_color_attachments: 1,
                max_color_attachment_bytes_per_sample: 1,
                max_inter_stage_shader_variables: 1,
            },
        ));
        let compute = crate::api::binding::BindingSupportQuery {
            visibility: ShaderStages::COMPUTE,
            kind: crate::api::binding::BindingKind::UniformBuffer { min_size: 1 },
            count: crate::api::binding::BindingCount::One,
            dynamic_offset: true,
        };
        assert_eq!(facts.binding_support(&compute), BindingSupport::Supported);
        let fragment = crate::api::binding::BindingSupportQuery {
            visibility: ShaderStages::FRAGMENT,
            ..compute
        };
        assert_eq!(
            facts.binding_support(&fragment),
            BindingSupport::Unsupported
        );
        assert_eq!(
            facts.binding_limit(
                crate::api::shader::ShaderStage::Compute,
                BindingLimitClass::UniformBuffers,
            ),
            Some(15)
        );
    }

    #[test]
    fn raster_facts_reserve_the_shared_vertex_buffer_namespace() {
        let facts = crate::api::capability::AvailableCapabilities::from_facts(probe(
            MetalCapabilityLimits {
                max_buffer_size: 1,
                max_texture_1d: 1,
                max_texture_2d: 1,
                max_texture_3d: 1,
                max_array_layers: 1,
                max_mip_levels: 1,
                max_sampler_anisotropy: None,
                bc: false,
                etc2_eac: false,
                astc_ldr: false,
                astc_hdr: false,
                astc_3d: false,
                texture_1d: false,
                texture_cube_array: false,
                depth16_unorm: false,
                rgba32_float_blend: false,
                occlusion_queries: false,
                indirect_commands: false,
                base_vertex_instance: false,
                depth_clip_control: false,
                read_write_texture_tier: 0,
                raster_storage_textures: false,
                sample_count_mask: 1 << 1,
                max_vertex_amplification_count: 1,
                comparison_samplers: false,
                sampler_clamp_to_border: false,
                storage_textures: false,
                compute_lowering: false,
                compute_binding_lowering: false,
                raster_lowering: true,
                raster_binding_lowering: true,
                texture_creation: true,
                texture_copy_lowering: false,
                buffer_texture_copy_lowering: false,
                texel_copy_buffer_offset_alignment: 1,
                max_color_attachments: 4,
                max_color_attachment_bytes_per_sample: 16,
                max_inter_stage_shader_variables: 15,
            },
        ));
        let vertex = crate::api::binding::BindingSupportQuery {
            visibility: ShaderStages::VERTEX,
            kind: crate::api::binding::BindingKind::UniformBuffer { min_size: 1 },
            count: crate::api::binding::BindingCount::One,
            dynamic_offset: true,
        };
        assert_eq!(facts.binding_support(&vertex), BindingSupport::Supported);
        assert_eq!(
            facts.binding_limit(
                crate::api::shader::ShaderStage::Vertex,
                BindingLimitClass::UniformBuffers,
            ),
            Some(7)
        );
        assert_eq!(facts.limit(LimitKey::MaxVertexBuffers), Some(15));
        // Immediates are globally advertised (rather than stage-scoped), so a
        // raster-only lowerer must not accidentally promise them to compute.
        assert_eq!(facts.limit(LimitKey::MaxImmediateSize), None);
        assert!(facts.supports_feature(OptionalFeature::PolygonModeLine));
        assert!(facts.supports_feature(OptionalFeature::IndependentBlend));
        assert!(!facts.supports_feature(OptionalFeature::BaseVertex));

        // Raster storage textures need a Metal stage/tier conformance path;
        // direct resource binding alone is insufficient proof.
        let storage_texture = crate::api::binding::BindingSupportQuery {
            visibility: ShaderStages::FRAGMENT,
            kind: crate::api::binding::BindingKind::StorageTexture {
                dimension: TextureViewDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                access: StorageAccess::WriteOnly,
            },
            count: crate::api::binding::BindingCount::One,
            dynamic_offset: false,
        };
        assert_eq!(
            facts.binding_support(&storage_texture),
            BindingSupport::Unsupported
        );
    }
}
