//! What a live Direct3D 12 device reports about itself, read through
//! `CheckFeatureSupport`.
//!
//! This module owns one step and no others: turning a created `ID3D12Device`
//! into a [`CapabilityFacts`]. It decides nothing about legality, holds no
//! resource, and is not consulted again after device creation — section 7.2
//! makes an enabled contract immutable, so the table built here is the table the
//! device reports for its whole life.
//!
//! # Why this runs against a device
//!
//! Direct3D 12 answers every capability question through a device. `DXGI` can
//! hand out an adapter name and a vendor id without one, but format support,
//! resource-binding tiers and multisample quality levels all come from
//! `ID3D12Device::CheckFeatureSupport`, and there is no adapter-level spelling of
//! them. The provider creates a temporary D3D12 device during adapter enumeration
//! and uses this probe to publish a complete `AdapterInfo` snapshot; device
//! creation runs the same probe for the final native device before validating its
//! requirements.
//!
//! # What is probed, what is structural, and what is still absent
//!
//! Being explicit about which is which, because "the table is full" and "the table
//! is complete" are different claims and only one of them is true here.
//!
//! - **Probed.** Per-format storage access and attachment support come from
//!   `D3D12_FEATURE_FORMAT_SUPPORT`. A refusal from `CheckFeatureSupport` on it
//!   is reported as a backend failure rather than swallowed: it is a mandatory
//!   question, so a device that cannot answer it is not a device this backend can
//!   describe.
//! - **Structural.** All three [`OptionalFeature`]s are recorded without a probe,
//!   and the reason is written at each call site. Two of them — `Compute` and
//!   `BindingArrays` — are properties of Direct3D 12 itself: every D3D12 command
//!   list has `Dispatch`, every D3D12 root signature has descriptor ranges. The
//!   third, `SamplerAnisotropy`, is structural in a subtler way that is worth
//!   knowing before writing a probe that cannot exist: Direct3D 12 has **no
//!   queryable maximum sampler anisotropy**. `MaxAnisotropy` is a field of
//!   `D3D12_SAMPLER_DESC` that the caller fills, and the specification requires
//!   every device to accept the whole range 1 through 16, so there is no
//!   `CheckFeatureSupport` question whose answer could be "this device cannot
//!   filter anisotropically" — a device like that would not be a D3D12 device.
//!   Recording these as structural facts rather than as unasked questions is what
//!   keeps a caller from being told a D3D12 device cannot dispatch or cannot
//!   filter.
//! - **Structural, with two probed rows.** [`record_binding_support`] fills the
//!   binding table, and most of it is a claim about Direct3D 12 rather than a
//!   reading: a buffer's role is decided by the root signature and the resource
//!   state, not by the resource. The storage-texture rows are the exception — they
//!   read the same two format-support bits the texture table reads — and the two
//!   limitations the table *does* record come from the header's own dimension
//!   enums, which have no multisampled 1D, 3D or cube spelling and no cube UAV
//!   member at all. See that function for why each negative is a fact instead of a
//!   hole.
//! - **Absent, and deliberately not guessed.** View-compatibility facts are not
//!   recorded yet, and seven of the twenty-seven
//!   [`crate::api::platform::LimitKey`]s are not either. The seven are the ones
//!   that name a ceiling Direct3D 12 does not state, and
//!   [`record_api_shape_limits`] lists them rather than filling them with a
//!   number borrowed from another API's convention.
//!   [`CapabilityFacts::binding_limit`] is absent for a related reason that is
//!   worth stating where the table is missing rather than where it is read: the
//!   answer it wants is a *per stage and class* ceiling, Direct3D 12 defines none,
//!   and the two constants in the neighbourhood — the descriptor-heap tiers and the
//!   sampler-heap size — are pool sizes shared by every stage and every pipeline.
//!   Recording one of those as `binding_limit(Vertex, SampledTextures)` would put a
//!   per-stage rule in the device's mouth. `None` is the spec's "inapplicable"
//!   answer and the portable layer's "no ceiling to impose", which is what this
//!   device is; what that costs is stated at [`probe`].
//!   Texture facts, route facts, binding facts and the other twenty limits *are*
//!   recorded. What is left is a real coverage gap and it is recorded as one rather
//!   than papered over: see [`probe`] for what a caller observes while it stands,
//!   and [`record_limits`] for why the missing limits are a *mapping* problem
//!   rather than a probing one.
//!
//! # The route table, and the one operation it refuses
//!
//! [`record_format_routes`] and [`record_buffer_route`] fill
//! [`crate::api::resource::route::RouteQuery`]'s table. The key is not one a
//! backend can walk in full — the two texture-to-texture routes carry a `u32`
//! sample count on each side — so this table is under the same obligation as the
//! texture one: record every *legal* key, because an unrecorded legal route is a
//! refusal to execute an operation the device can perform, and section 9.4 makes
//! that refusal final rather than advisory.
//!
//! The refusals in this table are therefore of two kinds, and they are worth
//! telling apart. A key naming a plane its format does not have, or a
//! texture-to-texture pair whose formats differ, is *not a route this device
//! lacks* — it is not an operation Direct3D 12 offers at all, and it is left to
//! the negative because there is nothing to record. A **filtered blit** is the
//! other kind and the more important one: Direct3D 12 has `CopyBufferRegion`,
//! `CopyTextureRegion`, `CopyResource`, `CopyTiles` and `ResolveSubresource`, and
//! no filtered or scaled blit at any of them. Recording `Unsupported` for every
//! blit key would be recording nothing, so the walk records none — and the refusal
//! is structural, which is what makes it safe to reach by absence. A test asserts
//! it on a real device so that a later change cannot quietly start promising a
//! lowering section 9.4 forbids.
//!
//! # Which texture keys can be asked and which cannot
//!
//! [`CapabilityFacts::texture_support`] is keyed on a combination a backend
//! cannot walk in full, so an absent key there answers the negative rather than
//! panicking — the same rule the buffer table inverts, for the reason the
//! capability module gives. The obligation that replaces "fill everything" is
//! weaker but real: a *legal* combination left unrecorded would tell a caller
//! that a texture the device can create cannot be created, and that failure is
//! silent because a negative is a well-formed answer. [`record_texture_support`]
//! therefore walks every dimension, sample count, usage mask and cube intent for
//! every format this backend can name, and both halves of that claim have a test:
//! the walk's coverage is asserted on a real device, and the rules that turn a
//! support word into an answer are asserted without one.
//!
//! # The one table that must be complete
//!
//! [`CapabilityFacts::buffer_support`] is keyed on [`BufferUsage`], whose space
//! is sixty-four masks a backend can walk in full, so
//! [`crate::api::capability::CapabilityFacts`]'s lookup rule makes an absent
//! entry there a hole in enumeration rather than an answer — it panics. That
//! makes this table the one place where a partial enumeration is not a smaller
//! answer but a crash, which is why it is filled here in full and first.
//!
//! Every non-empty combination is recorded as supported, and that is a claim
//! about Direct3D 12 rather than about a driver: a D3D12 buffer resource is
//! created with no usage flags at all, and what a buffer may be used for is
//! decided by the root signature that binds it and the resource states it is
//! transitioned through. Vertex, index, constant, and unordered-access uses are
//! orthogonal states, so any combination of them is expressible — it costs
//! transitions, not legality. The empty mask is recorded as unsupported, because
//! a buffer with no usage bit has no legal operation at all and section 12.3
//! refuses to create one.

use core::mem::size_of;

use windows::core::Interface;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
    D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, D3D12_CS_TGSM_REGISTER_COUNT,
    D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP, D3D12_CS_THREAD_GROUP_MAX_X,
    D3D12_CS_THREAD_GROUP_MAX_Y, D3D12_CS_THREAD_GROUP_MAX_Z, D3D12_FEATURE_D3D12_OPTIONS,
    D3D12_FEATURE_DATA_D3D12_OPTIONS, D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS, D3D12_FEATURE_FORMAT_SUPPORT,
    D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, D3D12_FORMAT_SUPPORT1,
    D3D12_FORMAT_SUPPORT1_BLENDABLE, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL,
    D3D12_FORMAT_SUPPORT1_RENDER_TARGET, D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE,
    D3D12_FORMAT_SUPPORT1_TEXTURE1D, D3D12_FORMAT_SUPPORT1_TEXTURE2D,
    D3D12_FORMAT_SUPPORT1_TEXTURE3D, D3D12_FORMAT_SUPPORT1_TEXTURECUBE,
    D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW, D3D12_FORMAT_SUPPORT2,
    D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE,
    D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT, D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT,
    D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS, D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT,
    D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT, D3D12_REQ_MIP_LEVELS,
    D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES, D3D12_REQ_TEXTURE1D_U_DIMENSION,
    D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION, D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
    D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION, D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT,
    D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, ID3D12Device,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC2_UNORM,
    DXGI_FORMAT_BC2_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM_SRGB,
    DXGI_FORMAT_BC4_SNORM, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC5_SNORM, DXGI_FORMAT_BC5_UNORM,
    DXGI_FORMAT_BC6H_SF16, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC7_UNORM,
    DXGI_FORMAT_BC7_UNORM_SRGB, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D32_FLOAT,
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8_SNORM,
    DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_R8G8_SNORM,
    DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8B8A8_SINT,
    DXGI_FORMAT_R8G8B8A8_SNORM, DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R10G10B10A2_UINT,
    DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R16_FLOAT,
    DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_UNORM,
    DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_R16G16_SNORM,
    DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT,
    DXGI_FORMAT_R16G16B16A16_SINT, DXGI_FORMAT_R16G16B16A16_SNORM, DXGI_FORMAT_R16G16B16A16_UINT,
    DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_SINT,
    DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_SINT,
    DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_SINT,
    DXGI_FORMAT_R32G32B32A32_UINT,
};

use crate::api::binding::vocabulary::BindableKind;
use crate::api::binding::{
    BindingSupport, BufferBindingAccess, SamplerKind, StorageAccess, TextureSampleType,
};
use crate::api::capability::{BindingSupportKey, CapabilityFacts, visibilities};
use crate::api::error::RhiResult;
use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery, format_aspects,
};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
use crate::api::resource::route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport, TexelCopyLayoutLimits,
};
use crate::api::resource::subresource::{TextureAspect, TextureAspects, aspect_bits};
use crate::api::resource::texture::{
    Extent3d, TextureDimension, TextureUsage, TextureViewCompatibility,
};
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::vocabulary::AcceptedCodeForm;
use crate::backend::dx12::ffi;

/// Reads every fact this backend enumerates from `device`.
///
/// Returns a `Result` rather than a bare [`CapabilityFacts`] because two of the
/// questions are mandatory — `D3D12_FEATURE_DATA_D3D12_OPTIONS` is defined for
/// every D3D12 device, and a format's support flags are always answerable — so a
/// refusal is a fault rather than a fact. Failing device creation on it is the
/// honest outcome: this backend cannot describe a device that will not say what
/// it is, and publishing an empty table instead would report a working device
/// with no capabilities.
///
/// # What a caller observes while the gaps above stand
///
/// The gaps are now three, and they do not cost the same thing, which is why they
/// are counted separately rather than gathered under one word.
///
/// A **view-compatibility** question is answered `Unsupported` when the relation
/// has no entry, and a `limit` question is answered `None` for the seven keys with
/// no Direct3D 12 ceiling to cite. Both are conservative: they refuse work the
/// hardware can do, and neither is silent, so the gap costs throughput rather than
/// correctness and is recorded here rather than left to be discovered.
///
/// The **binding table is no longer one of them**, and that difference is the one
/// worth knowing about. A binding answer of `Unsupported` is not conservative the
/// way those two are: it is a refusal of a legal layout, so an unrecorded legal
/// binding would silently forbid something the device can do. That is why
/// [`record_binding_support`] walks every combination and records an explicit
/// answer matching the currently implemented lowering, rather than recording
/// only the shapes a driver happened to make convenient. What remains absent from the
/// binding chapter is `binding_limit`, whose absence costs the *opposite* — no
/// per-stage ceiling is imposed, because Direct3D 12 does not state one — and
/// whose consequence belongs to the layer that does allocate descriptors: a
/// pipeline interface whose aggregate a real heap cannot hold is refused when the
/// heap is built, not when the interface is created.
///
/// The route table is no longer one of those gaps, and the difference is worth
/// being precise about. A route answer of `Unsupported` is not conservative the
/// way the two above are — section 9.4 makes it final, so an unrecorded route is
/// an operation the caller will never be able to run. That is why the route walk
/// records every legal key instead of relying on this rule, and why the one
/// operation it leaves to the negative (a filtered blit) is the one Direct3D 12
/// does not offer at any entry point rather than one this port has not got
/// around to lowering.
pub(super) fn probe(device: &ID3D12Device) -> RhiResult<CapabilityFacts> {
    let options = options(device)?;

    let mut facts = CapabilityFacts::empty();
    facts.record_transient_capabilities(crate::backend::dx12::resource::transient_capabilities());
    record_features(device, &mut facts);
    record_code_forms(&mut facts);
    record_limits(&options, &mut facts);
    record_buffer_support(&mut facts);
    record_binding_support(&mut facts);
    record_texture_limits(&mut facts);
    record_api_shape_limits(&mut facts);

    // One probe per format, shared by the two tables that are keyed on a format.
    // Asked once and not twice, because Direct3D 12 answers `FORMAT_SUPPORT` for
    // a format and not for a question about one: the same two words back both
    // "what can this format's storage be used for" and "which dimensions and
    // usages can a texture of this format have".
    for format in TextureFormat::all() {
        // A portable format with no single DXGI format is one this backend
        // cannot describe, and it is left out of the per-format table rather
        // than recorded as unsupported. The two are different answers: a missing
        // `FormatFacts` says the device was not asked, and a recorded all-false
        // one would say it was asked and said no. Only one of those is true.
        let Some(dxgi) = dxgi_format(format) else {
            continue;
        };

        let support = format_support(device, dxgi)?;

        // The quality levels per sample count are still queried even though this
        // correctness baseline deliberately refuses multisampled textures below.
        // Keeping the probe here makes the eventual MSAA lowering change local;
        // it must add RTV/DSV/resolve lowering and conformance coverage before
        // this table may publish a non-1x texture again.
        let quality = quality_levels(device, dxgi)?;

        record_format_facts(format, &support, &mut facts);
        record_texture_support(format, &support, &quality, &mut facts);
        record_format_routes(format, &mut facts);
        record_storage_texture_bindings(format, &support, &mut facts);
    }

    record_buffer_route(&mut facts);

    Ok(facts)
}

/// Records the code form this backend consumes.
///
/// One member, and structural rather than probed: Direct3D 12 takes shader
/// bytecode and nothing else. `D3D12_SHADER_BYTECODE` is a pointer and a length
/// with no format field, `CreateComputePipelineState` and
/// `CreateGraphicsPipelineState` accept exactly that struct, and there is no
/// `CheckFeatureSupport` question about shader input that could answer anything
/// else. A D3D12 device that refused DXIL would not be a D3D12 device.
///
/// The portable artifact need not be *validated* here — `decide` answers whether
/// this device will try, and `CreateComputePipelineState` remains the only thing
/// that can say whether the bytecode compiled. What this record settles is the
/// form, which is the part of that question a device can answer in advance.
fn record_code_forms(facts: &mut CapabilityFacts) {
    facts.record_code_form(AcceptedCodeForm::Dxil);
}

/// Records the three optional features this backend can answer for.
///
/// Takes no device and no probe result, deliberately: every feature here is
/// structural, so a parameter would be an unused argument that reads like a
/// question being asked. [`record_limits`] is the one that takes the options
/// struct, because the limit it derives is a real device reading.
fn record_features(device: &ID3D12Device, facts: &mut CapabilityFacts) {
    // Compute is structural. Every Direct3D 12 command list has `Dispatch`, every
    // D3D12 device has a compute queue family's worth of dispatch support, and
    // there is no D3D12 device that cannot run a compute pipeline — so there is
    // no question here for `CheckFeatureSupport` to answer, and asking the driver
    // to confirm the API it implements would be a probe with a foregone
    // conclusion. Section 0 forbids inferring capability from a trait's presence;
    // this is not that — it is a fact about the API revision the device was
    // created under, which is what `D3D12CreateDevice` already established.
    facts.record_feature(OptionalFeature::Compute);

    // Binding arrays are structural for the same reason, one level down. The
    // portable feature covers fixed-length arrays of buffers, textures and
    // samplers, and Direct3D 12 expresses exactly that with a descriptor range:
    // `NumDescriptors > 1` in one range is an array, and descriptor ranges are
    // part of every root signature at every resource-binding tier. The variants
    // the portable feature deliberately excludes — runtime-sized, partially
    // bound, update-after-bind, arbitrarily indexed — are the ones that *would*
    // need a tier probe, and none of them is being claimed here.
    facts.record_feature(OptionalFeature::BindingArrays);

    // Anisotropy is structural as well, and the reason is worth stating because
    // the obvious probe does not exist. Direct3D 12 has no queryable maximum
    // sampler anisotropy: `MaxAnisotropy` is a field of `D3D12_SAMPLER_DESC` that
    // the *caller* fills, and the specification requires every device to accept
    // the whole range 1 through 16. There is therefore no
    // `CheckFeatureSupport` question whose answer would be "this device cannot
    // filter anisotropically" — such a device would not be a D3D12 device. The
    // limit that goes with this feature is recorded in `record_limits`, and it is
    // the same API-defined ceiling rather than a second, independent probe.
    facts.record_feature(OptionalFeature::SamplerAnisotropy);
    // D3D12 sampler descriptors always expose comparison and border addressing;
    // neither answer is inferred from the Rust backend trait.
    facts.record_feature(OptionalFeature::ComparisonSamplers);
    facts.record_feature(OptionalFeature::SamplerClampToBorder);
    // D3D12 exposes a fixed palette of float border colors. `Zero` maps to the
    // all-zero float value and is not claimed as an integer-border capability.
    facts.record_feature(OptionalFeature::PolygonModeLine);
    facts.record_feature(OptionalFeature::DepthClipControl);
    facts.record_feature(OptionalFeature::DepthBiasClamp);
    facts.record_feature(OptionalFeature::DualSourceBlending);
    facts.record_feature(OptionalFeature::IndependentBlend);
    // Every D3D12 direct command list exposes occlusion query begin/end and a
    // query heap is an ordinary device allocation. Timestamp/statistics are
    // intentionally separate: their portable result conversion is not implied
    // by this structural occlusion path.
    facts.record_feature(OptionalFeature::OcclusionQuery);
    facts.record_feature(OptionalFeature::QueryResolve);
    facts.record_feature(OptionalFeature::IndirectDispatch);
    facts.record_feature(OptionalFeature::IndirectDraw);
    facts.record_feature(OptionalFeature::MultiDrawIndirect);
    facts.record_feature(OptionalFeature::MultiDrawIndirectCount);
    facts.record_feature(OptionalFeature::IndirectFirstInstance);
    // DrawIndexedInstanced carries BaseVertexLocation directly; the raster
    // lowerer forwards the recorded i32 without emulation.
    facts.record_feature(OptionalFeature::BaseVertex);
    facts.record_feature(OptionalFeature::ClearBuffer);
    // Color clear lowers through zeroed upload footprints; depth/stencil clear
    // uses a temporary DSV after recording-time validation requires both COPY_DST
    // and DEPTH_STENCIL_ATTACHMENT. Planar formats are not creatable here.
    facts.record_feature(OptionalFeature::ClearTexture);
    // UPLOAD/READBACK staging maps are real through the exact MAP_* support
    // rows below. They do not imply MappablePrimaryBuffers: D3D12's upload and
    // readback heaps cannot also express broader primary GPU usages.
    facts.record_feature(OptionalFeature::Immediates);
    // PipelineLibrary is an ID3D12Device1 extension. Its cast is the native
    // capability query; the cache lowering owns real Create/Load/Store/Serialize
    // paths and therefore both facts can be published together.
    if device
        .cast::<windows::Win32::Graphics::Direct3D12::ID3D12Device1>()
        .is_ok()
    {
        facts.record_feature(OptionalFeature::PipelineCache);
        facts.record_feature(OptionalFeature::PipelineCacheSerialization);
    }
}

/// Records the device limits this backend can ground.
///
/// Deliberately short. The remaining [`LimitKey`]s are not recorded yet for a
/// reason that is about *mapping* rather than about probing: several of them —
/// bind-group counts, dynamic-buffer counts per pipeline layout — are portable
/// questions whose Direct3D 12 counterpart is spread across root-signature
/// layout rules and resource-binding tiers, and answering one from the wrong
/// tier would be a number that looks probed and is not. `limit` answers `None`
/// for those today, which is a conservative refusal rather than a wrong answer.
///
/// The two recorded here are the two whose Direct3D 12 source is unambiguous.
fn record_limits(options: &D3D12_FEATURE_DATA_D3D12_OPTIONS, facts: &mut CapabilityFacts) {
    // D3D12_QUERY_HEAP_DESC carries Count as a u32 and the API specifies no
    // smaller device limit. Allocation availability is still a native resource
    // creation result, just like a buffer of a representable size.
    facts.record_limit(LimitKey::MaxQueriesPerQuerySet, u64::from(u32::MAX));
    facts.record_limit(LimitKey::QueryResolveBufferAlignment, 8);
    facts.record_limit(LimitKey::ImmediateDataAlignment, 4);
    // D3D12 Map itself permits byte-granular ranges on buffer resources.
    facts.record_limit(LimitKey::MapAlignment, 1);
    // 32 DWORDs are reserved for the one contiguous DX12 root-constant ABI.
    // The remaining 32 DWORDs leave room for descriptor-table parameters in
    // the native 64-DWORD root-signature budget.
    facts.record_limit(LimitKey::MaxImmediateSize, 128);
    // Anisotropic filtering's ceiling, from the sampler-descriptor range rather
    // than from a device query — see `record_features` for why there is no query
    // to make. `MaxAnisotropy` is clamped to 1..=16 by the API, so 16 is the
    // largest value any D3D12 device can be asked for.
    facts.record_limit(LimitKey::MaxSamplerAnisotropy, 16);

    // The largest single buffer, derived from the resource address space rather
    // than reported as a buffer cap, because Direct3D 12 does not define one: a
    // committed buffer is limited by the number of bits the device can address
    // per resource, which is what `MaxGPUVirtualAddressBitsPerResource` reports.
    // `1 << bits` is therefore an upper bound on any resource and not a promise
    // that the driver will hand out that much — a distinction worth keeping, so
    // this is recorded as a limit and not as a statement about what will be
    // allocated.
    //
    // Clamped rather than shifted blindly: a driver reporting 64 or more would
    // make the shift overflow, and `u64::MAX` is the honest ceiling in that case
    // rather than a wrapping number.
    let bits = options.MaxGPUVirtualAddressBitsPerResource;
    let ceiling = if bits >= 64 { u64::MAX } else { 1u64 << bits };
    facts.record_limit(LimitKey::MaxBufferSize, ceiling);

    // The storage-buffer binding ceiling is the same number, and it is recorded
    // here rather than in `record_api_shape_limits` because it is the same
    // *reading*: Direct3D 12 defines no cap specific to a storage binding, so
    // the address space is the answer to both questions, and asking the driver
    // twice for one fact is how two entries in a table start to disagree.
    facts.record_limit(LimitKey::MaxStorageBufferBindingSize, ceiling);
}

/// Fills the buffer-support table for its whole key space.
///
/// See the module documentation for why this table must be complete and why
/// every non-empty combination is a structural yes.
fn record_buffer_support(facts: &mut CapabilityFacts) {
    // The ceiling a supported answer carries. Direct3D 12 defines no per-buffer
    // size cap: a committed buffer is limited by the GPU virtual address space
    // the device exposes, which `D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT`
    // reports and which this backend does not yet read. The portable ceiling is
    // therefore reported as the largest size the *descriptor* can express rather
    // than as a device limit, and `LimitKey::MaxBufferSize` — which is where a
    // real device ceiling belongs — stays unrecorded rather than being answered
    // from here with a number that would be a guess dressed as a fact.
    let limits = BufferSupportLimits::new(u64::MAX);

    for usage in BufferUsage::all() {
        let support = if usage.is_empty() {
            // No legal operation at all, and section 12.3 refuses to create one.
            // Recorded rather than skipped: the space is enumerated in full, and
            // an entry left out of it would be a hole that panics.
            BufferSupport::Unsupported
        } else if usage.contains(BufferUsage::MAP_READ) && usage.contains(BufferUsage::MAP_WRITE) {
            // D3D12 has distinct READBACK and UPLOAD heaps. One allocation
            // cannot be both, so this portable combination must fail before
            // creation instead of selecting one direction arbitrarily.
            BufferSupport::Unsupported
        } else if usage.contains(BufferUsage::MAP_READ) {
            // READBACK heaps are permanently COPY_DEST. This currently closes
            // the map-read + buffer-copy route; query resolve and other uses
            // remain unsupported until their lowering is host-heap aware.
            let allowed = BufferUsage::MAP_READ.union(BufferUsage::COPY_DST);
            if usage.is_subset_of(allowed) {
                BufferSupport::Supported(limits)
            } else {
                BufferSupport::Unsupported
            }
        } else if usage.contains(BufferUsage::MAP_WRITE) {
            // UPLOAD heaps are permanently GENERIC_READ, which includes the
            // source side of a copy but not arbitrary DEFAULT-heap states.
            let allowed = BufferUsage::MAP_WRITE.union(BufferUsage::COPY_SRC);
            if usage.is_subset_of(allowed) {
                BufferSupport::Supported(limits)
            } else {
                BufferSupport::Unsupported
            }
        } else {
            BufferSupport::Supported(limits)
        };
        facts.record_buffer_support(usage, support);
    }
}

/// Fills the binding-support table for the families no format decides.
///
/// Capability records what this backend can lower end to end, rather than every
/// shape the native API could theoretically express. Static buffer descriptor
/// tables are implemented. Dynamic offsets need root descriptors.
///
/// Texture descriptor writing exists, but compute command lowering currently
/// refuses texture uses. Raster lowering can consume texture descriptor tables.
/// Sampler descriptor writing and both compute and graphics sampler-table binding
/// exist. The visibility-sensitive texture answers below make the remaining
/// compute-texture boundary part of the immutable device contract instead of
/// discovering it after native work was accepted.
///
/// Two limitations this API does have are recorded as the negatives they are, and
/// both come from the header rather than from a driver reading:
///
/// - A sampled texture cannot be multisampled in any dimension but two.
///   `D3D12_SRV_DIMENSION` has `TEXTURE2DMS` and `TEXTURE2DMSARRAY` and no
///   multisampled spelling of 1D, 3D or cube — which is section 13.4's rule seen
///   from the descriptor's side.
/// - A storage texture cannot be a cube. `D3D12_UAV_DIMENSION` has no cube
///   member at all, at any resource-binding tier, so there is nothing to probe for
///   and nothing a tier could change.
///
/// The two sample types are not read here and cannot be: whether a float format
/// may be sampled through a filtering sampler is a pairing of *view* and sampler,
/// which section 22.3 and pipeline validation decide against `FormatFacts`, not a
/// property of the binding. Enumerating them anyway is the difference between a
/// table with holes and a table with answers, and every hole in this table is a
/// silent refusal of a legal binding.
fn record_binding_support(facts: &mut CapabilityFacts) {
    for dynamic_offset in [false, true] {
        let answer = if dynamic_offset {
            BindingSupport::Unsupported
        } else {
            BindingSupport::Supported
        };
        record_bindable(facts, BindableKind::UniformBuffer, dynamic_offset, answer);
        for access in [
            BufferBindingAccess::ReadOnly,
            BufferBindingAccess::ReadWrite,
        ] {
            record_bindable(
                facts,
                BindableKind::StorageBuffer { access },
                dynamic_offset,
                answer,
            );
        }
    }

    // Enumerated although the answer does not vary with `sample_type`, for the
    // reason the walk's documentation gives.
    for dimension in VIEW_DIMENSIONS {
        for sample_type in [
            TextureSampleType::Float,
            TextureSampleType::UnfilterableFloat,
            TextureSampleType::Sint,
            TextureSampleType::Uint,
            TextureSampleType::Depth,
        ] {
            for multisampled in [false, true] {
                record_texture_bindable(
                    facts,
                    BindableKind::SampledTexture {
                        dimension,
                        sample_type,
                        multisampled,
                    },
                    false,
                    // The texture table deliberately refuses MSAA until its
                    // full raster/resolve lifecycle is lowered, so an MSAA
                    // binding shape cannot be useful before that same work.
                    // A portable depth sample is not just an SRV whose format
                    // happens to be D32/D16.  DX12 needs a typeless resource
                    // plus an R* view format, while this baseline's texture
                    // allocation intentionally preserves the exact portable
                    // depth DXGI format.  Publishing the generic depth-SRV
                    // row would therefore let validation reach a void native
                    // descriptor-write call that cannot report its rejection.
                    // Keep it absent until resource/view typeless families are
                    // lowered as one end-to-end path.
                    if multisampled || matches!(sample_type, TextureSampleType::Depth) {
                        BindingSupport::Unsupported
                    } else {
                        BindingSupport::Supported
                    },
                );
            }
        }
    }

    for kind in [
        SamplerKind::Filtering,
        SamplerKind::NonFiltering,
        SamplerKind::Comparison,
    ] {
        record_bindable(
            facts,
            BindableKind::Sampler { kind },
            false,
            BindingSupport::Supported,
        );
    }
}

/// Records one bindable kind's answer across every visibility and array shape.
///
/// The sweep is here rather than repeated at each call site so that "every
/// combination this device can be asked about" is one loop a reader can check,
/// and so that a call site states only what the device reports.
///
/// `dynamic_offset` is a parameter rather than part of the sweep because it is
/// only a question for the two buffer kinds: section 20.4 makes it "valid only for
/// UniformBuffer / StorageBuffer", the layout validator refuses it on anything else
/// before a query is made, and a direct caller who asks anyway is answered the
/// negative — a binding with a dynamic offset on a texture is not a binding that
/// exists.
fn record_bindable(
    facts: &mut CapabilityFacts,
    kind: BindableKind,
    dynamic_offset: bool,
    answer: BindingSupport,
) {
    for visibility in visibilities() {
        for array in [false, true] {
            facts.record_binding_support(
                BindingSupportKey {
                    visibility,
                    kind,
                    array,
                    runtime_sized: false,
                    dynamic_offset,
                },
                answer,
            );
        }
    }
}

/// Records the storage-texture rows for one format.
///
/// Per format because the portable key names one: a storage texture binding
/// declares the format the shader reads and writes, unlike a sampled one, where
/// the format belongs to the view.
///
/// The access answer is the conjunction of the real format probe and the
/// descriptor shape the lowering can emit. Read-only bindings copy an SRV;
/// write-only/read-write bindings build a typed UAV. D3D12 has no cube UAV.
fn record_storage_texture_bindings(
    format: TextureFormat,
    support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    facts: &mut CapabilityFacts,
) {
    for dimension in VIEW_DIMENSIONS {
        for access in [
            StorageAccess::ReadOnly,
            StorageAccess::WriteOnly,
            StorageAccess::ReadWrite,
        ] {
            let access_supported = match access {
                StorageAccess::ReadOnly => {
                    has_support2(support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD)
                }
                StorageAccess::WriteOnly => {
                    has_support2(support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)
                }
                StorageAccess::ReadWrite => {
                    has_support2(support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD)
                        && has_support2(support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)
                }
            };
            let answer = if access_supported
                && !matches!(
                    dimension,
                    TextureViewDimension::Cube | TextureViewDimension::CubeArray
                ) {
                BindingSupport::Supported
            } else {
                BindingSupport::Unsupported
            };
            record_texture_bindable(
                facts,
                BindableKind::StorageTexture {
                    dimension,
                    format,
                    access,
                },
                false,
                answer,
            );
        }
    }
}

/// Records a texture binding only where its command lowering exists today.
///
/// The binding packet itself is stage-agnostic, but the recorded command is not:
/// `lower_compute_dispatch` explicitly refuses `ResourceUse::Texture`, whereas
/// raster lowering transitions and retains sampled/storage textures. A capability
/// answer that ignored that distinction would let a compute pipeline pass all
/// public validation only to be rejected during submission.
fn record_texture_bindable(
    facts: &mut CapabilityFacts,
    kind: BindableKind,
    dynamic_offset: bool,
    answer: BindingSupport,
) {
    for visibility in visibilities() {
        let answer = if visibility.contains(crate::api::shader::ShaderStages::COMPUTE) {
            BindingSupport::Unsupported
        } else {
            answer
        };
        for array in [false, true] {
            facts.record_binding_support(
                BindingSupportKey {
                    visibility,
                    kind: kind.clone(),
                    array,
                    runtime_sized: false,
                    dynamic_offset,
                },
                answer,
            );
        }
    }
}

/// Records one portable format's storage access from its probed support words.
///
/// Direct3D 12 answers the three storage accesses with two independent bits, not
/// three: `UAV_TYPED_LOAD` and `UAV_TYPED_STORE`. Read-write is therefore the
/// conjunction rather than a third probe, and it is recorded as such instead of
/// being reported from a bit that does not exist. The two stay separate in the
/// record — a format can load without storing, which is a shader that reads a
/// storage texture it must not write — for the reason `StorageAccessSupport`
/// documents.
fn record_format_facts(
    format: TextureFormat,
    support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    facts: &mut CapabilityFacts,
) {
    let read_only = has_support2(support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD);
    let write_only = has_support2(support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
    let aspects = format_aspects(format);
    let depth_stencil = has_support1(support, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL);

    facts.record_format(
        format,
        FormatFacts::new(
            format,
            StorageAccessSupport::new(read_only, write_only, read_only && write_only),
            has_support1(support, D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
            depth_stencil && aspects.contains(TextureAspects::DEPTH),
            depth_stencil && aspects.contains(TextureAspects::STENCIL),
            has_support1(support, D3D12_FORMAT_SUPPORT1_BLENDABLE),
        ),
    );
}

/// Whether `support` carries one `D3D12_FORMAT_SUPPORT2` bit.
///
/// Compared through the raw word rather than through a `contains` helper so that
/// the test is visibly a mask test on the driver's own bit, and so that a bit the
/// binding crate renames cannot quietly change the meaning of this predicate.
fn has_support2(support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT, bit: D3D12_FORMAT_SUPPORT2) -> bool {
    support.Support2.0 & bit.0 != 0
}

/// Asks the device for its `D3D12_FEATURE_D3D12_OPTIONS`.
fn options(device: &ID3D12Device) -> RhiResult<D3D12_FEATURE_DATA_D3D12_OPTIONS> {
    let mut data = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();

    // SAFETY: `D3D12_FEATURE_DATA_D3D12_OPTIONS` is the struct
    // `D3D12_FEATURE_D3D12_OPTIONS` is defined to fill, which is the pairing
    // `CheckFeatureSupport` documents; the out-parameter points at a live value
    // of exactly that type, and the byte count handed over is that type's own
    // size. The struct takes no input field, so the driver reads nothing from it
    // before writing. This is the only `unsafe` in the module and it is confined
    // to the one call Direct3D 12 cannot avoid.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut data).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
    }
    .map_err(|error| ffi::to_rhi(&error, "ID3D12Device::CheckFeatureSupport"))?;

    Ok(data)
}

/// Asks the device what one DXGI format supports.
fn format_support(
    device: &ID3D12Device,
    format: DXGI_FORMAT,
) -> RhiResult<D3D12_FEATURE_DATA_FORMAT_SUPPORT> {
    // `Format` is the call's *input*: the driver reads it to decide which
    // format's two support words to write. Everything else is output.
    let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: format,
        ..Default::default()
    };

    // SAFETY: the same pairing argument as `options`, plus the input field this
    // feature has: `D3D12_FEATURE_FORMAT_SUPPORT` is defined to fill a
    // `D3D12_FEATURE_DATA_FORMAT_SUPPORT`, whose `Format` member is the question
    // being asked. The out-parameter points at a live value of exactly that type
    // and the size handed over is that type's own size.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut data).cast(),
            size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32,
        )
    }
    .map_err(|error| ffi::to_rhi(&error, "ID3D12Device::CheckFeatureSupport"))?;

    Ok(data)
}

/// The DXGI format a portable format lowers to.
///
/// `None` for a portable format Direct3D 12 has no single format for. Two of
/// section 8.1's P0 formats are in that position for the same reason:
/// [`TextureFormat::Depth24Plus`] explicitly permits a driver to choose between
/// 24-bit depth and 32-bit float depth, so there is no one DXGI format that is
/// the answer, and [`TextureFormat::Depth24PlusStencil8`] has the same latitude.
/// Returning `D24_UNORM_S8_UINT` for either would be this backend picking a
/// format on the caller's behalf and then reporting its facts as though the
/// caller had asked for that one.
///
/// The exact match rather than a nearest-fit: a portable format is a contract
/// about bit layout, so answering a `Depth24Plus` question with `D32_FLOAT`'s
/// facts would describe a resource the caller did not ask for.
pub(crate) fn dxgi_format(format: TextureFormat) -> Option<DXGI_FORMAT> {
    let mapped = match format {
        TextureFormat::R8Unorm => DXGI_FORMAT_R8_UNORM,
        TextureFormat::R8Snorm => DXGI_FORMAT_R8_SNORM,
        TextureFormat::R8Uint => DXGI_FORMAT_R8_UINT,
        TextureFormat::R8Sint => DXGI_FORMAT_R8_SINT,
        TextureFormat::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        TextureFormat::Rg8Snorm => DXGI_FORMAT_R8G8_SNORM,
        TextureFormat::Rg8Uint => DXGI_FORMAT_R8G8_UINT,
        TextureFormat::Rg8Sint => DXGI_FORMAT_R8G8_SINT,
        TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        TextureFormat::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        TextureFormat::Rgba8Snorm => DXGI_FORMAT_R8G8B8A8_SNORM,
        TextureFormat::Rgba8Uint => DXGI_FORMAT_R8G8B8A8_UINT,
        TextureFormat::Rgba8Sint => DXGI_FORMAT_R8G8B8A8_SINT,
        TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        TextureFormat::R16Uint => DXGI_FORMAT_R16_UINT,
        TextureFormat::R16Sint => DXGI_FORMAT_R16_SINT,
        TextureFormat::R16Float => DXGI_FORMAT_R16_FLOAT,
        TextureFormat::R16Unorm => DXGI_FORMAT_R16_UNORM,
        TextureFormat::R16Snorm => DXGI_FORMAT_R16_SNORM,
        TextureFormat::Rg16Uint => DXGI_FORMAT_R16G16_UINT,
        TextureFormat::Rg16Sint => DXGI_FORMAT_R16G16_SINT,
        TextureFormat::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        TextureFormat::Rg16Unorm => DXGI_FORMAT_R16G16_UNORM,
        TextureFormat::Rg16Snorm => DXGI_FORMAT_R16G16_SNORM,
        TextureFormat::Rgba16Uint => DXGI_FORMAT_R16G16B16A16_UINT,
        TextureFormat::Rgba16Sint => DXGI_FORMAT_R16G16B16A16_SINT,
        TextureFormat::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        TextureFormat::Rgba16Unorm => DXGI_FORMAT_R16G16B16A16_UNORM,
        TextureFormat::Rgba16Snorm => DXGI_FORMAT_R16G16B16A16_SNORM,
        TextureFormat::Rgb9e5Ufloat => DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        TextureFormat::Rgb10a2Uint => DXGI_FORMAT_R10G10B10A2_UINT,
        TextureFormat::Rgb10a2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
        TextureFormat::Rg11b10Ufloat => DXGI_FORMAT_R11G11B10_FLOAT,
        TextureFormat::R32Uint => DXGI_FORMAT_R32_UINT,
        TextureFormat::R32Sint => DXGI_FORMAT_R32_SINT,
        TextureFormat::R32Float => DXGI_FORMAT_R32_FLOAT,
        TextureFormat::R64Uint => return None,
        TextureFormat::Rg32Uint => DXGI_FORMAT_R32G32_UINT,
        TextureFormat::Rg32Sint => DXGI_FORMAT_R32G32_SINT,
        TextureFormat::Rg32Float => DXGI_FORMAT_R32G32_FLOAT,
        TextureFormat::Rgba32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
        TextureFormat::Rgba32Sint => DXGI_FORMAT_R32G32B32A32_SINT,
        TextureFormat::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        // BC is a native DXGI block-compressed family.  It remains individually
        // probed below: the mapping only makes `FORMAT_SUPPORT` queryable and
        // never turns a driver refusal into portable support.
        TextureFormat::Bc1RgbaUnorm => DXGI_FORMAT_BC1_UNORM,
        TextureFormat::Bc1RgbaUnormSrgb => DXGI_FORMAT_BC1_UNORM_SRGB,
        TextureFormat::Bc2RgbaUnorm => DXGI_FORMAT_BC2_UNORM,
        TextureFormat::Bc2RgbaUnormSrgb => DXGI_FORMAT_BC2_UNORM_SRGB,
        TextureFormat::Bc3RgbaUnorm => DXGI_FORMAT_BC3_UNORM,
        TextureFormat::Bc3RgbaUnormSrgb => DXGI_FORMAT_BC3_UNORM_SRGB,
        TextureFormat::Bc4RUnorm => DXGI_FORMAT_BC4_UNORM,
        TextureFormat::Bc4RSnorm => DXGI_FORMAT_BC4_SNORM,
        TextureFormat::Bc5RgUnorm => DXGI_FORMAT_BC5_UNORM,
        TextureFormat::Bc5RgSnorm => DXGI_FORMAT_BC5_SNORM,
        TextureFormat::Bc6hRgbUfloat => DXGI_FORMAT_BC6H_UF16,
        TextureFormat::Bc6hRgbFloat => DXGI_FORMAT_BC6H_SF16,
        TextureFormat::Bc7RgbaUnorm => DXGI_FORMAT_BC7_UNORM,
        TextureFormat::Bc7RgbaUnormSrgb => DXGI_FORMAT_BC7_UNORM_SRGB,
        TextureFormat::Depth16Unorm => DXGI_FORMAT_D16_UNORM,
        TextureFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        TextureFormat::Depth32FloatStencil8 => DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        // DX12 does not define ETC2/EAC/ASTC DXGI formats.  They deliberately
        // remain `None`: capability facts omit them and creation/view lowering
        // returns structured Unsupported rather than pretending BC is a
        // substitute codec.
        TextureFormat::Etc2Rgb8Unorm
        | TextureFormat::Etc2Rgb8UnormSrgb
        | TextureFormat::Etc2Rgb8A1Unorm
        | TextureFormat::Etc2Rgb8A1UnormSrgb
        | TextureFormat::Etc2Rgba8Unorm
        | TextureFormat::Etc2Rgba8UnormSrgb
        | TextureFormat::EacR11Unorm
        | TextureFormat::EacR11Snorm
        | TextureFormat::EacRg11Unorm
        | TextureFormat::EacRg11Snorm
        | TextureFormat::Astc4x4Unorm
        | TextureFormat::Astc4x4UnormSrgb
        | TextureFormat::Astc4x4Hdr
        | TextureFormat::Astc5x4Unorm
        | TextureFormat::Astc5x4UnormSrgb
        | TextureFormat::Astc5x4Hdr
        | TextureFormat::Astc5x5Unorm
        | TextureFormat::Astc5x5UnormSrgb
        | TextureFormat::Astc5x5Hdr
        | TextureFormat::Astc6x5Unorm
        | TextureFormat::Astc6x5UnormSrgb
        | TextureFormat::Astc6x5Hdr
        | TextureFormat::Astc6x6Unorm
        | TextureFormat::Astc6x6UnormSrgb
        | TextureFormat::Astc6x6Hdr
        | TextureFormat::Astc8x5Unorm
        | TextureFormat::Astc8x5UnormSrgb
        | TextureFormat::Astc8x5Hdr
        | TextureFormat::Astc8x6Unorm
        | TextureFormat::Astc8x6UnormSrgb
        | TextureFormat::Astc8x6Hdr
        | TextureFormat::Astc8x8Unorm
        | TextureFormat::Astc8x8UnormSrgb
        | TextureFormat::Astc8x8Hdr
        | TextureFormat::Astc10x5Unorm
        | TextureFormat::Astc10x5UnormSrgb
        | TextureFormat::Astc10x5Hdr
        | TextureFormat::Astc10x6Unorm
        | TextureFormat::Astc10x6UnormSrgb
        | TextureFormat::Astc10x6Hdr
        | TextureFormat::Astc10x8Unorm
        | TextureFormat::Astc10x8UnormSrgb
        | TextureFormat::Astc10x8Hdr
        | TextureFormat::Astc10x10Unorm
        | TextureFormat::Astc10x10UnormSrgb
        | TextureFormat::Astc10x10Hdr
        | TextureFormat::Astc12x10Unorm
        | TextureFormat::Astc12x10UnormSrgb
        | TextureFormat::Astc12x10Hdr
        | TextureFormat::Astc12x12Unorm
        | TextureFormat::Astc12x12UnormSrgb
        | TextureFormat::Astc12x12Hdr
        | TextureFormat::Depth24Plus
        | TextureFormat::Depth24PlusStencil8
        | TextureFormat::Stencil8
        | TextureFormat::Nv12
        | TextureFormat::P010 => {
            return None;
        }
    };

    Some(mapped)
}

/// The texture-dimension ceilings Direct3D 12 fixes for every device.
///
/// Recorded from the API's own requirement constants rather than probed, and the
/// distinction is the same one `record_features` draws for anisotropy: these are
/// not device-variable facts. `D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION` is both the
/// smallest maximum a conforming device must offer *and* the largest dimension a
/// D3D12 resource may have, so the constant is the answer for every device
/// rather than a floor this backend is rounding down to.
///
/// A backend that reported a lower number would refuse legal textures, which is
/// the failure this record exists to prevent.
fn record_texture_limits(facts: &mut CapabilityFacts) {
    facts.record_limit(
        LimitKey::MaxTexture1dDimension,
        u64::from(D3D12_REQ_TEXTURE1D_U_DIMENSION),
    );
    facts.record_limit(
        LimitKey::MaxTexture2dDimension,
        u64::from(D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION),
    );
    facts.record_limit(
        LimitKey::MaxTexture3dDimension,
        u64::from(D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION),
    );
    facts.record_limit(
        LimitKey::MaxTextureArrayLayers,
        u64::from(D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION),
    );
}

/// The sample counts this backend asks about.
///
/// Section 8.3 puts the sample count in the key, so the enumeration has to name
/// the counts it will answer for. Five is the API's own list: Direct3D 12
/// defines multisampling at 2, 4, 8 and 16 samples, and 16 is not universally
/// available — it is exactly the case a probe has to decide rather than assume.
/// One is always present, because a single-sampled texture is the degenerate
/// case of the same question.
const SAMPLE_COUNTS: [u32; 5] = [1, 2, 4, 8, 16];

/// Every view dimension section 14.2 declares.
///
/// The binding walk keys on the *view* dimension rather than the resource
/// dimension, because that is what a binding declares: a two-dimensional texture
/// bound as a cube and the same texture bound whole are two different bindings,
/// and `D3D12_SRV_DIMENSION` distinguishes them the same way.
const VIEW_DIMENSIONS: [TextureViewDimension; 6] = [
    TextureViewDimension::D1,
    TextureViewDimension::D2,
    TextureViewDimension::D2Array,
    TextureViewDimension::Cube,
    TextureViewDimension::CubeArray,
    TextureViewDimension::D3,
];

/// Fills the texture-support table for the key space this backend can express.
///
/// # Why this one may be partial, when the buffer table may not
///
/// [`crate::api::capability::CapabilityFacts`]'s rule is that a key space a
/// backend can walk in full must be filled in full, because an absent entry there
/// is a hole rather than an answer. A `TextureSupportKey` is not such a space: it
/// carries a sample count, a format, and a usage mask, and section 8.3 keeps
/// extent and mip and layer counts *out* of it precisely so the space stays
/// small — but "small" is not "walkable", because the usage mask alone is
/// sixty-four entries per dimension, format and sample count. So an absent key
/// here answers the negative rather than panicking, and the correctness
/// obligation is different: not "fill everything" but "do not leave a legal
/// combination unrecorded", because an unrecorded legal key would refuse a
/// texture the device can create.
///
/// That is why the enumeration below walks the combinations that are *legal*
/// rather than every combination the key type can spell. A single-sampled 3D
/// texture is legal and is recorded; a 3D texture with four samples is not a
/// question with an answer — section 13.4 refuses it before any query — and it
/// is left to the negative rather than recorded as a device fact.
fn record_texture_support(
    format: TextureFormat,
    support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    quality: &SampleQuality,
    facts: &mut CapabilityFacts,
) {
    for dimension in [
        TextureDimension::D1,
        TextureDimension::D2,
        TextureDimension::D3,
    ] {
        let dimension_bit = dimension_support_bit(dimension);

        for (index, sample_count) in SAMPLE_COUNTS.iter().enumerate() {
            let sample_count = *sample_count;

            // A multisampled texture exists in two dimensions only, and section
            // 13.4 says so before the device is asked.
            if sample_count > 1 && dimension != TextureDimension::D2 {
                continue;
            }

            for usage in TextureUsage::all() {
                for compatibility in [
                    TextureViewCompatibility::NONE,
                    TextureViewCompatibility::CUBE,
                ] {
                    let query = TextureSupportQuery::new(dimension, format, usage, sample_count)
                        .with_view_compatibility(compatibility);

                    let answer = texture_answer(
                        dimension_bit,
                        support,
                        usage,
                        sample_count,
                        compatibility,
                        quality[index],
                    );

                    facts.record_texture_support(&query, answer);
                }
            }
        }
    }
}

/// The answer for one texture key, assembled from the probed words.
///
/// Takes the already-probed facts rather than probing again, so that the
/// sixty-four usage masks of one (dimension, format, sample count) triple cost no
/// driver calls beyond the ones the caller already made.
fn texture_answer(
    dimension_bit: D3D12_FORMAT_SUPPORT1,
    support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    usage: TextureUsage,
    sample_count: u32,
    compatibility: TextureViewCompatibility,
    quality_levels: u32,
) -> TextureSupport {
    // The empty mask has no legal operation, exactly as for a buffer, and
    // section 13.4 refuses it. It is recorded rather than skipped so that the
    // negative is an answer rather than a hole.
    if usage.is_empty() {
        return TextureSupport::Unsupported;
    }

    // D3D12 itself can create multisampled resources, but this backend cannot
    // yet lower their attachment lifecycle: RTV/DSV creation and raster scope
    // transitions deliberately refuse them, and resolve has no command lowering.
    // Capability is an end-to-end promise, so publishing the native allocation
    // fact here would be a lie until all of those paths and their conformance
    // tests exist.
    if sample_count > 1 {
        return TextureSupport::Unsupported;
    }

    // The raster backend owns 2D DSVs plus 2D and explicit-slice 3D RTVs. D3D
    // depth/stencil views and 1D attachments are still absent, so do not turn a
    // format-support bit into a promise their lowering cannot keep.
    if (usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
        && dimension_bit != D3D12_FORMAT_SUPPORT1_TEXTURE2D)
        || (usage.contains(TextureUsage::COLOR_ATTACHMENT)
            && !matches!(
                dimension_bit,
                D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_TEXTURE3D
            ))
    {
        return TextureSupport::Unsupported;
    }

    // The dimension must be expressible at all before anything else is asked: a
    // format with no `TEXTURE3D` bit cannot back a 3D texture whatever its usage
    // or sample count.
    if !has_support1(support, dimension_bit) {
        return TextureSupport::Unsupported;
    }

    // A cube view needs a cube-capable format in two dimensions. Section 13.2
    // makes this creation-time, and `TEXTURECUBE` is the same fact stated by the
    // driver rather than derived from the dimension here.
    if compatibility.contains(TextureViewCompatibility::CUBE)
        && (!has_support1(support, D3D12_FORMAT_SUPPORT1_TEXTURECUBE)
            || dimension_bit != D3D12_FORMAT_SUPPORT1_TEXTURE2D)
    {
        return TextureSupport::Unsupported;
    }

    if sample_count > 1 && quality_levels == 0 {
        return TextureSupport::Unsupported;
    }

    for (usage_bit, support_bit) in USAGE_REQUIREMENTS {
        if usage.contains(usage_bit) && !has_support1(support, support_bit) {
            return TextureSupport::Unsupported;
        }
    }

    // Storage is the one usage Direct3D 12 splits across the two support words:
    // `TYPED_UNORDERED_ACCESS_VIEW` says a typed UAV exists at all, and the two
    // `SUPPORT2` bits say whether a shader may read and write through it. A
    // storage texture the portable layer describes is read-write, so both are
    // required, and requiring them is the conservative direction: a format with
    // only one would otherwise be reported as a storage texture that half works.
    if usage.contains(TextureUsage::STORAGE)
        && !(has_support2(support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD)
            && has_support2(support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE))
    {
        return TextureSupport::Unsupported;
    }

    TextureSupport::Supported(TextureSupportLimits::new(
        max_extent(dimension_bit),
        max_mip_levels(dimension_bit),
        max_array_layers(dimension_bit),
    ))
}

/// The usage bits that need a `D3D12_FORMAT_SUPPORT1` bit of their own.
///
/// [`TextureUsage::COPY_SRC`] and [`TextureUsage::COPY_DST`] are deliberately
/// absent: Direct3D 12 has no per-format copy bit, because every format can be
/// copied — a copy moves the resource's own texels and asks nothing of the
/// format's interpretation. Leaving them out of this table is what makes the
/// "copy is structural" claim visible at the one place it matters, rather than a
/// hole a reader has to notice.
const USAGE_REQUIREMENTS: [(TextureUsage, D3D12_FORMAT_SUPPORT1); 4] = [
    (TextureUsage::SAMPLED, D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE),
    (
        TextureUsage::STORAGE,
        D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW,
    ),
    (
        TextureUsage::COLOR_ATTACHMENT,
        D3D12_FORMAT_SUPPORT1_RENDER_TARGET,
    ),
    (
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL,
    ),
];

/// The `D3D12_FORMAT_SUPPORT1` bit that says a format can back `dimension`.
fn dimension_support_bit(dimension: TextureDimension) -> D3D12_FORMAT_SUPPORT1 {
    match dimension {
        TextureDimension::D1 => D3D12_FORMAT_SUPPORT1_TEXTURE1D,
        TextureDimension::D2 => D3D12_FORMAT_SUPPORT1_TEXTURE2D,
        TextureDimension::D3 => D3D12_FORMAT_SUPPORT1_TEXTURE3D,
    }
}

/// The largest extent a texture of the dimension `dimension_bit` names may have.
///
/// Keyed on the same bit the caller already matched on, rather than on a second
/// parameter carrying the dimension: the bit is what the probe returned, and two
/// spellings of the same fact are two chances to disagree.
fn max_extent(dimension_bit: D3D12_FORMAT_SUPPORT1) -> Extent3d {
    if dimension_bit == D3D12_FORMAT_SUPPORT1_TEXTURE1D {
        Extent3d::d1(D3D12_REQ_TEXTURE1D_U_DIMENSION)
    } else if dimension_bit == D3D12_FORMAT_SUPPORT1_TEXTURE3D {
        Extent3d::d3(
            D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
            D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
            D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
        )
    } else {
        Extent3d::d2(
            D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
            D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
        )
    }
}

/// The largest mip level count a texture of this dimension may have.
///
/// Derived from the ceiling rather than recorded as a constant: a mip chain
/// cannot be longer than the largest dimension it is built from, and
/// `D3D12_REQ_MIP_LEVELS` is the level count that goes with
/// `D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION`. Deriving the two from one another is
/// what keeps them consistent if either ceiling is ever corrected — and the
/// `min` is what makes "cannot be longer than the largest dimension" true in
/// both directions rather than only when the constants happen to agree.
fn max_mip_levels(dimension_bit: D3D12_FORMAT_SUPPORT1) -> u32 {
    let largest = if dimension_bit == D3D12_FORMAT_SUPPORT1_TEXTURE1D {
        D3D12_REQ_TEXTURE1D_U_DIMENSION
    } else if dimension_bit == D3D12_FORMAT_SUPPORT1_TEXTURE3D {
        D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION
    } else {
        D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
    };
    D3D12_REQ_MIP_LEVELS.min(u32::BITS - largest.leading_zeros())
}

/// The largest array layer count a texture of this dimension may have.
///
/// Only two dimensions carry layers in the portable model: a 1D array exists in
/// Direct3D 12 but is not part of section 8.1's P0 set, and a 3D texture is
/// addressed by Z slice rather than by layer. Reporting one for those is the
/// truthful answer to "how many layers may it have", not a refusal.
fn max_array_layers(dimension_bit: D3D12_FORMAT_SUPPORT1) -> u32 {
    if dimension_bit == D3D12_FORMAT_SUPPORT1_TEXTURE2D {
        D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION
    } else {
        1
    }
}

/// Whether `support` carries one `D3D12_FORMAT_SUPPORT1` bit.
fn has_support1(support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT, bit: D3D12_FORMAT_SUPPORT1) -> bool {
    support.Support1.0 & bit.0 != 0
}

/// One format's quality-level count at each of [`SAMPLE_COUNTS`], in that order.
///
/// A named type rather than a bare array because two tables read it and the
/// index is what ties an entry to a sample count: a `[u32; 5]` passed between
/// them could be reordered without anything failing to compile.
type SampleQuality = [u32; SAMPLE_COUNTS.len()];

/// Asks how many quality levels `format` has at each of [`SAMPLE_COUNTS`].
///
/// One call per sample count, and `NumQualityLevels == 0` is the API's way of
/// saying the combination does not exist — a different answer from a refusal,
/// and the reason both readers of this need the number rather than a boolean.
fn quality_levels(device: &ID3D12Device, format: DXGI_FORMAT) -> RhiResult<SampleQuality> {
    let mut levels = [0u32; SAMPLE_COUNTS.len()];

    for (index, sample_count) in SAMPLE_COUNTS.iter().enumerate() {
        let mut data = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: format,
            SampleCount: *sample_count,
            Flags: D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS(0),
            NumQualityLevels: 0,
        };

        // SAFETY: `D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS` is defined to fill a
        // `D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS`, whose `Format`,
        // `SampleCount` and `Flags` members are the question and whose
        // `NumQualityLevels` is the answer. The out-parameter points at a live
        // value of exactly that type and the size handed over is that type's own
        // size.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                (&raw mut data).cast(),
                size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32,
            )
        }
        .map_err(|error| ffi::to_rhi(&error, "ID3D12Device::CheckFeatureSupport"))?;

        levels[index] = data.NumQualityLevels;
    }

    Ok(levels)
}

/// The dimensions a copy route can name, in portable-vocabulary order.
const ROUTE_DIMENSIONS: [TextureDimension; 3] = [
    TextureDimension::D1,
    TextureDimension::D2,
    TextureDimension::D3,
];

/// The aspects a copy route can name, in portable-vocabulary order.
const ROUTE_ASPECTS: [TextureAspect; 3] = [
    TextureAspect::Color,
    TextureAspect::Depth,
    TextureAspect::Stencil,
];

/// Records the one route that does not depend on a format.
///
/// # Why the buffer copy's alignment is one byte
///
/// `CopyBufferRegion` takes two byte offsets and a byte count, and Direct3D 12
/// states no placement requirement for any of them: the two alignment constants
/// this backend reads for the *texel* route are the only copy alignments the API
/// fixes, and they describe a texture footprint rather than a buffer. So the
/// constraint a buffer copy carries is that a copy starts and ends on a byte
/// boundary, which is one byte on both axes.
///
/// One rather than the zero that `BufferCopyLayoutLimits::validate` reads as "no
/// constraint imposed": zero is a sentinel a reader has to already know, while
/// one is the claim itself — every offset satisfies it, and it says what the
/// device accepts instead of asking the reader to consult a convention.
fn record_buffer_route(facts: &mut CapabilityFacts) {
    facts.record_route(
        RouteQuery::BufferToBuffer,
        RouteSupport::Supported(RouteCapabilities::new(
            Some(BufferCopyLayoutLimits::new(1, 1)),
            None,
        )),
    );
}

/// Records the direct copy routes of one format.
///
/// # What is walked, and why it is the legal combinations rather than all of them
///
/// A [`RouteQuery`]'s key space is not one a backend can walk in full — the two
/// texture-to-texture routes carry a `u32` sample count on each side, so the
/// cross product is five figures before any format is named — which puts this
/// table under the same obligation as the texture-support walk next to it: not
/// "fill everything" but "do not leave a legal combination unrecorded". An
/// unrecorded legal route is a refusal to execute an operation the device can
/// perform, and section 9.4 makes that refusal final rather than advisory.
///
/// The walk below is over the combinations this backend can lower for every
/// public input accepted by the route vocabulary. Buffer↔texture carries its
/// D3D12 image-placement and 3D-slice constraints in `TexelCopyLayoutLimits`,
/// so those rejections occur at recording rather than Phase A. The two rules
/// that decide texture-copy membership are:
///
/// - **A copy covers one plane, and only a plane the format has.** Section 15.3
///   makes the aspect set a property of the format, so a color format has no
///   depth route and a depth-only format has no color one. The formats in section
///   8.1's P0 set that carry a stencil plane are the depth-stencil ones.
/// - **A texture-to-texture copy does not convert.** `CopyTextureRegion` moves
///   texels between resources of the same format, dimensionality and sample
///   count; a differently-typed pair is not a copy that needs a capability, it is
///   a copy the API does not offer. The sample counts are therefore walked once
///   and used for both sides rather than crossed with each other.
fn record_format_routes(format: TextureFormat, facts: &mut CapabilityFacts) {
    // Keep this helper total over the portable enum as well as correct at its
    // probe call site: an abstract/mobile format with no exact DXGI resource
    // must not gain route rows merely because its portable aspect set is known.
    if dxgi_format(format).is_none() {
        return;
    }
    let aspects = format_aspects(format);

    // A texture-to-texture copy and a resolve state no alignment, and the empty
    // pair is how this type says so: a route that reports neither layout is one
    // that has no copy alignment to declare.
    let none = RouteCapabilities::new(None, None);

    let texel = RouteCapabilities::new(
        None,
        Some(
            TexelCopyLayoutLimits::new(
                u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT),
                D3D12_TEXTURE_DATA_PITCH_ALIGNMENT,
            )
            // Array layers are separate placed footprints.  3D depth slices
            // are one footprint, so they require packed rows but not a 512-byte
            // offset per slice.
            .with_image_layout(u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT), true),
        ),
    );
    for dimension in ROUTE_DIMENSIONS {
        for aspect in ROUTE_ASPECTS {
            if !aspects.contains(aspect_bits(aspect)) || !copy_aspect_is_lowered(aspect) {
                continue;
            }
            let supported = RouteSupport::Supported(texel);
            facts.record_route(
                RouteQuery::BufferToTexture {
                    dimension,
                    format,
                    aspect,
                },
                supported,
            );
            facts.record_route(
                RouteQuery::TextureToBuffer {
                    dimension,
                    format,
                    aspect,
                },
                supported,
            );
        }
    }

    for dimension in ROUTE_DIMENSIONS {
        for src_aspect in ROUTE_ASPECTS {
            if !aspects.contains(aspect_bits(src_aspect)) || !copy_aspect_is_lowered(src_aspect) {
                continue;
            }
            for dst_aspect in ROUTE_ASPECTS {
                // `CopyTextureRegion` is an exact same-plane move.  Do not
                // turn the fact matrix's independent source/destination
                // fields into an invented depth-to-stencil conversion route.
                // Stencil needs DXGI plane arithmetic this backend does not
                // lower, so it is absent rather than nominally supported.
                if src_aspect != dst_aspect
                    || !aspects.contains(aspect_bits(dst_aspect))
                    || !copy_aspect_is_lowered(dst_aspect)
                {
                    continue;
                }
                facts.record_route(
                    RouteQuery::TextureToTexture {
                        src_dimension: dimension,
                        src_format: format,
                        src_aspect,
                        src_sample_count: 1,
                        dst_dimension: dimension,
                        dst_format: format,
                        dst_aspect,
                        dst_sample_count: 1,
                    },
                    RouteSupport::Supported(none),
                );
            }
        }
    }
}

/// Planes whose route facts have an exact counterpart in
/// `command::transfer::texture_subresource`.
///
/// D3D12 plane zero is sufficient for ordinary color and depth-only copies.
/// Stencil and multi-planar resources need format-specific plane index and
/// footprint lowering; publishing them before that work exists would cause the
/// public route validator to accept commands the submit path must later refuse.
fn copy_aspect_is_lowered(aspect: TextureAspect) -> bool {
    matches!(aspect, TextureAspect::Color | TextureAspect::Depth)
}

/// Records the limits Direct3D 12 fixes for every device.
///
/// # Why these can be constants rather than probes
///
/// The distinction is the one `record_features` draws for anisotropy and
/// `record_texture_limits` draws for extents: a `D3D12_REQ_*` or
/// `D3D12_*_COUNT` constant is not a floor a driver may undershoot. Direct3D 12
/// defines these as the API's own shape — thirty-two vertex-buffer slots, eight
/// simultaneous render targets, a 2048-byte input element, a 1024-thread
/// workgroup — and a device that could not meet them would not be a Direct3D 12
/// device. Probing for them would be asking the driver to confirm the API it
/// implements.
///
/// The distinction from what [`record_limits`] records is worth keeping
/// straight, because both live in the limits table: that function records the
/// two keys whose value is a *device reading* (the resource address space) or an
/// *API range* (anisotropy). This one records the keys whose value is the API's
/// fixed shape. All three are the device's answer; none of them is a guess.
///
/// # What is deliberately left out
///
/// Several keys have no Direct3D 12 counterpart to cite, and they stay
/// unrecorded rather than being filled with a number borrowed from another API's
/// convention. `MaxBindGroups` and `MaxBindingsPerGroup` are Vulkan's
/// descriptor-set vocabulary — a D3D12 root signature has no such ceiling, so
/// the honest answer is not a large number but the absence of the constraint.
/// `MaxBindGroupsPlusVertexBuffers` is explicitly optional in
/// [`LimitKey`]'s own documentation for that reason.
/// `MaxInterStageShaderVariables`, `MaxColorAttachmentBytesPerSample` and the two
/// dynamic-buffer-per-layout keys are in the same position: D3D12 constrains
/// none of them as a stated ceiling, and a plausible-looking constant would be
/// this backend inventing a rule.
fn record_api_shape_limits(facts: &mut CapabilityFacts) {
    // Vertex input. Thirty-two of each is the API's count for both slots and
    // elements, and the stride bound is the same 2048 bytes the multi-element
    // structure limit sets for one element's total size.
    facts.record_limit(
        LimitKey::MaxVertexBuffers,
        u64::from(D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT),
    );
    facts.record_limit(
        LimitKey::MaxVertexAttributes,
        u64::from(D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT),
    );
    facts.record_limit(
        LimitKey::MaxVertexBufferArrayStride,
        u64::from(D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES),
    );

    // Simultaneous render targets. The portable key counts color attachments, and
    // D3D12's count is for render targets as a class — but the two coincide
    // because a depth-stencil view is not a render target in this enumeration.
    facts.record_limit(
        LimitKey::MaxColorAttachments,
        u64::from(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT),
    );

    // The compute workgroup contract. `MAX_THREADS_PER_GROUP` is the product
    // bound; the three per-axis bounds are what a shader's `numthreads` is
    // checked against, and they are recorded separately because they are
    // separate questions — a workgroup of 1024 threads is legal in several
    // shapes, and only some of them satisfy all three axes.
    facts.record_limit(
        LimitKey::MaxComputeInvocationsPerWorkgroup,
        u64::from(D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupSizeX,
        u64::from(D3D12_CS_THREAD_GROUP_MAX_X),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupSizeY,
        u64::from(D3D12_CS_THREAD_GROUP_MAX_Y),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupSizeZ,
        u64::from(D3D12_CS_THREAD_GROUP_MAX_Z),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupsPerDimension,
        u64::from(D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION),
    );

    // Group-shared memory, in bytes. The API states it as a register count and
    // one register is four bytes, so the multiplication is the unit conversion
    // rather than a derived guess — and the shift is written out so the
    // conversion is visible instead of being a magic 32768.
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupStorageSize,
        u64::from(D3D12_CS_TGSM_REGISTER_COUNT) * 4,
    );

    // The constant-buffer binding ceiling, which is the placement size the API
    // fixes. Its storage-buffer counterpart is *not* here: D3D12 states no
    // separate cap for a storage binding, so it is the resource address space —
    // the same reading `record_limits` already makes for `MaxBufferSize`, and it
    // is recorded there beside it rather than read a second time here.
    facts.record_limit(
        LimitKey::MaxUniformBufferBindingSize,
        u64::from(D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT) * 16,
    );

    // The two minimum alignments. These are the keys for which a *smaller* value
    // is the stronger one — `LimitKey::larger_is_stronger` says so — and both
    // numbers are the API's placement requirements rather than driver
    // preferences, which is what makes them reportable at all.
    facts.record_limit(
        LimitKey::MinUniformBufferOffsetAlignment,
        u64::from(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT),
    );
    facts.record_limit(
        LimitKey::MinStorageBufferOffsetAlignment,
        u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT),
    );
}

/// Contract tests for the rules that turn a probed support word into an answer.
///
/// These run without a GPU, and that is the point of them. The real-device tests
/// in the provider assert that the table is *wired up*: that the walk reaches
/// every dimension, that a real format answers, that a real reading arrives. What
/// they cannot assert is how a rule behaves on an input no device on this machine
/// produces — this machine reports quality levels for 4x, so if the quality-level
/// requirement were deleted, every real-device assertion would still pass and the
/// deletion would be invisible. Handing the rule a support word with no quality
/// levels at all is what makes that deletion visible.
///
/// The support words here are written by hand rather than probed. They stand for
/// formats a device may or may not have, and the rule has to be correct for both;
/// a test that could only be written against the one device in front of it would
/// be a description of that device rather than of the rule.
#[cfg(test)]
mod tests {
    use super::*;

    use crate::api::binding::{
        BindingCount, BindingKind, BindingSupportQuery, SamplerKind, TextureSampleType,
    };
    use crate::api::capability::EnabledCapabilities;
    use crate::api::command::BlitFilter;
    use crate::api::submission::SubmissionCapabilities;

    /// A support word carrying `first` in the first word and `second` in the
    /// second, for a format whose identity the rules do not consult.
    fn support(first: i32, second: i32) -> D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Support1: D3D12_FORMAT_SUPPORT1(first),
            Support2: D3D12_FORMAT_SUPPORT2(second),
        }
    }

    /// Asks `texture_answer` the way the walk does, with everything the rules do
    /// not vary held still.
    fn answer(
        dimension: D3D12_FORMAT_SUPPORT1,
        support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT,
        usage: TextureUsage,
        sample_count: u32,
        quality_levels: u32,
    ) -> TextureSupport {
        texture_answer(
            dimension,
            support,
            usage,
            sample_count,
            TextureViewCompatibility::NONE,
            quality_levels,
        )
    }

    #[test]
    fn a_multisampled_key_without_quality_levels_is_not_a_texture_that_exists() {
        let bits = D3D12_FORMAT_SUPPORT1_TEXTURE2D.0 | D3D12_FORMAT_SUPPORT1_RENDER_TARGET.0;
        let word = support(bits, 0);

        assert!(
            !answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &word,
                TextureUsage::COLOR_ATTACHMENT,
                4,
                1
            )
            .is_supported(),
            "the baseline must not advertise native MSAA allocation before its raster \
             attachment and resolve lowering are implemented"
        );

        assert!(
            !answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &word,
                TextureUsage::COLOR_ATTACHMENT,
                4,
                0
            )
            .is_supported(),
            "a sample count the device reports no quality level for does not exist, and \
             reading the zero as support would let a caller ask for a render target that \
             cannot be created"
        );

        assert!(
            answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &word,
                TextureUsage::COLOR_ATTACHMENT,
                1,
                0
            )
            .is_supported(),
            "a single-sampled texture has no quality levels to have, so the zero that \
             means \"absent\" for a multisampled key means nothing here"
        );
    }

    #[test]
    fn attachment_support_matches_the_implemented_2d_rtv_dsv_lowering() {
        let bits = D3D12_FORMAT_SUPPORT1_TEXTURE1D.0
            | D3D12_FORMAT_SUPPORT1_TEXTURE2D.0
            | D3D12_FORMAT_SUPPORT1_TEXTURE3D.0
            | D3D12_FORMAT_SUPPORT1_RENDER_TARGET.0
            | D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL.0;
        let word = support(bits, 0);

        assert!(
            !answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE1D,
                &word,
                TextureUsage::COLOR_ATTACHMENT,
                1,
                0
            )
            .is_supported()
        );
        assert!(
            answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &word,
                TextureUsage::COLOR_ATTACHMENT,
                1,
                0
            )
            .is_supported()
        );
        assert!(
            answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE3D,
                &word,
                TextureUsage::COLOR_ATTACHMENT,
                1,
                0
            )
            .is_supported()
        );
        for dimension in [
            D3D12_FORMAT_SUPPORT1_TEXTURE1D,
            D3D12_FORMAT_SUPPORT1_TEXTURE2D,
            D3D12_FORMAT_SUPPORT1_TEXTURE3D,
        ] {
            assert_eq!(
                answer(
                    dimension,
                    &word,
                    TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                    1,
                    0
                )
                .is_supported(),
                dimension == D3D12_FORMAT_SUPPORT1_TEXTURE2D
            );
        }
    }

    #[test]
    fn a_storage_texture_needs_both_unordered_access_bits() {
        let typed_uav = D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0;
        let base = D3D12_FORMAT_SUPPORT1_TEXTURE2D.0 | typed_uav;

        assert!(
            !answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &support(base, 0),
                TextureUsage::STORAGE,
                1,
                0
            )
            .is_supported(),
            "a typed UAV with neither access bit is not a storage texture a shader can use"
        );

        assert!(
            !answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &support(base, D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD.0),
                TextureUsage::STORAGE,
                1,
                0
            )
            .is_supported(),
            "a storage texture the portable layer describes is read-write, so a format \
             that can only be read through is reported as unsupported rather than as a \
             storage texture that half works"
        );

        let both = D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD.0 | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0;
        assert!(
            answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &support(base, both),
                TextureUsage::STORAGE,
                1,
                0
            )
            .is_supported(),
            "both access bits is the read-write storage texture the rule exists to accept"
        );
    }

    #[test]
    fn an_empty_usage_mask_has_no_legal_operation() {
        let empty = TextureUsage::all()
            .find(|usage| usage.is_empty())
            .expect("the walk starts at the empty mask, so it is reachable here");

        let bits = D3D12_FORMAT_SUPPORT1_TEXTURE2D.0
            | D3D12_FORMAT_SUPPORT1_RENDER_TARGET.0
            | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE.0;

        assert!(
            !answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &support(bits, 0),
                empty,
                1,
                0
            )
            .is_supported(),
            "a texture with no usage at all has no operation to be legal for, whatever \
             the format can do"
        );
    }

    #[test]
    fn an_answer_is_refused_when_the_format_cannot_back_the_dimension() {
        // The 2D bit alone: a format that is not a volume format, asked about as
        // a volume. The dimension bit is the first thing the rule consults, so
        // the rest of the word is deliberately generous.
        let bits = D3D12_FORMAT_SUPPORT1_TEXTURE2D.0
            | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE.0
            | D3D12_FORMAT_SUPPORT1_RENDER_TARGET.0;

        assert!(
            !answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE3D,
                &support(bits, 0),
                TextureUsage::SAMPLED,
                1,
                0
            )
            .is_supported(),
            "a format with no TEXTURE3D bit cannot back a volume texture however its \
             usage is spelled"
        );

        assert!(
            answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &support(bits, 0),
                TextureUsage::SAMPLED,
                1,
                0
            )
            .is_supported(),
            "the same word asked about the dimension it does have must be accepted, or \
             the test above would pass for the wrong reason"
        );
    }

    #[test]
    fn a_cube_view_needs_a_cube_capable_format_in_two_dimensions() {
        let flat = D3D12_FORMAT_SUPPORT1_TEXTURE2D.0 | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE.0;
        let cubed = flat | D3D12_FORMAT_SUPPORT1_TEXTURECUBE.0;
        let volume = cubed | D3D12_FORMAT_SUPPORT1_TEXTURE3D.0;

        let ask = |word: &D3D12_FEATURE_DATA_FORMAT_SUPPORT, dimension, compatibility| {
            texture_answer(dimension, word, TextureUsage::SAMPLED, 1, compatibility, 0)
        };

        assert!(
            !ask(
                &support(flat, 0),
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                TextureViewCompatibility::CUBE
            )
            .is_supported(),
            "a format without the cube bit cannot back a cube view, which Direct3D 12 \
             states at creation time rather than deriving from the dimension"
        );

        assert!(
            ask(
                &support(cubed, 0),
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                TextureViewCompatibility::CUBE
            )
            .is_supported(),
            "the cube bit in two dimensions is exactly what a cube view needs"
        );

        assert!(
            !ask(
                &support(volume, 0),
                D3D12_FORMAT_SUPPORT1_TEXTURE3D,
                TextureViewCompatibility::CUBE
            )
            .is_supported(),
            "a volume texture is not an array of six faces, so the cube bit alone must \
             not make one cube-capable"
        );

        assert!(
            ask(
                &support(flat, 0),
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                TextureViewCompatibility::NONE
            )
            .is_supported(),
            "the cube bit is a question about the view and not about the format, so a \
             format without it still answers the ordinary question"
        );
    }

    #[test]
    fn a_mip_chain_is_no_longer_than_the_extent_it_is_built_from() {
        assert_eq!(max_mip_levels(D3D12_FORMAT_SUPPORT1_TEXTURE2D), 15);
        assert_eq!(max_mip_levels(D3D12_FORMAT_SUPPORT1_TEXTURE1D), 15);

        // 2048 is three halvings short of 16384, so a volume texture's chain is
        // capped by its own extent rather than by the 15 levels the API allows.
        // Recorded as a derived number rather than as a literal because that is
        // the claim: the two come from one another.
        assert_eq!(max_mip_levels(D3D12_FORMAT_SUPPORT1_TEXTURE3D), 12);
    }

    #[test]
    fn dxgi_mapping_keeps_bc_native_and_mobile_codecs_refused() {
        assert_eq!(
            dxgi_format(TextureFormat::Bc1RgbaUnorm),
            Some(DXGI_FORMAT_BC1_UNORM)
        );
        assert_eq!(
            dxgi_format(TextureFormat::Bc7RgbaUnormSrgb),
            Some(DXGI_FORMAT_BC7_UNORM_SRGB)
        );
        assert_eq!(dxgi_format(TextureFormat::Etc2Rgba8Unorm), None);
        assert_eq!(dxgi_format(TextureFormat::Astc4x4Unorm), None);
        assert_eq!(dxgi_format(TextureFormat::Astc4x4Hdr), None);
    }

    #[test]
    fn an_extent_is_the_one_the_dimension_actually_has() {
        assert_eq!(max_extent(D3D12_FORMAT_SUPPORT1_TEXTURE1D).height, 1);
        assert_eq!(max_extent(D3D12_FORMAT_SUPPORT1_TEXTURE1D).depth, 1);
        assert_eq!(
            max_extent(D3D12_FORMAT_SUPPORT1_TEXTURE2D).width,
            D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
        );
        assert_eq!(
            max_extent(D3D12_FORMAT_SUPPORT1_TEXTURE3D).depth,
            D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION
        );

        // Layers belong to the two dimensions that can be arrays. Reporting one
        // for a volume is the truthful answer to "how many layers may it have"
        // rather than a refusal, which is why it is asserted and not left out.
        assert_eq!(
            max_array_layers(D3D12_FORMAT_SUPPORT1_TEXTURE2D),
            D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION
        );
        assert_eq!(max_array_layers(D3D12_FORMAT_SUPPORT1_TEXTURE3D), 1);
        assert_eq!(max_array_layers(D3D12_FORMAT_SUPPORT1_TEXTURE1D), 1);
    }

    /// Wraps a filled record the way a completed device request would.
    ///
    /// The same three lines `api::capability`'s own tests use, repeated rather
    /// than shared: the rule under test here is a *fill*, and a helper that lived
    /// in the capability module would be that module asserting about a backend's
    /// enumeration.
    fn enabled_from(facts: CapabilityFacts) -> EnabledCapabilities {
        EnabledCapabilities::from_facts(facts, SubmissionCapabilities::new(Vec::new()))
    }

    /// Whether the formatted routes include `query`.
    fn routed(format: TextureFormat, query: &RouteQuery) -> bool {
        let mut facts = CapabilityFacts::empty();
        record_format_routes(format, &mut facts);
        enabled_from(facts).route(query).is_supported()
    }

    /// Section 9.4's refusal, exercised on the one operation Direct3D 12 has no
    /// path for at all.
    ///
    /// This is the case the route table exists for, and it is a *structural*
    /// negative rather than a probed one: Direct3D 12 has `CopyBufferRegion`,
    /// `CopyTextureRegion`, `CopyResource`, `CopyTiles` and `ResolveSubresource`,
    /// and no filtered or scaled blit at any of them. A backend that answered
    /// `Supported` here would be promising a lowering section 9.4 forbids it to
    /// perform silently, so the negative is the only honest answer and the walk
    /// records nothing — every blit key falls to the refusal.
    #[test]
    fn a_filtered_blit_has_no_direct_route_and_the_walk_records_none() {
        for filter in [BlitFilter::Nearest, BlitFilter::Linear] {
            assert!(
                !routed(
                    TextureFormat::Rgba8Unorm,
                    &RouteQuery::Blit {
                        src_dimension: TextureDimension::D2,
                        src_format: TextureFormat::Rgba8Unorm,
                        dst_dimension: TextureDimension::D2,
                        dst_format: TextureFormat::Rgba8Unorm,
                        filter,
                    },
                ),
                "Direct3D 12 has no blit for {filter:?} to lower onto"
            );
        }
    }

    /// Resolve remains absent until a command lowering exists. A D3D12 format
    /// support bit alone is not an end-to-end backend capability.
    #[test]
    fn resolve_is_not_advertised_before_command_lowering_exists() {
        let key = RouteQuery::Resolve {
            format: TextureFormat::Rgba8Unorm,
            src_sample_count: 4,
        };
        assert!(
            !routed(TextureFormat::Rgba8Unorm, &key),
            "a route must stay unsupported until Dx12CommandSpine lowers ResolveSubresource"
        );
    }

    /// The dedicated baseline refuses MSAA resources as well as MSAA copies.
    #[test]
    fn texture_copy_is_advertised_only_for_the_1x_baseline() {
        let supported = RouteQuery::TextureToTexture {
            src_dimension: TextureDimension::D2,
            src_format: TextureFormat::Rgba8Unorm,
            src_aspect: TextureAspect::Color,
            src_sample_count: 1,
            dst_dimension: TextureDimension::D2,
            dst_format: TextureFormat::Rgba8Unorm,
            dst_aspect: TextureAspect::Color,
            dst_sample_count: 1,
        };
        let multisampled = RouteQuery::TextureToTexture {
            src_dimension: TextureDimension::D2,
            src_format: TextureFormat::Rgba8Unorm,
            src_aspect: TextureAspect::Color,
            src_sample_count: 4,
            dst_dimension: TextureDimension::D2,
            dst_format: TextureFormat::Rgba8Unorm,
            dst_aspect: TextureAspect::Color,
            dst_sample_count: 4,
        };
        assert!(routed(TextureFormat::Rgba8Unorm, &supported));
        assert!(!routed(TextureFormat::Rgba8Unorm, &multisampled));
    }

    #[test]
    fn binding_facts_follow_the_command_lowering_stage_boundary() {
        let mut facts = CapabilityFacts::empty();
        record_binding_support(&mut facts);
        let enabled = enabled_from(facts);
        let sampled = |visibility| BindingSupportQuery {
            visibility,
            kind: BindingKind::SampledTexture {
                dimension: TextureViewDimension::D2,
                sample_type: TextureSampleType::Float,
                multisampled: false,
            },
            count: BindingCount::One,
            dynamic_offset: false,
        };
        let sampler = |visibility| BindingSupportQuery {
            visibility,
            kind: BindingKind::Sampler {
                kind: SamplerKind::Filtering,
            },
            count: BindingCount::One,
            dynamic_offset: false,
        };

        assert_eq!(
            enabled.binding_support(&sampled(crate::api::shader::ShaderStages::FRAGMENT)),
            BindingSupport::Supported,
            "raster lowering transitions and binds sampled texture descriptor tables"
        );
        assert_eq!(
            enabled.binding_support(&sampled(crate::api::shader::ShaderStages::COMPUTE)),
            BindingSupport::Unsupported,
            "compute lowering refuses texture ResourceUse until texture transitions are lowered"
        );
        assert_eq!(
            enabled.binding_support(&sampler(crate::api::shader::ShaderStages::COMPUTE)),
            BindingSupport::Supported,
            "compute lowering binds both descriptor heaps and the sampler root table"
        );
        assert_eq!(
            enabled.binding_support(&sampler(crate::api::shader::ShaderStages::FRAGMENT)),
            BindingSupport::Supported,
            "raster lowering binds both descriptor heaps and its graphics sampler root table"
        );
    }

    /// A route names a plane, and only a plane the format has.
    ///
    /// Section 15.3 makes the aspect set a property of the format rather than of
    /// the device, so the walk derives membership from the format name and the
    /// test asserts it on both sides of the line: a depth-stencil format has a
    /// stencil route, a colour format does not.
    #[test]
    fn a_copy_route_exists_only_for_a_plane_the_format_has() {
        let stencil_of = |format: TextureFormat, aspect: TextureAspect| {
            routed(
                format,
                &RouteQuery::TextureToBuffer {
                    dimension: TextureDimension::D2,
                    format,
                    aspect,
                },
            )
        };

        // These abstract formats intentionally have no single DXGI backing, so
        // they have no route rows at all; format aspect membership alone must
        // not manufacture an executable plane route.
        assert!(!stencil_of(
            TextureFormat::Depth24PlusStencil8,
            TextureAspect::Stencil
        ));
        assert!(!stencil_of(
            TextureFormat::Depth24PlusStencil8,
            TextureAspect::Depth
        ));
        assert!(!stencil_of(
            TextureFormat::Rgba8Unorm,
            TextureAspect::Stencil
        ));
        assert!(!stencil_of(TextureFormat::Rgba8Unorm, TextureAspect::Depth));
        assert!(stencil_of(TextureFormat::Rgba8Unorm, TextureAspect::Color));

        // A depth-only format is the third case, and the one that separates
        // "has a stencil" from "is not a colour format".
        assert!(stencil_of(
            TextureFormat::Depth32Float,
            TextureAspect::Depth
        ));
        assert!(!stencil_of(
            TextureFormat::Depth32Float,
            TextureAspect::Stencil
        ));
    }
}
