//! What a live Direct3D 12 device reports about itself, read through
//! `CheckFeatureSupport`.
//!
//! This module owns one step and no others: turning a created `ID3D12Device`
//! into a [`CapabilityFacts`]. It decides nothing about legality, holds no
//! resource, and is not consulted again after device creation — section 7.2
//! makes an enabled contract immutable, so the table built here is the table the
//! device reports for its whole life.
//!
//! # Why this runs at device creation and not before
//!
//! Direct3D 12 answers every capability question through a device. `DXGI` can
//! hand out an adapter name and a vendor id without one, but format support,
//! resource-binding tiers and multisample quality levels all come from
//! `ID3D12Device::CheckFeatureSupport`, and there is no adapter-level spelling of
//! them. That dependency — not a preference — is why
//! [`crate::base::platform::ProviderBackend::enumerate_adapters`] refuses while
//! `request_device` works, and why an [`crate::api::platform::AdapterInfo`] built
//! from `DXGI` alone carries no capability snapshot yet.
//!
//! # What is probed, what is structural, and what is still absent
//!
//! Being explicit about the three, because "the table is full" and "the table is
//! complete" are different claims and only one of them is true here.
//!
//! - **Probed.** Per-format storage access comes from
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
//! - **Absent, and deliberately not guessed.** Binding, route and
//!   view-compatibility facts are not recorded yet, and seven of the twenty-seven
//!   [`crate::api::platform::LimitKey`]s are not either. The seven are the ones
//!   that name a ceiling Direct3D 12 does not state, and
//!   [`record_api_shape_limits`] lists them rather than filling them with a
//!   number borrowed from another API's convention. Texture facts and the other
//!   twenty limits *are* recorded. What is left is a real coverage gap and it is
//!   recorded as one rather than papered over: see [`probe`] for what a caller
//!   observes while it stands, and [`record_limits`] for why the missing limits
//!   are a *mapping* problem rather than a probing one.
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

use windows::Win32::Graphics::Direct3D12::{
    D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
    D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, D3D12_CS_TGSM_REGISTER_COUNT,
    D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP, D3D12_CS_THREAD_GROUP_MAX_X,
    D3D12_CS_THREAD_GROUP_MAX_Y, D3D12_CS_THREAD_GROUP_MAX_Z, D3D12_FEATURE_D3D12_OPTIONS,
    D3D12_FEATURE_DATA_D3D12_OPTIONS, D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS, D3D12_FEATURE_FORMAT_SUPPORT,
    D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, D3D12_FORMAT_SUPPORT1,
    D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL, D3D12_FORMAT_SUPPORT1_RENDER_TARGET,
    D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE, D3D12_FORMAT_SUPPORT1_TEXTURE1D,
    D3D12_FORMAT_SUPPORT1_TEXTURE2D, D3D12_FORMAT_SUPPORT1_TEXTURE3D,
    D3D12_FORMAT_SUPPORT1_TEXTURECUBE, D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW,
    D3D12_FORMAT_SUPPORT2, D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD,
    D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE, D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT,
    D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT, D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS,
    D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT, D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT,
    D3D12_REQ_MIP_LEVELS, D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES,
    D3D12_REQ_TEXTURE1D_U_DIMENSION, D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
    D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION, D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
    D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT, ID3D12Device,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_UNORM,
    DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_R8G8_SNORM, DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_R8G8_UNORM,
    DXGI_FORMAT_R8G8B8A8_SINT, DXGI_FORMAT_R8G8B8A8_SNORM, DXGI_FORMAT_R8G8B8A8_UINT,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R16_FLOAT,
    DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_SINT,
    DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_SINT,
    DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_SINT,
    DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_SINT,
    DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_SINT,
    DXGI_FORMAT_R32G32B32A32_UINT,
};

use crate::api::capability::CapabilityFacts;
use crate::api::error::RhiResult;
use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery,
};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
use crate::api::resource::texture::{
    Extent3d, TextureDimension, TextureUsage, TextureViewCompatibility,
};
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
/// The accessors whose key spaces are not enumerable answer the negative when
/// their table has no entry, which is
/// [`crate::api::capability::CapabilityFacts`]'s rule and not a defect here: a
/// binding, route or view-compatibility question is answered `Unsupported`
/// rather than answered wrongly, and `limit` answers `None` for the seven keys
/// with no Direct3D 12 ceiling to cite. Both are conservative — they refuse work
/// the hardware can do — and neither is silent, which is the property that
/// matters: the gap costs throughput, not correctness, and it is recorded here
/// rather than left to be discovered.
pub(super) fn probe(device: &ID3D12Device) -> RhiResult<CapabilityFacts> {
    let options = options(device)?;

    let mut facts = CapabilityFacts::empty();
    record_features(&mut facts);
    record_limits(&options, &mut facts);
    record_buffer_support(&mut facts);
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

        record_format_facts(format, &support, &mut facts);
        record_texture_support(device, format, dxgi, &support, &mut facts)?;
    }

    Ok(facts)
}

/// Records the three optional features this backend can answer for.
///
/// Takes no device and no probe result, deliberately: every feature here is
/// structural, so a parameter would be an unused argument that reads like a
/// question being asked. [`record_limits`] is the one that takes the options
/// struct, because the limit it derives is a real device reading.
fn record_features(facts: &mut CapabilityFacts) {
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
        } else {
            BufferSupport::Supported(limits)
        };
        facts.record_buffer_support(usage, support);
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

    facts.record_format(
        format,
        FormatFacts::new(
            format,
            StorageAccessSupport::new(read_only, write_only, read_only && write_only),
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
fn dxgi_format(format: TextureFormat) -> Option<DXGI_FORMAT> {
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
        TextureFormat::Rg16Uint => DXGI_FORMAT_R16G16_UINT,
        TextureFormat::Rg16Sint => DXGI_FORMAT_R16G16_SINT,
        TextureFormat::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        TextureFormat::Rgba16Uint => DXGI_FORMAT_R16G16B16A16_UINT,
        TextureFormat::Rgba16Sint => DXGI_FORMAT_R16G16B16A16_SINT,
        TextureFormat::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        TextureFormat::R32Uint => DXGI_FORMAT_R32_UINT,
        TextureFormat::R32Sint => DXGI_FORMAT_R32_SINT,
        TextureFormat::R32Float => DXGI_FORMAT_R32_FLOAT,
        TextureFormat::Rg32Uint => DXGI_FORMAT_R32G32_UINT,
        TextureFormat::Rg32Sint => DXGI_FORMAT_R32G32_SINT,
        TextureFormat::Rg32Float => DXGI_FORMAT_R32G32_FLOAT,
        TextureFormat::Rgba32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
        TextureFormat::Rgba32Sint => DXGI_FORMAT_R32G32B32A32_SINT,
        TextureFormat::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        TextureFormat::Depth16Unorm => DXGI_FORMAT_D16_UNORM,
        TextureFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        TextureFormat::Depth32FloatStencil8 => DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => {
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
    device: &ID3D12Device,
    format: TextureFormat,
    dxgi: DXGI_FORMAT,
    support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    facts: &mut CapabilityFacts,
) -> RhiResult<()> {
    // The quality levels each sample count has, asked once per count rather than
    // once per key. `NumQualityLevels == 0` is the API's way of saying the
    // combination does not exist, which is a different answer from a refusal.
    let mut quality = [0u32; SAMPLE_COUNTS.len()];
    for (index, count) in SAMPLE_COUNTS.iter().enumerate() {
        quality[index] = quality_levels(device, dxgi, *count)?;
    }

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

    Ok(())
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

/// Asks how many quality levels `format` has at `sample_count`.
fn quality_levels(device: &ID3D12Device, format: DXGI_FORMAT, sample_count: u32) -> RhiResult<u32> {
    let mut data = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
        Format: format,
        SampleCount: sample_count,
        Flags: D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS(0),
        NumQualityLevels: 0,
    };

    // SAFETY: `D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS` is defined to fill a
    // `D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS`, whose `Format`,
    // `SampleCount` and `Flags` members are the question and whose
    // `NumQualityLevels` is the answer. The out-parameter points at a live value
    // of exactly that type and the size handed over is that type's own size.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
            (&raw mut data).cast(),
            size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32,
        )
    }
    .map_err(|error| ffi::to_rhi(&error, "ID3D12Device::CheckFeatureSupport"))?;

    Ok(data.NumQualityLevels)
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
            answer(
                D3D12_FORMAT_SUPPORT1_TEXTURE2D,
                &word,
                TextureUsage::COLOR_ATTACHMENT,
                4,
                1
            )
            .is_supported(),
            "a format that renders at 4x with a quality level is the case Direct3D 12 \
             mandates, and the rule must not refuse it"
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
}
