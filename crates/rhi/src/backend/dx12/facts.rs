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
//! - **Absent, and deliberately not guessed.** Texture, binding, route and
//!   view-compatibility facts are not recorded yet, and of the twenty-seven
//!   [`crate::api::platform::LimitKey`]s only two are. That is a real coverage gap
//!   and it is recorded as one rather than papered over: see [`probe`] for what a
//!   caller observes while it stands, and [`record_limits`] for why the missing
//!   limits are a *mapping* problem rather than a probing one.
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
    D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS,
    D3D12_FEATURE_DATA_FORMAT_SUPPORT, D3D12_FEATURE_FORMAT_SUPPORT,
    D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE, ID3D12Device,
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
use crate::api::format::{FormatFacts, StorageAccessSupport, TextureFormat};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
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
/// The four accessors whose key spaces are not enumerable answer the negative
/// when their table has no entry, which is
/// [`crate::api::capability::CapabilityFacts`]'s rule and not a defect here: a
/// texture, binding, or route question is answered `Unsupported` rather than
/// answered wrongly. `limit` answers `None` for every key. Both are conservative
/// — they refuse work the hardware can do — and neither is silent, which is the
/// property that matters: the gap costs throughput, not correctness, and it is
/// recorded here rather than left to be discovered.
pub(super) fn probe(device: &ID3D12Device) -> RhiResult<CapabilityFacts> {
    let options = options(device)?;

    let mut facts = CapabilityFacts::empty();
    record_features(&mut facts);
    record_limits(&options, &mut facts);
    record_buffer_support(&mut facts);
    record_format_facts(device, &mut facts)?;

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

/// Records what each portable format's storage access is on this device.
fn record_format_facts(device: &ID3D12Device, facts: &mut CapabilityFacts) -> RhiResult<()> {
    for format in TextureFormat::all() {
        // A portable format with no single DXGI format is one this backend cannot
        // describe, and it is left out rather than recorded as unsupported. The
        // two are different answers: `format()` returning `None` says the device
        // was not asked, and a recorded `StorageAccessSupport::new(false, false,
        // false)` would say it was asked and said no. Only one of those is true.
        let Some(dxgi) = dxgi_format(format) else {
            continue;
        };

        let support = format_support(device, dxgi)?;

        // Direct3D 12 answers the three storage accesses with two independent
        // bits, not three: `UAV_TYPED_LOAD` and `UAV_TYPED_STORE`. Read-write is
        // therefore the conjunction rather than a third probe, and it is
        // recorded as such instead of being reported from a bit that does not
        // exist. The two stay separate in the record — a format can load without
        // storing, which is a shader that reads a storage texture it must not
        // write — for the reason `StorageAccessSupport` documents.
        let read_only = has_support2(&support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD);
        let write_only = has_support2(&support, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);

        facts.record_format(
            format,
            FormatFacts::new(
                format,
                StorageAccessSupport::new(read_only, write_only, read_only && write_only),
            ),
        );
    }

    Ok(())
}

/// Whether `support` carries one `D3D12_FORMAT_SUPPORT2` bit.
///
/// Compared through the raw word rather than through a `contains` helper so that
/// the test is visibly a mask test on the driver's own bit, and so that a bit the
/// binding crate renames cannot quietly change the meaning of this predicate.
fn has_support2(
    support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    bit: windows::Win32::Graphics::Direct3D12::D3D12_FORMAT_SUPPORT2,
) -> bool {
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
