//! DX12 pipeline-library ownership, persistence, and stable entry names.
//!
//! A `PipelineLibrary` is useful only when a pipeline creation actually calls
//! `Load*Pipeline`/`StorePipeline`; this module owns that native object while the
//! compute and raster lowerers own their descriptor-specific calls.

use std::any::Any;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Device1, ID3D12PipelineLibrary};
use windows::core::Interface;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::backend::PipelineCacheBackend;
use crate::api::pipeline::{
    ComputePipelineDescriptor, PipelineCache, PipelineCacheDescriptor, PipelineCacheFallback,
    PipelineCacheValidationKey, RasterPipelineDescriptor,
};
use crate::api::platform::DeviceLossInfo;
use crate::backend::dx12::failure::Dx12Failure;
use crate::backend::dx12::platform::device::Dx12LossState;

/// Native cache guarded because applications may asynchronously create several
/// pipelines against one portable cache handle.
pub(crate) struct Dx12PipelineCache {
    library: Mutex<ID3D12PipelineLibrary>,
    // PipelineLibrary can itself be the first native call that observes device
    // removal (notably Serialize, which is invoked through the cache handle
    // rather than Device).  This is the device's existing one-way authority,
    // not a second cache-local liveness model.
    loss: Arc<Dx12LossState>,
}

impl Dx12PipelineCache {
    pub(crate) fn library(&self) -> std::sync::MutexGuard<'_, ID3D12PipelineLibrary> {
        self.library
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl PipelineCacheBackend for Dx12PipelineCache {
    fn serialized_data(&self) -> RhiResult<Vec<u8>> {
        let library = self.library();
        let mut bytes = vec![0; unsafe { library.GetSerializedSize() }];
        unsafe { library.Serialize(&mut bytes) }.map_err(|error| {
            observe_native_failure(
                &self.loss,
                crate::backend::dx12::ffi::NativeError::new(
                    &error,
                    "ID3D12PipelineLibrary::Serialize",
                ),
                "Dx12PipelineCache::serialized_data",
            )
        })?;
        Ok(bytes)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Creates a cache only on Device1+, which is where PipelineLibrary was added.
pub(crate) fn create_pipeline_cache(
    device: &ID3D12Device,
    loss: Arc<Dx12LossState>,
    descriptor: &PipelineCacheDescriptor,
) -> RhiResult<(Box<dyn PipelineCacheBackend>, PipelineCacheValidationKey)> {
    let device1 = device.cast::<ID3D12Device1>().map_err(|error| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            format!("DX12 PipelineLibrary requires ID3D12Device1: {error}"),
        )
        .at("Dx12Device::create_pipeline_cache")
    })?;
    let key = validation_key(device);
    let initial = match (
        descriptor.initial_data.as_deref(),
        descriptor.validation_key,
    ) {
        (Some(data), Some(saved)) if saved == key => data,
        (Some(_), Some(_)) if descriptor.fallback == PipelineCacheFallback::IgnoreInvalidData => {
            &[]
        }
        (Some(_), Some(_)) => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "pipeline cache bytes were created for a different DX12 adapter contract",
            )
            .at("Dx12Device::create_pipeline_cache"));
        }
        (None, None) => &[],
        _ => unreachable!("portable cache validation rejects partial initial data"),
    };
    let library = match unsafe { device1.CreatePipelineLibrary::<ID3D12PipelineLibrary>(initial) } {
        Ok(library) => library,
        Err(error) if descriptor.fallback == PipelineCacheFallback::IgnoreInvalidData => {
            let failure = crate::backend::dx12::ffi::NativeError::new(
                &error,
                "ID3D12Device1::CreatePipelineLibrary",
            );
            // IgnoreInvalidData recovers corrupted *cache bytes*, never a
            // removed device.  Retrying a terminal failure would conceal
            // the first loss observation behind an empty-cache fallback.
            if failure.failure().is_terminal() {
                return Err(observe_native_failure(
                    &loss,
                    failure,
                    "Dx12Device::create_pipeline_cache",
                ));
            }
            unsafe { device1.CreatePipelineLibrary::<ID3D12PipelineLibrary>(&[]) }.map_err(
                |fallback| {
                    observe_native_failure(
                        &loss,
                        crate::backend::dx12::ffi::NativeError::new(
                            &fallback,
                            "ID3D12Device1::CreatePipelineLibrary(empty)",
                        ),
                        "Dx12Device::create_pipeline_cache",
                    )
                },
            )?
        }
        Err(error) => {
            return Err(observe_native_failure(
                &loss,
                crate::backend::dx12::ffi::NativeError::new(
                    &error,
                    "ID3D12Device1::CreatePipelineLibrary",
                ),
                "Dx12Device::create_pipeline_cache",
            ));
        }
    };
    Ok((
        Box::new(Dx12PipelineCache {
            library: Mutex::new(library),
            loss,
        }),
        key,
    ))
}

/// The cache is reached directly by a public handle, outside `Dx12Device`'s
/// usual backend verb methods.  Keep its terminal HRESULT observation exactly
/// equivalent to that authority: first loss wins, all later cache calls expose
/// the same stable `DeviceLost` error.
fn observe_native_failure(
    loss: &Dx12LossState,
    failure: crate::backend::dx12::ffi::NativeError,
    operation: &'static str,
) -> RhiError {
    if failure.failure().is_terminal() {
        loss.mark_lost(DeviceLossInfo::new(format!(
            "Direct3D 12 reported a terminal failure in {operation}: {}",
            failure.as_error()
        )));
    }
    if let Some(info) = loss.loss_info() {
        RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned()).at(operation)
    } else {
        failure.into_rhi()
    }
}

pub(crate) fn native_cache(
    cache: Option<&PipelineCache>,
) -> Result<Option<&Dx12PipelineCache>, Dx12Failure> {
    let Some(cache) = cache else {
        return Ok(None);
    };
    cache
        .native()
        .as_any()
        .downcast_ref::<Dx12PipelineCache>()
        .map(Some)
        .ok_or(Dx12Failure::Unsupported {
            what: "a pipeline cache this DX12 device did not create",
            why: "its native PipelineLibrary belongs to another backend",
        })
}

/// Device LUID plus the Fluxel cache-key schema. The driver validates actual
/// library bytes at restore time; a changed driver that cannot consume them is
/// therefore rejected or falls back according to the portable descriptor.
fn validation_key(device: &ID3D12Device) -> PipelineCacheValidationKey {
    let luid = unsafe { device.GetAdapterLuid() };
    let mut bytes = [0; 32];
    bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
    bytes[4..8].copy_from_slice(&luid.HighPart.to_le_bytes());
    bytes[8..16].copy_from_slice(b"fluxel13");
    PipelineCacheValidationKey::from_bytes(bytes)
}

pub(crate) fn compute_name(descriptor: &ComputePipelineDescriptor) -> Vec<u16> {
    let mut key = Vec::new();
    key.extend_from_slice(b"compute-v1");
    key.extend_from_slice(&descriptor.shader.artifact().content_hash.0);
    key.extend_from_slice(&descriptor.interface.descriptor().canonical_bytes());
    name(key)
}

pub(crate) fn raster_name(descriptor: &RasterPipelineDescriptor) -> Vec<u16> {
    let mut key = Vec::new();
    key.extend_from_slice(b"raster-v1");
    key.extend_from_slice(&descriptor.vertex.artifact().content_hash.0);
    if let Some(fragment) = &descriptor.fragment {
        key.extend_from_slice(&fragment.artifact().content_hash.0);
    }
    key.extend_from_slice(&descriptor.interface.descriptor().canonical_bytes());
    // These data-only portable states are the complete graphics PSO shape after
    // excluding shader/interface/cache. Debug output is deterministic for the
    // current frozen enum/struct schema; the explicit v1 prefix invalidates old
    // cache names if that schema changes.
    key.extend_from_slice(
        format!(
            "{:?}{:?}{:?}{:?}{:?}{:?}",
            descriptor.vertex_input,
            descriptor.primitive,
            descriptor.depth_stencil,
            descriptor.multisample,
            descriptor.multiview_mask,
            descriptor.color_targets
        )
        .as_bytes(),
    );
    name(key)
}

fn name(bytes: Vec<u8>) -> Vec<u16> {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    format!("fluxel-{hash:016x}\0").encode_utf16().collect()
}
