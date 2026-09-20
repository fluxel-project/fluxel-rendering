//! Contract tests for opaque external sources and pipeline-cache descriptors.

use std::any::Any;
use std::sync::Arc;

use crate::api::RhiErrorKind;
use crate::api::binding::{BindingKind, BindingResource};
use crate::api::external::{
    ExternalImageSource, ExternalImageSourceBackend, ExternalMemoryCapabilities,
    ExternalMemoryHandleType, ExternalMemoryTextureSource, ExternalMemoryTextureSourceBackend,
    ExternalTexture, ExternalTextureDescriptor, ExternalTextureImportDescriptor,
};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::pipeline::{
    PipelineCacheDescriptor, PipelineCacheFallback, PipelineCacheValidationKey,
};
use crate::api::platform::OptionalFeature;
use crate::api::resource::{Extent3d, TextureDescriptor, TextureUsage};
use crate::api::tests::mock::{
    device_with_features_for_test, external_memory_device_for_test, paired_device_for_test,
};

struct TestExternalImage;
impl ExternalImageSourceBackend for TestExternalImage {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
struct TestExternalMemory;
impl ExternalMemoryTextureSourceBackend for TestExternalMemory {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn source(
    identity: DeviceIdentity,
    extent: Extent3d,
) -> crate::api::RhiResult<ExternalImageSource> {
    ExternalImageSource::new(
        ObjectId::new(700),
        identity,
        extent,
        Box::new(TestExternalImage),
    )
}

#[test]
fn external_image_source_keeps_device_identity_native_backing_and_extent() {
    let identity = DeviceIdentity::new(DeviceInstanceId::new(8));
    let source = source(identity, Extent3d::d2(1920, 1080))
        .expect("a platform bridge attaches native backing before publishing a handle");
    assert_eq!(source.id(), ObjectId::new(700));
    assert_eq!(source.device_identity(), identity);
    assert_eq!(source.extent(), Extent3d::d2(1920, 1080));
}

#[test]
fn external_image_source_refuses_zero_or_non_2d_extent_before_backend_use() {
    let identity = DeviceIdentity::new(DeviceInstanceId::new(8));
    assert_eq!(
        source(identity, Extent3d::d2(0, 1)).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        source(identity, Extent3d::d3(1, 1, 2)).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn external_memory_capabilities_are_per_handle_class_not_a_coarse_boolean() {
    let capabilities = ExternalMemoryCapabilities {
        supported_handle_types: vec![ExternalMemoryHandleType::DmaBuf],
    };
    assert!(capabilities.supports(ExternalMemoryHandleType::DmaBuf));
    assert!(!capabilities.supports(ExternalMemoryHandleType::OpaqueFd));
    assert!(!capabilities.supports(ExternalMemoryHandleType::Win32Handle));
}

#[test]
fn external_memory_import_descriptor_can_only_use_extension_owned_contract() {
    let identity = DeviceIdentity::new(DeviceInstanceId::new(8));
    let texture =
        TextureDescriptor::new_2d(64, 64, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED);
    let source = ExternalMemoryTextureSource::new(
        ObjectId::new(702),
        identity,
        ExternalMemoryHandleType::DmaBuf,
        texture,
        Box::new(TestExternalMemory),
    );
    let import = ExternalTextureImportDescriptor::new(source.clone()).with_label("shared-image");
    assert_eq!(import.source.id(), ObjectId::new(702));
    assert_eq!(import.source.device_identity(), identity);
    assert_eq!(
        import.source.handle_type(),
        ExternalMemoryHandleType::DmaBuf
    );
    assert_eq!(
        import.source.texture_descriptor().extent,
        Extent3d::d2(64, 64)
    );
    assert_eq!(import.label.as_deref(), Some("shared-image"));
}

#[test]
fn external_memory_source_identity_is_not_interchangeable_across_contexts() {
    let first = ExternalMemoryTextureSource::new(
        ObjectId::new(703),
        DeviceIdentity::new(DeviceInstanceId::new(8)),
        ExternalMemoryHandleType::DmaBuf,
        TextureDescriptor::new_2d(1, 1, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED),
        Box::new(TestExternalMemory),
    );
    let second = ExternalMemoryTextureSource::new(
        ObjectId::new(704),
        DeviceIdentity::new(DeviceInstanceId::new(9)),
        ExternalMemoryHandleType::DmaBuf,
        TextureDescriptor::new_2d(1, 1, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED),
        Box::new(TestExternalMemory),
    );
    assert_ne!(first.device_identity(), second.device_identity());
    assert_ne!(first.id(), second.id());
}

#[test]
fn external_texture_has_its_own_binding_kind_not_a_forged_texture_view() {
    let source = source(
        DeviceIdentity::new(DeviceInstanceId::new(8)),
        Extent3d::d2(64, 64),
    )
    .unwrap();
    let texture = ExternalTexture::new(
        ObjectId::new(701),
        DeviceIdentity::new(DeviceInstanceId::new(8)),
        ExternalTextureDescriptor::new(source),
    );
    let binding = BindingResource::ExternalTexture(texture.clone());
    assert!(matches!(binding, BindingResource::ExternalTexture(_)));
    assert!(matches!(
        BindingKind::ExternalTexture,
        BindingKind::ExternalTexture
    ));
    assert_eq!(
        texture.device_identity(),
        DeviceIdentity::new(DeviceInstanceId::new(8))
    );
}

#[test]
fn pipeline_cache_persistence_pairs_blob_with_validation_key() {
    let key = PipelineCacheValidationKey::from_bytes([7; 32]);
    let descriptor = PipelineCacheDescriptor::new()
        .with_label("shader-cache")
        .with_initial_data(key, Arc::<[u8]>::from([1_u8, 2, 3]))
        .with_fallback(PipelineCacheFallback::IgnoreInvalidData);
    assert_eq!(descriptor.validation_key, Some(key));
    assert_eq!(descriptor.initial_data.as_deref(), Some(&[1, 2, 3][..]));
    assert_eq!(
        descriptor.fallback,
        PipelineCacheFallback::IgnoreInvalidData
    );
    assert_eq!(key.as_bytes(), [7; 32]);
}

#[test]
fn pipeline_cache_empty_boundary_has_no_implicit_serialized_blob() {
    let descriptor = PipelineCacheDescriptor::new();
    assert!(descriptor.initial_data.is_none());
    assert!(descriptor.validation_key.is_none());
    assert_eq!(
        descriptor.fallback,
        PipelineCacheFallback::RejectInvalidData
    );
}

#[test]
fn pipeline_cache_device_facade_creates_and_serializes_when_both_features_exist() {
    let device = device_with_features_for_test(
        DeviceIdentity::new(DeviceInstanceId::new(21)),
        &[
            OptionalFeature::PipelineCache,
            OptionalFeature::PipelineCacheSerialization,
        ],
    );
    let cache = device
        .create_pipeline_cache(&PipelineCacheDescriptor::new())
        .unwrap();
    assert_eq!(cache.validation_key().as_bytes(), [9; 32]);
    assert_eq!(cache.serialized_data().unwrap(), vec![0xca, 0xce]);
}

#[test]
fn pipeline_cache_feature_and_blob_key_refusals_happen_before_native_creation() {
    let without_cache =
        device_with_features_for_test(DeviceIdentity::new(DeviceInstanceId::new(22)), &[]);
    assert_eq!(
        without_cache
            .create_pipeline_cache(&PipelineCacheDescriptor::new())
            .unwrap_err()
            .kind(),
        RhiErrorKind::Unsupported
    );
    let with_cache = device_with_features_for_test(
        DeviceIdentity::new(DeviceInstanceId::new(23)),
        &[OptionalFeature::PipelineCache],
    );
    let missing_key = PipelineCacheDescriptor {
        initial_data: Some(Arc::from([1_u8])),
        ..PipelineCacheDescriptor::new()
    };
    assert_eq!(
        with_cache
            .create_pipeline_cache(&missing_key)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    let keyed = PipelineCacheDescriptor::new().with_initial_data(
        PipelineCacheValidationKey::from_bytes([1; 32]),
        Arc::from([1_u8]),
    );
    assert_eq!(
        with_cache.create_pipeline_cache(&keyed).unwrap_err().kind(),
        RhiErrorKind::Unsupported
    );
    let empty_cache = with_cache
        .create_pipeline_cache(&PipelineCacheDescriptor::new())
        .unwrap();
    assert_eq!(
        empty_cache.serialized_data().unwrap_err().kind(),
        RhiErrorKind::Unsupported
    );
}

#[test]
fn pipeline_cache_creation_observes_device_loss_before_capability() {
    let (device, backend) = paired_device_for_test(DeviceIdentity::new(DeviceInstanceId::new(24)));
    backend.mark_lost(crate::api::platform::DeviceLossInfo::new("lost".to_owned()));
    assert_eq!(
        device
            .create_pipeline_cache(&PipelineCacheDescriptor::new())
            .unwrap_err()
            .kind(),
        RhiErrorKind::DeviceLost
    );
}

#[test]
fn external_memory_import_uses_device_facade_and_fixed_source_contract() {
    let identity = DeviceIdentity::new(DeviceInstanceId::new(25));
    let device = external_memory_device_for_test(identity);
    let source = ExternalMemoryTextureSource::new(
        ObjectId::new(705),
        identity,
        ExternalMemoryHandleType::DmaBuf,
        TextureDescriptor::new_2d(64, 64, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED),
        Box::new(TestExternalMemory),
    );
    let texture = device
        .import_external_memory_texture(&ExternalTextureImportDescriptor::new(source))
        .unwrap();
    assert_eq!(texture.device_identity(), identity);
    assert_eq!(texture.descriptor().extent, Extent3d::d2(64, 64));
}

#[test]
fn external_memory_import_rejects_wrong_device_and_unsupported_handle_class() {
    let device = external_memory_device_for_test(DeviceIdentity::new(DeviceInstanceId::new(26)));
    let wrong = ExternalMemoryTextureSource::new(
        ObjectId::new(706),
        DeviceIdentity::new(DeviceInstanceId::new(27)),
        ExternalMemoryHandleType::DmaBuf,
        TextureDescriptor::new_2d(1, 1, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED),
        Box::new(TestExternalMemory),
    );
    assert_eq!(
        device
            .import_external_memory_texture(&ExternalTextureImportDescriptor::new(wrong))
            .unwrap_err()
            .kind(),
        RhiErrorKind::WrongDevice
    );
    let unsupported = ExternalMemoryTextureSource::new(
        ObjectId::new(707),
        DeviceIdentity::new(DeviceInstanceId::new(26)),
        ExternalMemoryHandleType::OpaqueFd,
        TextureDescriptor::new_2d(1, 1, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED),
        Box::new(TestExternalMemory),
    );
    assert_eq!(
        device
            .import_external_memory_texture(&ExternalTextureImportDescriptor::new(unsupported))
            .unwrap_err()
            .kind(),
        RhiErrorKind::Unsupported
    );
}

#[test]
fn external_memory_import_observes_device_loss_before_capability() {
    let (device, backend) = paired_device_for_test(DeviceIdentity::new(DeviceInstanceId::new(28)));
    backend.mark_lost(crate::api::platform::DeviceLossInfo::new("lost".to_owned()));
    let source = ExternalMemoryTextureSource::new(
        ObjectId::new(708),
        DeviceIdentity::new(DeviceInstanceId::new(28)),
        ExternalMemoryHandleType::DmaBuf,
        TextureDescriptor::new_2d(1, 1, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED),
        Box::new(TestExternalMemory),
    );
    assert_eq!(
        device
            .import_external_memory_texture(&ExternalTextureImportDescriptor::new(source))
            .unwrap_err()
            .kind(),
        RhiErrorKind::DeviceLost
    );
}
