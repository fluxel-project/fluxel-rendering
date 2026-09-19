//! Contract tests for storage bindings: usage, ranges and images are all
//! checked before the binding is recorded.

use super::*;

#[test]
fn mock_storage_bindings_validate_usage_ranges_and_images_before_recording() {
    let mut inner = MockGlFamilyApi::from_discovery(compute_storage_snapshot(true));
    let non_storage = inner
        .create_buffer_resource(GlBufferDesc {
            size: 512,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("buffer");
    let image = inner
        .create_texture_resource(GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 1,
                height: 1,
                depth_or_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::STORAGE_BINDING,
        })
        .expect("storage image");
    let mut api = MockComputeStorageApi::new(inner).expect("proved optional domains");
    let trace_len = api.calls().len();
    assert!(
        api.bind_storage_buffer(
            0,
            GlStorageBufferRange {
                buffer: non_storage,
                offset: 0,
                size: 256,
                usage: GlStorageBufferUsage::ReadOnly,
            },
        )
        .is_err()
    );
    assert!(
        api.bind_storage_image(
            0,
            GlStorageImageBinding {
                texture: image,
                level: 1,
                sample_count: 1,
                layered: false,
                layer: Some(0),
                format: GlFormat::Rgba8Unorm,
                access: GlStorageImageAccess::ReadWrite,
            },
        )
        .is_err()
    );
    assert!(!api.calls()[trace_len..].iter().any(|call| {
        matches!(
            call,
            MockCall::BindStorageBuffer { .. } | MockCall::BindStorageImage { .. }
        )
    }));
}
