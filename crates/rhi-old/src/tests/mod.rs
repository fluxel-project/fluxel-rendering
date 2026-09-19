//! Tests device opening and owned-resource creation across supported native backends.

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
use super::{Backend, Device, DeviceOptions, Validation};

#[cfg(not(windows))]
#[test]
fn native_backends_are_rejected_off_windows() {
    use super::{Backend, Device, DeviceOptions, OpenError};

    assert!(matches!(
        Device::open(Backend::Vulkan, DeviceOptions::default()),
        Err(OpenError::PlatformUnsupported {
            backend: Backend::Vulkan
        })
    ));
}

#[cfg(all(windows, not(any(feature = "dx12", feature = "vulkan"))))]
#[test]
fn disabled_native_backends_are_rejected_on_windows() {
    use super::{Backend, Device, DeviceOptions, OpenError};

    for backend in [Backend::Dx12, Backend::Vulkan] {
        assert!(matches!(
            Device::open(backend, DeviceOptions::default()),
            Err(OpenError::BackendDisabled { backend: rejected }) if rejected == backend
        ));
    }
}

#[cfg(all(windows, feature = "dx12", feature = "vulkan"))]
#[test]
#[ignore = "requires a locally installed native GPU driver"]
fn opens_both_real_headless_backends() {
    for backend in [Backend::Dx12, Backend::Vulkan] {
        let device = Device::open(backend, DeviceOptions::default())
            .unwrap_or_else(|error| panic!("failed to open {backend:?}: {error}"));
        assert_eq!(device.hardware().backend, backend);
        assert_ne!(device.hardware().name, "");
        assert_ne!(device.capabilities().max_texture_dimension_2d, 0);
        eprintln!("{device:?}");
    }
}

#[cfg(all(windows, feature = "dx12", feature = "vulkan"))]
#[test]
#[ignore = "requires native validation facilities and a real GPU driver"]
fn creates_owned_resources_on_both_real_backends() {
    use crate::{
        BufferDescriptor, InvalidResourceReason, MemoryPolicy, ResourceCreateError, ResourceKind,
        TextureDescriptor,
    };
    use fluxel_rendergraph::{
        BufferDesc, BufferUsage, BufferUsageKind, Extent3d, TextureDesc, TextureDimension,
        TextureFormat, TextureUsage, TextureUsageKind,
    };

    for backend in [Backend::Dx12, Backend::Vulkan] {
        let before = crate::imp::destruction_counts();
        let device = Device::open(
            backend,
            DeviceOptions {
                validation: Validation::Required,
                ..DeviceOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("failed to open {backend:?}: {error}"));
        let buffer_usage = BufferUsage::from_kinds([
            BufferUsageKind::CopySource,
            BufferUsageKind::CopyDestination,
            BufferUsageKind::StorageRead,
            BufferUsageKind::StorageWrite,
        ]);
        let buffer = device
            .create_buffer(BufferDescriptor {
                buffer: BufferDesc { size: 4096 },
                usage: buffer_usage,
                memory: MemoryPolicy::DeviceOnly,
            })
            .unwrap_or_else(|error| panic!("{backend:?} A01/A03 failed: {error}"));
        assert_eq!(buffer.allowed_usage(), buffer_usage);
        assert_eq!(buffer.device_identity(), device.identity());
        let lease = buffer.lease();
        let lease_clone = lease.clone();
        drop(buffer);
        assert_eq!(crate::imp::destruction_counts(), before);
        drop(lease);
        assert_eq!(crate::imp::destruction_counts(), before);
        drop(device);
        assert_eq!(crate::imp::destruction_counts(), before);
        drop(lease_clone);
        assert_eq!(crate::imp::destruction_counts().0, before.0 + 1);

        let device = Device::open(
            backend,
            DeviceOptions {
                validation: Validation::Required,
                ..DeviceOptions::default()
            },
        )
        .unwrap();
        let texture_usage = TextureUsage::from_kinds([
            TextureUsageKind::CopySource,
            TextureUsageKind::CopyDestination,
            TextureUsageKind::ColorAttachment,
        ]);
        let texture_desc = TextureDescriptor {
            texture: TextureDesc {
                dimension: TextureDimension::D2,
                extent: Extent3d {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
            },
            usage: texture_usage,
            memory: MemoryPolicy::DeviceOnly,
        };
        let texture = device
            .create_texture(texture_desc)
            .unwrap_or_else(|error| panic!("{backend:?} A02 failed: {error}"));
        assert_eq!(texture.descriptor(), texture_desc);
        assert_eq!(texture.allowed_usage(), texture_usage);

        let invalid = TextureDescriptor {
            usage: TextureUsage::empty().with(TextureUsageKind::Present),
            ..texture_desc
        };
        assert_eq!(
            device.create_texture(invalid).unwrap_err(),
            ResourceCreateError::InvalidDescriptor {
                resource: ResourceKind::Texture,
                reason: InvalidResourceReason::PresentRequiresSurface,
            }
        );
        assert_eq!(crate::imp::destruction_counts().1, before.1);
        let incompatible = TextureDescriptor {
            texture: TextureDesc {
                format: TextureFormat::Depth32Float,
                ..texture_desc.texture
            },
            usage: TextureUsage::empty().with(TextureUsageKind::ColorAttachment),
            ..texture_desc
        };
        assert_eq!(
            device.create_texture(incompatible).unwrap_err(),
            ResourceCreateError::InvalidDescriptor {
                resource: ResourceKind::Texture,
                reason: InvalidResourceReason::IncompatibleUsage,
            }
        );
        let subsequent = device
            .create_texture(texture_desc)
            .unwrap_or_else(|error| panic!("{backend:?} A04 recovery failed: {error}"));
        assert_ne!(texture.identity(), subsequent.identity());
        drop(texture);
        drop(subsequent);
        assert_eq!(crate::imp::destruction_counts().1, before.1 + 2);
        eprintln!(
            "A01-A06 {backend:?}: hardware={:?}; buffer={:?}; texture={:?}; structured rejection recovered; destroyed buffer=1 texture=2; validation diagnostics=none observed",
            device.hardware(),
            buffer_usage,
            texture_desc
        );
    }
}

#[cfg(all(windows, feature = "dx12"))]
#[test]
#[ignore = "requires native validation facilities"]
fn opens_dx12_with_required_validation() {
    open_with_required_validation(Backend::Dx12);
}

#[cfg(all(windows, feature = "vulkan"))]
#[test]
#[ignore = "requires native validation facilities"]
fn opens_vulkan_with_required_validation() {
    open_with_required_validation(Backend::Vulkan);
}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
fn open_with_required_validation(backend: Backend) {
    let device = Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("failed to open {backend:?}: {error}"));
    assert_eq!(device.hardware().backend, backend);
    eprintln!("{device:?}");
}
