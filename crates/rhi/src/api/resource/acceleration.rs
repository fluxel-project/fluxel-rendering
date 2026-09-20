//! Portable acceleration-structure objects and build input validation.
//!
//! A structure is an opaque, device-owned object.  It deliberately has no GPU
//! address accessor: addresses are a native lowering detail, while the portable
//! handle is what shaders bind and commands retain.

use core::fmt;
use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::backend::AccelerationStructureBackend;
use crate::api::resource::{Buffer, BufferRange, BufferUsage};

/// The hierarchy represented by an acceleration structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelerationStructureKind {
    /// Bottom-level geometry hierarchy.
    BottomLevel,
    /// Top-level instance hierarchy.
    TopLevel,
}

/// Portable position encoding consumed by a triangle BLAS build.
///
/// `Float32x3` is the mandatory baseline. Every other encoding is explicitly
/// capability-gated because native ray-tracing APIs disagree on which packed,
/// normalized, and half-float position formats they accept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelerationStructureVertexFormat {
    /// Three 32-bit floating-point position components.
    Float32x3,
    /// Three 16-bit floating-point position components.
    Float16x3,
    /// Four 16-bit floating-point position components.
    Float16x4,
    /// Three unsigned normalized 16-bit components.
    Unorm16x3,
    /// Four unsigned normalized 8-bit components.
    Unorm8x4,
}

impl AccelerationStructureVertexFormat {
    const fn byte_size(self) -> u32 {
        match self {
            Self::Float32x3 => 12,
            Self::Float16x3 => 6,
            Self::Float16x4 => 8,
            Self::Unorm16x3 => 6,
            Self::Unorm8x4 => 4,
        }
    }
    const fn requires_extended_feature(self) -> bool {
        !matches!(self, Self::Float32x3)
    }
}

/// Element encoding for indexed triangle geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelerationStructureIndexFormat {
    /// Unsigned 16-bit indices.
    Uint16,
    /// Unsigned 32-bit indices.
    Uint32,
}
impl AccelerationStructureIndexFormat {
    const fn byte_size(self) -> u32 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// Geometry represented by indexed or non-indexed triangles.
#[derive(Clone, Debug)]
pub struct TrianglesGeometry {
    /// Vertex bytes used by the build.
    pub vertices: Buffer,
    /// Byte range of the vertex input.
    pub vertex_range: BufferRange,
    /// Encoding of each position record.
    pub vertex_format: AccelerationStructureVertexFormat,
    /// Vertex stride, in bytes. It must be non-zero and four-byte aligned.
    pub vertex_stride: u32,
    /// Number of addressable vertex records in `vertex_range`.
    pub vertex_count: u32,
    /// Number of triangle primitives this geometry contributes.
    pub primitive_count: u32,
    /// Optional index bytes and their element encoding.
    pub indices: Option<(Buffer, BufferRange, AccelerationStructureIndexFormat)>,
}

/// Axis-aligned-box geometry.
#[derive(Clone, Debug)]
pub struct AabbGeometry {
    /// Bytes holding the AABB records.
    pub boxes: Buffer,
    /// Byte range of the AABB records.
    pub range: BufferRange,
    /// Record stride, in bytes. It must be at least 24 and four-byte aligned.
    pub stride: u32,
    /// Number of AABB primitives.
    pub primitive_count: u32,
}

/// One BLAS geometry input.
#[derive(Clone, Debug)]
pub enum BlasGeometry {
    /// Triangle geometry.
    Triangles(TrianglesGeometry),
    /// Axis-aligned boxes.
    Aabbs(AabbGeometry),
}

/// Creation contract for a bottom-level acceleration structure.
#[derive(Clone, Debug)]
pub struct BottomLevelAccelerationStructureDescriptor {
    /// Diagnostic label.
    pub label: Label,
    /// Geometry inputs retained by the structure.
    pub geometries: Vec<BlasGeometry>,
    /// Optional behaviours the allocation must support.
    pub build_options: AccelerationStructureBuildOptions,
}

impl BottomLevelAccelerationStructureDescriptor {
    /// Creates an unlabelled BLAS descriptor.
    pub fn new(geometries: Vec<BlasGeometry>) -> Self {
        Self {
            label: Label::default(),
            geometries,
            build_options: AccelerationStructureBuildOptions::NONE,
        }
    }
    /// Adds a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
    /// Declares optional update/compaction storage requirements before sizing.
    pub fn with_build_options(mut self, options: AccelerationStructureBuildOptions) -> Self {
        self.build_options = options;
        self
    }
}

/// A portable top-level instance record.
#[derive(Clone, Debug)]
pub struct TlasInstance {
    /// Bottom-level structure instantiated by this record.
    pub bottom_level: AccelerationStructure,
    /// Row-major affine 3x4 transform.
    pub transform: [[f32; 4]; 3],
    /// Caller-defined instance mask.
    pub mask: u8,
    /// Caller-defined shader record offset.
    pub shader_record_offset: u32,
}

/// Creation contract for a top-level acceleration structure.
#[derive(Clone, Debug)]
pub struct TopLevelAccelerationStructureDescriptor {
    /// Diagnostic label.
    pub label: Label,
    /// Instances used to build this hierarchy.
    pub instances: Vec<TlasInstance>,
    /// Optional behaviours the allocation must support.
    pub build_options: AccelerationStructureBuildOptions,
}

impl TopLevelAccelerationStructureDescriptor {
    /// Creates an unlabelled TLAS descriptor.
    pub fn new(instances: Vec<TlasInstance>) -> Self {
        Self {
            label: Label::default(),
            instances,
            build_options: AccelerationStructureBuildOptions::NONE,
        }
    }
    /// Adds a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
    /// Declares optional update/compaction storage requirements before sizing.
    pub fn with_build_options(mut self, options: AccelerationStructureBuildOptions) -> Self {
        self.build_options = options;
        self
    }
}

/// Descriptor for either hierarchy kind.
#[derive(Clone, Debug)]
pub enum AccelerationStructureDescriptor {
    /// Bottom-level geometry hierarchy.
    BottomLevel(BottomLevelAccelerationStructureDescriptor),
    /// Top-level instance hierarchy.
    TopLevel(TopLevelAccelerationStructureDescriptor),
}

impl AccelerationStructureDescriptor {
    /// Hierarchy kind selected by this descriptor.
    pub fn kind(&self) -> AccelerationStructureKind {
        match self {
            Self::BottomLevel(_) => AccelerationStructureKind::BottomLevel,
            Self::TopLevel(_) => AccelerationStructureKind::TopLevel,
        }
    }
    /// Options that were part of the native allocation-size query.
    pub fn build_options(&self) -> AccelerationStructureBuildOptions {
        match self {
            Self::BottomLevel(desc) => desc.build_options,
            Self::TopLevel(desc) => desc.build_options,
        }
    }
}

/// Backend-calculated allocation requirements for one build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccelerationStructureBuildSizes {
    /// Final structure allocation size.
    pub acceleration_structure_size: u64,
    /// Scratch bytes required by a build.
    pub build_scratch_size: u64,
    /// Scratch bytes required by an update, when supported.
    pub update_scratch_size: u64,
}

/// Options which change the native build allocation contract.
///
/// They are declared when the structure is created, rather than inferred from a
/// later command.  Native APIs commonly require this choice while calculating
/// the exact result and scratch sizes; accepting an update or compact request
/// after allocation would make the portable handle promise storage it may not
/// have.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AccelerationStructureBuildOptions(u8);
impl AccelerationStructureBuildOptions {
    /// No optional build behaviour is requested.
    pub const NONE: Self = Self(0);
    /// The allocation may be updated in place with compatible build input.
    pub const ALLOW_UPDATE: Self = Self(1);
    /// The completed structure may have its compact size queried and be copied
    /// into a separately created compact destination.
    pub const ALLOW_COMPACTION: Self = Self(2);
    /// Whether all bits in `other` are selected.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// Whether a build creates a hierarchy or updates an existing compatible one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelerationStructureBuildMode {
    /// Build a new hierarchy from the descriptor's input.
    Build,
    /// Update a compatible existing hierarchy.
    Update,
}

/// A copy operation between compatible acceleration structures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelerationStructureCopyMode {
    /// Copy without changing the allocation representation.
    Clone,
    /// Copy into a compacted allocation after a backend-reported compact size.
    Compact,
}

/// Opaque acceleration-structure handle.
#[derive(Clone)]
pub struct AccelerationStructure {
    inner: Arc<AccelerationStructureInner>,
}
struct AccelerationStructureInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: AccelerationStructureDescriptor,
    build_sizes: AccelerationStructureBuildSizes,
    native: Box<dyn AccelerationStructureBackend>,
}
impl AccelerationStructure {
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: AccelerationStructureDescriptor,
        build_sizes: AccelerationStructureBuildSizes,
        native: Box<dyn AccelerationStructureBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(AccelerationStructureInner {
                id,
                device,
                descriptor,
                build_sizes,
                native,
            }),
        }
    }
    /// Process-local identity.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }
    /// Device that owns this structure.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }
    /// Immutable creation descriptor.
    pub fn descriptor(&self) -> &AccelerationStructureDescriptor {
        &self.inner.descriptor
    }
    /// Exact native allocation and scratch requirements calculated for this
    /// descriptor and its declared build options.
    pub fn build_sizes(&self) -> AccelerationStructureBuildSizes {
        self.inner.build_sizes
    }
    pub(crate) fn native(&self) -> &dyn AccelerationStructureBackend {
        self.inner.native.as_ref()
    }
}
impl fmt::Debug for AccelerationStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccelerationStructure")
            .field("id", &self.id())
            .field("device", &self.device_identity())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Creates an acceleration structure after portable capability and input validation.
    pub fn create_acceleration_structure(
        &self,
        descriptor: &AccelerationStructureDescriptor,
    ) -> RhiResult<AccelerationStructure> {
        self.require_active()
            .map_err(|e| e.at("Device::create_acceleration_structure"))?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::RayQuery)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not enable acceleration structures",
            )
            .at("Device::create_acceleration_structure"));
        }
        validate_descriptor(
            descriptor,
            self.identity(),
            |key| self.capabilities().limit(key),
            |feature| self.capabilities().supports_feature(feature),
        )?;
        let options = descriptor.build_options();
        if options.contains(AccelerationStructureBuildOptions::ALLOW_UPDATE)
            && !self
                .capabilities()
                .supports_feature(OptionalFeature::AccelerationStructureUpdate)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not enable acceleration-structure updates",
            )
            .at("Device::create_acceleration_structure"));
        }
        if options.contains(AccelerationStructureBuildOptions::ALLOW_COMPACTION)
            && !self
                .capabilities()
                .supports_feature(OptionalFeature::AccelerationStructureCompaction)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not enable acceleration-structure compaction",
            )
            .at("Device::create_acceleration_structure"));
        }
        let sizes = self
            .native()
            .acceleration_structure_build_sizes(descriptor)?;
        validate_build_sizes(sizes, descriptor)?;
        let native = self
            .native()
            .create_acceleration_structure(descriptor, sizes)?;
        Ok(AccelerationStructure::new(
            ObjectId::next(),
            self.identity(),
            descriptor.clone(),
            sizes,
            native,
        ))
    }
}

fn validate_build_sizes(
    sizes: AccelerationStructureBuildSizes,
    descriptor: &AccelerationStructureDescriptor,
) -> RhiResult<()> {
    if sizes.acceleration_structure_size == 0 || sizes.build_scratch_size == 0 {
        return Err(RhiError::new(
            RhiErrorKind::BackendFailure,
            "backend returned a zero acceleration-structure build size",
        ));
    }
    if descriptor
        .build_options()
        .contains(AccelerationStructureBuildOptions::ALLOW_UPDATE)
        && sizes.update_scratch_size == 0
    {
        return Err(RhiError::new(
            RhiErrorKind::BackendFailure,
            "backend returned no update scratch size for an update-capable structure",
        ));
    }
    Ok(())
}

fn require_blas_input(
    buffer: &Buffer,
    range: BufferRange,
    device: DeviceIdentity,
    what: &'static str,
) -> RhiResult<()> {
    if buffer.device_identity() != device {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            format!("{what} belongs to a different device"),
        )
        .with_object(buffer.id()));
    }
    if !buffer.descriptor().usage.contains(BufferUsage::BLAS_INPUT) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("{what} requires BLAS_INPUT usage"),
        ));
    }
    let Some(end) = range.end() else {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("{what} range overflows"),
        ));
    };
    if range.size == 0 || end > buffer.descriptor().size {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("{what} range is outside its buffer"),
        ));
    }
    Ok(())
}

fn require_count_coverage(
    range: BufferRange,
    count: u32,
    stride: u32,
    what: &'static str,
) -> RhiResult<()> {
    let required = u64::from(count)
        .checked_mul(u64::from(stride))
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("{what} byte count overflows"),
            )
        })?;
    if range.size < required {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("{what} range does not cover count × stride"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_descriptor(
    desc: &AccelerationStructureDescriptor,
    device: DeviceIdentity,
    limit: impl Fn(LimitKey) -> Option<u64>,
    supports_feature: impl Fn(OptionalFeature) -> bool,
) -> RhiResult<()> {
    match desc {
        AccelerationStructureDescriptor::BottomLevel(blas) => {
            if blas.geometries.is_empty() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a BLAS needs at least one geometry",
                ));
            }
            if let Some(max) = limit(LimitKey::MaxBlasGeometryCount) {
                if blas.geometries.len() as u64 > max {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "BLAS geometry count exceeds device limit",
                    ));
                }
            }
            let mut primitive_total = 0u64;
            for geometry in &blas.geometries {
                match geometry {
                    BlasGeometry::Triangles(g) => {
                        require_blas_input(
                            &g.vertices,
                            g.vertex_range,
                            device,
                            "BLAS vertex input",
                        )?;
                        if g.vertex_stride == 0
                            || g.vertex_stride % 4 != 0
                            || g.primitive_count == 0
                            || g.vertex_count == 0
                            || g.vertex_stride < g.vertex_format.byte_size()
                        {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "triangle geometry needs non-zero primitive count and four-byte-aligned non-zero stride",
                            ));
                        }
                        if g.vertex_format.requires_extended_feature()
                            && !supports_feature(
                                OptionalFeature::ExtendedAccelerationStructureVertexFormats,
                            )
                        {
                            return Err(RhiError::new(
                                RhiErrorKind::Unsupported,
                                "this device does not enable the requested acceleration-structure vertex format",
                            ));
                        }
                        require_count_coverage(
                            g.vertex_range,
                            g.vertex_count,
                            g.vertex_stride,
                            "BLAS vertex input",
                        )?;
                        let required_indices =
                            g.primitive_count.checked_mul(3).ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "triangle index count overflows",
                                )
                            })?;
                        if g.indices.is_none() && g.vertex_count < required_indices {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "non-indexed triangles need at least three vertices per primitive",
                            ));
                        }
                        if let Some((indices, range, format)) = &g.indices {
                            require_blas_input(indices, *range, device, "BLAS index input")?;
                            require_count_coverage(
                                *range,
                                required_indices,
                                format.byte_size(),
                                "BLAS index input",
                            )?;
                        }
                        primitive_total = primitive_total
                            .checked_add(u64::from(g.primitive_count))
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "BLAS primitive count overflows",
                                )
                            })?;
                    }
                    BlasGeometry::Aabbs(g) => {
                        require_blas_input(&g.boxes, g.range, device, "BLAS AABB input")?;
                        if g.stride < 24 || g.stride % 4 != 0 || g.primitive_count == 0 {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "AABB geometry needs non-zero primitive count and a four-byte-aligned stride of at least 24",
                            ));
                        }
                        require_count_coverage(
                            g.range,
                            g.primitive_count,
                            g.stride,
                            "BLAS AABB input",
                        )?;
                        primitive_total = primitive_total
                            .checked_add(u64::from(g.primitive_count))
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "BLAS primitive count overflows",
                                )
                            })?;
                    }
                }
            }
            if let Some(max) = limit(LimitKey::MaxBlasPrimitiveCount) {
                if primitive_total > max {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "BLAS primitive total exceeds device limit",
                    ));
                }
            }
        }
        AccelerationStructureDescriptor::TopLevel(tlas) => {
            if tlas.instances.is_empty() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a TLAS needs at least one instance",
                ));
            }
            if let Some(max) = limit(LimitKey::MaxTlasInstanceCount) {
                if tlas.instances.len() as u64 > max {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "TLAS instance count exceeds device limit",
                    ));
                }
            }
            for instance in &tlas.instances {
                if instance
                    .transform
                    .iter()
                    .flatten()
                    .any(|component| !component.is_finite())
                {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "TLAS transform components must be finite",
                    ));
                }
                // DXR encodes this field in 24 bits, and keeping the portable
                // subset avoids a backend-specific truncation on otherwise
                // valid Vulkan-sized u32 values.
                if instance.shader_record_offset > 0x00ff_ffff {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "TLAS shader record offset exceeds the portable 24-bit range",
                    ));
                }
                if instance.bottom_level.device_identity() != device {
                    return Err(RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "TLAS instance BLAS belongs to a different device",
                    )
                    .with_object(instance.bottom_level.id()));
                }
                if instance.bottom_level.descriptor().kind()
                    != AccelerationStructureKind::BottomLevel
                {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "TLAS instances must reference BLAS objects",
                    ));
                }
            }
        }
    }
    Ok(())
}
