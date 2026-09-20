//! Advanced command families: mesh/task dispatch, acceleration-structure work,
//! and ray-tracing dispatch.  They retain portable packets; native lowering is
//! selected only by a backend that published the matching capability.

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::record::{
    AccelerationStructureCommand, BoundGroup, ImmediateWrite, RayTracingBegin, RayTracingDispatch,
    RecordedPayload,
};
use crate::api::command::uses::{
    bound_group_uses, require_valid_dynamic_offsets, validate_bound_groups,
};
use crate::api::command::{AccessMask, CommandRecorder, PipelineScope, RecorderPhase, ResourceUse};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::Label;
use crate::api::pipeline::{MeshPipeline, PipelineInterface, RayTracingPipeline};
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::{
    AccelerationStructure, AccelerationStructureBuildMode, AccelerationStructureCopyMode, Buffer,
    BufferRange, BufferUsage,
};
use crate::api::submission::LaneWorkDomains;

const COMPUTE: LaneWorkDomains = LaneWorkDomains::COMPUTE;

/// Validates one portable immediate-data write shared by every programmable scope.
pub(crate) fn immediate_write(
    recorder: &CommandRecorder,
    interface: &PipelineInterface,
    offset: u32,
    bytes: &[u8],
    operation: &'static str,
) -> RhiResult<ImmediateWrite> {
    if !recorder
        .capabilities()
        .supports_feature(OptionalFeature::Immediates)
    {
        return unsupported(
            operation,
            "this device does not enable immediate pipeline data",
        );
    }
    let error = |message| RhiError::new(RhiErrorKind::InvalidUsage, message).at(operation);
    let size =
        u32::try_from(bytes.len()).map_err(|_| error("immediate-data write size overflows u32"))?;
    if size == 0 {
        return Err(error("immediate-data writes must be non-empty"));
    }
    let end = offset
        .checked_add(size)
        .ok_or_else(|| error("immediate-data write overflows"))?;
    let alignment = recorder
        .capabilities()
        .limit(LimitKey::ImmediateDataAlignment)
        .unwrap_or(1);
    if alignment == 0 || u64::from(offset) % alignment != 0 || u64::from(size) % alignment != 0 {
        return Err(error(
            "immediate-data offset and size must satisfy ImmediateDataAlignment",
        ));
    }
    let range = interface
        .descriptor()
        .immediate_ranges
        .iter()
        .find(|range| {
            range.offset <= offset
                && range
                    .offset
                    .checked_add(range.size)
                    .is_some_and(|limit| end <= limit)
        })
        .copied()
        .ok_or_else(|| {
            error("immediate-data write is not wholly contained in a declared pipeline range")
        })?;
    Ok(ImmediateWrite {
        offset,
        bytes: bytes.to_vec(),
        visibility: range.visibility,
    })
}

/// Descriptor for a ray-tracing scope.
#[derive(Clone, Default)]
pub struct RayTracingScopeDescriptor {
    /// Diagnostic label.
    pub label: Label,
}
/// One contiguous shader-table region. Native addresses are deliberately absent.
#[derive(Clone)]
pub struct RayTracingShaderTableRegion {
    /// Backing buffer, created with `INDIRECT` usage.
    pub buffer: Buffer,
    /// Byte range containing whole records.
    pub range: BufferRange,
    /// Byte stride of each record.
    pub stride: u64,
}
/// Shader-table regions consumed by one ray dispatch.
#[derive(Clone)]
pub struct RayTracingShaderTable {
    /// Exactly one ray-generation record.
    pub ray_generation: RayTracingShaderTableRegion,
    /// Optional miss records.
    pub miss: Option<RayTracingShaderTableRegion>,
    /// Optional hit records.
    pub hit: Option<RayTracingShaderTableRegion>,
}
impl RayTracingScopeDescriptor {
    /// Creates an unlabelled ray-tracing scope descriptor.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets its diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}

impl CommandRecorder {
    /// Records a build or compatible update of `destination` using `scratch`.
    pub fn build_acceleration_structure(
        &mut self,
        destination: &AccelerationStructure,
        scratch: &Buffer,
        mode: AccelerationStructureBuildMode,
    ) -> RhiResult<()> {
        self.require_open("build_acceleration_structure")?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::RayQuery)
        {
            return unsupported(
                "CommandRecorder::build_acceleration_structure",
                "this device does not enable acceleration structures",
            );
        }
        require_same(
            destination.device_identity(),
            self.device_identity(),
            "acceleration structure",
        )?;
        require_same(
            scratch.device_identity(),
            self.device_identity(),
            "build scratch buffer",
        )?;
        if !scratch
            .descriptor()
            .usage
            .contains(BufferUsage::ACCELERATION_STRUCTURE_SCRATCH)
        {
            return invalid(
                "CommandRecorder::build_acceleration_structure",
                "build scratch requires ACCELERATION_STRUCTURE_SCRATCH usage",
            );
        }
        if matches!(mode, AccelerationStructureBuildMode::Update)
            && !self
                .capabilities()
                .supports_feature(OptionalFeature::AccelerationStructureUpdate)
        {
            return unsupported(
                "CommandRecorder::build_acceleration_structure",
                "this device does not enable acceleration-structure updates",
            );
        }
        let required_scratch = match mode {
            AccelerationStructureBuildMode::Build => destination.build_sizes().build_scratch_size,
            AccelerationStructureBuildMode::Update => destination.build_sizes().update_scratch_size,
        };
        // The native AS constraint applies to the scratch GPU address (or to a
        // suballocation offset), not to the allocation's total byte length.
        // This API consumes the buffer from offset zero, so a buffer carrying
        // ACCELERATION_STRUCTURE_SCRATCH usage must receive a suitably aligned
        // native allocation from its backend. Requiring `size % alignment == 0`
        // here would reject valid allocations whose address is aligned and
        // whose capacity merely exceeds the requested scratch size.
        let alignment = self
            .capabilities()
            .limit(LimitKey::RayTracingScratchBufferAlignment)
            .unwrap_or(0);
        if required_scratch == 0 || scratch.descriptor().size < required_scratch || alignment == 0 {
            return invalid(
                "CommandRecorder::build_acceleration_structure",
                "build scratch must cover the mode size and the device must publish a scratch-address alignment",
            );
        }
        let mut uses = build_input_uses(destination)?;
        uses.push(ResourceUse::Buffer(crate::api::command::BufferUse {
            buffer: scratch.clone(),
            range: BufferRange::new(0, scratch.descriptor().size),
            stages: PipelineScope::COMPUTE,
            access: AccessMask::ACCELERATION_STRUCTURE_BUILD_WRITE,
        }));
        uses.push(ResourceUse::AccelerationStructure(
            crate::api::command::AccelerationStructureUse {
                structure: destination.clone(),
                stages: PipelineScope::COMPUTE,
                access: AccessMask::ACCELERATION_STRUCTURE_BUILD_WRITE,
            },
        ));
        self.record_command(
            RecordedPayload::AccelerationStructure(AccelerationStructureCommand::Build {
                destination: destination.clone(),
                scratch: scratch.clone(),
                mode,
            }),
            uses,
            COMPUTE,
        );
        Ok(())
    }

    /// Records a same-kind clone or compaction copy between acceleration structures.
    pub fn copy_acceleration_structure(
        &mut self,
        source: &AccelerationStructure,
        destination: &AccelerationStructure,
        mode: AccelerationStructureCopyMode,
    ) -> RhiResult<()> {
        self.require_open("copy_acceleration_structure")?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::RayQuery)
        {
            return unsupported(
                "CommandRecorder::copy_acceleration_structure",
                "this device does not enable acceleration structures",
            );
        }
        require_same(
            source.device_identity(),
            self.device_identity(),
            "source acceleration structure",
        )?;
        require_same(
            destination.device_identity(),
            self.device_identity(),
            "destination acceleration structure",
        )?;
        if source.id() == destination.id()
            || source.descriptor().kind() != destination.descriptor().kind()
        {
            return invalid(
                "CommandRecorder::copy_acceleration_structure",
                "source and destination must be distinct acceleration structures of the same kind",
            );
        }
        if matches!(mode, AccelerationStructureCopyMode::Compact)
            && !self
                .capabilities()
                .supports_feature(OptionalFeature::AccelerationStructureCompaction)
        {
            return unsupported(
                "CommandRecorder::copy_acceleration_structure",
                "this device does not enable acceleration-structure compaction",
            );
        }
        let uses = vec![
            ResourceUse::AccelerationStructure(crate::api::command::AccelerationStructureUse {
                structure: source.clone(),
                stages: PipelineScope::COPY,
                access: AccessMask::ACCELERATION_STRUCTURE_BUILD_READ,
            }),
            ResourceUse::AccelerationStructure(crate::api::command::AccelerationStructureUse {
                structure: destination.clone(),
                stages: PipelineScope::COPY,
                access: AccessMask::ACCELERATION_STRUCTURE_BUILD_WRITE,
            }),
        ];
        self.record_command(
            RecordedPayload::AccelerationStructure(AccelerationStructureCommand::Copy {
                source: source.clone(),
                destination: destination.clone(),
                mode,
            }),
            uses,
            COMPUTE,
        );
        Ok(())
    }

    /// Records a compacted-size query for `source` into one eight-byte buffer
    /// slot. The slot is produced by GPU execution, so it may be read only after
    /// the enclosing work's completion (normally by a readback recorded after
    /// this command). It is deliberately not returned synchronously.
    pub fn write_acceleration_structure_compaction_size(
        &mut self,
        source: &AccelerationStructure,
        destination: &Buffer,
        destination_offset: u64,
    ) -> RhiResult<()> {
        self.require_open("write_acceleration_structure_compaction_size")?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::AccelerationStructureCompaction)
        {
            return unsupported(
                "CommandRecorder::write_acceleration_structure_compaction_size",
                "this device does not enable acceleration-structure compaction",
            );
        }
        require_same(
            source.device_identity(),
            self.device_identity(),
            "acceleration structure",
        )?;
        require_same(
            destination.device_identity(),
            self.device_identity(),
            "compaction-size destination",
        )?;
        if !source
            .descriptor()
            .build_options()
            .contains(crate::api::resource::AccelerationStructureBuildOptions::ALLOW_COMPACTION)
        {
            return invalid(
                "CommandRecorder::write_acceleration_structure_compaction_size",
                "the source was not created with ALLOW_COMPACTION",
            );
        }
        if !destination
            .descriptor()
            .usage
            .contains(BufferUsage::QUERY_RESOLVE)
            || destination_offset % 8 != 0
            || destination_offset
                .checked_add(8)
                .is_none_or(|end| end > destination.descriptor().size)
        {
            return invalid(
                "CommandRecorder::write_acceleration_structure_compaction_size",
                "destination needs QUERY_RESOLVE usage and an in-range eight-byte-aligned u64 slot",
            );
        }
        self.record_command(
            RecordedPayload::AccelerationStructure(
                AccelerationStructureCommand::WriteCompactedSize {
                    source: source.clone(),
                    destination: destination.clone(),
                    destination_offset,
                },
            ),
            vec![
                ResourceUse::AccelerationStructure(crate::api::command::AccelerationStructureUse {
                    structure: source.clone(),
                    stages: PipelineScope::COPY,
                    access: AccessMask::ACCELERATION_STRUCTURE_BUILD_READ,
                }),
                ResourceUse::Buffer(crate::api::command::BufferUse {
                    buffer: destination.clone(),
                    range: BufferRange::new(destination_offset, 8),
                    stages: PipelineScope::COPY,
                    access: AccessMask::COPY_WRITE,
                }),
            ],
            COMPUTE,
        );
        Ok(())
    }

    /// Opens a ray-tracing scope.
    pub fn begin_ray_tracing<'a>(
        &'a mut self,
        desc: &RayTracingScopeDescriptor,
    ) -> RhiResult<RayTracingScope<'a>> {
        self.require_open("begin_ray_tracing")?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::RayTracingPipeline)
        {
            return unsupported(
                "CommandRecorder::begin_ray_tracing",
                "this device does not enable ray-tracing pipelines",
            );
        }
        self.record_command(
            RecordedPayload::RayTracingBegin(RayTracingBegin {
                label: desc.label.clone(),
            }),
            Vec::new(),
            COMPUTE,
        );
        self.set_phase(RecorderPhase::ComputeScopeOpen);
        Ok(RayTracingScope {
            recorder: self,
            pipeline: None,
            groups: Vec::new(),
            immediates: Vec::new(),
            ended: false,
        })
    }
}

/// An exclusive ray-tracing command scope.
pub struct RayTracingScope<'a> {
    recorder: &'a mut CommandRecorder,
    pipeline: Option<RayTracingPipeline>,
    groups: Vec<BoundGroup>,
    /// Validated writes current for the presently bound pipeline.  A pipeline
    /// change clears them: an immediate byte address space belongs to a
    /// `PipelineInterface`, so retaining writes across a different interface
    /// would make their native stage/range meaning ambiguous.
    immediates: Vec<ImmediateWrite>,
    ended: bool,
}
impl RayTracingScope<'_> {
    /// Binds a ray-tracing pipeline.
    pub fn set_pipeline(&mut self, pipeline: &RayTracingPipeline) -> RhiResult<()> {
        require_same(
            pipeline.device_identity(),
            self.recorder.device_identity(),
            "ray-tracing pipeline",
        )?;
        self.pipeline = Some(pipeline.clone());
        self.immediates.clear();
        Ok(())
    }
    /// Sets bytes in one declared immediate-data range of the bound pipeline.
    ///
    /// The range declaration supplies stage visibility; callers do not name a
    /// native push/root-constant stage mask.  The write must fit completely in
    /// one declared range, be non-empty, and respect the device alignment.  A
    /// later `set_pipeline` intentionally clears these writes because immediate
    /// addresses are owned by a pipeline interface.
    pub fn set_immediates(&mut self, offset: u32, bytes: &[u8]) -> RhiResult<()> {
        if !self
            .recorder
            .capabilities()
            .supports_feature(OptionalFeature::Immediates)
        {
            return unsupported(
                "RayTracingScope::set_immediates",
                "this device does not enable immediate pipeline data",
            );
        }
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "immediate data needs a bound ray-tracing pipeline",
            )
            .at("RayTracingScope::set_immediates")
        })?;
        if bytes.is_empty() {
            return invalid(
                "RayTracingScope::set_immediates",
                "immediate-data writes must be non-empty",
            );
        }
        let size = u32::try_from(bytes.len()).map_err(|_| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "immediate-data write size does not fit the portable address space",
            )
            .at("RayTracingScope::set_immediates")
        })?;
        let end = offset.checked_add(size).ok_or_else(|| {
            RhiError::new(RhiErrorKind::InvalidUsage, "immediate-data write overflows")
                .at("RayTracingScope::set_immediates")
        })?;
        let alignment = self
            .recorder
            .capabilities()
            .limit(LimitKey::ImmediateDataAlignment)
            .unwrap_or(1);
        if alignment == 0 || u64::from(offset) % alignment != 0 || u64::from(size) % alignment != 0
        {
            return invalid(
                "RayTracingScope::set_immediates",
                "immediate-data offset and size must satisfy ImmediateDataAlignment",
            );
        }
        let write_range = pipeline
            .descriptor()
            .interface
            .descriptor()
            .immediate_ranges
            .iter()
            .find(|range| {
                range.offset <= offset
                    && range
                        .offset
                        .checked_add(range.size)
                        .is_some_and(|range_end| end <= range_end)
            })
            .copied()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "immediate-data write is not wholly contained in a declared pipeline range",
                )
                .at("RayTracingScope::set_immediates")
            })?;
        self.immediates.retain(|write| write.offset != offset);
        self.immediates.push(ImmediateWrite {
            offset,
            bytes: bytes.to_vec(),
            visibility: write_range.visibility,
        });
        Ok(())
    }
    /// Binds a group used by the ray pipeline.
    pub fn set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        require_same(
            group.device_identity(),
            self.recorder.device_identity(),
            "bind group",
        )?;
        require_valid_dynamic_offsets(index, group, dynamic_offsets)?;
        let bound = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        if let Some(old) = self.groups.iter_mut().find(|entry| entry.index == index) {
            *old = bound;
        } else {
            self.groups.push(bound);
        }
        Ok(())
    }
    /// Dispatches rays over a non-empty three-dimensional extent.
    pub fn dispatch_rays(
        &mut self,
        table: &RayTracingShaderTable,
        width: u32,
        height: u32,
        depth: u32,
    ) -> RhiResult<()> {
        let pipeline = self.pipeline.clone().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a ray dispatch needs a bound ray-tracing pipeline",
            )
            .at("RayTracingScope::dispatch_rays")
        })?;
        if width == 0 || height == 0 || depth == 0 {
            return invalid(
                "RayTracingScope::dispatch_rays",
                "ray dispatch dimensions must be non-zero",
            );
        }
        let max = self
            .recorder
            .capabilities()
            .limit(LimitKey::MaxRayDispatchCount)
            .unwrap_or(0);
        if [width, height, depth]
            .into_iter()
            .any(|v| u64::from(v) > max)
        {
            return invalid(
                "RayTracingScope::dispatch_rays",
                "ray dispatch dimension exceeds MaxRayDispatchCount",
            );
        }
        validate_bound_groups(&pipeline.descriptor().interface, &self.groups)?;
        let mut uses = Vec::new();
        for group in &self.groups {
            uses.extend(bound_group_uses(&group.group)?);
        }
        for region in std::iter::once(&table.ray_generation)
            .chain(table.miss.iter())
            .chain(table.hit.iter())
        {
            validate_shader_table_region(
                region,
                self.recorder.device_identity(),
                self.recorder
                    .capabilities()
                    .limit(LimitKey::MaxRayTracingPipelineGroupDataSize),
                self.recorder
                    .capabilities()
                    .limit(LimitKey::RayTracingPipelineGroupDataAlignment),
                self.recorder
                    .capabilities()
                    .limit(LimitKey::RayTracingPipelineGroupDataOffsetAlignment),
            )?;
            uses.push(ResourceUse::Buffer(crate::api::command::BufferUse {
                buffer: region.buffer.clone(),
                range: region.range,
                stages: PipelineScope::RAY_TRACING,
                access: AccessMask::RAY_TRACING_SHADER_DATA_READ,
            }));
        }
        self.recorder.record_command(
            RecordedPayload::RayTracingDispatch(Box::new(RayTracingDispatch {
                pipeline,
                groups: self.groups.clone(),
                table: table.clone(),
                dimensions: (width, height, depth),
                immediates: self.immediates.clone(),
            })),
            uses,
            COMPUTE,
        );
        Ok(())
    }
    /// Ends the scope.
    pub fn end(mut self) -> RhiResult<()> {
        self.recorder
            .record_command(RecordedPayload::RayTracingEnd, Vec::new(), COMPUTE);
        self.recorder.set_phase(RecorderPhase::Open);
        self.ended = true;
        Ok(())
    }
}

fn validate_shader_table_region(
    region: &RayTracingShaderTableRegion,
    device: crate::api::identity::DeviceIdentity,
    maximum: Option<u64>,
    alignment: Option<u64>,
    offset_alignment: Option<u64>,
) -> RhiResult<()> {
    require_same(
        region.buffer.device_identity(),
        device,
        "shader-table buffer",
    )?;
    if !region
        .buffer
        .descriptor()
        .usage
        .contains(BufferUsage::INDIRECT)
    {
        return invalid(
            "RayTracingScope::dispatch_rays",
            "shader-table buffers require INDIRECT usage",
        );
    }
    if region.range.size == 0 || region.stride == 0 || region.range.size % region.stride != 0 {
        return invalid(
            "RayTracingScope::dispatch_rays",
            "shader-table ranges must contain whole non-zero records",
        );
    }
    if maximum.is_some_and(|value| region.stride > value)
        || alignment.is_some_and(|value| value != 0 && region.stride % value != 0)
        || offset_alignment.is_some_and(|value| value != 0 && region.range.offset % value != 0)
    {
        return invalid(
            "RayTracingScope::dispatch_rays",
            "shader-table size or alignment violates the enabled limits",
        );
    }
    crate::api::resource::buffer::validate_buffer_range(
        region.range,
        region.buffer.descriptor().size,
    )
}
impl Drop for RayTracingScope<'_> {
    fn drop(&mut self) {
        if !self.ended {
            self.recorder
                .poison("a ray-tracing scope was dropped without end()");
        }
    }
}

impl crate::api::command::RasterScope<'_> {
    /// Binds a mesh/task pipeline compatible with this raster attachment set.
    pub fn set_mesh_pipeline(&mut self, pipeline: &MeshPipeline) -> RhiResult<()> {
        self.set_mesh_pipeline_inner(pipeline)
    }
    /// Dispatches task/mesh workgroups using the bound mesh pipeline.
    pub fn dispatch_mesh(&mut self, x: u32, y: u32, z: u32) -> RhiResult<()> {
        self.dispatch_mesh_inner(x, y, z)
    }
    /// Dispatches mesh workgroups described by an indirect argument record.
    pub fn dispatch_mesh_indirect(&mut self, arguments: &Buffer, offset: u64) -> RhiResult<()> {
        self.dispatch_mesh_indirect_inner(arguments, offset, None)
    }
    /// Dispatches mesh workgroups with an indirect count record.
    pub fn dispatch_mesh_indirect_count(
        &mut self,
        arguments: &Buffer,
        offset: u64,
        count_buffer: &Buffer,
        count_offset: u64,
        max_count: u32,
    ) -> RhiResult<()> {
        self.dispatch_mesh_indirect_inner(
            arguments,
            offset,
            Some((count_buffer, count_offset, max_count)),
        )
    }
}

fn build_input_uses(destination: &AccelerationStructure) -> RhiResult<Vec<ResourceUse>> {
    use crate::api::resource::{AccelerationStructureDescriptor, BlasGeometry};
    let mut uses = Vec::new();
    match destination.descriptor() {
        AccelerationStructureDescriptor::BottomLevel(desc) => {
            for geometry in &desc.geometries {
                match geometry {
                    BlasGeometry::Triangles(g) => {
                        uses.push(buffer_read(&g.vertices, g.vertex_range));
                        if let Some((buffer, range, _)) = &g.indices {
                            uses.push(buffer_read(buffer, *range));
                        }
                    }
                    BlasGeometry::Aabbs(g) => uses.push(buffer_read(&g.boxes, g.range)),
                }
            }
        }
        AccelerationStructureDescriptor::TopLevel(desc) => {
            for instance in &desc.instances {
                uses.push(ResourceUse::AccelerationStructure(
                    crate::api::command::AccelerationStructureUse {
                        structure: instance.bottom_level.clone(),
                        stages: PipelineScope::COMPUTE,
                        access: AccessMask::ACCELERATION_STRUCTURE_BUILD_READ,
                    },
                ));
            }
        }
    }
    Ok(uses)
}
fn buffer_read(buffer: &Buffer, range: BufferRange) -> ResourceUse {
    ResourceUse::Buffer(crate::api::command::BufferUse {
        buffer: buffer.clone(),
        range,
        stages: PipelineScope::COMPUTE,
        access: AccessMask::ACCELERATION_STRUCTURE_BUILD_READ,
    })
}
fn require_same(
    actual: crate::api::identity::DeviceIdentity,
    expected: crate::api::identity::DeviceIdentity,
    what: &str,
) -> RhiResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            format!("{what} belongs to a different device"),
        ))
    }
}
fn invalid<T>(at: &'static str, message: &'static str) -> RhiResult<T> {
    Err(RhiError::new(RhiErrorKind::InvalidUsage, message).at(at))
}
fn unsupported<T>(at: &'static str, message: &'static str) -> RhiResult<T> {
    Err(RhiError::new(RhiErrorKind::Unsupported, message).at(at))
}
