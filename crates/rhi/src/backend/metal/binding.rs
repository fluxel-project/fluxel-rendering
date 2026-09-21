//! Immutable direct-binding packets and the native binding ABI for Metal.
//!
//! Metal deliberately has three independent argument index spaces (`buffer`,
//! `texture`, and `sampler`) for every shader stage.  Fluxel's group/slot
//! vocabulary is not any one of those spaces, so this file is the sole
//! translation authority.  The ABI is built once with the pipeline, while a
//! bind group only retains its immutable portable packet.  Keeping those two
//! concerns together prevents a pipeline and a command encoder from silently
//! deriving different indices for the same logical slot.

use std::any::Any;

use crate::api::binding::backend::BindGroupBackend;
use crate::api::binding::{
    BindGroup, BindGroupDescriptor, BindGroupIndex, BindingCount, BindingKind, BindingResource,
    BindingSlotId,
};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::PipelineInterface;
use crate::api::shader::{ShaderImmediateRequirement, ShaderInterface, ShaderStages};

/// Vertex fetch is part of the MSL vertex-stage buffer namespace.  This is a
/// fixed reservation, not the number of streams in a particular pipeline:
/// compiled MSL argument indices must not move when an otherwise compatible
/// vertex-input declaration gains or loses an unused stream.
const VERTEX_STREAM_RESERVED_BUFFERS: u32 = 15;

/// Metal has independent buffer, texture and sampler index spaces.  The packet
/// retains the portable resources (which in turn retain their native objects)
/// and leaves stage-specific encoder calls to command lowering.
pub(super) struct MetalBindGroup {
    entries: Vec<(BindingSlotId, BindingResource)>,
}

impl MetalBindGroup {
    pub(super) fn entries(&self) -> &[(BindingSlotId, BindingResource)] {
        &self.entries
    }
}

impl BindGroupBackend for MetalBindGroup {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) fn create_bind_group(descriptor: &BindGroupDescriptor) -> MetalBindGroup {
    MetalBindGroup {
        entries: descriptor
            .entries
            .iter()
            .map(|entry| (entry.slot, entry.resource.clone()))
            .collect(),
    }
}

/// Which one of Metal's independent argument namespaces a logical binding
/// occupies.  Storage and sampled textures share Metal's texture namespace;
/// their access semantics remain validated by the portable layout and shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MetalBindingClass {
    Buffer,
    Texture,
    Sampler,
}

/// Per-stage native argument indices for one logical slot.  An absent stage is
/// intentional: a fragment-only binding must not consume a vertex index merely
/// because both stages happen to be encoded by the same render command encoder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MetalStageBindingIndices {
    pub(super) vertex: Option<u32>,
    pub(super) fragment: Option<u32>,
    pub(super) compute: Option<u32>,
}

impl MetalStageBindingIndices {
    pub(super) fn for_visibility(self, visibility: ShaderStages) -> Self {
        Self {
            vertex: if visibility.contains(ShaderStages::VERTEX) {
                self.vertex
            } else {
                None
            },
            fragment: if visibility.contains(ShaderStages::FRAGMENT) {
                self.fragment
            } else {
                None
            },
            compute: if visibility.contains(ShaderStages::COMPUTE) {
                self.compute
            } else {
                None
            },
        }
    }
}

/// One logical layout slot after it has been assigned Metal argument indices.
#[derive(Clone, Debug)]
pub(super) struct MetalSlotBindingAbi {
    pub(super) slot: BindingSlotId,
    pub(super) class: MetalBindingClass,
    pub(super) first: MetalStageBindingIndices,
    pub(super) count: u32,
    pub(super) visibility: ShaderStages,
    pub(super) dynamic_offset: bool,
}

/// One logical bind group in a pipeline ABI.
#[derive(Clone, Debug, Default)]
pub(super) struct MetalGroupBindingAbi {
    slots: Vec<MetalSlotBindingAbi>,
}

impl MetalGroupBindingAbi {
    /// The slots are in the public layout's canonical ascending-slot order.
    pub(super) fn slots(&self) -> &[MetalSlotBindingAbi] {
        &self.slots
    }

    pub(super) fn slot(&self, slot: BindingSlotId) -> Option<&MetalSlotBindingAbi> {
        self.slots.iter().find(|candidate| candidate.slot == slot)
    }
}

/// Pipeline-private Metal argument ABI.
///
/// The allocation order is fixed and deliberately simple:
/// `group order -> ascending BindingSlotId -> element order`, independently for
/// each `(stage, class)` space.  MSL artifacts used with the direct Metal path
/// must use this ABI when spelling `[[buffer(n)]]`, `[[texture(n)]]`, and
/// `[[sampler(n)]]`; public group/slot numbers never leak into native calls.
#[derive(Clone, Debug, Default)]
pub(super) struct MetalBindingAbi {
    groups: Vec<MetalGroupBindingAbi>,
    /// One reserved direct-buffer index per visible stage for the complete
    /// portable immediate address space. It is allocated *after* ordinary
    /// resource buffers, so MSL can spell it without colliding with either a
    /// vertex stream or a bind-group buffer.
    immediates: MetalImmediateAbi,
}

#[derive(Clone, Debug, Default)]
pub(super) struct MetalImmediateAbi {
    pub(super) indices: MetalStageBindingIndices,
    /// The largest byte the executable shader interfaces can read.  This is
    /// intentionally not the size of the pipeline interface's superset.
    pub(super) size: u32,
    /// Artifact-declared intervals which must survive command lowering. Writes
    /// to a legal-but-unused interface range have no executable observer and
    /// are deliberately omitted from the native byte packet.
    pub(super) requirements: Vec<ShaderImmediateRequirement>,
}

impl MetalBindingAbi {
    /// Derives the complete ABI from an already-validated portable interface.
    ///
    /// Runtime-sized arrays cannot have a stable direct-Metal argument span:
    /// their active length belongs to a packet, while MSL direct argument
    /// indices are baked into a function.  They therefore remain fail-closed
    /// until this backend gains an argument-buffer ABI.
    pub(super) fn from_compute_interface(
        interface: &PipelineInterface,
        shader: &ShaderInterface,
    ) -> RhiResult<Self> {
        Self::from_stage_interfaces(interface, &[(ShaderStages::COMPUTE, shader)], 0)
    }

    /// Raster pipelines reserve the vertex-stage buffer indices occupied by the
    /// fixed vertex-input stream before assigning group bindings.  Metal uses
    /// the same `[[buffer(n)]]` namespace for vertex fetch buffers and ordinary
    /// vertex-stage arguments, so starting bind groups at zero would make a
    /// perfectly valid layout overwrite vertex data at draw time.
    pub(super) fn from_raster_interfaces(
        interface: &PipelineInterface,
        vertex: &ShaderInterface,
        fragment: Option<&ShaderInterface>,
    ) -> RhiResult<Self> {
        let mut stages = vec![(ShaderStages::VERTEX, vertex)];
        if let Some(fragment) = fragment {
            stages.push((ShaderStages::FRAGMENT, fragment));
        }
        Self::from_stage_interfaces(interface, &stages, VERTEX_STREAM_RESERVED_BUFFERS)
    }

    fn from_stage_interfaces(
        interface: &PipelineInterface,
        stages: &[(ShaderStages, &ShaderInterface)],
        vertex_buffer_reserve: u32,
    ) -> RhiResult<Self> {
        let mut next = NativeArgumentCursors {
            vertex: ClassCursors {
                buffers: vertex_buffer_reserve,
                ..ClassCursors::default()
            },
            ..NativeArgumentCursors::default()
        };
        let mut groups = Vec::with_capacity(interface.descriptor().groups.len());
        for (group_index, layout) in interface.descriptor().groups.iter().enumerate() {
            let mut group = MetalGroupBindingAbi::default();
            for entry in &layout.descriptor().entries {
                // A layout may intentionally be a superset of an entry point.
                // Only resources the compiled artifacts actually declare own an
                // MSL argument index.  Allocating the superset here would make
                // artifact ABI depend on unrelated layout declarations.
                let mut active_visibility = None;
                let mut active_kind = None;
                let mut active_count = None;
                for (stage, shader) in stages {
                    if let Some(requirement) = shader.resources().iter().find(|requirement| {
                        requirement.group == BindGroupIndex::new(group_index as u32)
                            && requirement.slot == entry.slot
                    }) {
                        active_visibility = Some(
                            active_visibility
                                .map_or(*stage, |known: ShaderStages| known.union(*stage)),
                        );
                        active_kind = Some(&requirement.kind);
                        active_count = Some(requirement.count);
                    }
                }
                let Some(visibility) = active_visibility else {
                    continue;
                };
                let kind = active_kind.expect("active Metal resource has a kind");
                let count = fixed_count(
                    active_count.expect("active Metal resource has a count"),
                    entry.slot,
                )?;
                let class = class_of(kind)?;
                let first = next.allocate(class, visibility, count)?;
                group.slots.push(MetalSlotBindingAbi {
                    slot: entry.slot,
                    class,
                    first,
                    count,
                    visibility,
                    dynamic_offset: entry.dynamic_offset,
                });
            }
            groups.push(group);
        }
        let mut visibility = None;
        let mut size = 0u32;
        let mut requirements = Vec::new();
        for (stage, shader) in stages {
            if shader.immediate_requirements().is_empty() {
                continue;
            }
            visibility = Some(visibility.map_or(*stage, |known: ShaderStages| known.union(*stage)));
            for requirement in shader.immediate_requirements() {
                size = size.max(
                    requirement
                        .offset
                        .checked_add(requirement.size)
                        .ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal immediate range overflows",
                            )
                        })?,
                );
                requirements.push(*requirement);
            }
        }
        let indices = if size == 0 {
            MetalStageBindingIndices::default()
        } else {
            next.allocate(
                MetalBindingClass::Buffer,
                visibility.ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal immediate ABI has no visible stages",
                    )
                })?,
                1,
            )?
        };
        for (name, cursors) in [
            ("vertex", &next.vertex),
            ("fragment", &next.fragment),
            ("compute", &next.compute),
        ] {
            if cursors.buffers > 31 || cursors.textures > 31 || cursors.samplers > 16 {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    format!(
                        "Metal direct {name}-stage argument ABI exceeds buffer/texture/sampler limits (31/31/16)"
                    ),
                ));
            }
        }
        Ok(Self {
            groups,
            immediates: MetalImmediateAbi {
                indices,
                size,
                requirements,
            },
        })
    }

    pub(super) fn group(&self, index: BindGroupIndex) -> Option<&MetalGroupBindingAbi> {
        self.groups.get(index.get() as usize)
    }
    pub(super) const fn immediates(&self) -> &MetalImmediateAbi {
        &self.immediates
    }

    /// Reifies the portable dynamic-offset sequence into named ABI elements.
    /// The recorder has already checked this contract, but native lowering
    /// repeats the bounded arithmetic at its trust boundary: otherwise a
    /// malformed replay/capture packet could move a Metal buffer binding beyond
    /// the range the bind group owns.
    pub(super) fn dynamic_offsets(
        &self,
        index: BindGroupIndex,
        group: &BindGroup,
        offsets: &[u32],
    ) -> RhiResult<Vec<MetalDynamicOffset>> {
        let Some(abi_group) = self.group(index) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("Metal pipeline has no bind group {}", index.get()),
            ));
        };
        let expected = group.layout().dynamic_offset_count() as usize;
        if offsets.len() != expected {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "Metal bind group {} requires {expected} dynamic offsets, got {}",
                    index.get(),
                    offsets.len()
                ),
            ));
        }

        let mut cursor = 0;
        let mut resolved = Vec::with_capacity(expected);
        for layout_slot in &group.layout().descriptor().entries {
            if !layout_slot.dynamic_offset {
                continue;
            }
            let entry = group
                .descriptor()
                .entries
                .iter()
                .find(|entry| entry.slot == layout_slot.slot)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        format!(
                            "bind group is missing dynamic slot {}",
                            layout_slot.slot.get()
                        ),
                    )
                })?;
            let bindings = buffer_elements(&entry.resource, layout_slot.slot)?;
            let abi_slot = abi_group.slot(layout_slot.slot);
            if let Some(abi_slot) = abi_slot {
                if abi_slot.class != MetalBindingClass::Buffer || !abi_slot.dynamic_offset {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal dynamic offset targets a non-buffer ABI slot",
                    ));
                }
            }
            if abi_slot.is_some_and(|slot| bindings.len() != slot.count as usize) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "Metal bind-group slot {} has {} buffer elements, ABI requires {}",
                        layout_slot.slot.get(),
                        bindings.len(),
                        abi_slot.map_or(0, |slot| slot.count)
                    ),
                ));
            }
            for (element, binding) in bindings.iter().enumerate() {
                let offset = *offsets.get(cursor).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "dynamic-offset sequence is truncated",
                    )
                })?;
                cursor += 1;
                let end = binding
                    .range
                    .offset
                    .checked_add(u64::from(offset))
                    .and_then(|effective| effective.checked_add(binding.range.size));
                if end.is_none_or(|end| end > binding.buffer.descriptor().size) {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        format!(
                            "Metal dynamic offset {offset} moves bind-group slot {} element {element} outside its buffer",
                            layout_slot.slot.get()
                        ),
                    ));
                }
                // Dynamic offsets remain a complete public packet sequence even
                // when this artifact does not use the layout slot. Consume and
                // validate it, but do not create a native binding for it.
                if abi_slot.is_some() {
                    resolved.push(MetalDynamicOffset {
                        slot: layout_slot.slot,
                        element: element as u32,
                        offset: u64::from(offset),
                    });
                }
            }
        }
        debug_assert_eq!(cursor, offsets.len());
        Ok(resolved)
    }
}

/// One checked dynamic offset paired with its logical buffer element.  Command
/// lowering uses `MetalBindingAbi::group(...).slot(...)` to select the matching
/// per-stage Metal buffer index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MetalDynamicOffset {
    pub(super) slot: BindingSlotId,
    pub(super) element: u32,
    pub(super) offset: u64,
}

#[derive(Default)]
struct NativeArgumentCursors {
    vertex: ClassCursors,
    fragment: ClassCursors,
    compute: ClassCursors,
}

#[derive(Default)]
struct ClassCursors {
    buffers: u32,
    textures: u32,
    samplers: u32,
}

impl NativeArgumentCursors {
    fn allocate(
        &mut self,
        class: MetalBindingClass,
        visibility: ShaderStages,
        count: u32,
    ) -> RhiResult<MetalStageBindingIndices> {
        Ok(MetalStageBindingIndices {
            vertex: visibility
                .contains(ShaderStages::VERTEX)
                .then(|| allocate_one(&mut self.vertex, class, count))
                .transpose()?,
            fragment: visibility
                .contains(ShaderStages::FRAGMENT)
                .then(|| allocate_one(&mut self.fragment, class, count))
                .transpose()?,
            compute: visibility
                .contains(ShaderStages::COMPUTE)
                .then(|| allocate_one(&mut self.compute, class, count))
                .transpose()?,
        })
    }
}

fn allocate_one(
    cursors: &mut ClassCursors,
    class: MetalBindingClass,
    count: u32,
) -> RhiResult<u32> {
    let cursor = match class {
        MetalBindingClass::Buffer => &mut cursors.buffers,
        MetalBindingClass::Texture => &mut cursors.textures,
        MetalBindingClass::Sampler => &mut cursors.samplers,
    };
    let first = *cursor;
    *cursor = cursor.checked_add(count).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal argument-index space overflows u32",
        )
    })?;
    Ok(first)
}

fn class_of(kind: &BindingKind) -> RhiResult<MetalBindingClass> {
    match kind {
        BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. } => {
            Ok(MetalBindingClass::Buffer)
        }
        BindingKind::SampledTexture { .. } | BindingKind::StorageTexture { .. } => {
            Ok(MetalBindingClass::Texture)
        }
        BindingKind::Sampler { .. } => Ok(MetalBindingClass::Sampler),
        BindingKind::AccelerationStructure | BindingKind::ExternalTexture => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal direct binding ABI does not implement acceleration structures or external textures",
        )),
    }
}

fn fixed_count(count: BindingCount, slot: BindingSlotId) -> RhiResult<u32> {
    match count {
        BindingCount::One => Ok(1),
        BindingCount::Fixed(count) if count >= 2 => Ok(count),
        BindingCount::Fixed(_) => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "Metal binding slot {} has an invalid fixed element count",
                slot.get()
            ),
        )),
        BindingCount::RuntimeSized => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!(
                "Metal direct binding ABI does not implement runtime-sized array at slot {}",
                slot.get()
            ),
        )),
    }
}

fn buffer_elements(
    resource: &BindingResource,
    slot: BindingSlotId,
) -> RhiResult<Vec<&crate::api::resource::BufferBinding>> {
    match resource {
        BindingResource::Buffer(binding) => Ok(vec![binding]),
        BindingResource::BufferArray(bindings) => Ok(bindings.iter().collect()),
        _ => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "Metal dynamic bind-group slot {} is not a buffer",
                slot.get()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::binding::LayoutFingerprint;
    use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
    use crate::api::pipeline::{
        ImmediateRange, PipelineInterfaceCompatibilityId, PipelineInterfaceDescriptor,
    };

    fn interface(ranges: Vec<ImmediateRange>) -> PipelineInterface {
        let mut descriptor = PipelineInterfaceDescriptor::new(Vec::new());
        for range in ranges {
            descriptor = descriptor.with_immediate_range(range);
        }
        PipelineInterface::new(
            ObjectId::new(1),
            DeviceIdentity::new(DeviceInstanceId::new(1)),
            descriptor,
            PipelineInterfaceCompatibilityId::new(1),
            LayoutFingerprint([1; 32]),
        )
    }

    #[test]
    fn unused_immediate_interface_ranges_do_not_change_the_metal_shader_abi() {
        let shader = ShaderInterface::new()
            .with_immediate_requirement(ShaderImmediateRequirement::new(0, 4));
        let narrow = MetalBindingAbi::from_compute_interface(
            &interface(vec![ImmediateRange::new(0, 4, ShaderStages::COMPUTE)]),
            &shader,
        )
        .unwrap();
        let superset = MetalBindingAbi::from_compute_interface(
            &interface(vec![
                ImmediateRange::new(0, 4, ShaderStages::COMPUTE),
                ImmediateRange::new(4, 4, ShaderStages::COMPUTE),
            ]),
            &shader,
        )
        .unwrap();

        assert_eq!(narrow.immediates().indices, superset.immediates().indices);
        assert_eq!(narrow.immediates().size, 4);
        assert_eq!(superset.immediates().size, 4);
    }
}
