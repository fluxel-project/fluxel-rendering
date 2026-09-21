//! Specification section 56: the portable command IR and captured recorded work.
//!
//! One responsibility: **what a recording did, in portable semantic terms.** This
//! file is the middle of section 52's chain — the live
//! [`crate::api::command::RecordedWork`] retains a compact internal IR, and
//! [`CapturedRecordedWork`] is the expansion a ReplayRuntime reads. It is an
//! in-memory semantic IR and **not a file format** (section 56), which is why
//! nothing here has an opcode, a version, or a byte layout.
//!
//! Not owned here: the live recorder and its internal types (`api::command`,
//! module 04), the values the commands carry ([`super::mutation`]), the object
//! definitions they name ([`super::definition`]), and the plan they are submitted
//! in ([`super::plan`]).
//!
//! # The lowering is a normalization, and this is why
//!
//! Section 52.2 requires a live `RecordedWork` to make "portable commands" and
//! the "actual command/use sequence" retrievable, and permits a compact internal
//! IR. Module 04's IR is compact in one specific way that a reader must know
//! about: a draw does **not** record a preceding run of `SetRasterPipeline`,
//! `SetBindGroup`, `SetViewport`, and so on. It records the state that was
//! current when the draw was issued, in module 04's crate-private `RasterDraw`
//! record, because that is what a backend lowers —
//! a later state change must not retroactively rewrite an earlier draw.
//!
//! So the [`PortableCommand`] list for such a recording is a normalized
//! expansion:
//!
//! ```text
//! internal:  RasterDraw { pipeline, groups, viewport, .., range, instances }
//! captured:  SetRasterPipeline, SetBindGroup.., SetViewport, .., Draw
//! ```
//!
//! and two consecutive draws that used identical state therefore produce two
//! identical state-setting runs. That is faithful — replaying the expansion
//! produces the same GPU work — but it is not a byte-for-byte echo of the
//! caller's recording, and a tool that diffs a captured command list against its
//! own source of truth should expect the difference. The alternative, recording
//! state changes as they were issued, is a second recording of the same thing and
//! section 52.2 refuses it in as many words.
//!
//! # The two invariants a reader of this file should carry
//!
//! 1. **Order within `commands` is real GPU order.** It is the one place a
//!    capture can learn execution order, because [`super::SemanticEventId`] is
//!    CPU observation order and lane order plus explicit dependencies are the
//!    only other sources (§53.3).
//! 2. **A `PortableCommand` discriminant is not an opcode.** Section 56 is
//!    explicit that a Rust enum discriminant or memory layout must never be
//!    written directly as an artifact opcode: the Artifact Layer must supply its
//!    own tagged, versioned, bounds-checked, canonical encoding. Nothing in this
//!    file is `repr`, and nothing may be made `repr` for an encoding's
//!    convenience.

use core::ops::Range;

use crate::api::binding::BindGroupIndex;
use crate::api::command::{AccessMask, PipelineScope, QueryAccess, TextureUseIntent};
use crate::api::command::{Color, IndexFormat, Rect, Viewport};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::presentation::AcquiredFrameId;
use crate::api::resource::buffer::BufferRange;
use crate::api::resource::subresource::TextureSubresourceRange;
use crate::api::resource::{AccelerationStructureBuildMode, AccelerationStructureCopyMode};
use crate::api::shader::ShaderStages;
use crate::api::submission::LaneWorkDomains;

use super::mutation::{
    CapturedBlit, CapturedBufferCopy, CapturedBufferTextureCopy, CapturedColorAttachment,
    CapturedColorAttachmentView, CapturedDepthStencilAttachment, CapturedExternalImageCopy,
    CapturedRasterScope, CapturedReadbackRequest, CapturedResolve, CapturedTextureCopy,
    CapturedUploadDefinition,
};

/// Captures a finished recording into values only.  This helper is deliberately
/// crate-private: public tooling reaches the stored snapshot by ObjectId.
pub(crate) fn capture_recorded_work(
    work: &crate::api::command::RecordedWork,
) -> CapturedRecordedWork {
    CapturedRecordedWork {
        work: work.id(),
        device: work.device_identity(),
        domains: work.work_domains(),
        commands: work.commands().iter().flat_map(capture_command).collect(),
        merged_use_summary: work.resource_uses().iter().map(capture_use).collect(),
    }
}

fn capture_command(command: &crate::api::command::record::RecordedCommand) -> Vec<CapturedCommand> {
    use crate::api::command::record::RecordedPayload as P;
    let uses: Vec<CapturedResourceUse> = command.uses.iter().map(capture_use).collect();
    let state = |command| CapturedCommand {
        command,
        actual_uses: Vec::new(),
    };
    let action = |command| CapturedCommand {
        command,
        actual_uses: uses.clone(),
    };
    let groups = |groups: &[crate::api::command::record::BoundGroup]| -> Vec<CapturedCommand> {
        groups
            .iter()
            .map(|group| {
                state(PortableCommand::SetBindGroup {
                    index: group.index,
                    group: group.group.id(),
                    dynamic_offsets: group.dynamic_offsets.clone(),
                })
            })
            .collect()
    };
    let immediates =
        |writes: &[crate::api::command::record::ImmediateWrite]| -> Vec<CapturedCommand> {
            writes
                .iter()
                .map(|write| {
                    state(PortableCommand::SetImmediates {
                        offset: write.offset,
                        bytes: write.bytes.clone(),
                        visibility: write.visibility,
                    })
                })
                .collect()
        };
    let raster_state = |pipeline: &crate::api::pipeline::RasterPipeline,
                        groups_: &[crate::api::command::record::BoundGroup],
                        vertex_buffers: &[(u32, crate::api::resource::buffer::BufferBinding)],
                        index: &Option<crate::api::command::record::BoundIndexBuffer>,
                        viewport: Option<Viewport>,
                        scissor: Option<Rect>,
                        blend_constant: Color,
                        stencil_reference: u32| {
        let mut result = vec![state(PortableCommand::SetRasterPipeline(pipeline.id()))];
        result.extend(groups(groups_));
        result.extend(vertex_buffers.iter().map(|(slot, binding)| {
            state(PortableCommand::SetVertexBuffer {
                slot: *slot,
                buffer: binding.buffer.id(),
                range: binding.range,
            })
        }));
        if let Some(index) = index {
            result.push(state(PortableCommand::SetIndexBuffer {
                buffer: index.binding.buffer.id(),
                range: index.binding.range,
                format: index.format,
            }));
        }
        if let Some(viewport) = viewport {
            result.push(state(PortableCommand::SetViewport(viewport)));
        }
        if let Some(scissor) = scissor {
            result.push(state(PortableCommand::SetScissor(scissor)));
        }
        result.push(state(PortableCommand::SetBlendConstant(blend_constant)));
        result.push(state(PortableCommand::SetStencilReference(
            stencil_reference,
        )));
        result
    };
    match &command.payload {
        P::DebugPush(label) => vec![action(PortableCommand::PushDebugGroup(
            label.0.clone().unwrap_or_default(),
        ))],
        P::DebugPop => vec![action(PortableCommand::PopDebugGroup)],
        P::DebugMarker(label) => vec![action(PortableCommand::DebugMarker(
            label.0.clone().unwrap_or_default(),
        ))],
        P::RasterBegin(begin) => vec![action(PortableCommand::BeginRaster(capture_raster_scope(
            begin,
        )))],
        P::RasterEnd => vec![action(PortableCommand::EndRaster)],
        P::ComputeBegin(begin) => vec![action(PortableCommand::BeginCompute {
            label: begin.label.clone(),
        })],
        P::ComputeEnd => vec![action(PortableCommand::EndCompute)],
        P::RayTracingBegin(begin) => vec![action(PortableCommand::BeginRayTracing {
            label: begin.label.clone(),
        })],
        P::RayTracingEnd => vec![action(PortableCommand::EndRayTracing)],
        P::TimestampWrite { set, index } => vec![action(PortableCommand::WriteTimestamp {
            query_set: set.id(),
            index: *index,
        })],
        P::QueryBegin { set, index } => vec![action(PortableCommand::BeginQuery {
            query_set: set.id(),
            index: *index,
        })],
        P::QueryEnd { set, index } => vec![action(PortableCommand::EndQuery {
            query_set: set.id(),
            index: *index,
        })],
        P::QueryResolve(resolve) => vec![action(PortableCommand::ResolveQuerySet {
            query_set: resolve.set.id(),
            first_query: resolve.first_query,
            query_count: resolve.query_count,
            destination: resolve.destination.id(),
            destination_offset: resolve.destination_offset,
        })],
        P::MeshDispatch(dispatch) => {
            let mut result = vec![state(PortableCommand::SetMeshPipeline(
                dispatch.pipeline.id(),
            ))];
            result.extend(groups(&dispatch.groups));
            result.push(action(PortableCommand::DispatchMesh {
                x: dispatch.workgroups.0,
                y: dispatch.workgroups.1,
                z: dispatch.workgroups.2,
            }));
            result
        }
        P::MeshIndirect(dispatch) => {
            let mut result = vec![state(PortableCommand::SetMeshPipeline(
                dispatch.pipeline.id(),
            ))];
            result.extend(groups(&dispatch.groups));
            result.push(action(PortableCommand::DispatchMeshIndirect {
                arguments: dispatch.arguments.id(),
                offset: dispatch.offset,
                count_buffer: dispatch
                    .count_buffer
                    .as_ref()
                    .map(|(b, o, c)| (b.id(), *o, *c)),
            }));
            result
        }
        P::RasterDraw(draw) => {
            let mut result = raster_state(
                &draw.pipeline,
                &draw.groups,
                &draw.vertex_buffers,
                &draw.index,
                draw.viewport,
                draw.scissor,
                draw.blend_constant,
                draw.stencil_reference,
            );
            result.extend(immediates(&draw.immediates));
            result.push(action(match draw.index {
                Some(_) => PortableCommand::DrawIndexed {
                    indices: draw.range.clone(),
                    base_vertex: draw.base_vertex,
                    instances: draw.instances.clone(),
                },
                None => PortableCommand::Draw {
                    vertices: draw.range.clone(),
                    instances: draw.instances.clone(),
                },
            }));
            result
        }
        P::RasterIndirect(draw) => {
            let mut result = raster_state(
                &draw.pipeline,
                &draw.groups,
                &draw.vertex_buffers,
                &draw.index,
                draw.viewport,
                draw.scissor,
                draw.blend_constant,
                draw.stencil_reference,
            );
            result.push(action(PortableCommand::DrawIndirect {
                arguments: draw.arguments.id(),
                offset: draw.arguments_offset,
                count: draw.draw_count,
                stride: draw.stride,
                indexed: draw.index.is_some(),
                count_buffer: draw
                    .count
                    .as_ref()
                    .map(|(buffer, offset, count)| (buffer.id(), *offset, *count)),
            }));
            result
        }
        P::ComputeDispatch(dispatch) => {
            let mut result = vec![state(PortableCommand::SetComputePipeline(
                dispatch.pipeline.id(),
            ))];
            result.extend(groups(&dispatch.groups));
            result.extend(immediates(&dispatch.immediates));
            result.push(action(PortableCommand::Dispatch {
                x: dispatch.workgroups.0,
                y: dispatch.workgroups.1,
                z: dispatch.workgroups.2,
            }));
            result
        }
        P::ComputeIndirect(dispatch) => {
            let mut result = vec![state(PortableCommand::SetComputePipeline(
                dispatch.pipeline.id(),
            ))];
            result.extend(groups(&dispatch.groups));
            result.push(action(PortableCommand::DispatchIndirect {
                arguments: dispatch.arguments.id(),
                offset: dispatch.arguments_offset,
            }));
            result
        }
        P::RayTracingDispatch(dispatch) => {
            let mut result = vec![state(PortableCommand::SetRayTracingPipeline(
                dispatch.pipeline.id(),
            ))];
            result.extend(groups(&dispatch.groups));
            result.extend(immediates(&dispatch.immediates));
            result.push(action(PortableCommand::TraceRays {
                table: capture_ray_table(&dispatch.table),
                dimensions: dispatch.dimensions,
            }));
            result
        }
        P::AccelerationStructure(command) => vec![action(match command {
            crate::api::command::record::AccelerationStructureCommand::Build {
                destination,
                scratch,
                mode,
            } => PortableCommand::BuildAccelerationStructure {
                destination: destination.id(),
                scratch: scratch.id(),
                mode: *mode,
            },
            crate::api::command::record::AccelerationStructureCommand::Copy {
                source,
                destination,
                mode,
            } => PortableCommand::CopyAccelerationStructure {
                source: source.id(),
                destination: destination.id(),
                mode: *mode,
            },
            crate::api::command::record::AccelerationStructureCommand::WriteCompactedSize {
                source,
                destination,
                destination_offset,
            } => PortableCommand::WriteAccelerationStructureCompactedSize {
                source: source.id(),
                destination: destination.id(),
                destination_offset: *destination_offset,
            },
        })],
        P::Copy(copy) => vec![action(capture_copy(copy))],
        P::Upload(upload) => vec![action(PortableCommand::Upload {
            upload: capture_upload(upload),
        })],
        P::Readback(ticket) => vec![action(PortableCommand::Readback {
            request: capture_readback(ticket),
        })],
    }
}

fn capture_raster_scope(begin: &crate::api::command::record::RasterBegin) -> CapturedRasterScope {
    let mut colors = Vec::new();
    for (location, attachment) in &begin.colors {
        colors.resize_with(*location as usize + 1, || None);
        colors[*location as usize] = Some(CapturedColorAttachment {
            view: capture_color_view(&attachment.view),
            load: attachment.load,
            store: attachment.store,
            depth_slice: attachment.depth_slice,
            resolve: attachment.resolve.as_ref().map(capture_color_view),
        });
    }
    CapturedRasterScope {
        label: begin.label.clone(),
        colors,
        depth_stencil: begin.depth_stencil.as_ref().map(|attachment| {
            CapturedDepthStencilAttachment {
                view: attachment.view.id(),
                depth: attachment.depth,
                stencil: attachment.stencil,
            }
        }),
    }
}

fn capture_color_view(
    view: &crate::api::command::ColorAttachmentView,
) -> CapturedColorAttachmentView {
    match view {
        crate::api::command::ColorAttachmentView::Texture(view) => {
            CapturedColorAttachmentView::TextureView(view.id())
        }
        crate::api::command::ColorAttachmentView::Frame(frame) => {
            CapturedColorAttachmentView::Frame(frame.frame_id())
        }
    }
}

fn capture_ray_table(
    table: &crate::api::command::RayTracingShaderTable,
) -> CapturedRayTracingShaderTable {
    let region = |region: &crate::api::command::RayTracingShaderTableRegion| {
        CapturedRayTracingShaderTableRegion {
            buffer: region.buffer.id(),
            range: region.range,
            stride: region.stride,
        }
    };
    CapturedRayTracingShaderTable {
        ray_generation: region(&table.ray_generation),
        miss: table.miss.as_ref().map(region),
        hit: table.hit.as_ref().map(region),
    }
}

fn capture_copy(copy: &crate::api::command::record::CopyRecord) -> PortableCommand {
    use crate::api::command::record::CopyRecord;
    match copy {
        CopyRecord::ExternalImage(copy) => {
            PortableCommand::CopyExternalImage(CapturedExternalImageCopy {
                source: copy.source.id(),
                source_origin: copy.source_origin,
                destination: copy.destination.id(),
                destination_subresource: copy.destination_subresource,
                destination_origin: copy.destination_origin,
                extent: copy.extent,
                flip_y: copy.flip_y,
                alpha_mode: copy.alpha_mode,
                color_space_conversion: copy.color_space_conversion,
            })
        }
        CopyRecord::ClearBuffer { buffer, range } => PortableCommand::ClearBuffer {
            buffer: buffer.id(),
            range: *range,
        },
        CopyRecord::ClearTexture {
            texture,
            subresources,
        } => PortableCommand::ClearTexture {
            texture: texture.id(),
            subresources: *subresources,
        },
        CopyRecord::Buffer(copy) => PortableCommand::CopyBuffer(CapturedBufferCopy {
            src: copy.src.id(),
            src_offset: copy.src_offset,
            dst: copy.dst.id(),
            dst_offset: copy.dst_offset,
            size: copy.size,
        }),
        CopyRecord::BufferToTexture(copy) => {
            PortableCommand::CopyBufferToTexture(capture_buffer_texture_copy(copy))
        }
        CopyRecord::TextureToBuffer(copy) => {
            PortableCommand::CopyTextureToBuffer(capture_buffer_texture_copy(copy))
        }
        CopyRecord::Texture(copy) => PortableCommand::CopyTexture(CapturedTextureCopy {
            src: copy.src.id(),
            src_subresource: copy.src_subresource,
            src_origin: copy.src_origin,
            dst: copy.dst.id(),
            dst_subresource: copy.dst_subresource,
            dst_origin: copy.dst_origin,
            extent: copy.extent,
        }),
        CopyRecord::Resolve(copy) => PortableCommand::Resolve(CapturedResolve {
            src: copy.src.id(),
            src_subresource: copy.src_subresource,
            src_origin: copy.src_origin,
            dst: copy.dst.id(),
            dst_subresource: copy.dst_subresource,
            dst_origin: copy.dst_origin,
            extent: copy.extent,
        }),
        CopyRecord::Blit(copy) => PortableCommand::Blit(CapturedBlit {
            src: copy.src.id(),
            src_subresource: copy.src_subresource,
            src_origin: copy.src_origin,
            src_extent: copy.src_extent,
            dst: copy.dst.id(),
            dst_subresource: copy.dst_subresource,
            dst_origin: copy.dst_origin,
            dst_extent: copy.dst_extent,
            filter: copy.filter,
        }),
    }
}

fn capture_buffer_texture_copy(
    copy: &crate::api::command::BufferTextureCopy,
) -> CapturedBufferTextureCopy {
    CapturedBufferTextureCopy {
        buffer: copy.buffer.id(),
        buffer_offset: copy.buffer_offset,
        bytes_per_row: copy.bytes_per_row,
        rows_per_image: copy.rows_per_image,
        texture: copy.texture.id(),
        texture_subresource: copy.texture_subresource,
        texture_origin: copy.texture_origin,
        extent: copy.extent,
    }
}
fn capture_upload(upload: &crate::api::resource::transfer::UploadJob) -> CapturedUploadDefinition {
    match upload.descriptor() {
        crate::api::resource::transfer::UploadDescriptor::Buffer(desc) => {
            CapturedUploadDefinition::Buffer {
                id: upload.id(),
                dst: desc.dst.id(),
                dst_offset: desc.dst_offset,
                bytes: desc.bytes.clone(),
            }
        }
        crate::api::resource::transfer::UploadDescriptor::Texture(desc) => {
            CapturedUploadDefinition::Texture {
                id: upload.id(),
                dst: desc.dst.id(),
                subresource: desc.subresource,
                origin: desc.origin,
                extent: desc.extent,
                source_layout: desc.source_layout,
                bytes: desc.bytes.clone(),
            }
        }
    }
}
fn capture_readback(
    ticket: &crate::api::resource::transfer::ReadbackTicket,
) -> CapturedReadbackRequest {
    match ticket.request() {
        crate::api::resource::transfer::ReadbackRequest::Buffer { src, range, .. } => {
            CapturedReadbackRequest::Buffer {
                ticket: ticket.id(),
                src: src.id(),
                range: *range,
            }
        }
        crate::api::resource::transfer::ReadbackRequest::Texture {
            src,
            subresource,
            origin,
            extent,
            ..
        } => CapturedReadbackRequest::Texture {
            ticket: ticket.id(),
            src: src.id(),
            subresource: *subresource,
            origin: *origin,
            extent: *extent,
        },
    }
}

fn capture_use(use_: &crate::api::command::ResourceUse) -> CapturedResourceUse {
    match use_ {
        crate::api::command::ResourceUse::Buffer(use_) => CapturedResourceUse::Buffer {
            buffer: use_.buffer.id(),
            range: use_.range,
            stages: use_.stages,
            access: use_.access,
        },
        crate::api::command::ResourceUse::Texture(use_) => CapturedResourceUse::Texture {
            texture: use_.texture.id(),
            subresources: use_.subresources,
            stages: use_.stages,
            access: use_.access,
            intent: use_.intent,
        },
        crate::api::command::ResourceUse::Frame(use_) => CapturedResourceUse::Frame {
            frame: use_.frame,
            stages: use_.stages,
            access: use_.access,
        },
        crate::api::command::ResourceUse::AccelerationStructure(use_) => {
            CapturedResourceUse::AccelerationStructure {
                structure: use_.structure.id(),
                stages: use_.stages,
                access: use_.access,
            }
        }
        crate::api::command::ResourceUse::Query(use_) => CapturedResourceUse::Query {
            set: use_.set.id(),
            first_query: use_.first_query,
            query_count: use_.query_count,
            stages: use_.stages,
            access: use_.access,
        },
    }
}

/// One command of a recording, in portable semantic terms.
///
/// The full vocabulary of what a recorder can have recorded. It is deliberately
/// *not* a mirror of a graphics API's command set: there is no descriptor-set
/// binding, no resource-barrier command, no root-signature command, and no
/// pipeline-cache command, because none of those is portable and none of them is
/// what a caller recorded. What a caller recorded is a pipeline, a group with its
/// dynamic offsets, buffers, viewport, scissor, and a draw.
///
/// # Why the draw carries instance and vertex ranges rather than counts
///
/// [`Self::Draw`] and [`Self::DrawIndexed`] carry [`Range`]s, because a draw's
/// range is a range rather than a count once `base_vertex` and a non-zero first
/// vertex are both in play. A count would force a replay to reconstruct an
/// offset from two other fields and would lose the `first_vertex` case entirely.
#[non_exhaustive]
#[derive(Clone)]
pub enum PortableCommand {
    /// A raster scope opened, with its attachment set.
    BeginRaster(CapturedRasterScope),

    /// The raster scope closed.
    EndRaster,

    /// The raster pipeline to draw with.
    SetRasterPipeline(ObjectId),

    /// The compute pipeline to dispatch with.
    SetComputePipeline(ObjectId),

    /// The mesh/task graphics pipeline used by following mesh dispatches.
    SetMeshPipeline(ObjectId),

    /// The ray-tracing pipeline used by following ray dispatches.
    SetRayTracingPipeline(ObjectId),

    /// A bind group was bound at a group index.
    SetBindGroup {
        /// Which group index, as the interface numbers it.
        index: BindGroupIndex,
        /// The group's identity.
        group: ObjectId,
        /// The dynamic offsets consumed, in the order section 21.3 fixes.
        dynamic_offsets: Vec<u32>,
    },

    /// A vertex buffer was bound at a slot.
    SetVertexBuffer {
        /// The vertex buffer slot.
        slot: u32,
        /// The buffer's identity.
        buffer: ObjectId,
        /// The range of it that is bound.
        range: BufferRange,
    },

    /// An index buffer was bound.
    SetIndexBuffer {
        /// The buffer's identity.
        buffer: ObjectId,
        /// The range of it that is bound.
        range: BufferRange,
        /// The element type indices are cut with.
        format: IndexFormat,
    },

    /// The viewport was set.
    SetViewport(Viewport),

    /// The scissor rect was set.
    SetScissor(Rect),

    /// The blend constant was set.
    SetBlendConstant(Color),

    /// The stencil reference value was set.
    SetStencilReference(u32),

    /// A non-indexed draw.
    Draw {
        /// Which vertices to draw.
        vertices: Range<u32>,
        /// Which instances to draw them as.
        instances: Range<u32>,
    },

    /// An indexed draw.
    DrawIndexed {
        /// Which indices to draw.
        indices: Range<u32>,
        /// The value added to each index before it is fetched.
        base_vertex: i32,
        /// Which instances to draw them as.
        instances: Range<u32>,
    },

    /// A compute scope opened.
    BeginCompute {
        /// The scope's diagnostic label.
        label: Label,
    },

    /// The compute scope closed.
    EndCompute,

    /// A ray-tracing scope opened.
    BeginRayTracing {
        /// The scope's diagnostic label.
        label: Label,
    },

    /// A ray-tracing scope closed.
    EndRayTracing,

    /// A dispatch.
    Dispatch {
        /// Workgroups along x.
        x: u32,
        /// Workgroups along y.
        y: u32,
        /// Workgroups along z.
        z: u32,
    },

    /// Direct mesh/task dispatch dimensions.
    DispatchMesh { x: u32, y: u32, z: u32 },

    /// Mesh/task dimensions read from a GPU argument buffer. `count_buffer`
    /// names the optional count source and portable maximum, never native bytes.
    DispatchMeshIndirect {
        arguments: ObjectId,
        offset: u64,
        count_buffer: Option<(ObjectId, u64, u32)>,
    },

    /// One immediate-data write active for a draw, dispatch, or ray dispatch.
    SetImmediates {
        offset: u32,
        bytes: Vec<u8>,
        visibility: ShaderStages,
    },

    /// A ray dispatch with shader-table ranges expressed by buffer IDs.
    TraceRays {
        table: CapturedRayTracingShaderTable,
        dimensions: (u32, u32, u32),
    },

    /// A BLAS/TLAS build or update.
    BuildAccelerationStructure {
        destination: ObjectId,
        scratch: ObjectId,
        mode: AccelerationStructureBuildMode,
    },

    /// An AS clone or compaction copy.
    CopyAccelerationStructure {
        source: ObjectId,
        destination: ObjectId,
        mode: AccelerationStructureCopyMode,
    },

    /// Writes a completed AS compacted size to a buffer.
    WriteAccelerationStructureCompactedSize {
        source: ObjectId,
        destination: ObjectId,
        destination_offset: u64,
    },

    /// A timestamp was written to a query-set slot.
    WriteTimestamp {
        /// Query-set identity.
        query_set: ObjectId,
        /// Slot within that set.
        index: u32,
    },

    /// A query began in the current scope.
    BeginQuery {
        /// Query-set identity.
        query_set: ObjectId,
        /// Slot within that set.
        index: u32,
    },

    /// A query ended in the current scope.
    EndQuery {
        /// Query-set identity.
        query_set: ObjectId,
        /// Slot within that set.
        index: u32,
    },

    /// Query results were resolved into a buffer.
    ResolveQuerySet {
        /// Query-set identity.
        query_set: ObjectId,
        /// First resolved slot.
        first_query: u32,
        /// Number of resolved slots.
        query_count: u32,
        /// Destination buffer identity.
        destination: ObjectId,
        /// Destination byte offset.
        destination_offset: u64,
    },

    /// A buffer range was cleared to zero.
    ClearBuffer {
        /// Buffer identity.
        buffer: ObjectId,
        /// Cleared byte range.
        range: BufferRange,
    },

    /// A texture subresource range was cleared to its native zero value.
    ClearTexture {
        /// Texture identity.
        texture: ObjectId,
        /// Cleared subresource range.
        subresources: TextureSubresourceRange,
    },

    /// One or more indirect draw arguments were consumed.
    DrawIndirect {
        /// Argument-buffer identity.
        arguments: ObjectId,
        /// First argument byte offset.
        offset: u64,
        /// Number of draw argument records.
        count: u32,
        /// Byte stride between records.
        stride: u32,
        /// Whether the arguments describe indexed draws.
        indexed: bool,
        /// Optional GPU count source and maximum record count.
        count_buffer: Option<(ObjectId, u64, u32)>,
    },

    /// An indirect compute dispatch consumed its argument record.
    DispatchIndirect {
        /// Argument-buffer identity.
        arguments: ObjectId,
        /// Argument byte offset.
        offset: u64,
    },

    /// An upload was encoded into this recording, including its portable bytes
    /// and destination. A job identity alone would not be replayable.
    Upload {
        /// Complete captured mutation definition.
        upload: CapturedUploadDefinition,
    },

    /// A readback was encoded into this recording, including its request shape.
    Readback {
        /// Complete captured request.
        request: CapturedReadbackRequest,
    },

    /// A host-owned external image was copied into an RHI texture.
    /// This contains no host-native image object.
    CopyExternalImage(CapturedExternalImageCopy),

    /// A buffer-to-buffer copy.
    CopyBuffer(CapturedBufferCopy),

    /// A buffer-to-texture copy.
    CopyBufferToTexture(CapturedBufferTextureCopy),

    /// A texture-to-buffer copy.
    CopyTextureToBuffer(CapturedBufferTextureCopy),

    /// A texture-to-texture copy.
    CopyTexture(CapturedTextureCopy),

    /// A multisampled resolve.
    Resolve(CapturedResolve),

    /// A filtered blit.
    Blit(CapturedBlit),

    /// A debug group was opened.
    PushDebugGroup(String),

    /// A debug group was closed.
    PopDebugGroup,

    /// A debug marker was inserted.
    DebugMarker(String),
}

/// One shader-table range without a live buffer handle.
#[derive(Clone)]
pub struct CapturedRayTracingShaderTableRegion {
    pub buffer: ObjectId,
    pub range: BufferRange,
    pub stride: u64,
}

/// Complete shader table for one portable ray dispatch.
#[derive(Clone)]
pub struct CapturedRayTracingShaderTable {
    pub ray_generation: CapturedRayTracingShaderTableRegion,
    pub miss: Option<CapturedRayTracingShaderTableRegion>,
    pub hit: Option<CapturedRayTracingShaderTableRegion>,
}

/// One captured command with the resource uses it produced.
///
/// The pairing is section 37.1's, carried through to the capture side: a flat use
/// list cannot say which command produced a use, and "this draw consumed that
/// binding" is exactly the question a diagnostics pass asks.
///
/// A command that reads nothing in particular — a viewport change, a debug
/// marker — has an empty `actual_uses`, because actual use is generated at draw
/// and dispatch rather than at the state-setting verbs.
#[derive(Clone)]
pub struct CapturedCommand {
    /// What the command was.
    pub command: PortableCommand,

    /// The actual uses this command produced.
    pub actual_uses: Vec<CapturedResourceUse>,
}

/// Everything a recording did, in portable semantic terms.
///
/// `merged_use_summary` and the per-command `actual_uses` are both present on
/// purpose, and are not redundant at two levels:
///
/// ```text
/// commands[].actual_uses   what each command consumed, in command order
/// merged_use_summary       what the recording as a whole touched, merged
/// ```
///
/// The merged summary is the live [`crate::api::command::RecordedWork`]'s own
/// public answer (`resource_uses`); the per-command lists are what a
/// *diagnostic* needs. A
/// capture that kept only the merged form could say that a texture was written
/// but not by which draw.
///
/// `domains` is [`LaneWorkDomains`] and not a lane: it is what the recording
/// contains, and it is the value a `SubmissionPlan` builder checks lanes against
/// (section 40.1). Recording it here is what lets a Replay decide which lane a
/// captured batch could run on without re-deriving it from the command list.
#[derive(Clone)]
pub struct CapturedRecordedWork {
    /// The recording's identity.
    pub work: ObjectId,

    /// The device every object in the recording belongs to.
    pub device: DeviceIdentity,

    /// Which execution domains the recording contains.
    pub domains: LaneWorkDomains,

    /// The commands, in recording order.
    pub commands: Vec<CapturedCommand>,

    /// The merged use summary the live recording reported.
    pub merged_use_summary: Vec<CapturedResourceUse>,
}

/// One resource use, named rather than held.
///
/// The tooling-side counterpart of
/// [`crate::api::command::ResourceUse`], and the difference is this chapter's
/// whole invariant: the live type holds `Buffer`, `Texture`, and `AcquiredFrameId`
/// — a live handle in two of the three arms — and this one holds [`ObjectId`]s.
/// Section 56 states the rule directly: a tooling-owned value may not retain a
/// live `Buffer` or `Texture` handle.
///
/// The stage and access fields reuse the RHI command vocabulary
/// ([`PipelineScope`], [`AccessMask`]) rather than creating tooling copies.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedResourceUse {
    /// A buffer was used.
    Buffer {
        /// The buffer's identity.
        buffer: ObjectId,
        /// The byte range used.
        range: BufferRange,
        /// The stages that used it.
        stages: PipelineScope,
        /// How they used it.
        access: AccessMask,
    },

    /// A texture was used.
    Texture {
        /// The texture's identity.
        texture: ObjectId,
        /// Which subresources were used.
        subresources: TextureSubresourceRange,
        /// The stages that used it.
        stages: PipelineScope,
        /// How they used it.
        access: AccessMask,
        /// What the use meant — attachment, sampled read, copy source, present.
        ///
        /// Carried in addition to `access` because the two answer different
        /// questions: `access` is a hazard class, and the intent is the role the
        /// use played. A snapshot decision needs the role and cannot recover it
        /// from the hazard class alone.
        intent: TextureUseIntent,
    },

    /// An acquired frame was used.
    Frame {
        /// The frame's identity.
        frame: AcquiredFrameId,
        /// The stages that used it.
        stages: PipelineScope,
        /// How they used it.
        access: AccessMask,
    },

    /// An acceleration structure. It has identity but no subrange vocabulary.
    AccelerationStructure {
        structure: ObjectId,
        stages: PipelineScope,
        access: AccessMask,
    },

    /// Query-set slots. Query sets have no buffer/image backing in the portable
    /// model, but this use preserves their scheduling dependency in a capture.
    Query {
        set: ObjectId,
        first_query: u32,
        query_count: u32,
        stages: PipelineScope,
        access: QueryAccess,
    },
}
