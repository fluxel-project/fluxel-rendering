//! Metal command submission for the single-queue v13 baseline.
//!
//! Metal has no fence object which maps naturally to a Fluxel completion point.
//! A retained command buffer and its completion handler are that primitive here.
//! The handler is installed before `commit`, so every accepted batch eventually
//! wakes its completion futures even if the application does not call `poll`.
//!
//! Submission is deliberately two phase.  We allocate and encode *all* command
//! buffers first; only then are they committed in plan order.  Consequently an
//! encoding error is still an honest `submit` error, rather than an ambiguous
//! partially accepted prefix.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::task::Waker;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLCullMode, MTLDevice,
    MTLIndexType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLResourceOptions, MTLScissorRect, MTLSize, MTLStoreAction,
    MTLTriangleFillMode, MTLViewport, MTLWinding,
};

use crate::api::binding::BindingResource;
use crate::api::command::ResourceUse;
use crate::api::command::record::{CopyRecord, RecordedPayload};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::{DeviceLossInfo, DeviceStatus};
use crate::api::resource::TextureDimension;
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackStatus, ReadbackTexelLayout, ReadbackTicket, UploadDescriptor,
};
use crate::api::submission::backend::{SubmissionOutcome, SubmissionRequest};
use crate::api::submission::{CompletionFailure, CompletionState};

use super::device::MetalShared;
use super::resource::{MetalBuffer, MetalTexture};

/// The mutable execution frontier.  This belongs to the command spine, not to
/// an individual command buffer: all command-buffer callbacks race through this
/// one authority and a terminal loss therefore wakes every pending receipt.
/// Shared completion authority also consulted by host-visible buffer mapping.
/// It is backend-private; only completion serials cross the resource seam.
pub(super) struct SpineState {
    issued: u64,
    completed: u64,
    finished: BTreeSet<u64>,
    failed: Option<(u64, CompletionFailure)>,
    lost: Option<DeviceLossInfo>,
    waiters: BTreeMap<u64, Vec<Waker>>,
    pending_readbacks: BTreeMap<u64, Vec<ReadbackTicket>>,
}

impl SpineState {
    fn new() -> Self {
        Self {
            issued: 0,
            completed: 0,
            finished: BTreeSet::new(),
            failed: None,
            lost: None,
            waiters: BTreeMap::new(),
            pending_readbacks: BTreeMap::new(),
        }
    }

    fn wake_through(&mut self, serial: u64) -> Vec<Waker> {
        let keys = self
            .waiters
            .range(..=serial)
            .map(|(&point, _)| point)
            .collect::<Vec<_>>();
        let mut out = Vec::new();
        for point in keys {
            if let Some(mut waiters) = self.waiters.remove(&point) {
                out.append(&mut waiters);
            }
        }
        out
    }

    fn wake_all(&mut self) -> Vec<Waker> {
        let mut out = Vec::new();
        for (_, mut waiters) in std::mem::take(&mut self.waiters) {
            out.append(&mut waiters);
        }
        out
    }
}

/// The backend-private execution domain used by `MetalDevice`.
pub(super) struct MetalCommandSpine {
    shared: Arc<MetalShared>,
    state: Arc<Mutex<SpineState>>,
    presentation_loss: Arc<super::presentation::MetalPresentationLoss>,
}

/// Native staging retained until the command buffer's completion handler has
/// copied it into the ticket. The public ticket owns the resulting bytes, not a
/// Metal mapping lease, so it remains valid after this native allocation drops.
struct MetalPendingReadback {
    ticket: ReadbackTicket,
    staging: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    byte_len: usize,
    layout: Option<ReadbackTexelLayout>,
}

/// Phase-A output: completion retention moves with the command buffer and is
/// never released merely because all batches finished recording.
struct EncodedBatch {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    readbacks: Vec<MetalPendingReadback>,
}

impl MetalCommandSpine {
    pub(super) fn new(
        shared: Arc<MetalShared>,
        presentation_loss: Arc<super::presentation::MetalPresentationLoss>,
    ) -> RhiResult<Self> {
        Ok(Self {
            shared,
            state: Arc::new(Mutex::new(SpineState::new())),
            presentation_loss,
        })
    }

    pub(super) fn poll(&self) -> RhiResult<()> {
        // Completion is callback driven.  Keeping this method intentionally
        // non-blocking is important for hosts whose platform pump owns Metal.
        Ok(())
    }

    pub(super) fn status(&self) -> DeviceStatus {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.lost.is_some() {
            DeviceStatus::Lost
        } else {
            DeviceStatus::Active
        }
    }

    pub(super) fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .lost
            .clone()
    }

    pub(super) fn wait_idle(&self) -> RhiResult<()> {
        let command_buffer = self.shared.queue.commandBuffer().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal failed to allocate idle command buffer",
            )
            .at("MetalCommandSpine::wait_idle")
        })?;
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            self.record_terminal_failure("Metal wait-idle command buffer failed");
            return Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "Metal device was lost while waiting idle",
            )
            .at("MetalCommandSpine::wait_idle"));
        }
        Ok(())
    }

    pub(super) fn completion(&self, serial: u64) -> CompletionState {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        completion_for(&state, serial)
    }

    pub(super) fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> CompletionState {
        completion_or_register_waker(&self.state, serial, waker)
    }

    pub(super) fn mapping_state(&self) -> Arc<Mutex<SpineState>> {
        Arc::clone(&self.state)
    }

    /// Records every batch before committing any of them.
    pub(super) fn submit(&self, request: &SubmissionRequest<'_>) -> RhiResult<SubmissionOutcome> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(info) = &state.lost {
            return Err(
                RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned())
                    .at("MetalCommandSpine::submit"),
            );
        }
        if request.batches.is_empty() {
            return Ok(SubmissionOutcome {
                completion: state.issued,
                points: Vec::new(),
            });
        }

        // Phase A: a local vector owns every native command buffer.  Dropping it
        // after an error releases uncommitted buffers without feeding Metal work.
        let mut encoded = Vec::with_capacity(request.batches.len());
        for batch in request.batches {
            encoded.push(self.encode_batch(batch)?);
        }
        // A present belongs to one plan point, hence to exactly one command
        // buffer in this single-queue baseline. Validate every association while
        // all buffers are still uncommitted: a bad plan must preserve v13's
        // `submit Err => no native work accepted` rule.
        for present in request.presents {
            if !request
                .batches
                .iter()
                .any(|batch| batch.point == present.after)
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal present refers to a plan point absent from this submission",
                )
                .at("MetalCommandSpine::submit"));
            }
            super::presentation::frame_attachment(&present.attachment)?;
        }
        let first = state.issued.checked_add(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal completion serial space is exhausted",
            )
        })?;
        let last = first.checked_add(encoded.len() as u64 - 1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal completion serial space is exhausted",
            )
        })?;

        // Phase B: handlers are installed before commit and the serial frontier
        // becomes visible before the first command buffer is handed to Metal.
        // `presentDrawable:` is deliberately issued before `commit`, rather
        // than using the attachment's direct-present fallback. This preserves
        // the required ordering between rendering the drawable and display.
        for present in request.presents {
            let index = request
                .batches
                .iter()
                .position(|batch| batch.point == present.after)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal present refers to a plan point absent from this submission",
                    )
                    .at("MetalCommandSpine::submit")
                })?;
            super::presentation::frame_attachment(&present.attachment)?
                .schedule_present(&encoded[index].command_buffer)?;
        }

        state.issued = last;
        for (index, encoded) in encoded.into_iter().enumerate() {
            let serial = first + index as u64;
            // Map requests consult this exact accepted serial.  It is recorded
            // only after Phase A succeeded for the full plan and directly
            // before the command buffer becomes native work, so a failed
            // submit never makes host mapping wait on imaginary GPU use.
            mark_batch_buffers_accepted(&request.batches[index], serial);
            let tickets = encoded
                .readbacks
                .iter()
                .map(|entry| entry.ticket.clone())
                .collect();
            state.pending_readbacks.insert(serial, tickets);
            install_completion_handler(
                &encoded.command_buffer,
                Arc::clone(&self.state),
                Arc::clone(&self.presentation_loss),
                serial,
                encoded.readbacks,
            );
            encoded.command_buffer.commit();
        }
        Ok(SubmissionOutcome {
            completion: last,
            points: request
                .batches
                .iter()
                .enumerate()
                .map(|(index, batch)| (batch.point, first + index as u64))
                .collect(),
        })
    }

    fn encode_batch(
        &self,
        batch: &crate::api::submission::plan::PlanBatch,
    ) -> RhiResult<EncodedBatch> {
        let command_buffer = self.shared.queue.commandBuffer().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal failed to allocate command buffer",
            )
            .at("MetalCommandSpine::encode_batch")
        })?;
        let mut blit: Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>> = None;
        let mut compute: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> = None;
        let mut render: Option<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> = None;
        let mut raster_extent: Option<(u32, u32)> = None;
        let mut readbacks = Vec::new();
        for work in &batch.work {
            for command in work.commands() {
                match &command.payload {
                    // A portable debug-group may span encoder boundaries.
                    // Metal groups cannot: ending an encoder implicitly ends its
                    // native group stack. Keep groups as recording diagnostics
                    // and lower only point markers, which cannot underflow or
                    // leak across a blit/compute/render transition.
                    RecordedPayload::DebugPush(_) | RecordedPayload::DebugPop => {}
                    RecordedPayload::DebugMarker(label) => {
                        let text = NSString::from_str(label.as_deref().unwrap_or("<debug-marker>"));
                        if let Some(encoder) = render.as_deref() {
                            encoder.insertDebugSignpost(&text);
                        } else if let Some(encoder) = compute.as_deref() {
                            encoder.insertDebugSignpost(&text);
                        } else if let Some(encoder) = blit.as_deref() {
                            encoder.insertDebugSignpost(&text);
                        }
                    }
                    RecordedPayload::RasterBegin(begin) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error(
                                "raster scope",
                                if compute.is_some() {
                                    "compute"
                                } else {
                                    "raster"
                                },
                            ));
                        }
                        end_blit(&mut blit);
                        let pass = render_pass_descriptor(begin)?;
                        raster_extent = Some(raster_scope_extent(begin)?);
                        render = Some(
                            command_buffer
                                .renderCommandEncoderWithDescriptor(&pass)
                                .ok_or_else(|| {
                                    RhiError::new(
                                        RhiErrorKind::BackendFailure,
                                        "Metal failed to create a render command encoder",
                                    )
                                    .at("MetalCommandSpine::encode_batch")
                                })?,
                        );
                    }
                    RecordedPayload::RasterDraw(draw) => {
                        let encoder = render.as_deref().ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal raster draw was recorded outside a raster scope",
                            )
                            .at("MetalCommandSpine::encode_batch")
                        })?;
                        let pipeline = draw
                            .pipeline
                            .native()
                            .as_any()
                            .downcast_ref::<super::pipeline::MetalRasterPipeline>()
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::WrongDevice,
                                    "raster pipeline is not backed by this Metal device",
                                )
                                .at("MetalCommandSpine::encode_batch")
                            })?;
                        encoder.setRenderPipelineState(pipeline.state());
                        if let Some(depth_stencil) = pipeline.depth_stencil() {
                            encoder.setDepthStencilState(Some(depth_stencil));
                        }
                        encoder.setStencilReferenceValue(draw.stencil_reference);
                        let primitive = &draw.pipeline.descriptor().primitive;
                        encoder.setCullMode(metal_cull_mode(primitive.cull_mode));
                        encoder.setFrontFacingWinding(metal_winding(primitive.front_face));
                        encoder.setTriangleFillMode(metal_fill_mode(primitive.polygon_mode)?);
                        encoder.setBlendColorRed_green_blue_alpha(
                            draw.blend_constant.r,
                            draw.blend_constant.g,
                            draw.blend_constant.b,
                            draw.blend_constant.a,
                        );
                        let extent = raster_extent.ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal raster draw has no attachment extent",
                            )
                        })?;
                        encoder.setViewport(metal_viewport(draw.viewport.unwrap_or(
                            crate::api::command::Viewport::new(
                                0.0,
                                0.0,
                                extent.0 as f32,
                                extent.1 as f32,
                                0.0,
                                1.0,
                            ),
                        )));
                        encoder.setScissorRect(metal_scissor(
                            draw.scissor.unwrap_or(crate::api::command::Rect::new(
                                0, 0, extent.0, extent.1,
                            )),
                        ));
                        bind_vertex_buffers(encoder, &draw.vertex_buffers)?;
                        bind_raster_groups(encoder, pipeline.binding_abi(), &draw.groups)?;
                        bind_raster_immediates(encoder, pipeline.binding_abi(), &draw.immediates)?;
                        let topology = metal_primitive(primitive.topology);
                        let instances = draw
                            .instances
                            .end
                            .checked_sub(draw.instances.start)
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "Metal raster instance range underflows",
                                )
                            })?;
                        let count =
                            draw.range
                                .end
                                .checked_sub(draw.range.start)
                                .ok_or_else(|| {
                                    RhiError::new(
                                        RhiErrorKind::InvalidUsage,
                                        "Metal raster vertex range underflows",
                                    )
                                })?;
                        if let Some(index) = &draw.index {
                            let buffer = metal_buffer(&index.binding.buffer)?;
                            let index_offset = index
                                .binding
                                .range
                                .offset
                                .checked_add(
                                    u64::from(draw.range.start)
                                        .checked_mul(index_element_size(index.format))
                                        .ok_or_else(|| {
                                            RhiError::new(
                                                RhiErrorKind::InvalidUsage,
                                                "Metal indexed-draw first-index offset overflow",
                                            )
                                        })?,
                                )
                                .ok_or_else(|| {
                                    RhiError::new(
                                        RhiErrorKind::InvalidUsage,
                                        "Metal indexed-draw buffer offset overflow",
                                    )
                                })?;
                            unsafe {
                                encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                                    topology, count as usize, metal_index_type(index.format), &buffer.raw,
                                index_offset as usize, instances as usize, draw.base_vertex as isize, draw.instances.start as usize,
                            );
                            }
                        } else {
                            unsafe {
                                encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                                topology, draw.range.start as usize, count as usize, instances as usize, draw.instances.start as usize,
                            );
                            }
                        }
                    }
                    RecordedPayload::RasterEnd => {
                        let encoder = render.take().ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal received a raster-scope end without a matching begin",
                            )
                            .at("MetalCommandSpine::encode_batch")
                        })?;
                        encoder.endEncoding();
                        raster_extent = None;
                    }
                    RecordedPayload::ComputeBegin(_) => {
                        end_blit(&mut blit);
                        if compute.is_some() || render.is_some() {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal received a nested compute scope",
                            )
                            .at("MetalCommandSpine::encode_batch"));
                        }
                        compute =
                            Some(command_buffer.computeCommandEncoder().ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::BackendFailure,
                                    "Metal failed to create a compute command encoder",
                                )
                                .at("MetalCommandSpine::encode_batch")
                            })?);
                    }
                    RecordedPayload::ComputeDispatch(dispatch) => {
                        let encoder = compute.as_deref().ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal compute dispatch was recorded outside a compute scope",
                            )
                            .at("MetalCommandSpine::encode_batch")
                        })?;
                        let pipeline = dispatch
                            .pipeline
                            .native()
                            .as_any()
                            .downcast_ref::<super::pipeline::MetalComputePipeline>()
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::WrongDevice,
                                    "compute pipeline is not backed by this Metal device",
                                )
                                .at("MetalCommandSpine::encode_batch")
                            })?;
                        encoder.setComputePipelineState(pipeline.state());
                        bind_compute_groups(encoder, pipeline.binding_abi(), &dispatch.groups)?;
                        bind_compute_immediates(
                            encoder,
                            pipeline.binding_abi(),
                            &dispatch.immediates,
                        )?;
                        let local = pipeline.workgroup_size();
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: dispatch.workgroups.0 as usize,
                                height: dispatch.workgroups.1 as usize,
                                depth: dispatch.workgroups.2 as usize,
                            },
                            MTLSize {
                                width: local.x as usize,
                                height: local.y as usize,
                                depth: local.z as usize,
                            },
                        );
                    }
                    RecordedPayload::ComputeEnd => {
                        let encoder = compute.take().ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal received a compute-scope end without a matching begin",
                            )
                            .at("MetalCommandSpine::encode_batch")
                        })?;
                        encoder.endEncoding();
                    }
                    RecordedPayload::Copy(CopyRecord::Buffer(copy)) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("copy", "compute"));
                        }
                        let encoder = ensure_blit(&command_buffer, &mut blit)?;
                        let source = copy
                            .src
                            .native()
                            .as_any()
                            .downcast_ref::<MetalBuffer>()
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::WrongDevice,
                                    "buffer is not backed by this Metal device",
                                )
                            })?;
                        let destination = copy
                            .dst
                            .native()
                            .as_any()
                            .downcast_ref::<MetalBuffer>()
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::WrongDevice,
                                    "buffer is not backed by this Metal device",
                                )
                            })?;
                        unsafe {
                            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                &source.raw,
                                copy.src_offset as usize,
                                &destination.raw,
                                copy.dst_offset as usize,
                                copy.size as usize,
                            );
                        }
                    }
                    RecordedPayload::Copy(CopyRecord::ClearBuffer { buffer, range }) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("copy", "compute"));
                        }
                        let encoder = ensure_blit(&command_buffer, &mut blit)?;
                        let native = metal_buffer(buffer)?;
                        encoder.fillBuffer_range_value(
                            &native.raw,
                            NSRange {
                                location: range.offset as usize,
                                length: range.size as usize,
                            },
                            0,
                        );
                    }
                    RecordedPayload::Copy(CopyRecord::Texture(copy)) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("copy", "compute"));
                        }
                        let encoder = ensure_blit(&command_buffer, &mut blit)?;
                        let source = metal_texture(&copy.src)?;
                        let destination = metal_texture(&copy.dst)?;
                        for layer in 0..copy.src_subresource.layer_count {
                            unsafe {
                                encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                                    &source.raw,
                                    (copy.src_subresource.base_layer + layer) as usize,
                                    copy.src_subresource.mip_level as usize,
                                    origin(copy.src_origin), size(copy.extent), &destination.raw,
                                    (copy.dst_subresource.base_layer + layer) as usize,
                                    copy.dst_subresource.mip_level as usize, origin(copy.dst_origin),
                                );
                            }
                        }
                    }
                    RecordedPayload::Copy(CopyRecord::BufferToTexture(copy)) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("copy", "compute"));
                        }
                        let encoder = ensure_blit(&command_buffer, &mut blit)?;
                        let source = metal_buffer(&copy.buffer)?;
                        let destination = metal_texture(&copy.texture)?;
                        let image_stride = u64::from(copy.bytes_per_row)
                            .checked_mul(u64::from(copy.rows_per_image))
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "Metal buffer-to-texture image stride overflow",
                                )
                            })?;
                        for layer in 0..copy.texture_subresource.layer_count {
                            unsafe {
                                encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                                    &source.raw, copy.buffer_offset.checked_add(image_stride.checked_mul(u64::from(layer)).ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal buffer-to-texture layer offset overflow"))?).ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal buffer-to-texture source offset overflow"))? as usize, copy.bytes_per_row as usize,
                                    image_stride as usize,
                                    size(copy.extent), &destination.raw,
                                    (copy.texture_subresource.base_layer + layer) as usize,
                                    copy.texture_subresource.mip_level as usize, origin(copy.texture_origin),
                                );
                            }
                        }
                    }
                    RecordedPayload::Copy(CopyRecord::TextureToBuffer(copy)) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("copy", "compute"));
                        }
                        let encoder = ensure_blit(&command_buffer, &mut blit)?;
                        let source = metal_texture(&copy.texture)?;
                        let destination = metal_buffer(&copy.buffer)?;
                        let image_stride = u64::from(copy.bytes_per_row)
                            .checked_mul(u64::from(copy.rows_per_image))
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "Metal texture-to-buffer image stride overflow",
                                )
                            })?;
                        for layer in 0..copy.texture_subresource.layer_count {
                            unsafe {
                                encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                                    &source.raw, (copy.texture_subresource.base_layer + layer) as usize,
                                    copy.texture_subresource.mip_level as usize, origin(copy.texture_origin), size(copy.extent),
                                    &destination.raw, copy.buffer_offset.checked_add(image_stride.checked_mul(u64::from(layer)).ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal texture-to-buffer layer offset overflow"))?).ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal texture-to-buffer destination offset overflow"))? as usize, copy.bytes_per_row as usize,
                                    image_stride as usize,
                                );
                            }
                        }
                    }
                    RecordedPayload::Upload(job) => {
                        if compute.is_some() {
                            return Err(scope_switch_error("upload", "compute"));
                        }
                        match job.descriptor() {
                            UploadDescriptor::Buffer(upload) => {
                                let destination = metal_buffer(&upload.dst)?;
                                let source = unsafe { self.shared.device.newBufferWithBytes_length_options(
                                std::ptr::NonNull::new(upload.bytes.as_ptr() as *mut core::ffi::c_void)
                                    .ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal upload cannot encode an empty byte payload"))?,
                                upload.bytes.len(), objc2_metal::MTLResourceOptions::StorageModeShared,
                            ) }.ok_or_else(|| RhiError::new(RhiErrorKind::OutOfMemory, "Metal upload staging allocation failed"))?;
                                let encoder = ensure_blit(&command_buffer, &mut blit)?;
                                unsafe {
                                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                &source, 0, &destination.raw, upload.dst_offset as usize, upload.bytes.len(),
                            );
                                }
                            }
                            UploadDescriptor::Texture(upload) => {
                                let destination = metal_texture(&upload.dst)?;
                                let source = unsafe { self.shared.device.newBufferWithBytes_length_options(
                                std::ptr::NonNull::new(upload.bytes.as_ptr() as *mut core::ffi::c_void)
                                    .ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal upload cannot encode an empty byte payload"))?,
                                upload.bytes.len(), objc2_metal::MTLResourceOptions::StorageModeShared,
                            ) }.ok_or_else(|| RhiError::new(RhiErrorKind::OutOfMemory, "Metal upload staging allocation failed"))?;
                                let encoder = ensure_blit(&command_buffer, &mut blit)?;
                                let image_stride = u64::from(upload.source_layout.bytes_per_row)
                                    .checked_mul(u64::from(upload.source_layout.rows_per_image))
                                    .ok_or_else(|| {
                                        RhiError::new(
                                            RhiErrorKind::InvalidUsage,
                                            "Metal texture-upload image stride overflow",
                                        )
                                    })?;
                                for layer in 0..upload.subresource.layer_count {
                                    unsafe {
                                        encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                                    &source, image_stride.checked_mul(u64::from(layer)).ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal texture-upload layer offset overflow"))? as usize, upload.source_layout.bytes_per_row as usize,
                                    image_stride as usize,
                                    size(upload.extent), &destination.raw,
                                    (upload.subresource.base_layer + layer) as usize, upload.subresource.mip_level as usize,
                                    origin(upload.origin),
                                );
                                    }
                                }
                            }
                        }
                    }
                    RecordedPayload::Readback(ticket) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("readback", "active GPU scope"));
                        }
                        lower_readback(
                            &self.shared.device,
                            &command_buffer,
                            &mut blit,
                            ticket,
                            &mut readbacks,
                        )?;
                    }
                    RecordedPayload::Copy(other) => {
                        return Err(unsupported_copy(other));
                    }
                    // The capabilities published by the baseline only include
                    // operations with a native lowering.  Refusing here is the
                    // defensive second line if a stale capability snapshot or a
                    // future recorder reaches this backend too early.
                    other => {
                        return Err(RhiError::new(
                            RhiErrorKind::Unsupported,
                            format!(
                                "Metal baseline has no lowering for recorded payload {}",
                                payload_name(other)
                            ),
                        )
                        .at("MetalCommandSpine::encode_batch"));
                    }
                }
            }
        }
        if let Some(encoder) = blit {
            encoder.endEncoding();
        }
        if compute.is_some() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal command batch ended with an unclosed compute scope",
            )
            .at("MetalCommandSpine::encode_batch"));
        }
        if render.is_some() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal command batch ended with an unclosed raster scope",
            )
            .at("MetalCommandSpine::encode_batch"));
        }
        Ok(EncodedBatch {
            command_buffer,
            readbacks,
        })
    }

    fn record_terminal_failure(&self, message: &str) {
        let waiters = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.lost.is_some() {
                return;
            }
            let info = DeviceLossInfo::new(message.to_owned());
            state.lost = Some(info.clone());
            for ticket in state.pending_readbacks.values().flatten() {
                ticket.set_status(ReadbackStatus::DeviceLost);
            }
            state.pending_readbacks.clear();
            self.presentation_loss.mark_lost(info);
            state.wake_all()
        };
        for waiter in waiters {
            waiter.wake();
        }
    }
}

fn end_blit(slot: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>) {
    if let Some(encoder) = slot.take() {
        encoder.endEncoding();
    }
}

fn scope_switch_error(next: &'static str, active: &'static str) -> RhiError {
    RhiError::new(
        RhiErrorKind::InvalidUsage,
        format!("Metal cannot encode {next} while a {active} scope is open"),
    )
    .at("MetalCommandSpine::encode_batch")
}

fn ensure_blit<'a>(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    slot: &'a mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
) -> RhiResult<&'a ProtocolObject<dyn MTLBlitCommandEncoder>> {
    if slot.is_none() {
        *slot = Some(command_buffer.blitCommandEncoder().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal failed to create a blit command encoder",
            )
            .at("MetalCommandSpine::encode_batch")
        })?);
    }
    slot.as_deref().ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::BackendFailure,
            "Metal did not retain the blit command encoder it created",
        )
        .at("MetalCommandSpine::encode_batch")
    })
}

fn metal_buffer(buffer: &crate::api::resource::Buffer) -> RhiResult<&MetalBuffer> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<MetalBuffer>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "buffer is not backed by this Metal device",
            )
            .at("MetalCommandSpine::encode_batch")
        })
}

fn metal_texture(texture: &crate::api::resource::Texture) -> RhiResult<&MetalTexture> {
    texture
        .native()
        .as_any()
        .downcast_ref::<MetalTexture>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "texture is not backed by this Metal device",
            )
            .at("MetalCommandSpine::encode_batch")
        })
}

fn render_pass_descriptor(
    begin: &crate::api::command::record::RasterBegin,
) -> RhiResult<Retained<MTLRenderPassDescriptor>> {
    use crate::api::command::attachment::{
        ColorAttachmentView, DepthAttachmentMode, StencilAttachmentMode,
    };
    use crate::api::command::geometry::{ColorClearValue, LoadOp, StoreOp};
    let pass = MTLRenderPassDescriptor::new();
    let colors = pass.colorAttachments();
    for (location, attachment) in &begin.colors {
        let native = unsafe { colors.objectAtIndexedSubscript(*location as usize) };
        match &attachment.view {
            ColorAttachmentView::Texture(view) => {
                let view = view
                    .native()
                    .as_any()
                    .downcast_ref::<super::resource::MetalTextureView>()
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::WrongDevice,
                            "color attachment view is not backed by Metal",
                        )
                    })?;
                native.setTexture(Some(&view.raw));
            }
            ColorAttachmentView::Frame(frame) => native.setTexture(Some(
                super::presentation::frame_attachment(frame)?.texture()?,
            )),
        }
        native.setLoadAction(match attachment.load {
            LoadOp::Load => MTLLoadAction::Load,
            LoadOp::Clear(value) => {
                native.setClearColor(match value {
                    ColorClearValue::Float(v) => MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    },
                    ColorClearValue::Sint(v) => MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    },
                    ColorClearValue::Uint(v) => MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    },
                });
                MTLLoadAction::Clear
            }
        });
        native.setStoreAction(match attachment.store {
            StoreOp::Store => MTLStoreAction::Store,
            StoreOp::Discard => MTLStoreAction::DontCare,
        });
        if let Some(resolve) = &attachment.resolve {
            let resolve = match resolve {
                ColorAttachmentView::Texture(view) => view
                    .native()
                    .as_any()
                    .downcast_ref::<super::resource::MetalTextureView>()
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::WrongDevice,
                            "resolve view is not backed by Metal",
                        )
                    })?
                    .raw
                    .clone(),
                ColorAttachmentView::Frame(_frame) => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "Metal direct resolve into a presentation frame is not implemented",
                    )
                    .at("MetalCommandSpine::encode_batch"));
                }
            };
            native.setResolveTexture(Some(&resolve));
            native.setStoreAction(match attachment.store {
                StoreOp::Store => MTLStoreAction::StoreAndMultisampleResolve,
                StoreOp::Discard => MTLStoreAction::MultisampleResolve,
            });
        }
    }
    if let Some(attachment) = &begin.depth_stencil {
        let view = attachment
            .view
            .native()
            .as_any()
            .downcast_ref::<super::resource::MetalTextureView>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "depth/stencil attachment view is not backed by Metal",
                )
            })?;
        if let Some(depth) = attachment.depth {
            let native = pass.depthAttachment();
            native.setTexture(Some(&view.raw));
            match depth {
                DepthAttachmentMode::ReadOnly => {
                    native.setLoadAction(MTLLoadAction::Load);
                    native.setStoreAction(MTLStoreAction::Store);
                }
                DepthAttachmentMode::ReadWrite { load, store } => {
                    native.setLoadAction(match load {
                        LoadOp::Load => MTLLoadAction::Load,
                        LoadOp::Clear(value) => {
                            native.setClearDepth(value as f64);
                            MTLLoadAction::Clear
                        }
                    });
                    native.setStoreAction(match store {
                        StoreOp::Store => MTLStoreAction::Store,
                        StoreOp::Discard => MTLStoreAction::DontCare,
                    });
                }
            }
        }
        if let Some(stencil) = attachment.stencil {
            let native = pass.stencilAttachment();
            native.setTexture(Some(&view.raw));
            match stencil {
                StencilAttachmentMode::ReadOnly => {
                    native.setLoadAction(MTLLoadAction::Load);
                    native.setStoreAction(MTLStoreAction::Store);
                }
                StencilAttachmentMode::ReadWrite { load, store } => {
                    native.setLoadAction(match load {
                        LoadOp::Load => MTLLoadAction::Load,
                        LoadOp::Clear(value) => {
                            native.setClearStencil(value);
                            MTLLoadAction::Clear
                        }
                    });
                    native.setStoreAction(match store {
                        StoreOp::Store => MTLStoreAction::Store,
                        StoreOp::Discard => MTLStoreAction::DontCare,
                    });
                }
            }
        }
    }
    Ok(pass)
}

fn raster_scope_extent(begin: &crate::api::command::record::RasterBegin) -> RhiResult<(u32, u32)> {
    if let Some((_, color)) = begin.colors.first() {
        let extent = color.view.extent();
        return Ok((extent.width, extent.height));
    }
    if let Some(depth_stencil) = &begin.depth_stencil {
        let extent = depth_stencil.view.extent();
        return Ok((extent.width, extent.height));
    }
    Err(RhiError::new(
        RhiErrorKind::InvalidUsage,
        "Metal raster scope has no attachment extent",
    ))
}

fn bind_vertex_buffers(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    bindings: &[(u32, crate::api::resource::BufferBinding)],
) -> RhiResult<()> {
    for (slot, binding) in bindings {
        let buffer = metal_buffer(&binding.buffer)?;
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&buffer.raw),
                binding.range.offset as usize,
                *slot as usize,
            )
        };
    }
    Ok(())
}

fn metal_primitive(value: crate::api::pipeline::PrimitiveTopology) -> MTLPrimitiveType {
    use crate::api::pipeline::PrimitiveTopology as P;
    match value {
        P::PointList => MTLPrimitiveType::Point,
        P::LineList => MTLPrimitiveType::Line,
        P::LineStrip => MTLPrimitiveType::LineStrip,
        P::TriangleList => MTLPrimitiveType::Triangle,
        P::TriangleStrip => MTLPrimitiveType::TriangleStrip,
    }
}
fn metal_index_type(value: crate::api::command::IndexFormat) -> MTLIndexType {
    match value {
        crate::api::command::IndexFormat::Uint16 => MTLIndexType::UInt16,
        crate::api::command::IndexFormat::Uint32 => MTLIndexType::UInt32,
    }
}
fn metal_cull_mode(value: crate::api::pipeline::CullMode) -> MTLCullMode {
    match value {
        crate::api::pipeline::CullMode::None => MTLCullMode::None,
        crate::api::pipeline::CullMode::Front => MTLCullMode::Front,
        crate::api::pipeline::CullMode::Back => MTLCullMode::Back,
    }
}
fn metal_winding(value: crate::api::pipeline::FrontFace) -> MTLWinding {
    match value {
        crate::api::pipeline::FrontFace::Ccw => MTLWinding::CounterClockwise,
        crate::api::pipeline::FrontFace::Cw => MTLWinding::Clockwise,
    }
}
fn metal_fill_mode(value: crate::api::pipeline::PolygonMode) -> RhiResult<MTLTriangleFillMode> {
    match value {
        crate::api::pipeline::PolygonMode::Fill => Ok(MTLTriangleFillMode::Fill),
        crate::api::pipeline::PolygonMode::Line => Ok(MTLTriangleFillMode::Lines),
        crate::api::pipeline::PolygonMode::Point => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal has no point polygon-mode lowering",
        )),
    }
}
fn metal_viewport(value: crate::api::command::Viewport) -> MTLViewport {
    MTLViewport {
        originX: value.x as f64,
        originY: value.y as f64,
        width: value.width as f64,
        height: value.height as f64,
        znear: value.min_depth as f64,
        zfar: value.max_depth as f64,
    }
}
fn metal_scissor(value: crate::api::command::Rect) -> MTLScissorRect {
    MTLScissorRect {
        x: value.x as usize,
        y: value.y as usize,
        width: value.width as usize,
        height: value.height as usize,
    }
}

/// Applies the immutable portable packets through the ABI constructed with the
/// pipeline.  The direct Metal path intentionally binds every packet at every
/// dispatch; state-diff caching is an optimization that must not weaken the
/// group/slot/dynamic-offset proof performed here.
fn bind_compute_groups(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    abi: &super::binding::MetalBindingAbi,
    groups: &[crate::api::command::record::BoundGroup],
) -> RhiResult<()> {
    for bound in groups {
        let packet = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<super::binding::MetalBindGroup>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "compute bind group is not backed by this Metal device",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
        let group_abi = abi.group(bound.index).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal compute pipeline has no ABI for a bound group",
            )
            .at("MetalCommandSpine::encode_batch")
        })?;
        let dynamic = abi.dynamic_offsets(bound.index, &bound.group, &bound.dynamic_offsets)?;
        for (slot, resource) in packet.entries() {
            let slot_abi = group_abi.slot(*slot).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal bind packet slot is absent from pipeline ABI",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
            let first = slot_abi.first.compute.ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "compute bind packet contains a non-compute-visible slot",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
            match resource {
                BindingResource::Buffer(binding) => bind_compute_buffer(
                    encoder,
                    first,
                    binding,
                    dynamic_offset(&dynamic, *slot, 0),
                )?,
                BindingResource::BufferArray(bindings) => {
                    for (element, binding) in bindings.iter().enumerate() {
                        bind_compute_buffer(
                            encoder,
                            first + element as u32,
                            binding,
                            dynamic_offset(&dynamic, *slot, element as u32),
                        )?;
                    }
                }
                BindingResource::Texture(view) => bind_compute_texture(encoder, first, view)?,
                BindingResource::TextureArray(views) => {
                    for (element, view) in views.iter().enumerate() {
                        bind_compute_texture(encoder, first + element as u32, view)?;
                    }
                }
                BindingResource::Sampler(sampler) => bind_compute_sampler(encoder, first, sampler)?,
                BindingResource::SamplerArray(samplers) => {
                    for (element, sampler) in samplers.iter().enumerate() {
                        bind_compute_sampler(encoder, first + element as u32, sampler)?;
                    }
                }
                BindingResource::AccelerationStructure(_)
                | BindingResource::AccelerationStructureArray(_)
                | BindingResource::ExternalTexture(_) => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "Metal direct binding does not implement this resource class",
                    )
                    .at("MetalCommandSpine::encode_batch"));
                }
            }
        }
    }
    Ok(())
}

fn bind_raster_groups(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    abi: &super::binding::MetalBindingAbi,
    groups: &[crate::api::command::record::BoundGroup],
) -> RhiResult<()> {
    for bound in groups {
        let packet = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<super::binding::MetalBindGroup>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "raster bind group is not backed by this Metal device",
                )
            })?;
        let group_abi = abi.group(bound.index).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal raster pipeline has no ABI for a bound group",
            )
        })?;
        let dynamic = abi.dynamic_offsets(bound.index, &bound.group, &bound.dynamic_offsets)?;
        for (slot, resource) in packet.entries() {
            let entry = group_abi.slot(*slot).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal raster binding slot is absent from ABI",
                )
            })?;
            match resource {
                BindingResource::Buffer(value) => bind_raster_buffer(
                    encoder,
                    entry.first.vertex,
                    entry.first.fragment,
                    value,
                    dynamic_offset(&dynamic, *slot, 0),
                )?,
                BindingResource::BufferArray(values) => {
                    for (i, value) in values.iter().enumerate() {
                        bind_raster_buffer(
                            encoder,
                            entry.first.vertex.map(|v| v + i as u32),
                            entry.first.fragment.map(|v| v + i as u32),
                            value,
                            dynamic_offset(&dynamic, *slot, i as u32),
                        )?
                    }
                }
                BindingResource::Texture(value) => {
                    bind_raster_texture(encoder, entry.first.vertex, entry.first.fragment, value)?
                }
                BindingResource::TextureArray(values) => {
                    for (i, value) in values.iter().enumerate() {
                        bind_raster_texture(
                            encoder,
                            entry.first.vertex.map(|v| v + i as u32),
                            entry.first.fragment.map(|v| v + i as u32),
                            value,
                        )?
                    }
                }
                BindingResource::Sampler(value) => {
                    bind_raster_sampler(encoder, entry.first.vertex, entry.first.fragment, value)?
                }
                BindingResource::SamplerArray(values) => {
                    for (i, value) in values.iter().enumerate() {
                        bind_raster_sampler(
                            encoder,
                            entry.first.vertex.map(|v| v + i as u32),
                            entry.first.fragment.map(|v| v + i as u32),
                            value,
                        )?
                    }
                }
                BindingResource::AccelerationStructure(_)
                | BindingResource::AccelerationStructureArray(_)
                | BindingResource::ExternalTexture(_) => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "Metal direct raster binding does not implement this resource class",
                    ));
                }
            }
        }
    }
    Ok(())
}
fn bind_raster_buffer(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    vertex: Option<u32>,
    fragment: Option<u32>,
    binding: &crate::api::resource::BufferBinding,
    dynamic: u64,
) -> RhiResult<()> {
    let buffer = metal_buffer(&binding.buffer)?;
    let offset = binding.range.offset.checked_add(dynamic).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal dynamic buffer offset overflow",
        )
    })? as usize;
    if let Some(index) = vertex {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&buffer.raw), offset, index as usize)
        };
    }
    if let Some(index) = fragment {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(&buffer.raw), offset, index as usize)
        };
    }
    Ok(())
}
fn bind_raster_texture(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    vertex: Option<u32>,
    fragment: Option<u32>,
    value: &crate::api::resource::TextureView,
) -> RhiResult<()> {
    let view = value
        .native()
        .as_any()
        .downcast_ref::<super::resource::MetalTextureView>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "texture view is not backed by this Metal device",
            )
        })?;
    if let Some(index) = vertex {
        unsafe { encoder.setVertexTexture_atIndex(Some(&view.raw), index as usize) };
    }
    if let Some(index) = fragment {
        unsafe { encoder.setFragmentTexture_atIndex(Some(&view.raw), index as usize) };
    }
    Ok(())
}
fn bind_raster_sampler(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    vertex: Option<u32>,
    fragment: Option<u32>,
    value: &crate::api::resource::Sampler,
) -> RhiResult<()> {
    let sampler = value
        .native()
        .as_any()
        .downcast_ref::<super::resource::MetalSampler>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "sampler is not backed by this Metal device",
            )
        })?;
    if let Some(index) = vertex {
        unsafe { encoder.setVertexSamplerState_atIndex(Some(&sampler.raw), index as usize) };
    }
    if let Some(index) = fragment {
        unsafe { encoder.setFragmentSamplerState_atIndex(Some(&sampler.raw), index as usize) };
    }
    Ok(())
}

fn immediate_bytes(
    abi: &super::binding::MetalBindingAbi,
    writes: &[crate::api::command::record::ImmediateWrite],
) -> RhiResult<Vec<u8>> {
    let immediate = abi.immediates();
    let mut bytes = vec![0; immediate.size as usize];
    for write in writes {
        let end = write
            .offset
            .checked_add(u32::try_from(write.bytes.len()).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal immediate write length exceeds u32",
                )
            })?)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal immediate write overflows",
                )
            })?;
        if end > immediate.size {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal immediate write exceeds pipeline ABI size",
            ));
        }
        bytes[write.offset as usize..end as usize].copy_from_slice(&write.bytes);
    }
    Ok(bytes)
}
fn bind_compute_immediates(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    abi: &super::binding::MetalBindingAbi,
    writes: &[crate::api::command::record::ImmediateWrite],
) -> RhiResult<()> {
    let immediate = abi.immediates();
    if immediate.size == 0 {
        return Ok(());
    }
    let bytes = immediate_bytes(abi, writes)?;
    let pointer =
        std::ptr::NonNull::new(bytes.as_ptr() as *mut core::ffi::c_void).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal immediate byte storage is null",
            )
        })?;
    if let Some(index) = immediate.indices.compute {
        unsafe { encoder.setBytes_length_atIndex(pointer, bytes.len(), index as usize) };
    }
    Ok(())
}
fn bind_raster_immediates(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    abi: &super::binding::MetalBindingAbi,
    writes: &[crate::api::command::record::ImmediateWrite],
) -> RhiResult<()> {
    let immediate = abi.immediates();
    if immediate.size == 0 {
        return Ok(());
    }
    let bytes = immediate_bytes(abi, writes)?;
    let pointer =
        std::ptr::NonNull::new(bytes.as_ptr() as *mut core::ffi::c_void).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal immediate byte storage is null",
            )
        })?;
    if let Some(index) = immediate.indices.vertex {
        unsafe { encoder.setVertexBytes_length_atIndex(pointer, bytes.len(), index as usize) };
    }
    if let Some(index) = immediate.indices.fragment {
        unsafe { encoder.setFragmentBytes_length_atIndex(pointer, bytes.len(), index as usize) };
    }
    Ok(())
}
fn index_element_size(value: crate::api::command::IndexFormat) -> u64 {
    match value {
        crate::api::command::IndexFormat::Uint16 => 2,
        crate::api::command::IndexFormat::Uint32 => 4,
    }
}

fn dynamic_offset(
    offsets: &[super::binding::MetalDynamicOffset],
    slot: crate::api::binding::BindingSlotId,
    element: u32,
) -> u64 {
    offsets
        .iter()
        .find(|offset| offset.slot == slot && offset.element == element)
        .map_or(0, |offset| offset.offset)
}

fn bind_compute_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u32,
    binding: &crate::api::resource::BufferBinding,
    dynamic: u64,
) -> RhiResult<()> {
    let buffer = metal_buffer(&binding.buffer)?;
    let offset = binding.range.offset.checked_add(dynamic).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal dynamic buffer offset overflow",
        )
    })?;
    unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer.raw), offset as usize, index as usize) };
    Ok(())
}

fn bind_compute_texture(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u32,
    view: &crate::api::resource::TextureView,
) -> RhiResult<()> {
    let view = view
        .native()
        .as_any()
        .downcast_ref::<super::resource::MetalTextureView>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "texture view is not backed by this Metal device",
            )
        })?;
    unsafe { encoder.setTexture_atIndex(Some(&view.raw), index as usize) };
    Ok(())
}

fn bind_compute_sampler(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u32,
    sampler: &crate::api::resource::Sampler,
) -> RhiResult<()> {
    let sampler = sampler
        .native()
        .as_any()
        .downcast_ref::<super::resource::MetalSampler>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "sampler is not backed by this Metal device",
            )
        })?;
    unsafe { encoder.setSamplerState_atIndex(Some(&sampler.raw), index as usize) };
    Ok(())
}

fn origin(value: crate::api::resource::Origin3d) -> MTLOrigin {
    MTLOrigin {
        x: value.x as usize,
        y: value.y as usize,
        z: value.z as usize,
    }
}

fn size(value: crate::api::resource::Extent3d) -> MTLSize {
    MTLSize {
        width: value.width as usize,
        height: value.height as usize,
        depth: value.depth as usize,
    }
}

fn install_completion_handler(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    state: Arc<Mutex<SpineState>>,
    presentation_loss: Arc<super::presentation::MetalPresentationLoss>,
    serial: u64,
    readbacks: Vec<MetalPendingReadback>,
) {
    let block = RcBlock::new(
        move |completed: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let command_buffer = unsafe { completed.as_ref() };
            let mut state = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let completed_ok = command_buffer.status() == MTLCommandBufferStatus::Completed;
            let waiters = if completed_ok {
                state.pending_readbacks.remove(&serial);
                state.finished.insert(serial);
                loop {
                    let next = state.completed + 1;
                    if !state.finished.remove(&next) {
                        break;
                    }
                    state.completed += 1;
                }
                let completed = state.completed;
                state.wake_through(completed)
            } else {
                let message = "Metal command buffer terminated with an execution error";
                // Metal gives a failed command buffer after `commit` no safe
                // recovery contract for this DeviceIdentity. Treat it as the
                // execution-domain terminal state rather than allowing later
                // mapping/acquire/present futures to remain Pending.
                let info = DeviceLossInfo::new(message.to_owned());
                state.lost.get_or_insert_with(|| info.clone());
                presentation_loss.mark_lost(info);
                for ticket in state.pending_readbacks.values().flatten() {
                    ticket.set_status(ReadbackStatus::DeviceLost);
                }
                state.pending_readbacks.clear();
                state.wake_all()
            };
            drop(state);
            if completed_ok {
                for readback in &readbacks {
                    // Shared staging has no separate map/unmap lease. The
                    // command-buffer completion establishes GPU visibility;
                    // only now may CPU bytes be copied into the public ticket.
                    let pointer = readback.staging.contents().cast::<u8>();
                    let Some(pointer) = std::ptr::NonNull::new(pointer.as_ptr()) else {
                        readback.ticket.set_status(ReadbackStatus::Failed);
                        continue;
                    };
                    let bytes = unsafe {
                        std::slice::from_raw_parts(pointer.as_ptr(), readback.byte_len).to_vec()
                    };
                    readback.ticket.publish(bytes, readback.layout);
                }
            } else {
                // A post-commit Metal execution error terminally invalidates
                // this DeviceIdentity. Do not publish stale staging contents.
                for readback in &readbacks {
                    readback.ticket.set_status(ReadbackStatus::DeviceLost);
                }
            }
            for waiter in waiters {
                waiter.wake();
            }
        },
    );
    unsafe {
        command_buffer.addCompletedHandler(RcBlock::as_ptr(&block));
    }
}

fn completion_for(state: &SpineState, serial: u64) -> CompletionState {
    if serial <= state.completed {
        return CompletionState::Complete;
    }
    if let Some(info) = &state.lost {
        return CompletionState::DeviceLost(info.clone());
    }
    if let Some((first, failure)) = &state.failed {
        if serial >= *first {
            return CompletionState::Failed(failure.clone());
        }
    }
    CompletionState::Pending
}

/// The mapping seam uses the same completion frontier as ordinary completion
/// futures.  Registering while the mutex is held closes the completion-before-
/// waker race; exposing a host-visible `contents()` pointer sooner would allow
/// CPU access concurrent with accepted GPU work.
pub(super) fn completion_or_register_waker(
    state: &Arc<Mutex<SpineState>>,
    serial: u64,
    waker: &Waker,
) -> CompletionState {
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let answer = completion_for(&state, serial);
    if matches!(answer, CompletionState::Pending) {
        let waiters = state.waiters.entry(serial).or_default();
        if !waiters.iter().any(|known| known.will_wake(waker)) {
            waiters.push(waker.clone());
        }
    }
    answer
}

fn mark_batch_buffers_accepted(batch: &crate::api::submission::plan::PlanBatch, serial: u64) {
    for use_record in batch
        .work
        .iter()
        .flat_map(crate::api::command::RecordedWork::resource_uses)
    {
        if let ResourceUse::Buffer(buffer) = use_record {
            if let Some(native) = buffer
                .buffer
                .native()
                .as_any()
                .downcast_ref::<MetalBuffer>()
            {
                native.mark_accepted(serial);
            }
        }
    }
}

/// Encodes a CPU-visible staging copy but deliberately does not publish it.
/// Metal can execute the blit asynchronously after this function returns; the
/// completion handler is the sole publisher so `ReadbackStatus::Ready` always
/// implies bytes from completed GPU work.
fn lower_readback(
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    blit: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    ticket: &ReadbackTicket,
    retained: &mut Vec<MetalPendingReadback>,
) -> RhiResult<()> {
    match ticket.request() {
        ReadbackRequest::Buffer { src, range, .. } => {
            let source = metal_buffer(src)?;
            let byte_len = usize::try_from(range.size).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal buffer readback exceeds host address space",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
            let staging = device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::OutOfMemory,
                        "Metal buffer-readback staging allocation failed",
                    )
                    .at("MetalCommandSpine::encode_batch")
                })?;
            let encoder = ensure_blit(command_buffer, blit)?;
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &source.raw,
                    range.offset as usize,
                    &staging,
                    0,
                    byte_len,
                );
            }
            retained.push(MetalPendingReadback {
                ticket: ticket.clone(),
                staging,
                byte_len,
                layout: None,
            });
        }
        ReadbackRequest::Texture {
            src,
            subresource,
            origin: source_origin,
            extent,
            ..
        } => {
            let bytes_per_block = crate::api::format::logical_bytes_per_block(
                src.descriptor().format,
            )
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "Metal texture readback format has no byte-copy block size",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
            let (_, block_height) = crate::api::format::block_extent(src.descriptor().format);
            let tight_row = extent
                .width
                .div_ceil(crate::api::format::block_extent(src.descriptor().format).0)
                .checked_mul(bytes_per_block)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture readback row pitch overflows",
                    )
                    .at("MetalCommandSpine::encode_batch")
                })?;
            // The documented portable layout makes no promise of tight rows.
            // A 256-byte pitch satisfies Metal's blit-buffer alignment on the
            // platform families this backend supports and keeps completion data
            // directly usable without a second CPU repack.
            let bytes_per_row = align_readback_row(u64::from(tight_row))?;
            let rows_per_image = extent.height.div_ceil(block_height);
            // Array layers occupy distinct Metal source slices, while a 3D
            // copy uses one slice and makes its Z extent part of the image
            // footprint. Both layouts expose every image/depth slice through
            // the same portable rows_per_image stride.
            let is_3d = src.descriptor().dimension == TextureDimension::D3;
            let images = if is_3d {
                u64::from(extent.depth)
            } else {
                u64::from(subresource.layer_count)
            };
            let image_stride = bytes_per_row
                .checked_mul(u64::from(rows_per_image))
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture readback image stride overflows",
                    )
                    .at("MetalCommandSpine::encode_batch")
                })?;
            let total_size = image_stride.checked_mul(images).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal texture readback staging size overflows",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
            let byte_len = usize::try_from(total_size).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal texture readback exceeds host address space",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
            let staging = device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::OutOfMemory,
                        "Metal texture-readback staging allocation failed",
                    )
                    .at("MetalCommandSpine::encode_batch")
                })?;
            let texture = metal_texture(src)?;
            let encoder = ensure_blit(command_buffer, blit)?;
            let copies = if is_3d { 1 } else { subresource.layer_count };
            for layer in 0..copies {
                let offset = image_stride.checked_mul(u64::from(layer)).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture readback layer offset overflows",
                    )
                    .at("MetalCommandSpine::encode_batch")
                })?;
                unsafe {
                    encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        &texture.raw,
                        if is_3d { 0 } else { (subresource.base_layer + layer) as usize },
                        subresource.mip_level as usize,
                        origin(*source_origin),
                        size(*extent),
                        &staging,
                        offset as usize,
                        bytes_per_row as usize,
                        image_stride as usize,
                    );
                }
            }
            retained.push(MetalPendingReadback {
                ticket: ticket.clone(),
                staging,
                byte_len,
                layout: Some(ReadbackTexelLayout {
                    bytes_per_row: u32::try_from(bytes_per_row).map_err(|_| {
                        RhiError::new(
                            RhiErrorKind::OutOfMemory,
                            "Metal texture readback row pitch exceeds public layout",
                        )
                        .at("MetalCommandSpine::encode_batch")
                    })?,
                    rows_per_image,
                    total_size,
                }),
            });
        }
    }
    Ok(())
}

fn align_readback_row(value: u64) -> RhiResult<u64> {
    value.checked_add(255).map(|row| row & !255).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal texture readback row pitch alignment overflows",
        )
        .at("MetalCommandSpine::encode_batch")
    })
}

fn unsupported_copy(copy: &CopyRecord) -> RhiError {
    let name = match copy {
        CopyRecord::ExternalImage(_) => "external-image copy",
        CopyRecord::ClearBuffer { .. } => "buffer clear",
        CopyRecord::ClearTexture { .. } => "texture clear",
        CopyRecord::Buffer(_) => "buffer copy",
        CopyRecord::BufferToTexture(_) => "buffer-to-texture copy",
        CopyRecord::TextureToBuffer(_) => "texture-to-buffer copy",
        CopyRecord::Texture(_) => "texture copy",
        CopyRecord::Resolve(_) => "texture resolve",
        CopyRecord::Blit(_) => "texture blit",
    };
    RhiError::new(
        RhiErrorKind::Unsupported,
        format!("Metal baseline has no lowering for {name}"),
    )
    .at("MetalCommandSpine::encode_batch")
}

fn payload_name(payload: &RecordedPayload) -> &'static str {
    match payload {
        RecordedPayload::MeshDispatch(_) => "mesh dispatch",
        RecordedPayload::MeshIndirect(_) => "indirect mesh dispatch",
        RecordedPayload::RayTracingBegin(_) => "ray-tracing begin",
        RecordedPayload::RayTracingDispatch(_) => "ray dispatch",
        RecordedPayload::RayTracingEnd => "ray-tracing end",
        RecordedPayload::AccelerationStructure(_) => "acceleration-structure command",
        RecordedPayload::RasterBegin(_) => "raster begin",
        RecordedPayload::RasterDraw(_) => "raster draw",
        RecordedPayload::RasterEnd => "raster end",
        RecordedPayload::ComputeBegin(_) => "compute begin",
        RecordedPayload::ComputeDispatch(_) => "compute dispatch",
        RecordedPayload::RasterIndirect(_) => "indirect raster draw",
        RecordedPayload::ComputeIndirect(_) => "indirect compute dispatch",
        RecordedPayload::QueryBegin { .. } => "query begin",
        RecordedPayload::QueryEnd { .. } => "query end",
        RecordedPayload::TimestampWrite { .. } => "timestamp write",
        RecordedPayload::QueryResolve(_) => "query resolve",
        RecordedPayload::ComputeEnd => "compute end",
        RecordedPayload::Copy(_) => "copy",
        RecordedPayload::Upload(_) => "upload",
        RecordedPayload::Readback(_) => "readback",
        RecordedPayload::DebugPush(_) => "debug push",
        RecordedPayload::DebugPop => "debug pop",
        RecordedPayload::DebugMarker(_) => "debug marker",
    }
}
