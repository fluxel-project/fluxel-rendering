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
    MTLTriangleFillMode, MTLViewport, MTLVisibilityResultMode, MTLWinding,
};

use crate::api::binding::BindingResource;
use crate::api::command::ResourceUse;
use crate::api::command::record::{CopyRecord, RecordedPayload};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{block_extent, logical_bytes_per_block};
use crate::api::platform::{DeviceLossInfo, DeviceStatus};
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackStatus, ReadbackTexelLayout, ReadbackTicket, UploadDescriptor,
};
use crate::api::resource::{TextureAspects, TextureDimension};
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
        let mut vertex_amplification_active = false;
        let mut readbacks = Vec::new();
        let occlusion_slot_count = batch
            .work
            .iter()
            .flat_map(|work| work.commands())
            .filter(|command| matches!(command.payload, RecordedPayload::QueryBegin { .. }))
            .count();
        let visibility_scratch = if occlusion_slot_count == 0 {
            None
        } else {
            let bytes = occlusion_slot_count.checked_mul(8).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal visibility scratch size overflows",
                )
            })?;
            Some(
                self.shared
                    .device
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::OutOfMemory,
                            "Metal visibility scratch allocation failed",
                        )
                    })?,
            )
        };
        let mut next_visibility_slot = 0usize;
        let mut active_visibility = None;
        let mut pending_visibility = Vec::new();
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
                        if let Some(scratch) = visibility_scratch.as_deref() {
                            pass.setVisibilityResultBuffer(Some(scratch));
                        }
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
                        vertex_amplification_active = false;
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
                        if let Some(mappings) = pipeline.vertex_amplification_mappings() {
                            // Pipeline creation set this exact count as its
                            // maximum only after the Metal device accepted it.
                            // The pipeline owns the slice for the whole native
                            // call, satisfying Metal's raw-pointer contract.
                            unsafe {
                                encoder.setVertexAmplificationCount_viewMappings(
                                    mappings.len(),
                                    mappings.as_ptr(),
                                );
                            }
                            vertex_amplification_active = true;
                        } else if vertex_amplification_active {
                            // Amplification count is encoder state. A normal
                            // pipeline following a multiview pipeline must
                            // explicitly restore the single-view count.
                            unsafe {
                                encoder
                                    .setVertexAmplificationCount_viewMappings(1, core::ptr::null());
                            }
                            vertex_amplification_active = false;
                        }
                        if let Some(mode) = pipeline.depth_clip_mode() {
                            encoder.setDepthClipMode(mode);
                        }
                        encoder.setDepthStencilState(pipeline.depth_stencil());
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
                            if self.shared.base_vertex_instance {
                                unsafe {
                                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                                        topology, count as usize, metal_index_type(index.format), &buffer.raw,
                                        index_offset as usize, instances as usize, draw.base_vertex as isize, draw.instances.start as usize,
                                    );
                                }
                            } else {
                                if draw.base_vertex != 0 || draw.instances.start != 0 {
                                    return Err(RhiError::new(
                                        RhiErrorKind::Unsupported,
                                        "Metal base-vertex/base-instance draw selector is unavailable on this device",
                                    ));
                                }
                                unsafe {
                                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(
                                        topology, count as usize, metal_index_type(index.format), &buffer.raw,
                                        index_offset as usize, instances as usize,
                                    );
                                }
                            }
                        } else {
                            if self.shared.base_vertex_instance {
                                unsafe {
                                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                                        topology, draw.range.start as usize, count as usize, instances as usize, draw.instances.start as usize,
                                    );
                                }
                            } else {
                                if draw.instances.start != 0 {
                                    return Err(RhiError::new(
                                        RhiErrorKind::Unsupported,
                                        "Metal base-instance draw selector is unavailable on this device",
                                    ));
                                }
                                unsafe {
                                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                                        topology,
                                        draw.range.start as usize,
                                        count as usize,
                                        instances as usize,
                                    );
                                }
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
                        if active_visibility.is_some() {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal raster scope ended with an active occlusion query",
                            )
                            .at("MetalCommandSpine::encode_batch"));
                        }
                        encoder.endEncoding();
                        if !pending_visibility.is_empty() {
                            let scratch = visibility_scratch.as_deref().ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::BackendFailure,
                                    "Metal visibility copies have no scratch buffer",
                                )
                            })?;
                            let encoder = ensure_blit(&command_buffer, &mut blit)?;
                            for (set, index, scratch_offset) in pending_visibility.drain(..) {
                                let native = metal_query_set(&set)?;
                                unsafe {
                                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                        scratch,
                                        scratch_offset,
                                        &native.raw,
                                        usize::try_from(u64::from(index) * 8).map_err(|_| {
                                            RhiError::new(
                                                RhiErrorKind::InvalidUsage,
                                                "Metal query destination offset exceeds host size",
                                            )
                                        })?,
                                        8,
                                    );
                                }
                            }
                        }
                        raster_extent = None;
                        vertex_amplification_active = false;
                    }
                    RecordedPayload::QueryBegin { set, index } => {
                        let encoder = render.as_deref().ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal occlusion query began outside a raster scope",
                            )
                        })?;
                        if active_visibility.is_some() {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal received nested occlusion queries",
                            ));
                        }
                        let native = metal_query_set(set)?;
                        if *index >= native.count {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal occlusion query index exceeds its native query set",
                            ));
                        }
                        let offset = next_visibility_slot.checked_mul(8).ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal visibility query offset overflows",
                            )
                        })?;
                        next_visibility_slot =
                            next_visibility_slot.checked_add(1).ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "Metal visibility query count overflows",
                                )
                            })?;
                        encoder.setVisibilityResultMode_offset(
                            MTLVisibilityResultMode::Counting,
                            offset,
                        );
                        active_visibility = Some((set.clone(), *index, offset));
                    }
                    RecordedPayload::QueryEnd { set, index } => {
                        let encoder = render.as_deref().ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal occlusion query ended outside a raster scope",
                            )
                        })?;
                        let Some((active_set, active_index, offset)) = active_visibility.take()
                        else {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal received an occlusion-query end without a begin",
                            ));
                        };
                        if active_set.id() != set.id() || active_index != *index {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal occlusion-query end does not match its begin",
                            ));
                        }
                        encoder
                            .setVisibilityResultMode_offset(MTLVisibilityResultMode::Disabled, 0);
                        pending_visibility.push((active_set, active_index, offset));
                    }
                    RecordedPayload::QueryResolve(resolve) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("query resolve", "active GPU scope"));
                        }
                        let source = metal_query_set(&resolve.set)?;
                        if resolve
                            .first_query
                            .checked_add(resolve.query_count)
                            .is_none_or(|end| end > source.count)
                        {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal query resolve range exceeds its native query set",
                            ));
                        }
                        let destination = metal_buffer(&resolve.destination)?;
                        let source_offset = usize::try_from(u64::from(resolve.first_query) * 8)
                            .map_err(|_| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "Metal query source offset exceeds host size",
                                )
                            })?;
                        let destination_offset = usize::try_from(resolve.destination_offset)
                            .map_err(|_| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "Metal query resolve destination exceeds host size",
                                )
                            })?;
                        let bytes =
                            usize::try_from(u64::from(resolve.query_count) * 8).map_err(|_| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "Metal query resolve size exceeds host size",
                                )
                            })?;
                        let encoder = ensure_blit(&command_buffer, &mut blit)?;
                        unsafe {
                            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                &source.raw,
                                source_offset,
                                &destination.raw,
                                destination_offset,
                                bytes,
                            );
                        }
                    }
                    RecordedPayload::RasterIndirect(draw) => {
                        let encoder = render.as_deref().ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal indirect raster draw was recorded outside a raster scope",
                            )
                            .at("MetalCommandSpine::encode_batch")
                        })?;
                        if draw.count.is_some() {
                            return Err(RhiError::new(
                                RhiErrorKind::Unsupported,
                                "Metal indirect-count raster draws are not implemented",
                            )
                            .at("MetalCommandSpine::encode_batch"));
                        }
                        let pipeline = draw
                            .pipeline
                            .native()
                            .as_any()
                            .downcast_ref::<super::pipeline::MetalRasterPipeline>()
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::WrongDevice,
                                    "indirect raster pipeline is not backed by this Metal device",
                                )
                                .at("MetalCommandSpine::encode_batch")
                            })?;
                        encoder.setRenderPipelineState(pipeline.state());
                        if let Some(mappings) = pipeline.vertex_amplification_mappings() {
                            // The same pipeline can be used by direct and
                            // indirect draws; both must install the selected
                            // layers rather than inheriting a prior encoder
                            // mapping.
                            unsafe {
                                encoder.setVertexAmplificationCount_viewMappings(
                                    mappings.len(),
                                    mappings.as_ptr(),
                                );
                            }
                            vertex_amplification_active = true;
                        } else if vertex_amplification_active {
                            unsafe {
                                encoder
                                    .setVertexAmplificationCount_viewMappings(1, core::ptr::null());
                            }
                            vertex_amplification_active = false;
                        }
                        if let Some(mode) = pipeline.depth_clip_mode() {
                            encoder.setDepthClipMode(mode);
                        }
                        encoder.setDepthStencilState(pipeline.depth_stencil());
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
                                "Metal indirect raster draw has no attachment extent",
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
                        bind_raster_immediates(encoder, pipeline.binding_abi(), &[])?;
                        let arguments = metal_buffer(&draw.arguments)?;
                        let topology = metal_primitive(primitive.topology);
                        for draw_index in 0..draw.draw_count {
                            let offset = draw
                                .arguments_offset
                                .checked_add(
                                    u64::from(draw_index)
                                        .checked_mul(u64::from(draw.stride))
                                        .ok_or_else(|| {
                                            RhiError::new(
                                                RhiErrorKind::InvalidUsage,
                                                "Metal indirect draw stride offset overflows",
                                            )
                                        })?,
                                )
                                .ok_or_else(|| {
                                    RhiError::new(
                                        RhiErrorKind::InvalidUsage,
                                        "Metal indirect draw offset overflows",
                                    )
                                })?;
                            let offset = usize::try_from(offset).map_err(|_| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "Metal indirect draw offset exceeds host address space",
                                )
                            })?;
                            if let Some(index) = &draw.index {
                                let index_buffer = metal_buffer(&index.binding.buffer)?;
                                unsafe {
                                    encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(topology, metal_index_type(index.format), &index_buffer.raw, index.binding.range.offset as usize, &arguments.raw, offset);
                                }
                            } else {
                                unsafe {
                                    encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                                        topology,
                                        &arguments.raw,
                                        offset,
                                    );
                                }
                            }
                        }
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
                    RecordedPayload::ComputeIndirect(dispatch) => {
                        let encoder = compute.as_deref().ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal indirect compute dispatch was recorded outside a compute scope").at("MetalCommandSpine::encode_batch"))?;
                        let pipeline = dispatch
                            .pipeline
                            .native()
                            .as_any()
                            .downcast_ref::<super::pipeline::MetalComputePipeline>()
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::WrongDevice,
                                    "indirect compute pipeline is not backed by this Metal device",
                                )
                                .at("MetalCommandSpine::encode_batch")
                            })?;
                        let arguments = metal_buffer(&dispatch.arguments)?;
                        encoder.setComputePipelineState(pipeline.state());
                        bind_compute_groups(encoder, pipeline.binding_abi(), &dispatch.groups)?;
                        bind_compute_immediates(encoder, pipeline.binding_abi(), &[])?;
                        let local = pipeline.workgroup_size();
                        let offset = usize::try_from(dispatch.arguments_offset).map_err(|_| {
                            RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                "Metal indirect compute offset exceeds host address space",
                            )
                        })?;
                        unsafe {
                            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(&arguments.raw, offset, MTLSize { width: local.x as usize, height: local.y as usize, depth: local.z as usize });
                        }
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
                    RecordedPayload::Copy(CopyRecord::ClearTexture {
                        texture,
                        subresources,
                    }) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("texture clear", "active GPU scope"));
                        }
                        lower_clear_texture(
                            &self.shared.device,
                            &command_buffer,
                            &mut blit,
                            texture,
                            *subresources,
                        )?;
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
                    RecordedPayload::Copy(CopyRecord::Resolve(resolve)) => {
                        if compute.is_some() || render.is_some() {
                            return Err(scope_switch_error("texture resolve", "active GPU scope"));
                        }
                        // A blit encoder and a compute encoder cannot overlap.
                        // End the former before the resolver installs its own
                        // compute state, while retaining the same command-buffer
                        // ordering as surrounding copy records.
                        end_blit(&mut blit);
                        let mut cache = self
                            .shared
                            .resolve_pipeline
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if cache.is_none() {
                            *cache = Some(super::resolve::create_pipeline(&self.shared.device)?);
                        }
                        let pipeline = cache.as_ref().ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::BackendFailure,
                                "Metal did not retain standalone resolve pipeline",
                            )
                            .at("MetalCommandSpine::encode_batch")
                        })?;
                        super::resolve::encode(&command_buffer, pipeline, resolve)?;
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
                                let repacked = repack_texture_upload(upload)?;
                                let source = unsafe { self.shared.device.newBufferWithBytes_length_options(
                                std::ptr::NonNull::new(repacked.bytes.as_ptr() as *mut core::ffi::c_void)
                                    .ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal upload cannot encode an empty byte payload"))?,
                                repacked.bytes.len(), objc2_metal::MTLResourceOptions::StorageModeShared,
                            ) }.ok_or_else(|| RhiError::new(RhiErrorKind::OutOfMemory, "Metal upload staging allocation failed"))?;
                                let encoder = ensure_blit(&command_buffer, &mut blit)?;
                                for layer in 0..upload.subresource.layer_count {
                                    unsafe {
                                        encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                                    &source, repacked.bytes_per_image.checked_mul(u64::from(layer)).ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "Metal texture-upload layer offset overflow"))? as usize, repacked.bytes_per_row as usize,
                                    repacked.bytes_per_image as usize,
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

struct RepackedTextureUpload {
    bytes: Vec<u8>,
    bytes_per_row: u64,
    bytes_per_image: u64,
}

/// Converts the caller-owned host layout into Metal's buffer-copy layout.
/// Host rows are deliberately allowed to be tightly packed; imposing Metal's
/// four-byte row requirement on asset bytes would violate the upload contract.
/// Copy only logical block bytes, so caller padding never leaks into native
/// staging and compressed formats follow their block-row geometry exactly.
fn repack_texture_upload(
    upload: &crate::api::resource::transfer::TextureUploadDescriptor,
) -> RhiResult<RepackedTextureUpload> {
    let format = upload.dst.descriptor().format;
    let block_bytes = u64::from(logical_bytes_per_block(format).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal texture upload cannot determine this format's host block size",
        )
    })?);
    let (block_width, block_height) = block_extent(format);
    let logical_row = u64::from(upload.extent.width.div_ceil(block_width))
        .checked_mul(block_bytes)
        .ok_or_else(|| {
            RhiError::new(RhiErrorKind::InvalidUsage, "Metal upload row size overflow")
        })?;
    let bytes_per_row = logical_row
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal upload row alignment overflow",
            )
        })?;
    let block_rows = u64::from(upload.extent.height.div_ceil(block_height));
    let bytes_per_image = bytes_per_row.checked_mul(block_rows).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal texture-upload image stride overflow",
        )
    })?;
    let image_count = match upload.dst.descriptor().dimension {
        TextureDimension::D3 => u64::from(upload.extent.depth),
        TextureDimension::D1 | TextureDimension::D2 => u64::from(upload.subresource.layer_count),
    };
    let total = bytes_per_image.checked_mul(image_count).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "Metal texture-upload staging size overflow",
        )
    })?;
    let total = usize::try_from(total).map_err(|_| {
        RhiError::new(
            RhiErrorKind::OutOfMemory,
            "Metal texture-upload staging size does not fit the host address space",
        )
    })?;
    let mut bytes = vec![0_u8; total];
    let source_row = u64::from(upload.source_layout.bytes_per_row);
    let source_image = source_row
        .checked_mul(u64::from(upload.source_layout.rows_per_image))
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal texture-upload source image stride overflow",
            )
        })?;
    for image in 0..image_count {
        for row in 0..block_rows {
            let src = image
                .checked_mul(source_image)
                .and_then(|value| value.checked_add(row.checked_mul(source_row)?))
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture-upload source offset overflow",
                    )
                })?;
            let dst = image
                .checked_mul(bytes_per_image)
                .and_then(|value| value.checked_add(row.checked_mul(bytes_per_row)?))
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "Metal texture-upload staging offset overflow",
                    )
                })?;
            let src = usize::try_from(src).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal upload source offset is too large",
                )
            })?;
            let dst = usize::try_from(dst).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal upload staging offset is too large",
                )
            })?;
            let count = usize::try_from(logical_row).map_err(|_| {
                RhiError::new(RhiErrorKind::InvalidUsage, "Metal upload row is too large")
            })?;
            let source = upload.bytes.get(src..src + count).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "Metal texture-upload source bytes do not cover the copied row",
                )
            })?;
            bytes[dst..dst + count].copy_from_slice(source);
        }
    }
    Ok(RepackedTextureUpload {
        bytes,
        bytes_per_row,
        bytes_per_image,
    })
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

fn metal_query_set(set: &crate::api::query::QuerySet) -> RhiResult<&super::query::MetalQuerySet> {
    set.native()
        .as_any()
        .downcast_ref::<super::query::MetalQuerySet>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "query set is not backed by this Metal device",
            )
            .at("MetalCommandSpine::encode_batch")
        })
}

pub(super) fn metal_texture(texture: &crate::api::resource::Texture) -> RhiResult<&MetalTexture> {
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
    // RasterScope validation gives every main attachment one common layer
    // count. Establish the physical array length before draw-specific vertex
    // amplification mappings select layers within it.
    pass.setRenderTargetArrayLength(render_target_array_length(begin));
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

/// The recorder already validated a common attachment layer count. This mirrors
/// its primary-attachment order: color first, then depth/stencil-only scopes.
fn render_target_array_length(begin: &crate::api::command::record::RasterBegin) -> usize {
    begin
        .colors
        .first()
        .map(|(_, color)| match &color.view {
            crate::api::command::ColorAttachmentView::Texture(view) => view.layer_count(),
            crate::api::command::ColorAttachmentView::Frame(_) => 1,
        })
        .or_else(|| {
            begin
                .depth_stencil
                .as_ref()
                .map(|attachment| attachment.view.layer_count())
        })
        .unwrap_or(1) as usize
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
            // Layout packets may contain resources unused by this entry point.
            // The pipeline ABI intentionally gives those no MSL index.
            let Some(slot_abi) = group_abi.slot(*slot) else {
                continue;
            };
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
            // See compute binding above: a portable bind group may be a
            // layout superset of the compiled vertex/fragment artifacts.
            let Some(entry) = group_abi.slot(*slot) else {
                continue;
            };
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
            let completed_ok = command_buffer.status() == MTLCommandBufferStatus::Completed;
            // A completion point that owns readback is not logically Complete
            // until the CPU-visible ticket has been published. Do this before
            // advancing the shared frontier, otherwise another thread can
            // observe CompletionState::Complete while the ticket is Pending.
            let mut publication_failed = false;
            if completed_ok {
                for readback in &readbacks {
                    // Shared staging has no separate map/unmap lease. The
                    // command-buffer callback establishes GPU visibility.
                    let pointer = readback.staging.contents().cast::<u8>();
                    let Some(pointer) = std::ptr::NonNull::new(pointer.as_ptr()) else {
                        readback.ticket.set_status(ReadbackStatus::Failed);
                        publication_failed = true;
                        continue;
                    };
                    let bytes = unsafe {
                        std::slice::from_raw_parts(pointer.as_ptr(), readback.byte_len).to_vec()
                    };
                    readback.ticket.publish(bytes, readback.layout);
                }
            }

            let mut state = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let waiters = if completed_ok {
                state.pending_readbacks.remove(&serial);
                if publication_failed {
                    state.failed.get_or_insert_with(|| {
                        (
                            serial,
                            CompletionFailure::new(
                                "Metal readback staging was unavailable at completion",
                            ),
                        )
                    });
                    state.wake_all()
                } else {
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
                }
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
            if !completed_ok {
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
/// Clears whole selected subresources to Metal's byte-zero value.
///
/// Metal has no format-independent blit clear. Color subresources therefore
/// use a zero-filled shared buffer and the ordinary buffer-to-texture blit
/// route. This is deliberately not a shader fallback: it also works for the
/// block-compressed formats whose zero block is the command's documented
/// backend-defined zero value. Depth/stencil has no byte-copy route in the
/// published table, so it uses a one-attachment render pass instead. The
/// recorder requires `DEPTH_STENCIL_ATTACHMENT` for that case, which is the
/// native `RenderTarget` usage needed by Metal.
fn lower_clear_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    blit: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    texture: &crate::api::resource::Texture,
    range: crate::api::resource::TextureSubresourceRange,
) -> RhiResult<()> {
    if range.aspects == TextureAspects::COLOR {
        return lower_color_clear(device, command_buffer, blit, texture, range);
    }
    lower_depth_stencil_clear(command_buffer, blit, texture, range)
}

fn lower_color_clear(
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    blit: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    texture: &crate::api::resource::Texture,
    range: crate::api::resource::TextureSubresourceRange,
) -> RhiResult<()> {
    let native = metal_texture(texture)?;
    let descriptor = texture.descriptor();
    let bytes_per_block = crate::api::format::logical_bytes_per_block(descriptor.format)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal cannot byte-clear a color format without a fixed block layout",
            )
            .at("MetalCommandSpine::encode_batch")
        })?;
    let (block_width, block_height) = crate::api::format::block_extent(descriptor.format);
    let is_3d = descriptor.dimension == TextureDimension::D3;
    let copies_per_mip = if is_3d { 1 } else { range.layer_count };

    for mip_level in range.base_mip..range.base_mip + range.mip_count {
        let extent = crate::api::resource::texture::mip_extent(
            descriptor.extent,
            descriptor.dimension,
            mip_level,
        );
        let block_columns = extent.width.div_ceil(block_width);
        let block_rows = extent.height.div_ceil(block_height);
        let tight_row = u64::from(block_columns)
            .checked_mul(u64::from(bytes_per_block))
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal clear texture row pitch overflows",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
        // The Metal copy facts for this backend publish the native four-byte
        // row-pitch alignment. Padding is zero too, so it cannot change a
        // copied block even for the final partial block row of a mip.
        let bytes_per_row = align_clear_row(tight_row)?;
        let bytes_per_image = bytes_per_row
            .checked_mul(u64::from(block_rows))
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal clear texture image stride overflows",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
        let byte_len = bytes_per_image
            .checked_mul(if is_3d { u64::from(extent.depth) } else { 1 })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal clear texture staging allocation overflows",
                )
                .at("MetalCommandSpine::encode_batch")
            })?;
        let byte_len = usize::try_from(byte_len).map_err(|_| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "Metal clear texture staging allocation exceeds host address space",
            )
            .at("MetalCommandSpine::encode_batch")
        })?;
        let bytes_per_row = usize::try_from(bytes_per_row).map_err(|_| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "Metal clear texture row pitch exceeds host address space",
            )
            .at("MetalCommandSpine::encode_batch")
        })?;
        let bytes_per_image = usize::try_from(bytes_per_image).map_err(|_| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "Metal clear texture image stride exceeds host address space",
            )
            .at("MetalCommandSpine::encode_batch")
        })?;

        for layer in 0..copies_per_mip {
            let staging = device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::OutOfMemory,
                        "Metal clear texture staging allocation failed",
                    )
                    .at("MetalCommandSpine::encode_batch")
                })?;
            // Allocation contents are intentionally written, rather than
            // relying on a platform allocator's initialisation convention.
            unsafe { std::ptr::write_bytes(staging.contents().as_ptr(), 0, byte_len) };
            let encoder = ensure_blit(command_buffer, blit)?;
            unsafe {
                encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &staging,
                    0,
                    bytes_per_row,
                    bytes_per_image,
                    size(extent),
                    &native.raw,
                    if is_3d { 0 } else { (range.base_layer + layer) as usize },
                    mip_level as usize,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
        }
    }
    Ok(())
}

fn lower_depth_stencil_clear(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    blit: &mut Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    texture: &crate::api::resource::Texture,
    range: crate::api::resource::TextureSubresourceRange,
) -> RhiResult<()> {
    if range.aspects.contains(TextureAspects::COLOR)
        || !(range.aspects.contains(TextureAspects::DEPTH)
            || range.aspects.contains(TextureAspects::STENCIL))
    {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal clear texture cannot mix color with depth/stencil aspects",
        )
        .at("MetalCommandSpine::encode_batch"));
    }
    let native = metal_texture(texture)?;
    let descriptor = texture.descriptor();
    // A render-pass attachment is a 2D texture or 2D array slice. This
    // defensive check mirrors capability creation; it prevents a stale facts
    // snapshot from treating a 3D depth allocation as clearable.
    if descriptor.dimension != TextureDimension::D2 || descriptor.sample_count != 1 {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal depth/stencil ClearTexture requires a single-sampled 2D texture",
        )
        .at("MetalCommandSpine::encode_batch"));
    }
    let format_aspects = crate::api::format::format_aspects(descriptor.format);
    end_blit(blit);
    for mip_level in range.base_mip..range.base_mip + range.mip_count {
        for layer in range.base_layer..range.base_layer + range.layer_count {
            let pass = MTLRenderPassDescriptor::new();
            if format_aspects.contains(TextureAspects::DEPTH) {
                let attachment = pass.depthAttachment();
                attachment.setTexture(Some(&native.raw));
                attachment.setLevel(mip_level as usize);
                attachment.setSlice(layer as usize);
                if range.aspects.contains(TextureAspects::DEPTH) {
                    attachment.setClearDepth(0.0);
                    attachment.setLoadAction(MTLLoadAction::Clear);
                } else {
                    attachment.setLoadAction(MTLLoadAction::Load);
                }
                attachment.setStoreAction(MTLStoreAction::Store);
            }
            if format_aspects.contains(TextureAspects::STENCIL) {
                let attachment = pass.stencilAttachment();
                attachment.setTexture(Some(&native.raw));
                attachment.setLevel(mip_level as usize);
                attachment.setSlice(layer as usize);
                if range.aspects.contains(TextureAspects::STENCIL) {
                    attachment.setClearStencil(0);
                    attachment.setLoadAction(MTLLoadAction::Clear);
                } else {
                    attachment.setLoadAction(MTLLoadAction::Load);
                }
                attachment.setStoreAction(MTLStoreAction::Store);
            }
            let encoder = command_buffer
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "Metal failed to create a depth/stencil clear render encoder",
                    )
                    .at("MetalCommandSpine::encode_batch")
                })?;
            encoder.endEncoding();
        }
    }
    Ok(())
}

fn align_clear_row(value: u64) -> RhiResult<u64> {
    value.checked_add(3).map(|row| row & !3).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::OutOfMemory,
            "Metal clear texture row alignment overflows",
        )
        .at("MetalCommandSpine::encode_batch")
    })
}

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
