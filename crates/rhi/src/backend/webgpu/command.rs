//! Transactional WebGPU command submission.
//!
//! Browser WebGPU has no fence object.  `queue.onSubmittedWorkDone()` is its
//! completion primitive, so a submitted RHI serial owns the corresponding
//! promise in this spine.  Importantly, promise creation is *after* the whole
//! plan passed [`preflight`]: an `Err` from `submit` consequently means that no
//! command encoder, queue submit, or browser promise was created for the plan.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::task::Waker;

use js_sys::{Array, Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};

use crate::api::command::copy::{BufferCopy, BufferTextureCopy, TextureCopy};
use crate::api::command::record::{
    ComputeDispatch, ComputeIndirect, CopyRecord, RasterBegin, RasterDraw, RasterIndirect,
    RecordedPayload,
};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{block_extent, logical_bytes_per_block};
use crate::api::query::{QuerySet, QueryType};
use crate::api::resource::{ReadbackRequest, ReadbackTexelLayout};
use crate::api::submission::backend::{SubmissionOutcome, SubmissionRequest};
use crate::api::submission::{CompletionFailure, CompletionState};

use super::binding::WebGpuBindGroup;
use super::js;
use super::pipeline::{WebGpuComputePipeline, WebGpuRasterPipeline};
use super::registry::{
    self, PromisePoll, WebGpuDriver, WebGpuObjectId, WebGpuRegistration, WebGpuRequestId,
};
use super::resource::WebGpuTextureView;
use super::resource::{WebGpuBuffer, WebGpuQuerySet, WebGpuTexture};

#[derive(Clone, Debug)]
enum SerialState {
    Pending(WebGpuRequestId),
    /// GPU queue work ended but an attached readback mapping lease has not
    /// published CPU bytes yet.
    AwaitReadbacks(WebGpuRequestId),
    Complete,
    Failed(String),
}

#[derive(Default)]
struct State {
    issued: u64,
    serials: BTreeMap<u64, SerialState>,
    readbacks: Vec<PendingReadback>,
}

/// Registry-only state for one submitted map-read staging allocation.  Keeping
/// an object ID rather than a `JsValue` is what lets `WebGpuCommandSpine`
/// remain `Send + Sync` even though browser handles themselves are affine.
struct PendingReadback {
    ticket: crate::api::resource::ReadbackTicket,
    staging: WebGpuObjectId,
    bytes: u64,
    layout: Option<ReadbackTexelLayout>,
    map: Option<WebGpuRequestId>,
    /// The submitted-work promise for this plan. A plan completion cannot be
    /// published Complete until every ticket attached to it has been copied
    /// into CPU-owned bytes and published.
    completion: Option<WebGpuRequestId>,
}

/// The browser-side execution timeline for one WebGPU device registration.
pub(crate) struct WebGpuCommandSpine {
    driver: WebGpuDriver,
    state: Mutex<State>,
}

impl WebGpuCommandSpine {
    pub(crate) fn new(driver: WebGpuDriver) -> Self {
        Self {
            driver,
            state: Mutex::new(State::default()),
        }
    }

    fn registration(&self) -> WebGpuRegistration {
        self.driver.registration()
    }

    /// Does the pure, whole-plan half of submission.
    ///
    /// This deliberately does not merely check the first command: all batches
    /// are walked before any WebGPU call.  A newly added portable payload must
    /// be admitted here *and* have a Phase-B lowering below; otherwise it is a
    /// fail-closed `Unsupported`, never an accidental partial submit.
    fn preflight(&self, request: &SubmissionRequest<'_>) -> RhiResult<()> {
        if registry::device_status(self.registration())
            != Some(crate::api::platform::DeviceStatus::Active)
        {
            return Err(lost("WebGpuCommandSpine::submit"));
        }
        for present in request.presents {
            // Backend ownership is checked before Phase B; this only reads the
            // registry-owned acquired view and never creates a JS object.
            super::presentation::frame_view(&present.attachment)?;
        }
        for batch in request.batches {
            for work in &batch.work {
                for command in work.commands() {
                    match &command.payload {
                        RecordedPayload::Copy(CopyRecord::Buffer(copy)) => {
                            buffer(copy.src.native(), self.registration(), "copy source")?;
                            buffer(copy.dst.native(), self.registration(), "copy destination")?;
                        }
                        RecordedPayload::Copy(CopyRecord::BufferToTexture(copy))
                        | RecordedPayload::Copy(CopyRecord::TextureToBuffer(copy)) => {
                            buffer(
                                copy.buffer.native(),
                                self.registration(),
                                "buffer/texture copy buffer",
                            )?;
                            texture(
                                copy.texture.native(),
                                self.registration(),
                                "buffer/texture copy texture",
                            )?;
                        }
                        RecordedPayload::Copy(CopyRecord::Texture(copy)) => {
                            texture(
                                copy.src.native(),
                                self.registration(),
                                "texture copy source",
                            )?;
                            texture(
                                copy.dst.native(),
                                self.registration(),
                                "texture copy destination",
                            )?;
                        }
                        RecordedPayload::Copy(CopyRecord::ClearBuffer {
                            buffer: value, ..
                        }) => {
                            buffer(value.native(), self.registration(), "clear buffer")?;
                        }
                        // These are metadata-only on all WebGPU encoders and are
                        // therefore supported as part of the baseline.
                        RecordedPayload::DebugPush(_)
                        | RecordedPayload::DebugPop
                        | RecordedPayload::DebugMarker(_) => {}
                        // WebGPU exposes parts of the query API, but not the
                        // stronger portable recording contract. A render pass
                        // fixes one occlusionQuerySet in its descriptor, while
                        // RasterScope permits several sequentially; its
                        // timestamp descriptors denote boundaries rather than
                        // TimestampWrite's exact command position. This is an
                        // intentional fail-closed Phase-A refusal, not a
                        // missing method wrapper.
                        RecordedPayload::RasterBegin(begin) => {
                            preflight_raster_begin(begin, self.registration())?
                        }
                        RecordedPayload::QueryBegin { set, .. }
                        | RecordedPayload::QueryEnd { set, .. } => {
                            query_set(set, self.registration(), "occlusion query")?;
                            if set.descriptor().ty != QueryType::Occlusion {
                                return Err(unsupported("non-occlusion query"));
                            }
                        }
                        RecordedPayload::QueryResolve(resolve) => {
                            query_set(&resolve.set, self.registration(), "query resolve")?;
                            buffer(
                                resolve.destination.native(),
                                self.registration(),
                                "query resolve destination",
                            )?;
                            if resolve.set.descriptor().ty != QueryType::Occlusion {
                                return Err(unsupported("non-occlusion query resolve"));
                            }
                        }
                        RecordedPayload::RasterDraw(draw) => {
                            preflight_raster_draw(draw, self.registration())?
                        }
                        RecordedPayload::RasterIndirect(draw) => {
                            preflight_raster_indirect(draw, self.registration())?
                        }
                        RecordedPayload::RasterEnd
                        | RecordedPayload::ComputeBegin(_)
                        | RecordedPayload::ComputeEnd => {}
                        RecordedPayload::ComputeDispatch(dispatch) => {
                            preflight_compute_draw(dispatch, self.registration())?
                        }
                        RecordedPayload::ComputeIndirect(dispatch) => {
                            preflight_compute_indirect(dispatch, self.registration())?
                        }
                        RecordedPayload::Upload(job) => preflight_upload(job, self.registration())?,
                        RecordedPayload::Readback(ticket) => {
                            preflight_readback(ticket, self.registration())?
                        }
                        RecordedPayload::Copy(CopyRecord::ClearTexture { .. })
                        | RecordedPayload::Copy(CopyRecord::Resolve(_))
                        | RecordedPayload::Copy(CopyRecord::Blit(_))
                        | RecordedPayload::Copy(CopyRecord::ExternalImage(_))
                        | RecordedPayload::TimestampWrite { .. }
                        | RecordedPayload::MeshDispatch(_)
                        | RecordedPayload::MeshIndirect(_)
                        | RecordedPayload::RayTracingBegin(_)
                        | RecordedPayload::RayTracingDispatch(_)
                        | RecordedPayload::RayTracingEnd
                        | RecordedPayload::AccelerationStructure(_) => {
                            return Err(unsupported(payload_name(&command.payload)));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn submit(&self, request: &SubmissionRequest<'_>) -> RhiResult<SubmissionOutcome> {
        self.preflight(request)?;
        if request.batches.is_empty() {
            let issued = self.state.lock().unwrap_or_else(|p| p.into_inner()).issued;
            return Ok(SubmissionOutcome {
                completion: issued,
                points: Vec::new(),
            });
        }

        // Phase B begins here.  From this point a browser exception cannot be
        // returned as `Err`: no matter whether `queue.submit` accepted prior
        // commands before reporting an error, the RHI must expose a terminal
        // completion instead of lying that nothing happened.
        let phase_b = self.encode(request);
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let first = state.issued + 1;
        let last = first + request.batches.len() as u64 - 1;
        state.issued = last;
        match phase_b {
            Ok((promise, mut readbacks)) => {
                let request_id = registry::start_device_promise(self.registration(), promise);
                for serial in first..=last {
                    state
                        .serials
                        .insert(serial, SerialState::Pending(request_id));
                }
                for readback in &mut readbacks {
                    readback.completion = Some(request_id);
                }
                state.readbacks.extend(readbacks);
            }
            Err(message) => {
                for present in request.presents {
                    let outcome = if let Some(info) = registry::device_loss(self.registration()) {
                        crate::api::presentation::PresentState::DeviceLost(info)
                    } else {
                        crate::api::presentation::PresentState::Failed(
                            crate::api::presentation::PresentFailure::new(message.clone()),
                        )
                    };
                    present
                        .attachment
                        .terminate_present(present.receipt, outcome);
                }
                for serial in first..=last {
                    state
                        .serials
                        .insert(serial, SerialState::Failed(message.clone()));
                }
            }
        }
        Ok(SubmissionOutcome {
            completion: last,
            points: request
                .batches
                .iter()
                .enumerate()
                .map(|(n, batch)| (batch.point, first + n as u64))
                .collect(),
        })
    }

    fn encode(
        &self,
        request: &SubmissionRequest<'_>,
    ) -> Result<(Promise, Vec<PendingReadback>), String> {
        let (device, queue) = registry::with_device_handles(self.registration(), |h| {
            (h.device.clone(), h.queue.clone())
        })
        .ok_or_else(|| "WebGPU device registration was retired".to_owned())?;
        let encoder = call0(&device, "createCommandEncoder")?;
        let mut raster_pass = None;
        let mut compute_pass = None;
        let mut readbacks = Vec::new();
        for batch in request.batches {
            for work in &batch.work {
                for command in work.commands() {
                    match &command.payload {
                        RecordedPayload::RasterBegin(begin) => {
                            if raster_pass.is_some() || compute_pass.is_some() {
                                return Err(
                                    "portable command scope nesting reached WebGPU lowering".into(),
                                );
                            }
                            raster_pass = Some(begin_raster(&encoder, begin, self.registration())?);
                        }
                        RecordedPayload::QueryBegin { index, .. } => {
                            let pass = raster_pass.as_ref().ok_or_else(|| {
                                "occlusion query begin outside render pass".to_owned()
                            })?;
                            call1(pass, "beginOcclusionQuery", &num(*index as u64))?;
                        }
                        RecordedPayload::QueryEnd { .. } => {
                            let pass = raster_pass.as_ref().ok_or_else(|| {
                                "occlusion query end outside render pass".to_owned()
                            })?;
                            call0(pass, "endOcclusionQuery")?;
                        }
                        RecordedPayload::QueryResolve(resolve) => {
                            let set = object(
                                query_set(&resolve.set, self.registration(), "query resolve")
                                    .map_err(|error| error.to_string())?,
                            )
                            .ok_or_else(|| "query set was retired".to_owned())?;
                            let destination = object(
                                buffer(
                                    resolve.destination.native(),
                                    self.registration(),
                                    "query resolve destination",
                                )
                                .map_err(|error| error.to_string())?,
                            )
                            .ok_or_else(|| "query resolve destination was retired".to_owned())?;
                            call5(
                                &encoder,
                                "resolveQuerySet",
                                &set,
                                &num(resolve.first_query as u64),
                                &num(resolve.query_count as u64),
                                &destination,
                                &num(resolve.destination_offset),
                            )?;
                        }
                        RecordedPayload::RasterDraw(draw) => {
                            let pass = raster_pass
                                .as_ref()
                                .ok_or_else(|| "raster draw without render pass".to_owned())?;
                            lower_raster_draw(pass, draw, self.registration())?;
                        }
                        RecordedPayload::RasterIndirect(draw) => {
                            let pass = raster_pass
                                .as_ref()
                                .ok_or_else(|| "raster indirect outside render pass".to_owned())?;
                            lower_raster_indirect(pass, draw, self.registration())?;
                        }
                        RecordedPayload::RasterEnd => {
                            let pass = raster_pass
                                .take()
                                .ok_or_else(|| "render pass end without begin".to_owned())?;
                            call0(&pass, "end")?;
                        }
                        RecordedPayload::ComputeBegin(_) => {
                            if raster_pass.is_some() || compute_pass.is_some() {
                                return Err(
                                    "portable command scope nesting reached WebGPU lowering".into(),
                                );
                            }
                            compute_pass =
                                Some(call1(&encoder, "beginComputePass", &Object::new().into())?);
                        }
                        RecordedPayload::ComputeDispatch(draw) => {
                            let pass = compute_pass.as_ref().ok_or_else(|| {
                                "compute dispatch without compute pass".to_owned()
                            })?;
                            lower_compute_dispatch(pass, draw, self.registration())?;
                        }
                        RecordedPayload::ComputeIndirect(draw) => {
                            let pass = compute_pass.as_ref().ok_or_else(|| {
                                "compute indirect outside compute pass".to_owned()
                            })?;
                            lower_compute_indirect(pass, draw, self.registration())?;
                        }
                        RecordedPayload::ComputeEnd => {
                            let pass = compute_pass
                                .take()
                                .ok_or_else(|| "compute pass end without begin".to_owned())?;
                            call0(&pass, "end")?;
                        }
                        RecordedPayload::Upload(job) => {
                            lower_upload(&queue, job, self.registration())?
                        }
                        RecordedPayload::Readback(ticket) => {
                            lower_readback(&encoder, ticket, self.registration(), &mut readbacks)?
                        }
                        RecordedPayload::Copy(CopyRecord::Buffer(copy)) => {
                            lower_buffer_copy(&encoder, copy)?
                        }
                        RecordedPayload::Copy(CopyRecord::BufferToTexture(copy)) => {
                            lower_buffer_texture_copy(&encoder, copy, true)?
                        }
                        RecordedPayload::Copy(CopyRecord::TextureToBuffer(copy)) => {
                            lower_buffer_texture_copy(&encoder, copy, false)?
                        }
                        RecordedPayload::Copy(CopyRecord::Texture(copy)) => {
                            lower_texture_copy(&encoder, copy)?
                        }
                        RecordedPayload::Copy(CopyRecord::ClearBuffer {
                            buffer: value,
                            range,
                        }) => {
                            let native =
                                buffer(value.native(), self.registration(), "clear buffer")
                                    .map_err(|e| e.to_string())?;
                            let value = object(native)
                                .ok_or_else(|| "clear buffer object retired".to_owned())?;
                            call3(
                                &encoder,
                                "clearBuffer",
                                &value,
                                &num(range.offset),
                                &num(range.size),
                            )?;
                        }
                        RecordedPayload::DebugPush(label) => {
                            let _ = call1(
                                &encoder,
                                "pushDebugGroup",
                                &JsValue::from_str(&label.to_string()),
                            )?;
                        }
                        RecordedPayload::DebugPop => {
                            call0(&encoder, "popDebugGroup")?;
                        }
                        RecordedPayload::DebugMarker(label) => {
                            let _ = call1(
                                &encoder,
                                "insertDebugMarker",
                                &JsValue::from_str(&label.to_string()),
                            )?;
                        }
                        _ => return Err("Phase-A admitted an unsupported WebGPU payload".into()),
                    }
                }
            }
        }
        if raster_pass.is_some() || compute_pass.is_some() {
            return Err("portable recorder emitted an unterminated scope".into());
        }
        let commands = call0(&encoder, "finish")?;
        let list = Array::new();
        list.push(&commands);
        let list: JsValue = list.into();
        call1(&queue, "submit", &list)?;
        for present in request.presents {
            // WebGPU has no explicit swapchain Present: submitting work that
            // references the current canvas view transfers the acquired frame
            // to the browser compositor.
            present.attachment.present(present.receipt);
        }
        for pending in &mut readbacks {
            let staging = registry::with_object(self.registration(), pending.staging, Clone::clone)
                .ok_or_else(|| {
                    "readback staging allocation was retired before mapAsync".to_owned()
                })?;
            let promise = call3_value(&staging, "mapAsync", &num(1), &num(0), &num(pending.bytes))?;
            pending.map = Some(registry::start_device_promise(
                self.registration(),
                Promise::from(promise),
            ));
        }
        let completion = call0(&queue, "onSubmittedWorkDone")
            .map(Promise::from)
            // `queue.submit` is already the acceptance boundary. Preserve the
            // v13 `submit Err => zero accepted work` rule by converting a later
            // completion-observation failure into a rejected completion future.
            .unwrap_or_else(|message| Promise::reject(&JsValue::from_str(&message)));
        Ok((completion, readbacks))
    }

    pub(crate) fn poll(&self) -> RhiResult<()> {
        self.advance();
        Ok(())
    }
    pub(crate) fn completion(&self, serial: u64) -> CompletionState {
        self.advance();
        self.state(serial, None)
    }
    pub(crate) fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> CompletionState {
        self.advance();
        self.state(serial, Some(waker))
    }

    fn advance(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        // Publish readback bytes before completion. `mapAsync` resolves only
        // after the copy that precedes it, so this may finish before the queue
        // promise. If it does not, the completion below remains Pending until
        // the ticket reaches its terminal state.
        let mut failed_readbacks = Vec::new();
        let mut still_pending = Vec::with_capacity(state.readbacks.len());
        for pending in state.readbacks.drain(..) {
            let map = pending
                .map
                .expect("mapAsync is started after queue submission");
            match registry::try_take_settled_promise(map) {
                PromisePoll::Pending => still_pending.push(pending),
                PromisePoll::Failed(_) => {
                    if registry::device_loss(self.registration()).is_some() {
                        pending
                            .ticket
                            .set_status(crate::api::resource::ReadbackStatus::DeviceLost);
                    } else {
                        pending
                            .ticket
                            .set_status(crate::api::resource::ReadbackStatus::Failed);
                        if let Some(completion) = pending.completion {
                            failed_readbacks.push(completion);
                        }
                    }
                    let _ = registry::remove_object(self.registration(), pending.staging);
                }
                PromisePoll::Ready(_) => {
                    let bytes =
                        registry::with_object(self.registration(), pending.staging, |value| {
                            read_mapped_bytes(value, pending.bytes)
                        })
                        .flatten();
                    match bytes {
                        Some(bytes) => pending.ticket.publish(bytes, pending.layout),
                        None => {
                            pending
                                .ticket
                                .set_status(crate::api::resource::ReadbackStatus::Failed);
                            if let Some(completion) = pending.completion {
                                failed_readbacks.push(completion);
                            }
                        }
                    }
                    let _ =
                        registry::with_object(self.registration(), pending.staging, unmap_buffer);
                    let _ = registry::remove_object(self.registration(), pending.staging);
                }
            }
        }
        state.readbacks = still_pending;
        // Each plan shares one promise. Poll it only once, then fan the result
        // out to each point carrying that serial; `poll_promise` consumes a
        // settled request, so collecting IDs first is essential.
        let mut ids = Vec::new();
        for id in state.serials.values().filter_map(|entry| match entry {
            SerialState::Pending(id) => Some(*id),
            _ => None,
        }) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        for id in ids {
            match registry::try_take_settled_promise(id) {
                PromisePoll::Pending => {}
                PromisePoll::Ready(_) => {
                    if !state
                        .readbacks
                        .iter()
                        .any(|readback| readback.completion == Some(id))
                    {
                        for entry in state.serials.values_mut() {
                            if matches!(entry, SerialState::Pending(current) if *current == id) {
                                *entry = SerialState::Complete;
                            }
                        }
                    } else {
                        for entry in state.serials.values_mut() {
                            if matches!(entry, SerialState::Pending(current) if *current == id) {
                                *entry = SerialState::AwaitReadbacks(id);
                            }
                        }
                    }
                }
                PromisePoll::Failed(message) => {
                    for entry in state.serials.values_mut() {
                        if matches!(entry, SerialState::Pending(current) if *current == id) {
                            *entry = SerialState::Failed(message.clone());
                        }
                    }
                }
            }
        }
        for id in failed_readbacks {
            for entry in state.serials.values_mut() {
                if matches!(entry, SerialState::Pending(current) | SerialState::AwaitReadbacks(current) if *current == id)
                {
                    *entry = SerialState::Failed("WebGPU readback mapping failed".into());
                }
            }
        }
        let published_readback_completions: Vec<_> = state
            .serials
            .values()
            .filter_map(|entry| match entry {
                SerialState::AwaitReadbacks(id)
                    if !state
                        .readbacks
                        .iter()
                        .any(|readback| readback.completion == Some(*id)) =>
                {
                    Some(*id)
                }
                _ => None,
            })
            .collect();
        for entry in state.serials.values_mut() {
            if matches!(entry, SerialState::AwaitReadbacks(id) if published_readback_completions.contains(id))
            {
                *entry = SerialState::Complete;
            }
        }
    }

    fn state(&self, serial: u64, waker: Option<&Waker>) -> CompletionState {
        if let Some(info) = registry::device_loss(self.registration()) {
            return CompletionState::DeviceLost(info);
        }
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(SerialState::AwaitReadbacks(id)) = state.serials.get(&serial) {
            let maps = state
                .readbacks
                .iter()
                .filter(|readback| readback.completion == Some(*id))
                .filter_map(|readback| readback.map)
                .collect::<Vec<_>>();
            // A browser promise callback may wake an executor synchronously.
            // Never retain the spine mutex across that foreign wake boundary.
            drop(state);
            if let Some(waker) = waker {
                for map in maps {
                    registry::register_promise_waker(map, waker);
                }
            }
            return CompletionState::Pending;
        }
        match state.serials.get(&serial) {
            Some(SerialState::Complete) => CompletionState::Complete,
            Some(SerialState::Failed(message)) => {
                CompletionState::Failed(CompletionFailure::new(message.clone()))
            }
            Some(SerialState::Pending(id)) => {
                if let Some(waker) = waker {
                    let _ = registry::poll_promise(*id, waker);
                }
                CompletionState::Pending
            }
            Some(SerialState::AwaitReadbacks(_)) => unreachable!("handled above"),
            None if serial <= state.issued => CompletionState::Complete,
            None => {
                CompletionState::Failed(CompletionFailure::new("unknown WebGPU completion serial"))
            }
        }
    }
}

fn buffer<'a>(
    value: &'a dyn crate::api::resource::backend::BufferBackend,
    registration: WebGpuRegistration,
    what: &'static str,
) -> RhiResult<&'a WebGpuBuffer> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuBuffer>()
        .ok_or_else(|| unsupported(what))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another WebGPU device"),
            )
        })
}
fn texture<'a>(
    value: &'a dyn crate::api::resource::backend::TextureBackend,
    registration: WebGpuRegistration,
    what: &'static str,
) -> RhiResult<&'a WebGpuTexture> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuTexture>()
        .ok_or_else(|| unsupported(what))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another WebGPU device"),
            )
        })
}
fn texture_view<'a>(
    value: &'a dyn crate::api::resource::backend::TextureViewBackend,
    registration: WebGpuRegistration,
    what: &'static str,
) -> RhiResult<&'a WebGpuTextureView> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuTextureView>()
        .ok_or_else(|| unsupported(what))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another WebGPU device"),
            )
        })
}
fn query_set<'a>(
    value: &'a QuerySet,
    registration: WebGpuRegistration,
    what: &'static str,
) -> RhiResult<&'a WebGpuQuerySet> {
    let value = value
        .native()
        .as_any()
        .downcast_ref::<WebGpuQuerySet>()
        .ok_or_else(|| unsupported(what))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another WebGPU device"),
            )
        })
}
fn bind_group<'a>(
    value: &'a dyn crate::api::binding::backend::BindGroupBackend,
    registration: WebGpuRegistration,
) -> RhiResult<&'a WebGpuBindGroup> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuBindGroup>()
        .ok_or_else(|| unsupported("bind group"))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "bind group belongs to another WebGPU device",
            )
        })
}
fn raster_pipeline<'a>(
    value: &'a dyn crate::api::pipeline::backend::RasterPipelineBackend,
    registration: WebGpuRegistration,
) -> RhiResult<&'a WebGpuRasterPipeline> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuRasterPipeline>()
        .ok_or_else(|| unsupported("raster pipeline"))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "raster pipeline belongs to another WebGPU device",
            )
        })
}
fn compute_pipeline<'a>(
    value: &'a dyn crate::api::pipeline::backend::ComputePipelineBackend,
    registration: WebGpuRegistration,
) -> RhiResult<&'a WebGpuComputePipeline> {
    let value = value
        .as_any()
        .downcast_ref::<WebGpuComputePipeline>()
        .ok_or_else(|| unsupported("compute pipeline"))?;
    (value.registration() == registration)
        .then_some(value)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "compute pipeline belongs to another WebGPU device",
            )
        })
}
fn object<T>(value: &T) -> Option<JsValue>
where
    T: Registered,
{
    registry::with_object(value.registration(), value.object(), Clone::clone)
}
trait Registered {
    fn registration(&self) -> WebGpuRegistration;
    fn object(&self) -> super::registry::WebGpuObjectId;
}
impl Registered for WebGpuBuffer {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuTexture {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuTextureView {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuQuerySet {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuBindGroup {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuRasterPipeline {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}
impl Registered for WebGpuComputePipeline {
    fn registration(&self) -> WebGpuRegistration {
        self.registration()
    }
    fn object(&self) -> super::registry::WebGpuObjectId {
        self.object()
    }
}

fn lower_buffer_copy(encoder: &JsValue, copy: &BufferCopy) -> Result<(), String> {
    let registration = buffer_registration(&copy.src).map_err(|e| e.to_string())?;
    let src =
        object(buffer(copy.src.native(), registration, "copy source").map_err(|e| e.to_string())?)
            .ok_or_else(|| "copy source retired".to_owned())?;
    let dst = object(
        buffer(
            copy.dst.native(),
            buffer_registration(&copy.dst).map_err(|e| e.to_string())?,
            "copy destination",
        )
        .map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "copy destination retired".to_owned())?;
    call5(
        encoder,
        "copyBufferToBuffer",
        &src,
        &num(copy.src_offset),
        &dst,
        &num(copy.dst_offset),
        &num(copy.size),
    )
}
fn buffer_registration(value: &crate::api::resource::Buffer) -> RhiResult<WebGpuRegistration> {
    value
        .native()
        .as_any()
        .downcast_ref::<WebGpuBuffer>()
        .map(WebGpuBuffer::registration)
        .ok_or_else(|| unsupported("buffer"))
}
fn texture_registration(value: &crate::api::resource::Texture) -> RhiResult<WebGpuRegistration> {
    value
        .native()
        .as_any()
        .downcast_ref::<WebGpuTexture>()
        .map(WebGpuTexture::registration)
        .ok_or_else(|| unsupported("texture"))
}

fn lower_buffer_texture_copy(
    encoder: &JsValue,
    copy: &BufferTextureCopy,
    to_texture: bool,
) -> Result<(), String> {
    let registration = buffer_registration(&copy.buffer).map_err(|e| e.to_string())?;
    if texture_registration(&copy.texture).map_err(|e| e.to_string())? != registration {
        return Err("buffer/texture copy crosses WebGPU devices".into());
    }
    let buffer_value = object(
        buffer(copy.buffer.native(), registration, "copy buffer").map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "copy buffer retired".to_owned())?;
    let texture_value = object(
        texture(copy.texture.native(), registration, "copy texture").map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "copy texture retired".to_owned())?;
    let buffer_desc = Object::new();
    set(&buffer_desc, "buffer", &buffer_value)?;
    set(&buffer_desc, "offset", &num(copy.buffer_offset))?;
    set(&buffer_desc, "bytesPerRow", &num(copy.bytes_per_row as u64))?;
    set(
        &buffer_desc,
        "rowsPerImage",
        &num(copy.rows_per_image as u64),
    )?;
    let texture_desc = texture_copy_desc(
        &texture_value,
        copy.texture_subresource.mip_level,
        copy.texture_subresource.base_layer,
        copy.texture_subresource.aspect,
        copy.texture_origin,
    )?;
    let extent = extent(copy.extent);
    if to_texture {
        call3(
            encoder,
            "copyBufferToTexture",
            &buffer_desc.into(),
            &texture_desc.into(),
            &extent,
        )
    } else {
        call3(
            encoder,
            "copyTextureToBuffer",
            &texture_desc.into(),
            &buffer_desc.into(),
            &extent,
        )
    }
}

fn lower_texture_copy(encoder: &JsValue, copy: &TextureCopy) -> Result<(), String> {
    let registration = texture_registration(&copy.src).map_err(|e| e.to_string())?;
    if texture_registration(&copy.dst).map_err(|e| e.to_string())? != registration {
        return Err("texture copy crosses WebGPU devices".into());
    }
    let src =
        object(texture(copy.src.native(), registration, "copy source").map_err(|e| e.to_string())?)
            .ok_or_else(|| "copy source retired".to_owned())?;
    let dst = object(
        texture(copy.dst.native(), registration, "copy destination").map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "copy destination retired".to_owned())?;
    let src = texture_copy_desc(
        &src,
        copy.src_subresource.mip_level,
        copy.src_subresource.base_layer,
        copy.src_subresource.aspect,
        copy.src_origin,
    )?;
    let dst = texture_copy_desc(
        &dst,
        copy.dst_subresource.mip_level,
        copy.dst_subresource.base_layer,
        copy.dst_subresource.aspect,
        copy.dst_origin,
    )?;
    call3(
        encoder,
        "copyTextureToTexture",
        &src.into(),
        &dst.into(),
        &extent(copy.extent),
    )
}

fn preflight_raster_begin(begin: &RasterBegin, registration: WebGpuRegistration) -> RhiResult<()> {
    if let Some(set) = &begin.occlusion_query_set {
        query_set(set, registration, "raster occlusion query set")?;
        if set.descriptor().ty != QueryType::Occlusion {
            return Err(unsupported("non-occlusion raster query set"));
        }
    }
    for (_, attachment) in &begin.colors {
        match &attachment.view {
            crate::api::command::attachment::ColorAttachmentView::Texture(view) => {
                texture_view(view.native(), registration, "raster color attachment")?;
            }
            crate::api::command::attachment::ColorAttachmentView::Frame(frame) => {
                // `frame_view` owns the acquired lease check. It does not touch
                // JS; acquisition happened before recording/submission.
                super::presentation::frame_view(frame)?;
            }
        }
        if let Some(crate::api::command::attachment::ColorAttachmentView::Texture(view)) =
            &attachment.resolve
        {
            texture_view(view.native(), registration, "raster resolve attachment")?;
        } else if attachment.resolve.is_some() {
            return Err(unsupported("a frame resolve attachment"));
        }
    }
    if let Some(depth) = &begin.depth_stencil {
        texture_view(
            depth.view.native(),
            registration,
            "depth/stencil attachment",
        )?;
    }
    Ok(())
}

fn preflight_raster_draw(draw: &RasterDraw, registration: WebGpuRegistration) -> RhiResult<()> {
    raster_pipeline(draw.pipeline.native(), registration)?;
    preflight_groups(&draw.groups, registration)?;
    for (_, binding) in &draw.vertex_buffers {
        buffer(binding.buffer.native(), registration, "vertex buffer")?;
    }
    if let Some(index) = &draw.index {
        buffer(index.binding.buffer.native(), registration, "index buffer")?;
    }
    if !draw.immediates.is_empty() {
        return Err(unsupported("immediate data"));
    }
    Ok(())
}
fn preflight_compute_draw(
    draw: &ComputeDispatch,
    registration: WebGpuRegistration,
) -> RhiResult<()> {
    compute_pipeline(draw.pipeline.native(), registration)?;
    preflight_groups(&draw.groups, registration)?;
    if !draw.immediates.is_empty() {
        return Err(unsupported("immediate data"));
    }
    Ok(())
}
fn preflight_raster_indirect(
    draw: &RasterIndirect,
    registration: WebGpuRegistration,
) -> RhiResult<()> {
    raster_pipeline(draw.pipeline.native(), registration)?;
    preflight_groups(&draw.groups, registration)?;
    for (_, binding) in &draw.vertex_buffers {
        buffer(binding.buffer.native(), registration, "vertex buffer")?;
    }
    if let Some(index) = &draw.index {
        buffer(index.binding.buffer.native(), registration, "index buffer")?;
    }
    buffer(
        draw.arguments.native(),
        registration,
        "indirect argument buffer",
    )?;
    if draw.count.is_some() {
        return Err(unsupported("multi-draw indirect count buffer"));
    }
    Ok(())
}
fn preflight_compute_indirect(
    draw: &ComputeIndirect,
    registration: WebGpuRegistration,
) -> RhiResult<()> {
    compute_pipeline(draw.pipeline.native(), registration)?;
    preflight_groups(&draw.groups, registration)?;
    buffer(
        draw.arguments.native(),
        registration,
        "indirect argument buffer",
    )?;
    Ok(())
}
fn preflight_upload(
    job: &crate::api::resource::UploadJob,
    registration: WebGpuRegistration,
) -> RhiResult<()> {
    match job.descriptor() {
        crate::api::resource::UploadDescriptor::Buffer(value) => {
            buffer(value.dst.native(), registration, "upload destination")?;
        }
        crate::api::resource::UploadDescriptor::Texture(value) => {
            texture(value.dst.native(), registration, "upload destination")?;
        }
    }
    Ok(())
}
fn preflight_readback(
    ticket: &crate::api::resource::ReadbackTicket,
    registration: WebGpuRegistration,
) -> RhiResult<()> {
    match ticket.request() {
        crate::api::resource::ReadbackRequest::Buffer { src, .. } => {
            buffer(src.native(), registration, "readback source")?;
            Ok(())
        }
        crate::api::resource::ReadbackRequest::Texture { .. } => {
            let ReadbackRequest::Texture { src, .. } = ticket.request() else {
                unreachable!();
            };
            texture(src.native(), registration, "readback texture source")?;
            texture_readback_layout(ticket).map(|_| ())
        }
    }
}

fn lower_readback(
    encoder: &JsValue,
    ticket: &crate::api::resource::ReadbackTicket,
    registration: WebGpuRegistration,
    out: &mut Vec<PendingReadback>,
) -> Result<(), String> {
    if let ReadbackRequest::Texture { .. } = ticket.request() {
        return lower_texture_readback(encoder, ticket, registration, out);
    }
    let ReadbackRequest::Buffer { src, range, .. } = ticket.request() else {
        unreachable!()
    };
    let source =
        object(buffer(src.native(), registration, "readback source").map_err(|e| e.to_string())?)
            .ok_or_else(|| "readback source retired".to_owned())?;
    let device = registry::with_device_handles(registration, |h| h.device.clone())
        .ok_or_else(|| "WebGPU device retired".to_owned())?;
    let desc = Object::new();
    set(&desc, "size", &num(range.size))?;
    set(&desc, "usage", &num(1 | 8))?;
    let staging = call1(&device, "createBuffer", &desc.into())?;
    let staging_id = registry::insert_object(registration, staging.clone())
        .ok_or_else(|| "WebGPU device retired while retaining readback staging".to_owned())?;
    call5(
        encoder,
        "copyBufferToBuffer",
        &source,
        &num(range.offset),
        &staging,
        &num(0),
        &num(range.size),
    )?;
    out.push(PendingReadback {
        ticket: ticket.clone(),
        staging: staging_id,
        bytes: range.size,
        layout: None,
        map: None,
        completion: None,
    });
    Ok(())
}

fn texture_readback_layout(
    ticket: &crate::api::resource::ReadbackTicket,
) -> RhiResult<ReadbackTexelLayout> {
    let ReadbackRequest::Texture {
        src,
        subresource,
        extent,
        ..
    } = ticket.request()
    else {
        unreachable!();
    };
    let bytes = logical_bytes_per_block(src.descriptor().format).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "texture format has no WebGPU copy footprint",
        )
    })?;
    let (block_width, block_height) = block_extent(src.descriptor().format);
    let columns = extent.width.div_ceil(block_width);
    let rows = extent.height.div_ceil(block_height);
    let unaligned = columns.checked_mul(bytes).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture readback row size overflow",
        )
    })?;
    // A private staging buffer is a GPU copy buffer, so its rows obey WebGPU's
    // 256-byte footprint alignment; the public ticket reports that padding.
    let bytes_per_row = unaligned.checked_add(255).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture readback row alignment overflow",
        )
    })? / 256
        * 256;
    let images = if matches!(
        src.descriptor().dimension,
        crate::api::resource::TextureDimension::D3
    ) {
        extent.depth
    } else {
        subresource.layer_count
    };
    let total_size = u64::from(bytes_per_row)
        .checked_mul(u64::from(rows))
        .and_then(|value| value.checked_mul(u64::from(images)))
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture readback staging size overflow",
            )
        })?;
    Ok(ReadbackTexelLayout {
        bytes_per_row,
        rows_per_image: rows,
        total_size,
    })
}

fn lower_texture_readback(
    encoder: &JsValue,
    ticket: &crate::api::resource::ReadbackTicket,
    registration: WebGpuRegistration,
    out: &mut Vec<PendingReadback>,
) -> Result<(), String> {
    let ReadbackRequest::Texture {
        src,
        subresource,
        origin,
        extent,
        ..
    } = ticket.request()
    else {
        unreachable!()
    };
    let layout = texture_readback_layout(ticket).map_err(|error| error.to_string())?;
    let source = object(
        texture(src.native(), registration, "readback texture source")
            .map_err(|error| error.to_string())?,
    )
    .ok_or_else(|| "readback texture source retired".to_owned())?;
    let device = registry::with_device_handles(registration, |h| h.device.clone())
        .ok_or_else(|| "WebGPU device retired".to_owned())?;
    let descriptor = Object::new();
    set(&descriptor, "size", &num(layout.total_size))?;
    set(&descriptor, "usage", &num(1 | 8))?;
    let staging = call1(&device, "createBuffer", &descriptor.into())?;
    let staging_id = registry::insert_object(registration, staging.clone()).ok_or_else(|| {
        "WebGPU device retired while retaining texture readback staging".to_owned()
    })?;
    let source_desc = Object::new();
    set(&source_desc, "texture", &source)?;
    set(&source_desc, "mipLevel", &num(subresource.mip_level as u64))?;
    let aspect = match subresource.aspect {
        crate::api::resource::TextureAspect::Color => "all",
        crate::api::resource::TextureAspect::Depth => "depth-only",
        crate::api::resource::TextureAspect::Stencil => "stencil-only",
        crate::api::resource::TextureAspect::Plane0
        | crate::api::resource::TextureAspect::Plane1
        | crate::api::resource::TextureAspect::Plane2 => {
            return Err("WebGPU has no portable multi-planar texture readback lowering".into());
        }
    };
    set(&source_desc, "aspect", &JsValue::from_str(aspect))?;
    let source_origin = Object::new();
    set(&source_origin, "x", &num(origin.x as u64))?;
    set(&source_origin, "y", &num(origin.y as u64))?;
    let source_z = if matches!(
        src.descriptor().dimension,
        crate::api::resource::TextureDimension::D3
    ) {
        origin.z
    } else {
        subresource.base_layer
    };
    set(&source_origin, "z", &num(source_z as u64))?;
    set(&source_desc, "origin", &source_origin.into())?;
    let destination = Object::new();
    set(&destination, "buffer", &staging)?;
    set(&destination, "offset", &num(0))?;
    set(
        &destination,
        "bytesPerRow",
        &num(layout.bytes_per_row as u64),
    )?;
    set(
        &destination,
        "rowsPerImage",
        &num(layout.rows_per_image as u64),
    )?;
    let copy_extent = Object::new();
    set(&copy_extent, "width", &num(extent.width as u64))?;
    set(&copy_extent, "height", &num(extent.height as u64))?;
    set(
        &copy_extent,
        "depthOrArrayLayers",
        &num(
            if matches!(
                src.descriptor().dimension,
                crate::api::resource::TextureDimension::D3
            ) {
                extent.depth as u64
            } else {
                subresource.layer_count as u64
            },
        ),
    )?;
    call3(
        encoder,
        "copyTextureToBuffer",
        &source_desc.into(),
        &destination.into(),
        &copy_extent.into(),
    )?;
    out.push(PendingReadback {
        ticket: ticket.clone(),
        staging: staging_id,
        bytes: layout.total_size,
        layout: Some(layout),
        map: None,
        completion: None,
    });
    Ok(())
}
fn preflight_groups(
    groups: &[crate::api::command::record::BoundGroup],
    registration: WebGpuRegistration,
) -> RhiResult<()> {
    for group in groups {
        bind_group(group.group.native(), registration)?;
    }
    Ok(())
}

fn begin_raster(
    encoder: &JsValue,
    begin: &RasterBegin,
    registration: WebGpuRegistration,
) -> Result<JsValue, String> {
    let descriptor = Object::new();
    let colors = Array::new();
    // WebGPU's colorAttachments array permits null holes, exactly matching the
    // portable sparse MRT locations rather than compacting their indices.
    let mut next = 0u32;
    for (location, attachment) in &begin.colors {
        while next < *location {
            colors.push(&JsValue::NULL);
            next += 1;
        }
        let item = Object::new();
        let view = color_view(&attachment.view, registration).map_err(|e| e.to_string())?;
        set(&item, "view", &view)?;
        set(
            &item,
            "loadOp",
            &JsValue::from_str(match attachment.load {
                crate::api::command::LoadOp::Load => "load",
                crate::api::command::LoadOp::Clear(_) => "clear",
            }),
        )?;
        set(
            &item,
            "storeOp",
            &JsValue::from_str(match attachment.store {
                crate::api::command::StoreOp::Store => "store",
                crate::api::command::StoreOp::Discard => "discard",
            }),
        )?;
        if let crate::api::command::LoadOp::Clear(value) = attachment.load {
            set(&item, "clearValue", &clear_color(value).into())?;
        }
        if let Some(resolve) = &attachment.resolve {
            set(
                &item,
                "resolveTarget",
                &color_view(resolve, registration).map_err(|e| e.to_string())?,
            )?;
        }
        colors.push(&item);
        next += 1;
    }
    set(&descriptor, "colorAttachments", &colors.into())?;
    if let Some(depth) = &begin.depth_stencil {
        set(
            &descriptor,
            "depthStencilAttachment",
            &depth_attachment(depth, registration)
                .map_err(|e| e.to_string())?
                .into(),
        )?;
    }
    if let Some(query) = &begin.occlusion_query_set {
        let native_set = object(
            query_set(query, registration, "raster occlusion query set")
                .map_err(|error| error.to_string())?,
        )
        .ok_or_else(|| "raster occlusion query set was retired".to_owned())?;
        set(&descriptor, "occlusionQuerySet", &native_set)?;
    }
    call1(encoder, "beginRenderPass", &descriptor.into())
}

fn color_view(
    view: &crate::api::command::attachment::ColorAttachmentView,
    registration: WebGpuRegistration,
) -> RhiResult<JsValue> {
    match view {
        crate::api::command::attachment::ColorAttachmentView::Texture(view) => object(
            texture_view(view.native(), registration, "color attachment")?,
        )
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::DeviceLost,
                "color attachment view was retired",
            )
        }),
        crate::api::command::attachment::ColorAttachmentView::Frame(frame) => {
            super::presentation::frame_view(frame)
        }
    }
}
fn depth_attachment(
    value: &crate::api::command::attachment::DepthStencilAttachment,
    registration: WebGpuRegistration,
) -> RhiResult<Object> {
    let out = Object::new();
    set_rhi(
        &out,
        "view",
        object(texture_view(
            value.view.native(),
            registration,
            "depth/stencil attachment",
        )?)
        .ok_or_else(|| RhiError::new(RhiErrorKind::DeviceLost, "depth/stencil view retired"))?,
    )?;
    if let Some(depth) = value.depth {
        match depth {
            crate::api::command::attachment::DepthAttachmentMode::ReadOnly => {
                set_rhi(&out, "depthReadOnly", JsValue::TRUE)?;
            }
            crate::api::command::attachment::DepthAttachmentMode::ReadWrite { load, store } => {
                set_rhi(
                    &out,
                    "depthLoadOp",
                    JsValue::from_str(match load {
                        crate::api::command::LoadOp::Load => "load",
                        crate::api::command::LoadOp::Clear(_) => "clear",
                    }),
                )?;
                set_rhi(
                    &out,
                    "depthStoreOp",
                    JsValue::from_str(match store {
                        crate::api::command::StoreOp::Store => "store",
                        crate::api::command::StoreOp::Discard => "discard",
                    }),
                )?;
                if let crate::api::command::LoadOp::Clear(v) = load {
                    set_rhi(&out, "depthClearValue", JsValue::from_f64(v as f64))?;
                }
            }
        }
    }
    if let Some(stencil) = value.stencil {
        match stencil {
            crate::api::command::attachment::StencilAttachmentMode::ReadOnly => {
                set_rhi(&out, "stencilReadOnly", JsValue::TRUE)?;
            }
            crate::api::command::attachment::StencilAttachmentMode::ReadWrite { load, store } => {
                set_rhi(
                    &out,
                    "stencilLoadOp",
                    JsValue::from_str(match load {
                        crate::api::command::LoadOp::Load => "load",
                        crate::api::command::LoadOp::Clear(_) => "clear",
                    }),
                )?;
                set_rhi(
                    &out,
                    "stencilStoreOp",
                    JsValue::from_str(match store {
                        crate::api::command::StoreOp::Store => "store",
                        crate::api::command::StoreOp::Discard => "discard",
                    }),
                )?;
                if let crate::api::command::LoadOp::Clear(v) = load {
                    set_rhi(&out, "stencilClearValue", num(v as u64))?;
                }
            }
        }
    }
    Ok(out)
}

fn lower_raster_draw(
    pass: &JsValue,
    draw: &RasterDraw,
    registration: WebGpuRegistration,
) -> Result<(), String> {
    call1(
        pass,
        "setPipeline",
        &object(raster_pipeline(draw.pipeline.native(), registration).map_err(|e| e.to_string())?)
            .ok_or_else(|| "raster pipeline retired".to_owned())?,
    )?;
    lower_groups(pass, &draw.groups, registration)?;
    for (slot, binding) in &draw.vertex_buffers {
        let native = object(
            buffer(binding.buffer.native(), registration, "vertex buffer")
                .map_err(|e| e.to_string())?,
        )
        .ok_or_else(|| "vertex buffer retired".to_owned())?;
        call4(
            pass,
            "setVertexBuffer",
            &num(*slot as u64),
            &native,
            &num(binding.range.offset),
            &num(binding.range.size),
        )?;
    }
    if let Some(index) = &draw.index {
        let native = object(
            buffer(index.binding.buffer.native(), registration, "index buffer")
                .map_err(|e| e.to_string())?,
        )
        .ok_or_else(|| "index buffer retired".to_owned())?;
        let format = match index.format {
            crate::api::command::IndexFormat::Uint16 => "uint16",
            crate::api::command::IndexFormat::Uint32 => "uint32",
        };
        call4(
            pass,
            "setIndexBuffer",
            &native,
            &JsValue::from_str(format),
            &num(index.binding.range.offset),
            &num(index.binding.range.size),
        )?;
    }
    if let Some(v) = draw.viewport {
        call6(
            pass,
            "setViewport",
            &JsValue::from_f64(v.x as f64),
            &JsValue::from_f64(v.y as f64),
            &JsValue::from_f64(v.width as f64),
            &JsValue::from_f64(v.height as f64),
            &JsValue::from_f64(v.min_depth as f64),
            &JsValue::from_f64(v.max_depth as f64),
        )?;
    }
    if let Some(s) = draw.scissor {
        call4(
            pass,
            "setScissorRect",
            &num(s.x as u64),
            &num(s.y as u64),
            &num(s.width as u64),
            &num(s.height as u64),
        )?;
    }
    let blend = Object::new();
    set(
        &blend,
        "r",
        &JsValue::from_f64(draw.blend_constant.r as f64),
    )?;
    set(
        &blend,
        "g",
        &JsValue::from_f64(draw.blend_constant.g as f64),
    )?;
    set(
        &blend,
        "b",
        &JsValue::from_f64(draw.blend_constant.b as f64),
    )?;
    set(
        &blend,
        "a",
        &JsValue::from_f64(draw.blend_constant.a as f64),
    )?;
    let blend: JsValue = blend.into();
    call1(pass, "setBlendConstant", &blend)?;
    call1(
        pass,
        "setStencilReference",
        &num(draw.stencil_reference as u64),
    )?;
    if draw.index.is_some() {
        call5(
            pass,
            "drawIndexed",
            &num((draw.range.end - draw.range.start) as u64),
            &num((draw.instances.end - draw.instances.start) as u64),
            &num(draw.range.start as u64),
            &JsValue::from_f64(draw.base_vertex as f64),
            &num(draw.instances.start as u64),
        )?;
    } else {
        call4(
            pass,
            "draw",
            &num((draw.range.end - draw.range.start) as u64),
            &num((draw.instances.end - draw.instances.start) as u64),
            &num(draw.range.start as u64),
            &num(draw.instances.start as u64),
        )?;
    }
    Ok(())
}
fn bind_raster_state(
    pass: &JsValue,
    draw: &RasterDraw,
    registration: WebGpuRegistration,
) -> Result<(), String> {
    call1(
        pass,
        "setPipeline",
        &object(raster_pipeline(draw.pipeline.native(), registration).map_err(|e| e.to_string())?)
            .ok_or_else(|| "raster pipeline retired".to_owned())?,
    )?;
    lower_groups(pass, &draw.groups, registration)?;
    for (slot, binding) in &draw.vertex_buffers {
        let native = object(
            buffer(binding.buffer.native(), registration, "vertex buffer")
                .map_err(|e| e.to_string())?,
        )
        .ok_or_else(|| "vertex buffer retired".to_owned())?;
        call4(
            pass,
            "setVertexBuffer",
            &num(*slot as u64),
            &native,
            &num(binding.range.offset),
            &num(binding.range.size),
        )?;
    }
    if let Some(index) = &draw.index {
        let native = object(
            buffer(index.binding.buffer.native(), registration, "index buffer")
                .map_err(|e| e.to_string())?,
        )
        .ok_or_else(|| "index buffer retired".to_owned())?;
        let format = match index.format {
            crate::api::command::IndexFormat::Uint16 => "uint16",
            crate::api::command::IndexFormat::Uint32 => "uint32",
        };
        call4(
            pass,
            "setIndexBuffer",
            &native,
            &JsValue::from_str(format),
            &num(index.binding.range.offset),
            &num(index.binding.range.size),
        )?;
    }
    if let Some(v) = draw.viewport {
        call6(
            pass,
            "setViewport",
            &JsValue::from_f64(v.x as f64),
            &JsValue::from_f64(v.y as f64),
            &JsValue::from_f64(v.width as f64),
            &JsValue::from_f64(v.height as f64),
            &JsValue::from_f64(v.min_depth as f64),
            &JsValue::from_f64(v.max_depth as f64),
        )?;
    }
    if let Some(s) = draw.scissor {
        call4(
            pass,
            "setScissorRect",
            &num(s.x as u64),
            &num(s.y as u64),
            &num(s.width as u64),
            &num(s.height as u64),
        )?;
    }
    Ok(())
}
fn lower_compute_dispatch(
    pass: &JsValue,
    draw: &ComputeDispatch,
    registration: WebGpuRegistration,
) -> Result<(), String> {
    call1(
        pass,
        "setPipeline",
        &object(compute_pipeline(draw.pipeline.native(), registration).map_err(|e| e.to_string())?)
            .ok_or_else(|| "compute pipeline retired".to_owned())?,
    )?;
    lower_groups(pass, &draw.groups, registration)?;
    call3(
        pass,
        "dispatchWorkgroups",
        &num(draw.workgroups.0 as u64),
        &num(draw.workgroups.1 as u64),
        &num(draw.workgroups.2 as u64),
    )
}
fn lower_raster_indirect(
    pass: &JsValue,
    draw: &RasterIndirect,
    registration: WebGpuRegistration,
) -> Result<(), String> {
    let full = RasterDraw {
        pipeline: draw.pipeline.clone(),
        groups: draw.groups.clone(),
        vertex_buffers: draw.vertex_buffers.clone(),
        index: draw.index.clone(),
        viewport: draw.viewport,
        scissor: draw.scissor,
        blend_constant: draw.blend_constant,
        stencil_reference: draw.stencil_reference,
        range: 0..0,
        instances: 0..0,
        base_vertex: 0,
        immediates: Vec::new(),
    };
    bind_raster_state(pass, &full, registration)?;
    let args = object(
        buffer(
            draw.arguments.native(),
            registration,
            "indirect argument buffer",
        )
        .map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "indirect argument buffer retired".to_owned())?;
    if draw.index.is_some() {
        call2(
            pass,
            "drawIndexedIndirect",
            &args,
            &num(draw.arguments_offset),
        )?;
    } else {
        call2(pass, "drawIndirect", &args, &num(draw.arguments_offset))?;
    }
    Ok(())
}
fn lower_compute_indirect(
    pass: &JsValue,
    draw: &ComputeIndirect,
    registration: WebGpuRegistration,
) -> Result<(), String> {
    call1(
        pass,
        "setPipeline",
        &object(compute_pipeline(draw.pipeline.native(), registration).map_err(|e| e.to_string())?)
            .ok_or_else(|| "compute pipeline retired".to_owned())?,
    )?;
    lower_groups(pass, &draw.groups, registration)?;
    let args = object(
        buffer(
            draw.arguments.native(),
            registration,
            "indirect argument buffer",
        )
        .map_err(|e| e.to_string())?,
    )
    .ok_or_else(|| "indirect argument buffer retired".to_owned())?;
    call2(
        pass,
        "dispatchWorkgroupsIndirect",
        &args,
        &num(draw.arguments_offset),
    )
}
fn lower_groups(
    pass: &JsValue,
    groups: &[crate::api::command::record::BoundGroup],
    registration: WebGpuRegistration,
) -> Result<(), String> {
    for group in groups {
        let native =
            object(bind_group(group.group.native(), registration).map_err(|e| e.to_string())?)
                .ok_or_else(|| "bind group retired".to_owned())?;
        let offsets = Array::new();
        for offset in &group.dynamic_offsets {
            offsets.push(&num(*offset as u64));
        }
        let offsets: JsValue = offsets.into();
        call3(
            pass,
            "setBindGroup",
            &num(group.index.get() as u64),
            &native,
            &offsets,
        )?;
    }
    Ok(())
}
fn lower_upload(
    queue: &JsValue,
    job: &crate::api::resource::UploadJob,
    registration: WebGpuRegistration,
) -> Result<(), String> {
    match job.descriptor() {
        crate::api::resource::UploadDescriptor::Buffer(value) => {
            let dst = object(
                buffer(value.dst.native(), registration, "upload destination")
                    .map_err(|e| e.to_string())?,
            )
            .ok_or_else(|| "upload destination retired".to_owned())?;
            let bytes = js_sys::Uint8Array::from(value.bytes.as_ref());
            call3(
                queue,
                "writeBuffer",
                &dst,
                &num(value.dst_offset),
                &bytes.into(),
            )?;
        }
        crate::api::resource::UploadDescriptor::Texture(value) => {
            let dst = object(
                texture(value.dst.native(), registration, "upload destination")
                    .map_err(|e| e.to_string())?,
            )
            .ok_or_else(|| "upload destination retired".to_owned())?;
            let destination = texture_copy_desc(
                &dst,
                value.subresource.mip_level,
                value.subresource.base_layer,
                value.subresource.aspect,
                value.origin,
            )?;
            let layout = Object::new();
            set(&layout, "offset", &num(0))?;
            set(
                &layout,
                "bytesPerRow",
                &num(value.source_layout.bytes_per_row as u64),
            )?;
            set(
                &layout,
                "rowsPerImage",
                &num(value.source_layout.rows_per_image as u64),
            )?;
            let bytes = js_sys::Uint8Array::from(value.bytes.as_ref());
            let destination: JsValue = destination.into();
            let layout: JsValue = layout.into();
            call4(
                queue,
                "writeTexture",
                &destination,
                &bytes.into(),
                &layout,
                &extent(value.extent),
            )?;
        }
    }
    Ok(())
}

fn texture_copy_desc(
    texture: &JsValue,
    mip: u32,
    layer: u32,
    aspect: crate::api::resource::TextureAspect,
    mut origin: crate::api::resource::Origin3d,
) -> Result<Object, String> {
    // WebGPU represents the base array layer as the z coordinate of an
    // `ImageCopyTexture` origin.  Portable 3D copies carry layer zero, so this
    // addition is also the exact 3D answer.
    origin.z = origin
        .z
        .checked_add(layer)
        .ok_or_else(|| "texture array layer plus origin overflows".to_owned())?;
    let out = Object::new();
    set(&out, "texture", texture)?;
    set(&out, "mipLevel", &num(mip as u64))?;
    set(&out, "origin", &origin_obj(origin).into())?;
    let aspect = match aspect {
        crate::api::resource::TextureAspect::Color => "all",
        crate::api::resource::TextureAspect::Depth => "depth-only",
        crate::api::resource::TextureAspect::Stencil => "stencil-only",
        _ => return Err("WebGPU does not lower multi-planar texture copies".into()),
    };
    set(&out, "aspect", &JsValue::from_str(aspect))?;
    Ok(out)
}
fn origin_obj(origin: crate::api::resource::Origin3d) -> Object {
    let out = Object::new();
    let _ = set(&out, "x", &num(origin.x as u64));
    let _ = set(&out, "y", &num(origin.y as u64));
    let _ = set(&out, "z", &num(origin.z as u64));
    out
}
fn extent(value: crate::api::resource::Extent3d) -> JsValue {
    let out = Object::new();
    let _ = set(&out, "width", &num(value.width as u64));
    let _ = set(&out, "height", &num(value.height as u64));
    let _ = set(&out, "depthOrArrayLayers", &num(value.depth as u64));
    out.into()
}

fn lost(at: &'static str) -> RhiError {
    RhiError::new(RhiErrorKind::DeviceLost, "the WebGPU device is lost").at(at)
}
fn unsupported(what: &'static str) -> RhiError {
    RhiError::new(
        RhiErrorKind::Unsupported,
        format!("WebGPU command lowering does not support {what}"),
    )
    .at("WebGpuCommandSpine::preflight")
}
fn payload_name(value: &RecordedPayload) -> &'static str {
    match value {
        RecordedPayload::MeshDispatch(_) => "mesh dispatch",
        RecordedPayload::MeshIndirect(_) => "mesh indirect",
        RecordedPayload::RayTracingBegin(_)
        | RecordedPayload::RayTracingDispatch(_)
        | RecordedPayload::RayTracingEnd => "ray tracing",
        RecordedPayload::AccelerationStructure(_) => "acceleration structure",
        RecordedPayload::RasterBegin(_)
        | RecordedPayload::RasterDraw(_)
        | RecordedPayload::RasterIndirect(_)
        | RecordedPayload::RasterEnd => "raster command",
        RecordedPayload::ComputeBegin(_)
        | RecordedPayload::ComputeDispatch(_)
        | RecordedPayload::ComputeIndirect(_)
        | RecordedPayload::ComputeEnd => "compute command",
        RecordedPayload::QueryBegin { .. }
        | RecordedPayload::QueryEnd { .. }
        | RecordedPayload::TimestampWrite { .. }
        | RecordedPayload::QueryResolve(_) => "query command",
        RecordedPayload::Copy(CopyRecord::ClearTexture { .. }) => "clear texture",
        RecordedPayload::Copy(CopyRecord::Resolve(_)) => "texture resolve",
        RecordedPayload::Copy(CopyRecord::Blit(_)) => "texture blit",
        RecordedPayload::Copy(CopyRecord::ExternalImage(_)) => "external image copy",
        RecordedPayload::Upload(_) => "upload",
        RecordedPayload::Readback(_) => "readback",
        _ => "command",
    }
}

fn num(value: u64) -> JsValue {
    JsValue::from_f64(value as f64)
}
fn clear_color(value: crate::api::command::ColorClearValue) -> Object {
    let out = Object::new();
    match value {
        crate::api::command::ColorClearValue::Float([r, g, b, a]) => {
            let _ = set(&out, "r", &JsValue::from_f64(r as f64));
            let _ = set(&out, "g", &JsValue::from_f64(g as f64));
            let _ = set(&out, "b", &JsValue::from_f64(b as f64));
            let _ = set(&out, "a", &JsValue::from_f64(a as f64));
        }
        crate::api::command::ColorClearValue::Sint([r, g, b, a]) => {
            let _ = set(&out, "r", &JsValue::from_f64(r as f64));
            let _ = set(&out, "g", &JsValue::from_f64(g as f64));
            let _ = set(&out, "b", &JsValue::from_f64(b as f64));
            let _ = set(&out, "a", &JsValue::from_f64(a as f64));
        }
        crate::api::command::ColorClearValue::Uint([r, g, b, a]) => {
            let _ = set(&out, "r", &num(r as u64));
            let _ = set(&out, "g", &num(g as u64));
            let _ = set(&out, "b", &num(b as u64));
            let _ = set(&out, "a", &num(a as u64));
        }
    }
    out
}
fn set(object: &Object, name: &str, value: &JsValue) -> Result<(), String> {
    Reflect::set(object, &JsValue::from_str(name), value)
        .map_err(|e| js::message(&e))
        .and_then(|ok| {
            ok.then_some(())
                .ok_or_else(|| "WebGPU descriptor field rejected".into())
        })
}
fn set_rhi(object: &Object, name: &str, value: JsValue) -> RhiResult<()> {
    set(object, name, &value).map_err(|message| {
        RhiError::new(RhiErrorKind::BackendFailure, message).at("WebGPU render-pass descriptor")
    })
}
fn call0(receiver: &JsValue, name: &str) -> Result<JsValue, String> {
    function(receiver, name)?
        .call0(receiver)
        .map_err(|e| js::message(&e))
}
fn call1(receiver: &JsValue, name: &str, a: &JsValue) -> Result<JsValue, String> {
    function(receiver, name)?
        .call1(receiver, a)
        .map_err(|e| js::message(&e))
}
fn call2(receiver: &JsValue, name: &str, a: &JsValue, b: &JsValue) -> Result<(), String> {
    function(receiver, name)?
        .call2(receiver, a, b)
        .map(|_| ())
        .map_err(|e| js::message(&e))
}
fn call3(
    receiver: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
) -> Result<(), String> {
    function(receiver, name)?
        .call3(receiver, a, b, c)
        .map(|_| ())
        .map_err(|e| js::message(&e))
}
fn call3_value(
    receiver: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
) -> Result<JsValue, String> {
    function(receiver, name)?
        .call3(receiver, a, b, c)
        .map_err(|e| js::message(&e))
}
fn call4(
    receiver: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
    d: &JsValue,
) -> Result<(), String> {
    function(receiver, name)?
        .call4(receiver, a, b, c, d)
        .map(|_| ())
        .map_err(|e| js::message(&e))
}
fn call5(
    receiver: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
    d: &JsValue,
    e: &JsValue,
) -> Result<(), String> {
    function(receiver, name)?
        .call5(receiver, a, b, c, d, e)
        .map(|_| ())
        .map_err(|e| js::message(&e))
}
fn call6(
    receiver: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
    d: &JsValue,
    e: &JsValue,
    f: &JsValue,
) -> Result<(), String> {
    function(receiver, name)?
        .call6(receiver, a, b, c, d, e, f)
        .map(|_| ())
        .map_err(|e| js::message(&e))
}
fn function(receiver: &JsValue, name: &str) -> Result<Function, String> {
    js::property(receiver, name)
        .map_err(|e| js::message(&e))?
        .dyn_into::<Function>()
        .map_err(|e| js::message(&e))
}
fn read_mapped_bytes(buffer: &JsValue, bytes: u64) -> Option<Vec<u8>> {
    let get = function(buffer, "getMappedRange").ok()?;
    let range = get.call2(buffer, &num(0), &num(bytes)).ok()?;
    Some(js_sys::Uint8Array::new(&range).to_vec())
}
fn unmap_buffer(buffer: &JsValue) {
    if let Ok(unmap) = function(buffer, "unmap") {
        let _ = unmap.call0(buffer);
    }
}

#[cfg(test)]
mod tests {
    // Browser-native objects cannot be constructed on the host. The invariant
    // above is intentionally structural: unsupported variants are exhausted in
    // preflight before `encode`, whose first operation is createCommandEncoder.
    #[test]
    fn unsupported_payload_names_are_never_empty() {
        assert_ne!(
            super::payload_name(&crate::api::command::record::RecordedPayload::ComputeEnd),
            ""
        );
    }
}
