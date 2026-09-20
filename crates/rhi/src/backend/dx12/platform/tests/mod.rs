//! The DX12 provider's real-GPU evidence.
//!
//! These tests are not a mock conformance run and must not be read as one: they
//! call `CreateDXGIFactory2` and `D3D12CreateDevice` on whatever adapter the
//! machine actually has. `CLAUDE.md` section 4.8 is explicit that a `TestRhi` or
//! a mock cannot stand in for this, and section 9 is explicit about the reverse
//! substitution too — passing here proves that this backend's native path works
//! on this driver, not that the portable contract holds.
//!
//! What they cover, and what they deliberately do not, recorded rather than left
//! to be assumed:
//!
//! - **Copy, upload and readback move bytes on the GPU.** The command spine
//!   lowers `encode_upload`, `copy_buffer` and `encode_readback` into real
//!   command lists, `ExecuteCommandLists` runs them, and
//!   `a_real_device_moves_bytes_from_the_cpu_to_a_buffer_and_back_to_the_cpu`
//!   below compares what the GPU wrote against the pattern the CPU uploaded. That is
//!   `version-plan.md` section 4's copy/upload/readback requirement met with a
//!   read-back result rather than with an absence of errors.
//! - **Nothing is rasterized or dispatched.** `RasterBegin`/`RasterDraw` and
//!   `ComputeBegin`/`ComputeDispatch` have no lowering — the spine refuses a plan
//!   containing one rather than skipping it — so section 4's other two
//!   requirements, real headless raster and compute on Windows DX12, are **not
//!   met by this module** and no test here may be read as if they were.
//! - **A shader entry point is accepted, not compiled.** Section 19.10's verdict
//!   is exercised against a real device by
//!   `a_real_device_accepts_the_dxil_form_and_refuses_another`, and what that
//!   establishes is narrower than "a shader was compiled": Direct3D 12 has no
//!   shader-module object, so the test shows the recorded code-form fact reaching
//!   the portable verdict through a real `ID3D12Device`, the module carrying the
//!   checked-in DXIL blob unchanged, and a WGSL artifact being refused. The
//!   driver's opinion of the bytecode arrives at `CreateComputePipelineState`,
//!   which is not written — see [`crate::backend::dx12::shader`].
//!
//! Allocation is exercised below too, and it is a prerequisite for the byte
//! movement rather than a part of it.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
};

use super::provider::{Candidate, Dx12Provider};
use crate::api::binding::vocabulary::StorageAccess;
use crate::api::binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayoutDescriptor, BindingCount,
    BindingKind, BindingLimitClass, BindingResource, BindingSlot, BindingSlotId,
    BindingSupportQuery, BufferBindingAccess, SamplerKind, TextureSampleType,
};
use crate::api::command::{
    BlitFilter, BufferCopy, BufferTextureCopy, RecorderDescriptor, TextureCopy,
};
use crate::api::error::RhiErrorKind;
use crate::api::format::{TextureFormat, TextureSupportQuery};
use crate::api::identity::{DeviceInstanceId, Label, ObjectId};
use crate::api::pipeline::{ComputePipelineDescriptor, PipelineInterfaceDescriptor};
use crate::api::platform::backend::{DeviceBackend, ProviderBackend, RequestProgress};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::{DeviceRequirements, LimitKey, OptionalFeature};
use crate::api::platform::{
    AdapterId, BackendKind, DeviceLossInfo, DeviceStatus, PlatformProvider,
};
use crate::api::presentation::PresentationTarget;
use crate::api::resource::buffer::{
    Buffer, BufferDescriptor, BufferRange, BufferSupportQuery, BufferUsage,
    ResourceMemoryPreference,
};
use crate::api::resource::route::RouteQuery;
use crate::api::resource::subresource::{
    HostTexelLayout, Origin3d, TextureAspect, TextureSubresourceLayers,
};
use crate::api::resource::texture::{
    Extent3d, TextureDescriptor, TextureDimension, TextureUsage, TextureViewCompatibility,
};
use crate::api::resource::transfer::{
    BufferUploadDescriptor, ReadbackRequest, ReadbackViewData, TextureUploadDescriptor,
};
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ShaderAbiVersion, ShaderArtifact, ShaderInterface,
    ShaderRequirements, ShaderResourceRequirement, ShaderStage, ShaderStages,
};
use crate::api::submission::{
    CompletionPoint, CompletionState, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder,
};
use crate::backend::dx12::resource::Dx12Buffer;

/// A fresh provider instance identity, as host integration would mint.
fn instance() -> DeviceInstanceId {
    DeviceInstanceId::new(0x0D12_0001)
}

/// Minimal executor for the synchronous DX12 request future in these evidence tests.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {}
        }
    }
}

/// A provider over this machine's DXGI factory.
fn provider() -> Dx12Provider {
    Dx12Provider::new(instance()).expect("DXGI must be reachable on a Windows test host")
}

/// The same provider behind the portable façade host integration would build.
///
/// This is what makes the tests below evidence about the *portable* path rather
/// than only about this backend: the identity check, the request retirement and
/// the device handle all come from the portable v13 RHI layer.
fn portable_provider() -> PlatformProvider {
    PlatformProvider::new(
        BackendKind::Dx12,
        instance(),
        Box::new(provider()) as Box<dyn ProviderBackend>,
    )
}

/// Asks for a headless device satisfying nothing, which is the one request shape
/// this backend can currently honour.
fn headless_request(selection: AdapterSelection) -> DeviceRequestDescriptor {
    DeviceRequestDescriptor::new(selection, DeviceRequirements::new())
}

/// The candidate the default selection lands on, for tests that need to name one.
fn default_candidate(provider: &Dx12Provider) -> Candidate {
    provider
        .select(AdapterSelection::Default)
        .expect("a Windows host with DXGI exposes a usable adapter")
}

/// A portable device over this machine's default adapter, through the full path.
///
/// Built the way host integration builds one — the portable `PlatformProvider`
/// awaited a device — rather than by reaching for the backend directly, so a
/// test using this observes the capability facts
/// the *portable* layer would hand a caller. The capability table is filled
/// during device creation, so anything that wants to examine it has to come
/// through a created device.
fn portable_device() -> crate::api::platform::Device {
    let provider = portable_provider();
    block_on(provider.request_device(headless_request(AdapterSelection::Default)))
        .expect("a headless device request must succeed on a machine with DXGI")
}

unsafe extern "system" fn hidden_presentation_window_proc(
    hwnd: windows::Win32::Foundation::HWND,
    message: u32,
    wparam: windows::Win32::Foundation::WPARAM,
    lparam: windows::Win32::Foundation::LPARAM,
) -> windows::Win32::Foundation::LRESULT {
    unsafe {
        windows::Win32::UI::WindowsAndMessaging::DefWindowProcW(hwnd, message, wparam, lparam)
    }
}

#[test]
fn a_hidden_hwnd_frame_clears_presents_and_can_be_acquired_again() {
    use windows::Win32::Foundation::{HINSTANCE, HWND};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, RegisterClassW, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER,
        SetWindowPos, UnregisterClassW, WINDOW_EX_STYLE, WNDCLASSW, WS_OVERLAPPED,
    };
    use windows::core::w;

    struct HiddenWindow {
        hwnd: HWND,
        instance: HINSTANCE,
    }
    impl Drop for HiddenWindow {
        fn drop(&mut self) {
            unsafe {
                let _ = DestroyWindow(self.hwnd);
                let _ = UnregisterClassW(w!("FluxelDx12PresentationTest"), Some(self.instance));
            }
        }
    }
    let instance = unsafe { GetModuleHandleW(None) }.expect("module handle");
    let class = WNDCLASSW {
        hInstance: instance.into(),
        lpszClassName: w!("FluxelDx12PresentationTest"),
        lpfnWndProc: Some(hidden_presentation_window_proc),
        ..Default::default()
    };
    unsafe { RegisterClassW(&class) };
    let hwnd = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("FluxelDx12PresentationTest"),
            w!("fluxel-hidden"),
            WS_OVERLAPPED,
            0,
            0,
            64,
            64,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
    .expect("hidden HWND");
    let _window = HiddenWindow {
        hwnd,
        instance: instance.into(),
    };
    let device = portable_device();
    let native = device
        .native()
        .as_any()
        .downcast_ref::<super::device::Dx12Device>()
        .expect("DX12 device");
    let target = native.register_test_presentation_target(hwnd);
    let capabilities = device
        .presentation_capabilities(&target)
        .expect("presentation facts");
    assert_eq!(
        capabilities.present_modes(),
        &[crate::api::presentation::PresentMode::Fifo],
        "DX12 may only report modes that its Present arguments actually implement",
    );
    let config =
        crate::api::presentation::PresentationConfiguration::new(TextureFormat::Bgra8Unorm)
            .with_present_mode(crate::api::presentation::PresentMode::Fifo);
    let mut surface = block_on(device.configure_presentation(&target, &config)).expect("configure");
    let frame = block_on(surface.acquire()).expect("acquire");
    let attachment = frame.attachment();
    let scope = crate::api::command::RasterScopeDescriptor::new().with_color(
        crate::api::shader::ShaderLocation::new(0),
        crate::api::command::ColorAttachment {
            view: crate::api::command::ColorAttachmentView::Frame(attachment),
            load: crate::api::command::LoadOp::Clear(crate::api::command::ColorClearValue::Float(
                [0.0, 0.0, 0.0, 1.0],
            )),
            store: crate::api::command::StoreOp::Store,
            resolve: None,
        },
    );
    let mut recorder = device
        .create_recorder(&crate::api::command::RecorderDescriptor::new())
        .expect("recorder");
    recorder
        .begin_raster(&scope)
        .expect("raster")
        .end()
        .expect("end");
    let work = recorder.finish().expect("finish");
    let lane = device.capabilities().submission().lanes()[0].id();
    let mut plan = crate::api::submission::SubmissionPlanBuilder::new(&device);
    let point = plan.add_batch(lane, vec![work]).expect("batch");
    plan.present_after(frame, point).expect("present plan");
    let receipt = block_on(device.submit(plan.build().expect("plan"))).expect("submit");
    assert!(matches!(
        block_on(device.wait_present(receipt.presents()[0].id())),
        Ok(crate::api::presentation::PresentState::Accepted)
    ));
    assert!(matches!(
        block_on(device.wait_completion(receipt.completion())),
        Ok(crate::api::submission::CompletionState::Complete)
    ));
    // `scope` owns a FrameAttachment clone. DXGI requires every old backbuffer
    // reference to be released before ResizeBuffers, including this recorded
    // description after its submitted work is complete.
    drop(scope);
    // Flip-model DXGI permits only one HWND-associated swapchain.  Change the
    // native client size, reconfigure the *same* portable lease, then prove the
    // resized chain can again acquire and present rather than merely being
    // recreated beside the old chain.
    unsafe {
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            96,
            80,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
        )
    }
    .expect("resize hidden HWND");
    block_on(surface.reconfigure(&config)).expect("ResizeBuffers reconfigure");

    let frame = block_on(surface.acquire()).expect("acquire after resize");
    let attachment = frame.attachment();
    let scope = crate::api::command::RasterScopeDescriptor::new().with_color(
        crate::api::shader::ShaderLocation::new(0),
        crate::api::command::ColorAttachment {
            view: crate::api::command::ColorAttachmentView::Frame(attachment),
            load: crate::api::command::LoadOp::Clear(crate::api::command::ColorClearValue::Float(
                [0.0, 0.0, 0.0, 1.0],
            )),
            store: crate::api::command::StoreOp::Store,
            resolve: None,
        },
    );
    let mut recorder = device
        .create_recorder(&crate::api::command::RecorderDescriptor::new())
        .expect("recorder after resize");
    recorder
        .begin_raster(&scope)
        .expect("raster after resize")
        .end()
        .expect("end after resize");
    let work = recorder.finish().expect("finish after resize");
    let mut plan = crate::api::submission::SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(lane, vec![work])
        .expect("batch after resize");
    plan.present_after(frame, point)
        .expect("present plan after resize");
    let receipt = block_on(device.submit(plan.build().expect("plan after resize")))
        .expect("submit after resize");
    assert!(matches!(
        block_on(device.wait_present(receipt.presents()[0].id())),
        Ok(crate::api::presentation::PresentState::Accepted)
    ));
    let _again = block_on(surface.acquire()).expect("reacquire after resize and present");
}

#[test]
fn a_provider_reports_the_adapters_dxgi_lists() {
    let candidates = provider()
        .candidates()
        .expect("adapter enumeration must not fail on a machine with DXGI");

    assert!(
        !candidates.is_empty(),
        "a Windows host with DXGI exposes at least one adapter"
    );
    for candidate in &candidates {
        assert!(
            !candidate.name.is_empty(),
            "DXGI must give a non-empty adapter description"
        );
    }

    // The mapping `Explicit` selection relies on: distinct adapters must produce
    // distinct serials, or two adapters could not be told apart by the id a
    // caller holds.
    let mut serials: Vec<u64> = candidates
        .iter()
        .map(|candidate| candidate.serial)
        .collect();
    serials.sort_unstable();
    let before = serials.len();
    serials.dedup();
    assert_eq!(before, serials.len(), "adapter serials must be distinct");
}

#[test]
fn a_headless_device_is_created_on_the_default_adapter() {
    let provider = provider();
    let mut request = provider
        .request_device(&headless_request(AdapterSelection::Default))
        .expect("a headless device request must succeed on a machine with DXGI");

    let progress = request
        .poll()
        .expect("a synchronous backend must answer on the first poll");
    let RequestProgress::Ready(device) = progress else {
        panic!("the DX12 path is synchronous, so the first poll must be Ready");
    };

    assert_eq!(device.backend_kind(), BackendKind::Dx12);
    assert_eq!(
        device.status(),
        DeviceStatus::Active,
        "a freshly created device is active"
    );
    assert!(
        device.loss_info().is_none(),
        "an active device has no loss to report"
    );
    assert!(
        !device.adapter_info().name().is_empty(),
        "the device must carry the adapter it was actually created on"
    );
    assert!(
        device.poll().is_ok(),
        "polling an idle device must not fail"
    );
    assert!(
        device.wait_idle().is_ok(),
        "a device with no submitted work is idle"
    );
}

#[test]
fn the_portable_path_produces_a_real_device() {
    // The end-to-end shape: a portable `PlatformProvider` over this real DX12
    // backend and an awaited portable `Device` backed by an actual `ID3D12Device`.
    let provider = portable_provider();
    let device = block_on(provider.request_device(headless_request(AdapterSelection::Default)))
        .expect("a headless device request must succeed on a machine with DXGI");

    assert_eq!(device.backend(), BackendKind::Dx12);
    assert_eq!(device.status(), DeviceStatus::Active);
    assert!(device.loss_info().is_none());
    assert!(
        !device.adapter_info().name().is_empty(),
        "the portable device must carry the adapter its native device was created on"
    );
    assert!(device.poll().is_ok());
    assert!(block_on(device.wait_idle()).is_ok());
}

/// The checked-in DXIL blob, summarised rather than only embedded.
///
/// The bytes come from `scripts/tests/data/dxil/fill_cs.dxil`, whose README records
/// the `dxc` version, the command line and the hash. **No test runs `dxc`**: a test
/// that generated its own input could not tell "the driver accepted this program"
/// apart from "this machine's toolchain produced something", and the fixture would
/// change under the test without anyone deciding to.
const FILL_CS_DXIL: &[u8] =
    include_bytes!("../../../../../../../scripts/tests/data/dxil/fill_cs.dxil");

/// A compute artifact over the DXIL fixture: the closure's payload.
fn fill_cs_artifact(code: crate::api::shader::ShaderCode) -> ShaderArtifact {
    ShaderArtifact::new(
        ShaderStage::Compute,
        "main",
        code,
        ShaderAbiVersion { major: 1, minor: 0 },
        // `fill_cs.hlsl` declares `RWByteAddressBuffer output : register(u0)`.
        // ABI 1.0 maps that to group 0 / slot 0, so the PSO test
        // below exercises the same TablePlan/register mapping the fixture needs.
        ShaderInterface::new().with_resource(ShaderResourceRequirement {
            group: crate::api::binding::BindGroupIndex::new(0),
            slot: BindingSlotId::new(0),
            kind: BindingKind::StorageBuffer {
                access: BufferBindingAccess::ReadWrite,
                min_size: 4,
            },
            count: BindingCount::One,
        }),
        ShaderRequirements::new(),
        ArtifactHash([0x51; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

/// The driver's compute-compiler verdict over the checked-in DXIL fixture.
///
/// DX12 shader-module creation preserves DXIL, while
/// `CreateComputePipelineState` compiles and validates it. This reaches that
/// latter call through the portable device verb with the `u0, space0` interface
/// declared by the fixture.
#[test]
fn a_real_device_creates_a_compute_pipeline_from_checked_in_dxil() {
    let device = portable_device();
    let shader = block_on(device.create_shader(&fill_cs_artifact(
        crate::api::shader::ShaderCode::Dxil(Arc::from(FILL_CS_DXIL)),
    )))
    .expect("the checked-in compute artifact must create its DX12 shader module");
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
            BindingSlotId::new(0),
            ShaderStages::COMPUTE,
            BindingKind::StorageBuffer {
                access: BufferBindingAccess::ReadWrite,
                min_size: 4,
            },
        )]))
        .expect("the fixture's u0 storage binding must have a portable layout");
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout]))
        .expect("the fixture's one group layout must make a pipeline interface");

    let pipeline = block_on(
        device.create_compute_pipeline(&ComputePipelineDescriptor::new(shader, interface)),
    )
    .expect("CreateComputePipelineState must accept the checked-in fill_cs DXIL");
    let native = pipeline
        .native()
        .as_any()
        .downcast_ref::<crate::backend::dx12::pipeline::Dx12ComputePipeline>()
        .expect("the portable pipeline must retain its DX12 PSO");
    // Reaching both accessors proves the native objects are retained by the
    // portable handle for later command lowering, rather than being temporaries
    // that only survived the creation call.
    let _root_signature = native.root_signature();
    let _pipeline_state = native.pipeline_state();
    assert_eq!(native.view_root_parameter(0), Some(0));
}

/// A real dispatch writes a storage buffer and the copy path returns the bytes.
#[test]
fn a_real_compute_dispatch_writes_a_storage_buffer() {
    const WORDS: u64 = 8;
    const SIZE: u64 = WORDS * 4;

    let device = portable_device();
    let shader = block_on(device.create_shader(&fill_cs_artifact(
        crate::api::shader::ShaderCode::Dxil(Arc::from(FILL_CS_DXIL)),
    )))
    .expect("the checked-in compute artifact must create its DX12 shader module");
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
            BindingSlotId::new(0),
            ShaderStages::COMPUTE,
            BindingKind::StorageBuffer {
                access: BufferBindingAccess::ReadWrite,
                min_size: SIZE,
            },
        )]))
        .expect("u0 must have a portable storage-buffer layout");
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))
        .expect("the storage-buffer group must make a pipeline interface");
    let pipeline = block_on(
        device.create_compute_pipeline(&ComputePipelineDescriptor::new(shader, interface)),
    )
    .expect("the checked-in DXIL must create a compute PSO");
    let output = device
        .create_buffer(&BufferDescriptor::new(
            SIZE,
            BufferUsage::STORAGE.union(BufferUsage::COPY_SRC),
        ))
        .expect("the compute output must be usable as UAV and readback source");
    let group = device
        .create_bind_group(
            &BindGroupDescriptor::new(layout).with_entry(BindGroupEntry::new(
                BindingSlotId::new(0),
                BindingResource::Buffer(crate::api::resource::BufferBinding::new(
                    output.clone(),
                    BufferRange::new(0, SIZE),
                )),
            )),
        )
        .expect("the output buffer must lower to a DX12 UAV descriptor");

    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("a real device opens a recorder");
    {
        let mut compute = recorder
            .begin_compute(&crate::api::command::ComputeScopeDescriptor::new())
            .expect("the device reports compute support");
        compute
            .set_pipeline(&pipeline)
            .expect("the pipeline belongs to this device");
        compute
            .set_bind_group(BindGroupIndex::new(0), &group, &[])
            .expect("the bind group matches the pipeline interface");
        compute
            .dispatch(1, 1, 1)
            .expect("one workgroup is within the device limit");
        compute.end().expect("the compute scope is balanced");
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some("dx12 compute output".to_string())),
            src: output,
            range: BufferRange::new(0, SIZE),
        })
        .expect("the output carries COPY_SRC");
    let work = recorder.finish().expect("the recording is complete");

    let mut builder = SubmissionPlanBuilder::new(&device);
    let point = builder
        .add_batch(copy_lane(&device), vec![work])
        .expect("the direct lane accepts compute and copy work");
    let receipt = block_on(device.submit(builder.build().expect("one batch is acyclic")))
        .expect("the dispatch and readback lower before commit");
    let completion = receipt
        .completion_for(point)
        .expect("the receipt knows its batch point");
    assert!(matches!(
        settle(&device, completion),
        CompletionState::Complete
    ));

    let view = ticket
        .try_read()
        .expect("the completed dispatch readback is not terminal")
        .expect("the completion drain publishes the bytes");
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("a buffer request must return buffer bytes");
    };
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("one u32")))
        .collect();
    assert_eq!(words, (0..WORDS as u32).collect::<Vec<_>>());
    println!(
        "dx12 compute evidence: adapter={:?} workgroups=(1,1,1) words={words:?}",
        device.adapter_info().name(),
    );
}

/// The acceptance verdict, on the device this machine actually has.
///
/// This is the real-GPU half of the shader step, and what it establishes is
/// narrower than the paragraph that follows might suggest. The accepted code forms
/// are a recorded device fact, and for Direct3D 12 that record is *structural* —
/// there is no `CheckFeatureSupport` question whose answer could be otherwise, and
/// [`super::facts`] records the form for that reason rather than from a
/// probe. So this test does not check a driver's opinion. What it checks is that
/// the fact reaches the portable verdict through a real device: the same
/// `EnabledCapabilities` a caller holds, built from a real `ID3D12Device` by the
/// production path, answers `Accepted` for a DXIL artifact and refuses a WGSL one.
///
/// The WGSL case is the one that can fail if the shortcut section 6.3 forbids is
/// ever reintroduced: a device that answered "a DX12 backend consumes DXIL" would
/// look identical here, which is why the refusal is asserted alongside.
#[test]
fn a_real_device_accepts_the_dxil_form_and_refuses_another() {
    let device = portable_device();

    assert_eq!(
        device.backend(),
        BackendKind::Dx12,
        "this evidence is about the Direct3D 12 backend"
    );

    let dxil = block_on(device.create_shader(&fill_cs_artifact(
        crate::api::shader::ShaderCode::Dxil(Arc::from(FILL_CS_DXIL)),
    )))
    .expect("a real DX12 device consumes DXIL, which its own facts record");

    assert_eq!(dxil.stage(), ShaderStage::Compute);
    assert_eq!(dxil.artifact().entry_point, "main");

    // The seam, on real hardware: the portable handle hands back this backend's own
    // object, holding the very bytes the fixture supplies. This is what the pipeline
    // lowering will hand to `D3D12_SHADER_BYTECODE`.
    let held = dxil
        .native()
        .as_any()
        .downcast_ref::<crate::backend::dx12::shader::Dx12ShaderModule>()
        .expect("a DX12 device's module is the DX12 backend's own type");
    assert_eq!(
        held.dxil(),
        Some(FILL_CS_DXIL),
        "the module must carry the fixture's bytes unchanged"
    );

    // The other direction. A `ShaderCode::Wgsl` artifact is well-formed — the
    // portable rules pass — and this device still cannot consume it, so the refusal
    // is `Unsupported` and it is the *device's* answer rather than a complaint about
    // the artifact.
    let error = block_on(device.create_shader(&fill_cs_artifact(
        crate::api::shader::ShaderCode::Wgsl(Arc::from(
            "@compute @workgroup_size(8, 8, 1) fn main() {}",
        )),
    )))
    .expect_err("a DX12 device has no WGSL compiler, and says so");
    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert_eq!(error.operation(), Some("Device::create_shader"));
    assert!(
        error.message().contains("UnsupportedCodeFormat"),
        "the verdict names the reason rather than only the refusal: {}",
        error.message()
    );
}

#[test]
fn a_loss_recorded_on_the_native_device_is_terminal_and_stable() {
    let provider = provider();
    let native = provider
        .create_native(AdapterSelection::Default)
        .expect("a headless device must be creatable");

    assert_eq!(native.status(), DeviceStatus::Active);
    assert!(native.loss_info().is_none());

    native.mark_lost(DeviceLossInfo::new("DXGI_ERROR_DEVICE_REMOVED".to_string()));

    assert_eq!(native.status(), DeviceStatus::Lost);
    assert_eq!(
        native.status(),
        DeviceStatus::Lost,
        "loss is terminal: a second read must not revive it"
    );
    assert_eq!(
        native.loss_info().map(|info| info.message().to_string()),
        Some("DXGI_ERROR_DEVICE_REMOVED".to_string()),
        "section 6.5 makes the summary stable rather than a one-shot notification"
    );
}

#[test]
fn adapter_enumeration_publishes_complete_snapshots() {
    let adapters = provider()
        .enumerate_adapters()
        .expect("DX12 adapter enumeration must probe complete capability snapshots")
        .expect("DX12 exposes explicit adapter enumeration");
    assert!(
        !adapters.is_empty(),
        "a machine that created the test provider must publish at least one usable adapter"
    );
    for adapter in adapters {
        assert!(
            adapter
                .available_capabilities()
                .limit(LimitKey::MaxBufferSize)
                .is_some(),
            "every published adapter must carry its probed capability snapshot"
        );
    }
}

#[test]
fn presentation_support_is_refused_rather_than_answered_no() {
    let provider = provider();
    let candidate = default_candidate(&provider);
    let adapter = AdapterId::new(instance().as_u64(), candidate.serial);
    let target = PresentationTarget::new(ObjectId::new(0x9001));

    let error = provider
        .supports_presentation(adapter, &target)
        .expect_err("there is no channel from an ObjectId to an HWND, so this cannot be answered");

    assert_eq!(
        error.kind(),
        RhiErrorKind::Unsupported,
        "Ok(false) would turn 'no way to ask' into 'the hardware says no'"
    );
}

#[test]
fn a_presentation_requirement_is_refused_rather_than_dropped() {
    let descriptor = headless_request(AdapterSelection::Default)
        .require_presentation_target(PresentationTarget::new(ObjectId::new(0x9002)));

    // `.err().expect(..)` rather than `expect_err`: the private backend request
    // result owns a trait object that is deliberately not `Debug`. A backend
    // object has no portable rendering, and printing one would be the native leak
    // the seam exists to prevent.
    let error = provider().request_device(&descriptor).err().expect(
        "a device required to present cannot be created while presentation cannot be resolved",
    );

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
}

#[test]
fn a_device_requirement_is_refused_rather_than_dropped() {
    let requirements =
        DeviceRequirements::new().require_limit_at_least(LimitKey::MaxBufferSize, u64::MAX);
    let descriptor = DeviceRequestDescriptor::new(AdapterSelection::Default, requirements);

    let error = provider()
        .request_device(&descriptor)
        .err()
        .expect("a requirement above the probed device limit must be refused");

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert!(
        error.message().contains("MaxBufferSize"),
        "the refusal must name the unmet contract: {}",
        error.message()
    );
}

#[test]
fn the_high_performance_preference_never_picks_the_software_adapter() {
    // A preference is not a guarantee — section 5.6 says so — but it must at
    // least not select the opposite of what was asked when a hardware adapter
    // exists. Which adapter that is depends on the machine, so this asserts the
    // property rather than a name.
    let provider = provider();
    let candidate = provider
        .select(AdapterSelection::PreferHighPerformance)
        .expect("a Windows host exposes a usable adapter");
    let hardware_exists = provider
        .candidates()
        .expect("enumeration must not fail")
        .iter()
        .any(|other| !other.software);

    assert!(
        !hardware_exists || !candidate.software,
        "a hardware adapter exists, so the high-performance preference must not resolve to WARP"
    );
}

#[test]
fn what_the_machine_actually_reported() {
    // Not an assertion about the hardware: `CLAUDE.md` section 4.8 requires the
    // exact backend, adapter and vendor/device pair to be recorded with the
    // evidence, and section 9 forbids recording an observation that was not made.
    // Run with `--nocapture` and this is what the run above used.
    let provider = provider();
    for candidate in provider.candidates().expect("enumeration must not fail") {
        println!(
            "dx12 evidence: name={:?} vendor=0x{:04x} device=0x{:04x} \
             dedicated_video_memory={} software={}",
            candidate.name,
            candidate.vendor,
            candidate.device,
            candidate.dedicated_video_memory,
            candidate.software
        );
    }
    assert_eq!(provider.instance(), instance());
}

/// The enumerated facts answer the one table that must be complete.
///
/// This is the test that closes the landmine `facts`' module documentation names.
/// Before the enumeration landed, `buffer_support` panicked for every query on a
/// real DX12 device, because the table was empty and its key space is one a
/// backend can walk in full. The assertions below therefore do two separate
/// things: they check the *answers*, and — by the mere fact of returning — they
/// check that a query against a real device no longer panics.
#[test]
fn a_real_device_answers_every_buffer_usage_combination() {
    let device = portable_device();

    for usage in BufferUsage::all() {
        let query = BufferSupportQuery::new(usage);
        let support = device.capabilities().buffer_support(&query);

        if usage.is_empty() {
            // Section 12.3 refuses to create a buffer with no usage bit at all,
            // so the honest answer is a refusal rather than a supported entry
            // with a ceiling nobody can reach.
            assert!(
                !support.is_supported(),
                "an empty usage set has no legal operation and must be refused"
            );
            continue;
        }

        assert!(
            support.is_supported(),
            "Direct3D 12 expresses {usage} as resource states rather than as creation \
             flags, so it must be reported creatable"
        );
        assert!(
            support.limits().is_some_and(|limits| limits.max_size() > 0),
            "a supported answer must carry a non-zero ceiling: {usage}"
        );
    }
}

/// The three optional features Direct3D 12 answers structurally are reported.
///
/// They are recorded without a probe, and the reason is at each call site in
/// `facts`. What matters for the portable contract is the consequence: a caller
/// reading this device's enabled capabilities must not be told that a D3D12
/// device cannot dispatch, cannot filter anisotropically, or cannot use binding
/// arrays. The lane assertion is the other half of the same fact — section 7.2's
/// base guarantee relates the `Compute` feature to a lane accepting compute work,
/// and the two are now consistent rather than deliberately under-reported.
#[test]
fn a_real_device_reports_the_features_direct3d_12_mandates() {
    let device = portable_device();
    let capabilities = device.capabilities();

    for feature in [
        OptionalFeature::Compute,
        OptionalFeature::SamplerAnisotropy,
        OptionalFeature::BindingArrays,
    ] {
        assert!(
            capabilities.supports_feature(feature),
            "{feature:?} is a property of Direct3D 12 rather than a driver's choice"
        );
    }

    let accepts_compute = capabilities
        .submission()
        .lanes()
        .iter()
        .any(|lane| lane.domains().contains(LaneWorkDomains::COMPUTE));

    assert!(
        accepts_compute,
        "section 7.2's base guarantee ties a compute-accepting lane to the Compute feature, \
         and this device reports both"
    );
}

/// A format table that covers the portable set, minus the two it cannot name.
///
/// The two omissions are the assertion worth reading: `Depth24Plus` and
/// `Depth24PlusStencil8` explicitly permit a driver to choose a bit layout, so
/// there is no single DXGI format that is the answer and `facts` returns none.
/// The portable accessor answers `Option`, so "not asked" and "asked and refused"
/// stay distinguishable — and this test pins which of the two a caller sees.
#[test]
fn a_real_device_reports_what_each_namable_format_can_do() {
    let device = portable_device();
    let capabilities = device.capabilities();

    // The two permitted-layout formats are the only ones this backend declines to
    // answer, and that is asserted as an exact set rather than one format at a
    // time: a third silent omission is the failure mode worth catching, and
    // checking only the two known names would not catch it.
    let unanswered: Vec<TextureFormat> = TextureFormat::all()
        .filter(|format| capabilities.format(*format).is_none())
        .collect();

    assert_eq!(
        unanswered,
        vec![
            TextureFormat::Depth24Plus,
            TextureFormat::Depth24PlusStencil8
        ],
        "exactly the two formats whose bit layout Direct3D 12 leaves to the driver \
         may go unanswered; anything else absent is a hole in the table"
    );

    // One answered format, read end to end: `None` above is only meaningful if
    // `Some` carries real facts, and `Rgba8Unorm` is the format every backend
    // must support for a storage write or the portable P0 set is not viable.
    let sampled = capabilities
        .format(TextureFormat::Rgba8Unorm)
        .expect("Rgba8Unorm is a DXGI format on every device");

    assert!(
        sampled.storage_access().supports(StorageAccess::ReadOnly),
        "an Rgba8Unorm texture is readable as a storage resource"
    );
}

/// The two limits whose Direct3D 12 source is unambiguous are recorded.
///
/// Not a claim that the limit table is complete — it is not, and `facts` records
/// which seven keys are still unrecorded. This pins the two that are, so that a
/// later mapping change cannot quietly drop them.
#[test]
fn a_real_device_reports_the_two_limits_it_can_ground() {
    let device = portable_device();
    let capabilities = device.capabilities();

    assert_eq!(
        capabilities.limit(LimitKey::MaxSamplerAnisotropy),
        Some(16),
        "the D3D12 sampler descriptor clamps MaxAnisotropy to 1..=16"
    );

    let max_buffer = capabilities
        .limit(LimitKey::MaxBufferSize)
        .expect("the resource address space is always reported");
    assert!(
        max_buffer >= (1 << 32),
        "a D3D12 device addresses at least 32 bits per resource; got {max_buffer}"
    );

    // Direct3D 12 defines no cap specific to a storage binding, so both keys are
    // answered by the one address-space reading. Asserting they agree is what
    // keeps them from drifting into two entries that disagree; the value itself
    // is a reading and is printed by the evidence test instead.
    assert_eq!(
        capabilities.limit(LimitKey::MaxStorageBufferBindingSize),
        Some(max_buffer),
        "a storage binding is bounded by the same address space as any other \
         resource, because Direct3D 12 states no separate cap"
    );
}

/// The routes a real device answers, including the one it must refuse.
///
/// Section 9.4 makes a route answer final: `Unsupported` means the command returns
/// `Unsupported` rather than the backend quietly lowering a blit into a shader or
/// a copy into a CPU round-trip. So the blit half of this test is the load-bearing
/// one — Direct3D 12 has no filtered or scaled blit, and a route table that
/// answered `Supported` here would be promising a lowering the specification
/// forbids.
#[test]
fn a_real_device_answers_the_routes_a_renderer_asks() {
    let device = portable_device();
    let capabilities = device.capabilities();

    // The buffer copy is the one route with no format in its key, and the device
    // imposes nothing on it: `CopyBufferRegion` takes byte offsets and a byte
    // count.
    let buffer_copy = capabilities.route(&RouteQuery::BufferToBuffer);
    assert!(
        buffer_copy.is_supported(),
        "a D3D12 device copies one buffer to another"
    );
    let layout = buffer_copy
        .capabilities()
        .and_then(|capabilities| capabilities.buffer_copy_layout())
        .expect("a supported buffer copy states its alignment");
    assert_eq!((layout.offset_alignment(), layout.size_alignment()), (1, 1));

    // A buffer-texture copy is a placed footprint, so the two alignment numbers
    // are the API's placement constants rather than a driver preference.
    for query in [
        RouteQuery::BufferToTexture {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            aspect: TextureAspect::Color,
        },
        RouteQuery::TextureToBuffer {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            aspect: TextureAspect::Color,
        },
    ] {
        let support = capabilities.route(&query);
        assert!(
            support.is_supported(),
            "{query:?} is a copy Direct3D 12 offers"
        );

        let texel = support
            .capabilities()
            .and_then(|capabilities| capabilities.texel_copy_layout())
            .expect("a supported texel copy states its alignment");
        assert_eq!(texel.buffer_offset_alignment(), 512);
        assert_eq!(texel.bytes_per_row_alignment(), 256);
    }

    // A texture-to-texture copy does not convert. The same key with a different
    // format on one side is therefore not a route this device lacks — it is not a
    // copy the API offers at all, and the negative is the honest answer.
    let copy = |src_format, dst_format| RouteQuery::TextureToTexture {
        src_dimension: TextureDimension::D2,
        src_format,
        src_aspect: TextureAspect::Color,
        src_sample_count: 1,
        dst_dimension: TextureDimension::D2,
        dst_format,
        dst_aspect: TextureAspect::Color,
        dst_sample_count: 1,
    };
    assert!(
        capabilities
            .route(&copy(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm))
            .is_supported()
    );
    assert!(
        !capabilities
            .route(&copy(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Uint))
            .is_supported(),
        "CopyTextureRegion moves texels between like formats; it is not a converter"
    );

    // The native adapter can expose ResolveSubresource, but this correctness
    // baseline has no command lowering for it yet. Capability is an end-to-end
    // promise, not a raw CheckFeatureSupport mirror.
    assert!(
        !capabilities
            .route(&RouteQuery::Resolve {
                format: TextureFormat::Rgba8Unorm,
                src_sample_count: 4,
            })
            .is_supported(),
        "resolve stays unavailable until Dx12CommandSpine lowers it"
    );
    assert!(
        !capabilities
            .route(&RouteQuery::Resolve {
                format: TextureFormat::Rgba8Unorm,
                src_sample_count: 1,
            })
            .is_supported(),
        "a single-sampled source has nothing to resolve"
    );

    // And the refusal. Direct3D 12 has CopyBufferRegion, CopyTextureRegion,
    // CopyResource, CopyTiles and ResolveSubresource, and no filtered blit at any
    // of them.
    for filter in [BlitFilter::Nearest, BlitFilter::Linear] {
        for (src, dst) in [
            (TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm),
            (TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Uint),
        ] {
            assert!(
                !capabilities
                    .route(&RouteQuery::Blit {
                        src_dimension: TextureDimension::D2,
                        src_format: src,
                        dst_dimension: TextureDimension::D2,
                        dst_format: dst,
                        filter,
                    })
                    .is_supported(),
                "there is no {filter:?} blit for Direct3D 12 to lower onto"
            );
        }
    }
}

/// The route walk reaches every legal key rather than a sample of them.
///
/// Asserted as a count rather than trusted, for the reason the texture walk's
/// equivalent test gives: a walk that silently stopped early would otherwise pass
/// every per-key assertion in the test above.
#[test]
fn every_legal_route_key_is_recorded_rather_than_left_to_the_negative() {
    let device = portable_device();
    let capabilities = device.capabilities();

    let mut legal = 0usize;
    let mut refused = 0usize;

    // Three formats, one per aspect shape section 8.1's P0 set contains: a colour
    // format, a depth-only format, and a depth-stencil format. The other two
    // depth formats are left out deliberately — `Depth24Plus` and
    // `Depth24PlusStencil8` have no single DXGI format, so this backend has no
    // route table for them at all, and including one here would be asserting about
    // an enumeration that was never made.
    for format in [
        TextureFormat::Rgba8Unorm,
        TextureFormat::Depth32Float,
        TextureFormat::Depth32FloatStencil8,
    ] {
        for dimension in [
            TextureDimension::D1,
            TextureDimension::D2,
            TextureDimension::D3,
        ] {
            for aspect in [
                TextureAspect::Color,
                TextureAspect::Depth,
                TextureAspect::Stencil,
            ] {
                let mut has_plane = false;
                for query in [
                    RouteQuery::BufferToTexture {
                        dimension,
                        format,
                        aspect,
                    },
                    RouteQuery::TextureToBuffer {
                        dimension,
                        format,
                        aspect,
                    },
                ] {
                    if capabilities.route(&query).is_supported() {
                        legal += 1;
                        has_plane = true;
                    } else {
                        refused += 1;
                    }
                }

                // The same format on both sides, which is the only shape a copy
                // covers, must agree with the two directions above: the plane a
                // copy can cover is the plane the format has. Asserting the
                // agreement rather than the value is what makes this a statement
                // about the walk — a key the walk skipped in one table and not the
                // other is exactly the silent refusal this test exists to catch.
                let query = RouteQuery::TextureToTexture {
                    src_dimension: dimension,
                    src_format: format,
                    src_aspect: aspect,
                    src_sample_count: 1,
                    dst_dimension: dimension,
                    dst_format: format,
                    dst_aspect: aspect,
                    dst_sample_count: 1,
                };
                assert_eq!(
                    capabilities.route(&query).is_supported(),
                    has_plane,
                    "a same-format copy covers the plane the format has: {query:?}"
                );
                assert_eq!(
                    capabilities
                        .route(&RouteQuery::BufferToTexture {
                            dimension,
                            format,
                            aspect,
                        })
                        .is_supported(),
                    capabilities
                        .route(&RouteQuery::TextureToBuffer {
                            dimension,
                            format,
                            aspect,
                        })
                        .is_supported(),
                    "the two buffer-texture directions are one path, so they cannot disagree"
                );
            }
        }
    }

    // The count the rule requires, derived rather than observed. Each of the three
    // formats has its own planes — colour for one, depth for the next, depth and
    // stencil for the third — over three dimensions and two directions, so
    // `3 * (1 + 1 + 2) * 2` keys are legal and the other thirty of the fifty-four
    // walked name a plane their format does not have. Asserted as a count because a
    // walk that silently stopped early would satisfy every per-key assertion above.
    assert_eq!(
        legal, 24,
        "the walk must reach every legal key, not merely some"
    );
    assert_eq!(
        refused, 30,
        "a plane the format does not have is not a refusal by the device"
    );
}

/// Records what this machine's device enumerated, for the evidence binding.
///
/// Run with `--nocapture`. `CLAUDE.md` section 9 forbids recording an observation
/// that was not made, so this prints rather than asserts a hardware-specific
/// value; the assertions that do bind are in the tests above.
#[test]
fn what_the_capability_enumeration_actually_reported() {
    let device = portable_device();
    let capabilities = device.capabilities();

    println!(
        "dx12 capability evidence: backend={:?} adapter={:?} id={:?} fingerprint={:?}",
        device.backend(),
        device.adapter_info().name(),
        capabilities.compatibility_id(),
        capabilities.fingerprint(),
    );

    for format in TextureFormat::all() {
        let Some(facts) = capabilities.format(format) else {
            continue;
        };
        println!(
            "dx12 format {format:?}: storage read={} write={} read_write={}",
            facts.storage_access().supports(StorageAccess::ReadOnly),
            facts.storage_access().supports(StorageAccess::WriteOnly),
            facts.storage_access().supports(StorageAccess::ReadWrite),
        );
    }

    println!(
        "dx12 limits: max_buffer_size={:?} max_sampler_anisotropy={:?}",
        capabilities.limit(LimitKey::MaxBufferSize),
        capabilities.limit(LimitKey::MaxSamplerAnisotropy),
    );

    let buffer_to_buffer = capabilities.route(&RouteQuery::BufferToBuffer);
    println!(
        "dx12 route buffer->buffer: supported={} buffer_layout={:?}",
        buffer_to_buffer.is_supported(),
        buffer_to_buffer.capabilities().map(|route| route
            .buffer_copy_layout()
            .map(|layout| (layout.offset_alignment(), layout.size_alignment()))),
    );

    let buffer_to_texture = capabilities.route(&RouteQuery::BufferToTexture {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        aspect: TextureAspect::Color,
    });
    println!(
        "dx12 route buffer->texture rgba8 D2 color: supported={} texel_layout={:?}",
        buffer_to_texture.is_supported(),
        buffer_to_texture
            .capabilities()
            .map(|route| route.texel_copy_layout().map(|layout| (
                layout.buffer_offset_alignment(),
                layout.bytes_per_row_alignment()
            ))),
    );

    for sample_count in [1u32, 2, 4, 8] {
        println!(
            "dx12 route resolve rgba8 from {sample_count} samples: supported={}",
            capabilities
                .route(&RouteQuery::Resolve {
                    format: TextureFormat::Rgba8Unorm,
                    src_sample_count: sample_count,
                })
                .is_supported()
        );
    }

    for filter in [BlitFilter::Nearest, BlitFilter::Linear] {
        println!(
            "dx12 route blit rgba8 D2 -> rgba8 D2 {filter:?}: supported={}",
            capabilities
                .route(&RouteQuery::Blit {
                    src_dimension: TextureDimension::D2,
                    src_format: TextureFormat::Rgba8Unorm,
                    dst_dimension: TextureDimension::D2,
                    dst_format: TextureFormat::Rgba8Unorm,
                    filter,
                })
                .is_supported()
        );
    }
}

/// A real device answers the texture questions a renderer actually asks.
///
/// The assertions are chosen to be about Direct3D 12 rather than about this
/// machine, so they hold on any conforming device: a format's own attachment
/// role is a property of the format, and the portable layer's two depth formats
/// differ in exactly one way that matters here. Anything that varies by driver —
/// quality levels at sixteen samples, `TEXTURE3D` on a compressed format — is
/// printed by the evidence test below rather than asserted.
#[test]
fn a_real_device_answers_the_texture_questions_a_renderer_asks() {
    let device = portable_device();
    let capabilities = device.capabilities();

    let queried = |usage, compatibility| {
        let query =
            TextureSupportQuery::new(TextureDimension::D2, TextureFormat::Rgba8Unorm, usage, 1)
                .with_view_compatibility(compatibility);
        capabilities.texture_support(&query).is_supported()
    };

    assert!(
        queried(TextureUsage::SAMPLED, TextureViewCompatibility::NONE),
        "Rgba8Unorm is the one format every backend must be able to sample"
    );
    assert!(
        queried(
            TextureUsage::COLOR_ATTACHMENT,
            TextureViewCompatibility::NONE
        ),
        "Rgba8Unorm is the format a color attachment is required to support"
    );
    assert!(
        queried(TextureUsage::COPY_DST, TextureViewCompatibility::NONE),
        "an upload target is the minimum a renderer needs, and Direct3D 12 \
         expresses copy as a resource state rather than as a per-format bit"
    );

    // The empty usage set is deliberately not asserted here. The portable
    // texture vocabulary has no way to spell it outside `TextureUsage::all()`,
    // whose walk starts at the empty mask, so a direct query for it is not a
    // question a caller can ask; the walk below reaches it and checks it there.
    assert!(
        !queried(
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
            TextureViewCompatibility::NONE
        ),
        "a color format is not a depth attachment, and reporting it as one would \
         let a caller build a depth pass that cannot work"
    );

    // The cube intent is a creation-time fact, so it is a different key rather
    // than a different answer to the same one: Direct3D 12 states cube support
    // with its own bit, and for Rgba8Unorm it agrees with every other backend.
    assert!(
        queried(TextureUsage::SAMPLED, TextureViewCompatibility::CUBE),
        "Rgba8Unorm in two dimensions is cube-capable on any Direct3D 12 device"
    );

    // The two portable depth formats are the exception the format table already
    // records, seen from the texture side: one lowers to a single DXGI format
    // and the other does not, so one can be asked about and the other cannot.
    let depth32 = TextureSupportQuery::new(
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        1,
    );
    assert!(
        capabilities.texture_support(&depth32).is_supported(),
        "Depth32Float lowers to one DXGI format and is the depth attachment P0 \
         renderers use"
    );

    // The dimension cases above all ask about two dimensions. A walk that covered
    // two of the three and stopped would pass every assertion so far, so the other
    // two are bound here. The extents asserted are the Direct3D 12 requirements
    // `D3D12_REQ_TEXTURE{1,2}D_*_DIMENSION` and `D3D12_REQ_TEXTURE3D_*` state,
    // which are also the API maxima, so they are structural rather than readings
    // of this adapter.
    let one = capabilities.texture_support(&TextureSupportQuery::new(
        TextureDimension::D1,
        TextureFormat::Rgba8Unorm,
        TextureUsage::SAMPLED,
        1,
    ));
    assert!(
        one.is_supported(),
        "a one-dimensional texture is a separate key, not a separate answer"
    );
    assert_eq!(
        one.limits()
            .expect("a supported answer carries its maxima")
            .max_extent()
            .width,
        16384,
        "Direct3D 12 requires 16384 texels in the one dimension a 1D texture has"
    );

    let three = capabilities.texture_support(&TextureSupportQuery::new(
        TextureDimension::D3,
        TextureFormat::Rgba8Unorm,
        TextureUsage::SAMPLED,
        1,
    ));
    assert!(
        three.is_supported(),
        "a three-dimensional texture is a separate key, not a separate answer"
    );
    let three_limits = three
        .limits()
        .expect("a supported answer carries its maxima");
    assert_eq!(
        three_limits.max_extent().depth,
        2048,
        "Direct3D 12 requires 2048 texels along the third axis, which is smaller \
         than the 16384 the first two allow"
    );
    assert_eq!(
        three_limits.max_array_layers(),
        1,
        "a volume texture has no array of layers to index"
    );

    // Native MSAA facts are probed, but the capability is deliberately withheld
    // until attachment and resolve lowering are complete. Capability is an
    // end-to-end promise, not merely an allocation fact.
    let multisampled = |sample_count| {
        capabilities
            .texture_support(&TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT,
                sample_count,
            ))
            .is_supported()
    };
    assert!(multisampled(1));
    assert!(!multisampled(2));
    assert!(!multisampled(4));
}

/// Every legal usage combination is recorded rather than left to the negative.
///
/// The texture key space is not one a backend can walk in full, so an absent key
/// answers the negative rather than panicking — which means this test cannot
/// detect a hole the way the buffer one does, because a hole and a refusal are
/// the same value. What it checks instead is the obligation that replaces "fill
/// everything": the walk reaches all sixty-four masks, and every one of them that
/// is *legal* for a color format comes back supported. A legal key left
/// unrecorded would tell a caller that a texture the device can create cannot be
/// created, and that failure is silent precisely because a negative is a
/// well-formed answer.
///
/// The depth bit is excluded, and the exclusion is part of the claim rather than
/// a workaround: `Rgba8Unorm` is a color format, so a depth-stencil usage on it
/// has no legal meaning and refusing it is correct. The earlier test asserts that
/// refusal directly; this one asserts its complement.
#[test]
fn every_legal_usage_combination_is_recorded_rather_than_left_to_the_negative() {
    let device = portable_device();
    let capabilities = device.capabilities();

    let mut legal_seen = 0usize;

    for usage in TextureUsage::all() {
        let query =
            TextureSupportQuery::new(TextureDimension::D2, TextureFormat::Rgba8Unorm, usage, 1);
        let support = capabilities.texture_support(&query);

        if usage.is_empty() {
            assert!(
                !support.is_supported(),
                "an empty usage set has no legal operation and must be refused"
            );
            continue;
        }

        if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
            assert!(
                !support.is_supported(),
                "a color format cannot be a depth-stencil attachment, so every mask \
                 carrying that bit must be refused: {usage}"
            );
            continue;
        }

        legal_seen += 1;
        assert!(
            support.is_supported(),
            "Rgba8Unorm in two dimensions supports {usage}, so leaving the key \
             unrecorded would refuse a texture the device can create"
        );

        let limits = support
            .limits()
            .expect("a supported answer carries its maxima");
        assert!(
            limits.max_extent().width > 0 && limits.max_array_layers() > 0,
            "a supported answer must carry real maxima: {usage}"
        );
    }

    // Half of the sixty-four masks carry the depth bit, and the empty mask is one
    // more, so thirty-one are legal. Asserted as a count rather than trusted: a
    // walk that silently stopped early would otherwise pass.
    assert_eq!(
        legal_seen, 31,
        "the walk must reach every legal mask, not merely some of them"
    );
}

/// Records the texture table this machine enumerated, for the evidence binding.
///
/// Run with `--nocapture`. Only the entries that carry information vary between
/// devices, so this prints those rather than every recorded key.
#[test]
fn what_the_texture_enumeration_actually_reported() {
    let device = portable_device();
    let capabilities = device.capabilities();

    for dimension in [
        TextureDimension::D1,
        TextureDimension::D2,
        TextureDimension::D3,
    ] {
        let query = TextureSupportQuery::new(
            dimension,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED,
            1,
        );
        let support = capabilities.texture_support(&query);
        println!(
            "dx12 texture {dimension:?} sampled rgba8: supported={} limits={:?}",
            support.is_supported(),
            support.limits().map(|limits| (
                limits.max_extent().width,
                limits.max_extent().height,
                limits.max_extent().depth,
                limits.max_mip_levels(),
                limits.max_array_layers(),
            )),
        );
    }

    for sample_count in [1u32, 2, 4, 8, 16] {
        let query = TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
            sample_count,
        );
        println!(
            "dx12 texture rgba8 color_attachment at {sample_count} samples: supported={}",
            capabilities.texture_support(&query).is_supported()
        );
    }

    println!(
        "dx12 texture limits: max_1d={:?} max_2d={:?} max_3d={:?} max_layers={:?} \
         min_uniform_offset={:?} min_storage_offset={:?} max_uniform_binding={:?} \
         max_storage_binding={:?}",
        capabilities.limit(LimitKey::MaxTexture1dDimension),
        capabilities.limit(LimitKey::MaxTexture2dDimension),
        capabilities.limit(LimitKey::MaxTexture3dDimension),
        capabilities.limit(LimitKey::MaxTextureArrayLayers),
        capabilities.limit(LimitKey::MinUniformBufferOffsetAlignment),
        capabilities.limit(LimitKey::MinStorageBufferOffsetAlignment),
        capabilities.limit(LimitKey::MaxUniformBufferBindingSize),
        capabilities.limit(LimitKey::MaxStorageBufferBindingSize),
    );

    println!(
        "dx12 pipeline limits: vertex_buffers={:?} vertex_attributes={:?} stride={:?} \
         color_attachments={:?} invocations={:?} workgroup=({:?},{:?},{:?}) \
         workgroups_per_dim={:?} workgroup_storage={:?}",
        capabilities.limit(LimitKey::MaxVertexBuffers),
        capabilities.limit(LimitKey::MaxVertexAttributes),
        capabilities.limit(LimitKey::MaxVertexBufferArrayStride),
        capabilities.limit(LimitKey::MaxColorAttachments),
        capabilities.limit(LimitKey::MaxComputeInvocationsPerWorkgroup),
        capabilities.limit(LimitKey::MaxComputeWorkgroupSizeX),
        capabilities.limit(LimitKey::MaxComputeWorkgroupSizeY),
        capabilities.limit(LimitKey::MaxComputeWorkgroupSizeZ),
        capabilities.limit(LimitKey::MaxComputeWorkgroupsPerDimension),
        capabilities.limit(LimitKey::MaxComputeWorkgroupStorageSize),
    );
}

// ---------------------------------------------------------------------------
// The binding-support table on real hardware.
//
// The binding table is the one capability whose *absence* is not conservative.
// `route`, `texture_support`, `view_compatibility` and `binding_limit` all answer
// something safe when they have no entry — a refusal costs throughput, or no
// ceiling is imposed — but a binding answer of `Unsupported` refuses a layout the
// device can actually bind, and section 9.4 makes that refusal final. So this
// table is walked rather than sampled, and the tests below check the walk reached
// the answers rather than checking a handful of convenient ones.
//
// What only this module can show: `D3D12_FEATURE_FORMAT_SUPPORT`'s
// `TYPED_UNORDERED_ACCESS_VIEW` and `D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD/STORE`
// bits are what the storage answers are derived from, and a mock cannot tell
// whether this driver sets them.
// ---------------------------------------------------------------------------

// The assertions below read a binding answer through
// `BindingSupport::is_supported`, which the crate carries as `pub(crate)` because
// section 20.4 freezes that enum at two members and no *public* accessor. This
// module had an extension trait of its own before that method existed, which was
// the same reader written twice.

/// Builds the query a shader would ask, with the magnitude fields filled in.
///
/// The magnitudes vary across the tests below on purpose. Two queries that differ
/// only in `min_size` or in a `Fixed(n)` count must answer identically, because
/// neither magnitude reaches the recorded key — a binding's size envelope is
/// section 22.3's `MaxUniformBufferBindingSize` and a binding's element count is
/// section 23.1's `binding_limit`. Passing a real value here rather than a token
/// one is what makes these tests able to fail if that narrowing is ever undone.
fn binding_query(
    visibility: ShaderStages,
    kind: BindingKind,
    count: BindingCount,
    dynamic_offset: bool,
) -> BindingSupportQuery {
    BindingSupportQuery {
        visibility,
        kind,
        count,
        dynamic_offset,
    }
}

/// The answers a renderer's layouts actually depend on, on this driver.
#[test]
fn a_real_device_answers_the_binding_questions_a_renderer_asks() {
    let device = portable_device();
    let capabilities = device.capabilities();

    let uniform = |count| {
        binding_query(
            ShaderStages::FRAGMENT,
            BindingKind::UniformBuffer { min_size: 256 },
            count,
            false,
        )
    };

    assert!(
        capabilities
            .binding_support(&uniform(BindingCount::One))
            .is_supported(),
        "a uniform buffer in the fragment stage is the least exotic binding there \
         is; a device that refuses it cannot run any renderer"
    );
    assert!(
        capabilities
            .binding_support(&uniform(BindingCount::Fixed(4)))
            .is_supported(),
        "a fixed-length uniform array is what Direct3D 12 calls a descriptor range, \
         and refusing it would refuse every array-typed layout"
    );

    // Dynamic offsets require root descriptors; this backend currently lowers
    // static descriptor tables only and must not advertise the missing path.
    let dynamic = binding_query(
        ShaderStages::VERTEX,
        BindingKind::StorageBuffer {
            min_size: 256,
            access: BufferBindingAccess::ReadOnly,
        },
        BindingCount::One,
        true,
    );
    assert!(!capabilities.binding_support(&dynamic).is_supported());

    // Every stage must be answered, including the compute stage a dispatch uses.
    for visibility in [
        ShaderStages::VERTEX,
        ShaderStages::FRAGMENT,
        ShaderStages::COMPUTE,
    ] {
        let query = binding_query(
            visibility,
            BindingKind::SampledTexture {
                dimension: TextureViewDimension::D2,
                sample_type: TextureSampleType::Float,
                multisampled: false,
            },
            BindingCount::One,
            false,
        );
        assert_eq!(
            capabilities.binding_support(&query).is_supported(),
            visibility != ShaderStages::COMPUTE,
            "compute texture uses remain unsupported until command lowering can transition them"
        );
    }

    for kind in [
        SamplerKind::Filtering,
        SamplerKind::NonFiltering,
        SamplerKind::Comparison,
    ] {
        let query = binding_query(
            ShaderStages::COMPUTE,
            BindingKind::Sampler { kind },
            BindingCount::Fixed(4),
            false,
        );
        assert!(capabilities.binding_support(&query).is_supported());
    }
}

/// The two negatives, asserted where they are decided rather than where they are
/// convenient.
///
/// Both are answers Direct3D 12 gives structurally — one through the absence of a
/// member in `D3D12_UAV_DIMENSION`, the other through the absence of a
/// multisampled spelling in `D3D12_SRV_DIMENSION` — and a backend that answered
/// them from a guess rather than from the enumeration would be indistinguishable
/// from one that answered them from the device. These tests are here so that the
/// guess and the reading are not.
#[test]
fn a_real_device_refuses_the_two_binding_shapes_direct3d_12_cannot_express() {
    let device = portable_device();
    let capabilities = device.capabilities();

    for dimension in [TextureViewDimension::Cube, TextureViewDimension::CubeArray] {
        let query = binding_query(
            ShaderStages::COMPUTE,
            BindingKind::StorageTexture {
                dimension,
                format: TextureFormat::Rgba8Unorm,
                access: StorageAccess::ReadWrite,
            },
            BindingCount::One,
            false,
        );
        assert!(
            !capabilities.binding_support(&query).is_supported(),
            "Direct3D 12 has no cube member in D3D12_UAV_DIMENSION at any resource \
             binding tier, so a cube storage texture cannot be bound: {dimension:?}"
        );
    }

    for dimension in [
        TextureViewDimension::D1,
        TextureViewDimension::D3,
        TextureViewDimension::Cube,
    ] {
        let query = binding_query(
            ShaderStages::FRAGMENT,
            BindingKind::SampledTexture {
                dimension,
                sample_type: TextureSampleType::Float,
                multisampled: true,
            },
            BindingCount::One,
            false,
        );
        assert!(
            !capabilities.binding_support(&query).is_supported(),
            "D3D12_SRV_DIMENSION spells multisampling only for 2D and 2D-array \
             views, so this cannot be sampled: {dimension:?}"
        );
    }
}

/// Every legal binding shape is recorded rather than left to the negative.
///
/// This is the obligation that replaces "fill everything": the binding key space
/// is not enumerable, so an absent key and a refusal are the same value and a hole
/// cannot be detected the way the buffer test detects one. What can be checked is
/// that the walk reaches the whole space and that nothing *legal* in it comes back
/// refused. The count is asserted rather than trusted, because a walk that stopped
/// early would otherwise pass.
#[test]
fn every_binding_shape_matches_the_current_dx12_lowering() {
    let device = portable_device();
    let capabilities = device.capabilities();

    // The shapes a storage texture can legally take on a color format: cube views
    // are excluded structurally, and a storage texture cannot be multisampled.
    let dimensions = [
        TextureViewDimension::D1,
        TextureViewDimension::D2,
        TextureViewDimension::D2Array,
        TextureViewDimension::D3,
    ];
    let accesses = [
        StorageAccess::ReadOnly,
        StorageAccess::WriteOnly,
        StorageAccess::ReadWrite,
    ];

    let mut seen = 0usize;
    for visibility in [
        ShaderStages::VERTEX,
        ShaderStages::FRAGMENT,
        ShaderStages::COMPUTE,
    ] {
        // Static buffer tables are implemented; dynamic offsets are not.
        for kind in [
            BindingKind::UniformBuffer { min_size: 64 },
            BindingKind::StorageBuffer {
                min_size: 64,
                access: BufferBindingAccess::ReadOnly,
            },
            BindingKind::StorageBuffer {
                min_size: 64,
                access: BufferBindingAccess::ReadWrite,
            },
        ] {
            for count in [BindingCount::One, BindingCount::Fixed(4)] {
                for dynamic_offset in [false, true] {
                    let query = binding_query(visibility, kind.clone(), count, dynamic_offset);
                    seen += 1;
                    assert_eq!(
                        capabilities.binding_support(&query).is_supported(),
                        !dynamic_offset,
                        "capability must match the descriptor-table lowering: \
                         {visibility:?} {kind:?} {count:?} dynamic={dynamic_offset}"
                    );
                }
            }
        }

        // Storage textures over every legal view dimension and access.
        for dimension in dimensions {
            for access in accesses {
                let query = binding_query(
                    visibility,
                    BindingKind::StorageTexture {
                        dimension,
                        format: TextureFormat::Rgba8Unorm,
                        access,
                    },
                    BindingCount::One,
                    false,
                );
                seen += 1;
                let expected = visibility != ShaderStages::COMPUTE
                    && capabilities
                        .format(TextureFormat::Rgba8Unorm)
                        .is_some_and(|facts| facts.storage_access().supports(access));
                assert_eq!(
                    capabilities.binding_support(&query).is_supported(),
                    expected
                );
            }
        }
    }

    // Per stage: three buffer kinds times (two counts times two dynamic-offset
    // settings) is twelve, plus four view dimensions times three accesses is
    // twelve — twenty-four. Three stages makes seventy-two. Asserted as a count
    // rather than trusted: a walk that silently stopped early would otherwise pass.
    let expected = 3 * (3 * 2 * 2 + 4 * 3);
    assert_eq!(
        seen, expected,
        "the walk must reach every key it claims to have checked"
    );
}

/// Records the binding table this machine enumerated, for the evidence binding.
///
/// Run with `--nocapture`. The binding answers are almost entirely structural, so
/// what varies between devices is the storage-texture half — which is a format
/// question, and therefore the half a driver is free to differ on.
#[test]
fn what_the_binding_enumeration_actually_reported() {
    let device = portable_device();
    let capabilities = device.capabilities();

    for kind in [
        BindingKind::UniformBuffer { min_size: 64 },
        BindingKind::StorageBuffer {
            min_size: 64,
            access: BufferBindingAccess::ReadOnly,
        },
        BindingKind::StorageBuffer {
            min_size: 64,
            access: BufferBindingAccess::ReadWrite,
        },
    ] {
        for dynamic_offset in [false, true] {
            println!(
                "dx12 binding {kind:?} dynamic={dynamic_offset} one={} array={}",
                capabilities
                    .binding_support(&binding_query(
                        ShaderStages::COMPUTE,
                        kind.clone(),
                        BindingCount::One,
                        dynamic_offset
                    ))
                    .is_supported(),
                capabilities
                    .binding_support(&binding_query(
                        ShaderStages::COMPUTE,
                        kind.clone(),
                        BindingCount::Fixed(4),
                        dynamic_offset
                    ))
                    .is_supported(),
            );
        }
    }

    for dimension in [
        TextureViewDimension::D1,
        TextureViewDimension::D2,
        TextureViewDimension::D2Array,
        TextureViewDimension::Cube,
        TextureViewDimension::CubeArray,
        TextureViewDimension::D3,
    ] {
        print!("dx12 binding storage_texture rgba8 {dimension:?}:");
        for access in [
            StorageAccess::ReadOnly,
            StorageAccess::WriteOnly,
            StorageAccess::ReadWrite,
        ] {
            print!(
                " {access:?}={}",
                capabilities
                    .binding_support(&binding_query(
                        ShaderStages::COMPUTE,
                        BindingKind::StorageTexture {
                            dimension,
                            format: TextureFormat::Rgba8Unorm,
                            access,
                        },
                        BindingCount::One,
                        false
                    ))
                    .is_supported()
            );
        }
        println!();
    }

    for multisampled in [false, true] {
        print!("dx12 binding sampled_texture float multisampled={multisampled}:");
        for dimension in [
            TextureViewDimension::D1,
            TextureViewDimension::D2,
            TextureViewDimension::D2Array,
            TextureViewDimension::Cube,
            TextureViewDimension::CubeArray,
            TextureViewDimension::D3,
        ] {
            print!(
                " {dimension:?}={}",
                capabilities
                    .binding_support(&binding_query(
                        ShaderStages::FRAGMENT,
                        BindingKind::SampledTexture {
                            dimension,
                            sample_type: TextureSampleType::Float,
                            multisampled,
                        },
                        BindingCount::One,
                        false
                    ))
                    .is_supported()
            );
        }
        println!();
    }

    // The absence that is a decision rather than a gap, printed where a reader of
    // the evidence will meet it.
    println!(
        "dx12 binding_limit: fragment/uniform={:?} (absent by design: Direct3D 12 \
         states no per-stage-class ceiling, and the descriptor-heap sizes are pools \
         shared by every stage and pipeline)",
        capabilities.binding_limit(ShaderStage::Fragment, BindingLimitClass::UniformBuffers),
    );
}

// ---------------------------------------------------------------------------
// Buffer allocation on real hardware.
//
// The portable half of this verb — that a refusal happens before a backend is
// touched, and what a created handle reports — is asserted over `api::tests::mock` in
// `api::tests::resource::buffer`. What only this module can show is that the
// lowering itself works: that Direct3D 12 accepts the descriptors the portable
// rules admit, that the native description is the one the caller asked for, and
// that the one creation-time flag this backend sets is set for exactly the usage
// that requires it.
//
// None of this moves a byte. `CopyBufferRegion` needs a command queue, an
// allocator, a command list and a fence, and those arrive in a later round; until
// they do, an allocation that nothing reads and nothing writes is a weaker claim
// than it looks, and the module documentation above says so.
// ---------------------------------------------------------------------------

/// The native allocation behind `buffer`, as this backend committed it.
///
/// The downcast is the seam doing its job: only a caller that knows which backend
/// produced this handle can name the type behind `dyn BufferBackend`, and the
/// portable layer above never can.
fn native_buffer(buffer: &Buffer) -> &Dx12Buffer {
    buffer
        .native()
        .as_any()
        .downcast_ref::<Dx12Buffer>()
        .expect("a buffer created through this device carries a DX12 allocation")
}

/// A descriptor of `size` bytes with exactly one usage bit, so that each case
/// below isolates one class rather than a combination whose flag could be
/// explained by either half.
fn single_usage(size: u64, usage: BufferUsage) -> BufferDescriptor {
    BufferDescriptor::new(size, usage)
}

#[test]
fn a_real_device_allocates_a_buffer_for_every_usage_class() {
    // Section 11.1's six bits, one at a time. Direct3D 12 expresses five of them
    // as resource states rather than as creation flags, so all six must be
    // creatable — an `Unsupported` here would mean the portable capability table
    // and this lowering disagree about the same device, which is the failure this
    // test exists to catch.
    let device = portable_device();

    for usage in [
        BufferUsage::COPY_SRC,
        BufferUsage::COPY_DST,
        BufferUsage::VERTEX,
        BufferUsage::INDEX,
        BufferUsage::UNIFORM,
        BufferUsage::STORAGE,
    ] {
        let buffer = device
            .create_buffer(&single_usage(4096, usage))
            .unwrap_or_else(|error| {
                panic!(
                    "Direct3D 12 allocates a {usage} buffer: {}",
                    error.message()
                )
            });

        let native = native_buffer(&buffer);
        // SAFETY: `GetDesc` writes into a by-value return of a POD struct and
        // reads nothing but the resource this handle owns.
        let desc = unsafe { native.resource().GetDesc() };

        assert_eq!(desc.Width, 4096, "the width of a buffer is its byte size");
        assert_eq!(
            desc.Dimension, D3D12_RESOURCE_DIMENSION_BUFFER,
            "a buffer is a buffer, not a texture shaped like one"
        );
        // The four fields Direct3D 12 requires a buffer to pin. `D3D12_RESOURCE_DESC`
        // is one struct for every resource kind, so the fields a buffer does not use
        // are not "don't care" — they are checked, and `CreateCommittedResource`
        // fails on a wrong value rather than ignoring it.
        assert_eq!(desc.Height, 1);
        assert_eq!(desc.DepthOrArraySize, 1);
        assert_eq!(desc.MipLevels, 1);
        assert_eq!(desc.SampleDesc.Count, 1);
        assert_eq!(desc.SampleDesc.Quality, 0);
    }
}

#[test]
fn a_real_device_grants_unordered_access_to_exactly_the_usage_that_needs_it() {
    // `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` is a creation-time *grant*
    // rather than a hint: a buffer committed without it can never be a UAV, and no
    // later barrier or descriptor makes it one. So this is not a preference being
    // pinned but a correctness contract — section 11.1 makes usage a creation-time
    // fact for precisely this reason.
    //
    // Both directions are asserted. Granting it unconditionally would pass a test
    // that only checked the storage case, and it would quietly make every buffer a
    // UAV candidate, relaxing the contract the caller stated.
    let device = portable_device();

    let storage = device
        .create_buffer(&single_usage(1024, BufferUsage::STORAGE))
        .expect("a storage buffer is allocatable on a real device");
    // SAFETY: `GetDesc` returns a POD struct and reads only this resource.
    let desc = unsafe { native_buffer(&storage).resource().GetDesc() };
    assert!(
        desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0,
        "a buffer created as STORAGE without the UAV flag could never be a UAV"
    );

    let vertex = device
        .create_buffer(&single_usage(1024, BufferUsage::VERTEX))
        .expect("a vertex buffer is allocatable on a real device");
    // SAFETY: as above.
    let desc = unsafe { native_buffer(&vertex).resource().GetDesc() };
    assert_eq!(
        desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0,
        0,
        "a vertex buffer granted unordered access would be a relaxation the caller \
         never asked for, and one no portable rule could then recover"
    );
}

#[test]
fn a_real_allocation_is_addressable_and_each_one_is_a_different_place() {
    // A non-zero GPU virtual address is the cheapest real evidence that
    // `CreateCommittedResource` committed memory rather than returning a handle to
    // nothing: Direct3D 12 returns zero for a resource with no GPU address, and
    // `CopyBufferRegion` takes exactly this value as its operand.
    //
    // Two allocations are made because one non-zero address proves only that *an*
    // address exists. Distinct addresses are what show the second call allocated
    // rather than re-returning the first.
    let device = portable_device();

    let first = device
        .create_buffer(&single_usage(65536, BufferUsage::COPY_DST))
        .expect("a 64 KiB copy destination is allocatable on a real device");
    let second = device
        .create_buffer(&single_usage(65536, BufferUsage::COPY_SRC))
        .expect("a 64 KiB copy source is allocatable on a real device");

    // SAFETY: `GetGPUVirtualAddress` reads one field of this resource and has no
    // preconditions beyond the handle being live, which the portable `Buffer`
    // guarantees by holding the allocation.
    let first_address = unsafe { native_buffer(&first).resource().GetGPUVirtualAddress() };
    // SAFETY: as above.
    let second_address = unsafe { native_buffer(&second).resource().GetGPUVirtualAddress() };

    assert_ne!(
        first_address, 0,
        "a committed resource has a GPU virtual address; zero means nothing was committed"
    );
    assert_ne!(second_address, 0);
    assert_ne!(
        first_address, second_address,
        "two separate creations must be two separate allocations, or the second \
         creation silently aliased the first"
    );
}

#[test]
fn a_clone_holds_the_one_allocation_rather_than_a_second_copy_of_it() {
    // Section 18.6's last-owner rule, seen from the native side: the portable
    // layer shares one `Arc` between clones, so the two handles must resolve to the
    // same `ID3D12Resource`. If they did not, one of them would be freed while the
    // other still referred to it — the double-free the rule exists to prevent —
    // and this is where that would be caught before a copy ever used one.
    let device = portable_device();
    let buffer = device
        .create_buffer(&single_usage(4096, BufferUsage::COPY_SRC))
        .expect("a 4 KiB copy source is allocatable on a real device");

    let clone = buffer.clone();

    assert!(
        std::ptr::eq(clone.native(), buffer.native()),
        "a clone must share the backend allocation, not allocate a second one"
    );
    // SAFETY: both handles are live and `GetGPUVirtualAddress` only reads.
    let buffer_address = unsafe { native_buffer(&buffer).resource().GetGPUVirtualAddress() };
    // SAFETY: as above.
    let clone_address = unsafe { native_buffer(&clone).resource().GetGPUVirtualAddress() };
    assert_eq!(buffer_address, clone_address);
}

#[test]
fn a_real_device_allocates_under_either_placement_preference() {
    // Section 11.2 makes `ResourceMemoryPreference` a performance hint and never a
    // correctness guarantee, and both variants lower onto
    // `D3D12_HEAP_TYPE_DEFAULT` — section 11.2 deleted host-visible buffers, so no
    // caller-stated preference can select `UPLOAD` or `READBACK`. What this asserts
    // is the consequence a caller can depend on: the hint is accepted rather than
    // refused, on either setting, and neither setting changes the descriptor the
    // driver is handed.
    let device = portable_device();

    let mut descriptors = Vec::new();
    for memory in [
        ResourceMemoryPreference::Automatic,
        ResourceMemoryPreference::DeviceLocalPreferred,
    ] {
        descriptors
            .push(BufferDescriptor::new(2048, BufferUsage::UNIFORM).with_memory_preference(memory));
    }

    for descriptor in descriptors {
        let buffer = device.create_buffer(&descriptor).unwrap_or_else(|error| {
            panic!(
                "a {:?} placement is a hint, not a request the device may refuse: {}",
                descriptor.memory,
                error.message()
            )
        });
        // SAFETY: `GetDesc` returns a POD struct and reads only this resource.
        let desc = unsafe { native_buffer(&buffer).resource().GetDesc() };
        assert_eq!(desc.Width, 2048);
        assert_eq!(desc.Dimension, D3D12_RESOURCE_DIMENSION_BUFFER);
    }
}

#[test]
fn a_real_device_refuses_a_buffer_that_the_portable_rules_reject() {
    // The refusal half, and the reason it belongs *here* rather than only in the
    // mock: this asserts that a device which really can allocate still answers
    // `InvalidUsage` for a zero-byte buffer, rather than passing it to
    // `CreateCommittedResource` and reporting whatever the driver says. Section
    // 12.3 is a portable rule, so its verdict must not depend on the driver's
    // opinion — and a driver's opinion here is `E_INVALIDARG`, which would be a
    // `BackendFailure` naming an HRESULT that means nothing to a caller.
    let device = portable_device();

    let error = device
        .create_buffer(&single_usage(0, BufferUsage::VERTEX))
        .expect_err("section 12.3 refuses a zero-byte buffer before any backend sees it");

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("Device::create_buffer"));
}

// ---------------------------------------------------------------------------
// Bytes on the GPU: upload, copy and readback.
//
// Everything under this heading executes on the real queue. The command spine
// records real command lists, `ExecuteCommandLists` runs them, and the
// assertions read back what the GPU wrote. `CLAUDE.md` section 4.8 is the rule
// that makes these the only evidence of their kind here: a status code from a
// mock, or a validation layer that stayed quiet, would say nothing about whether
// a byte moved.
// ---------------------------------------------------------------------------

/// The lane of `device` that accepts `COPY` work.
///
/// Read out of the portable capability table rather than written as
/// `SubmissionLaneId::new(0)`, because the id is the device's fact to state: a
/// test that hard-coded one would keep passing against a device whose lane table
/// lies, which is exactly what section 40.1's
/// `lane.domains().contains(work.work_domains())` check exists to prevent.
fn copy_lane(device: &crate::api::platform::Device) -> SubmissionLaneId {
    device
        .capabilities()
        .submission()
        .lanes()
        .iter()
        .find(|lane| lane.domains().contains(LaneWorkDomains::COPY))
        .map(|lane| lane.id())
        .expect("section 7.2's base guarantee requires every device to offer a COPY lane")
}

/// A pattern no partial or misplaced write could imitate.
///
/// Built from a plain LCG so the bytes are reproducible without a dependency,
/// and with a full 32-bit state reduced to one byte per element, so that no
/// short-range structure — a repeated word, a ramp, a constant run — would let a
/// copy that moved the wrong four kilobytes compare equal by accident.
fn pattern_of(size: usize) -> Vec<u8> {
    let mut pattern = Vec::with_capacity(size);
    let mut state: u32 = 0x1234_5678;
    while pattern.len() < size {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pattern.push((state >> 24) as u8);
    }
    pattern
}

/// A pattern written on the CPU comes back off the GPU byte for byte.
///
/// This is `version-plan.md` section 4's real headless **copy/upload/readback**
/// requirement on Windows DX12, and the shape is chosen so that each hop is
/// load-bearing: the pattern reaches the source buffer only through an upload
/// job, the destination only through a device-side buffer copy, and the CPU only
/// through a readback ticket. A spine that dropped any one of the three fails the
/// final comparison — including one that silently aliased the two buffers, which
/// a copy that did nothing at all would produce, and including one that forgot
/// the staging copy, which would read back the destination's original contents.
///
/// Both buffers carry `COPY_SRC | COPY_DST`, because both roles are used: the
/// source is written by the upload and read by the copy, and the destination is
/// written by the copy and read by the readback.
#[test]
fn a_real_device_moves_bytes_from_the_cpu_to_a_buffer_and_back_to_the_cpu() {
    const SIZE: u64 = 4096;

    let device = portable_device();
    let pattern = pattern_of(SIZE as usize);

    let source = device
        .create_buffer(&BufferDescriptor::new(
            SIZE,
            BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        ))
        .expect("a 4 KiB copy source is allocatable on a real device");
    let destination = device
        .create_buffer(&BufferDescriptor::new(
            SIZE,
            BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        ))
        .expect("a 4 KiB copy destination is allocatable on a real device");

    let job = device
        .create_buffer_upload(BufferUploadDescriptor {
            label: Label(Some("dx12 end-to-end upload".to_string())),
            dst: source.clone(),
            dst_offset: 0,
            bytes: Arc::from(pattern.as_slice()),
        })
        .expect("a 4 KiB upload into a COPY_DST buffer meets section 17.3's list");

    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("a real device opens a recorder");
    recorder
        .encode_upload(&job)
        .expect("a job validated at creation encodes without asking the device anything");
    recorder
        .copy_buffer(&BufferCopy {
            src: source.clone(),
            src_offset: 0,
            dst: destination.clone(),
            dst_offset: 0,
            size: SIZE,
        })
        .expect("the device reports a buffer-to-buffer route this copy meets");
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some("dx12 end-to-end readback".to_string())),
            src: destination.clone(),
            range: BufferRange::new(0, SIZE),
        })
        .expect("the destination carries COPY_SRC, so section 18.1 permits reading it");
    let work = recorder.finish().expect("no scope was left open");

    let mut builder = SubmissionPlanBuilder::new(&device);
    let point = builder
        .add_batch(copy_lane(&device), vec![work])
        .expect("the lane that accepts COPY accepts a recording whose only domain is COPY");
    let plan = builder
        .build()
        .expect("a single batch has no edge that could be cyclic");

    let receipt = block_on(device.submit(plan))
        .expect("acceptance does not await completion, and section 41.7 forbids it doing so");
    let token = receipt
        .completion_for(point)
        .expect("the point came from this plan's own builder, so the receipt must know it");

    // Section 41.10 makes polling the way a frame loop makes progress and section
    // 41.7 makes acceptance a separate fact from completion, so this loop is the
    // caller's only way to learn that the GPU finished. It is bounded on purpose:
    // an unbounded wait would hang the suite instead of reporting one, and a
    // bound that trips is a failure here, not a slow machine.
    let mut terminal = None;
    for _ in 0..10_000 {
        device
            .poll()
            .expect("a poll on a live device must succeed; section 6.5 requires progress");
        match device
            .completion_state(token)
            .expect("the token came from this device")
        {
            CompletionState::Pending => {}
            // Every terminal state ends the loop, including the two failures:
            // section 41.8 makes them terminal, so polling through one would spin
            // forever and report a timeout where the device had already answered.
            other => {
                terminal = Some(other);
                break;
            }
        }
    }
    let terminal = terminal.expect(
        "one 4 KiB copy must reach a terminal state; never observing one means the \
         fence never advanced, which is a spin a bound must report rather than hide",
    );
    assert!(
        matches!(terminal, CompletionState::Complete),
        "the upload, copy and readback must complete on a live device, but the device \
         reported {terminal:?}"
    );

    // The bytes themselves. Section 18.4 makes a ticket the only place they live,
    // so this is where the GPU's answer becomes comparable to the CPU's input.
    let view = ticket
        .try_read()
        .expect("a completed readback with published bytes is readable, not terminal")
        .expect("the ticket is Ready once its work reached a terminal Complete state");
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("the request was a buffer range, so the data must be a buffer range too");
    };
    assert_eq!(
        bytes,
        pattern.as_slice(),
        "the bytes the GPU wrote back must be the bytes the CPU uploaded"
    );

    // The evidence binding, for a `--nocapture` run: `CLAUDE.md` section 5 wants
    // the adapter, the size and the compared bytes attached to the result, and
    // printing them here is what makes one run's record reproducible rather than
    // only its verdict.
    println!(
        "dx12 byte-movement evidence: adapter={:?} backend={:?} bytes={} \
         expected_first8={:?} actual_first8={:?}",
        device.adapter_info().name(),
        device.backend(),
        bytes.len(),
        &pattern[..8],
        &bytes[..8],
    );
}

/// Exercises all direct texture transfer lowerings on a real adapter: host
/// upload, texture copy, texture-to-buffer, buffer-to-texture, then readback.
#[test]
fn a_real_device_round_trips_texels_through_every_dx12_copy_path() {
    let device = portable_device();
    let usage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    let make_texture = || {
        device
            .create_texture(&TextureDescriptor::new_2d(
                4,
                2,
                TextureFormat::Rgba8Unorm,
                usage,
            ))
            .unwrap()
    };
    let source = make_texture();
    let middle = make_texture();
    let destination = make_texture();
    let buffer = device
        .create_buffer(&BufferDescriptor::new(
            512,
            BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        ))
        .unwrap();
    let bytes: [u8; 32] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    ];
    let layers = TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let origin = Origin3d { x: 0, y: 0, z: 0 };
    let extent = Extent3d::d2(4, 2);
    let upload = device
        .create_texture_upload(TextureUploadDescriptor {
            label: Label(Some("dx12 texture transfer".into())),
            dst: source.clone(),
            subresource: layers,
            origin,
            extent,
            source_layout: HostTexelLayout {
                bytes_per_row: 16,
                rows_per_image: 2,
            },
            bytes: Arc::from(bytes.as_slice()),
        })
        .unwrap();
    let buffer_copy = BufferTextureCopy {
        buffer: buffer.clone(),
        buffer_offset: 0,
        bytes_per_row: 256,
        rows_per_image: 2,
        texture: middle.clone(),
        texture_subresource: layers,
        texture_origin: origin,
        extent,
    };
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    recorder.encode_upload(&upload).unwrap();
    recorder
        .copy_texture(&TextureCopy {
            src: source,
            src_subresource: layers,
            src_origin: origin,
            dst: middle.clone(),
            dst_subresource: layers,
            dst_origin: origin,
            extent,
        })
        .unwrap();
    recorder.copy_texture_to_buffer(&buffer_copy).unwrap();
    recorder
        .copy_buffer_to_texture(&BufferTextureCopy {
            texture: destination.clone(),
            ..buffer_copy
        })
        .unwrap();
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some("dx12 texture readback".into())),
            src: destination,
            subresource: layers,
            origin,
            extent,
        })
        .unwrap();
    let work = recorder.finish().unwrap();
    let mut builder = SubmissionPlanBuilder::new(&device);
    let point = builder.add_batch(copy_lane(&device), vec![work]).unwrap();
    let receipt = block_on(device.submit(builder.build().unwrap())).unwrap();
    let completion = receipt.completion_for(point).unwrap();
    for _ in 0..10_000 {
        device.poll().unwrap();
        if matches!(
            device.completion_state(completion).unwrap(),
            CompletionState::Complete
        ) {
            break;
        }
    }
    let view = ticket
        .try_read()
        .unwrap()
        .expect("texture readback must be ready after completion");
    let ReadbackViewData::Texture {
        bytes: actual,
        layout,
    } = view.data()
    else {
        panic!("texture request must return texture layout")
    };
    assert_eq!(layout.bytes_per_row, 256);
    assert_eq!(&actual[..16], &bytes[..16]);
    assert_eq!(&actual[256..272], &bytes[16..]);
}

/// One upload job encoded twice writes its bytes twice.
///
/// Section 17.2 makes a job repeatable rather than one-shot, and the DX12 spine
/// honours that by allocating staging inside the recording rather than holding it
/// on the job — so the second encode needs a *second* staging allocation, and the
/// first batch's must stay alive until its own fence reports it finished. A spine
/// that shared one staging buffer between the two encodes would have the second
/// CPU write race the first copy; a spine that freed the first at encode time
/// would hand the GPU freed memory. Both are invisible to a single-encode test,
/// and both are what this one is for.
///
/// The two destinations make the result checkable: each must hold the pattern,
/// which a shared staging buffer would still produce, so the assertion that
/// matters is that both *submissions* succeed and both readbacks compare equal —
/// a use-after-free here would be a device removal, not a wrong byte.
#[test]
fn one_upload_job_encodes_repeatedly_and_each_encoding_writes_its_own_bytes() {
    const SIZE: u64 = 1024;

    let device = portable_device();
    let pattern = pattern_of(SIZE as usize);

    let source = device
        .create_buffer(&BufferDescriptor::new(
            SIZE,
            BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        ))
        .expect("a 1 KiB copy source is allocatable on a real device");
    let first = device
        .create_buffer(&BufferDescriptor::new(
            SIZE,
            BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        ))
        .expect("a 1 KiB destination is allocatable on a real device");
    let second = device
        .create_buffer(&BufferDescriptor::new(
            SIZE,
            BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        ))
        .expect("a second 1 KiB destination is allocatable on a real device");

    let job = device
        .create_buffer_upload(BufferUploadDescriptor {
            label: Label(Some("dx12 repeatable upload".to_string())),
            dst: source.clone(),
            dst_offset: 0,
            bytes: Arc::from(pattern.as_slice()),
        })
        .expect("a 1 KiB upload into a COPY_DST buffer meets section 17.3's list");

    let mut tickets = Vec::new();
    let mut tokens = Vec::new();
    for destination in [&first, &second] {
        let mut recorder = device
            .create_recorder(&RecorderDescriptor::new())
            .expect("a real device opens a recorder");
        recorder
            .encode_upload(&job)
            .expect("the same job encodes a second time; section 17.2 makes it repeatable");
        recorder
            .copy_buffer(&BufferCopy {
                src: source.clone(),
                src_offset: 0,
                dst: destination.clone(),
                dst_offset: 0,
                size: SIZE,
            })
            .expect("the device reports a buffer-to-buffer route this copy meets");
        tickets.push(
            recorder
                .encode_readback(ReadbackRequest::Buffer {
                    label: Label(Some("dx12 repeatable upload readback".to_string())),
                    src: destination.clone(),
                    range: BufferRange::new(0, SIZE),
                })
                .expect("the destination carries COPY_SRC"),
        );

        let work = recorder.finish().expect("no scope was left open");
        let mut builder = SubmissionPlanBuilder::new(&device);
        let point = builder
            .add_batch(copy_lane(&device), vec![work])
            .expect("the COPY lane accepts a recording whose only domain is COPY");
        let plan = builder.build().expect("a single batch has no edge");

        // The two batches are submitted separately rather than in one plan,
        // because the spine allocates staging per batch: a second plan submitted
        // while the first is still in flight is the case where a spine that
        // reused one staging allocation would corrupt the first copy.
        let receipt = block_on(device.submit(plan))
            .expect("a second submission is accepted while the first may still be running");
        tokens.push(
            receipt
                .completion_for(point)
                .expect("the point came from this plan's own builder"),
        );
    }

    let mut terminal = Vec::new();
    for token in &tokens {
        terminal.push(settle(&device, *token));
    }
    for state in &terminal {
        assert!(
            matches!(state, CompletionState::Complete),
            "both submissions must complete on a live device, but one reported {state:?}"
        );
    }

    for ticket in &tickets {
        let view = ticket
            .try_read()
            .expect("a completed readback with published bytes is readable")
            .expect("the ticket is Ready once its work completed");
        let ReadbackViewData::Buffer { bytes } = view.data() else {
            panic!("the request was a buffer range, so the data must be a buffer range too");
        };
        assert_eq!(
            bytes,
            pattern.as_slice(),
            "each encoding of the job must have written the pattern into its own destination"
        );
    }
}

/// Polls `device` until `token` reaches a terminal state, and returns that state.
///
/// The bounded form of the loop the test above spells out inline, factored out
/// because two callers need the identical rule: section 41.10 forbids a blocking
/// wait, section 41.7 makes `poll` the only progress verb, and section 41.8 makes
/// both failure states terminal. The bound is deliberately far longer than any
/// 4 KiB copy needs, because its purpose is to report a fence that never
/// advanced, not to measure a slow GPU. Its input is a completion token rather
/// than a plan point, because turning one into the other is the receipt's job and
/// a point on its own names work only to the plan that produced it.
fn settle(device: &crate::api::platform::Device, token: CompletionPoint) -> CompletionState {
    for _ in 0..10_000 {
        device.poll().expect("a poll on a live device must succeed");
        match device
            .completion_state(token)
            .expect("the token came from this device")
        {
            CompletionState::Pending => {}
            other => return other,
        }
    }
    panic!("the point never reached a terminal state; the fence is not advancing");
}

/// Records what a real allocation actually looked like, for the evidence binding.
///
/// Run with `--nocapture`. The values that matter here — the GPU virtual addresses
/// and the adapter's own buffer ceiling — vary by machine and driver, and
/// `CLAUDE.md` section 9 forbids asserting a reading that was not taken. The
/// assertions that do bind are in the tests above.
///
/// One reading needs a warning attached, because it looks like a contradiction of
/// the test above and is not one. Each iteration drops its buffer before the next
/// is made, so Direct3D 12 is free to hand the next allocation the address the
/// freed one had — and on this driver it does. A repeated address here therefore
/// means *the allocator reused a freed region*, not that two live allocations
/// aliased. The claim that two live allocations have different addresses is the
/// one asserted in `a_real_allocation_is_addressable_and_each_one_is_a_different_place`,
/// where both handles are held at once.
#[test]
fn what_the_buffer_allocation_actually_reported() {
    let device = portable_device();

    println!(
        "dx12 allocation evidence: backend={:?} adapter={:?} max_buffer_size={:?}",
        device.backend(),
        device.adapter_info().name(),
        device.capabilities().limit(LimitKey::MaxBufferSize),
    );

    for usage in [
        BufferUsage::COPY_SRC,
        BufferUsage::COPY_DST,
        BufferUsage::VERTEX,
        BufferUsage::INDEX,
        BufferUsage::UNIFORM,
        BufferUsage::STORAGE,
    ] {
        let Ok(buffer) = device.create_buffer(&single_usage(4096, usage)) else {
            println!("dx12 allocation {usage}: refused");
            continue;
        };
        let resource = native_buffer(&buffer).resource();
        // SAFETY: both calls read one field of a live resource and write nothing.
        let (desc, address) = unsafe { (resource.GetDesc(), resource.GetGPUVirtualAddress()) };
        println!(
            "dx12 allocation {usage}: gpu_address=0x{address:016x} dimension={:?} \
             width={} flags=0x{:x}",
            desc.Dimension, desc.Width, desc.Flags.0,
        );
    }
}
