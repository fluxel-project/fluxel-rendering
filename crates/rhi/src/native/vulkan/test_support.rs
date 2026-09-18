//! Shared scaffolding for the Vulkan tests that need a real window, a real surface,
//! or a retained raster recipe.
//!
//! Several step 10 modules are only provable against a real window: the surface
//! handle, the facts it reports, and the swapchain created over it. The Win32
//! window lowering and the "which enumerated adapter is this surface on" search are
//! written once here rather than copied into each of their test modules, which is
//! also what keeps the window's lifetime rule in one place.
//!
//! Step 12's recorder tests need the same minimal raster recipe the pipeline tests
//! create against -- the position-only vertex stream, the colour-only fixed-function
//! state, and the two Naga-emitted SPIR-V modules -- so those are written once here
//! too. A second copy in `command`'s tests could drift from the pipeline's, and the
//! two would then be testing different recipes while both compiling.
//!
//! Compiled under `cfg(test)` only: nothing here is part of the backend.

use ash::vk;
use fluxel_rendergraph::{
    AttachmentOps, Extent3d, LoadOp, RasterColorAttachment, StoreOp, TextureDesc, TextureDimension,
    TextureFormat, TextureRange, TextureUsage, TextureUsageKind, WriteCoverage,
};
use raw_window_handle::{RawWindowHandle, Win32WindowHandle};

use super::format;
use super::framebuffer::Framebuffer;
use super::memory;
use super::open::OpenedVulkan;
use super::pipeline::RasterShaders;
use super::presentation::{Surface, SurfaceFacts};
use super::render_pass;
use super::resource::ResourceTable;
use crate::common::pipeline::{ColorTargetState, ColorWriteMask, PipelineState, PrimitiveState};
use crate::common::vertex::{
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, VertexStepMode,
};

#[link(name = "user32")]
unsafe extern "system" {
    fn CreateWindowExW(
        ex_style: u32,
        class_name: *const u16,
        window_name: *const u16,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: isize,
        menu: isize,
        instance: isize,
        param: *mut core::ffi::c_void,
    ) -> isize;
    fn DestroyWindow(hwnd: isize) -> i32;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(module_name: *const u16) -> isize;
}

/// A fabricated Win32 handle pair, for the pure lowering tests.
pub(crate) fn win32(hwnd: isize, hinstance: Option<isize>) -> RawWindowHandle {
    let mut handle = Win32WindowHandle::new(
        core::num::NonZeroIsize::new(hwnd).expect("the test's hwnd is non-zero"),
    );
    handle.hinstance = hinstance.and_then(core::num::NonZeroIsize::new);
    RawWindowHandle::Win32(handle)
}

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A hidden top-level window, destroyed when dropped.
pub(crate) struct TestWindow {
    hwnd: isize,
    hinstance: isize,
}

impl TestWindow {
    /// Creates a hidden `STATIC` window, or `None` where the session has no
    /// window station that can back one.
    ///
    /// `STATIC` is a system class, so no class registration is needed; the
    /// window is never shown and exists only to have a real `HWND`.
    pub(crate) fn open() -> Option<Self> {
        let class = wide("STATIC");
        let title = wide("fluxel-rhi vulkan test");
        // SAFETY: both are the documented Win32 contract. The null module name
        // asks for this process's own module, and the two string pointers are
        // NUL-terminated locals that outlive the call.
        let hinstance = unsafe { GetModuleHandleW(core::ptr::null()) };
        if hinstance == 0 {
            return None;
        }
        // SAFETY: as above; the window's parent and menu are null, so no other
        // object is referenced.
        let hwnd = unsafe {
            CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                0x00CF_0000, // WS_OVERLAPPEDWINDOW, never shown
                0,
                0,
                64,
                64,
                0,
                0,
                hinstance,
                core::ptr::null_mut(),
            )
        };
        (hwnd != 0).then_some(Self { hwnd, hinstance })
    }

    pub(crate) fn raw(&self) -> RawWindowHandle {
        win32(self.hwnd, Some(self.hinstance))
    }
}

impl Drop for TestWindow {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the handle `open` created.
        unsafe { DestroyWindow(self.hwnd) };
    }
}

/// The physical device that actually owns `surface`, with the facts it reported, or
/// `None` where no enumerated adapter reports a presentable surface.
///
/// A multi-adapter machine can enumerate an adapter the window is not attached
/// to; `Vulkan` answers that with an empty format list rather than a driver
/// error, so the first adapter that reports a format is the one this surface is
/// on.
pub(crate) fn presenting_adapter(
    surface: &Surface<'_>,
    adapters: &[vk::PhysicalDevice],
) -> Option<(vk::PhysicalDevice, SurfaceFacts)> {
    adapters.iter().find_map(|adapter| {
        let facts = surface.facts(*adapter).ok()?;
        (!facts.formats.is_empty()).then_some((*adapter, facts))
    })
}

/// A real colour target and the framebuffer a raster pass begins over it.
///
/// Returns the framebuffer, the image a transition must move into the pass's initial
/// layout, and that image's mapped `Vulkan` format. The recorder's draw tests and the
/// bind-group tests both need a real pass to record into, so the fixture is written
/// once here rather than copied into each module, which is section 37.5's rule applied
/// to the pass target.
pub(crate) fn colour_target_pass(
    opened: &OpenedVulkan,
    table: &mut ResourceTable,
    memory_types: &[vk::MemoryType],
) -> (Framebuffer, vk::Image, vk::Format) {
    let target = table
        .create_texture(
            TextureDesc {
                dimension: TextureDimension::D2,
                extent: Extent3d {
                    width: 16,
                    height: 8,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
            },
            TextureUsage::from_kinds([TextureUsageKind::ColorAttachment]),
            memory_types,
            memory::MemoryPurpose::DeviceLocal,
        )
        .expect("a device-local colour target");
    let colors = [RasterColorAttachment {
        index: 0,
        texture: &target,
        range: TextureRange::Whole,
        operations: AttachmentOps {
            load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Full,
        },
    }];
    let admitted = render_pass::admit(&colors, None).expect("one colour at index zero");
    let resolved = table
        .texture_desc(*admitted.texture)
        .expect("a live texture");
    let view = table.texture_view(*admitted.texture).expect("a live view");
    let image = table.texture_image(*admitted.texture).expect("a live image");
    let mapped = format::image_format(resolved.format).expect("a mapped format");
    let framebuffer = Framebuffer::create(opened.device.device(), admitted, &resolved, view)
        .expect("a render pass and framebuffer for a real colour target");
    (framebuffer, image, mapped)
}

/// The retained colour-only raster state: one `Rgba8Unorm` target written in full,
/// one sample, triangle lists and no culling.
pub(crate) fn colour_only_state() -> PipelineState {
    PipelineState {
        primitive: PrimitiveState::triangle_list(),
        depth_stencil: None,
        sample_count: 1,
        color_targets: vec![ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            write_mask: ColorWriteMask::ALL,
        }],
    }
}

/// The position-only vertex stream of the retained `Float32x3` recipes.
pub(crate) fn position_stream() -> VertexLayout {
    VertexLayout {
        buffers: vec![VertexBufferLayout {
            slot: 0,
            stride: 12,
            step_mode: VertexStepMode::Vertex,
        }],
        attributes: vec![VertexAttribute {
            location: 0,
            buffer_slot: 0,
            format: VertexFormat::Float32x3,
            offset: 0,
        }],
    }
}

/// The vertex and fragment modules a minimal raster pipeline is created from.
pub(crate) fn raster_shaders() -> RasterShaders<'static> {
    RasterShaders {
        vertex: &MINIMAL_RASTER_VERTEX_SPIRV,
        vertex_entry: c"main",
        fragment: &MINIMAL_RASTER_FRAGMENT_SPIRV,
        fragment_entry: c"main",
    }
}

/// The vertex module of a minimal drawable raster recipe.
///
/// Emitted once by Naga 30's `spv-out` for exactly this WGSL, targeting SPIR-V 1.0:
///
/// ```text
/// @vertex
/// fn main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
///     return vec4<f32>(position, 1.0);
/// }
/// ```
///
/// It is a fixed payload independent of the Naga lowering path, exactly as
/// [`MINIMAL_COMPUTE_SPIRV`](super::shader::MINIMAL_COMPUTE_SPIRV) is for the compute
/// half: the pipeline and recorder tests exercise creation and recording rather than
/// [`super::wgsl`]. Because Naga emitted it, the words are valid SPIR-V without a
/// second assembler in this repository.
pub(crate) const MINIMAL_RASTER_VERTEX_SPIRV: [u32; 129] = [
    0x0723_0203, 0x0001_0000, 0x0000_001c, 0x0000_0017, 0x0000_0000, 0x0002_0011,
    0x0000_0001, 0x0006_000b, 0x0000_0001, 0x4c53_4c47, 0x6474_732e, 0x3035_342e,
    0x0000_0000, 0x0003_000e, 0x0000_0000, 0x0000_0001, 0x0007_000f, 0x0000_0000,
    0x0000_000c, 0x6e69_616d, 0x0000_0000, 0x0000_0007, 0x0000_000a, 0x0005_0005,
    0x0000_0007, 0x6973_6f70, 0x6e6f_6974, 0x0000_0000, 0x0004_0005, 0x0000_000c,
    0x6e69_616d, 0x0000_0000, 0x0004_0047, 0x0000_0007, 0x0000_001e, 0x0000_0000,
    0x0004_0047, 0x0000_000a, 0x0000_000b, 0x0000_0000, 0x0002_0013, 0x0000_0002,
    0x0003_0016, 0x0000_0004, 0x0000_0020, 0x0004_0017, 0x0000_0003, 0x0000_0004,
    0x0000_0003, 0x0004_0017, 0x0000_0005, 0x0000_0004, 0x0000_0004, 0x0004_0020,
    0x0000_0008, 0x0000_0001, 0x0000_0003, 0x0004_003b, 0x0000_0008, 0x0000_0007,
    0x0000_0001, 0x0004_0020, 0x0000_000b, 0x0000_0003, 0x0000_0005, 0x0004_003b,
    0x0000_000b, 0x0000_000a, 0x0000_0003, 0x0003_0021, 0x0000_000d, 0x0000_0002,
    0x0004_002b, 0x0000_0004, 0x0000_000e, 0x3f80_0000, 0x0004_0020, 0x0000_0011,
    0x0000_0003, 0x0000_0004, 0x0004_0015, 0x0000_0013, 0x0000_0020, 0x0000_0000,
    0x0004_002b, 0x0000_0013, 0x0000_0012, 0x0000_0001, 0x0005_0036, 0x0000_0002,
    0x0000_000c, 0x0000_0000, 0x0000_000d, 0x0002_00f8, 0x0000_0006, 0x0004_003d,
    0x0000_0003, 0x0000_0009, 0x0000_0007, 0x0002_00f9, 0x0000_000f, 0x0002_00f8,
    0x0000_000f, 0x0005_0050, 0x0000_0005, 0x0000_0010, 0x0000_0009, 0x0000_000e,
    0x0003_003e, 0x0000_000a, 0x0000_0010, 0x0005_0041, 0x0000_0011, 0x0000_0014,
    0x0000_000a, 0x0000_0012, 0x0004_003d, 0x0000_0004, 0x0000_0015, 0x0000_0014,
    0x0004_007f, 0x0000_0004, 0x0000_0016, 0x0000_0015, 0x0003_003e, 0x0000_0014,
    0x0000_0016, 0x0001_00fd, 0x0001_0038,
];

/// The fragment module of the same minimal recipe, from this WGSL:
///
/// ```text
/// @fragment
/// fn main() -> @location(0) vec4<f32> {
///     return vec4<f32>(1.0, 0.0, 0.0, 1.0);
/// }
/// ```
pub(crate) const MINIMAL_RASTER_FRAGMENT_SPIRV: [u32; 84] = [
    0x0723_0203, 0x0001_0000, 0x0000_001c, 0x0000_000e, 0x0000_0000, 0x0002_0011,
    0x0000_0001, 0x0006_000b, 0x0000_0001, 0x4c53_4c47, 0x6474_732e, 0x3035_342e,
    0x0000_0000, 0x0003_000e, 0x0000_0000, 0x0000_0001, 0x0006_000f, 0x0000_0004,
    0x0000_0008, 0x6e69_616d, 0x0000_0000, 0x0000_0006, 0x0003_0010, 0x0000_0008,
    0x0000_0007, 0x0004_0005, 0x0000_0008, 0x6e69_616d, 0x0000_0000, 0x0004_0047,
    0x0000_0006, 0x0000_001e, 0x0000_0000, 0x0002_0013, 0x0000_0002, 0x0003_0016,
    0x0000_0004, 0x0000_0020, 0x0004_0017, 0x0000_0003, 0x0000_0004, 0x0000_0004,
    0x0004_0020, 0x0000_0007, 0x0000_0003, 0x0000_0003, 0x0004_003b, 0x0000_0007,
    0x0000_0006, 0x0000_0003, 0x0003_0021, 0x0000_0009, 0x0000_0002, 0x0004_002b,
    0x0000_0004, 0x0000_000a, 0x3f80_0000, 0x0004_002b, 0x0000_0004, 0x0000_000b,
    0x0000_0000, 0x0007_002c, 0x0000_0003, 0x0000_000c, 0x0000_000a, 0x0000_000b,
    0x0000_000b, 0x0000_000a, 0x0005_0036, 0x0000_0002, 0x0000_0008, 0x0000_0000,
    0x0000_0009, 0x0002_00f8, 0x0000_0005, 0x0002_00f9, 0x0000_000d, 0x0002_00f8,
    0x0000_000d, 0x0003_003e, 0x0000_0006, 0x0000_000c, 0x0001_00fd, 0x0001_0038,
];
