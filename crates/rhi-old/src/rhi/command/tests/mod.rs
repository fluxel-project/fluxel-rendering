//! Contract tests for [`crate::rhi::command`], run against the portable mock.
//!
//! Every test here is a rule from rhi-design sections 29 and 32 through 38,
//! exercised through the recorder's own API over [`Mock`]. The mock executes
//! nothing, so what these tests prove is what the *library* guarantees before a
//! backend is involved: which commands are refused, which failures poison a
//! recorder, and which resource uses a recording actually declares.
//!
//! The mock's journal is what makes the two failure policies distinguishable. A
//! command the recorder refused before reaching a backend leaves no journal
//! entry; a command a backend refused leaves one. Several tests below assert
//! exactly that, because the consequences differ: the first rejects one command
//! and leaves the recorder usable, the second poisons it.
//!
//! # What cannot be tested here
//!
//! Three section 29.2 rules are unreachable at runtime because the type system
//! already forbids them: a second `begin_raster` without `end_raster`, a draw
//! outside a raster scope, and `finish` from inside a scope. `RasterScope<'a>`
//! and `ComputeScope<'a>` hold `&'a mut CommandRecorder`, so those programs do
//! not compile and there is no runtime state to observe. Their reachable
//! equivalents — a scope dropped without `end()`, a scope whose `end()` is
//! refused — are tested instead.

use std::sync::Arc;

use crate::rhi::binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayoutDescriptor,
    BindGroupLayout, BindingCount, BindingKind, BindingResource, BindingSlot, BindingSlotId,
};
use crate::rhi::command::{
    BufferCopy, BufferTextureCopy, ColorAttachment, ColorAttachmentView, CommandRecorder,
    ComputeScope, IndexFormat, RasterScope, RasterScopeDescriptor, RecorderDescriptor,
    TextureCopy, TextureResolve,
};
use crate::rhi::diagnostics::DiagnosticSeverity;
use crate::rhi::format::{LaneWorkDomains, TextureFormat};
use crate::rhi::graph_bridge::{AccessMask, ResourceUse};
use crate::rhi::mock::Mock;
use crate::rhi::pipeline::{
    ColorTargetState, ComputePipelineDescriptor, PipelineInterfaceDescriptor, PipelineInterface,
    PrimitiveState, PrimitiveTopology, RasterPipeline, RasterPipelineDescriptor, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexInputState, VertexStepMode,
};
use crate::rhi::platform::RhiErrorKind;
use crate::rhi::resource::{
    Buffer, BufferBinding, BufferRange, BufferUsage, Extent3d, Origin3d, TextureAspect,
    TextureDescriptor, TextureSubresourceLayers, TextureUsage, TextureView,
};
use crate::rhi::shader::{
    ArtifactHash, ArtifactProducerId, ArtifactProducerVersion, ComputeWorkgroupRequirements,
    ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderInterface, ShaderLocation,
    ShaderLocationInterface, ShaderModule, ShaderNumericType, ShaderRequirements,
    ShaderResourceRequirement, ShaderStage, ShaderStages,
};

/// The color attachment format every texture-attachment test uses.
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// The binding shape every uniform-buffer test uses.
const UNIFORM_MIN_SIZE: u64 = 64;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A shader module for `interface`, accepted by the mock's capability database.
///
/// The mock declares the Vulkan backend family and the SpirV code format at the
/// current ABI, so an artifact built that way is accepted. Anything else would
/// be refused by `Device::create_shader`, which is the acceptance rule and not
/// a mock quirk.
fn shader(mock: &Mock, stage: ShaderStage, interface: ShaderInterface) -> ShaderModule {
    let requirements = match stage {
        ShaderStage::Compute => ShaderRequirements::new()
            .with_compute_workgroup(ComputeWorkgroupRequirements::new(1, 1, 1, 1, 0)),
        ShaderStage::Vertex | ShaderStage::Fragment => ShaderRequirements::new(),
    };
    let artifact = ShaderArtifact::new(
        stage,
        "main",
        ShaderCode::SpirV(Arc::from(vec![0u32; 4])),
        ShaderAbiVersion::CURRENT,
        interface,
        requirements,
        ArtifactHash([0u8; 32]),
        ArtifactProducerId("fluxel-rhi-mock-contract-tests".to_string()),
        ArtifactProducerVersion { major: 1, minor: 0 },
    );
    mock.device()
        .create_shader(&artifact)
        .expect("the mock device accepts a current-ABI SpirV artifact")
}

/// A fragment output at location 0 carrying a Float32 rgba.
fn rgba_output() -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(0),
        numeric_type: ShaderNumericType::Float32,
        components: 4,
        interpolation: None,
    }
}

/// A vertex shader that writes clip position and reads no bindings.
fn plain_vertex(mock: &Mock) -> ShaderModule {
    shader(
        mock,
        ShaderStage::Vertex,
        ShaderInterface::new().with_writes_position(true),
    )
}

/// A fragment shader that writes a Float32 rgba at location 0.
fn rgba_fragment(mock: &Mock) -> ShaderModule {
    shader(
        mock,
        ShaderStage::Fragment,
        ShaderInterface::new().with_output(rgba_output()),
    )
}

/// An interface over no groups.
fn empty_interface(mock: &Mock) -> PipelineInterface {
    mock.device()
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(Vec::new()))
        .expect("an interface with no groups is legal")
}

/// A raster pipeline over one color target in `format`, with no vertex fetch.
fn plain_pipeline(mock: &Mock, format: TextureFormat) -> RasterPipeline {
    let descriptor = RasterPipelineDescriptor::new(plain_vertex(mock), empty_interface(mock))
        .with_fragment(rgba_fragment(mock))
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(format));
    mock.device()
        .create_raster_pipeline(&descriptor)
        .expect("the mock device accepts a minimal raster pipeline")
}

/// A raster pipeline whose vertex stage fetches from vertex slot 0.
fn fetch_pipeline(mock: &Mock, format: TextureFormat) -> RasterPipeline {
    let vertex = shader(
        mock,
        ShaderStage::Vertex,
        ShaderInterface::new()
            .with_input(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 2,
                interpolation: None,
            })
            .with_writes_position(true),
    );
    let descriptor = RasterPipelineDescriptor::new(vertex, empty_interface(mock))
        .with_fragment(rgba_fragment(mock))
        .with_vertex_input(
            VertexInputState::new().with_buffer(
                VertexBufferLayout::new(8, VertexStepMode::Vertex).with_attribute(
                    VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x2, 0),
                ),
            ),
        )
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(format));
    mock.device()
        .create_raster_pipeline(&descriptor)
        .expect("the mock device accepts a one-slot vertex fetch pipeline")
}

/// A raster pipeline built for a triangle strip with 16-bit strip indices.
fn strip_pipeline(mock: &Mock, format: TextureFormat) -> RasterPipeline {
    let descriptor = RasterPipelineDescriptor::new(plain_vertex(mock), empty_interface(mock))
        .with_fragment(rgba_fragment(mock))
        .with_primitive(
            PrimitiveState::new(PrimitiveTopology::TriangleStrip)
                .with_strip_index_format(IndexFormat::Uint16),
        )
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(format));
    mock.device()
        .create_raster_pipeline(&descriptor)
        .expect("the mock device accepts a strip pipeline with a strip index format")
}

/// The layout of `SLOTS` uniform-buffer slots, all visible to both graphics stages.
fn uniform_layout(mock: &Mock, slots: u32) -> BindGroupLayout {
    let visibility = ShaderStages::VERTEX.union(ShaderStages::FRAGMENT);
    let entries = (0..slots)
        .map(|slot| {
            BindingSlot::new(
                BindingSlotId::new(slot),
                visibility,
                BindingKind::UniformBuffer {
                    min_size: UNIFORM_MIN_SIZE,
                },
            )
        })
        .collect();
    mock.device()
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(entries))
        .expect("the mock device accepts ten uniform-buffer slots")
}

/// The layout of `slots` uniform-buffer slots visible to the compute stage.
///
/// Visibility is per stage and a slot that is not visible to a stage the shader
/// runs in is `IncompatibleInterface`, so the compute tests need their own
/// layout rather than the graphics one.
fn compute_uniform_layout(mock: &Mock, slots: u32) -> BindGroupLayout {
    let entries = (0..slots)
        .map(|slot| {
            BindingSlot::new(
                BindingSlotId::new(slot),
                ShaderStages::COMPUTE,
                BindingKind::UniformBuffer {
                    min_size: UNIFORM_MIN_SIZE,
                },
            )
        })
        .collect();
    mock.device()
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(entries))
        .expect("the mock device accepts a compute-visible uniform layout")
}

/// A shader requirement for one uniform buffer in group 0.
fn uniform_requirement(slot: u32) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(slot),
        kind: BindingKind::UniformBuffer {
            min_size: UNIFORM_MIN_SIZE,
        },
        count: BindingCount::One,
    }
}

/// A raster pipeline whose vertex reads slots 0 and 1 and whose fragment reads
/// slots 1 and 2 of a ten-slot layout, so the merge is observable.
///
/// The layout, the interface, and the pipeline are returned together because the
/// bind group a test binds must be built over exactly the layout the interface
/// declares; that is what `check_shader_groups` compares.
fn selective_pipeline(mock: &Mock) -> (BindGroupLayout, RasterPipeline) {
    let layout = uniform_layout(mock, 10);
    let vertex = shader(
        mock,
        ShaderStage::Vertex,
        ShaderInterface::new()
            .with_resource(uniform_requirement(0))
            .with_resource(uniform_requirement(1))
            .with_writes_position(true),
    );
    let fragment = shader(
        mock,
        ShaderStage::Fragment,
        ShaderInterface::new()
            .with_resource(uniform_requirement(1))
            .with_resource(uniform_requirement(2))
            .with_output(rgba_output()),
    );
    let interface = mock
        .device()
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))
        .expect("the mock device accepts a one-group interface");
    let descriptor = RasterPipelineDescriptor::new(vertex, interface)
        .with_fragment(fragment)
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(COLOR_FORMAT));
    let pipeline = mock
        .device()
        .create_raster_pipeline(&descriptor)
        .expect("the mock device accepts the selective pipeline");
    (layout, pipeline)
}

/// A bind group over `layout` filling every slot with a distinct uniform buffer.
fn uniform_group(mock: &Mock, layout: &BindGroupLayout, slots: u32) -> crate::rhi::binding::BindGroup {
    let entries = (0..slots).map(|slot| {
        let buffer = mock.buffer(UNIFORM_MIN_SIZE, BufferUsage::UNIFORM);
        BindGroupEntry::new(
            BindingSlotId::new(slot),
            BindingResource::Buffer(BufferBinding::new(
                buffer,
                BufferRange::new(0, UNIFORM_MIN_SIZE),
            )),
        )
    });
    mock.device()
        .create_bind_group(&BindGroupDescriptor::new(layout.clone()).with_entries(entries))
        .expect("the mock device accepts a full uniform packet")
}

/// A color-attachment view over a fresh four-by-four texture.
fn color_view(mock: &Mock) -> TextureView {
    let texture = mock
        .device()
        .create_texture(&TextureDescriptor::new_2d(
            4,
            4,
            COLOR_FORMAT,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ))
        .expect("the mock device creates a color attachment texture");
    mock.view(&texture)
}

/// A raster scope descriptor with one color attachment at location 0.
fn one_color_scope(view: &TextureView) -> RasterScopeDescriptor {
    RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment::new(ColorAttachmentView::Texture(view.clone())),
    )
}

/// A whole-texture subresource of mip 0, layer 0, color aspect.
fn color_layers() -> TextureSubresourceLayers {
    TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    }
}

/// A source buffer and a destination buffer of `size` bytes each.
fn copy_pair(mock: &Mock, size: u64) -> (Buffer, Buffer) {
    (
        mock.buffer(size, BufferUsage::COPY_SRC),
        mock.buffer(size, BufferUsage::COPY_DST),
    )
}

/// A half-open range built from values rather than written as a literal.
///
/// The inverted-range tests below deliberately pass `start > end`, which is the
/// input under test. Written as a literal, `3..1` trips clippy's
/// `reversed_empty_ranges`, a deny-by-default lint about iterating nothing; the
/// range is never iterated here, so it is constructed as a value instead.
fn range(start: u32, end: u32) -> core::ops::Range<u32> {
    core::ops::Range { start, end }
}

/// How many times the journal contains exactly `entry`.
fn journal_count(mock: &Mock, entry: &str) -> usize {
    mock.journal()
        .iter()
        .filter(|item| item.as_str() == entry)
        .count()
}

/// A compute scope over `recorder`, for the tests that only need a scope.
fn open_compute<'a>(recorder: &'a mut CommandRecorder, mock: &Mock) -> ComputeScope<'a> {
    let _ = mock;
    recorder
        .begin_compute()
        .expect("the mock device enables the compute feature")
}


// ---------------------------------------------------------------------------
// Test modules
// ---------------------------------------------------------------------------
//
// The tests are split by the design section they cover rather than kept in one
// file: a single file for all of them would exceed the size at which a module
// has to prove it is one responsibility, and "recording lifecycle", "generated
// resource uses", "draw validation", and "copy validation" are four separate
// ones. Everything they share lives here.

mod copies;
mod draws;
mod lifecycle;
mod uses;
