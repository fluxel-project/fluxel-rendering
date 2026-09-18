//! Step 5's pipeline half: the pipeline layout and the compute pipeline.
//!
//! # The pipeline owns the layout it was created with
//!
//! In `Vulkan` a `VkPipeline` refers to its `VkPipelineLayout` when it is bound, so
//! destroying the layout before the pipeline is invalid. The layout is therefore an
//! owned field of [`ComputePipeline`] rather than a borrow a caller has to remember
//! to keep alive, and the two are destroyed together in the order the dependency
//! requires: the pipeline first, then the layout it refers to.
//!
//! # The shader module is transient
//!
//! `Vulkan` reads a module when a pipeline is created and never again, so a module
//! may be destroyed as soon as its pipeline exists. [`create_compute`] creates one,
//! records it into the create-info, and lets it drop on the way out of the call --
//! after `vkCreateComputePipelines` returned and not before. That order is a scope
//! rather than a comment, because [`shader::Module`] destroys its handle in `Drop`.
//!
//! # What the layout carries today, and what it will carry
//!
//! The descriptor set layouts are owned by the pipeline layout rather than borrowed
//! from the caller. `Vulkan` requires every set layout referenced by a pipeline
//! layout to outlive it, so the only shape that cannot be misused is the
//! longer-lived object containing the shorter ones: [`PipelineLayout`] holds the
//! [`SetLayout`]s it was created over, and field order destroys the pipeline layout
//! before them. Pass an empty vector for a shader that declares no bindings.
//!
//! No push constant ranges yet: a push constant range would claim a mechanism no
//! retained recipe uses. The borrowed path being replaced passes an immediate-data
//! size of zero, which is "no push constants" here.
//!
//! # A failed creation is not partially usable
//!
//! `vkCreateComputePipelines` reports failure with the contents of its output array
//! **undefined**, and the specification directs an application not to use them. The
//! partially filled vector `ash` returns on that path is therefore dropped without
//! being read. That is deliberate: destroying a handle the driver did not promise to
//! have created would be worse than not destroying it.
//!
//! # The raster half, and the three states it pins
//!
//! [`create_raster`] builds the graphics pipeline for a fixed-function description
//! from [`crate::common::pipeline`] over a
//! [`crate::common::vertex::VertexLayout`]. Three facts about that lowering are
//! worth stating, because each is a place a fresh implementation quietly claims
//! something:
//!
//! 1. **Viewport and scissor are dynamic.** The graphics family's `set_viewport`
//!    and `set_scissor` are the two verbs that write them, so a pipeline that baked
//!    them in would silently ignore those verbs. The viewport state therefore
//!    declares one viewport and one scissor with null pointers, which is the legal
//!    "both are dynamic" shape; a zero count would be invalid.
//! 2. **The depth-stencil state is present exactly when the description names a
//!    depth-stencil attachment**, because `Vulkan` validates a pipeline against the
//!    subpass it is created for: without the state it is invalid against a render
//!    pass that has a depth attachment, and with it, against one that does not.
//! 3. **Everything the portable description does not state is pinned to the value
//!    that claims nothing** -- no blending, no logic op, no sample shading, no depth
//!    bias, no stencil, fill polygon mode, a line width of one and the default
//!    sample mask. Leaving a field at `ash`'s default would be an unproved claim:
//!    `PipelineColorBlendAttachmentState::default()` writes **no** colour
//!    components, so a lowering that forgot the write mask would produce a pipeline
//!    that renders nothing and still creates successfully.
//!
//! # The render pass is a creation companion
//!
//! The device this backend opens is a `Vulkan` 1.0 device with no extensions, so
//! there is no dynamic-rendering path: `vkCreateGraphicsPipelines` needs a render
//! pass. [`create_raster`] therefore creates one from the pipeline's attachment
//! signature and destroys it as soon as the pipeline exists. That is sound because
//! a pipeline does not refer to a render pass after creation -- `vkDestroyRenderPass`
//! requires only that submitted commands referring to it have completed -- and the
//! render passes the recording step begins are *compatible* with this pipeline when
//! their attachment formats, sample counts and reference layouts agree, which is
//! exactly what [`signature`] states. The creation render pass's load and store
//! operations are `DONT_CARE` on purpose: contents operations belong to the compiled
//! pass (RenderGraph's `AttachmentOps`), not to a pipeline, and they do not
//! participate in render pass compatibility.

use core::ffi::CStr;

use ash::vk;
use fluxel_rendergraph::TextureFormat;

use crate::common::pipeline::{
    ColorWriteMask, CullMode, FrontFace, PipelineState, PipelineStateError, PrimitiveTopology,
};
use crate::common::vertex::{VertexFormat, VertexLayout, VertexLayoutError, VertexStepMode};
use crate::shader_contract::ShaderStage;

use super::descriptor::SetLayout;
use super::format;
use super::render_pass;
use super::sampler;
use super::shader::{self, Module, ShaderError};

/// Why a pipeline layout or a pipeline could not be created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PipelineError {
    /// The driver refused to create the pipeline layout.
    Layout(vk::Result),
    /// The shader module could not be described or created.
    Shader(ShaderError),
    /// The driver refused to create the pipeline.
    Creation(vk::Result),
    /// The vertex-input description cannot describe a pipeline.
    VertexLayout(VertexLayoutError),
    /// The fixed-function description cannot describe a pipeline.
    State(PipelineStateError),
    /// A portable format this backend has not been taught.
    UnsupportedFormat(TextureFormat),
    /// A colour target named a format that carries depth rather than colour.
    ColorTargetFormatIsDepth {
        /// Which target named it.
        index: usize,
        /// The format that was named.
        format: TextureFormat,
    },
    /// A depth-stencil state named a format that carries no depth.
    DepthStencilFormatNotDepth(TextureFormat),
    /// The sample count is not one of the counts `Vulkan` names.
    UnsupportedSampleCount(u32),
    /// The driver refused to create the render pass the pipeline is created against.
    RenderPass(vk::Result),
}

/// A `VkPipelineLayout` this backend owns exactly once.
pub(crate) struct PipelineLayout {
    device: ash::Device,
    handle: vk::PipelineLayout,
    /// The descriptor set layouts this pipeline layout refers to at bind time.
    ///
    /// Owned, not borrowed: `Vulkan` requires a set layout to outlive every pipeline
    /// layout that names it, and the declaration order here is what destroys the
    /// pipeline layout before them. An empty vector is a shader declaring no
    /// bindings and claims nothing.
    set_layouts: Vec<SetLayout>,
}

impl PipelineLayout {
    /// Returns the driver handle a pipeline is created against.
    pub(crate) const fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        // SAFETY: the handle was created by this device and is destroyed once here.
        // The device outlives this layout because the caller that created it holds
        // both, and every pipeline created against this layout is destroyed first.
        // The set layouts are released by field order *after* this body, which is
        // the order Vulkan requires: the pipeline layout names them, not the reverse.
        unsafe { self.device.destroy_pipeline_layout(self.handle, None) };
    }
}

impl core::fmt::Debug for PipelineLayout {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PipelineLayout")
            .field("handle", &self.handle)
            .field("set_layout_count", &self.set_layouts.len())
            .finish_non_exhaustive()
    }
}

/// A `vk::Pipeline` for a compute kernel, with the layout it refers to.
pub(crate) struct ComputePipeline {
    device: ash::Device,
    handle: vk::Pipeline,
    /// The layout this pipeline refers to at bind time.
    ///
    /// An owned field, not a borrow: a pipeline that outlives its layout is invalid,
    /// so the only safe shape is for the longer-lived object to contain the shorter.
    layout: PipelineLayout,
}

impl ComputePipeline {
    /// Returns the driver handle the compute recorder binds.
    pub(crate) const fn handle(&self) -> vk::Pipeline {
        self.handle
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        // SAFETY: the handle was created by this device and is destroyed once here.
        // The layout field is released by field order *after* this body, which is the
        // order Vulkan requires: the pipeline refers to the layout, not the reverse.
        unsafe { self.device.destroy_pipeline(self.handle, None) };
    }
}

impl core::fmt::Debug for ComputePipeline {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ComputePipeline")
            .field("handle", &self.handle)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// Creates a pipeline layout over `set_layouts`.
///
/// The set layouts are consumed rather than borrowed because the resulting pipeline
/// layout names them and they must outlive it; see the module docs. An empty vector
/// is legal and is what a shader that declares no bindings asks for.
pub(crate) fn create_layout(
    device: &ash::Device,
    set_layouts: Vec<SetLayout>,
) -> Result<PipelineLayout, PipelineError> {
    // The create-info borrows handles for the duration of the call only: `Vulkan`
    // copies them into the layout it creates. The owners are moved into the
    // returned value, which is what keeps the handles alive.
    let handles: Vec<vk::DescriptorSetLayout> =
        set_layouts.iter().map(SetLayout::handle).collect();
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&handles);
    // SAFETY: the device is live; every set layout is a live handle this device
    // created and is moved into the returned value, and the handle slice outlives
    // the call.
    let handle = unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(PipelineError::Layout)?;
    Ok(PipelineLayout {
        device: device.clone(),
        handle,
        set_layouts,
    })
}

/// Creates a compute pipeline from `words`, entering `entry_point`, over `layout`.
///
/// `layout` is consumed rather than borrowed because the resulting pipeline keeps it
/// alive; see the module docs. `words` is the same SPIR-V payload
/// [`shader::create_module`] describes, and is validated before the driver sees it.
pub(crate) fn create_compute(
    device: &ash::Device,
    layout: PipelineLayout,
    words: &[u32],
    entry_point: &CStr,
) -> Result<ComputePipeline, PipelineError> {
    // The module lives exactly as long as it is needed: the driver reads it during
    // creation, and it is destroyed by its own drop on the way out of this call.
    let module: Module =
        shader::create_module(device, words).map_err(PipelineError::Shader)?;
    let stage = shader::stage(ShaderStage::Compute, module.handle(), entry_point);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout.handle());
    // The create-info slice must outlive the call, so it is a binding rather than an
    // inline array literal.
    let infos = [info];
    // SAFETY: the device is live; the layout and module are live handles this device
    // created; the create-info borrows only locals that outlive the call, and a null
    // pipeline cache is the valid "no cache" value.
    let created = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
    };
    let handle = match created {
        // One create-info yields exactly one pipeline.
        Ok(mut pipelines) => pipelines
            .pop()
            .expect("one create-info yields one pipeline"),
        // The output array is undefined on failure and is deliberately not read;
        // see the module docs.
        Err((_undefined, error)) => return Err(PipelineError::Creation(error)),
    };
    Ok(ComputePipeline {
        device: device.clone(),
        handle,
        layout,
    })
}

/// The two shader modules one raster pipeline is created from.
pub(crate) struct RasterShaders<'a> {
    /// The SPIR-V words of the vertex module.
    pub vertex: &'a [u32],
    /// The entry point the vertex stage enters.
    pub vertex_entry: &'a CStr,
    /// The SPIR-V words of the fragment module.
    pub fragment: &'a [u32],
    /// The entry point the fragment stage enters.
    pub fragment_entry: &'a CStr,
}

/// A `vk::Pipeline` for a raster draw, with the layout it refers to.
///
/// The layout is an owned field for [`ComputePipeline`]'s reason: a pipeline refers
/// to its layout at bind time, so the only shape that cannot be misused is the
/// longer-lived object containing the shorter one, and field order destroys the
/// pipeline before its layout.
pub(crate) struct RasterPipeline {
    device: ash::Device,
    handle: vk::Pipeline,
    /// The layout this pipeline refers to at bind time.
    layout: PipelineLayout,
}

impl RasterPipeline {
    /// Returns the driver handle the raster recorder binds.
    pub(crate) const fn handle(&self) -> vk::Pipeline {
        self.handle
    }
}

impl Drop for RasterPipeline {
    fn drop(&mut self) {
        // SAFETY: the handle was created by this device and is destroyed once here.
        // The layout field is released by field order *after* this body, which is
        // the order Vulkan requires: the pipeline refers to the layout, not the
        // reverse.
        unsafe { self.device.destroy_pipeline(self.handle, None) };
    }
}

impl core::fmt::Debug for RasterPipeline {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RasterPipeline")
            .field("handle", &self.handle)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// The attachment facts a raster pipeline and its render pass must agree on.
///
/// The formats are `Vulkan` formats rather than portable ones because that is what
/// both the render pass and the pipeline are built from; asking the portable enum
/// again here would be a second mapping to keep in step with
/// [`format::image_format`], which is the single place a portable format is
/// translated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RasterSignature {
    /// The colour attachment formats, in index order.
    pub colors: Vec<vk::Format>,
    /// The depth-stencil attachment format, when the description names one.
    pub depth_stencil: Option<vk::Format>,
    /// The sample count both the attachments and the pipeline are built for.
    pub samples: vk::SampleCountFlags,
}

/// The `Vulkan` primitive topology for one portable topology.
///
/// Exhaustive rather than `Option`-returning, because [`PrimitiveTopology`] is this
/// crate's own closed enum: a variant added later must be taught to this match at
/// compile time rather than falling through to no topology.
pub(crate) const fn topology(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        PrimitiveTopology::Points => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::Lines => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveTopology::Triangles => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

/// The `Vulkan` cull flags for one portable cull mode.
///
/// `None` is `NONE` rather than `FRONT_AND_BACK`: the portable mode says "discard
/// nothing", not "discard both", and folding the two would discard every primitive.
pub(crate) const fn cull_mode(mode: CullMode) -> vk::CullModeFlags {
    match mode {
        CullMode::None => vk::CullModeFlags::NONE,
        CullMode::Front => vk::CullModeFlags::FRONT,
        CullMode::Back => vk::CullModeFlags::BACK,
    }
}

/// The `Vulkan` front face for one portable winding order.
pub(crate) const fn front_face(face: FrontFace) -> vk::FrontFace {
    match face {
        FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
    }
}

/// The `Vulkan` attribute format for one portable vertex format.
///
/// Exhaustive for [`topology`]'s reason: the portable vertex vocabulary is closed
/// and owned by this crate.
pub(crate) const fn vertex_format(format: VertexFormat) -> vk::Format {
    match format {
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        // Normalized, so the stored integers reach the shader as `[0, 1]`, which is
        // what the portable type documents the format to mean.
        VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
    }
}

/// The `Vulkan` input rate for one portable step mode.
pub(crate) const fn input_rate(step_mode: VertexStepMode) -> vk::VertexInputRate {
    match step_mode {
        VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
        VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
    }
}

/// The `Vulkan` component mask for one portable write mask.
///
/// Folded from the four named components rather than cast from the portable
/// representation: a cast would make the portable bitset's bit order a second,
/// unwritten statement about `Vulkan`'s, and the two would silently diverge if
/// either changed.
pub(crate) fn color_write_mask(mask: ColorWriteMask) -> vk::ColorComponentFlags {
    let mut flags = vk::ColorComponentFlags::empty();
    if mask.contains(ColorWriteMask::R) {
        flags |= vk::ColorComponentFlags::R;
    }
    if mask.contains(ColorWriteMask::G) {
        flags |= vk::ColorComponentFlags::G;
    }
    if mask.contains(ColorWriteMask::B) {
        flags |= vk::ColorComponentFlags::B;
    }
    if mask.contains(ColorWriteMask::A) {
        flags |= vk::ColorComponentFlags::A;
    }
    flags
}

/// The `Vulkan` sample-count bit for a portable count, or `None` where `Vulkan`
/// names no such count.
///
/// The constants are matched rather than built from the raw number, because the
/// flag is a *bit set*, not a count: `SampleCountFlags::from_raw(3)` is a value
/// `Vulkan` does not define, and this match is what refuses it. The named counts
/// happen to equal their bit, so a cast would work for every legal value and fail
/// only for the illegal ones -- which is the worst possible place for a silent
/// acceptance.
pub(crate) const fn sample_count(count: u32) -> Option<vk::SampleCountFlags> {
    Some(match count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => return None,
    })
}

/// The binding descriptions a vertex layout lowers to.
pub(crate) fn vertex_bindings(layout: &VertexLayout) -> Vec<vk::VertexInputBindingDescription> {
    layout
        .buffers
        .iter()
        .map(|buffer| {
            vk::VertexInputBindingDescription::default()
                // The portable slot is the `Vulkan` binding number directly: both
                // name the same input index, and offsetting it would bind a stream
                // the shader never reads.
                .binding(buffer.slot)
                .stride(buffer.stride)
                .input_rate(input_rate(buffer.step_mode))
        })
        .collect()
}

/// The attribute descriptions a vertex layout lowers to.
pub(crate) fn vertex_attributes(layout: &VertexLayout) -> Vec<vk::VertexInputAttributeDescription> {
    layout
        .attributes
        .iter()
        .map(|attribute| {
            vk::VertexInputAttributeDescription::default()
                .location(attribute.location)
                .binding(attribute.buffer_slot)
                .format(vertex_format(attribute.format))
                .offset(attribute.offset)
        })
        .collect()
}

/// Lowers the attachment half of a description into the formats a render pass and
/// a pipeline are built from.
///
/// This is the pure half of [`create_raster`], and it is where a format or sample
/// count this backend has not been taught is refused -- before the driver is
/// reached, so the caller reads a sentence naming the value rather than a
/// validation error naming a handle. The two format refusals are deliberately
/// separate from [`PipelineState::validate`]: whether a format carries depth is a
/// backend fact, and `TextureFormat` is `#[non_exhaustive]`, so this layer is the
/// first place that *can* answer it.
pub(crate) fn signature(state: &PipelineState) -> Result<RasterSignature, PipelineError> {
    let samples = sample_count(state.sample_count)
        .ok_or(PipelineError::UnsupportedSampleCount(state.sample_count))?;
    let mut colors = Vec::with_capacity(state.color_targets.len());
    for (index, target) in state.color_targets.iter().enumerate() {
        let format = format::image_format(target.format)
            .ok_or(PipelineError::UnsupportedFormat(target.format))?;
        if format::is_depth(format) {
            return Err(PipelineError::ColorTargetFormatIsDepth {
                index,
                format: target.format,
            });
        }
        colors.push(format);
    }
    let depth_stencil = match state.depth_stencil {
        None => None,
        Some(depth) => {
            let format = format::image_format(depth.format)
                .ok_or(PipelineError::UnsupportedFormat(depth.format))?;
            if !format::is_depth(format) {
                return Err(PipelineError::DepthStencilFormatNotDepth(depth.format));
            }
            Some(format)
        }
    };
    Ok(RasterSignature {
        colors,
        depth_stencil,
        samples,
    })
}

/// The render pass one raster pipeline is created against.
///
/// It is a creation companion and not a field of [`RasterPipeline`]; the module
/// docs state why destroying it as soon as the pipeline exists is sound. Its
/// contents operations are `DONT_CARE` because the operations a pass actually
/// performs come from the graph's attachment operations, which are the recording
/// step's business.
struct RenderPass {
    device: ash::Device,
    handle: vk::RenderPass,
}

impl RenderPass {
    /// Creates one subpass whose attachments are `signature`'s.
    fn create(device: &ash::Device, signature: &RasterSignature) -> Result<Self, PipelineError> {
        let mut attachments = Vec::with_capacity(signature.colors.len() + 1);
        let mut colors = Vec::with_capacity(signature.colors.len());
        for format in &signature.colors {
            colors.push(
                vk::AttachmentReference::default()
                    .attachment(attachments.len() as u32)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            );
            // The one place a colour attachment's description is written, so the
            // creation pass and the recording pass cannot disagree about the format
            // or sample count `Vulkan` compares two passes by. Their contents
            // operations differ deliberately: creation performs nothing.
            attachments.push(render_pass::color_description(
                *format,
                signature.samples,
                vk::AttachmentLoadOp::DONT_CARE,
                vk::AttachmentStoreOp::DONT_CARE,
            ));
        }
        let depth_stencil = signature.depth_stencil.map(|format| {
            let reference = vk::AttachmentReference::default()
                .attachment(attachments.len() as u32)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(format)
                    .samples(signature.samples)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            );
            reference
        });
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&colors);
        if let Some(reference) = &depth_stencil {
            subpass = subpass.depth_stencil_attachment(reference);
        }
        let subpasses = [subpass];
        // No dependencies: the implicit external-to-first-subpass dependency is the
        // one Vulkan adds for a pass with a single subpass, and inventing an
        // explicit one here would state a synchronization the recording step owns.
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);
        // SAFETY: the device is live, every slice the create-info points at is a
        // local that outlives the call, and no allocation callbacks are supplied.
        let handle =
            unsafe { device.create_render_pass(&info, None) }.map_err(PipelineError::RenderPass)?;
        Ok(Self {
            device: device.clone(),
            handle,
        })
    }

    /// Returns the handle a graphics pipeline is created against.
    const fn handle(&self) -> vk::RenderPass {
        self.handle
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        // SAFETY: the handle was created by this device and is destroyed once here.
        // No command buffer has referred to it, because none exists yet.
        unsafe { self.device.destroy_render_pass(self.handle, None) };
    }
}

/// Creates a raster pipeline from `shaders`, `vertex_input` and `state`.
///
/// `layout` is consumed rather than borrowed because the resulting pipeline keeps it
/// alive, exactly as [`create_compute`] does. The vertex layout and the pipeline
/// state are validated before either shader module exists, so a refused description
/// reaches no driver call at all. Both modules are created inside this call and
/// destroyed on the way out of it, after `vkCreateGraphicsPipelines` has returned
/// and not before.
pub(crate) fn create_raster(
    device: &ash::Device,
    layout: PipelineLayout,
    shaders: &RasterShaders<'_>,
    vertex_input: &VertexLayout,
    state: &PipelineState,
) -> Result<RasterPipeline, PipelineError> {
    vertex_input
        .validate()
        .map_err(PipelineError::VertexLayout)?;
    state.validate().map_err(PipelineError::State)?;
    let signature = signature(state)?;

    // The modules live exactly as long as the driver needs them: it reads them
    // during creation, and each is destroyed by its own drop on the way out.
    let vertex_module =
        shader::create_module(device, shaders.vertex).map_err(PipelineError::Shader)?;
    let fragment_module =
        shader::create_module(device, shaders.fragment).map_err(PipelineError::Shader)?;
    let stages = [
        shader::stage(
            ShaderStage::Vertex,
            vertex_module.handle(),
            shaders.vertex_entry,
        ),
        shader::stage(
            ShaderStage::Fragment,
            fragment_module.handle(),
            shaders.fragment_entry,
        ),
    ];

    let bindings = vertex_bindings(vertex_input);
    let attributes = vertex_attributes(vertex_input);
    let vertex_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(topology(state.primitive.topology))
        // No retained recipe declares a restart index, and enabling restart would
        // change what an index value means.
        .primitive_restart_enable(false);

    // Viewport and scissor are dynamic; see the module docs. The counts are one
    // because `Vulkan` requires a non-zero count even when the pointers are null.
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        // Fill is the only polygon mode the portable vocabulary can express, and
        // line and point modes are named refusals.
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(cull_mode(state.primitive.cull_mode))
        .front_face(front_face(state.primitive.front_face))
        // Depth bias is not in the portable vocabulary, so bias is disabled and
        // its three factors are written as the zeroes that bias nothing: the
        // explicit values are what make this a lowering rather than a reliance on
        // `ash`'s default.
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0)
        // Zero is invalid for `Vulkan`; one is the width that changes nothing.
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(signature.samples)
        .sample_shading_enable(false)
        .min_sample_shading(0.0)
        // A null sample mask is the "every sample is covered" default, not an
        // absent requirement.
        .sample_mask(&[])
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    // One blend attachment per colour target. No retained recipe blends, so each
    // target is replaced; the factors are still written as the replace identity so
    // that enabling blending later is a visible edit rather than a default.
    let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = state
        .color_targets
        .iter()
        .map(|target| {
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ZERO)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
                // A colour write mask of zero writes nothing, which is `ash`'s
                // default and a pipeline that renders nothing; it is stated here.
                .color_write_mask(color_write_mask(target.write_mask))
        })
        .collect();
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&blend_attachments)
        .blend_constants([0.0; 4]);

    // The state is present exactly when the description names a depth-stencil
    // attachment; see the module docs.
    let depth_stencil = state.depth_stencil.map(|depth| {
        vk::PipelineDepthStencilStateCreateInfo::default()
            // The presence of the description is what enables the test: a depth
            // state that does not test depth is not one a recipe declares.
            .depth_test_enable(true)
            .depth_write_enable(depth.depth_write_enabled)
            .depth_compare_op(sampler::compare_function(depth.depth_compare))
            .depth_bounds_test_enable(false)
            // Stencil is refused by name; a pipeline that enabled it would claim a
            // state the pass model cannot reach.
            .stencil_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
    });

    let render_pass = RenderPass::create(device, &signature)?;
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_state)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout.handle())
        .render_pass(render_pass.handle())
        .subpass(0);
    if let Some(depth_stencil) = &depth_stencil {
        info = info.depth_stencil_state(depth_stencil);
    }
    // The create-info slice must outlive the call, so it is a binding rather than an
    // inline array literal.
    let infos = [info];
    // SAFETY: the device is live; the layout, the render pass and both modules are
    // live handles this device created; the create-info borrows only locals that
    // outlive the call, and a null pipeline cache is the valid "no cache" value.
    let created =
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &infos, None) };
    let handle = match created {
        // One create-info yields exactly one pipeline.
        Ok(mut pipelines) => pipelines
            .pop()
            .expect("one create-info yields one pipeline"),
        // The output array is undefined on failure and is deliberately not read;
        // see the module docs.
        Err((_undefined, error)) => return Err(PipelineError::Creation(error)),
    };
    // The pipeline does not refer to the render pass after creation; see the module
    // docs. Dropping it here rather than storing it keeps the pipeline from owning
    // an object whose operations belong to the recording step.
    drop(render_pass);
    Ok(RasterPipeline {
        device: device.clone(),
        handle,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::common::binding::{
        BindGroupLayout, BindGroupLayoutEntry, BindingKind, BufferBindingType, SamplerBindingType,
        ShaderVisibility, TextureSampleType, ViewDimension,
    };
    use crate::common::pipeline::{ColorTargetState, DepthStencilState};
    use crate::common::sampler::CompareFunction;
    use crate::native::vulkan::descriptor;
    use crate::native::vulkan::open;
    use crate::native::vulkan::shader::MINIMAL_COMPUTE_SPIRV;
    use crate::native::vulkan::test_support::{
        MINIMAL_RASTER_FRAGMENT_SPIRV, colour_only_state, position_stream, raster_shaders,
    };

    /// The exact bind-group layout the linear-clamp raster artifact declares.
    fn textured_frame() -> BindGroupLayout {
        let entry = |binding, visibility, kind| BindGroupLayoutEntry {
            binding,
            visibility,
            kind,
        };
        BindGroupLayout {
            entries: vec![
                entry(
                    0,
                    ShaderVisibility::VERTEX_FRAGMENT,
                    BindingKind::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(80),
                    },
                ),
                entry(
                    1,
                    ShaderVisibility::FRAGMENT,
                    BindingKind::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: ViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    ShaderVisibility::FRAGMENT,
                    BindingKind::Sampler(SamplerBindingType::Filtering),
                ),
            ],
        }
    }

    #[test]
    fn a_real_compute_pipeline_is_created_from_a_spirv_module_on_this_machine() {
        // Step 5's compute half against the real driver: the module is read during
        // creation, the pipeline refers to the layout, and dropping the pipeline
        // destroys both in the order Vulkan requires. Skips where no adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let layout =
            create_layout(opened.device.device(), Vec::new()).expect("an empty pipeline layout");
        let pipeline = create_compute(
            opened.device.device(),
            layout,
            &MINIMAL_COMPUTE_SPIRV,
            c"main",
        )
        .expect("a compute pipeline whose module is valid SPIR-V");
        assert_ne!(pipeline.handle(), vk::Pipeline::null());
        // The drop destroys the pipeline and then its layout; nothing else releases
        // either handle.
        drop(pipeline);
    }

    #[test]
    fn a_pipeline_layout_owns_the_descriptor_set_layouts_it_names() {
        // The dependency `Vulkan` enforces is set layout -> pipeline layout ->
        // pipeline, and the owner chain states it as fields rather than as a comment.
        // Dropping the pipeline releases all three handles in that order, so a
        // layout that outlived its set layouts cannot be written here.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let set = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid descriptor set layout");
        let layout = create_layout(opened.device.device(), vec![set])
            .expect("a pipeline layout over one set layout");
        let pipeline = create_compute(
            opened.device.device(),
            layout,
            &MINIMAL_COMPUTE_SPIRV,
            c"main",
        )
        .expect("a compute pipeline over a real descriptor set layout");
        assert_ne!(pipeline.handle(), vk::Pipeline::null());
        drop(pipeline);
    }

    #[test]
    fn a_payload_that_is_not_spirv_is_refused_before_the_driver_is_reached() {
        // The refusal happens inside `create_compute`, before a pipeline exists, and
        // the layout it consumed is released with it rather than leaked. A real
        // descriptor set layout is consumed too, which is what proves the release
        // walks the whole owner chain.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let set = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid descriptor set layout");
        let layout = create_layout(opened.device.device(), vec![set])
            .expect("a pipeline layout over one set layout");
        let refused = create_compute(opened.device.device(), layout, &[0x0000_0001], c"main");
        assert_eq!(
            refused.err(),
            Some(PipelineError::Shader(ShaderError::NotSpirV { found: 1 }))
        );
    }

    /// The depth sibling every retained raster recipe also creates.
    fn depth_state() -> PipelineState {
        PipelineState {
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
            }),
            ..colour_only_state()
        }
    }

    #[test]
    fn the_five_topologies_map_to_five_distinct_vulkan_topologies() {
        use PrimitiveTopology::{LineStrip, Lines, Points, TriangleStrip, Triangles};

        let portable = [Points, Lines, LineStrip, Triangles, TriangleStrip];
        let mapped: Vec<vk::PrimitiveTopology> = portable.iter().copied().map(topology).collect();
        assert_eq!(mapped[3], vk::PrimitiveTopology::TRIANGLE_LIST);
        for (index, value) in mapped.iter().enumerate() {
            for (other_index, other) in mapped.iter().enumerate() {
                if index != other_index {
                    assert_ne!(
                        value, other,
                        "{:?} and {:?} share a Vulkan topology",
                        portable[index], portable[other_index]
                    );
                }
            }
        }
    }

    #[test]
    fn the_three_cull_modes_map_to_three_distinct_vulkan_flag_sets() {
        // `None` is `NONE` and not `FRONT_AND_BACK`: the portable mode means
        // "discard nothing", and the two are not synonyms.
        let portable = [CullMode::None, CullMode::Front, CullMode::Back];
        let mapped: Vec<vk::CullModeFlags> = portable.iter().copied().map(cull_mode).collect();
        assert_eq!(mapped[0], vk::CullModeFlags::NONE);
        assert!(mapped[0].is_empty());
        assert_ne!(mapped[0], vk::CullModeFlags::FRONT_AND_BACK);
        assert!(!mapped[1].contains(mapped[2]));
        assert!(!mapped[2].contains(mapped[1]));
        for (index, value) in mapped.iter().enumerate() {
            for (other_index, other) in mapped.iter().enumerate() {
                if index != other_index {
                    assert_ne!(
                        value, other,
                        "{:?} and {:?} share a Vulkan cull mode",
                        portable[index], portable[other_index]
                    );
                }
            }
        }
    }

    #[test]
    fn both_winding_orders_map_to_their_own_front_face() {
        assert_eq!(front_face(FrontFace::Clockwise), vk::FrontFace::CLOCKWISE);
        assert_eq!(
            front_face(FrontFace::CounterClockwise),
            vk::FrontFace::COUNTER_CLOCKWISE
        );
        assert_ne!(
            front_face(FrontFace::Clockwise),
            front_face(FrontFace::CounterClockwise)
        );
    }

    #[test]
    fn each_vertex_format_maps_to_its_own_attribute_format() {
        let portable = [
            VertexFormat::Float32x2,
            VertexFormat::Float32x3,
            VertexFormat::Unorm8x4,
        ];
        let mapped: Vec<vk::Format> = portable.iter().copied().map(vertex_format).collect();
        assert_eq!(mapped[0], vk::Format::R32G32_SFLOAT);
        assert_eq!(mapped[1], vk::Format::R32G32B32_SFLOAT);
        // Normalized, so the stored integers reach the shader as `[0, 1]`.
        assert_eq!(mapped[2], vk::Format::R8G8B8A8_UNORM);
        for (index, value) in mapped.iter().enumerate() {
            for (other_index, other) in mapped.iter().enumerate() {
                if index != other_index {
                    assert_ne!(
                        value, other,
                        "{:?} and {:?} share a Vulkan format",
                        portable[index], portable[other_index]
                    );
                }
            }
        }
    }

    #[test]
    fn both_step_modes_map_to_their_own_input_rate() {
        assert_eq!(
            input_rate(VertexStepMode::Vertex),
            vk::VertexInputRate::VERTEX
        );
        assert_eq!(
            input_rate(VertexStepMode::Instance),
            vk::VertexInputRate::INSTANCE
        );
        assert_ne!(
            input_rate(VertexStepMode::Vertex),
            input_rate(VertexStepMode::Instance)
        );
    }

    #[test]
    fn the_write_mask_folds_into_the_named_components() {
        // The value a lowering that forgot the mask would produce is "write
        // nothing", which is `ash`'s default and a pipeline that renders nothing.
        assert_eq!(
            color_write_mask(ColorWriteMask::ALL),
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A
        );
        assert_eq!(
            color_write_mask(ColorWriteMask::R),
            vk::ColorComponentFlags::R
        );
        assert_eq!(
            color_write_mask(ColorWriteMask::A),
            vk::ColorComponentFlags::A
        );
        let rg = color_write_mask(ColorWriteMask::R.union(ColorWriteMask::G));
        assert!(rg.contains(vk::ColorComponentFlags::R));
        assert!(rg.contains(vk::ColorComponentFlags::G));
        assert!(!rg.contains(vk::ColorComponentFlags::B));
    }

    #[test]
    fn only_the_counts_vulkan_names_lower_and_the_flag_bit_is_not_the_count() {
        assert_eq!(sample_count(1), Some(vk::SampleCountFlags::TYPE_1));
        assert_eq!(sample_count(2), Some(vk::SampleCountFlags::TYPE_2));
        assert_eq!(sample_count(4), Some(vk::SampleCountFlags::TYPE_4));
        assert_eq!(sample_count(8), Some(vk::SampleCountFlags::TYPE_8));
        assert_eq!(sample_count(16), Some(vk::SampleCountFlags::TYPE_16));
        assert_eq!(sample_count(32), Some(vk::SampleCountFlags::TYPE_32));
        assert_eq!(sample_count(64), Some(vk::SampleCountFlags::TYPE_64));
        for count in [0, 3, 5, 6, 7, 128, u32::MAX] {
            assert_eq!(sample_count(count), None, "{count} is not a named count");
        }
        // The named counts happen to equal their bit, so the value returned is not
        // a renaming: it is the list of counts `Vulkan` defines, and the refusals
        // above are the ones a raw cast would have accepted.
        for count in [1u32, 2, 4, 8, 16, 32, 64] {
            assert_eq!(
                sample_count(count).map(vk::SampleCountFlags::as_raw),
                Some(count)
            );
        }
    }

    #[test]
    fn the_signature_names_the_formats_the_pass_and_the_pipeline_share() {
        let sign = signature(&colour_only_state()).expect("a description this backend maps");
        assert_eq!(sign.colors, vec![vk::Format::R8G8B8A8_UNORM]);
        assert_eq!(sign.depth_stencil, None);
        assert_eq!(sign.samples, vk::SampleCountFlags::TYPE_1);

        let sign = signature(&depth_state()).expect("the depth sibling this backend maps");
        assert_eq!(sign.colors, vec![vk::Format::R8G8B8A8_UNORM]);
        assert_eq!(sign.depth_stencil, Some(vk::Format::D32_SFLOAT));
        assert_eq!(sign.samples, vk::SampleCountFlags::TYPE_1);
    }

    #[test]
    fn a_colour_target_that_carries_depth_is_refused_before_the_driver_is_reached() {
        // The depth and colour vocabularies meet here, and this is the first layer
        // that can tell them apart: the portable enum is non-exhaustive, so a
        // classification written there would need a wildcard arm.
        let state = PipelineState {
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Depth32Float,
                write_mask: ColorWriteMask::ALL,
            }],
            ..colour_only_state()
        };
        assert_eq!(
            signature(&state),
            Err(PipelineError::ColorTargetFormatIsDepth {
                index: 0,
                format: TextureFormat::Depth32Float,
            })
        );
    }

    #[test]
    fn a_depth_state_that_carries_no_depth_is_refused_before_the_driver_is_reached() {
        let state = PipelineState {
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Rgba8Unorm,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
            }),
            ..colour_only_state()
        };
        assert_eq!(
            signature(&state),
            Err(PipelineError::DepthStencilFormatNotDepth(
                TextureFormat::Rgba8Unorm
            ))
        );
    }

    #[test]
    fn an_unnamed_sample_count_is_refused_before_the_driver_is_reached() {
        let state = PipelineState {
            sample_count: 3,
            ..colour_only_state()
        };
        assert_eq!(
            signature(&state),
            Err(PipelineError::UnsupportedSampleCount(3))
        );
    }

    #[test]
    fn a_real_colour_only_raster_pipeline_is_created_on_this_machine() {
        // Step 5's raster half against the real driver: both modules are read
        // during creation, the render pass is created and destroyed around it, and
        // dropping the pipeline destroys it and then its layout. Skips where no
        // adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let layout =
            create_layout(opened.device.device(), Vec::new()).expect("an empty pipeline layout");
        let pipeline = create_raster(
            opened.device.device(),
            layout,
            &raster_shaders(),
            &position_stream(),
            &colour_only_state(),
        )
        .expect("a raster pipeline whose modules, layout and render pass are valid");
        assert_ne!(pipeline.handle(), vk::Pipeline::null());
        // The drop destroys the pipeline and then its layout; the render pass was
        // released as soon as the pipeline existed.
        drop(pipeline);
    }

    #[test]
    fn a_real_depth_raster_pipeline_is_created_over_a_real_set_layout_on_this_machine() {
        // The depth-stencil branch reaches a real driver, over a pipeline layout
        // that owns a real descriptor set layout, so the whole owner chain is
        // exercised by the drop.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let set = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid descriptor set layout");
        let layout = create_layout(opened.device.device(), vec![set]).expect("a pipeline layout");
        let pipeline = create_raster(
            opened.device.device(),
            layout,
            &raster_shaders(),
            &position_stream(),
            &depth_state(),
        )
        .expect("a raster pipeline with a depth attachment");
        assert_ne!(pipeline.handle(), vk::Pipeline::null());
        drop(pipeline);
    }

    #[test]
    fn a_payload_that_is_not_spirv_is_refused_before_the_render_pass_exists() {
        // The refusal happens while the modules are being described, so no render
        // pass is created and the layout the call consumed is released with it.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let set = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid descriptor set layout");
        let layout = create_layout(opened.device.device(), vec![set]).expect("a pipeline layout");
        let refused = create_raster(
            opened.device.device(),
            layout,
            &RasterShaders {
                vertex: &[0x0000_0001],
                vertex_entry: c"main",
                fragment: &MINIMAL_RASTER_FRAGMENT_SPIRV,
                fragment_entry: c"main",
            },
            &position_stream(),
            &colour_only_state(),
        );
        assert_eq!(
            refused.err(),
            Some(PipelineError::Shader(ShaderError::NotSpirV { found: 1 }))
        );
    }
}
