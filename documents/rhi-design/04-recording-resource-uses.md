# RHI API freeze v13. Recording and actual resource uses

> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the root
> specification and this module in full before implementation. No other
> document may redefine the interfaces in this module.

# 29. CommandRecorder

This chapter has entered **FROZEN/P0**.

`CommandRecorder` is a CPU-side portable command builder:

```text
not a native command list / encoder
does not accept an external scheduling contract
does not expose barriers / transitions
does not bind a submission lane
```

A Recorder can contain Copy, Raster, Compute, Upload, and Readback in sequence; finally `RecordedWork` gets `LaneWorkDomains` based on the actual command.

## 29.1 Descriptor / creation

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RecorderDescriptor {
    pub label: Label,
}

impl RecorderDescriptor {
    pub fn new() -> Self;
    pub fn with_label(mut self, label: impl Into<String>) -> Self;
}

pub struct CommandRecorder {
    /* opaque */
}

impl Device {
    pub fn create_recorder(
        &self,
        desc: &RecorderDescriptor,
    ) -> RhiResult<CommandRecorder>;
}
```

Recorder is bound to `DeviceIdentity`; all input objects undergo identity validation before backend.

## 29.2 State machine

```text
Open
  ├─ begin_raster  -> RasterScopeOpen
  ├─ begin_compute -> ComputeScopeOpen
  ├─ copy/upload/readback/debug -> Open
  ├─ finish        -> RecordedWork
  └─ backend/internal failure -> Poisoned

RasterScopeOpen
  ├─ raster commands
  ├─ end -> Open
  └─ drop without end -> Poisoned

ComputeScopeOpen
  ├─ descriptor label + compute commands
  ├─ end -> Open
  └─ drop without end -> Poisoned

Poisoned
  └─ finish -> Err(...)
```

Rust mutable borrow prevents the scope from directly issuing another type of command to the Recorder while it is alive.

Scope still requires explicit `end()`; Drop does not perform a backend finalize that may fail. Metal's command encoder also has an explicit `endEncoding()` life cycle. (See References at the end of the article)

## 29.3 Error policy

Parameter/capability error:

```text
InvalidUsage / WrongDevice / DeviceLost / Unsupported
```

Only the current command is rejected, and the Recorder can still continue.

backend recording failure, scope-finalization failure, or internal-invariant failure:

```text
Recorder -> Poisoned
```

---

# 30. Geometry / dynamic command types

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorClearValue {
    Float([f32; 4]),
    Sint([i32; 4]),
    Uint([u32; 4]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    Load,
    Clear(T),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    Discard,
}
```

Portable validation:

```text
Rect:
    width/height may be 0
    x+width / y+height must not overflow

Viewport:
    all values finite
    width >= 0
    height >= 0
    0 <= min_depth <= max_depth <= 1
```

P0 does not use negative viewport height to express Y flip; coordinate system adaptation belongs to shader/toolchain/backend convention.

---

# 31. Raster attachments

Attachment set is fixed within a RasterScope.

## 31.1 Color attachment

```rust
#[non_exhaustive]
#[derive(Clone)]
pub enum ColorAttachmentView {
    Texture(TextureView),
    Frame(FrameAttachment),
}

impl ColorAttachmentView {
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn format(&self) -> TextureFormat;
    pub fn extent(&self) -> Extent3d;
    pub fn sample_count(&self) -> u32;
}

#[derive(Clone)]
pub struct ColorAttachment {
    pub view: ColorAttachmentView,
    pub load: LoadOp<ColorClearValue>,
    pub store: StoreOp,
    pub resolve: Option<ColorAttachmentView>,
}
```

TextureView as color attachment:

```text
TextureUsage::COLOR_ATTACHMENT
FormatFacts.color_attachment == true
```

For `LoadOp::Clear`, `ColorClearValue` must match the numeric class of the
color attachment format: `Float` for floating-point or normalized formats,
`Sint` for signed-integer formats, and `Uint` for unsigned-integer formats.

If `resolve != None`:

```text
source.sample_count > 1
resolve.sample_count == 1
source.format == resolve.format
source.width/height == resolve.width/height
source / resolve must not overlap
```

Texture resolve target also requires `COLOR_ATTACHMENT` usage.

Raster attachment resolve does not require COPY_SRC/COPY_DST; it is a different route than the standalone `resolve_texture()` route.

When FrameAttachment is used as the main color target, `store` must be `Store`; P0 does not allow the content of the frame to be presented to be marked as Discard.

## 31.2 Depth / stencil mode

The old model `read_only + load/store` will produce a conflicting combination of `read_only + Clear`, which is changed to:

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum DepthAttachmentMode {
    ReadOnly,

    ReadWrite {
        load: LoadOp<f32>,
        store: StoreOp,
    },
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum StencilAttachmentMode {
    ReadOnly,

    ReadWrite {
        load: LoadOp<u32>,
        store: StoreOp,
    },
}

#[derive(Clone)]
pub struct DepthStencilAttachment {
    pub view: TextureView,
    pub depth: Option<DepthAttachmentMode>,
    pub stencil: Option<StencilAttachmentMode>,
}
```

Must verify:

```text
TextureUsage::DEPTH_STENCIL_ATTACHMENT

depth != None   -> view aspects include DEPTH
stencil != None -> view aspects include STENCIL

depth == None && stencil == None -> InvalidUsage
```

ReadOnly does not generate attachment write use.

For `DepthAttachmentMode::ReadWrite { load: LoadOp::Clear(value), .. }`,
`value` must be finite and satisfy `0.0 <= value <= 1.0`. This validation is
performed before the command enters the semantic stream.

## 31.3 RasterScopeDescriptor

```rust
#[non_exhaustive]
#[derive(Clone)]
pub struct RasterScopeDescriptor {
    pub label: Label,

    /// index = color attachment location.
    /// None = this location has no attachment.
    pub colors: Vec<Option<ColorAttachment>>,

    pub depth_stencil: Option<DepthStencilAttachment>,
}

impl RasterScopeDescriptor {
    pub fn new() -> Self;
    pub fn with_label(mut self, label: impl Into<String>) -> Self;

    pub fn with_color(
        mut self,
        location: ShaderLocation,
        attachment: ColorAttachment,
    ) -> Self;

    pub fn with_depth_stencil(
        mut self,
        attachment: DepthStencilAttachment,
    ) -> Self;
}
```

Canonicalize/remove trailing `None`.

## 31.4 Scope attachment invariants

`begin_raster()` verifies:

```text
all active attachments have the same DeviceIdentity
same width/height
same sample_count (except resolve targets)
```

Single-view pipelines require `layer_count == 1`. A multiview pipeline's
non-zero mask selects attachment layers: its highest selected bit must be less
than every main attachment's common `layer_count`; all color and depth/stencil
main attachments must have identical layer counts. `begin_raster()` validates
the common geometry, while `set_pipeline()` validates the pipeline mask after
the pipeline is known. Resolve views remain separate from the main attachment
geometry.

At least one color or depth/stencil attachment exists; an empty attachment RasterScope does not enter P0.

---



# 32. RasterScope

~~~rust
pub struct RasterScope<'a> {
    _marker: std::marker::PhantomData<&'a mut CommandRecorder>,
}

impl CommandRecorder {
    pub fn begin_raster<'a>(
        &'a mut self,
        desc: &RasterScopeDescriptor,
    ) -> RhiResult<RasterScope<'a>>;
}
~~~

All portable logical state resets when a scope begins:

~~~text
pipeline       = unbound
bind groups    = unbound
vertex buffers = unbound
index buffer   = unbound
viewport       = default-unset
scissor        = default-unset
blend constant = [0,0,0,0]
stencil ref    = 0
~~~

No state is inherited from the prior scope.

## 32.1 Commands

~~~rust
impl<'a> RasterScope<'a> {
    pub fn set_pipeline(&mut self, pipeline: &RasterPipeline) -> RhiResult<()>;

    pub fn set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()>;

    pub fn set_vertex_buffer(
        &mut self,
        slot: u32,
        binding: &BufferBinding,
    ) -> RhiResult<()>;

    pub fn set_index_buffer(
        &mut self,
        binding: &BufferBinding,
        format: IndexFormat,
    ) -> RhiResult<()>;

    pub fn set_viewport(&mut self, viewport: Viewport) -> RhiResult<()>;
    pub fn set_scissor(&mut self, rect: Rect) -> RhiResult<()>;
    pub fn set_blend_constant(&mut self, color: Color) -> RhiResult<()>;
    pub fn set_stencil_reference(&mut self, value: u32) -> RhiResult<()>;

    pub fn draw(
        &mut self,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> RhiResult<()>;

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> RhiResult<()>;

    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()>;
    pub fn pop_debug_group(&mut self) -> RhiResult<()>;
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()>;

    pub fn end(self) -> RhiResult<()>;
}
~~~

## 32.2 Pipeline / attachment compatibility

set_pipeline() must validate:

~~~text
DeviceIdentity
pipeline.target_signature == current RasterScope primary attachment signature
~~~

The signature includes sparse color formats, depth/stencil format, and sample count.

A resolve target is not part of the pipeline signature; the pipeline sees the multisample source attachment.

## 32.3 Draw validation

Before draw/draw_indexed enters the semantic stream, validate at least:

~~~text
RasterPipeline bound

every group actually used by shader bound
BindGroup layout exactly compatible
dynamic offsets valid

every buffer slot used by VertexInputState bound
BufferUsage::VERTEX
attribute/stride access in bounds
~~~

draw_indexed additionally requires:

~~~text
IndexBuffer bound
BufferUsage::INDEX
index byte range in bounds

strip topology:
    pipeline.strip_index_format != None
    and == current IndexFormat
~~~

Default dynamic state:

~~~text
viewport unset -> full attachment extent
scissor unset  -> full attachment extent
blend constant -> [0,0,0,0]
stencil ref    -> 0
~~~

These are Fluxel portable defaults, not backend-selected behavior.

## 32.4 Actual resource use

Resources are actually consumed by shader/vertex fetch at:

~~~text
draw / draw_indexed
~~~

not at set_bind_group().

Therefore:

~~~text
BindGroup has 10 resources
Shader actually references only 3
~~~

generates shader use only for those 3.

Raster actual use includes at least:

~~~text
active attachments
shader actually referenced binding resources
vertex buffers actually required by VertexInputState
index buffer (indexed draw)
~~~

Attachment-scope semantics:

~~~text
Load    -> scope begin read
Clear   -> scope begin write
Store   -> result remains defined
Discard -> contents become undefined after scope end
~~~

Thus even with no draw:

~~~text
Clear + Store
~~~

is a valid write.

---

# 33. ComputeScope

~~~rust
pub struct ComputeScope<'a> {
    _marker: std::marker::PhantomData<&'a mut CommandRecorder>,
}

impl CommandRecorder {
    pub fn begin_compute<'a>(&'a mut self) -> RhiResult<ComputeScope<'a>>;
}

impl<'a> ComputeScope<'a> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) -> RhiResult<()>;

    pub fn set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()>;

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> RhiResult<()>;

    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()>;
    pub fn pop_debug_group(&mut self) -> RhiResult<()>;
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()>;

    pub fn end(self) -> RhiResult<()>;
}
~~~

When OptionalFeature::Compute is not enabled:

~~~text
begin_compute -> Unsupported
~~~

Before dispatch(), validate:

~~~text
ComputePipeline bound
BindGroups actually used by shader bound and compatible
x/y/z <= MaxComputeWorkgroupsPerDimension
~~~

x/y/z == 0 is legal.

Actual shader use is generated only at dispatch():

~~~text
UniformBuffer              -> UNIFORM_READ
StorageBuffer ReadOnly     -> SHADER_READ
StorageBuffer ReadWrite    -> SHADER_READ | SHADER_WRITE
SampledTexture             -> SHADER_READ
StorageTexture ReadOnly    -> SHADER_READ
StorageTexture WriteOnly   -> SHADER_WRITE
StorageTexture ReadWrite   -> SHADER_READ | SHADER_WRITE
Sampler                    -> no memory hazard
~~~

---

# 34. Copy / Resolve / Blit

Copy-family commands may be recorded only in Recorder::Open.

An unsupported route returns Unsupported; a backend must not secretly insert shader/CPU fallback.

## 34.1 BufferCopy

~~~rust
#[derive(Clone)]
pub struct BufferCopy {
    pub src: Buffer,
    pub src_offset: u64,
    pub dst: Buffer,
    pub dst_offset: u64,
    pub size: u64,
}
~~~

Validation:

~~~text
size > 0
src COPY_SRC
dst COPY_DST
range bounds
DeviceIdentity
RouteQuery::BufferToBuffer Supported
BufferCopyLayoutLimits offset/size alignment satisfied
~~~

Source/destination byte ranges in the same Buffer must not overlap; P0 defines no memmove.

## 34.2 BufferTextureCopy

~~~rust
#[derive(Clone)]
pub struct BufferTextureCopy {
    pub buffer: Buffer,
    pub buffer_offset: u64,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,

    pub texture: Texture,
    pub texture_subresource: TextureSubresourceLayers,
    pub texture_origin: Origin3d,
    pub extent: Extent3d,
}
~~~

Buffer→Texture:

~~~text
buffer COPY_SRC
texture COPY_DST
texture.sample_count == 1
BufferToTexture route key includes dimension/format/aspect
route Supported
~~~

Texture→Buffer reverses the direction and also requires texture.sample_count == 1; its route key likewise includes dimension/format/aspect.

Both require:

~~~text
buffer_offset / bytes_per_row satisfy TexelCopyLayoutLimits
array image stride satisfies TexelCopyLayoutLimits when more than one layer is copied
3D rows_per_image satisfies the route's packed-slice rule when it has one
bytes_per_row > 0
rows_per_image > 0
layout footprint sufficiently covers region
~~~

This is GPU copy-buffer layout, not Upload HostTexelLayout. Row-pitch/placement alignment in routes such as D3D12 is a typical example (see References).

## 34.3 TextureCopy

~~~rust
#[derive(Clone)]
pub struct TextureCopy {
    pub src: Texture,
    pub src_subresource: TextureSubresourceLayers,
    pub src_origin: Origin3d,

    pub dst: Texture,
    pub dst_subresource: TextureSubresourceLayers,
    pub dst_origin: Origin3d,

    pub extent: Extent3d,
}
~~~

P0 direct copy:

~~~text
src/dst formats identical
src/dst aspects identical
src/dst layer_count identical
src/dst sample_count explicitly matched by RouteQuery
src COPY_SRC
dst COPY_DST
TextureToTexture route Supported
~~~

Corresponding texel regions in the same Texture must not overlap.

## 34.4 Direct resolve

~~~rust
#[derive(Clone)]
pub struct TextureResolve {
    pub src: Texture,
    pub src_subresource: TextureSubresourceLayers,
    pub src_origin: Origin3d,
    pub dst: Texture,
    pub dst_subresource: TextureSubresourceLayers,
    pub dst_origin: Origin3d,
    pub extent: Extent3d,
}
~~~

P0 direct resolve freezes Color only:

~~~text
src/dst aspect == Color
src format == dst format
src sample_count > 1
dst sample_count == 1
src COPY_SRC
dst COPY_DST
layer_count identical
src_origin + extent in source subresource bounds
dst_origin + extent in destination subresource bounds
Resolve route Supported
~~~

Vulkan direct resolve itself requires a multisampled source, a single-sample destination, and matching format (see References).

If a WebGPU backend lacks encoder-level direct resolve:

~~~text
RouteSupport::Unsupported
~~~

is legal; RasterScope attachment resolve may still be supported.

## 34.5 Blit

~~~rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlitFilter {
    Nearest,
    Linear,
}

#[derive(Clone)]
pub struct TextureBlit {
    pub src: Texture,
    pub src_subresource: TextureSubresourceLayers,
    pub src_origin: Origin3d,
    pub src_extent: Extent3d,

    pub dst: Texture,
    pub dst_subresource: TextureSubresourceLayers,
    pub dst_origin: Origin3d,
    pub dst_extent: Extent3d,

    pub filter: BlitFilter,
}
~~~

P0 direct blit freezes Color only:

~~~text
src/dst aspect == Color
src/dst sample_count == 1
src COPY_SRC
dst COPY_DST
layer_count identical
Blit route key includes src/dst dimension + format + filter
Blit route Supported
~~~

Linear is legal only where the route explicitly supports it.

Source/destination regions in the same Texture must not overlap.

## 34.6 Commands

~~~rust
impl CommandRecorder {
    pub fn copy_buffer(&mut self, copy: &BufferCopy) -> RhiResult<()>;

    pub fn copy_buffer_to_texture(
        &mut self,
        copy: &BufferTextureCopy,
    ) -> RhiResult<()>;

    pub fn copy_texture_to_buffer(
        &mut self,
        copy: &BufferTextureCopy,
    ) -> RhiResult<()>;

    pub fn copy_texture(&mut self, copy: &TextureCopy) -> RhiResult<()>;
    pub fn resolve_texture(&mut self, resolve: &TextureResolve) -> RhiResult<()>;
    pub fn blit_texture(&mut self, blit: &TextureBlit) -> RhiResult<()>;
}
~~~

A Copy-family command is itself an actual resource-use point.

---

# 35. Upload / Readback encoding

```rust
impl CommandRecorder {
    pub fn encode_upload(
        &mut self,
        upload: &UploadJob,
    ) -> RhiResult<()>;

    pub fn encode_readback(
        &mut self,
        request: ReadbackRequest,
    ) -> RhiResult<ReadbackTicket>;
}
```

Can only be called from `Recorder::Open`.

Actual use:

```text
Upload   -> destination COPY_WRITE
Readback -> source COPY_READ
```

If Recorder/RecordedWork/SubmissionPlan is abandoned before successful submit and there is no other submission owner:

```text
ReadbackTicket::NotSubmitted -> Abandoned
```

---

# 36. Debug labels / markers

```rust
impl CommandRecorder {
    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()>;
    pub fn pop_debug_group(&mut self) -> RhiResult<()>;
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()>;
}
```

rule:

```text
Recorder::Open, RasterScope, and ComputeScope each own an independent debug-group stack
push/pop on a given object operate only on that object's stack
pop without a matching push on that stack -> InvalidUsage
Recorder::finish requires the Recorder::Open stack to be empty
RasterScope::end requires the RasterScope stack to be empty
ComputeScope::end requires the ComputeScope stack to be empty
```

P0 debug-group stacks do not span Recorder::Open, RasterScope, or
ComputeScope. A scope label establishes diagnostics nesting separately.

---

# 37. Actual ResourceUse

`ResourceUse` is **the RHI's own execution vocabulary**.

Its public path is:

```text
rhi::command::ResourceUse
```

Callers normally do not construct it by hand; the Recorder generates it from
the actual portable commands.

```rust
pub mod command {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PipelineScope(u32);

    impl PipelineScope {
        pub const VERTEX: Self = Self(1 << 0);
        pub const FRAGMENT: Self = Self(1 << 1);
        pub const COMPUTE: Self = Self(1 << 2);
        pub const COPY: Self = Self(1 << 3);

        pub fn contains(self, other: Self) -> bool;
        pub fn union(self, other: Self) -> Self;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccessMask(u32);

    impl AccessMask {
        pub const VERTEX_READ: Self = Self(1 << 0);
        pub const INDEX_READ: Self = Self(1 << 1);
        pub const UNIFORM_READ: Self = Self(1 << 2);
        pub const SHADER_READ: Self = Self(1 << 3);
        pub const SHADER_WRITE: Self = Self(1 << 4);
        pub const COLOR_READ: Self = Self(1 << 5);
        pub const COLOR_WRITE: Self = Self(1 << 6);
        pub const DEPTH_READ: Self = Self(1 << 7);
        pub const DEPTH_WRITE: Self = Self(1 << 8);
        pub const STENCIL_READ: Self = Self(1 << 9);
        pub const STENCIL_WRITE: Self = Self(1 << 10);
        pub const COPY_READ: Self = Self(1 << 11);
        pub const COPY_WRITE: Self = Self(1 << 12);
        pub fn contains(self, other: Self) -> bool;
        pub fn union(self, other: Self) -> Self;
    }

    #[non_exhaustive]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum TextureUseIntent {
        ShaderRead,
        ShaderReadWrite,
        ColorAttachment,
        DepthStencilRead,
        DepthStencilWrite,
        CopySrc,
        CopyDst,
        ResolveSrc,
        ResolveDst,
    }

    #[derive(Clone)]
    pub struct BufferUse {
        pub buffer: Buffer,
        pub range: BufferRange,
        pub stages: PipelineScope,
        pub access: AccessMask,
    }

    #[derive(Clone)]
    pub struct TextureUse {
        pub texture: Texture,
        pub subresources: TextureSubresourceRange,
        pub stages: PipelineScope,
        pub access: AccessMask,
        pub intent: TextureUseIntent,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct FrameAttachmentUse {
        pub frame: AcquiredFrameId,
        pub stages: PipelineScope,
        pub access: AccessMask,
    }

    #[non_exhaustive]
    #[derive(Clone)]
    pub enum ResourceUse {
        Buffer(BufferUse),
        Texture(TextureUse),
        Frame(FrameAttachmentUse),
    }
}
```

## 37.1 Command-level use sequence

RHI internal recording must be retained:

```text
Command #N -> uses [...]
```

Because within a RecordedWork it is also possible:

```text
copy A -> B
dispatch reads B writes C
draw reads C
```

Need to keep happens-before/hazard lowering.

The merged summary cannot replace command-level ordering.

## 37.2 BindGroup use

shader actual use source:

```text
current pipeline ShaderInterface
+
current BindGroup
+
dynamic offsets
```

Not all resources in BindGroup.

Sampler does not generate memory hazard, but still enters command semantics/statistics.

## 37.3 Attachment use / definedness

```text
Load    -> begin read
Clear   -> begin write
draw    -> raster attachment access
Store   -> result preserved
Discard -> result becomes undefined
```

`ResourceUse` only describes access/hazard; it does not pretend to infer
data-dependent full-write coverage by a shader.

The RHI can validate explicit command semantics:

```text
Load/Clear
Store/Discard
copy destination
resolve destination
```

It cannot infer from `SHADER_WRITE` that an entire resource range was covered.

---

# 38. RecordedWork

## 38.1 RecordedWork

```rust
pub struct RecordedWork {
    /* opaque, single-device */
}

impl CommandRecorder {
    pub fn finish(self) -> RhiResult<RecordedWork>;
}

impl RecordedWork {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;

    /// Execution domains actually contained.
    pub fn work_domains(&self) -> LaneWorkDomains;

    /// Merged actual-use summary.
    pub fn resource_uses(&self) -> &[command::ResourceUse];
}
```

domains:

```text
Raster             -> RASTER
Compute            -> COMPUTE
Copy/Upload/Readback -> COPY
```

`RecordedWork` strongly owns the logical objects required for execution:

```text
Buffer / Texture / View / Sampler
Shader / BindGroup / Pipeline
Upload payload
Readback state
```

Therefore, if the user drops the original resource handle after recording, it will not affect the correct submission of the RecordedWork.

## 38.2 Recording freeze decision

0.16 does not exist:

```text
public Barrier / Transition
public Fence / Event
native command-list escape
user-selected native queue
implicit shader fallback
implicit CPU-copy fallback
scope auto-end-on-Drop
```

Stable semantics:

```text
explicit scopes
+ portable logical state
+ validated commands
+ command-level actual-use sequence
+ RecordedWork
```
