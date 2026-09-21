# RHI API freeze v13. Resources, upload, and readback

> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the root
> specification and this module in full before implementation. No other
> document may redefine the interfaces in this module.

# 8. Format / Texture facts

This chapter is **FROZEN / P0**.

Two questions are explicitly separated here:

```text
FormatFacts
    = What can this format itself do under the current capability contract?

TextureSupportQuery
    = Can a Texture with this dimension + format + usage + sample_count be created?

BindingSupportQuery
    = Can this stage visibility + kind + count + dynamic-offset semantic be expressed?
```

These two concepts must not be merged again.

---

## 8.1 TextureFormat

v13 freezes the core formats plus a per-format block-compression vocabulary:

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,

    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,

    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,

    Bgra8Unorm,
    Bgra8UnormSrgb,

    R16Uint,
    R16Sint,
    R16Float,

    Rg16Uint,
    Rg16Sint,
    Rg16Float,

    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,

    R32Uint,
    R32Sint,
    R32Float,

    Rg32Uint,
    Rg32Sint,
    Rg32Float,

    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,

    Depth16Unorm,

    /// Portable depth semantic.
    ///
    /// The backend may choose the actual backing precision/layout implementation
    /// that satisfies the contract. Therefore this does not promise fixed
    /// bytes-per-block and cannot be used for bit-exact VRAM estimates.
    Depth24Plus,

    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
}
```

The enum additionally includes these concrete compressed families (with the
linear/sRGB variants named by the API):

- BC: BC1/2/3 RGBA, BC4 R, BC5 RG, BC6H RGB float/ufloat, and BC7 RGBA.
- ETC2/EAC: RGB8, RGB8A1, RGBA8, R11/RG11, including signed EAC and sRGB
  variants where the encoded color representation permits one.
- ASTC: 4x4, 5x4, 5x5, 6x5, 6x6, 8x5, 8x6, 8x8, 10x5, 10x6, 10x8, 10x10,
  12x10 and 12x12, each in linear and sRGB form.

Support is never represented by a family-wide boolean. `FormatFacts(format)`
and `TextureSupportQuery { format, ... }` answer every concrete enum member;
thus an ETC2-capable device does not accidentally promise BC or an unsupported
ASTC block size. `block_width`, `block_height`, and `logical_bytes_per_block`
are the one source of truth for upload/readback/copy layouts. Copy origins are
block aligned and extents may end part-way through a block only at a mip edge.

Planar/video formats (`NV12`, `P010`) and plane aspects are part of the same
per-format vocabulary, with their plane geometry and copy/view restrictions
defined by `TextureSupportQuery`. Do not create a `CompressedTextureApi` trait
or a family-wide compressed capability.

---

## 8.2 FormatFacts

`filterable` cannot by itself replace “sampleable.”

For example, integer / unfilterable-float formats:

```text
can be shader read
but cannot be filtered
```

Therefore, the sample type is frozen as follows:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageAccessSupport {
    read_only: bool,
    write_only: bool,
    read_write: bool,
}

impl StorageAccessSupport {
    pub fn supports(
        &self,
        access: StorageAccess,
    ) -> bool;
}

/// Format facts under the current Device/Adapter contract.
///
/// Fields are opaque, so adding a format fact in the future does not cause a
/// public struct-literal breaking change.
#[derive(Clone, Copy, Debug)]
pub struct FormatFacts {
    /* opaque */
}

impl FormatFacts {
    /// Set of Color / Depth / Stencil aspects.
    pub fn aspects(&self) -> TextureAspects;

    /// Portable sample type for a shader sampled binding.
    ///
    /// None means it cannot be used as a sampled Texture.
    ///
    /// Float       = a filtering sampler is legal;
    /// UnfilterableFloat = non-filtering only;
    /// Sint/Uint/Depth use their corresponding binding semantics.
    pub fn sample_type(&self) -> Option<TextureSampleType>;

    pub fn storage_access(&self) -> StorageAccessSupport;

    pub fn color_attachment(&self) -> bool;
    pub fn depth_attachment(&self) -> bool;
    pub fn stencil_attachment(&self) -> bool;

    /// Can be true only for color-attachment formats.
    pub fn blendable(&self) -> bool;

    /// Whether the color format has an alpha component.
    pub fn has_alpha_channel(&self) -> bool;

    /// Numeric class required when a fragment shader writes this color attachment.
    ///
    /// normalized / float format -> Float32
    /// signed integer format     -> Sint32
    /// unsigned integer format   -> Uint32
    ///
    /// Returns None for depth/stencil formats.
    pub fn color_output_type(&self) -> Option<ShaderNumericType>;

    /// How many texels one addressable texel/block covers.
    pub fn block_width(&self) -> u32;
    pub fn block_height(&self) -> u32;

    /// Bytes/block that may be used for a descriptor-based logical memory estimate.
    ///
    /// Implementation-defined backing such as `Depth24Plus` may return None.
    pub fn logical_bytes_per_block(&self) -> Option<u32>;
}
```

### Content excluded from FormatFacts

The following are not simple format facts:

```text
whether a particular 3D texture can be created
whether a particular usage combination can be created
MSAA sample count
copy row-pitch alignment
surface presentability
whether a viewFormats list is legal
```

They belong respectively to:

```text
TextureSupport
RouteSupport
SurfaceFact
TextureView validation
```

---

## 8.3 TextureSupportQuery

Vulkan image-format support itself depends on:

```text
format
image type
usage
sample count / creation constraints
```

Other backends have similar descriptor-dependent restrictions.

Therefore, a capability query must contain at least:

```rust
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureSupportQuery {
    dimension: TextureDimension,
    format: TextureFormat,
    usage: TextureUsage,
    sample_count: u32,

    /// Alternate view formats declared as permitted when the Texture is created.
    view_formats: Vec<TextureFormat>,

    /// View-compatibility intent that must be fixed when the Texture is created.
    ///
    /// For example, a Vulkan cube view requires the image to have cube-compatible
    /// semantics at creation time.
    view_compatibility: TextureViewCompatibility,
}

impl TextureSupportQuery {
    pub fn new(
        dimension: TextureDimension,
        format: TextureFormat,
        usage: TextureUsage,
        sample_count: u32,
    ) -> Self;

    pub fn with_view_format(
        self,
        format: TextureFormat,
    ) -> Self;

    pub fn with_view_compatibility(
        self,
        compatibility: TextureViewCompatibility,
    ) -> Self;

    pub fn dimension(&self) -> TextureDimension;
    pub fn format(&self) -> TextureFormat;
    pub fn usage(&self) -> TextureUsage;
    pub fn sample_count(&self) -> u32;
    pub fn view_formats(&self) -> &[TextureFormat];
    pub fn view_compatibility(&self) -> TextureViewCompatibility;
}
```

Do not put extent/mip/layer into the query key.

Those limits are returned by the `Supported` result.

---

## 8.4 TextureSupport

```rust
#[derive(Clone, Copy, Debug)]
pub struct TextureSupportLimits {
    max_extent: Extent3d,
    max_mip_levels: u32,
    max_array_layers: u32,
}

impl TextureSupportLimits {
    pub fn max_extent(&self) -> Extent3d;
    pub fn max_mip_levels(&self) -> u32;
    pub fn max_array_layers(&self) -> u32;
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum TextureSupport {
    Unsupported,

    Supported(TextureSupportLimits),
}

impl TextureSupport {
    pub fn is_supported(&self) -> bool;
    pub fn limits(&self) -> Option<&TextureSupportLimits>;
}
```

Final validation for Texture creation:

```text
TextureSupportQuery
    +
TextureSupportLimits
    +
TextureDescriptor extent/mips/layers
    =
whether it is legal
```

This correctly expresses:

```text
RGBA16Float 2D color attachment is supported
but a 3D color attachment is not supported

a format with sample_count=1 is supported
but sample_count=8 is not supported
```

without adding a pile of fixed booleans.

---

## 8.5 Texture view format compatibility

Support for a Texture base format does not mean that arbitrary reinterpretation is legal.

```rust
impl EnabledCapabilities {
    /// Whether `view_format` can be used as the view format of a `base_format` Texture.
    ///
    /// This answers format compatibility only;
    /// whether TextureDescriptor declares this view format, and whether the
    /// aspect/dimension/range are legal, remain the responsibility of
    /// create_texture_view validation.
    pub fn texture_view_format_compatible(
        &self,
        base_format: TextureFormat,
        view_format: TextureFormat,
    ) -> bool;
}
```

Do not assume:

```text
same byte size
    =>
view-compatible
```

---

# 9. Route facts

This chapter has entered **FROZEN/P0**.

Route Fact Answer:

> **Does a certain portable operation have a legal direct RHI route? **

no:

> Can the backend be secretly replaced by a shader implementation?

---

## 9.1 RouteQuery

Route key must contain texture shape/sample facts that will change native legality;
Otherwise, `Blit/Copy` will have a "query Supported, but the actual descriptor cannot be executed" vulnerability.

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RouteQuery {
    BufferToBuffer,

    BufferToTexture {
        dimension: TextureDimension,
        format: TextureFormat,
        aspect: TextureAspect,
    },

    TextureToBuffer {
        dimension: TextureDimension,
        format: TextureFormat,
        aspect: TextureAspect,
    },

    TextureToTexture {
        src_dimension: TextureDimension,
        src_format: TextureFormat,
        src_aspect: TextureAspect,
        src_sample_count: u32,

        dst_dimension: TextureDimension,
        dst_format: TextureFormat,
        dst_aspect: TextureAspect,
        dst_sample_count: u32,
    },

    Resolve {
        format: TextureFormat,
        src_sample_count: u32,
    },

    Blit {
        src_dimension: TextureDimension,
        src_format: TextureFormat,

        dst_dimension: TextureDimension,
        dst_format: TextureFormat,

        filter: BlitFilter,
    },
}
```

Buffer↔Texture P0 only allows single-sample texture;
TextureSupport + command validation are jointly guaranteed.

---

## 9.2 Copy layout limits

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferCopyLayoutLimits {
    offset_alignment: u64,
    size_alignment: u64,
}

impl BufferCopyLayoutLimits {
    pub fn offset_alignment(&self) -> u64;
    pub fn size_alignment(&self) -> u64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelCopyLayoutLimits {
    /* opaque */
}

impl TexelCopyLayoutLimits {
    pub fn buffer_offset_alignment(&self) -> u64;
    pub fn bytes_per_row_alignment(&self) -> u32;
    pub fn image_stride_alignment(&self) -> u64;
    pub fn tightly_packed_3d_slices(&self) -> bool;
}
```

`image_stride_alignment` applies only between independently addressed array
images. It does not impose a fictitious placement alignment on 3D Z slices:
those belong to one native footprint. `tightly_packed_3d_slices` instead says
whether `rows_per_image` must equal the copied physical block-row count because
the backend has no independent 3D slice pitch. All rules remain route facts,
verified after the region shape is known and before recording.

---

## 9.3 RouteSupport

```rust
#[derive(Clone, Copy, Debug)]
pub struct RouteCapabilities {
    buffer_copy_layout: Option<BufferCopyLayoutLimits>,
    texel_copy_layout: Option<TexelCopyLayoutLimits>,
}

impl RouteCapabilities {
    pub fn buffer_copy_layout(
        &self,
    ) -> Option<BufferCopyLayoutLimits>;

    pub fn texel_copy_layout(
        &self,
    ) -> Option<TexelCopyLayoutLimits>;
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum RouteSupport {
    Unsupported,
    Supported(RouteCapabilities),
}

impl RouteSupport {
    pub fn is_supported(&self) -> bool;
    pub fn capabilities(&self) -> Option<&RouteCapabilities>;
}
```

typical:

```text
BufferToBuffer
    -> buffer_copy_layout = Some(...)

BufferToTexture
    -> texel_copy_layout = Some(...)

Blit
    -> Unsupported / Supported(None layouts)
```

---

## 9.4 Fallback rule

Freeze rules:

```text
RouteSupport::Unsupported
    =>
RHI command returns Unsupported
```

RHI/backend prohibits: without the user's knowledge:

```text
Blit -> insert fullscreen shader
Copy -> staging CPU round-trip
Resolve -> compute shader
```

If Renderer / RenderGraph accepts fallback:

```text
query RouteFact
    -> select another explicit Graph pass / command route
```

In this way, Capture / statistics / performance model will not be secretly modified by backend.

---

# 10. Submission capability

This chapter has entered **FROZEN/P0**.

`SubmissionLane` definition:

> **A logically ordered submission domain. **

It is not equivalent to native queue family / command queue / GPU engine.

---

## 10.1 Lane identity / class / executable domains

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionLaneId(u16);

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmissionLaneClass {
    General,
    Graphics,
    Compute,
    Transfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaneWorkDomains(u8);

impl LaneWorkDomains {
    pub const RASTER: Self = Self(1 << 0);
    pub const COMPUTE: Self = Self(1 << 1);
    pub const COPY: Self = Self(1 << 2);

    pub fn contains(self, other: Self) -> bool;
    pub fn union(self, other: Self) -> Self;
}

#[derive(Clone, Debug)]
pub struct SubmissionLaneInfo {
    id: SubmissionLaneId,
    class: SubmissionLaneClass,
    domains: LaneWorkDomains,
}

impl SubmissionLaneInfo {
    pub fn id(&self) -> SubmissionLaneId;
    pub fn class(&self) -> SubmissionLaneClass;

    /// Correctness fact: which RecordedWork domains may be submitted to this lane.
    pub fn domains(&self) -> LaneWorkDomains;
}
```

`class` is the scheduling/diagnostic classification.

What really determines command legality is:

```rust
domains()
```

For example backend can be exposed:

```text
Lane 0:
    class = Graphics
    domains = Raster | Compute | Copy

Lane 1:
    class = Transfer
    domains = Copy
```

Instead of guessing the legal command from its name.

---

## 10.2 Lane dependency route

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneDependencyRoute {
    /// Producer/consumer are already on the same ordered lane;
    /// lane order itself satisfies happens-before.
    Ordered,

    /// Different lanes; a GPU-side dependency can be established.
    Gpu,

    /// The two logical lanes must be collapsed to the same ordered execution domain during lowering.
    Collapse,

    Unsupported,
}

#[derive(Clone, Debug)]
pub struct SubmissionCapabilities {
    lanes: Vec<SubmissionLaneInfo>,
}

impl SubmissionCapabilities {
    pub fn lanes(&self) -> &[SubmissionLaneInfo];

    pub fn lane(
        &self,
        id: SubmissionLaneId,
    ) -> Option<&SubmissionLaneInfo>;

    pub fn dependency_route(
        &self,
        from: SubmissionLaneId,
        to: SubmissionLaneId,
    ) -> LaneDependencyRoute;
}
```

### Base guarantee

Each Device has at least one lane:

```text
domains include Raster + Copy
```

If Device enables `OptionalFeature::Compute`:

```text
at least one lane has domains including Compute
```

This lane can be the Base lane.

---

## 10.3 Explicit capability that does not exist

```text
supports_real_overlap
supports_gpu_lane_dependencies: bool
queue_family_index
native_queue_count
```

reason:

```text
multiple logical lanes
    !=
real hardware parallelism

multiple native queues
    !=
guaranteed overlap

one native queue
    !=
cannot express multiple logical scheduling lanes
```

The true overlap belongs to the profiler/statistics/benchmark observation, not to the correctness fact.

---

# 11. Resource common types

This group has entered **FROZEN/P0**.

The resource layer only freezes the semantics that the user can actually observe:

```text
size / extent / mip / layer
format
usage
view compatibility
device identity
logical lifetime
upload/readback behavior
```

Not freezing:

```text
native heap type
memory type index
storage mode
tiling mode
allocation offset
resource state
native map/unmap primitive
```

---

## 11.1 Usage flags

No dependencies on the third-party bitflags crate; only std-only API shapes are shown here.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferUsage(u32);

impl BufferUsage {
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const VERTEX: Self = Self(1 << 2);
    pub const INDEX: Self = Self(1 << 3);
    pub const UNIFORM: Self = Self(1 << 4);
    pub const STORAGE: Self = Self(1 << 5);
    pub const MAP_READ: Self = Self(1 << 6);
    pub const MAP_WRITE: Self = Self(1 << 7);
    pub const INDIRECT: Self = Self(1 << 8);
    pub const QUERY_RESOLVE: Self = Self(1 << 9);
    pub const BLAS_INPUT: Self = Self(1 << 10);
    pub const TLAS_INPUT: Self = Self(1 << 11);

    pub fn contains(self, other: Self) -> bool;
    pub fn union(self, other: Self) -> Self;
    pub fn is_empty(self) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureUsage(u32);

impl TextureUsage {
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const SAMPLED: Self = Self(1 << 2);
    pub const STORAGE: Self = Self(1 << 3);
    pub const COLOR_ATTACHMENT: Self = Self(1 << 4);
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(1 << 5);

    pub fn contains(self, other: Self) -> bool;
    pub fn union(self, other: Self) -> Self;
    pub fn is_empty(self) -> bool;
}
```

Usage is a creation-time correctness contract.

For example:

```text
COPY_DST not declared
    -> cannot use Upload / copy destination

COPY_SRC not declared
    -> cannot use Readback / copy source

STORAGE not declared
    -> cannot be a storage binding
```

The backend cannot bypass portable usage validation just because a platform "happens to allow it".

---

## 11.2 Resource memory preference

General mapping is an explicit usage/capability contract, while placement
remains a **pure performance preference**:

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceMemoryPreference {
    /// Chosen by the backend.
    Automatic,

    /// Prefer GPU/device-local placement where possible.
    ///
    /// This is a preference, not a correctness guarantee.
    /// UMA / WebGPU / GL backends may treat it equivalently or ignore it.
    DeviceLocalPreferred,
}
```

`ResourceMemoryPreference` does not imply host visibility. An ordinary host
map lease requires the matching `MAP_READ` or `MAP_WRITE` bit at creation, a
supported exact `BufferSupportQuery`, and an available mapping lease. It does
**not** require `MappablePrimaryBuffers`: `MAP_READ | COPY_DST` readback and
`MAP_WRITE | COPY_SRC` upload are ordinary staging-buffer contracts.

`MappablePrimaryBuffers` answers the narrower, additional question of whether
a map usage may coexist with broader primary GPU usages such as `VERTEX`,
`UNIFORM`, or `STORAGE`. That question is still represented by the exact
`BufferSupportQuery`; the optional feature lets requirements distinguish a
backend that supports those primary combinations from one that supports only
ordinary staging buffers. The remaining mapping facts are:

```text
PersistentMapping
CoherentMapping or explicit flush/invalidate
MapAlignment
```

`Device::map_buffer` returns `RhiResult<MapBufferFuture<'_>>`. The future may
remain pending until conflicting GPU use retires, registers its executor waker,
and terminates as `DeviceLost` if the device is lost. Awaiting it yields
`MappedRange<'buffer>`, a borrowing RAII lease; dropping either the pending
future or the ready range releases exactly one exclusive mapping reservation. A
write range exposes mutable bytes and may be flushed; a read range exposes
immutable bytes and may be invalidated. Wrong usage, device identity,
empty/out-of-bounds/overflowing range, alignment, and a second concurrent map
are refused before native mapping begins.

Mapped leases also participate in submit Phase-A validation. On a device without
`PersistentMapping`, a buffer with a pending or ready mapping lease cannot be
accepted by `submit`, even if the eventual command uses a disjoint byte range;
the caller must drop/unmap first. With `PersistentMapping` enabled, an active
lease is allowed, but it does not make simultaneous CPU/GPU access to overlapping
bytes data-race-free. The caller owns that synchronization: CPU writes to a
non-coherent lease must be flushed before submitting GPU work that reads the
same bytes; CPU reads must wait for the last GPU writer to complete and then
invalidate non-coherent memory; and CPU code must not read or write bytes while
accepted GPU work may concurrently write those bytes. Coherent memory removes
cache-maintenance calls, not this execution-order requirement. A backend may
publish `PersistentMapping` only when its native allocation and submission
barriers uphold these rules. This is checked before native work is accepted, so
failure cannot produce a partial submission.

The borrowing result is deliberate: the lease needs no extra cloned resource
handle or reference-count operation merely to keep the buffer alive.

---

# 12. Buffer

## 12.1 Buffer capability

Texture already has descriptor-dependent `TextureSupportQuery`;
Buffer must also have same level facts.

Otherwise for example:

```text
WebGL2:
    VERTEX / INDEX / UNIFORM / COPY available
    STORAGE unavailable
```

There is no stable query entry.

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferSupportQuery {
    usage: BufferUsage,
}

impl BufferSupportQuery {
    pub fn new(
        usage: BufferUsage,
    ) -> Self;

    pub fn usage(&self) -> BufferUsage;
}

#[derive(Clone, Copy, Debug)]
pub struct BufferSupportLimits {
    max_size: u64,
}

impl BufferSupportLimits {
    pub fn max_size(&self) -> u64;
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum BufferSupport {
    Unsupported,
    Supported(BufferSupportLimits),
}

impl BufferSupport {
    pub fn is_supported(&self) -> bool;
    pub fn limits(&self) -> Option<&BufferSupportLimits>;
}
```

`BufferUsage::STORAGE` on a backend that does not support shader storage buffers:

```text
buffer_support(STORAGE)
    -> Unsupported
```

Instead of creating it successfully first and then failing secretly when waiting for BindGroup.

---

## 12.2 BufferDescriptor

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BufferDescriptor {
    pub label: Label,
    pub size: u64,
    pub usage: BufferUsage,
    pub memory: ResourceMemoryPreference,
}

impl BufferDescriptor {
    pub fn new(
        size: u64,
        usage: BufferUsage,
    ) -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;

    pub fn with_memory_preference(
        mut self,
        preference: ResourceMemoryPreference,
    ) -> Self;
}
```

Delete old fields:

```text
element_stride_hint
host_access
```

reason:

- P0 Buffer is byte-addressed;
- structured/texel stride should belong to future `BufferView`;
- CPU host access is governed by the separately frozen General Mapping
  capability and lease contract, not a placement field.

---

## 12.3 Buffer object

```rust
#[derive(Clone)]
pub struct Buffer {
    /* opaque */
}

impl Device {
    pub fn create_buffer(
        &self,
        desc: &BufferDescriptor,
    ) -> RhiResult<Buffer>;
}

impl Buffer {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn descriptor(&self) -> &BufferDescriptor;
}
```

Must verify when creating:

```text
size > 0
usage non-empty
BufferSupportQuery(usage) == Supported
size <= BufferSupportLimits.max_size
```

If the Buffer size is 0, the P0 portable contract will not be entered.

---

## 12.4 Buffer range / binding

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferRange {
    pub offset: u64,
    pub size: u64,
}

impl BufferRange {
    pub fn new(
        offset: u64,
        size: u64,
    ) -> Self;

    pub fn end(&self) -> Option<u64>;
}

#[derive(Clone)]
pub struct BufferBinding {
    pub buffer: Buffer,
    pub range: BufferRange,
}

impl BufferBinding {
    pub fn new(
        buffer: Buffer,
        range: BufferRange,
    ) -> Self;
}
```

All points of use must verify:

```text
size > 0
offset + size has no integer overflow
offset + size <= buffer.size
corresponding binding/copy alignment
DeviceIdentity
```

P0 does not use the `{ offset, size = WHOLE_BUFFER }` type of magic sentinel.

---

# 13. Texture

## 13.1 Dimension / Extent

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent3d {
    pub fn d1(width: u32) -> Self;
    pub fn d2(width: u32, height: u32) -> Self;
    pub fn d3(width: u32, height: u32, depth: u32) -> Self;
}
```

Portable creation invariants：

```text
all extent components > 0
mip_levels > 0
array_layers > 0
sample_count > 0

D1:
    height = 1
    depth = 1
    array_layers = 1
    sample_count = 1

D2:
    depth = 1

D3:
    array_layers = 1
    sample_count = 1

sample_count > 1:
    dimension = D2
    mip_levels = 1

mip_levels <= floor(log2(max(width, height, depth))) + 1

view_formats:
    canonicalize as a set (sort, deduplicate)
    may not include the base format itself
```

The finer maximum value is determined by `TextureSupport`.

---

## 13.2 Texture view creation compatibility

Certain view semantics must be declared in advance when the Texture is created.

The most important P0 example is Cube:

- Vulkan cube view requires the cube-compatible intent to be used when creating the image;
- Cannot wait for `create_texture_view()` to add the native create flag.

therefore:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureViewCompatibility(u32);

impl TextureViewCompatibility {
    pub const NONE: Self = Self(0);

    /// Texture permits creation of Cube / CubeArray views.
    pub const CUBE: Self = Self(1 << 0);

    pub fn contains(self, other: Self) -> bool;
    pub fn union(self, other: Self) -> Self;
}
```

P0:

```text
CUBE:
    dimension must be D2
    width == height
    array_layers >= 6
    sample_count == 1
```

future:

```text
3D -> 2D sliced view
block-texel reinterpretation
video/planar view
```

Add compatibility bit/extension respectively; do not secretly borrow `TextureViewDimension` and only ask for the backend's miraculous complement capability after creation.

---

## 13.3 TextureDescriptor

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TextureDescriptor {
    pub label: Label,

    pub dimension: TextureDimension,
    pub extent: Extent3d,

    pub mip_levels: u32,
    pub array_layers: u32,
    pub sample_count: u32,

    pub format: TextureFormat,
    pub usage: TextureUsage,

    /// Set of formats allowed for alternate-format views.
    pub view_formats: Vec<TextureFormat>,

    /// View intent that must be known when Texture is created.
    pub view_compatibility: TextureViewCompatibility,

    pub memory: ResourceMemoryPreference,
}

impl TextureDescriptor {
    pub fn new_1d(
        width: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self;

    pub fn new_2d(
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self;

    pub fn new_3d(
        width: u32,
        height: u32,
        depth: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;

    pub fn with_mip_levels(
        mut self,
        levels: u32,
    ) -> Self;

    pub fn with_array_layers(
        mut self,
        layers: u32,
    ) -> Self;

    pub fn with_sample_count(
        mut self,
        samples: u32,
    ) -> Self;

    pub fn with_view_format(
        mut self,
        format: TextureFormat,
    ) -> Self;

    pub fn with_view_compatibility(
        mut self,
        compatibility: TextureViewCompatibility,
    ) -> Self;

    pub fn with_memory_preference(
        mut self,
        preference: ResourceMemoryPreference,
    ) -> Self;
}
```

The final Texture capability query must use the same creation semantics:

```rust
let query = TextureSupportQuery::new(
    desc.dimension,
    desc.format,
    desc.usage,
    desc.sample_count,
)
.with_view_compatibility(desc.view_compatibility);

// Then add desc.view_formats
```

so:

```text
capability query
create_texture validation
backend image creation
```

Three different sets of conditions will not be used.

---

## 13.4 Texture object

```rust
#[derive(Clone)]
pub struct Texture {
    /* opaque */
}

impl Device {
    pub fn create_texture(
        &self,
        desc: &TextureDescriptor,
    ) -> RhiResult<Texture>;
}

impl Texture {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn descriptor(&self) -> &TextureDescriptor;
}
```

Texture creation must verify:

```text
dimension invariants
extent/mip/layer/sample count
usage non-empty
view format compatibility
view creation compatibility
TextureSupportQuery
TextureSupportLimits
DeviceIdentity
```

---



# 14. Texture subresource / texel layout types

Tracking/View and Copy/Upload/Readback use distinct types.

---

## 14.1 Aspect

~~~rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureAspect {
    Color,
    Depth,
    Stencil,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureAspects(u8);

impl TextureAspects {
    pub const COLOR: Self = Self(1 << 0);
    pub const DEPTH: Self = Self(1 << 1);
    pub const STENCIL: Self = Self(1 << 2);

    pub fn contains(self, other: Self) -> bool;
    pub fn union(self, other: Self) -> Self;
}
~~~

---

## 14.2 View / hazard range

~~~rust
/// View / hazard tracking: may cover multiple mips + multiple array layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSubresourceRange {
    pub aspects: TextureAspects,
    pub base_mip: u32,
    pub mip_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}
~~~

For TextureDimension::D3:

~~~text
base_layer = 0
layer_count = 1
~~~

A 3D texture Z slice is not an independent array subresource; its Z range is expressed only by copy origin/extent.

---

## 14.3 Copy / Upload / Readback layers

~~~rust
/// One mip + a range of array layers.
///
/// Semantically analogous to Vulkan ImageSubresourceLayers, but not a native struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSubresourceLayers {
    pub aspect: TextureAspect,
    pub mip_level: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}
~~~

Frozen rules:

~~~text
Each layers value may select only one aspect

D1:
    base_layer = 0
    layer_count = 1

D2:
    base_layer/layer_count select array layers

D3:
    base_layer = 0
    layer_count = 1
~~~

For multi-aspect depth-stencil formats:

~~~text
Depth / Stencil copy routes are queried and encoded separately
~~~

TextureSubresourceLayers may not carry both Depth|Stencil bits at once.

---

## 14.4 Origin / extent semantics

~~~rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Origin3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
~~~

Copy/Upload/Readback:

~~~text
D1:
    origin.y = origin.z = 0
    extent.height = extent.depth = 1

D2 / D2Array:
    origin.z = 0
    extent.depth = 1
    array range is expressed by TextureSubresourceLayers

D3:
    array layer is fixed at 0/1
    origin.z + extent.depth express the Z-slice range
~~~

Thus D2 array layers and D3 Z slices are not conflated into one depth_or_layers in the portable API.

---

## 14.5 Host texel data layout

An upload source is ordinary CPU bytes; its layout **is not the GPU copy-buffer layout requirement**.

~~~rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTexelLayout {
    /// Byte distance between starts of adjacent rows in the CPU source.
    pub bytes_per_row: u32,

    /// Number of rows between starts of adjacent images/layers/depth slices in the CPU source.
    pub rows_per_image: u32,
}
~~~

It must satisfy:

~~~text
bytes_per_row >= logical row bytes
bytes_per_row is aligned to format block bytes
rows_per_image >= logical block-row count
the source byte range covers the final copied texel/block
~~~

But it **need not** satisfy:

~~~text
the 256-byte bytesPerRow required by WebGPU command copy
the 256-byte row-pitch / placement alignment of a D3D12 copy footprint
~~~

If source layout does not meet backend staging-copy alignment:

~~~text
UploadJob
    -> RHI-private staging/repack
    -> GPU copy
~~~

This is host-to-GPU staging behavior promised by Upload itself, not a backend secretly applying fallback to a Copy command. D3D12 and WebGPU command-buffer texture copies have explicit row-pitch alignment, so these native alignments must not be imposed on an asset loader's CPU source bytes.

---

# 15. TextureView

## 15.1 View dimension

~~~rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureViewDimension {
    D1,
    D1Array,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}
~~~

`D1Array` and `D3 -> D2/D2Array` sliced views are descriptor-dependent view
routes. They are present only when the concrete texture/view route fact reports
support; the descriptor validates the selected layer/slice bounds and otherwise
returns `Unsupported`. A `ColorAttachment::depth_slice` is separately valid for
3D attachment rendering and does not require exposing a native view handle.

---

## 15.2 TextureViewDescriptor

~~~rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TextureViewDescriptor {
    pub label: Label,

    pub dimension: TextureViewDimension,

    /// None = use the base Texture format.
    pub format: Option<TextureFormat>,

    pub aspects: TextureAspects,

    /// Requested uses of this view. Must be a subset of the parent texture's
    /// usage and supported by the concrete format/aspect/view route.
    pub usage: TextureUsage,

    pub base_mip: u32,
    pub mip_count: u32,

    pub base_layer: u32,
    pub layer_count: u32,
}

impl TextureViewDescriptor {
    pub fn new(
        dimension: TextureViewDimension,
        aspects: TextureAspects,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
        layer_count: u32,
    ) -> Self;

    /// Constructs a view covering the complete logical subresource range from a Texture descriptor.
    ///
    /// Cube/CubeArray compatibility validation is still performed.
    pub fn whole(
        texture: &Texture,
        dimension: TextureViewDimension,
    ) -> RhiResult<Self>;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;

    pub fn with_format(
        mut self,
        format: TextureFormat,
    ) -> Self;
}
~~~

The old whole_2d() is removed because it is not a complete Base constructor for 1D/3D/array/cube.

---

## 15.3 TextureView object

~~~rust
#[derive(Clone)]
pub struct TextureView {
    /* opaque */
}

impl Device {
    pub fn create_texture_view(
        &self,
        texture: &Texture,
        desc: &TextureViewDescriptor,
    ) -> RhiResult<TextureView>;
}

impl TextureView {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn texture(&self) -> &Texture;
    pub fn descriptor(&self) -> &TextureViewDescriptor;

    /// Actual view format.
    pub fn format(&self) -> TextureFormat;

    pub fn aspects(&self) -> TextureAspects;

    /// Logical texel extent of base_mip; array layers do not contribute to depth.
    pub fn extent(&self) -> Extent3d;

    pub fn sample_count(&self) -> u32;
    pub fn layer_count(&self) -> u32;
}
~~~

The following must be validated:

~~~text
DeviceIdentity
format reinterpretation
TextureDescriptor.view_formats
aspect
mip/layer range
dimension compatibility
TextureDescriptor.view_compatibility
sample count restrictions
~~~

### Aspect compatibility

~~~text
Color format view:
    aspects == COLOR

Sampled depth view:
    aspects == DEPTH

P0 provides no sampled stencil semantics

StorageTexture:
    aspects == COLOR

Depth/stencil attachment:
    DEPTH, STENCIL, or both may be selected (depending on format)
~~~

Specifically:

~~~text
D1      -> D1
D2      -> D2
D2Array -> D2 texture with array layers
Cube    -> D2 + CUBE compatibility + exactly 6 layers
CubeArray -> D2 + CUBE compatibility + layer_count multiple of 6
D3      -> D3
~~~

---

# 16. Sampler

## 16.1 Descriptor

~~~rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SamplerDescriptor {
    pub label: Label,

    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,

    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mip_filter: FilterMode,

    pub lod_min: f32,
    pub lod_max: f32,

    pub compare: Option<CompareFunction>,

    /// 1 = anisotropy disabled.
    pub max_anisotropy: u16,
}

impl SamplerDescriptor {
    pub fn new() -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;

    pub fn with_address_modes(
        mut self,
        u: AddressMode,
        v: AddressMode,
        w: AddressMode,
    ) -> Self;

    pub fn with_filters(
        mut self,
        mag: FilterMode,
        min: FilterMode,
        mip: FilterMode,
    ) -> Self;

    pub fn with_lod_clamp(
        mut self,
        min: f32,
        max: f32,
    ) -> Self;

    pub fn with_compare(
        mut self,
        compare: CompareFunction,
    ) -> Self;

    pub fn with_max_anisotropy(
        mut self,
        value: u16,
    ) -> Self;
}
~~~

Default:

~~~text
max_anisotropy = 1
~~~

If max_anisotropy > 1:

~~~text
OptionalFeature::SamplerAnisotropy must be enabled
value <= DeviceLimits::MaxSamplerAnisotropy
~~~

This is an RHI-wide optional capability, not a WebGL-only escape hatch. DX12
has a descriptor-defined `1..=16` range, Vulkan publishes the enabled
`samplerAnisotropy` feature together with its physical-device limit, and the
GL family publishes the acquired extension's queried limit. Consequently an
application may request a value greater than one whenever the selected
device publishes both facts; it is never silently degraded to isotropic
filtering.

WebGL2 anisotropy comes from an extension and therefore cannot be an
unconditional Base capability. WebGPU is deliberately more conservative for
now: its `GPUSamplerDescriptor.maxAnisotropy` field is lowered privately, but
the standard API does not expose a `GPUSupportedLimits` value equivalent to
`MaxSamplerAnisotropy`. Browser implementations clamp to their own supported
ceiling. Fluxel therefore does **not** publish `SamplerAnisotropy` or
`MaxSamplerAnisotropy` for WebGPU, and public requests above one fail before
native creation rather than relying on an unqueryable clamp. Should that fact
become portable and queryable, enabling it is a capability-probe change, not a
sampler API redesign.

WebGPU additionally requires all of `mag_filter`, `min_filter`, and
`mip_filter` to be `Linear` when `max_anisotropy > 1`. This backend-specific
native validation is stricter than the portable sampler-kind classification;
while WebGPU remains fail-closed above one it cannot be reached through the
public API.

The following must also be validated:

~~~text
lod_min / lod_max finite
lod_min <= lod_max
max_anisotropy >= 1
~~~

Backend-specific sampler-object / texture-state lowering is an implementation detail.

### SamplerKind compatibility

~~~text
Comparison:
    sampler.compare != None

NonFiltering:
    sampler.compare == None
    mag/min/mip == Nearest
    max_anisotropy == 1

Filtering:
    sampler.compare == None
    (Nearest or Linear is legal; Linear/anisotropy require the corresponding capability)
~~~

SamplerKind describes the binding contract; it is not whether a sampler descriptor “currently uses linear.”

---

## 16.2 Sampler object

~~~rust
#[derive(Clone)]
pub struct Sampler {
    /* opaque */
}

impl Device {
    pub fn create_sampler(
        &self,
        desc: &SamplerDescriptor,
    ) -> RhiResult<Sampler>;
}

impl Sampler {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn descriptor(&self) -> &SamplerDescriptor;
}
~~~



# 17. Upload

Upload is **host -> GPU resource mutation workflow**.

It is not a general mapping, nor a normal GPU Copy command.

---

## 17.1 Upload descriptor

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BufferUploadDescriptor {
    pub label: Label,

    pub dst: Buffer,
    pub dst_offset: u64,

    /// Retained immutable source bytes。
    pub bytes: std::sync::Arc<[u8]>,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TextureUploadDescriptor {
    pub label: Label,

    pub dst: Texture,

    pub subresource: TextureSubresourceLayers,
    pub origin: Origin3d,
    pub extent: Extent3d,

    /// CPU source layout, not native GPU copy layout.
    pub source_layout: HostTexelLayout,

    /// Retained immutable source bytes。
    pub bytes: std::sync::Arc<[u8]>,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum UploadDescriptor {
    Buffer(BufferUploadDescriptor),
    Texture(TextureUploadDescriptor),
}
```

---

## 17.2 UploadJob

```rust
#[derive(Clone)]
pub struct UploadJob {
    /* opaque; owns retained source payload */
}

impl Device {
    pub fn create_buffer_upload(
        &self,
        desc: BufferUploadDescriptor,
    ) -> RhiResult<UploadJob>;

    pub fn create_texture_upload(
        &self,
        desc: TextureUploadDescriptor,
    ) -> RhiResult<UploadJob>;
}

impl UploadJob {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;

    /// Capture/tooling can obtain the complete portable mutation descriptor.
    pub fn descriptor(&self) -> &UploadDescriptor;
}
```

`UploadJob` uses `&UploadJob` encoding, so P0 allows the same retained payload to be encoded repeatedly;
Each encode represents an independent resource mutation.

---

## 17.3 Upload validation

### Buffer

must:

```text
dst usage includes COPY_DST
dst_offset + bytes.len() does not overflow
dst_offset + bytes.len() <= dst.size
bytes non-empty
BufferCopyLayoutLimits of RouteQuery::BufferToBuffer satisfied
DeviceIdentity matches
```

### Texture

must:

```text
dst usage includes COPY_DST
TextureSubresourceLayers valid
origin/extent valid
dst.sample_count == 1
source_layout sufficiently covers source bytes
format/aspect valid
corresponding upload/copy route realizable
DeviceIdentity matches
```

### Native copy alignment

Upload **does not require the caller to meet native staging alignment**.

For example, the row pitch of the D3D12 texture upload footprint is usually aligned by 256, and the `bytesPerRow` of the WebGPU command `copyBufferToTexture` also requires a multiple of 256; RHI can repack the caller's normal CPU layout into private staging.

therefore:

```text
public HostTexelLayout
    !=
RouteSupport.texel_copy_layout()
```

The former is the semantics of caller bytes.

The latter is the legality/alignment of the GPU-side buffer<->texture route.

---

# 18. Readback

Readback is:

```text
encode request
    ->
RecordedWork
    ->
successful async submit
    ->
GPU terminal completion
    ->
CPU-visible scoped read
```

Not map immediately.

---

## 18.1 Request

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ReadbackRequest {
    Buffer {
        label: Label,
        src: Buffer,
        range: BufferRange,
    },

    Texture {
        label: Label,

        src: Texture,

        subresource: TextureSubresourceLayers,
        origin: Origin3d,
        extent: Extent3d,
    },
}
```

Validation：

```text
Buffer:
    usage includes COPY_SRC
    range valid
    BufferCopyLayoutLimits of RouteQuery::BufferToBuffer satisfied

Texture:
    usage includes COPY_SRC
    src.sample_count == 1
    subresource/origin/extent valid
    TextureToBuffer route Supported

all input DeviceIdentity values match
```

---

## 18.2 State

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadbackStatus {
    NotSubmitted,
    Pending,
    Ready,
    Abandoned,
    DeviceLost,
    Failed,
}
```

State machine:

```text
NotSubmitted
    ├─ submit accepted -> Pending
    ├─ owner dropped   -> Abandoned
    └─ device loss     -> DeviceLost

Pending
    ├─ success         -> Ready
    ├─ device loss     -> DeviceLost
    └─ backend failure -> Failed
```

Won't be permanently stuck in:

```text
NotSubmitted
Pending
```

And there is no owner.

---

## 18.3 Result layout

Readback does not promise to return tightly-packed texture bytes.

This is intentional:

- D3D12 copy footprint distinguishes between unpadded row size and aligned row pitch;
- Different backend staging layouts may be different;
- Forcing RHI to do another CPU repack will increase unnecessary costs.

So explicitly return layout.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadbackTexelLayout {
    /// Byte distance between starts of adjacent valid rows.
    pub bytes_per_row: u32,

    /// Number of rows between starts of adjacent images/layers/depth slices.
    pub rows_per_image: u32,

    /// Total length of returned byte slice.
    pub total_size: u64,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ReadbackViewData<'a> {
    Buffer {
        bytes: &'a [u8],
    },

    Texture {
        bytes: &'a [u8],
        layout: ReadbackTexelLayout,
    },
}
```

`bytes_per_row` can be larger than logical row bytes.

Capture Artifact if required canonical tightly-packed blob:

```text
ReadbackViewData
    -> Capture layer canonicalize/repack
    -> BlobStore
```

Don't stuff artifact policy into RHI.

---

/// Scoped CPU view.
///
/// If a backend uses map/unmap internally, Drop releases the mapping lease.
pub struct ReadbackView<'a> {
    /* opaque RAII guard */
    _marker: std::marker::PhantomData<&'a ReadbackTicket>,
}

impl<'a> ReadbackView<'a> {
    pub fn data(&self) -> ReadbackViewData<'_>;
}

This prevents a backend mapping lifetime from escaping as a bare `&[u8]`.

---

## 18.4 Ticket

```rust
#[derive(Clone)]
pub struct ReadbackTicket {
    /* opaque, shared device-scoped state */
}

impl ReadbackTicket {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;

    pub fn request(&self) -> &ReadbackRequest;
    pub fn status(&self) -> ReadbackStatus;
    pub fn completion(&self) -> Option<CompletionPoint>;

    /// Non-blocking fast path.
    pub fn try_read(&self) -> RhiResult<Option<ReadbackView<'_>>>;

    /// Waits for readback data; it does not busy-loop Device::poll().
    pub async fn read(&self) -> RhiResult<ReadbackView<'_>>;
}
```

---

## 18.5 Encode

`CommandRecorder` provides:

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

Both operations enter `RecordedWork` actual resource use: Upload is destination
`COPY_WRITE`; Readback is source `COPY_READ`. This is RHI’s own
`rhi::command::ResourceUse` vocabulary and has no renderer-scheduler contract.

---

## 18.6 Resource lifetime / retirement

All resource objects obey the same rules:

```text
public handle drop
    !=
logical object is no longer referenced
    !=
native backing may be released immediately
```

RHI must maintain backing until at least:

```text
final CPU logical owner no longer references it
+
all accepted GPU work referencing that object is terminal
```

where terminal:

```text
Complete
DeviceLost
Failed
```

For example:

```text
Buffer handle dropped by user
but RecordedWork still references it
    -> resource remains valid

RecordedWork submitted
user drops all Buffer clones
    -> native backing still cannot be released

Completion terminal
and no other logical owner
    -> may reclaim
```

Descriptor storage / staging allocation / view backing / sampler backing also comply with completion-safe retirement.

---

## 18.7 Device loss

After Device loss:

```text
UploadJob
ReadbackTicket
Buffer / Texture / View / Sampler
```

All belong to lost device.

Readback tickets must be entered:

```text
DeviceLost
```

Rather than being permanently pending.

The resource handle of the lost Device is passed into the Device obtained by the new request:

```text
WrongDevice
```

The native handle/value cannot be restored to validity because it is reused by the driver.

---

## 18.8 Capture / Statistics implication

The Resource layer must guarantee for future Capture/Replay:

```text
Buffer / Texture / View / Sampler descriptors recoverable
UploadJob descriptor + retained bytes observable
Readback returns explicit layout
ObjectId / DeviceIdentity stable
resource lifetime/reclaim event observable
```

Statistics：

```text
create count
reclaim count
live inventory
logical memory estimate
```
