//! Shared resource leases, opaque accessors, and fixed binding contracts.
//!
//! This module owns the safe handles that retain native resource state. Upload
//! operation state lives in sibling modules so completion and Drop invariants stay
//! local to the operation they protect.

use super::*;
/// A type-erased strong lease retained by native frame submissions.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ResourceLease {
    /// Retains one buffer allocation.
    Buffer(BufferLease),
    /// Retains one texture allocation.
    Texture(TextureLease),
    /// Retains a native compute pipeline, module, and layout.
    ComputePipeline(ComputePipelineLease),
    /// Retains a native compute binding object, pipeline, and buffer.
    ComputeBindings(ComputeBindingsLease),
    /// Retains a native raster pipeline, module, and layout.
    RasterPipeline(RasterPipelineLease),
    /// Retains the closed camera/material raster uniform binding.
    RasterUniformBindings(RasterUniformBindingsLease),
    /// Retains the closed textured raster binding.
    RasterTextureBindings(RasterTextureBindingsLease),
    /// Retains the closed explicit-UV textured raster binding.
    RasterUvTextureBindings(RasterUvTextureBindingsLease),
    /// Retains the closed explicit-UV linear-clamp sampler binding.
    RasterUvLinearClampTextureBindings(RasterUvLinearClampTextureBindingsLease),
    /// Retains the closed normal-Lambert binding and both vertex streams.
    RasterNormalBindings(RasterNormalBindingsLease),
    /// Retains the closed position-and-RGBA8 vertex-color raster binding.
    RasterVertexColorBindings(RasterVertexColorBindingsLease),
    /// Retains the closed sampled-texture-to-storage-buffer binding object.
    TexturePackBindings(TexturePackBindingsLease),
}

macro_rules! resource_accessors {
    ($resource:ident, $lease:ident, $shared:ident, $desc:ty, $usage:ty) => {
        impl $resource {
            /// Returns the descriptor validated at creation.
            pub fn descriptor(&self) -> $desc {
                self.0.descriptor
            }
            /// Returns operations proven by the final native creation facts.
            ///
            /// This is an authorization upper bound, not necessarily the exact
            /// requested set. Native normalization may widen a request; for
            /// example, buffer storage write is reported as storage read/write.
            pub fn allowed_usage(&self) -> $usage {
                self.0.allowed_usage
            }
            /// Returns this physical generation's opaque identity.
            pub fn identity(&self) -> PhysicalResourceIdentity {
                self.0.identity
            }
            /// Returns the owning device identity.
            pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
                self.0.device
            }
            /// Acquires a token that keeps the native object and device alive.
            pub fn lease(&self) -> $lease {
                $lease(Arc::clone(&self.0))
            }
        }
        impl fmt::Debug for $resource {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($resource))
                    .field("descriptor", &self.0.descriptor)
                    .field("allowed_usage", &self.0.allowed_usage)
                    .field("identity", &self.0.identity)
                    .finish_non_exhaustive()
            }
        }
        impl fmt::Debug for $lease {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($lease))
                    .field(&self.0.identity)
                    .finish()
            }
        }
    };
}

resource_accessors!(
    Buffer,
    BufferLease,
    BufferShared,
    BufferDescriptor,
    BufferUsage
);
resource_accessors!(
    Texture,
    TextureLease,
    TextureShared,
    TextureDescriptor,
    TextureUsage
);

impl BufferLease {
    #[allow(
        dead_code,
        reason = "the test-only exported-buffer wrapper consumes this exact lease identity"
    )]
    pub(crate) fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    #[allow(
        dead_code,
        reason = "the test-only exported-buffer wrapper consumes this exact lease identity"
    )]
    pub(crate) fn identity(&self) -> PhysicalResourceIdentity {
        self.0.identity
    }
}

impl TextureLease {
    #[allow(
        dead_code,
        reason = "the test-only exported-texture wrapper consumes this exact lease identity"
    )]
    pub(crate) fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }

    #[allow(
        dead_code,
        reason = "the test-only exported-texture wrapper consumes this exact lease identity"
    )]
    pub(crate) fn identity(&self) -> PhysicalResourceIdentity {
        self.0.identity
    }
}

impl From<BufferLease> for ResourceLease {
    fn from(value: BufferLease) -> Self {
        Self::Buffer(value)
    }
}

impl From<TextureLease> for ResourceLease {
    fn from(value: TextureLease) -> Self {
        Self::Texture(value)
    }
}

impl From<ComputePipelineLease> for ResourceLease {
    fn from(value: ComputePipelineLease) -> Self {
        Self::ComputePipeline(value)
    }
}

impl From<ComputeBindingsLease> for ResourceLease {
    fn from(value: ComputeBindingsLease) -> Self {
        Self::ComputeBindings(value)
    }
}

impl From<RasterPipelineLease> for ResourceLease {
    fn from(value: RasterPipelineLease) -> Self {
        Self::RasterPipeline(value)
    }
}

impl From<RasterUniformBindingsLease> for ResourceLease {
    fn from(value: RasterUniformBindingsLease) -> Self {
        Self::RasterUniformBindings(value)
    }
}
impl From<RasterTextureBindingsLease> for ResourceLease {
    fn from(value: RasterTextureBindingsLease) -> Self {
        Self::RasterTextureBindings(value)
    }
}
impl From<RasterUvTextureBindingsLease> for ResourceLease {
    fn from(value: RasterUvTextureBindingsLease) -> Self {
        Self::RasterUvTextureBindings(value)
    }
}
impl From<RasterUvLinearClampTextureBindingsLease> for ResourceLease {
    fn from(value: RasterUvLinearClampTextureBindingsLease) -> Self {
        Self::RasterUvLinearClampTextureBindings(value)
    }
}

impl From<RasterNormalBindingsLease> for ResourceLease {
    fn from(value: RasterNormalBindingsLease) -> Self {
        Self::RasterNormalBindings(value)
    }
}
impl From<RasterVertexColorBindingsLease> for ResourceLease {
    fn from(value: RasterVertexColorBindingsLease) -> Self {
        Self::RasterVertexColorBindings(value)
    }
}

impl From<TexturePackBindingsLease> for ResourceLease {
    fn from(value: TexturePackBindingsLease) -> Self {
        Self::TexturePackBindings(value)
    }
}

impl Buffer {
    pub(crate) fn native(&self) -> &crate::imp::OwnedBuffer {
        &self.0._native
    }
}

impl Texture {
    pub(crate) fn native(&self) -> &crate::imp::OwnedTexture {
        &self.0._native
    }
}

impl ComputePipeline {
    /// Returns the fixed kernel selected during creation.
    pub fn kernel(&self) -> ComputeKernel {
        self.0.kernel
    }
    pub(crate) fn same_object(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining the native pipeline and its artifacts.
    pub fn lease(&self) -> ComputePipelineLease {
        ComputePipelineLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeComputePipeline {
        &self.0._native
    }
}

impl ComputeBindings {
    /// Returns the pipeline this binding object was created for.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.0.pipeline
    }
    /// Returns the exact authorized storage-buffer byte range.
    pub fn range(&self) -> (u64, u64) {
        (self.0.offset, self.0.size)
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining all native objects referenced by this binding.
    pub fn lease(&self) -> ComputeBindingsLease {
        ComputeBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeComputeBindings {
        &self.0._native
    }
}

impl RasterPipeline {
    /// Returns the fixed artifact selected during creation.
    pub fn kernel(&self) -> RasterKernel {
        self.0.kernel
    }
    pub(crate) fn same_object(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining the native pipeline and its artifacts.
    pub fn lease(&self) -> RasterPipelineLease {
        RasterPipelineLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterPipeline {
        &self.0._native
    }
}

impl RasterUniformBindings {
    /// Returns the sole camera/material pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining the binding, pipeline, and uniform buffer.
    pub fn lease(&self) -> RasterUniformBindingsLease {
        RasterUniformBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterUniformBindings {
        &self.0._native
    }
}

impl RasterTextureBindings {
    /// Returns the textured raster pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a strong lifetime lease.
    pub fn lease(&self) -> RasterTextureBindingsLease {
        RasterTextureBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterTextureBindings {
        &self.0._native
    }
}

impl RasterUvTextureBindings {
    /// Returns the sole explicit-UV pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the physical buffer generation required at vertex slot zero.
    pub fn position_identity(&self) -> PhysicalResourceIdentity {
        self.0.position_identity
    }
    /// Returns the physical buffer generation required at vertex slot one.
    pub fn texture_coordinate_identity(&self) -> PhysicalResourceIdentity {
        self.0.texture_coordinate_identity
    }
    /// Returns the exact number of vertices represented by both streams.
    pub fn vertex_count(&self) -> u32 {
        self.0.vertex_count
    }
    /// Acquires a terminal-lifetime lease for the complete binding.
    pub fn lease(&self) -> RasterUvTextureBindingsLease {
        RasterUvTextureBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterTextureBindings {
        &self.0._native
    }
    pub(crate) fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
}

impl RasterUvLinearClampTextureBindings {
    /// Returns the sole explicit-UV linear-clamp pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the physical buffer generation required at vertex slot zero.
    pub fn position_identity(&self) -> PhysicalResourceIdentity {
        self.0.position_identity
    }
    /// Returns the physical buffer generation required at vertex slot one.
    pub fn texture_coordinate_identity(&self) -> PhysicalResourceIdentity {
        self.0.texture_coordinate_identity
    }
    /// Returns the exact number of vertices represented by both streams.
    pub fn vertex_count(&self) -> u32 {
        self.0.vertex_count
    }
    /// Acquires a terminal-lifetime lease for the complete binding and its
    /// private native sampler.
    pub fn lease(&self) -> RasterUvLinearClampTextureBindingsLease {
        RasterUvLinearClampTextureBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterTextureBindings {
        &self.0._native
    }
    pub(crate) fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
}

impl RasterNormalBindings {
    /// Returns the only normal-Lambert pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the physical generation required at vertex slot zero.
    pub fn position_identity(&self) -> PhysicalResourceIdentity {
        self.0.position_identity
    }
    /// Returns the physical generation required at vertex slot one.
    pub fn normal_identity(&self) -> PhysicalResourceIdentity {
        self.0.normal_identity
    }
    /// Returns the exact number of vertices represented by both streams.
    pub fn vertex_count(&self) -> u32 {
        self.0.vertex_count
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a terminal-lifetime lease for the complete binding.
    pub fn lease(&self) -> RasterNormalBindingsLease {
        RasterNormalBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterUniformBindings {
        &self.0._native
    }
}

impl RasterVertexColorBindings {
    /// Returns the only vertex-color pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the position generation required at slot zero.
    pub fn position_identity(&self) -> PhysicalResourceIdentity {
        self.0.position_identity
    }
    /// Returns the RGBA8 color generation required at slot one.
    pub fn color_identity(&self) -> PhysicalResourceIdentity {
        self.0.color_identity
    }
    /// Returns the exact count represented by both streams.
    pub fn vertex_count(&self) -> u32 {
        self.0.vertex_count
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a terminal-lifetime lease for this complete binding.
    pub fn lease(&self) -> RasterVertexColorBindingsLease {
        RasterVertexColorBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterUniformBindings {
        &self.0._native
    }
}

impl TexturePackBindings {
    /// Returns the `TexturePackRgba8` pipeline selected during creation.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.0.pipeline
    }
    /// Returns the exact storage-buffer range receiving packed pixels.
    pub fn range(&self) -> (u64, u64) {
        (self.0.offset, self.0.size)
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining all native objects referenced by this binding.
    pub fn lease(&self) -> TexturePackBindingsLease {
        TexturePackBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeTexturePackBindings {
        &self.0._native
    }
}

impl fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputePipeline")
            .field("kernel", &self.0.kernel)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for ComputeBindings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputeBindings")
            .field("range", &self.range())
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for ComputePipelineLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ComputePipelineLease")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}
impl fmt::Debug for ComputeBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ComputeBindingsLease")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}

impl fmt::Debug for RasterPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RasterPipeline")
            .field("kernel", &self.0.kernel)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for RasterPipelineLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RasterPipelineLease")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}
impl fmt::Debug for TexturePackBindings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TexturePackBindings")
            .field("range", &self.range())
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for TexturePackBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TexturePackBindingsLease")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}
