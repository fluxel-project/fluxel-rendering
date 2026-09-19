//! The mock's shader modules, bind groups, interfaces, and pipelines.
//!
//! These backends store the canonical value they were handed and mint an
//! object identity. What they do *not* do is re-decide portable legality,
//! because the `Device` façade already did: the one exception is the raster,
//! compute, and pipeline-interface descriptors, which the façade delegates
//! unvalidated and the mock refuses to accept unvalidated.
//!
//! Interning compares canonical values, never a hash. Two layout descriptors
//! that canonicalize to the same value intern to one compatibility id, and a
//! fingerprint is derived from that id without ever being compared.

use crate::rhi::binding::{
    BindGroupBackend, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutBackend,
    BindGroupLayoutCompatibilityId, BindGroupLayoutDescriptor, LayoutFingerprint,
    PipelineInterfaceCompatibilityId,
};
use crate::rhi::pipeline::{
    ComputePipelineBackend, ComputePipelineDescriptor, PipelineInterfaceBackend, PipelineInterfaceDescriptor, RasterPipelineBackend,
    RasterPipelineDescriptor, RenderTargetSignature, ResourceMergeOutcome,
};
use crate::rhi::platform::{DeviceIdentity, ObjectId};
use crate::rhi::shader::{ShaderArtifact, ShaderModuleBackend};

use super::platform::fingerprint_of;

/// The backend half of the mock's shader modules.
pub(super) struct MockShaderModule {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) artifact: ShaderArtifact,
}

impl ShaderModuleBackend for MockShaderModule {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn artifact(&self) -> &ShaderArtifact {
        &self.artifact
    }
}

/// The backend half of the mock's bind group layouts.
pub(super) struct MockBindGroupLayout {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) descriptor: BindGroupLayoutDescriptor,
    pub(super) compatibility: BindGroupLayoutCompatibilityId,
}

impl BindGroupLayoutBackend for MockBindGroupLayout {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &BindGroupLayoutDescriptor {
        &self.descriptor
    }

    fn compatibility_id(&self) -> BindGroupLayoutCompatibilityId {
        self.compatibility
    }

    fn fingerprint(&self) -> LayoutFingerprint {
        fingerprint_of(self.compatibility.as_u64())
    }

    fn dynamic_offset_count(&self) -> u32 {
        // The frozen order is ascending slot id and, within one slot, ascending
        // element index. A canonicalized descriptor is already in slot order,
        // so summing in that order produces the count the lowering contract
        // names.
        self.descriptor
            .entries
            .iter()
            .filter(|entry| entry.dynamic_offset)
            .map(|entry| entry.count.elements())
            .sum()
    }
}

/// The backend half of the mock's bind groups.
pub(super) struct MockBindGroup {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) layout: BindGroupLayout,
    pub(super) descriptor: BindGroupDescriptor,
}

impl BindGroupBackend for MockBindGroup {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }

    fn descriptor(&self) -> &BindGroupDescriptor {
        &self.descriptor
    }
}

/// The backend half of the mock's pipeline interfaces.
pub(super) struct MockPipelineInterface {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) descriptor: PipelineInterfaceDescriptor,
    pub(super) compatibility: PipelineInterfaceCompatibilityId,
}

impl PipelineInterfaceBackend for MockPipelineInterface {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &PipelineInterfaceDescriptor {
        &self.descriptor
    }

    fn compatibility_id(&self) -> PipelineInterfaceCompatibilityId {
        self.compatibility
    }

    fn fingerprint(&self) -> LayoutFingerprint {
        fingerprint_of(self.compatibility.as_u64())
    }
}

/// The backend half of the mock's raster pipelines.
///
/// The used-binding list is not empty by accident: it is what the real merge
/// produced from the pipeline's shader interfaces, which is the only source of
/// "this draw reads three of the ten bound resources".
pub(super) struct MockRasterPipeline {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) descriptor: RasterPipelineDescriptor,
    pub(super) target_signature: RenderTargetSignature,
    pub(super) used_bindings: Vec<ResourceMergeOutcome>,
}

impl RasterPipelineBackend for MockRasterPipeline {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &RasterPipelineDescriptor {
        &self.descriptor
    }

    fn target_signature(&self) -> &RenderTargetSignature {
        &self.target_signature
    }

    fn used_bindings(&self) -> &[ResourceMergeOutcome] {
        &self.used_bindings
    }
}

/// The backend half of the mock's compute pipelines.
pub(super) struct MockComputePipeline {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) descriptor: ComputePipelineDescriptor,
    pub(super) used_bindings: Vec<ResourceMergeOutcome>,
}

impl ComputePipelineBackend for MockComputePipeline {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &ComputePipelineDescriptor {
        &self.descriptor
    }

    fn used_bindings(&self) -> &[ResourceMergeOutcome] {
        &self.used_bindings
    }
}
