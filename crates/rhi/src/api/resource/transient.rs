//! Frozen transient-resource allocation contract (specification section 50).
//!
//! Transient allocation is an RHI capability, not a render-graph service. A
//! backend may satisfy this contract with a dedicated allocation for every
//! resource and later add aliasing without changing this public vocabulary.

use crate::api::error::RhiResult;
use crate::api::format::TextureSupportQuery;
use crate::api::identity::DeviceIdentity;
use crate::api::platform::Device;
use crate::api::resource::backend::BufferBackend;
use crate::api::resource::buffer::{
    Buffer, BufferDescriptor, BufferSupportQuery, validate_buffer_descriptor,
};
use crate::api::resource::texture::{Texture, TextureDescriptor, validate_texture_descriptor};
use crate::api::submission::{PlanPoint, SubmissionPlanId};
use std::any::Any;
use std::sync::{Arc, Mutex};

/// How a resource kind is physically allocated by this device.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransientAllocationSupport {
    /// Every transient resource receives independent backing. This is the
    /// required correctness-complete implementation for every backend.
    Dedicated,
    /// Non-overlapping lifetimes may reuse physical memory.
    Aliasing,
}

/// The device's transient allocation capabilities.
#[derive(Clone, Copy, Debug)]
pub struct TransientCapabilities {
    /// Allocation support for buffers.
    pub buffers: TransientAllocationSupport,
    /// Allocation support for textures.
    pub textures: TransientAllocationSupport,
    /// Whether buffers and textures may occupy one physical alias pool/class.
    pub mixed_resource_aliasing: bool,
}

/// One resource descriptor accepted by the transient requirements query.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum TransientResourceDescriptor {
    /// A transient buffer.
    Buffer(BufferDescriptor),
    /// A transient texture.
    Texture(TextureDescriptor),
}

/// Opaque backend-specific class used for physical alias compatibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransientCompatibilityClass(u64);

/// Allocation information a backend can expose without allocating.
#[derive(Clone, Copy, Debug)]
pub struct TransientAllocationRequirements {
    /// Descriptor-based logical byte estimate, when the format defines one.
    pub logical_size: Option<u64>,
    /// Physical allocation size, when this backend exposes a stable value.
    pub physical_size: Option<u64>,
    /// Physical allocation alignment, when exposed by this backend.
    pub alignment: Option<u64>,
    /// Opaque physical aliasing class, when exposed by this backend.
    pub class: Option<TransientCompatibilityClass>,
}

/// Implementation-observable transient backing statistics.
///
/// These counters are neither total VRAM nor a residency report. They describe
/// only transient backing the current backend implementation can observe.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct TransientMemoryStatistics {
    /// Sum of logical descriptor byte estimates for realized resources.
    pub logical_bytes: u64,
    /// Physical transient backing bytes currently observable by this backend.
    pub physical_backing_bytes: u64,
    /// Number of transient resources physically realized.
    pub resources_realized: u64,
    /// Number of physical allocation reuses caused by aliasing.
    pub alias_reuses: u64,
}

/// The plan-point interval in which a transient resource may be used.
#[derive(Clone, Debug)]
pub struct TransientLifetime {
    acquire: PlanPoint,
    release_frontier: Vec<PlanPoint>,
}

/// Crate-private execution metadata carried by ordinary transient handles.
#[derive(Clone, Debug)]
pub(crate) struct TransientResourceMetadata {
    lifetime: TransientLifetime,
}

/// Shared builder/allocator registry of all transient lifetimes in one plan.
///
/// Submission owns validation of this registry because only it owns the plan
/// DAG; resource allocation merely makes every allocated lifetime visible.
pub(crate) type TransientLifetimeRegistry = Arc<Mutex<Vec<TransientLifetime>>>;

impl TransientResourceMetadata {
    fn new(lifetime: TransientLifetime) -> Self {
        Self { lifetime }
    }

    /// The resource's immutable execution lifetime.
    pub(crate) fn lifetime(&self) -> &TransientLifetime {
        &self.lifetime
    }
}

/// Placeholder backing for a logical resource awaiting submission realization.
struct DeferredTransientBuffer;

impl BufferBackend for DeferredTransientBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl TransientLifetime {
    /// Starts a lifetime at `acquire`.
    pub fn new(acquire: PlanPoint) -> Self {
        Self {
            acquire,
            release_frontier: Vec::new(),
        }
    }

    /// Adds a terminal release frontier point.
    pub fn release_at(mut self, point: PlanPoint) -> Self {
        self.release_frontier.push(point);
        self
    }

    /// The point at or after which the resource may be used.
    pub fn acquire(&self) -> PlanPoint {
        self.acquire
    }

    /// The terminal points reachable from every permitted resource use.
    pub fn release_frontier(&self) -> &[PlanPoint] {
        &self.release_frontier
    }
}

/// A handle which creates transient resources for one submission plan.
#[derive(Clone)]
pub struct TransientAllocator {
    device: DeviceIdentity,
    plan: SubmissionPlanId,
    registry: TransientLifetimeRegistry,
}

impl TransientAllocator {
    /// Assembles the plan-scoped allocator returned by the submission builder.
    pub(crate) fn new(
        device: DeviceIdentity,
        plan: SubmissionPlanId,
        registry: TransientLifetimeRegistry,
    ) -> Self {
        Self {
            device,
            plan,
            registry,
        }
    }

    /// Assembles an allocator sharing its plan builder's lifetime registry.
    pub(crate) fn new_with_registry(
        device: DeviceIdentity,
        plan: SubmissionPlanId,
        registry: TransientLifetimeRegistry,
    ) -> Self {
        Self::new(device, plan, registry)
    }

    /// The device that owns resources created by this allocator.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The submission plan to which created resources are bound.
    pub fn plan_id(&self) -> SubmissionPlanId {
        self.plan
    }

    /// Creates a logical transient buffer. Physical realization may be deferred
    /// until submission lowering.
    pub fn create_buffer(
        &self,
        desc: &BufferDescriptor,
        lifetime: TransientLifetime,
    ) -> RhiResult<Buffer> {
        self.validate_lifetime(&lifetime)?;
        validate_transient_buffer_descriptor(desc)?;
        let buffer = Buffer::new_transient(
            crate::api::identity::ObjectId::next(),
            self.device,
            desc.clone(),
            Arc::new(DeferredTransientBuffer),
            TransientResourceMetadata::new(lifetime.clone()),
        );
        self.registry
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(lifetime);
        Ok(buffer)
    }

    /// Creates a logical transient texture. Physical realization may be deferred
    /// until submission lowering.
    pub fn create_texture(
        &self,
        desc: &TextureDescriptor,
        lifetime: TransientLifetime,
    ) -> RhiResult<Texture> {
        self.validate_lifetime(&lifetime)?;
        validate_transient_texture_descriptor(desc)?;
        let texture = Texture::new_transient(
            crate::api::identity::ObjectId::next(),
            self.device,
            desc.clone(),
            TransientResourceMetadata::new(lifetime.clone()),
        );
        self.registry
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(lifetime);
        Ok(texture)
    }

    fn validate_lifetime(&self, lifetime: &TransientLifetime) -> RhiResult<()> {
        use crate::api::error::{RhiError, RhiErrorKind};
        if lifetime.release_frontier().is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a transient lifetime must have at least one release frontier point",
            ));
        }
        if lifetime.acquire().plan() != self.plan
            || lifetime
                .release_frontier()
                .iter()
                .any(|point| point.plan() != self.plan)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "every transient lifetime point must belong to the allocator's submission plan",
            ));
        }
        Ok(())
    }
}

fn validate_transient_buffer_descriptor(desc: &BufferDescriptor) -> RhiResult<()> {
    use crate::api::error::{RhiError, RhiErrorKind};
    if desc.size == 0 || desc.usage.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a transient buffer descriptor must have non-zero size and non-empty usage",
        ));
    }
    Ok(())
}

fn validate_transient_texture_descriptor(desc: &TextureDescriptor) -> RhiResult<()> {
    use crate::api::error::{RhiError, RhiErrorKind};
    if desc.extent.width == 0
        || desc.extent.height == 0
        || desc.extent.depth == 0
        || desc.mip_levels == 0
        || desc.array_layers == 0
        || desc.sample_count == 0
        || desc.usage.is_empty()
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a transient texture descriptor has an invalid zero extent, count, or usage",
        ));
    }
    Ok(())
}

impl Device {
    /// Returns allocation requirements for a transient descriptor without allocating it.
    pub fn transient_requirements(
        &self,
        desc: &TransientResourceDescriptor,
    ) -> RhiResult<TransientAllocationRequirements> {
        self.require_active()?;
        match desc {
            TransientResourceDescriptor::Buffer(desc) => {
                let support = self
                    .capabilities()
                    .buffer_support(&BufferSupportQuery::new(desc.usage));
                validate_buffer_descriptor(desc, &support)?;
            }
            TransientResourceDescriptor::Texture(desc) => {
                let mut accepted = desc.clone();
                let mut query = TextureSupportQuery::new(
                    accepted.dimension,
                    accepted.format,
                    accepted.usage,
                    accepted.sample_count,
                )
                .with_view_compatibility(accepted.view_compatibility);
                for format in &accepted.view_formats {
                    query = query.with_view_format(*format);
                }
                let support = self.capabilities().texture_support(&query);
                validate_texture_descriptor(&mut accepted, &support)?;
            }
        }
        Ok(TransientAllocationRequirements {
            logical_size: match desc {
                TransientResourceDescriptor::Buffer(desc) => Some(desc.size),
                // Backend-defined layouts, including Depth24Plus, need not expose
                // a descriptor-only byte count.
                TransientResourceDescriptor::Texture(_) => None,
            },
            physical_size: None,
            alignment: None,
            class: None,
        })
    }
}
