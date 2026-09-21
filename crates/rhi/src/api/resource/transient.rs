//! Frozen transient-resource allocation contract (specification section 50).
//!
//! Transient allocation is an RHI capability, not a render-graph service. A
//! backend may satisfy this contract with a dedicated allocation for every
//! resource and later add aliasing without changing this public vocabulary.
//! `TransientLifetime` supplies the ordering input, while recorded
//! [`crate::api::command::ResourceUse`] supplies the access input; an aliasing
//! backend combines those facts with its own physical compatibility classes to
//! lower native aliasing barriers or heap synchronization. Neither heap/page
//! ownership nor a render-graph scheduling concept is part of this API.

use crate::api::error::RhiResult;
use crate::api::format::TextureSupportQuery;
use crate::api::identity::DeviceIdentity;
use crate::api::platform::Device;
use crate::api::resource::backend::{BufferBackend, TextureBackend};
use crate::api::resource::buffer::{
    Buffer, BufferDescriptor, BufferSupportQuery, validate_buffer_descriptor,
};
use crate::api::resource::texture::{Texture, TextureDescriptor, validate_texture_descriptor};
use crate::api::submission::{PlanPoint, SubmissionPlanId, TransientLifetimeRegistry};
use std::any::Any;

/// How a resource kind is physically allocated by this device.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransientAllocationSupport {
    /// Every transient resource receives independent backing. This is the
    /// required correctness-complete implementation for every backend.
    Dedicated,
    /// Non-overlapping lifetimes may reuse physical memory.
    ///
    /// A backend reports this only after it actually lowers the required alias
    /// synchronization. It remains free to use dedicated backing for an
    /// individual incompatible allocation.
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
///
/// This is deliberately execution vocabulary rather than an allocator-specific
/// interval format. An aliasing implementation may derive placement and
/// synchronization from this lifetime, plan ordering, and actual `ResourceUse`s;
/// the caller never supplies native heap offsets, alias pairs, or barriers.
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
pub(crate) struct DeferredTransientBuffer;

impl BufferBackend for DeferredTransientBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Placeholder backend object for a texture whose dedicated backing will be
/// materialized by submission lowering. It keeps the public handle's ownership
/// invariant intact; unlike the removed `Option`, it is always a concrete seam
/// object.
pub(crate) struct DeferredTransientTexture;

impl TextureBackend for DeferredTransientTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Whether a buffer still carries the crate-local deferred fixture backing.
///
/// A public plan builder always creates transient resources through a live
/// [`Device`], so a submitted plan must never contain this marker.  It remains
/// useful for the fact-only builder used by DAG tests: those tests deliberately
/// have no native device from which an allocation could be requested.
pub(crate) fn is_deferred_buffer(buffer: &Buffer) -> bool {
    buffer.native().as_any().is::<DeferredTransientBuffer>()
}

/// Whether a texture still carries the crate-local deferred fixture backing.
pub(crate) fn is_deferred_texture(texture: &Texture) -> bool {
    texture.native().as_any().is::<DeferredTransientTexture>()
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
pub struct TransientAllocator<'plan> {
    device: DeviceIdentity,
    plan: SubmissionPlanId,
    registry: &'plan TransientLifetimeRegistry,
    /// Present for builders opened from a real device.  The fact-only builder
    /// used by contract tests intentionally has no backend to allocate from.
    device_handle: Option<&'plan Device>,
}

impl<'plan> TransientAllocator<'plan> {
    /// Assembles an allocator sharing its plan builder's lifetime registry.
    pub(crate) fn new_with_registry(
        device: DeviceIdentity,
        plan: SubmissionPlanId,
        registry: &'plan TransientLifetimeRegistry,
        device_handle: Option<&'plan Device>,
    ) -> Self {
        Self {
            device,
            plan,
            registry,
            device_handle,
        }
    }

    /// The device that owns resources created by this allocator.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The submission plan to which created resources are bound.
    pub fn plan_id(&self) -> SubmissionPlanId {
        self.plan
    }

    /// Creates a transient buffer with native backing appropriate to this
    /// device's advertised transient capability.
    ///
    /// Dedicated is the frozen baseline: its allocation happens here so the
    /// returned handle is immediately a normal native-backed `Buffer`.  A future
    /// aliasing implementation may defer *placement* until submit, but may not
    /// expose a public handle without a valid backend object.
    pub fn create_buffer(
        &self,
        desc: &BufferDescriptor,
        lifetime: TransientLifetime,
    ) -> RhiResult<Buffer> {
        self.validate_lifetime(&lifetime)?;
        validate_transient_buffer_descriptor(desc)?;
        let native = if let Some(device) = self.device_handle {
            device
                .require_active()
                .map_err(|error| error.at("TransientAllocator::create_buffer"))?;
            let support = device
                .capabilities()
                .buffer_support(&BufferSupportQuery::new(desc.usage));
            validate_buffer_descriptor(desc, &support)
                .map_err(|error| error.at("TransientAllocator::create_buffer"))?;
            device.native().create_buffer(desc)?
        } else {
            // `SubmissionPlanBuilder::with_facts` has no native device.  This
            // path is solely a crate-local plan-validation fixture; submit
            // rejects it before reaching a backend.
            Box::new(DeferredTransientBuffer)
        };
        let buffer = Buffer::new_transient(
            crate::api::identity::ObjectId::next(),
            self.device,
            desc.clone(),
            native,
            TransientResourceMetadata::new(lifetime.clone()),
        );
        self.registry.record(lifetime);
        Ok(buffer)
    }

    /// Creates a transient texture with native backing appropriate to this
    /// device's advertised transient capability.
    ///
    /// See [`Self::create_buffer`] for the dedicated-baseline and future aliasing
    /// placement rule; texture allocation follows the same public contract.
    pub fn create_texture(
        &self,
        desc: &TextureDescriptor,
        lifetime: TransientLifetime,
    ) -> RhiResult<Texture> {
        self.validate_lifetime(&lifetime)?;
        validate_transient_texture_descriptor(desc)?;
        let (descriptor, native) = if let Some(device) = self.device_handle {
            device
                .require_active()
                .map_err(|error| error.at("TransientAllocator::create_texture"))?;
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
            let support = device.capabilities().texture_support(&query);
            validate_texture_descriptor(&mut accepted, &support)
                .map_err(|error| error.at("TransientAllocator::create_texture"))?;
            let native = device.native().create_texture(&accepted)?;
            (accepted, native)
        } else {
            (
                desc.clone(),
                Box::new(DeferredTransientTexture) as Box<dyn TextureBackend>,
            )
        };
        let texture = Texture::new_transient(
            crate::api::identity::ObjectId::next(),
            self.device,
            descriptor,
            native,
            TransientResourceMetadata::new(lifetime.clone()),
        );
        self.registry.record(lifetime);
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
