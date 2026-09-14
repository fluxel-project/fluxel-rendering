//! Generation-safe identities for objects owned by a GL-family context.

use core::marker::PhantomData;
use core::num::NonZeroU64;

/// Identity of the thread which owns a GL-family context.
///
/// This is a Rust runtime identity, never a platform thread handle.  Keeping
/// it opaque lets every provider enforce affinity before crossing its driver
/// boundary on native and browser targets alike.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OwnerThreadIdentity(std::thread::ThreadId);

impl OwnerThreadIdentity {
    /// Captures the calling thread.
    pub fn current() -> Self {
        Self(std::thread::current().id())
    }
}

/// A stable identity for the Fluxel device which owns a context.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceIdentity(NonZeroU64);

impl DeviceIdentity {
    /// Creates a nonzero device identity from allocator-owned metadata.
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Returns the allocator-owned numeric identity.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// A monotonically increasing context recreation generation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContextEpoch(NonZeroU64);

impl ContextEpoch {
    /// The first live context generation.
    pub const INITIAL: Self = Self(NonZeroU64::MIN);

    /// Advances the epoch, returning `None` rather than wrapping.
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.get().checked_add(1) {
            Some(value) => match NonZeroU64::new(value) {
                Some(value) => Some(Self(value)),
                None => None,
            },
            None => None,
        }
    }

    /// Returns the monotonically allocated generation number.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// The context provenance carried by all object and cache identities.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContextStamp {
    /// Device which owns the context.
    pub device: DeviceIdentity,
    /// Generation of that context on the device.
    pub epoch: ContextEpoch,
}

impl ContextStamp {
    /// Creates a context stamp.
    pub const fn new(device: DeviceIdentity, epoch: ContextEpoch) -> Self {
        Self { device, epoch }
    }
}

/// Marks a specific kind of GL object.
pub trait GlObjectKind: 'static {}

/// Buffer-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BufferObject;
impl GlObjectKind for BufferObject {}

/// Texture-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TextureObject;
impl GlObjectKind for TextureObject {}

/// Program-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProgramObject;
impl GlObjectKind for ProgramObject {}

/// Query-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QueryObject;
impl GlObjectKind for QueryObject {}

/// Sampler-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SamplerObject;
impl GlObjectKind for SamplerObject {}

/// Shader-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShaderObject;
impl GlObjectKind for ShaderObject {}

/// Vertex-array-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VertexArrayObject;
impl GlObjectKind for VertexArrayObject {}

/// Framebuffer-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FramebufferObject;
impl GlObjectKind for FramebufferObject {}

/// Renderbuffer-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RenderbufferObject;
impl GlObjectKind for RenderbufferObject {}

/// Sync-object marker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SyncObject;
impl GlObjectKind for SyncObject {}

/// Surface-image marker.
///
/// A surface image is a Fluxel presentation allocation, not a native GL name.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SurfaceImageObject;
impl GlObjectKind for SurfaceImageObject {}

/// A typed, structural GL object identity.
///
/// `slot` and `generation` are Fluxel allocation-table values. Native integer
/// names and browser handles deliberately do not participate in this identity.
#[derive(Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ObjectIdentity<K: GlObjectKind> {
    /// Owning context and its generation.
    pub context: ContextStamp,
    /// Stable allocation-table slot.
    pub slot: u32,
    /// Reuse generation for this allocation-table slot.
    pub generation: u32,
    marker: PhantomData<K>,
}

impl<K: GlObjectKind> Copy for ObjectIdentity<K> {}

impl<K: GlObjectKind> Clone for ObjectIdentity<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: GlObjectKind> ObjectIdentity<K> {
    /// Creates a typed identity from Fluxel-owned allocation metadata.
    pub(super) const fn new(context: ContextStamp, slot: u32, generation: u32) -> Self {
        Self {
            context,
            slot,
            generation,
            marker: PhantomData,
        }
    }
}

/// Typed buffer identity.
pub type BufferId = ObjectIdentity<BufferObject>;
/// Typed texture identity.
pub type TextureId = ObjectIdentity<TextureObject>;
/// Typed program identity.
pub type ProgramId = ObjectIdentity<ProgramObject>;
/// Typed query identity.
pub type QueryId = ObjectIdentity<QueryObject>;
/// Typed sampler identity.
pub type SamplerId = ObjectIdentity<SamplerObject>;
/// Typed shader identity.
pub type ShaderId = ObjectIdentity<ShaderObject>;
/// Typed vertex-array identity.
pub type VertexArrayId = ObjectIdentity<VertexArrayObject>;
/// Typed framebuffer identity.
pub type FramebufferId = ObjectIdentity<FramebufferObject>;
/// Typed renderbuffer identity.
pub type RenderbufferId = ObjectIdentity<RenderbufferObject>;
/// Typed sync identity.
pub type SyncId = ObjectIdentity<SyncObject>;
/// Typed presentation surface-image identity.
pub type SurfaceImageId = ObjectIdentity<SurfaceImageObject>;

#[cfg(test)]
mod tests {
    use core::any::TypeId;

    use super::{
        BufferId, ContextEpoch, ContextStamp, DeviceIdentity, FramebufferId, ObjectIdentity,
        SamplerId, ShaderId, SurfaceImageId, SyncId, TextureId, VertexArrayId,
    };

    fn stamp() -> ContextStamp {
        ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
    }

    #[test]
    fn object_kinds_have_distinct_static_identity_types() {
        assert_ne!(TypeId::of::<BufferId>(), TypeId::of::<TextureId>());
        assert_ne!(TypeId::of::<SamplerId>(), TypeId::of::<ShaderId>());
        assert_ne!(TypeId::of::<VertexArrayId>(), TypeId::of::<FramebufferId>());
        assert_ne!(TypeId::of::<SyncId>(), TypeId::of::<SurfaceImageId>());
    }

    #[test]
    fn typed_identities_keep_only_fluxel_owned_metadata() {
        let context = stamp();
        let sampler = SamplerId::new(context, 7, 3);

        assert_eq!(sampler.context, context);
        assert_eq!(sampler.slot, 7);
        assert_eq!(sampler.generation, 3);

        fn require_surface_image(_: SurfaceImageId) {}
        require_surface_image(ObjectIdentity::new(context, 2, 1));
    }
}
