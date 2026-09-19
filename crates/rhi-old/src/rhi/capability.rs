//! Device capability facts and their exact identity.
//!
//! This module owns rhi-design section 7 (available vs enabled capabilities,
//! their identity, and the device limit map) and section 8.5 (texture view
//! format compatibility).
//!
//! # What it is
//!
//! One immutable fact database per adapter and one per device, both speaking the
//! same vocabulary: features, limits, format facts, buffer/texture/binding
//! support, per-stage binding counts, routes, and submission lanes. A fact has
//! exactly one canonical home. There is no second boolean field, no
//! `ResourceCapabilities { supports_texture_1d }`, and no trait per domain that
//! could disagree with this database.
//!
//! # Fail-closed by construction
//!
//! Every question whose answer the backend did not explicitly declare answers
//! "no": an unproved route is [`RouteSupport::Unsupported`], an undeclared
//! format is `None`, an undeclared binding shape is
//! [`BindingSupport::Unsupported`]. Forgetting a declaration therefore produces
//! a portable refusal, never an optimistic `true` that reaches a driver.
//!
//! # Two kinds of digest, deliberately not interchangeable
//!
//! [`CapabilityCompatibilityId`] is exact: values are interned process-wide and
//! compared structurally, so two equal ids mean equal facts with no
//! hash-collision risk, and a re-created device with identical facts gets the
//! same id again. [`CapabilityFingerprint`] is a hash and is only ever a cache
//! key, a log token, or provenance.
//!
//! # What it deliberately does not own
//!
//! Presentation capability is not here. Whether a surface can be acquired,
//! configured, or presented is a
//! [`SurfaceFact`](super::presentation::PresentationTargetCapabilities) of a
//! `(Device, PresentationTarget)` pair, because the same device answers
//! differently for two windows.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock};

use super::binding::{
    BindingCount, BindingKind, BindingLimitClass, BindingSupport, BindingSupportQuery,
    BufferBindingAccess, SamplerKind, StorageAccess, TextureSampleType,
};
use super::format::{
    BlitFilter, BufferSupport, BufferSupportLimits, BufferSupportQuery, FormatFacts, LaneDependencyRoute,
    LaneWorkDomains, RouteCapabilities, RouteQuery, RouteSupport, StorageAccessSupport,
    SubmissionCapabilities, SubmissionLaneClass, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery, TextureViewCompatibility, format_facts,
};
use super::hash::CanonicalHasher;
use super::platform::{BackendKind, LimitKey, OptionalFeature, RhiError, RhiErrorKind, RhiResult};
use super::resource::{
    BufferUsage, Extent3d, TextureAspect, TextureDimension, TextureUsage, TextureViewDimension,
};
use super::shader::{
    ArtifactAcceptance, ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderNumericType, ShaderStage,
    ShaderStages,
};

/// The process-local exact capability-contract intern token.
///
/// Produced by RHI through process-wide interning of canonical
/// [`EnabledCapabilities`] semantics. A caller cannot construct it, and it never
/// comes from a hash: correctness that depends on "the same capability contract"
/// compares this value and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapabilityCompatibilityId(u64);

impl CapabilityCompatibilityId {
    /// The underlying intern serial, for diagnostics and logs.
    ///
    /// The value is only meaningful within one process; persistence belongs to
    /// [`CapabilityFingerprint`].
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A cache, diagnostics, and capture-provenance fingerprint.
///
/// Equal hashes cannot alone carry correctness. Reuse of a compiled pipeline or
/// graph must be keyed on [`CapabilityCompatibilityId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapabilityFingerprint(pub [u8; 32]);

/// An immutable device limit map.
///
/// Larger is stronger for `Max*` keys and smaller is stronger for `Min*`
/// alignment keys, which is why requirements are expressed as
/// [`LimitRequirement`](super::platform::LimitRequirement) rather than a single
/// "minimum limit". A key the device did not report is absent, not zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    values: Arc<BTreeMap<LimitKey, u64>>,
}

impl DeviceLimits {
    /// The value of `key`, when the device reports it.
    pub fn get(&self, key: LimitKey) -> Option<u64> {
        self.values.get(&key).copied()
    }

    /// How many limits the device reports.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the device reports no limits at all, which no real device does.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The complete fact database behind both capability views.
///
/// Equality is exact structural equality over canonical containers, which is
/// what makes interning a comparison rather than a hash lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CapabilityData {
    backend: BackendKind,
    abi: ShaderAbiVersion,
    code_formats: BTreeSet<u8>,
    limits: DeviceLimits,
    features: BTreeSet<OptionalFeature>,
    formats: BTreeMap<TextureFormat, FormatFacts>,
    view_formats: BTreeMap<TextureFormat, BTreeSet<TextureFormat>>,
    buffers: BTreeMap<BufferUsage, BufferSupport>,
    textures: BTreeMap<TextureSupportQuery, TextureSupport>,
    bindings: BTreeMap<BindingSupportQuery, BindingSupport>,
    binding_limits: BTreeMap<(ShaderStage, BindingLimitClass), u32>,
    routes: BTreeMap<RouteQuery, RouteSupport>,
    submission: SubmissionCapabilities,
}

impl CapabilityData {
    /// This contract's backend.
    pub(crate) fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Whether the backend declared this optional feature.
    fn has_feature(&self, feature: OptionalFeature) -> bool {
        self.features.contains(&feature)
    }

    /// The declared value of `key`.
    fn limit(&self, key: LimitKey) -> Option<u64> {
        self.limits.get(key)
    }

    /// The declared facts of `format`.
    fn format(&self, format: TextureFormat) -> Option<FormatFacts> {
        self.formats.get(&format).copied()
    }

    /// The declared verdict on a buffer usage set.
    fn buffer_support(&self, query: &BufferSupportQuery) -> BufferSupport {
        self.buffers
            .get(&query.usage())
            .copied()
            .unwrap_or(BufferSupport::Unsupported)
    }

    /// The declared verdict on a texture shape.
    fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        self.textures
            .get(query)
            .copied()
            .unwrap_or(TextureSupport::Unsupported)
    }

    /// The declared verdict on a binding shape.
    fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        self.bindings
            .get(query)
            .copied()
            .unwrap_or(BindingSupport::Unsupported)
    }

    /// The declared per-stage binding count.
    fn binding_limit(&self, stage: ShaderStage, class: BindingLimitClass) -> Option<u32> {
        self.binding_limits.get(&(stage, class)).copied()
    }

    /// The declared verdict on a route.
    fn route(&self, query: &RouteQuery) -> RouteSupport {
        self.routes
            .get(query)
            .copied()
            .unwrap_or(RouteSupport::Unsupported)
    }

    /// Whether `view_format` may be the view format of a `base_format` texture.
    fn texture_view_format_compatible(
        &self,
        base_format: TextureFormat,
        view_format: TextureFormat,
    ) -> bool {
        self.view_formats
            .get(&base_format)
            .is_some_and(|views| views.contains(&view_format))
    }

    /// Whether this device can consume `artifact`'s code form and interface.
    fn shader_acceptance(&self, artifact: &ShaderArtifact) -> ArtifactAcceptance {
        if artifact.code.backend_affinity() != self.backend {
            return ArtifactAcceptance::UnsupportedCodeFormat;
        }
        if !self.code_formats.contains(&code_format_code(&artifact.code)) {
            return ArtifactAcceptance::UnsupportedCodeFormat;
        }
        if !self.abi.accepts(artifact.abi_version) {
            return ArtifactAcceptance::UnsupportedAbi;
        }
        if artifact.stage == ShaderStage::Compute && !self.has_feature(OptionalFeature::Compute) {
            return ArtifactAcceptance::MissingFeature;
        }
        if artifact
            .requirements
            .required_features()
            .iter()
            .any(|feature| !self.has_feature(*feature))
        {
            return ArtifactAcceptance::MissingFeature;
        }
        if !artifact
            .requirements
            .limit_satisfied_by(|key| self.limit(key))
        {
            return ArtifactAcceptance::LimitExceeded;
        }
        if let Some(workgroup) = artifact.requirements.compute_workgroup() {
            let within = |key, value: u64| self.limit(key).is_some_and(|actual| actual >= value);
            if !within(LimitKey::MaxComputeWorkgroupSizeX, u64::from(workgroup.x))
                || !within(LimitKey::MaxComputeWorkgroupSizeY, u64::from(workgroup.y))
                || !within(LimitKey::MaxComputeWorkgroupSizeZ, u64::from(workgroup.z))
                || !within(
                    LimitKey::MaxComputeInvocationsPerWorkgroup,
                    u64::from(workgroup.total_invocations),
                )
                || !within(
                    LimitKey::MaxComputeWorkgroupStorageSize,
                    workgroup.workgroup_storage_bytes,
                )
            {
                return ArtifactAcceptance::LimitExceeded;
            }
        }
        let visibility = ShaderStages::from_stage(artifact.stage);
        for resource in artifact.interface.resources() {
            let query = BindingSupportQuery::new(visibility, resource.kind.clone())
                .with_count(resource.count);
            if !self.binding_support(&query).is_supported() {
                return ArtifactAcceptance::InterfaceUnsupported;
            }
        }
        ArtifactAcceptance::Accepted
    }

    /// The canonical byte encoding used for fingerprints and provenance.
    fn canonical_hash(&self) -> [u8; 32] {
        let mut hasher = CanonicalHasher::new();
        hasher
            .tag(0x01)
            .u8(self.backend.canonical_code())
            .tag(0x02)
            .u16(self.abi.major)
            .u16(self.abi.minor)
            .tag(0x03);
        encode_set(&mut hasher, &self.code_formats);
        hasher.tag(0x04);
        encode_map(&mut hasher, &self.limits.values);
        hasher.tag(0x05);
        encode_set(&mut hasher, &self.features);
        hasher.tag(0x06);
        encode_map(&mut hasher, &self.formats);
        hasher.tag(0x07);
        encode_map(&mut hasher, &self.view_formats);
        hasher.tag(0x08);
        encode_map(&mut hasher, &self.buffers);
        hasher.tag(0x09);
        encode_map(&mut hasher, &self.textures);
        hasher.tag(0x0a);
        encode_map(&mut hasher, &self.bindings);
        hasher.tag(0x0b);
        encode_map(&mut hasher, &self.binding_limits);
        hasher.tag(0x0c);
        encode_map(&mut hasher, &self.routes);
        hasher.tag(0x0d);
        encode_submission(&mut hasher, &self.submission);
        hasher.finish()
    }
}

/// Builds a [`CapabilityData`] from explicit backend declarations.
///
/// Backends declare facts and nothing else. There is deliberately no
/// `set_all_limits` shortcut and no "assume supported" default: every
/// declaration is a claim the backend is willing to have validated on a real
/// device.
#[derive(Clone, Debug)]
pub(crate) struct CapabilityDataBuilder {
    data: CapabilityData,
}

impl CapabilityDataBuilder {
    /// A database for `backend` that speaks `abi` and exposes `submission`.
    pub(crate) fn new(
        backend: BackendKind,
        abi: ShaderAbiVersion,
        submission: SubmissionCapabilities,
    ) -> Self {
        Self {
            data: CapabilityData {
                backend,
                abi,
                code_formats: BTreeSet::new(),
                limits: DeviceLimits {
                    values: Arc::new(BTreeMap::new()),
                },
                features: BTreeSet::new(),
                formats: BTreeMap::new(),
                view_formats: BTreeMap::new(),
                buffers: BTreeMap::new(),
                textures: BTreeMap::new(),
                bindings: BTreeMap::new(),
                binding_limits: BTreeMap::new(),
                routes: BTreeMap::new(),
                submission,
            },
        }
    }

    /// Declares that the enabled contract includes `feature`.
    pub(crate) fn enable_feature(&mut self, feature: OptionalFeature) -> &mut Self {
        self.data.features.insert(feature);
        self
    }

    /// Declares the value of `key`.
    pub(crate) fn set_limit(&mut self, key: LimitKey, value: u64) -> &mut Self {
        Arc::make_mut(&mut self.data.limits.values).insert(key, value);
        self
    }

    /// Declares that `format` is usable, with the facts of [`format_facts`].
    pub(crate) fn declare_format(&mut self, format: TextureFormat) -> &mut Self {
        self.data.formats.insert(format, format_facts(format));
        self
    }

    /// Declares that every format in `group` may be viewed as every other.
    ///
    /// View compatibility is declared as a group because that is how the rule
    /// reads in every backend: a set of formats that share a memory layout.
    /// A base format that was never declared has no compatible views at all.
    pub(crate) fn declare_view_compatibility_group(&mut self, group: &[TextureFormat]) -> &mut Self {
        for base in group {
            let entry = self.data.view_formats.entry(*base).or_default();
            for view in group {
                entry.insert(*view);
            }
        }
        self
    }

    /// Declares that `usage` is creatable within `limits`.
    pub(crate) fn declare_buffer_support(
        &mut self,
        usage: BufferUsage,
        limits: BufferSupportLimits,
    ) -> &mut Self {
        self.data
            .buffers
            .insert(usage, BufferSupport::Supported(limits));
        self
    }

    /// Declares that `query` is creatable within `limits`.
    pub(crate) fn declare_texture_support(
        &mut self,
        query: TextureSupportQuery,
        limits: TextureSupportLimits,
    ) -> &mut Self {
        self.data
            .textures
            .insert(query, TextureSupport::Supported(limits));
        self
    }

    /// Declares that `query` is legal.
    pub(crate) fn declare_binding_support(&mut self, query: BindingSupportQuery) -> &mut Self {
        self.data.bindings.insert(query, BindingSupport::Supported);
        self
    }

    /// Declares the per-stage binding count for `class`.
    pub(crate) fn declare_binding_limit(
        &mut self,
        stage: ShaderStage,
        class: BindingLimitClass,
        count: u32,
    ) -> &mut Self {
        self.data.binding_limits.insert((stage, class), count);
        self
    }

    /// Declares that `query` has a legal direct route with `capabilities`.
    pub(crate) fn declare_route(
        &mut self,
        query: RouteQuery,
        capabilities: RouteCapabilities,
    ) -> &mut Self {
        self.data
            .routes
            .insert(query, RouteSupport::Supported(capabilities));
        self
    }

    /// Declares that this device consumes `code`'s form.
    ///
    /// A backend declares the forms it can actually consume, which may be a
    /// strict subset of what [`ShaderCode::backend_affinity`] maps to it.
    pub(crate) fn declare_code_format(&mut self, code: &ShaderCode) -> &mut Self {
        self.data.code_formats.insert(code_format_code(code));
        self
    }

    /// Finishes the database, rejecting a self-inconsistent lane set.
    pub(crate) fn build(self) -> RhiResult<Arc<CapabilityData>> {
        let compute = self
            .data
            .features
            .contains(&OptionalFeature::Compute);
        if !self.data.submission.satisfies_base_guarantee(compute) {
            return Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "capability database declares no base submission lane",
            ));
        }
        Ok(Arc::new(self.data))
    }
}

/// The facts an adapter or provider reports before a device exists.
///
/// A fact here means "the adapter offers this", not "a device enabled this".
/// A platform where a format requires feature enablement legally answers
/// `Some(..)` here and `None` from [`EnabledCapabilities::format`].
#[derive(Clone, Debug)]
pub struct AvailableCapabilities {
    data: Arc<CapabilityData>,
}

impl AvailableCapabilities {
    /// Wraps a finished database.
    pub(crate) fn new(data: Arc<CapabilityData>) -> Self {
        Self { data }
    }

    /// The backend these facts describe.
    pub fn backend(&self) -> BackendKind {
        self.data.backend()
    }

    /// Whether the adapter offers `feature`.
    pub fn supports_feature(&self, feature: OptionalFeature) -> bool {
        self.data.has_feature(feature)
    }

    /// The adapter's value for `key`.
    pub fn limit(&self, key: LimitKey) -> Option<u64> {
        self.data.limit(key)
    }

    /// The adapter's facts for `format`, when it offers the format at all.
    pub fn format(&self, format: TextureFormat) -> Option<FormatFacts> {
        self.data.format(format)
    }

    /// Whether a buffer usage set is creatable.
    pub fn buffer_support(&self, query: &BufferSupportQuery) -> BufferSupport {
        self.data.buffer_support(query)
    }

    /// Whether a texture shape is creatable.
    pub fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        self.data.texture_support(query)
    }

    /// Whether a binding shape is legal.
    pub fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        self.data.binding_support(query)
    }

    /// The portable binding-count limit for one shader stage and resource class.
    ///
    /// `None` means the stage or resource class is inapplicable to this
    /// capability contract.
    pub fn binding_limit(&self, stage: ShaderStage, class: BindingLimitClass) -> Option<u32> {
        self.data.binding_limit(stage, class)
    }

    /// Whether a portable operation has a legal direct route.
    pub fn route(&self, query: &RouteQuery) -> RouteSupport {
        self.data.route(query)
    }

    /// The adapter's lanes and the relations between them.
    pub fn submission(&self) -> &SubmissionCapabilities {
        &self.data.submission
    }
}

/// The facts a created device actually enabled.
///
/// Correctness always depends on this value rather than on an adapter snapshot:
/// enabling a feature, requesting a limit, or selecting a backend all change
/// what the device may legally do.
#[derive(Clone, Debug)]
pub struct EnabledCapabilities {
    data: Arc<CapabilityData>,
    compatibility: CapabilityCompatibilityId,
    fingerprint: CapabilityFingerprint,
}

impl EnabledCapabilities {
    /// Interns a finished database and fixes its identity.
    pub(crate) fn new(data: Arc<CapabilityData>) -> Self {
        let (compatibility, fingerprint) = intern(&data);
        Self {
            data,
            compatibility,
            fingerprint,
        }
    }

    /// The exact capability-contract identity.
    pub fn compatibility_id(&self) -> CapabilityCompatibilityId {
        self.compatibility
    }

    /// The cache/diagnostics fingerprint of the same contract.
    pub fn fingerprint(&self) -> CapabilityFingerprint {
        self.fingerprint
    }

    /// The backend this device enabled.
    pub fn backend(&self) -> BackendKind {
        self.data.backend()
    }

    /// The lowering ABI this device speaks.
    pub fn shader_abi(&self) -> ShaderAbiVersion {
        self.data.abi
    }

    /// Whether the device enabled `feature`.
    pub fn supports_feature(&self, feature: OptionalFeature) -> bool {
        self.data.has_feature(feature)
    }

    /// The device's value for `key`.
    pub fn limit(&self, key: LimitKey) -> Option<u64> {
        self.data.limit(key)
    }

    /// The device's facts for `format`, when the device enabled it.
    pub fn format(&self, format: TextureFormat) -> Option<FormatFacts> {
        self.data.format(format)
    }

    /// Whether a buffer usage set is creatable.
    pub fn buffer_support(&self, query: &BufferSupportQuery) -> BufferSupport {
        self.data.buffer_support(query)
    }

    /// Whether a texture shape is creatable.
    pub fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        self.data.texture_support(query)
    }

    /// Whether a binding shape is legal.
    pub fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        self.data.binding_support(query)
    }

    /// The portable binding-count limit for one shader stage and resource class.
    ///
    /// `None` means the stage or resource class is inapplicable to this
    /// capability contract.
    pub fn binding_limit(&self, stage: ShaderStage, class: BindingLimitClass) -> Option<u32> {
        self.data.binding_limit(stage, class)
    }

    /// Whether a portable operation has a legal direct route.
    pub fn route(&self, query: &RouteQuery) -> RouteSupport {
        self.data.route(query)
    }

    /// The device's lanes and the relations between them.
    pub fn submission(&self) -> &SubmissionCapabilities {
        &self.data.submission
    }

    /// The device's limit map.
    pub fn limits(&self) -> &DeviceLimits {
        &self.data.limits
    }

    /// Whether `view_format` may be the view format of a `base_format` texture.
    ///
    /// This answers format compatibility only. Whether the texture descriptor
    /// declares that view format, and whether the aspect, dimension, and
    /// subresource range are legal, remain the responsibility of
    /// `create_texture_view` validation. Equal byte size never implies
    /// view compatibility.
    pub fn texture_view_format_compatible(
        &self,
        base_format: TextureFormat,
        view_format: TextureFormat,
    ) -> bool {
        self.data
            .texture_view_format_compatible(base_format, view_format)
    }

    /// The device's verdict on one shader artifact.
    ///
    /// Checking before module creation turns "the driver rejected it" into a
    /// portable, structured refusal.
    pub fn shader_acceptance(&self, artifact: &ShaderArtifact) -> ArtifactAcceptance {
        self.data.shader_acceptance(artifact)
    }
}

/// The process-wide capability interning table.
#[derive(Default)]
struct Interner {
    next: u64,
    by_fingerprint: BTreeMap<[u8; 32], Vec<(u64, Arc<CapabilityData>)>>,
}

static INTERNER: OnceLock<Mutex<Interner>> = OnceLock::new();

/// Interns `data`, returning its exact identity and fingerprint.
///
/// The bucket is keyed by fingerprint for speed but resolving a hit compares the
/// databases themselves, so a hash collision can only cost an extra bucket
/// entry; it can never make two different capability contracts share an id.
fn intern(data: &Arc<CapabilityData>) -> (CapabilityCompatibilityId, CapabilityFingerprint) {
    let fingerprint = CapabilityFingerprint(data.canonical_hash());
    let interner = INTERNER.get_or_init(|| Mutex::new(Interner::default()));
    // A panic while holding this lock must not permanently disable device
    // creation, so poisoning is recovered rather than propagated.
    let mut interner = interner.lock().unwrap_or_else(|error| error.into_inner());
    let Interner {
        next,
        by_fingerprint,
    } = &mut *interner;
    let bucket = by_fingerprint.entry(fingerprint.0).or_default();
    if let Some((id, _)) = bucket
        .iter()
        .find(|(_, existing)| Arc::ptr_eq(existing, data) || **existing == **data)
    {
        return (CapabilityCompatibilityId(*id), fingerprint);
    }
    let id = *next;
    *next = next.wrapping_add(1);
    bucket.push((id, Arc::clone(data)));
    (CapabilityCompatibilityId(id), fingerprint)
}

/// The canonical discriminant of a shader code form.
fn code_format_code(code: &ShaderCode) -> u8 {
    match code {
        ShaderCode::Wgsl(_) => 1,
        ShaderCode::SpirV(_) => 2,
        ShaderCode::Dxil(_) => 3,
        ShaderCode::Msl(_) => 4,
        ShaderCode::Metallib(_) => 5,
        ShaderCode::Glsl { .. } => 6,
        ShaderCode::GlslEs { .. } => 7,
    }
}

/// Emits a stable discriminant for simple enums.
///
/// The match is exhaustive inside this crate, so adding a variant without
/// giving it a canonical code is a compile error rather than a silent change of
/// every existing fingerprint.
macro_rules! canonical_codes {
    ($($ty:ty { $($variant:ident => $code:literal),+ $(,)? })+) => { $(
        impl $ty {
            /// The stable discriminant used by canonical capability encoding.
            pub(crate) const fn canonical_code(self) -> u8 {
                match self { $(Self::$variant => $code),+ }
            }
        }
    )+ };
}

canonical_codes! {
    BackendKind {
        Dx12 => 1, Vulkan => 2, Metal => 3, WebGpu => 4, OpenGl => 5, WebGl2 => 6,
    }
    OptionalFeature {
        Compute => 1, SamplerAnisotropy => 2, BindingArrays => 3,
    }
    LimitKey {
        MaxBufferSize => 1,
        MaxTexture1dDimension => 2, MaxTexture2dDimension => 3, MaxTexture3dDimension => 4,
        MaxTextureArrayLayers => 5,
        MaxBindGroups => 6, MaxBindingsPerGroup => 7, MaxBindGroupsPlusVertexBuffers => 8,
        MaxUniformBufferBindingSize => 9, MaxStorageBufferBindingSize => 10,
        MaxDynamicUniformBuffersPerPipelineLayout => 11,
        MaxDynamicStorageBuffersPerPipelineLayout => 12,
        MaxSamplerAnisotropy => 13,
        MaxColorAttachments => 14, MaxColorAttachmentBytesPerSample => 15,
        MaxVertexBuffers => 16, MaxVertexAttributes => 17, MaxVertexBufferArrayStride => 18,
        MaxInterStageShaderVariables => 19,
        MaxComputeInvocationsPerWorkgroup => 20,
        MaxComputeWorkgroupSizeX => 21, MaxComputeWorkgroupSizeY => 22,
        MaxComputeWorkgroupSizeZ => 23, MaxComputeWorkgroupsPerDimension => 24,
        MaxComputeWorkgroupStorageSize => 25,
        MinUniformBufferOffsetAlignment => 26, MinStorageBufferOffsetAlignment => 27,
    }
    TextureFormat {
        R8Unorm => 1, R8Snorm => 2, R8Uint => 3, R8Sint => 4,
        Rg8Unorm => 5, Rg8Snorm => 6, Rg8Uint => 7, Rg8Sint => 8,
        Rgba8Unorm => 9, Rgba8UnormSrgb => 10, Rgba8Snorm => 11, Rgba8Uint => 12,
        Rgba8Sint => 13,
        Bgra8Unorm => 14, Bgra8UnormSrgb => 15,
        R16Uint => 16, R16Sint => 17, R16Float => 18,
        Rg16Uint => 19, Rg16Sint => 20, Rg16Float => 21,
        Rgba16Uint => 22, Rgba16Sint => 23, Rgba16Float => 24,
        R32Uint => 25, R32Sint => 26, R32Float => 27,
        Rg32Uint => 28, Rg32Sint => 29, Rg32Float => 30,
        Rgba32Uint => 31, Rgba32Sint => 32, Rgba32Float => 33,
        Depth16Unorm => 34, Depth24Plus => 35, Depth24PlusStencil8 => 36,
        Depth32Float => 37, Depth32FloatStencil8 => 38,
    }
    TextureDimension { D1 => 1, D2 => 2, D3 => 3 }
    TextureAspect { Color => 1, Depth => 2, Stencil => 3 }
    TextureViewDimension {
        D1 => 1, D2 => 2, D2Array => 3, Cube => 4, CubeArray => 5, D3 => 6,
    }
    BlitFilter { Nearest => 1, Linear => 2 }
    TextureSampleType {
        Float => 1, UnfilterableFloat => 2, Sint => 3, Uint => 4, Depth => 5,
    }
    StorageAccess { ReadOnly => 1, WriteOnly => 2, ReadWrite => 3 }
    ShaderNumericType { Float32 => 1, Sint32 => 2, Uint32 => 3 }
    SamplerKind { Filtering => 1, NonFiltering => 2, Comparison => 3 }
    BufferBindingAccess { ReadOnly => 1, ReadWrite => 2 }
    ShaderStage { Vertex => 1, Fragment => 2, Compute => 3 }
    BindingLimitClass {
        UniformBuffers => 1, StorageBuffers => 2, SampledTextures => 3, StorageTextures => 4,
        Samplers => 5,
    }
    SubmissionLaneClass { General => 1, Graphics => 2, Compute => 3, Transfer => 4 }
    LaneDependencyRoute { Ordered => 1, Gpu => 2, Collapse => 3, Unsupported => 4 }
}

impl BindingCount {
    /// The stable discriminant used by canonical capability encoding.
    ///
    /// The element count is written separately, so `One` and `Fixed(n)` never
    /// encode alike.
    pub(crate) const fn canonical_code(self) -> u8 {
        match self {
            Self::One => 0,
            Self::Fixed(_) => 1,
        }
    }
}

impl TextureViewCompatibility {
    /// The stable discriminant used by canonical capability encoding.
    pub(crate) fn canonical_code(self) -> u32 {
        self.bits()
    }
}

impl TextureUsage {
    /// The stable discriminant used by canonical capability encoding.
    pub(crate) fn canonical_code(self) -> u32 {
        self.bits()
    }
}

impl BufferUsage {
    /// The stable discriminant used by canonical capability encoding.
    pub(crate) fn canonical_code(self) -> u32 {
        self.bits()
    }
}

impl LaneWorkDomains {
    /// The stable discriminant used by canonical capability encoding.
    pub(crate) fn canonical_code(self) -> u8 {
        self.bits()
    }
}

impl StorageAccessSupport {
    /// Writes the three independent storage accesses.
    fn canonical(&self, hasher: &mut CanonicalHasher) {
        hasher
            .bool(self.supports(StorageAccess::ReadOnly))
            .bool(self.supports(StorageAccess::WriteOnly))
            .bool(self.supports(StorageAccess::ReadWrite));
    }
}

impl Extent3d {
    /// Writes the three extents.
    fn canonical(&self, hasher: &mut CanonicalHasher) {
        hasher.u32(self.width).u32(self.height).u32(self.depth);
    }
}

/// Writes a length-prefixed `Option`.
fn encode_option<T>(hasher: &mut CanonicalHasher, value: Option<&T>, encode: impl FnOnce(&T, &mut CanonicalHasher)) {
    match value {
        None => {
            hasher.tag(0);
        }
        Some(inner) => {
            hasher.tag(1);
            encode(inner, hasher);
        }
    }
}

/// Writes a canonical (already sorted) set.
fn encode_set<T>(hasher: &mut CanonicalHasher, set: &BTreeSet<T>)
where
    T: CanonicalFact,
{
    hasher.u64(set.len() as u64);
    for value in set {
        value.canonical_encode(hasher);
    }
}

/// Writes a canonical (already sorted) map.
fn encode_map<K, V>(hasher: &mut CanonicalHasher, map: &BTreeMap<K, V>)
where
    K: CanonicalFact,
    V: CanonicalFact,
{
    hasher.u64(map.len() as u64);
    for (key, value) in map {
        key.canonical_encode(hasher);
        value.canonical_encode(hasher);
    }
}

/// Writes the submission lane set and its pairwise routes.
fn encode_submission(hasher: &mut CanonicalHasher, submission: &SubmissionCapabilities) {
    hasher.u64(submission.lanes().len() as u64);
    for lane in submission.lanes() {
        hasher
            .u16(lane.id().as_u16())
            .u8(lane.class().canonical_code())
            .u8(lane.domains().canonical_code());
    }
    let mut relations = 0u64;
    for from in submission.lanes() {
        for to in submission.lanes() {
            if from.id() != to.id() {
                relations += 1;
            }
        }
    }
    hasher.u64(relations);
    for from in submission.lanes() {
        for to in submission.lanes() {
            if from.id() == to.id() {
                continue;
            }
            hasher
                .u16(from.id().as_u16())
                .u16(to.id().as_u16())
                .u8(submission.dependency_route(from.id(), to.id()).canonical_code());
        }
    }
}

/// A fact that can write itself into the canonical encoding.
trait CanonicalFact {
    /// Writes this fact.
    fn canonical_encode(&self, hasher: &mut CanonicalHasher);
}

impl CanonicalFact for u8 {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u8(*self);
    }
}

impl CanonicalFact for u16 {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u16(*self);
    }
}

impl CanonicalFact for u32 {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u32(*self);
    }
}

impl CanonicalFact for u64 {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u64(*self);
    }
}

impl CanonicalFact for TextureFormat {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u8(self.canonical_code());
    }
}

impl CanonicalFact for LimitKey {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u8(self.canonical_code());
    }
}

impl CanonicalFact for OptionalFeature {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u8(self.canonical_code());
    }
}

impl CanonicalFact for BufferUsage {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u32(self.canonical_code());
    }
}

impl CanonicalFact for ShaderStage {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u8(self.canonical_code());
    }
}

impl CanonicalFact for BindingLimitClass {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u8(self.canonical_code());
    }
}

impl<A: CanonicalFact, B: CanonicalFact> CanonicalFact for (A, B) {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        self.0.canonical_encode(hasher);
        self.1.canonical_encode(hasher);
    }
}

impl CanonicalFact for FormatFacts {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher
            .u8(self.aspects().bits())
            .tag(0x01);
        encode_option(hasher, self.sample_type().as_ref(), |value, hasher| {
            hasher.u8(value.canonical_code());
        });
        hasher.tag(0x02);
        self.storage_access().canonical(hasher);
        hasher
            .tag(0x03)
            .bool(self.color_attachment())
            .bool(self.depth_attachment())
            .bool(self.stencil_attachment())
            .bool(self.blendable())
            .bool(self.has_alpha_channel())
            .tag(0x04);
        encode_option(hasher, self.color_output_type().as_ref(), |value, hasher| {
            hasher.u8(value.canonical_code());
        });
        hasher
            .tag(0x05)
            .u32(self.block_width())
            .u32(self.block_height())
            .tag(0x06);
        encode_option(hasher, self.logical_bytes_per_block().as_ref(), |value, hasher| {
            hasher.u32(*value);
        });
    }
}

impl CanonicalFact for BTreeSet<TextureFormat> {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        encode_set(hasher, self);
    }
}

impl CanonicalFact for BufferSupport {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        match self {
            Self::Unsupported => {
                hasher.tag(0);
            }
            Self::Supported(limits) => {
                hasher.tag(1).u64(limits.max_size());
            }
        }
    }
}

impl CanonicalFact for BufferSupportQuery {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u32(self.usage().canonical_code());
    }
}

impl CanonicalFact for TextureSupportLimits {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        self.max_extent().canonical(hasher);
        hasher
            .u32(self.max_mip_levels())
            .u32(self.max_array_layers());
    }
}

impl CanonicalFact for TextureSupport {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        match self {
            Self::Unsupported => {
                hasher.tag(0);
            }
            Self::Supported(limits) => {
                hasher.tag(1);
                limits.canonical_encode(hasher);
            }
        }
    }
}

impl CanonicalFact for TextureSupportQuery {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher
            .u8(self.dimension().canonical_code())
            .u8(self.format().canonical_code())
            .u32(self.usage().canonical_code())
            .u32(self.sample_count())
            .u32(self.view_compatibility().canonical_code());
        hasher.u64(self.view_formats().len() as u64);
        for format in self.view_formats() {
            hasher.u8(format.canonical_code());
        }
    }
}

impl CanonicalFact for BindingKind {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        match self {
            Self::UniformBuffer { min_size } => {
                hasher.tag(1).u64(*min_size);
            }
            Self::StorageBuffer { access, min_size } => {
                hasher.tag(2).u8(access.canonical_code()).u64(*min_size);
            }
            Self::SampledTexture {
                dimension,
                sample_type,
                multisampled,
            } => {
                hasher
                    .tag(3)
                    .u8(dimension.canonical_code())
                    .u8(sample_type.canonical_code())
                    .bool(*multisampled);
            }
            Self::StorageTexture {
                dimension,
                format,
                access,
            } => {
                hasher
                    .tag(4)
                    .u8(dimension.canonical_code())
                    .u8(format.canonical_code())
                    .u8(access.canonical_code());
            }
            Self::Sampler { kind } => {
                hasher.tag(5).u8(kind.canonical_code());
            }
        }
    }
}

impl CanonicalFact for BindingSupportQuery {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.u8(self.visibility.bits()).tag(0x01);
        self.kind.canonical_encode(hasher);
        hasher.tag(0x02).u8(self.count.canonical_code());
        match self.count {
            BindingCount::One => {}
            BindingCount::Fixed(count) => {
                hasher.u32(count);
            }
        }
        hasher.tag(0x03).bool(self.dynamic_offset);
    }
}

impl CanonicalFact for BindingSupport {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.tag(u8::from(self.is_supported()));
    }
}

impl CanonicalFact for RouteCapabilities {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        hasher.tag(0x01);
        encode_option(
            hasher,
            self.buffer_copy_layout().as_ref(),
            |limits, hasher| {
                hasher
                    .u64(limits.offset_alignment())
                    .u64(limits.size_alignment());
            },
        );
        hasher.tag(0x02);
        encode_option(hasher, self.texel_copy_layout().as_ref(), |limits, hasher| {
            hasher
                .u64(limits.buffer_offset_alignment())
                .u32(limits.bytes_per_row_alignment());
        });
    }
}

impl CanonicalFact for RouteSupport {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        match self {
            Self::Unsupported => {
                hasher.tag(0);
            }
            Self::Supported(capabilities) => {
                hasher.tag(1);
                capabilities.canonical_encode(hasher);
            }
        }
    }
}

impl CanonicalFact for RouteQuery {
    fn canonical_encode(&self, hasher: &mut CanonicalHasher) {
        match self {
            Self::BufferToBuffer => {
                hasher.tag(1);
            }
            Self::BufferToTexture {
                dimension,
                format,
                aspect,
            }
            | Self::TextureToBuffer {
                dimension,
                format,
                aspect,
            } => {
                hasher
                    .tag(if matches!(self, Self::BufferToTexture { .. }) {
                        2
                    } else {
                        3
                    })
                    .u8(dimension.canonical_code())
                    .u8(format.canonical_code())
                    .u8(aspect.canonical_code());
            }
            Self::TextureToTexture {
                src_dimension,
                src_format,
                src_aspect,
                src_sample_count,
                dst_dimension,
                dst_format,
                dst_aspect,
                dst_sample_count,
            } => {
                hasher
                    .tag(4)
                    .u8(src_dimension.canonical_code())
                    .u8(src_format.canonical_code())
                    .u8(src_aspect.canonical_code())
                    .u32(*src_sample_count)
                    .u8(dst_dimension.canonical_code())
                    .u8(dst_format.canonical_code())
                    .u8(dst_aspect.canonical_code())
                    .u32(*dst_sample_count);
            }
            Self::Resolve {
                format,
                src_sample_count,
            } => {
                hasher
                    .tag(5)
                    .u8(format.canonical_code())
                    .u32(*src_sample_count);
            }
            Self::Blit {
                src_dimension,
                src_format,
                dst_dimension,
                dst_format,
                filter,
            } => {
                hasher
                    .tag(6)
                    .u8(src_dimension.canonical_code())
                    .u8(src_format.canonical_code())
                    .u8(dst_dimension.canonical_code())
                    .u8(dst_format.canonical_code())
                    .u8(filter.canonical_code());
            }
        }
    }
}

#[cfg(test)]
mod tests;
