//! Capability facts (specification section 7).
//!
//! Capability is not a bag of booleans. Section 7 defines it as a fact database
//! at three levels, and this module owns exactly those three:
//!
//! ```text
//! AvailableCapabilities     what the selected adapter could do
//! EnabledCapabilities       what the created device actually got
//! DeviceLimits              the limit subset of the enabled facts
//! ```
//!
//! The shape is the point. Section 7.3 forbids collapsing this into
//! `HardwareCapabilities { supports_x: bool, .. }`, because every fact must have
//! exactly one canonical source: texture dimension, usage, and sample count are
//! answered by a [`TextureSupportQuery`]; per-stage resource counts by
//! [`EnabledCapabilities::binding_limit`]; binding kind, count, and
//! dynamic-offset legality by a [`BindingSupportQuery`]; format specifics by
//! [`FormatFacts`]; presentation by surface facts. A second `supports_*` boolean
//! for any of those is a defect, not a convenience.
//!
//! # What this module does not own
//!
//! The *vocabulary* of a fact stays with the thing the fact describes — the
//! buffer and texture query types in [`crate::api::resource`], the format facts
//! in [`crate::api::format`], the binding queries in [`crate::api::binding`], and
//! the feature and limit keys in [`crate::api::platform::requirements`]. This
//! module owns only the database that answers them, so a caller comparing what
//! was available against what was enabled learns one shape rather than two.
//!
//! Presentation facts are deliberately absent. Section 7 scopes them to a
//! `(Device, PresentationTarget)` pair rather than to a device, which is why they
//! are a query in [`crate::api::presentation`] and not a field here. A device-wide
//! "presentation capabilities" value would be the degenerate form section 7
//! rules out, because one device can serve two targets with different answers.
//!
//! # Two completeness rules, and they differ
//!
//! [`EnabledCapabilities::format`] answers `Option<FormatFacts>`, and `None` is a
//! real answer: the format is unavailable to this contract. Section 7.2 gives the
//! case that makes the distinction load-bearing — a WebGPU adapter may report a
//! format as available while the created device reports it as unavailable, and
//! that is legal as long as the device was created without enabling the format's
//! required semantics.
//!
//! The remaining queries answer a support *enum* whose negative variant is the
//! real answer. For those there is no `None`, so an absent entry is not a fact at
//! all: it means enumeration never asked. Answering `Supported` would be a lie and
//! answering `Unsupported` would be a different lie, so the query panics and names
//! the query. A backend that leaves a hole in its snapshot has produced a bug, and
//! the rule that a portable defect may not be left for a driver to discover cuts
//! the same way here.
//!
//! A third shape exists, and it is neither of those. [`EnabledCapabilities::supports_feature`]
//! and [`EnabledCapabilities::texture_view_format_compatible`] answer membership in
//! a set, so an absent entry *is* the honest negative — a feature the device did
//! not enable is not enabled, and a format pair enumeration did not list is not
//! compatible. Both relations are recorded positively and exhaustively, which is
//! what makes the default sound rather than a silent guess. The direction matters
//! too: this negative refuses an operation, so a hole in enumeration costs a
//! capability rather than permitting something a driver would reject.
//!
//! # Why the snapshot is data rather than `unimplemented!()`
//!
//! Every accessor in this module is a lookup over facts the backend recorded
//! during enumeration. Writing them as panics would leave the *shape* of the
//! interface unreviewable — the tests in `api::tests::capability` exist to answer
//! "can a caller compare available against enabled without learning two
//! vocabularies", and they cannot answer it against a panic. The storage below is
//! crate-private and is what the backend port fills; the public contract is the
//! accessor list, which is transcribed from section 7.2 unchanged.

use std::collections::{HashMap, HashSet};

use crate::api::binding::{BindingLimitClass, BindingSupport, BindingSupportQuery};
use crate::api::format::{FormatFacts, TextureFormat, TextureSupport, TextureSupportQuery};
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportQuery};
use crate::api::resource::route::{RouteQuery, RouteSupport};
use crate::api::shader::{ArtifactAcceptance, ShaderArtifact, ShaderStage};
use crate::api::submission::SubmissionCapabilities;

/// Process-local exact capability-contract intern token.
///
/// Produced by the RHI through process-wide interning of canonical
/// [`EnabledCapabilities`] semantics, which is what makes it a token a compiled
/// graph can key correctness on without a hash-collision risk. A caller cannot
/// construct one, and there is no public accessor for the integer: the value is
/// evidence of equality, not an ordinal.
///
/// Re-creating a device from the same capability facts yields the same id. That
/// does not make the two devices interchangeable: section 7.1 keeps
/// [`crate::api::identity::DeviceIdentity`] as the isolation boundary, so handles
/// from one are still rejected by the other even when this token matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityCompatibilityId(u64);

impl CapabilityCompatibilityId {
    /// Interns a capability contract.
    ///
    /// Crate-private: only the RHI's interning table may mint one, because a
    /// freely constructible token would let a caller assert an equality the
    /// capability facts do not support.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "minted by the interning table the device port builds"
        )
    )]
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }
}

/// Cache, diagnostics, and capture-provenance fingerprint.
///
/// The field is public where [`CapabilityCompatibilityId`]'s is not, and that
/// asymmetry is deliberate: a fingerprint is a provenance and cache-key token
/// that tooling must be able to write into a report and compare across processes,
/// while section 7.1 is explicit that equal fingerprints cannot alone carry
/// correctness. Nothing in the RHI makes a correctness decision from one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityFingerprint(pub [u8; 32]);

/// The limit subset of the enabled facts.
///
/// A separate type rather than a plain map because section 7.4 makes the
/// *direction* of a limit part of the contract, not a property of each key's
/// name: `MaxFoo` grows stronger with size and `MinFooAlignment` grows stronger as
/// it shrinks. Callers that need to compare requirements against answers express
/// which direction they mean through
/// [`crate::api::platform::requirements::LimitRequirement`], and
/// `crate::api::platform::requirements::LimitKey::larger_is_stronger` is the one
/// place that classifies a key. A `minimum_limit()`-style unifier is ruled out by
/// section 7.4 precisely because it would erase that distinction.
#[derive(Clone, Debug)]
pub struct DeviceLimits {
    entries: HashMap<LimitKey, u64>,
}

impl DeviceLimits {
    /// The value reported for `key`, or `None` when the contract does not define
    /// this limit at all.
    ///
    /// `None` is a fact, not a failure: a limit the capability contract does not
    /// define imposes no requirement.
    pub fn get(&self, key: LimitKey) -> Option<u64> {
        self.entries.get(&key).copied()
    }

    /// The keys this contract defines, in unspecified order.
    ///
    /// For diagnostics and capture provenance. Correctness paths ask
    /// [`Self::get`] with the key they need rather than iterating.
    pub fn keys(&self) -> impl Iterator<Item = LimitKey> + '_ {
        self.entries.keys().copied()
    }
}

/// The fact database both capability levels answer from.
///
/// Section 7.2 gives `AvailableCapabilities` and `EnabledCapabilities` the same
/// portable vocabulary on purpose, so the shared body lives here once and the two
/// public types differ only in what they add — an enabled database carries its
/// compatibility id and fingerprint, and reaches submission and shader-artifact
/// facts that an adapter cannot answer.
///
/// Module-private: it is a storage shape, not part of the contract, and both
/// public types are opaque to a caller.
#[derive(Clone, Debug)]
struct Facts {
    features: HashSet<OptionalFeature>,
    limits: DeviceLimits,
    formats: HashMap<TextureFormat, FormatFacts>,
    buffer_support: HashMap<BufferSupportQuery, BufferSupport>,
    texture_support: HashMap<TextureSupportQuery, TextureSupport>,
    binding_support: HashMap<BindingSupportQuery, BindingSupport>,
    binding_limits: HashMap<(ShaderStage, BindingLimitClass), u32>,
    routes: HashMap<RouteQuery, RouteSupport>,
    /// The *compatible* `(base, view)` format pairs, and only those.
    ///
    /// A set rather than a map of answers, because view compatibility is a
    /// positive relation: a pair that enumeration did not record is not
    /// compatible, and there is no third answer to store. This is the shape
    /// [`Self::features`] has, and for the same reason.
    view_compatibility: HashSet<(TextureFormat, TextureFormat)>,
}

impl Facts {
    fn empty() -> Self {
        Self {
            features: HashSet::new(),
            limits: DeviceLimits {
                entries: HashMap::new(),
            },
            formats: HashMap::new(),
            buffer_support: HashMap::new(),
            texture_support: HashMap::new(),
            binding_support: HashMap::new(),
            binding_limits: HashMap::new(),
            routes: HashMap::new(),
            view_compatibility: HashSet::new(),
        }
    }

    /// Answers a support query that must have been recorded.
    ///
    /// See the module docs for why an absent entry panics instead of defaulting:
    /// neither `Supported` nor `Unsupported` is an honest answer to a question
    /// enumeration never asked.
    fn recorded<V: Copy>(answer: Option<V>, query: &str) -> V {
        match answer {
            Some(value) => value,
            None => panic!(
                "capability snapshot has no recorded answer for this {query}; \
                 enumeration must record an answer for every query a caller can ask"
            ),
        }
    }
}

/// What the selected adapter could do.
///
/// A snapshot of the adapter's own answers, which is *not* a statement about the
/// device that was created from it. Section 7.2 makes the relationship one-way:
///
/// ```text
/// EnabledOnDevice  subset-of  AvailableOnAdapter
/// ```
///
/// so a plan built against this snapshot may still be wrong for the device.
/// Correctness always reads [`crate::api::platform::Device::capabilities`].
#[derive(Clone, Debug)]
pub struct AvailableCapabilities {
    facts: Facts,
}

impl AvailableCapabilities {
    /// An adapter that has been asked nothing yet.
    ///
    /// Crate-private: a caller may not fabricate capability facts, because a
    /// fabricated snapshot would let a caller bypass the very device query
    /// section 7.2 requires correctness to read.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "filled by adapter enumeration when the backend port lands"
        )
    )]
    pub(crate) fn new() -> Self {
        Self {
            facts: Facts::empty(),
        }
    }

    /// Records that the adapter offers `feature`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "filled by adapter enumeration when the backend port lands"
        )
    )]
    pub(crate) fn record_feature(&mut self, feature: OptionalFeature) {
        self.facts.features.insert(feature);
    }

    /// Records the adapter's value for `key`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "filled by adapter enumeration when the backend port lands"
        )
    )]
    pub(crate) fn record_limit(&mut self, key: LimitKey, value: u64) {
        self.facts.limits.entries.insert(key, value);
    }

    /// Records the adapter's facts for `format`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "filled by adapter enumeration when the backend port lands"
        )
    )]
    pub(crate) fn record_format(&mut self, format: TextureFormat, facts: FormatFacts) {
        self.facts.formats.insert(format, facts);
    }

    /// Records the adapter's answer to `query`.
    #[expect(
        dead_code,
        reason = "adapter enumeration records the answer to a buffer support query; nothing reads it yet"
    )]
    pub(crate) fn record_buffer_support(
        &mut self,
        query: BufferSupportQuery,
        support: BufferSupport,
    ) {
        self.facts.buffer_support.insert(query, support);
    }

    /// Records the adapter's answer to `query`.
    #[expect(
        dead_code,
        reason = "adapter enumeration records the answer to a texture support query; nothing reads it yet"
    )]
    pub(crate) fn record_texture_support(
        &mut self,
        query: TextureSupportQuery,
        support: TextureSupport,
    ) {
        self.facts.texture_support.insert(query, support);
    }

    /// Records the adapter's answer to `query`.
    #[expect(
        dead_code,
        reason = "adapter enumeration records the answer to a binding support query; nothing reads it yet"
    )]
    pub(crate) fn record_binding_support(
        &mut self,
        query: BindingSupportQuery,
        support: BindingSupport,
    ) {
        self.facts.binding_support.insert(query, support);
    }

    /// Records the adapter's binding-count ceiling for one stage and class.
    #[expect(
        dead_code,
        reason = "adapter enumeration records the binding-count ceiling; nothing reads it yet"
    )]
    pub(crate) fn record_binding_limit(
        &mut self,
        stage: ShaderStage,
        class: BindingLimitClass,
        limit: u32,
    ) {
        self.facts.binding_limits.insert((stage, class), limit);
    }

    /// Records the adapter's answer to `query`.
    #[expect(
        dead_code,
        reason = "adapter enumeration records the answer to a route query; nothing reads it yet"
    )]
    pub(crate) fn record_route(&mut self, query: RouteQuery, support: RouteSupport) {
        self.facts.routes.insert(query, support);
    }

    /// Whether the adapter offers `feature`.
    pub fn supports_feature(&self, feature: OptionalFeature) -> bool {
        self.facts.features.contains(&feature)
    }

    /// The adapter's value for `key`, or `None` when it defines none.
    pub fn limit(&self, key: LimitKey) -> Option<u64> {
        self.facts.limits.get(key)
    }

    /// The adapter's facts for `format`, or `None` when the format is unavailable
    /// to this contract.
    pub fn format(&self, format: TextureFormat) -> Option<FormatFacts> {
        self.facts.formats.get(&format).copied()
    }

    /// Whether, and within what ceiling, the adapter can create the described
    /// buffer.
    pub fn buffer_support(&self, query: &BufferSupportQuery) -> BufferSupport {
        Facts::recorded(
            self.facts.buffer_support.get(query).copied(),
            "buffer query",
        )
    }

    /// Whether, and within what maxima, the adapter can create the described
    /// texture.
    pub fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        Facts::recorded(
            self.facts.texture_support.get(query).copied(),
            "texture query",
        )
    }

    /// Whether, and how, the adapter can satisfy the described binding.
    pub fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        Facts::recorded(
            self.facts.binding_support.get(query).copied(),
            "binding query",
        )
    }

    /// The adapter's binding-count ceiling for one shader stage and resource
    /// class.
    ///
    /// `None` means the stage and resource class are inapplicable to this
    /// capability contract — a distinction that matters, because "inapplicable"
    /// and "zero allowed" would otherwise be the same answer.
    pub fn binding_limit(&self, stage: ShaderStage, class: BindingLimitClass) -> Option<u32> {
        self.facts.binding_limits.get(&(stage, class)).copied()
    }

    /// Whether, and with what capabilities, the described transfer route exists.
    pub fn route(&self, query: &RouteQuery) -> RouteSupport {
        Facts::recorded(self.facts.routes.get(query).copied(), "route query")
    }
}

/// What the created device actually got.
///
/// The only correct source of capability answers. It carries everything
/// [`AvailableCapabilities`] carries, plus the three things only a device can
/// answer — its compatibility id and fingerprint, its logical submission lanes,
/// and whether it accepts a given shader artifact.
///
/// Immutable by contract: section 7.2 calls it an "opaque immutable device query
/// database". There is no verb that enables a feature after creation, so a caller
/// that needs a capability it did not request must create a new device and accept
/// a new [`crate::api::identity::DeviceIdentity`].
#[derive(Clone, Debug)]
pub struct EnabledCapabilities {
    compatibility_id: CapabilityCompatibilityId,
    fingerprint: CapabilityFingerprint,
    facts: Facts,
    submission: SubmissionCapabilities,
}

impl EnabledCapabilities {
    /// A device that has been asked nothing yet, under the given identity tokens.
    ///
    /// Crate-private: the pair is minted by the RHI's interning table when a
    /// device request completes, and a caller that could choose its own would be
    /// able to claim a compatibility it does not have.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "filled when a device request completes")
    )]
    pub(crate) fn new(
        compatibility_id: CapabilityCompatibilityId,
        fingerprint: CapabilityFingerprint,
    ) -> Self {
        Self {
            compatibility_id,
            fingerprint,
            facts: Facts::empty(),
            // An empty lane set, not a guess. Section 7.2's base lane guarantee
            // is checked separately by `SubmissionCapabilities`' own validator,
            // which the device-request path calls; a constructor that enforced it
            // here would put the rule in two places.
            submission: SubmissionCapabilities::new(Vec::new()),
        }
    }

    /// Records that the device enabled `feature`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "filled when a device request completes")
    )]
    pub(crate) fn record_feature(&mut self, feature: OptionalFeature) {
        self.facts.features.insert(feature);
    }

    /// Records the device's value for `key`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "filled when a device request completes")
    )]
    pub(crate) fn record_limit(&mut self, key: LimitKey, value: u64) {
        self.facts.limits.entries.insert(key, value);
    }

    /// Records the device's facts for `format`.
    #[expect(
        dead_code,
        reason = "a completed device request records the format's facts; nothing reads them yet"
    )]
    pub(crate) fn record_format(&mut self, format: TextureFormat, facts: FormatFacts) {
        self.facts.formats.insert(format, facts);
    }

    /// Records the device's answer to `query`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "filled when a device request completes")
    )]
    pub(crate) fn record_buffer_support(
        &mut self,
        query: BufferSupportQuery,
        support: BufferSupport,
    ) {
        self.facts.buffer_support.insert(query, support);
    }

    /// Records the device's answer to `query`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "filled when a device request completes")
    )]
    pub(crate) fn record_texture_support(
        &mut self,
        query: TextureSupportQuery,
        support: TextureSupport,
    ) {
        self.facts.texture_support.insert(query, support);
    }

    /// Records the device's answer to `query`.
    #[expect(
        dead_code,
        reason = "a completed device request records the answer to a binding support query; nothing reads it yet"
    )]
    pub(crate) fn record_binding_support(
        &mut self,
        query: BindingSupportQuery,
        support: BindingSupport,
    ) {
        self.facts.binding_support.insert(query, support);
    }

    /// Records the device's binding-count ceiling for one stage and class.
    #[expect(
        dead_code,
        reason = "a completed device request records the binding-count ceiling; nothing reads it yet"
    )]
    pub(crate) fn record_binding_limit(
        &mut self,
        stage: ShaderStage,
        class: BindingLimitClass,
        limit: u32,
    ) {
        self.facts.binding_limits.insert((stage, class), limit);
    }

    /// Records the device's answer to `query`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "filled when a device request completes")
    )]
    pub(crate) fn record_route(&mut self, query: RouteQuery, support: RouteSupport) {
        self.facts.routes.insert(query, support);
    }

    /// Records that a texture of `base_format` may be viewed as `view_format`.
    ///
    /// Records the *positive* relation only. Section 8.5 answers a device fact
    /// rather than a rule: two formats of equal byte size are not thereby
    /// view-compatible, and a caller that assumed it would create a view the
    /// driver rejects. Enumeration therefore records every pair its backend
    /// permits, and an unrecorded pair answers `false`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "filled by device enumeration when the backend port lands"
        )
    )]
    pub(crate) fn record_view_compatibility(
        &mut self,
        base_format: TextureFormat,
        view_format: TextureFormat,
    ) {
        self.facts
            .view_compatibility
            .insert((base_format, view_format));
    }

    /// Whether the device's `base_format` may be viewed as `view_format`.
    ///
    /// Answers format compatibility and nothing else. Whether the texture's
    /// descriptor declared the view format, and whether the aspect, dimension, and
    /// mip range are legal, remain the job of view creation's own validation — so
    /// a `true` here is a necessary condition, not permission to skip that check.
    ///
    /// A pair enumeration did not record answers `false`. That direction is the
    /// safe one: refusing a view that would have been legal costs a capability,
    /// while permitting one that is not costs correctness. Section 8.5's warning
    /// that equal byte size does not imply compatibility is the case this
    /// prevents.
    pub fn texture_view_format_compatible(
        &self,
        base_format: TextureFormat,
        view_format: TextureFormat,
    ) -> bool {
        self.facts
            .view_compatibility
            .contains(&(base_format, view_format))
    }
    ///
    /// This is what a compiled graph keys correctness reuse on, together with the
    /// resource, import, and presentation contracts. It is not derived from
    /// [`Self::fingerprint`], and a caller must not substitute one for the other:
    /// section 7.1 keeps the fingerprint out of correctness precisely because
    /// equal hashes cannot carry it.
    pub fn compatibility_id(&self) -> CapabilityCompatibilityId {
        self.compatibility_id
    }

    /// The cache, diagnostics, and provenance fingerprint for this device.
    pub fn fingerprint(&self) -> CapabilityFingerprint {
        self.fingerprint
    }

    /// Whether the device enabled `feature`.
    ///
    /// An adapter that reported the feature available may still answer `false`
    /// here, and this answer is the one that decides whether the operation is
    /// legal.
    pub fn supports_feature(&self, feature: OptionalFeature) -> bool {
        self.facts.features.contains(&feature)
    }

    /// The device's value for `key`, or `None` when it defines none.
    pub fn limit(&self, key: LimitKey) -> Option<u64> {
        self.facts.limits.get(key)
    }

    /// The device's limits as a set.
    ///
    /// The same answers [`Self::limit`] gives, for a caller that reports or diffs
    /// a whole contract rather than probing one key.
    pub fn limits(&self) -> &DeviceLimits {
        &self.facts.limits
    }

    /// The device's facts for `format`, or `None` when the format is unavailable
    /// to this contract.
    ///
    /// Section 7.2 makes the `None` case load-bearing rather than an edge case: on
    /// a platform where a format requires feature enablement, the adapter may
    /// answer `Some` while the device answers `None`, and that is legal. The
    /// device's answer is the one that decides whether a texture can be created.
    pub fn format(&self, format: TextureFormat) -> Option<FormatFacts> {
        self.facts.formats.get(&format).copied()
    }

    /// Whether, and within what ceiling, the device can create the described
    /// buffer.
    pub fn buffer_support(&self, query: &BufferSupportQuery) -> BufferSupport {
        Facts::recorded(
            self.facts.buffer_support.get(query).copied(),
            "buffer query",
        )
    }

    /// Whether, and within what maxima, the device can create the described
    /// texture.
    pub fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        Facts::recorded(
            self.facts.texture_support.get(query).copied(),
            "texture query",
        )
    }

    /// Whether, and how, the device can satisfy the described binding.
    pub fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        Facts::recorded(
            self.facts.binding_support.get(query).copied(),
            "binding query",
        )
    }

    /// The device's binding-count ceiling for one shader stage and resource class.
    ///
    /// `None` means the stage and resource class are inapplicable to this
    /// capability contract, which is not the same answer as zero.
    ///
    /// This is the canonical source for per-stage resource counts. Section 7.3
    /// lists a per-stage count as exactly the kind of fact that must not also
    /// appear as a `ShaderCapabilities` boolean.
    pub fn binding_limit(&self, stage: ShaderStage, class: BindingLimitClass) -> Option<u32> {
        self.facts.binding_limits.get(&(stage, class)).copied()
    }

    /// Whether, and with what capabilities, the described transfer route exists.
    pub fn route(&self, query: &RouteQuery) -> RouteSupport {
        Facts::recorded(self.facts.routes.get(query).copied(), "route query")
    }

    /// The device's logical submission lanes and what each one guarantees.
    ///
    /// Section 7.2 gives this accessor to an enabled device and no matching one to
    /// [`AvailableCapabilities`]: which lanes actually exist is a property of the
    /// created device, not of the adapter it was requested from, so an adapter
    /// cannot answer it and a caller that planned against the adapter would be
    /// wrong in the same way it would be wrong about an unenabled feature.
    ///
    /// The snapshot is immutable like the rest of this type, and it is a *borrow*
    /// of storage this module owns rather than a value computed per call.
    pub fn submission(&self) -> &SubmissionCapabilities {
        &self.submission
    }

    /// Whether the device accepts `artifact`.
    ///
    /// The decision rule is section 19.8's, alongside the artifact provenance the
    /// answer is derived from, so this delegates rather than re-deriving it from
    /// the artifact's fields here. It panics until that module exists.
    ///
    /// Note the asymmetry with [`Self::supports_feature`]: a feature is a fact the
    /// device either has or lacks, while acceptance is a judgement about one
    /// artifact's provenance. Section 7.2 models it as a query for that reason.
    pub fn shader_acceptance(&self, artifact: &ShaderArtifact) -> ArtifactAcceptance {
        let _ = artifact;
        unimplemented!(
            "shader artifact acceptance is decided by module 03 section 19.8; the \
             contract is fixed, the decision rule is not built"
        )
    }
}
