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
//! # Four shapes of answer, and the rule that chooses between two of them
//!
//! The accessors below do not all treat "enumeration did not record this" the same
//! way, and the difference is not taste. The rule is:
//!
//! > **A recorded key must be drawn from a space a backend can walk in full.**
//! > Where it is, an absent entry is a hole in enumeration. Where it is not, an
//! > absent entry is an answer.
//!
//! It is worth stating why that is the rule rather than "always be conservative".
//! A hole and an answer are different failures: a hole costs a capability and is
//! found by whoever needed it, while a hole read as `Supported` permits an
//! operation the device cannot perform and is found by a driver, later, somewhere
//! else. Only one of those is recoverable, so where the two cannot be told apart
//! the answer must be the conservative one — but where they *can* be told apart,
//! collapsing them would delete a bug detector that costs nothing to keep.
//!
//! The four shapes, in the order they appear below:
//!
//! 1. **`Option<V>`, where `None` is a real answer.** [`EnabledCapabilities::format`]
//!    answers `Option<FormatFacts>`. Section 7.2 gives the case that makes the
//!    distinction load-bearing — a WebGPU adapter may report a format as available
//!    while the created device reports it as unavailable, and that is legal as long
//!    as the device was created without enabling the format's required semantics.
//! 2. **Membership in a recorded relation.** [`EnabledCapabilities::supports_feature`]
//!    and [`EnabledCapabilities::texture_view_format_compatible`] answer membership,
//!    so an absent entry *is* the honest negative: a feature the device did not
//!    enable is not enabled, and a format pair enumeration did not list is not
//!    compatible. Both relations are recorded positively, and the direction is the
//!    safe one — this negative refuses an operation, so a hole here costs a
//!    capability rather than permitting something a driver would reject.
//! 3. **A negative-variant answer over an enumerable key.** [`EnabledCapabilities::buffer_support`]
//!    answers a support enum, and its key space is [`BufferUsage`]'s sixty-four
//!    masks — a set a backend can record in full. So an absent entry here is not an
//!    answer to anything: it means enumeration never asked. Answering `Supported`
//!    would be a lie and answering `Unsupported` would be a *different* lie that
//!    nobody would ever trace back to enumeration, so the query panics and names
//!    itself. A backend that leaves a hole in this snapshot has a bug, and section
//!    6.9's rule that a portable defect may not be left for a driver to discover
//!    cuts the same way.
//! 4. **A support enum whose answers no single enumeration settles.**
//!    [`EnabledCapabilities::route`] answers a support enum over keys carrying a
//!    sample count, which has no last element a backend could stop at; there is no
//!    enumeration that could have been complete, so an absent entry cannot be a
//!    hole, and the query answers its negative instead of panicking.
//!    [`EnabledCapabilities::texture_support`] and
//!    [`EnabledCapabilities::binding_support`] are here for that reason too, and
//!    for a second one that outlives it: each is answered at *both* capability
//!    levels, and an adapter-level snapshot has no device to ask, so its answer is
//!    the negative whether or not the key space is walkable. The recorded binding
//!    key is what narrowed the second of those to a finite space; a shape-3 panic
//!    there would land on every adapter instead of on a backend that forgot to fill
//!    a table, which is a bug detector pointed at the wrong party.
//!
//! # What shape 4 costs, and what is done about it
//!
//! Shape 4 trades a bug detector for a correct answer, and it is worth naming the
//! price instead of leaving it implied: a backend that neglects to record a texture
//! format will not panic anywhere, it will refuse textures in that format, and the
//! symptom will look like a driver limitation.
//!
//! Two things keep that from being guesswork. The view-formats field — the one
//! unbounded component that is a *list* rather than a number — is not in the key at
//! all: the recorded key type drops it, and the query answers it from the
//! pairwise relation of shape 2, which is already recorded and already has the
//! right default. And the mock backend's conformance suite records a full snapshot,
//! so a fill that is incomplete in a way that matters shows up as a mock-versus-
//! native disagreement rather than as a mystery on one machine.
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
use std::sync::{LazyLock, Mutex};

use crate::api::binding::vocabulary::BindableKind;
use crate::api::binding::{BindingCount, BindingLimitClass, BindingSupport, BindingSupportQuery};
use crate::api::format::{FormatFacts, TextureFormat, TextureSupport, TextureSupportQuery};
use crate::api::internal::digest::sha256;
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::query::{PipelineStatistics, TimestampQueryCapabilities};
use crate::api::resource::buffer::{BufferSupport, BufferSupportQuery, BufferUsage};
use crate::api::resource::route::{RouteQuery, RouteSupport};
use crate::api::resource::texture::{TextureDimension, TextureUsage, TextureViewCompatibility};
use crate::api::resource::transient::{TransientAllocationSupport, TransientCapabilities};
use crate::api::shader::vocabulary::AcceptedCodeForm;
use crate::api::shader::{ArtifactAcceptance, ShaderArtifact, ShaderStage, ShaderStages};
use crate::api::shader::{CooperativeMatrixProperties, SubgroupSizeRange};
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
/// The recorded key for one texture support question.
///
/// [`TextureSupportQuery`] is not its own key, and the difference is exactly one
/// field. `view_formats` is a `Vec`, so a table keyed on the whole query could
/// never be complete — and worse, it would be asking the wrong question: whether a
/// list of alternate views is legal is not a fact about one format, it is the
/// conjunction of facts about several, and section 8.2 already places it with
/// TextureView validation rather than with the format facts. So it is not in the
/// key, and [`CapabilityFacts`] answers it from the pairwise
/// [`CapabilityFacts::view_compatibility`] relation it already holds.
///
/// Everything left is bounded: three enums, a bitmask, and a small integer. The
/// first four are what a backend's own probe takes as its arguments — DX12's
/// `CheckFormatSupport` is asked about a format *and* a resource dimension *and* a
/// usage — so this key is not a decomposition invented for storage, it is the
/// question the hardware is actually asked.
///
/// Private to this module: nothing outside the capability database needs to name
/// it, and the public `record_texture_support` takes the query a caller has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TextureSupportKey {
    dimension: TextureDimension,
    format: TextureFormat,
    usage: TextureUsage,
    sample_count: u32,
    view_compatibility: TextureViewCompatibility,
}

impl TextureSupportKey {
    /// The part of `query` that a backend can record an answer for.
    fn of(query: &TextureSupportQuery) -> Self {
        Self {
            dimension: query.dimension(),
            format: query.format(),
            usage: query.usage(),
            sample_count: query.sample_count(),
            view_compatibility: query.view_compatibility(),
        }
    }

    /// Writes this key's canonical bytes, field by field in declaration order.
    ///
    /// Every field is written, for the reason [`CapabilityFacts::canonical_bytes`]
    /// gives: two keys that differ anywhere are two different questions, and a
    /// texture the device can create with one sample count is not thereby
    /// creatable with another. `view_compatibility` is written even though most
    /// devices will answer the same for both of its values — section 13.2 makes it
    /// a creation-time fact, and a contract that ignored it would intern two
    /// devices whose cube-view behaviour differs.
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.dimension.encode_into(out);
        self.format.encode_into(out);
        self.usage.encode_into(out);
        out.extend_from_slice(&self.sample_count.to_le_bytes());
        self.view_compatibility.encode_into(out);
    }
}

/// The recorded key for one binding support question.
///
/// [`BindingSupportQuery`] is not its own key, for the same reason
/// [`TextureSupportKey`] is not [`TextureSupportQuery`], and here the reason runs
/// deeper than table size: two of the query's fields are **magnitudes**, and a
/// magnitude is not what a support answer is about.
///
/// The first is `BindingKind`'s `min_size`, carried by both buffer kinds. Whether
/// a *range* is within the device's reach is section 22.3's question, measured
/// against `MaxUniformBufferBindingSize` / `MaxStorageBufferBindingSize` when a
/// BindGroup is created — and that is already where it is asked. The second is
/// [`BindingCount::Fixed`]'s element count, which section 23.1 aggregates per
/// stage and class and compares against `binding_limit(stage, class)`, also
/// already where it is asked. Section 7.3's closing rule — "a fact can only have
/// one canonical source" — makes leaving either of them in this key a second
/// authority for a rule that already has one, and
/// [`CapabilityFacts::binding_support`] states the same conclusion from the other
/// end: this accessor answers whether the *kind* of binding is expressible.
///
/// So the key keeps what the answer actually depends on. `Fixed(n)` survives as
/// the boolean "is this an array", which is what section 20.4's "fixed resource
/// array" limitation is about and what section 7.3 pairs with
/// [`OptionalFeature::BindingArrays`]; how long the array is belongs to the limit.
/// `min_size` does not survive at all, because there is no support-shaped question
/// left in it once its magnitude is section 22.3's.
///
/// Everything left is enumerable — a stage set, small enums, and two booleans —
/// which is what lets a backend record the table in full. The negative-on-absence
/// behaviour is unchanged and is *not* a consequence of the space being
/// unenumerable: an adapter-level snapshot has no device to ask, and answering its
/// negative is the only thing it can honestly do.
///
/// Crate-visible rather than module-private, unlike its texture sibling: a backend
/// records into this table, and the recorder takes a key. The alternative — taking
/// the whole [`BindingSupportQuery`] and discarding two fields inside — would ask
/// every backend to name a `min_size` and an array length it has no answer about,
/// which is a fabricated value in a call the callee immediately throws away.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BindingSupportKey {
    /// The stages that will see the binding.
    pub(crate) visibility: ShaderStages,
    /// The magnitude-free resource semantics.
    pub(crate) kind: BindableKind,
    /// Whether the binding is a fixed-length array rather than a single element.
    pub(crate) array: bool,
    /// Whether this is a runtime-sized descriptor-indexed array.  It must not
    /// collapse with fixed arrays: native descriptor indexing is a distinct
    /// capability and an omitted fact is a fail-closed answer.
    pub(crate) runtime_sized: bool,
    /// Whether a dynamic offset will be applied. Only ever true for buffer kinds.
    pub(crate) dynamic_offset: bool,
}

impl BindingSupportKey {
    /// The part of `query` that a backend can record an answer for.
    ///
    /// [`BindingCount::One`] and every `Fixed(n)` collapse onto `array`, for the
    /// reason the type's documentation gives.
    pub(crate) fn of(query: &BindingSupportQuery) -> Self {
        Self {
            visibility: query.visibility,
            kind: BindableKind::of(&query.kind),
            array: matches!(
                query.count,
                BindingCount::Fixed(_) | BindingCount::RuntimeSized
            ),
            runtime_sized: matches!(query.count, BindingCount::RuntimeSized),
            dynamic_offset: query.dynamic_offset,
        }
    }

    /// Writes this key's canonical bytes, field by field in declaration order.
    ///
    /// Every field is written, for the reason [`CapabilityFacts::canonical_bytes`]
    /// gives: two keys that differ anywhere are two different questions. The
    /// payload inside [`BindableKind`] is written by its own encoder, including
    /// the fields that most devices will answer the same way for — a contract that
    /// dropped them would intern two devices whose `multisampled` or
    /// `dynamic_offset` behaviour differs.
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.visibility.encode_into(out);
        self.kind.encode_into(out);
        out.push(u8::from(self.array));
        out.push(u8::from(self.runtime_sized));
        out.push(u8::from(self.dynamic_offset));
    }
}

/// The non-empty stage sets of section 19.1, in one fixed order.
///
/// Here rather than in a backend because it is a property of the *key space* a
/// capability table has to cover, not of any device: every backend's binding walk
/// needs the same seven, and a walk that enumerated a different seven would leave
/// holes in a table whose holes are silent refusals. A function rather than a
/// `const` because [`ShaderStages::union`] is not a `const fn` — and narrow enough
/// that making it one for this caller's sake would be a change to a public type's
/// declared surface for no other reason.
///
/// The order is by increasing mask, so that a reader can see the sweep is complete
/// by inspection: the three singletons, the three pairs, then the whole set.
#[cfg_attr(
    not(feature = "dx12"),
    expect(
        dead_code,
        reason = "the DX12 binding walk is the only caller, and it is compiled out without the dx12 feature"
    )
)]
pub(crate) fn visibilities() -> [ShaderStages; 7] {
    [
        ShaderStages::VERTEX,
        ShaderStages::FRAGMENT,
        ShaderStages::COMPUTE,
        ShaderStages::VERTEX.union(ShaderStages::FRAGMENT),
        ShaderStages::VERTEX.union(ShaderStages::COMPUTE),
        ShaderStages::FRAGMENT.union(ShaderStages::COMPUTE),
        ShaderStages::VERTEX
            .union(ShaderStages::FRAGMENT)
            .union(ShaderStages::COMPUTE),
    ]
}

/// Crate-private, but a named type rather than an anonymous storage shape,
/// because a backend has to be able to hand one back: `DeviceBackend` cannot
/// return a private field bundle, and the portable layer must not accept a
/// fact-by-fact mutation from a backend that could then hand back a half-filled
/// record. See [`Self::canonical_bytes`] for what the name is also used for.
#[derive(Clone, Debug)]
pub(crate) struct CapabilityFacts {
    features: HashSet<OptionalFeature>,
    /// The [`ShaderCode`](crate::api::shader::ShaderCode) forms this device will
    /// consume.
    ///
    /// A relation rather than a single answer, because a device may consume more
    /// than one form — the GL family backend is the case that forces it, and the
    /// shape is section 7.2's for accepting a code form at all. A form that is
    /// absent is not a form this device refuses *in particular*; it is one this
    /// device does not consume, and `shader_acceptance` answers
    /// [`ArtifactAcceptance::UnsupportedCodeFormat`] for it.
    ///
    /// Not derived from the backend kind. Section 6.3 states in as many words that
    /// the kind a device reports is diagnostics and tooling provenance and **not**
    /// a capability oracle, and two devices of one kind can disagree here.
    code_forms: HashSet<AcceptedCodeForm>,
    limits: DeviceLimits,
    formats: HashMap<TextureFormat, FormatFacts>,
    buffer_support: HashMap<BufferUsage, BufferSupport>,
    /// Keyed on [`TextureSupportKey`], which is *not* the whole query. See that
    /// type for which field is left out and why the answer is still complete.
    texture_support: HashMap<TextureSupportKey, TextureSupport>,
    /// Keyed on [`BindingSupportKey`], which is *not* the whole query. See that
    /// type for which two fields are left out and why the answer is still an answer.
    binding_support: HashMap<BindingSupportKey, BindingSupport>,
    binding_limits: HashMap<(ShaderStage, BindingLimitClass), u32>,
    routes: HashMap<RouteQuery, RouteSupport>,
    /// The *compatible* `(base, view)` format pairs, and only those.
    ///
    /// A set rather than a map of answers, because view compatibility is a
    /// positive relation: a pair that enumeration did not record is not
    /// compatible, and there is no third answer to store. This is the shape
    /// [`Self::features`] has, and for the same reason.
    view_compatibility: HashSet<(TextureFormat, TextureFormat)>,
    transient: TransientCapabilities,
    pipeline_statistics: PipelineStatistics,
    timestamp_queries: TimestampQueryCapabilities,
    subgroup_size: Option<SubgroupSizeRange>,
    cooperative_matrices: Vec<CooperativeMatrixProperties>,
}

impl CapabilityFacts {
    /// A database that has been asked nothing yet.
    ///
    /// Crate-private: a caller may not fabricate capability facts, because a
    /// fabricated snapshot would let a caller bypass the very device query
    /// section 7.2 requires correctness to read. The one legitimate producer is a
    /// backend's enumeration, which starts here and fills the record in.
    ///
    /// The result is *incomplete*, and the record methods are the only way to
    /// complete it. Note what an incomplete database does when published: see
    /// [`Self::recorded`], and see the note in `backend::dx12::provider` about why
    /// the provider that builds one today does not publish it.
    #[cfg_attr(
        all(
            not(test),
            not(any(feature = "dx12", feature = "vulkan", feature = "gl-family"))
        ),
        expect(
            dead_code,
            reason = "the DX12 and Vulkan providers start enumeration here and the mock backend starts its own here; with those backends compiled out nothing reaches this"
        )
    )]
    pub(crate) fn empty() -> Self {
        Self {
            features: HashSet::new(),
            code_forms: HashSet::new(),
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
            transient: TransientCapabilities {
                buffers: TransientAllocationSupport::Dedicated,
                textures: TransientAllocationSupport::Dedicated,
                mixed_resource_aliasing: false,
            },
            pipeline_statistics: PipelineStatistics::NONE,
            timestamp_queries: TimestampQueryCapabilities::NONE,
            subgroup_size: None,
            cooperative_matrices: Vec::new(),
        }
    }

    /// The canonical encoding of these facts.
    ///
    /// # What "canonical" has to mean here
    ///
    /// Two devices that recorded the same facts must encode to the same bytes,
    /// byte for byte, and two devices that recorded different facts must not.
    /// Everything below follows from those two sentences:
    ///
    /// - **Every section is written, and in a fixed order set by this function.**
    ///   Nothing the provider does — which order it enumerated in, which map it
    ///   filled first — reaches the output.
    /// - **Every entry within a section is sorted by its encoded bytes.** The
    ///   facts live in `HashMap`s and `HashSet`s, whose iteration order is
    ///   deliberately unspecified and differs between runs; without this sort,
    ///   identical facts would intern to different ids depending on where the
    ///   hasher happened to put them. Sorting the bytes rather than the key values
    ///   needs no `Ord` on the key types — which matters, because the
    ///   specification fixes every one of their derive lists and none includes
    ///   `Ord`.
    /// - **Every field of every key and value is written**, including fields a
    ///   given variant seems to make redundant. Two records that differ anywhere
    ///   are different contracts, and interning them together is the one wrong
    ///   answer this function can give. Each type's own `encode_into`, in the
    ///   module that declares it, is where that is enforced; this function only
    ///   decides the order the sections go in.
    /// - **The domain string goes first and carries a version.** See
    ///   [`ENCODING_DOMAIN`].
    ///
    /// # Why the submission snapshot is a parameter
    ///
    /// Because the token this feeds is not an id for these facts — it is an id for
    /// the whole [`EnabledCapabilities`] contract, and section 7.1 says so in as
    /// many words: it is "produced by RHI through process-wide interning of
    /// canonical *EnabledCapabilities* semantics". A device's lanes are part of
    /// those semantics: `submission()` is declared on `EnabledCapabilities` and on
    /// nothing else, so a lane layout that this encoding ignored is a contract
    /// difference nothing would record.
    ///
    /// The consequence is not academic. Section 7.1 keys `CompiledGraph`
    /// correctness reuse on the id, and a compiled plan names the lanes it submits
    /// to. Two devices that agreed on every query fact but laid out lanes
    /// differently would otherwise intern to one id, and a plan interned under the
    /// first would be reused against a device that cannot accept its batches.
    ///
    /// The section goes last, after every query section, so that the bytes a
    /// backend can produce without knowing anything about lanes are a prefix of the
    /// bytes it produces with them.
    ///
    /// The bytes are compared exactly by [`intern`] and hashed by
    /// [`EnabledCapabilities::from_facts`]. The comparison is what carries
    /// correctness and the hash is what carries provenance, per section 7.1 and
    /// the module documentation above.
    pub(crate) fn canonical_bytes(&self, submission: &SubmissionCapabilities) -> Vec<u8> {
        let mut out = Vec::with_capacity(4096);
        out.extend_from_slice(ENCODING_DOMAIN);

        write_section(
            &mut out,
            self.features
                .iter()
                .map(|feature| encode_entry(|key| feature.encode_into(key), |_| {}))
                .collect(),
        );
        write_section(
            &mut out,
            self.limits
                .entries
                .iter()
                .map(|(key, value)| {
                    encode_entry(
                        |out| key.encode_into(out),
                        |out| out.extend_from_slice(&value.to_le_bytes()),
                    )
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.formats
                .iter()
                .map(|(format, facts)| {
                    encode_entry(|out| format.encode_into(out), |out| facts.encode_into(out))
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.buffer_support
                .iter()
                .map(|(usage, support)| {
                    encode_entry(|out| usage.encode_into(out), |out| support.encode_into(out))
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.texture_support
                .iter()
                .map(|(key, support)| {
                    encode_entry(|out| key.encode_into(out), |out| support.encode_into(out))
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.binding_support
                .iter()
                .map(|(key, support)| {
                    encode_entry(|out| key.encode_into(out), |out| support.encode_into(out))
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.binding_limits
                .iter()
                .map(|((stage, class), limit)| {
                    encode_entry(
                        |out| {
                            stage.encode_into(out);
                            class.encode_into(out);
                        },
                        |out| out.extend_from_slice(&limit.to_le_bytes()),
                    )
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.routes
                .iter()
                .map(|(query, support)| {
                    encode_entry(|out| query.encode_into(out), |out| support.encode_into(out))
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.view_compatibility
                .iter()
                .map(|(base, view)| {
                    encode_entry(
                        |out| {
                            base.encode_into(out);
                            view.encode_into(out);
                        },
                        |_| {},
                    )
                })
                .collect(),
        );

        // Appended last, after every section that was here before it, so that a
        // section added in a later revision never moves an earlier section's
        // offset. It still needs the domain bump: an encoding that grew a section
        // is a different encoding, and two devices that differ only here would
        // otherwise intern together.
        write_section(
            &mut out,
            self.code_forms
                .iter()
                .map(|form| encode_entry(|out| form.encode_into(out), |_| {}))
                .collect(),
        );

        write_section(
            &mut out,
            vec![encode_entry(
                |out| out.push(0),
                |out| {
                    out.push(match self.transient.buffers {
                        TransientAllocationSupport::Dedicated => 0,
                        TransientAllocationSupport::Aliasing => 1,
                    });
                    out.push(match self.transient.textures {
                        TransientAllocationSupport::Dedicated => 0,
                        TransientAllocationSupport::Aliasing => 1,
                    });
                    out.push(u8::from(self.transient.mixed_resource_aliasing));
                },
            )],
        );
        write_section(
            &mut out,
            vec![encode_entry(
                |out| out.push(0),
                |out| self.timestamp_queries.encode_into(out),
            )],
        );

        write_section(
            &mut out,
            vec![encode_entry(
                |out| out.push(0),
                |out| self.pipeline_statistics.encode_into(out),
            )],
        );
        write_section(
            &mut out,
            self.subgroup_size
                .into_iter()
                .map(|range| encode_entry(|out| out.push(0), |out| range.encode_into(out)))
                .collect(),
        );

        write_section(
            &mut out,
            self.cooperative_matrices
                .iter()
                .map(|properties| encode_entry(|out| properties.encode_into(out), |_| {}))
                .collect(),
        );

        submission.encode_into(&mut out);

        out
    }

    /// Answers a support query whose key space a backend can record in full.
    ///
    /// An absent entry is a hole in enumeration, and the module documentation
    /// explains at length why that is a panic rather than a negative. The short
    /// form: answering `Supported` would permit an operation the device cannot
    /// perform, and answering `Unsupported` would hide a backend bug behind a
    /// driver-shaped symptom. Only the panic is recoverable.
    fn recorded<V: Copy>(answer: Option<V>, query: &str) -> V {
        match answer {
            Some(value) => value,
            None => panic!(
                "capability snapshot has no recorded answer for this {query}; \
                 enumeration must record an answer for every query a caller can ask"
            ),
        }
    }

    /// Answers a support query whose key space no backend can record in full.
    ///
    /// The mirror of [`Self::recorded`], and the difference is the whole reason
    /// both exist. A [`TextureSupportQuery`] carries an unbounded sample count, a
    /// [`BindingSupportQuery`] a
    /// [`crate::api::binding::BindingCount::Fixed`] element count, and a
    /// [`RouteQuery`] a sample count — so "every key in the space" is not a set
    /// anybody can walk, no enumeration could have been complete, and an absent
    /// entry therefore cannot be a hole. It is the honest negative.
    ///
    /// `negative` is passed by the caller rather than produced here because only
    /// the answer type knows which of its variants is the refusing one, and a
    /// helper that guessed would be a second, invisible statement of that fact.
    fn not_enumerable<V: Copy>(answer: Option<V>, negative: V) -> V {
        answer.unwrap_or(negative)
    }
}

/// The fills a backend's enumeration performs on a [`CapabilityFacts`].
///
/// # Why the remaining methods carry an expectation individually
///
/// This block was one expectation over nine methods, on the reasoning that they
/// were one body of code with one fate and that the expectation going
/// *unfulfilled* when the DX12 fill arrived would be the gate saying "these are
/// live now, delete the crutch". That is what happened, and it happened in
/// several parts rather than one — which is the useful part of the story.
///
/// The form the arrival takes is a *narrowing* rather than a deletion: `not(test)`
/// becomes `all(not(test), not(feature = "dx12"))`. A build without the `dx12`
/// feature still compiles these methods with no caller at all, so the expectation
/// is the true statement about that build rather than a crutch being left behind;
/// what the arrival deletes is the *claim* that the DX12 fill has not landed, and
/// that claim lives in the reason string, which is rewritten in the same commit
/// that gives the method its caller. An `expect` whose reason has stopped being
/// true is the failure mode [`Self::record_feature`]'s own reason string was
/// written to announce.
///
/// The block holds ten methods: the nine it started with, plus
/// [`Self::record_code_form`]. Eight of them have made that move
/// ([`Self::record_feature`], [`Self::record_limit`], [`Self::record_format`],
/// [`Self::record_buffer_support`], [`Self::record_texture_support`],
/// [`Self::record_binding_support`], [`Self::record_route`], and
/// [`Self::record_code_form`]): they are reached from a non-test build by
/// `crate::backend::dx12::facts`.
///
/// Two still have no caller outside this crate's tests, and they are not the same
/// kind of gap:
///
/// - [`Self::record_binding_limit`]'s own doc explains why no Direct3D 12 backend
///   will ever call it: this device states no per-stage, per-class binding ceiling,
///   so an entry here would be this port inventing one.
/// - [`Self::record_view_compatibility`] is the fill that has not landed yet.
///
/// One expectation per method is what keeps the signal working when the block
/// splits, and this block has now split five times: each time the method that moved
/// lost its expectation and the ones that had not kept theirs — the crutch was
/// still there to be removed a method at a time, which is exactly what per-method
/// expectations buy.
impl CapabilityFacts {
    /// Records that the contract offers `feature`.
    #[cfg_attr(
        all(not(test), not(any(feature = "dx12", feature = "gl-family"))),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller until Vulkan publishes an implemented optional feature"
        )
    )]
    pub(crate) fn record_feature(&mut self, feature: OptionalFeature) {
        self.features.insert(feature);
    }

    /// Records that the device consumes this code form.
    ///
    /// Some backends have nothing to ask here and record a structural fact — the
    /// DX12 port records `Dxil` because Direct3D 12's only shader input is
    /// bytecode, while Vulkan records `SpirV`; this is why the method takes no
    /// device and no probe result.
    #[cfg_attr(
        all(
            not(test),
            not(any(feature = "dx12", feature = "vulkan", feature = "gl-family"))
        ),
        expect(
            dead_code,
            reason = "DX12 and Vulkan record the code forms their implemented module paths consume; without either backend this is test-only"
        )
    )]
    pub(crate) fn record_code_form(&mut self, form: AcceptedCodeForm) {
        self.code_forms.insert(form);
    }

    /// Records the value for `key`.
    #[cfg_attr(
        all(not(test), not(any(feature = "dx12", feature = "vulkan"))),
        expect(
            dead_code,
            reason = "the DX12 and Vulkan capability ports record probed native limits; without either backend this is test-only"
        )
    )]
    pub(crate) fn record_limit(&mut self, key: LimitKey, value: u64) {
        self.limits.entries.insert(key, value);
    }

    /// Records the exact pipeline-statistics counters native queries can return.
    pub(crate) fn record_pipeline_statistics(&mut self, counters: PipelineStatistics) {
        self.pipeline_statistics = counters;
    }

    /// Records timestamp conversion and resolve facts probed for this device.
    pub(crate) fn record_timestamp_queries(&mut self, facts: TimestampQueryCapabilities) {
        self.timestamp_queries = facts;
    }

    /// Records the valid subgroup-size interval after native feature probing.
    pub(crate) fn record_subgroup_size(&mut self, range: SubgroupSizeRange) {
        self.subgroup_size = Some(range);
    }

    /// Records the facts for `format`.
    #[cfg_attr(
        all(not(test), not(any(feature = "dx12", feature = "gl-family"))),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller, and it is compiled out without the dx12 feature"
        )
    )]
    pub(crate) fn record_format(&mut self, format: TextureFormat, facts: FormatFacts) {
        self.formats.insert(format, facts);
    }

    /// Records the answer to a buffer support query.
    #[cfg_attr(
        all(
            not(test),
            not(any(feature = "dx12", feature = "vulkan", feature = "gl-family"))
        ),
        expect(
            dead_code,
            reason = "the DX12 and Vulkan capability ports call this while building their complete buffer-support tables; without either feature it is unreachable outside tests"
        )
    )]
    pub(crate) fn record_buffer_support(&mut self, usage: BufferUsage, support: BufferSupport) {
        self.buffer_support.insert(usage, support);
    }

    /// Records the answer to a texture support query.
    ///
    /// Takes the whole query and keys on [`TextureSupportKey::of`] of it, so that a
    /// caller records the question it has rather than reconstructing the key by
    /// hand. Two queries that differ only in their alternate view formats therefore
    /// land on one entry, and the *second* recording wins — which is consistent
    /// rather than lossy, because the answer no longer depends on that field: the
    /// accessor applies it as a predicate. A backend that recorded contradictory
    /// answers for one key would be stating that its own probe is unstable, and no
    /// shape of table can repair that.
    #[cfg_attr(
        all(not(test), not(any(feature = "dx12", feature = "vulkan"))),
        expect(
            dead_code,
            reason = "the DX12 and Vulkan capability ports record probed texture support; without either backend this is test-only"
        )
    )]
    pub(crate) fn record_texture_support(
        &mut self,
        query: &TextureSupportQuery,
        support: TextureSupport,
    ) {
        self.texture_support
            .insert(TextureSupportKey::of(query), support);
    }

    /// Records the answer for one binding support key.
    ///
    /// Takes the key rather than the query a caller asks with, because two of the
    /// query's fields are magnitudes no backend has a support answer about — see
    /// the key's documentation — and a recorder that demanded them anyway would ask
    /// every backend to invent a value it is then told to ignore. A backend records
    /// what it knows; the accessor derives the same key from whatever query arrives.
    #[cfg_attr(
        all(not(test), not(any(feature = "dx12", feature = "gl-family"))),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller, and it is compiled out without the dx12 feature"
        )
    )]
    pub(crate) fn record_binding_support(
        &mut self,
        key: BindingSupportKey,
        support: BindingSupport,
    ) {
        self.binding_support.insert(key, support);
    }

    /// Records the binding-count ceiling for one stage and class.
    ///
    /// Nothing fills this yet, and the reason is not that the work is pending: see
    /// [`Self::binding_limit`] for why Direct3D 12 states no per-stage-class
    /// ceiling to record. The test-side callers keep it exercised, and the
    /// expectation below is what keeps a reader from reading "no caller" as "the
    /// port forgot".
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no backend has a per-stage-class ceiling to record yet, which is a fact about the backends rather than an unimplemented port"
        )
    )]
    pub(crate) fn record_binding_limit(
        &mut self,
        stage: ShaderStage,
        class: BindingLimitClass,
        limit: u32,
    ) {
        self.binding_limits.insert((stage, class), limit);
    }

    /// Records the answer to a route query.
    #[cfg_attr(
        all(
            not(test),
            not(any(feature = "dx12", feature = "vulkan", feature = "gl-family"))
        ),
        expect(
            dead_code,
            reason = "the DX12 and Vulkan capability ports record implemented routes; without either backend this is test-only"
        )
    )]
    pub(crate) fn record_route(&mut self, query: RouteQuery, support: RouteSupport) {
        self.routes.insert(query, support);
    }

    /// Records that `base` may be viewed as `view`.
    ///
    /// One direction per call, and only the compatible direction: the relation is
    /// not symmetric — a format may be viewable as one with the same channel
    /// widths but not the reverse — so there is no "record both" convenience here
    /// that could paper over the difference.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
        )
    )]
    pub(crate) fn record_view_compatibility(&mut self, base: TextureFormat, view: TextureFormat) {
        self.view_compatibility.insert((base, view));
    }

    /// Records the device's transient allocation strategy.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "backend capability enumeration will record aliasing support when a backend exposes it"
        )
    )]
    pub(crate) fn record_transient_capabilities(&mut self, transient: TransientCapabilities) {
        self.transient = transient;
    }

    /// Records one concrete cooperative-matrix shape reported by the device.
    pub(crate) fn record_cooperative_matrix(&mut self, properties: CooperativeMatrixProperties) {
        if !self.cooperative_matrices.contains(&properties) {
            self.cooperative_matrices.push(properties);
        }
    }

    pub(crate) fn supports_cooperative_matrix(
        &self,
        requirement: crate::api::shader::CooperativeMatrixRequirement,
    ) -> bool {
        self.cooperative_matrices
            .iter()
            .copied()
            .any(|properties| properties.satisfies(requirement))
    }
}

/// The four support questions, answered once for both snapshots.
///
/// [`AvailableCapabilities`] and [`EnabledCapabilities`] expose the same four
/// verbs, and every rule in them is a rule about *facts* rather than about which
/// snapshot is asking — so the bodies live here and each type delegates. Writing
/// them twice would be two chances for the rule to drift, and the pair they would
/// drift across is exactly the pair section 7.2 requires a caller to be able to
/// compare: an adapter that answered `Supported` where its device answers
/// `Unsupported` is legal only when the difference is feature enablement, and an
/// accidental difference here would be indistinguishable from that.
impl CapabilityFacts {
    /// Whether, and within what ceiling, the described buffer can be created.
    ///
    /// Keyed on [`BufferUsage`], whose space is sixty-four masks, so this is shape
    /// 3 of the module documentation: enumeration can be complete and a hole is a
    /// bug.
    fn buffer_support(&self, query: &BufferSupportQuery) -> BufferSupport {
        Self::recorded(
            self.buffer_support.get(&query.usage()).copied(),
            "buffer query",
        )
    }

    /// Whether, and within what maxima, the described texture can be created.
    ///
    /// Two questions in one, and the module documentation is where the split is
    /// argued. The recorded key answers the shape question; the alternate view
    /// formats are answered here, from the pairwise relation, because a list of
    /// formats is not a fact a table can be keyed on and section 8.2 places its
    /// legality with view validation anyway.
    ///
    /// The conjunction is ordered so that the shape answer short-circuits: a format
    /// the device cannot create at all is not made creatable by listing views for
    /// it, and asking the pairwise relation first would report a missing view
    /// format for a texture that was never creatable — a diagnostic that names the
    /// wrong cause.
    fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        let support = Self::not_enumerable(
            self.texture_support
                .get(&TextureSupportKey::of(query))
                .copied(),
            TextureSupport::Unsupported,
        );
        if !support.is_supported() {
            return support;
        }
        let views_are_permitted = query
            .view_formats()
            .iter()
            .all(|view| self.view_compatibility.contains(&(query.format(), *view)));
        if views_are_permitted {
            support
        } else {
            TextureSupport::Unsupported
        }
    }

    /// Whether the device consumes this code form.
    ///
    /// Crate-visible rather than private because
    /// [`crate::api::shader::acceptance::decide`] reads it. It is deliberately not
    /// a public accessor: section 7.2's frozen `EnabledCapabilities` surface has
    /// no `code_forms` query, and the whole of what a caller may ask about a code
    /// form is [`EnabledCapabilities::shader_acceptance`]'s verdict.
    pub(crate) fn accepts_code_form(&self, form: AcceptedCodeForm) -> bool {
        self.code_forms.contains(&form)
    }

    /// Whether the device enabled `feature`.
    ///
    /// Crate-visible for the same reason: the acceptance rule reads it, and
    /// [`EnabledCapabilities::supports_feature`] is the public spelling.
    pub(crate) fn has_feature(&self, feature: OptionalFeature) -> bool {
        self.features.contains(&feature)
    }

    /// The device's value for `key`, or `None` when it defines none.
    ///
    /// Crate-visible for the same reason;
    /// [`EnabledCapabilities::limit`] is the public spelling.
    pub(crate) fn limit(&self, key: LimitKey) -> Option<u64> {
        self.limits.get(key)
    }

    /// Whether, and how, the described binding can be satisfied.
    ///
    /// Shape 4, and the two magnitudes [`BindingSupportKey`] drops are why: a
    /// `min_size` and a [`crate::api::binding::BindingCount::Fixed`] element count
    /// are both values no enumeration could have walked to the end of, so a miss
    /// here answers the negative rather than panicking. After the narrowing the
    /// recorded space is finite, and the negative stays anyway for a reason that
    /// has nothing to do with size: an adapter-level snapshot has no device to ask,
    /// and its honest answer is the same negative.
    ///
    /// Neither magnitude is checked here. Section 20.4 puts the two questions in
    /// different places on purpose — a count a device cannot reach is a limit, which
    /// `binding_limit` answers per stage and class, and a range beyond the device's
    /// reach is section 22.3's, answered against `MaxUniformBufferBindingSize` /
    /// `MaxStorageBufferBindingSize` when a BindGroup is created — and re-deriving
    /// either here would give one fact two sources, which section 7.3 forbids. This
    /// accessor answers whether the *kind* of binding is expressible; the ceilings
    /// are asked separately.
    pub(crate) fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        Self::not_enumerable(
            self.binding_support
                .get(&BindingSupportKey::of(query))
                .copied(),
            BindingSupport::Unsupported,
        )
    }

    /// Whether, and with what capabilities, the described route exists.
    ///
    /// Shape 4 again: the two texture-to-texture routes carry sample counts, which
    /// are `u32`. Section 9.4 makes `Unsupported` here a refusal to invent a route
    /// rather than a statement about hardware, which is why the negative is the
    /// answer a caller that asked about something absurd should get.
    fn route(&self, query: &RouteQuery) -> RouteSupport {
        Self::not_enumerable(self.routes.get(query).copied(), RouteSupport::Unsupported)
    }
}

/// The domain separator every canonical encoding begins with.
///
/// The trailing version is load-bearing. A change to the encoding rules — a new
/// section, a reordered field, a narrower integer — must produce different bytes
/// for the same facts, or a fingerprint recorded before the change would compare
/// equal to one recorded after it while describing something else. Bumping this
/// string is what makes that true, and it is the one step of an encoding change
/// that a compiler cannot be made to insist on.
///
/// `v3` added the accepted-code-form section. `v2` narrowed the texture-support
/// key's `view_formats` away (see [`TextureSupportKey`]); `v1` was the first
/// encoding.
const ENCODING_DOMAIN: &[u8] = b"fluxel-rhi/capability-facts/v3";

/// Encodes one section entry: the key's length, the key, then the value.
///
/// The key is length-prefixed because it is not always fixed-width — a
/// [`BindingSupportQuery`] carries a [`crate::api::binding::BindingKind`] with a
/// payload, and a [`RouteQuery`]'s variants are different sizes — and without the
/// prefix the key/value boundary would depend on the reader already knowing how
/// long the key is. The value needs no prefix: every value encoding is
/// self-delimiting once the key is known, and the entry's own length bounds it.
///
/// A section with no value — a feature, a view-compatibility pair — passes a
/// no-op for `value` rather than a second helper, so that the element's *key* is
/// still the thing that gets length-prefixed and sorted.
///
/// Crate-visible because a capability contract is not only query facts:
/// [`crate::api::submission::SubmissionCapabilities::encode_into`] contributes the
/// lane snapshot to the same byte string and sorts its two sections the same way.
/// The alternative would be a second copy of the two rules that make the encoding
/// canonical — length-prefix the key, sort the section — in a module that would
/// then have to keep them in step by hand.
pub(crate) fn encode_entry(
    key: impl FnOnce(&mut Vec<u8>),
    value: impl FnOnce(&mut Vec<u8>),
) -> Vec<u8> {
    let mut key_bytes = Vec::new();
    key(&mut key_bytes);
    let mut entry = Vec::with_capacity(key_bytes.len() + 8);
    entry.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
    entry.extend_from_slice(&key_bytes);
    value(&mut entry);
    entry
}

/// Writes one section: a count, then each entry, sorted, each length-prefixed.
///
/// The sort is what makes the encoding independent of the order a provider
/// recorded its facts in; see [`CapabilityFacts::canonical_bytes`].
///
/// Crate-visible for the reason given on [`encode_entry`].
pub(crate) fn write_section(out: &mut Vec<u8>, mut entries: Vec<Vec<u8>>) {
    entries.sort_unstable();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in &entries {
        out.extend_from_slice(&(entry.len() as u32).to_le_bytes());
        out.extend_from_slice(entry);
    }
}

/// The process-wide table that assigns compatibility ids.
///
/// A `Mutex<HashMap>` rather than anything lock-free: this is touched once per
/// device creation, and a device creation is already tens of milliseconds of
/// driver work, so no caller can observe the lock.
///
/// The key is the canonical encoding **itself**, not a digest of it. That is the
/// whole point of the type: section 7.1 requires an id a compiled graph can key
/// correctness on "without hash-collision correctness risk", and a table keyed by
/// the bytes decides equality by comparing them, so there is no collision to be
/// wrong about. Hashing is what [`CapabilityFingerprint`] is for, and section 7.1
/// is explicit that equal fingerprints cannot alone carry correctness.
static COMPATIBILITY_IDS: LazyLock<Mutex<HashMap<Vec<u8>, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Interns a canonical encoding, minting an id the first time it is seen.
fn intern(canonical: &[u8]) -> CapabilityCompatibilityId {
    // A poisoned lock is recovered rather than propagated. The only mutation
    // under it is the insert of a `Vec` that is already built, so a panic while
    // holding it cannot leave the table in a state that matters, and letting one
    // unrelated panic make every later device creation fail would be the worse
    // failure by far.
    let mut table = COMPATIBILITY_IDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Ids start at one, so that zero stays available as the "not an id" value a
    // log or a debugger shows for an uninitialised slot. They are minting order,
    // not ranks: nothing may read meaning into which of two ids is larger, which
    // is why there is no public accessor for the number.
    let next = table.len() as u64 + 1;
    CapabilityCompatibilityId::new(*table.entry(canonical.to_vec()).or_insert(next))
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
    facts: CapabilityFacts,
}

impl AvailableCapabilities {
    /// An adapter snapshot over the facts enumeration produced.
    ///
    /// Crate-private: a caller may not fabricate capability facts, because a
    /// fabricated snapshot would let a caller bypass the very device query
    /// section 7.2 requires correctness to read.
    ///
    /// It takes the whole record rather than pairing a `new()` with the
    /// `record_*` fills, and that is the point of the signature: a snapshot that a
    /// caller may build part-way is a snapshot that will answer "unsupported" to a
    /// question nobody asked it, and [`CapabilityFacts::recorded`] is a panic
    /// precisely because that answer would be indistinguishable from a real one.
    /// Handing over a finished record makes the half-built state unreachable
    /// instead of merely discouraged.
    ///
    /// The expectation is absent whenever *any* caller could exist, and the
    /// contract tests are callers too: it is gated on `all(not(test), not(feature
    /// = "dx12", feature = "vulkan"))` rather than on either alone. See the
    /// providers for why a `not(test)` expectation on an item they reference would sit
    /// unfulfilled whenever that backend is compiled.
    #[cfg_attr(
        all(
            not(test),
            not(any(feature = "dx12", feature = "vulkan", feature = "gl-family"))
        ),
        expect(
            dead_code,
            reason = "the only callers are the contract tests and the DX12/Vulkan providers; with both backends compiled out, adapter enumeration is what will publish one"
        )
    )]
    pub(crate) fn from_facts(facts: CapabilityFacts) -> Self {
        Self { facts }
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
        self.facts.buffer_support(query)
    }

    /// Whether, and within what maxima, the adapter can create the described
    /// texture.
    ///
    /// The answer accounts for the query's alternate view formats as well as its
    /// shape. Whether a list of views is legal is not a fact about one format but
    /// the conjunction of facts about several, so it is answered from the same
    /// pairwise relation that backs
    /// [`EnabledCapabilities::texture_view_format_compatible`] rather than from a
    /// table keyed on the list — a list has no last element to stop at.
    pub fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        self.facts.texture_support(query)
    }

    /// Whether, and how, the adapter can satisfy the described binding.
    ///
    /// A binding count beyond what the adapter can express answers
    /// [`BindingSupport::Unsupported`] rather than panicking, because its key is
    /// not a space enumeration could have covered. The count *ceiling* for one
    /// stage and class is [`Self::binding_limit`]'s question, not this one's.
    pub fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        self.facts.binding_support(query)
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
    ///
    /// A route the adapter has no recorded answer for answers
    /// [`RouteSupport::Unsupported`] rather than panicking, for the reason given on
    /// [`Self::binding_support`]: section 9.4 makes "no direct route exists" a
    /// refusal to invent one, so the negative is an answer here and not a
    /// placeholder for one.
    pub fn route(&self, query: &RouteQuery) -> RouteSupport {
        self.facts.route(query)
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
    facts: CapabilityFacts,
    submission: SubmissionCapabilities,
}

impl EnabledCapabilities {
    /// Exact pipeline-statistics counters supported by this device.
    ///
    /// An empty set is the normal answer when pipeline-statistics queries are not
    /// enabled; it never implies that all counters are available.
    pub fn pipeline_statistics(&self) -> PipelineStatistics {
        self.facts.pipeline_statistics
    }
    /// Timestamp conversion and non-blocking-resolve facts for this device.
    pub fn timestamp_queries(&self) -> TimestampQueryCapabilities {
        self.facts.timestamp_queries
    }
    /// Valid subgroup sizes, or `None` when subgroup operations are unavailable.
    pub fn subgroup_size_range(&self) -> Option<SubgroupSizeRange> {
        self.facts.subgroup_size
    }
    /// Concrete cooperative-matrix shapes the enabled device can lower.
    pub fn cooperative_matrix_properties(&self) -> &[CooperativeMatrixProperties] {
        &self.facts.cooperative_matrices
    }
    /// The device's transient resource allocation capabilities.
    pub fn transient(&self) -> TransientCapabilities {
        self.facts.transient
    }

    /// The device's facts, interned and fingerprinted.
    ///
    /// Crate-private: the two tokens are minted here and nowhere else, because a
    /// caller that could choose its own would be able to claim a compatibility it
    /// does not have.
    ///
    /// It takes the facts rather than the tokens, and that signature is the point
    /// of the constructor. A `new(compatibility_id, fingerprint)` would let a
    /// caller pair an id with a fact set it does not describe — a state this
    /// type's own documentation calls impossible, and which nothing in the crate
    /// could have caught. Here the id *is* a function of the facts: the interning
    /// table decides it by exact comparison of the canonical encoding, so the only
    /// way to obtain one is to hold the facts it stands for.
    ///
    /// Both tokens come from the same bytes on purpose. Interning compares those
    /// bytes and the fingerprint hashes them; section 7.1 lets only the first
    /// carry correctness and only the second cross a process boundary.
    ///
    /// The submission lanes are the caller's, not defaulted here, and the
    /// canonical encoding is over both halves — see
    /// [`CapabilityFacts::canonical_bytes`]. Section 7.2's base lane guarantee is
    /// checked by `SubmissionCapabilities`' own validator, which
    /// [`crate::api::platform::Device::new`] calls; a constructor that also
    /// enforced it here would put the rule in two places.
    ///
    /// That caller settles what the two tokens are minted *over*: the id is the
    /// interning of the whole enabled contract, so it is minted at the one moment
    /// the whole contract is in hand.
    pub(crate) fn from_facts(facts: CapabilityFacts, submission: SubmissionCapabilities) -> Self {
        let canonical = facts.canonical_bytes(&submission);
        let compatibility_id = intern(&canonical);
        let fingerprint = CapabilityFingerprint(sha256(&canonical));
        Self {
            compatibility_id,
            fingerprint,
            facts,
            submission,
        }
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
        self.facts.buffer_support(query)
    }

    /// Whether, and within what maxima, the device can create the described
    /// texture.
    ///
    /// The answer accounts for the query's alternate view formats as well as its
    /// shape, so a texture whose declared views the device cannot form answers
    /// [`TextureSupport::Unsupported`] here rather than at view creation. Section
    /// 13.2 makes the view intent a creation-time fact, which is what puts it in
    /// this question.
    pub fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        self.facts.texture_support(query)
    }

    /// Whether, and how, the device can satisfy the described binding.
    ///
    /// A count beyond what the device can express answers
    /// [`BindingSupport::Unsupported`] rather than panicking; the ceiling itself is
    /// [`Self::binding_limit`]'s question, kept separate by section 20.4 and by
    /// section 7.3's rule that a fact has one canonical source.
    pub fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        self.facts.binding_support(query)
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
        self.facts.route(query)
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
    /// the artifact's fields here.
    ///
    /// Note the asymmetry with [`Self::supports_feature`]: a feature is a fact the
    /// device either has or lacks, while acceptance is a judgement about one
    /// artifact's provenance. Section 7.2 models it as a query for that reason.
    pub fn shader_acceptance(&self, artifact: &ShaderArtifact) -> ArtifactAcceptance {
        crate::api::shader::acceptance::decide(&self.facts, artifact)
    }
}
