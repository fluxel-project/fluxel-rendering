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
use std::sync::{LazyLock, Mutex};

use crate::api::binding::{BindingLimitClass, BindingSupport, BindingSupportQuery};
use crate::api::format::{FormatFacts, TextureFormat, TextureSupport, TextureSupportQuery};
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportQuery};
use crate::api::resource::route::{RouteQuery, RouteSupport};
use crate::api::shader::{ArtifactAcceptance, ShaderArtifact, ShaderStage};
use crate::api::submission::SubmissionCapabilities;
use crate::base::digest::sha256;

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
/// Crate-private, but a named type rather than an anonymous storage shape,
/// because a backend has to be able to hand one back: `DeviceBackend` cannot
/// return a private field bundle, and the portable layer must not accept a
/// fact-by-fact mutation from a backend that could then hand back a half-filled
/// record. See [`Self::canonical_bytes`] for what the name is also used for.
#[derive(Clone, Debug)]
pub(crate) struct CapabilityFacts {
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
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the DX12 provider starts its enumeration here and the mock backend starts its own here, so with that backend compiled out nothing reaches this"
        )
    )]
    pub(crate) fn empty() -> Self {
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
                .map(|(query, support)| {
                    encode_entry(|out| query.encode_into(out), |out| support.encode_into(out))
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.texture_support
                .iter()
                .map(|(query, support)| {
                    encode_entry(|out| query.encode_into(out), |out| support.encode_into(out))
                })
                .collect(),
        );
        write_section(
            &mut out,
            self.binding_support
                .iter()
                .map(|(query, support)| {
                    encode_entry(|out| query.encode_into(out), |out| support.encode_into(out))
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

        submission.encode_into(&mut out);

        out
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

/// The fills a backend's enumeration performs on a [`CapabilityFacts`].
///
/// # Why these are one expectation rather than nine
///
/// None of them has a caller outside this crate's tests yet: the DX12 capability
/// port that will call them is the next block of this series, and it is a large
/// one because section 7.2's completeness rule means an enumeration has to answer
/// every query the portable layer can be asked, not a representative sample. Until
/// it lands, these are dead in every non-test configuration, and one expectation
/// on the block says so once.
///
/// It is deliberately a block-level expectation and not nine method-level ones:
/// they are one body of code with one fate, deleted together, and the expectation
/// going *unfulfilled* when the DX12 fill arrives is a useful signal rather than a
/// nuisance — it is the gate saying "these are live now, delete the crutch".
impl CapabilityFacts {
    /// Records that the contract offers `feature`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
        )
    )]
    pub(crate) fn record_feature(&mut self, feature: OptionalFeature) {
        self.features.insert(feature);
    }

    /// Records the value for `key`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
        )
    )]
    pub(crate) fn record_limit(&mut self, key: LimitKey, value: u64) {
        self.limits.entries.insert(key, value);
    }

    /// Records the facts for `format`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
        )
    )]
    pub(crate) fn record_format(&mut self, format: TextureFormat, facts: FormatFacts) {
        self.formats.insert(format, facts);
    }

    /// Records the answer to a buffer support query.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
        )
    )]
    pub(crate) fn record_buffer_support(
        &mut self,
        query: BufferSupportQuery,
        support: BufferSupport,
    ) {
        self.buffer_support.insert(query, support);
    }

    /// Records the answer to a texture support query.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
        )
    )]
    pub(crate) fn record_texture_support(
        &mut self,
        query: TextureSupportQuery,
        support: TextureSupport,
    ) {
        self.texture_support.insert(query, support);
    }

    /// Records the answer to a binding support query.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
        )
    )]
    pub(crate) fn record_binding_support(
        &mut self,
        query: BindingSupportQuery,
        support: BindingSupport,
    ) {
        self.binding_support.insert(query, support);
    }

    /// Records the binding-count ceiling for one stage and class.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
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
        not(test),
        expect(
            dead_code,
            reason = "the DX12 capability port is what fills these, and it has not landed yet"
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
}

/// The domain separator every canonical encoding begins with.
///
/// The trailing version is load-bearing. A change to the encoding rules — a new
/// section, a reordered field, a narrower integer — must produce different bytes
/// for the same facts, or a fingerprint recorded before the change would compare
/// equal to one recorded after it while describing something else. Bumping this
/// string is what makes that true, and it is the one step of an encoding change
/// that a compiler cannot be made to insist on.
const ENCODING_DOMAIN: &[u8] = b"fluxel-rhi/capability-facts/v1";

/// Encodes one section entry: the key's length, the key, then the value.
///
/// The key is length-prefixed because it is not always fixed-width — a
/// [`TextureSupportQuery`] carries a `Vec` of alternate view formats — and without
/// the prefix the key/value boundary would depend on the reader already knowing
/// how long the key is. The value needs no prefix: every value encoding is
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
    /// = "dx12"))` rather than on either alone. See `backend::dx12::provider` for
    /// why a `not(test)` expectation on an item the provider references would sit
    /// unfulfilled whenever that backend is compiled.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the only callers are the contract tests and the DX12 provider; with that backend compiled out, adapter enumeration is what will publish one"
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
        CapabilityFacts::recorded(
            self.facts.buffer_support.get(query).copied(),
            "buffer query",
        )
    }

    /// Whether, and within what maxima, the adapter can create the described
    /// texture.
    pub fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        CapabilityFacts::recorded(
            self.facts.texture_support.get(query).copied(),
            "texture query",
        )
    }

    /// Whether, and how, the adapter can satisfy the described binding.
    pub fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        CapabilityFacts::recorded(
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
        CapabilityFacts::recorded(self.facts.routes.get(query).copied(), "route query")
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
        CapabilityFacts::recorded(
            self.facts.buffer_support.get(query).copied(),
            "buffer query",
        )
    }

    /// Whether, and within what maxima, the device can create the described
    /// texture.
    pub fn texture_support(&self, query: &TextureSupportQuery) -> TextureSupport {
        CapabilityFacts::recorded(
            self.facts.texture_support.get(query).copied(),
            "texture query",
        )
    }

    /// Whether, and how, the device can satisfy the described binding.
    pub fn binding_support(&self, query: &BindingSupportQuery) -> BindingSupport {
        CapabilityFacts::recorded(
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
        CapabilityFacts::recorded(self.facts.routes.get(query).copied(), "route query")
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
