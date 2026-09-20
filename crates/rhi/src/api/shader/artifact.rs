//! Sections 19.8-19.9: the artifact and the created module.
//!
//! The content hash the producer computed, the toolchain version, and the finished
//! [`ShaderArtifact`] — entry point, code, ABI, interface, and hash in one value. The
//! created [`ShaderModule`] is here too, because it is an artifact plus the
//! device identity that accepted it.
//!
//! Not owned here: the interface and requirements the artifact carries (section
//! 19.5-19.7, in `requirements.rs`) and the rule that decides whether an artifact
//! is internally consistent (in `validation.rs`).

use core::fmt;
use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::shader::backend::ShaderModuleBackend;

use super::requirements::{ShaderInterface, ShaderRequirements};
use super::validation::validate_shader_artifact;
use super::vocabulary::{ArtifactAcceptance, ShaderAbiVersion, ShaderCode, ShaderStage};

/// The content-address key computed by the artifact producer.
///
/// A newtype with a public array field, unlike the identity tokens of section 3,
/// and section 19.8 says why in the same breath as it defines it: it is supplied
/// by the producer, it is a *candidate cache key*, and "correctness must not rely
/// only on equal hashes". Object and interface compatibility therefore still use
/// complete canonical semantic validation, which is what keeps this field from
/// being an identity a caller could forge.
///
/// The canonical content domain is fixed by section 19.8:
///
/// ```text
/// stage + entry_point + ShaderCode + ShaderAbiVersion
/// canonical ShaderInterface + ShaderRequirements
/// ```
///
/// `label` is explicitly excluded from it, as are temporary paths, process
/// addresses, and other build-local values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactHash(pub [u8; 32]);

/// The version of the toolchain that produced an artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactProducerVersion {
    /// Breaking component of the toolchain's own version.
    pub major: u16,
    /// Compatible-extension component of the toolchain's own version.
    pub minor: u16,
}

/// One entry point, fully described, ready to be accepted or refused.
///
/// Every field is public because an artifact is data the toolchain produces and
/// the RHI consumes; there is no device fact in it, so there is nothing a caller
/// could claim that the RHI would have to verify against hardware.
///
/// A P0 artifact must have **pipeline specialization closed**: no WGSL required
/// override without a default, no unresolved Vulkan specialization constant, no
/// Metal required function constant (section 19.9). Supplying values at pipeline
/// creation belongs to a future capability family, and P0 does not carry a
/// half-complete constants map in anticipation of it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ShaderArtifact {
    /// Diagnostic label. Excluded from the canonical hash domain (section 19.8).
    pub label: Label,

    /// The stage of this entry point.
    pub stage: ShaderStage,
    /// The entry-point name within the code.
    pub entry_point: String,

    /// The code, in a form the target device must be able to consume.
    pub code: ShaderCode,

    /// The lowering ABI this code was produced against.
    pub abi_version: ShaderAbiVersion,

    /// The portable semantics of this entry point.
    pub interface: ShaderInterface,
    /// What this entry point needs from the device.
    pub requirements: ShaderRequirements,

    /// The producer-computed content hash.
    pub content_hash: ArtifactHash,
    /// That toolchain's version.
    pub producer_version: ArtifactProducerVersion,
}

impl ShaderArtifact {
    /// Assembles an artifact from the facts the producer knows.
    ///
    /// Checks nothing: `validate_shader_artifact` is the check, and it is run by
    /// `create_shader` before the artifact reaches a backend.
    ///
    // Section 19.9 freezes this constructor with its eight parameters and no
    // builder for the fields it sets, so the clippy suggestion to fold them into
    // an argument struct is declined: that struct would be public API the
    // specification did not declare. `expect` rather than `allow`, for the same
    // reason as `SamplerDescriptor`: a suppression that cannot expire is a
    // suppression nobody re-checks.
    #[expect(
        clippy::too_many_arguments,
        reason = "section 19.9 freezes this constructor with eight parameters and no builder; folding them into a struct would add public API the specification did not declare"
    )]
    pub fn new(
        stage: ShaderStage,
        entry_point: impl Into<String>,
        code: ShaderCode,
        abi_version: ShaderAbiVersion,
        interface: ShaderInterface,
        requirements: ShaderRequirements,
        content_hash: ArtifactHash,
        producer_version: ArtifactProducerVersion,
    ) -> Self {
        Self {
            label: Label::default(),
            stage,
            entry_point: entry_point.into(),
            code,
            abi_version,
            interface,
            requirements,
            content_hash,
            producer_version,
        }
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}

/// A created entry point on one device.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the
/// [`DeviceIdentity`] that created it. It owns its artifact rather than borrowing
/// it, because a module's interface outlives the call that created
/// it: section 28.1 requires everything a pipeline needs to be re-describable from
/// the artifacts and interfaces it was built from.
#[derive(Clone)]
pub struct ShaderModule {
    inner: Arc<ShaderModuleInner>,
}

/// The one ownership domain of a created shader module.
///
/// A module is a single logical handle.  Keeping its portable description and
/// native backing together means cloning the public handle performs one shared
/// ownership operation rather than independently sharing only the native half.
struct ShaderModuleInner {
    id: ObjectId,
    device: DeviceIdentity,
    artifact: ShaderArtifact,
    /// The backend's own entry point, in the same shape
    /// [`crate::api::resource::Buffer`] holds its allocation: `Arc` so that the
    /// handle stays `Clone` without the backend cloning a native device, and
    /// behind `dyn` so that no native type reaches the exported surface (section
    /// 59). Section 19.10 declares no accessor for it, and it is reached only by a
    /// later chapter's device verb, which downcasts inside its own backend.
    native: Box<dyn ShaderModuleBackend>,
}

impl ShaderModule {
    /// Assembles a created module.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only `Device::create_shader` may produce one, and the identity is minted
    /// there rather than here — a constructor that minted its own would give a
    /// caller who assembled one directly an object the process has never recorded.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        artifact: ShaderArtifact,
        native: Box<dyn ShaderModuleBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(ShaderModuleInner {
                id,
                device,
                artifact,
                native,
            }),
        }
    }

    /// The native entry point behind this module.
    ///
    /// Crate-private for the reason the whole seam is: section 59 keeps native
    /// types off the exported surface. The caller will be the pipeline chapter's
    /// device verb, which reaches each stage's own backend type to fill a native
    /// shader bytecode struct; it is a backend's own lowering that may cross here,
    /// exactly as [`crate::api::resource::Buffer::native`] documents for its side
    /// of the seam.
    ///
    /// # Why the expectation is gated on `test` alone, unlike `Buffer::native`'s
    ///
    /// `Buffer::native` narrows to the backend feature list because the DX12 command
    /// spine is a real caller. This accessor has none in any configuration: the DX12
    /// pipeline lowering is what will read it, and that lowering is not written.
    /// Gating it on a backend feature would therefore be a claim that some backend
    /// reads it, which is not yet true of any of them.
    ///
    /// So the gate is `test`, and the reason says what is actually the case. When
    /// the pipeline lowering lands, this `expect` sits *unfulfilled* in that
    /// configuration and the build fails — which is the intended forcing function
    /// rather than an accident: the attribute's reason has stopped being true, and
    /// `expect` is what makes that a compile error instead of a stale comment.
    #[cfg_attr(
        all(not(test), not(any(feature = "dx12", feature = "vulkan"))),
        expect(dead_code, reason = "read by backend pipeline lowering")
    )]
    pub(crate) fn native(&self) -> &dyn ShaderModuleBackend {
        self.inner.native.as_ref()
    }

    /// This module's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }

    /// The device that created this module.
    ///
    /// Section 3.3 makes this the only answer to a cross-device use: there is no
    /// implicit recompile or module transfer, so a module from another device is a
    /// refusal.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }

    /// The artifact this module was created from.
    ///
    /// Still needed after creation: a pipeline re-describes its stages from their
    /// artifacts (section 28.1).
    pub fn artifact(&self) -> &ShaderArtifact {
        &self.inner.artifact
    }

    /// The stage of this module's entry point.
    ///
    /// A convenience over `module.artifact().stage`, and the only place the stage
    /// is decided: a module is exactly one artifact entry point (section 19.10),
    /// so there is no second stage to disagree with.
    pub fn stage(&self) -> ShaderStage {
        self.inner.artifact.stage
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived: section 19.10 declares
/// `#[derive(Clone)]` and no `Debug` on this handle, while section 28 declares
/// descriptors that *contain* a module and do derive `Debug` — the specification
/// is internally inconsistent here, and the resolution (defect D6 of the 0.16
/// plan) is that every public opaque handle implements `Debug` portably.
///
/// It prints the identity rather than the artifact because section 7.1 describes
/// an object by its identity, because the backend port will add a native field
/// that has no reason to be `Debug`, and because printing a native handle into a
/// log would leak it. `finish_non_exhaustive()` is what makes it honest that the
/// artifact is not shown — a caller who needs it calls [`ShaderModule::artifact`].
impl fmt::Debug for ShaderModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ShaderModule")
            .field("id", &self.inner.id)
            .field("device", &self.inner.device)
            .finish_non_exhaustive()
    }
}

/// Section 19.10's creation verb, defined in the chapter that owns the type it
/// produces.
///
/// The placement is section 19.10's own: the specification writes an `impl Device`
/// in each chapter for that chapter's verbs, so the definition site is the owner.
/// Reading "what can a `Device` create" therefore takes a search across this
/// tree rather than one file, which is the deliberate cost of the rule.
impl Device {
    /// Compiles an artifact into a module on this device.
    ///
    /// Everything section 19.10 lists is checked before the stop, through
    /// `validate_shader_artifact`, including the binding-support question every
    /// resource in the artifact's interface asks. Nothing portable is left to the
    /// backend: a [`ShaderArtifact`] carries no [`DeviceIdentity`] — it is
    /// producer-side data with a content hash — so there is no wrong-device
    /// argument here to refuse, and the device's own answers are the only input
    /// the check needs.
    ///
    /// # The two questions, and why both are asked
    ///
    /// `validate_shader_artifact` asks whether the *artifact* is internally
    /// consistent — a producer's question, answered with
    /// [`RhiErrorKind::InvalidUsage`].
    /// [`EnabledCapabilities::shader_acceptance`](crate::api::capability::EnabledCapabilities::shader_acceptance)
    /// then asks whether **this device** may use it, which is a fact about the
    /// device and not a mistake by the caller.
    ///
    /// Both run, and the order is not a preference: a consumer may call the second
    /// on its own — it is the question "should I build this artifact at all" — so
    /// the two are allowed to overlap on the binding-support step, and
    /// `acceptance.rs` records why that overlap is deliberate. What the order here
    /// settles is which answer a caller gets when the artifact is *both*
    /// ill-formed and unacceptable: the refusal it can act on without touching a
    /// device, which is the artifact's own.
    ///
    /// # What a returned module does not prove
    ///
    /// That its bytes are a legal program. For a source form, a runtime compiler
    /// error arrives here (section 19.10: [`crate::api::RhiError`] plus a
    /// `DiagnosticEvent`), but for a form a backend copies through to a later
    /// native call there is nothing to compile yet, and a backend must not pretend
    /// otherwise — `Dx12ShaderModule` records the concrete case.
    pub async fn create_shader(&self, artifact: &ShaderArtifact) -> RhiResult<ShaderModule> {
        // Section 6.5 refuses creation through a lost device. There is no
        // ownership comparison ahead of it here because a `ShaderArtifact`
        // carries no `DeviceIdentity` — it is producer-side data with a content
        // hash, as the doc above states — so the device's own liveness is the
        // first device-side question this verb can ask.
        self.require_active()
            .map_err(|error| error.at("Device::create_shader"))?;

        let capabilities = self.capabilities();
        validate_shader_artifact(artifact, |query| capabilities.binding_support(query))
            .map_err(|error| error.at("Device::create_shader"))?;

        // Section 19.10's verdict, and the reason it is not an `RhiError`: the
        // artifact is well-formed and the device is the thing that cannot use it,
        // so the answer is a value the caller reads rather than a failure the
        // caller unwraps. A refusal carries the same information either way — the
        // caller must not build this — but `Unsupported` would say "the RHI does
        // not implement this" about a device that simply answered a question.
        let acceptance = capabilities.shader_acceptance(artifact);
        if acceptance != ArtifactAcceptance::Accepted {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device refused the {:?} entry point {:?}: {acceptance:?}",
                    artifact.stage, artifact.entry_point
                ),
            )
            // Named here rather than in the message, so that a caller reading
            // `operation()` and one reading the text are told the same thing once.
            .at("Device::create_shader"));
        }

        // The one backend call. Everything above it is a portable verdict about
        // the artifact and about this device; this is the backend's answer to how
        // its own API wants the bytes.
        let native = self.native().create_shader(artifact)?;
        Ok(ShaderModule::new(
            ObjectId::next(),
            self.identity(),
            artifact.clone(),
            native,
        ))
    }
}
