//! CPU-only provisional extension points for render-scene snapshots.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::FrameSnapshot;

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

/// An opaque handle naming one statically selected CPU feature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureHandle {
    scene: u64,
    issuer: u64,
    identity: u64,
    name: &'static str,
}

impl FeatureHandle {
    /// Returns the diagnostic feature name.
    pub const fn name(self) -> &'static str {
        self.name
    }
}

/// Scene-controlled issuer for opaque feature handles and contexts.
pub struct FeatureIssuer {
    scene: u64,
    issuer: u64,
}
impl FeatureIssuer {
    pub(crate) const fn new(scene: u64, issuer: u64) -> Self {
        Self { scene, issuer }
    }
    /// Issues an opaque handle for one static feature.
    pub fn issue(&self, name: &'static str) -> Result<FeatureHandle, FeatureError> {
        let identity = NEXT_HANDLE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| FeatureError::HandleExhausted)?;
        Ok(FeatureHandle {
            scene: self.scene,
            issuer: self.issuer,
            identity,
            name,
        })
    }
    /// Creates context after checking scene provenance.
    pub fn context<'a, R>(
        &self,
        snapshot: &'a FrameSnapshot<R>,
        phase: CpuPhase,
        feature: FeatureHandle,
    ) -> Result<FeatureContext<'a, R>, FeatureError> {
        if snapshot.provenance() != self.scene
            || feature.scene != self.scene
            || feature.issuer != self.issuer
        {
            return Err(FeatureError::FeatureHandleMismatch {
                expected: "scene-issued feature",
                actual: feature.name,
            });
        }
        Ok(FeatureContext {
            snapshot,
            phase,
            feature,
        })
    }
    /// Creates typed storage bound to this issuer and scene.
    pub fn storage<T>(&self) -> TypedFeatureStorage<T> {
        TypedFeatureStorage::new(self.scene, self.issuer)
    }
}

/// CPU-only phases available to the provisional facade.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuPhase {
    /// Snapshot-local reference preparation.
    Reference,
}

/// An immutable typed payload pinned to one scene snapshot version.
pub struct TypedFeatureSnapshot<T> {
    scene: u64,
    issuer: u64,
    scene_version: u64,
    data: Arc<T>,
}

impl<T> Clone for TypedFeatureSnapshot<T> {
    fn clone(&self) -> Self {
        Self {
            scene: self.scene,
            issuer: self.issuer,
            scene_version: self.scene_version,
            data: Arc::clone(&self.data),
        }
    }
}

/// Renderer-owned storage for one statically selected typed feature payload.
pub struct TypedFeatureStorage<T> {
    scene: u64,
    issuer: u64,
    current: Option<TypedFeatureSnapshot<T>>,
}

impl<T> TypedFeatureStorage<T> {
    pub(crate) const fn new(scene: u64, issuer: u64) -> Self {
        Self {
            scene,
            issuer,
            current: None,
        }
    }
    /// Publishes data pinned to a scene snapshot and returns the retained snapshot.
    pub fn publish<R>(
        &mut self,
        context: &FeatureContext<'_, R>,
        data: T,
    ) -> Result<TypedFeatureSnapshot<T>, FeatureError> {
        if context.snapshot.provenance() != self.scene || context.feature.issuer != self.issuer {
            return Err(FeatureError::FeatureHandleMismatch {
                expected: "storage issuer",
                actual: context.feature.name,
            });
        }
        let snapshot = TypedFeatureSnapshot::for_context(context, data);
        self.current = Some(snapshot.clone());
        Ok(snapshot)
    }
    /// Returns the most recently published typed snapshot.
    pub fn current(&self) -> Option<&TypedFeatureSnapshot<T>> {
        self.current.as_ref()
    }
}

impl<T> TypedFeatureSnapshot<T> {
    /// Pins payload data to an issuer-controlled scene context.
    pub fn for_context<R>(context: &FeatureContext<'_, R>, data: T) -> Self {
        Self {
            scene: context.snapshot.provenance(),
            issuer: context.feature.issuer,
            scene_version: context.snapshot.version(),
            data: Arc::new(data),
        }
    }
    /// Borrows the typed payload.
    pub fn data(&self) -> &T {
        &self.data
    }
    /// Returns the pinned scene version.
    pub const fn scene_version(&self) -> u64 {
        self.scene_version
    }
}

/// Controlled CPU-only access to an immutable scene snapshot.
pub struct FeatureContext<'a, R> {
    snapshot: &'a FrameSnapshot<R>,
    phase: CpuPhase,
    feature: FeatureHandle,
}

/// CPU-only contribution bound to one feature context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuPhaseContribution {
    scene: u64,
    issuer: u64,
    feature: FeatureHandle,
    version: u64,
}

/// Controlled collector for CPU-only phase contributions.
#[derive(Default)]
pub struct CpuPhaseContributions {
    binding: Option<(u64, u64, u64, CpuPhase)>,
    contributions: Vec<CpuPhaseContribution>,
}

impl CpuPhaseContributions {
    /// Accepts a contribution only from the supplied context.
    pub fn accept<R>(
        &mut self,
        context: &FeatureContext<'_, R>,
        contribution: CpuPhaseContribution,
    ) -> Result<(), FeatureError> {
        let binding = (
            context.snapshot.provenance(),
            context.feature.issuer,
            context.snapshot.version(),
            context.phase,
        );
        if contribution.scene != binding.0
            || contribution.issuer != binding.1
            || contribution.feature != context.feature
            || contribution.version != binding.2
            || self.binding.is_some_and(|current| current != binding)
            || self.contributions.contains(&contribution)
        {
            return Err(FeatureError::FeatureHandleMismatch {
                expected: context.feature.name,
                actual: contribution.feature.name,
            });
        }
        self.binding = Some(binding);
        self.contributions.push(contribution);
        Ok(())
    }
    /// Returns accepted CPU contribution count.
    pub fn len(&self) -> usize {
        self.contributions.len()
    }
    /// Returns whether no CPU contribution has been accepted.
    pub fn is_empty(&self) -> bool {
        self.contributions.is_empty()
    }
}

impl<'a, R> FeatureContext<'a, R> {
    /// Creates a controlled CPU-only phase contribution.
    pub fn contribute(&self) -> CpuPhaseContribution {
        CpuPhaseContribution {
            scene: self.snapshot.provenance(),
            issuer: self.feature.issuer,
            feature: self.feature,
            version: self.snapshot.version(),
        }
    }
    /// Returns the immutable scene snapshot.
    pub fn snapshot(&self) -> &FrameSnapshot<R> {
        self.snapshot
    }
    /// Returns the controlled CPU phase.
    pub const fn phase(&self) -> CpuPhase {
        self.phase
    }
    /// Returns the requesting feature handle.
    pub const fn feature(&self) -> FeatureHandle {
        self.feature
    }
}

/// Structured refusal from CPU feature preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FeatureError {
    /// Typed data belongs to another scene.
    ForeignScene,
    /// Typed data belongs to another issuer in this scene.
    ForeignIssuer,
    /// The process cannot issue another opaque feature handle.
    HandleExhausted,
    /// Typed feature data belongs to another scene version.
    SnapshotVersionMismatch {
        /// Feature name.
        feature: &'static str,
        /// Scene version.
        scene_version: u64,
        /// Payload version.
        feature_version: u64,
    },
    /// The context was created for another feature.
    FeatureHandleMismatch {
        /// Expected name.
        expected: &'static str,
        /// Actual name.
        actual: &'static str,
    },
    /// Feature data was invalid.
    InvalidPreparedData {
        /// Feature name.
        feature: &'static str,
        /// Stable diagnostic.
        detail: &'static str,
    },
}

impl std::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for FeatureError {}

/// A statically dispatched CPU-only scene feature.
pub trait CpuFeature<R> {
    /// Application-owned typed input.
    type Input;
    /// Application-owned prepared output.
    type Prepared;
    /// Prepares a CPU-only contribution from one immutable scene snapshot.
    fn prepare(
        &self,
        context: &FeatureContext<'_, R>,
        input: &TypedFeatureSnapshot<Self::Input>,
    ) -> Result<Self::Prepared, FeatureError>;
}

/// Validates feature identity and snapshot version before preparation.
pub fn validate_context<R, T>(
    context: &FeatureContext<'_, R>,
    input: &TypedFeatureSnapshot<T>,
    expected: FeatureHandle,
) -> Result<(), FeatureError> {
    if context.feature != expected {
        return Err(FeatureError::FeatureHandleMismatch {
            expected: expected.name(),
            actual: context.feature.name(),
        });
    }
    if input.scene != context.snapshot.provenance() {
        return Err(FeatureError::ForeignScene);
    }
    if input.issuer != context.feature.issuer {
        return Err(FeatureError::ForeignIssuer);
    }
    if input.scene_version != context.snapshot.version() {
        return Err(FeatureError::SnapshotVersionMismatch {
            feature: expected.name(),
            scene_version: context.snapshot.version(),
            feature_version: input.scene_version,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/feature/contract.rs"]
mod tests;
