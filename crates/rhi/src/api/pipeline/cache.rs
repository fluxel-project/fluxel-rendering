//! Portable driver pipeline-cache objects.
//!
//! A cache is deliberately an opaque, device-scoped object.  Its bytes are a
//! backend implementation artifact, while its validation key is the portable
//! statement of the adapter/device contract those bytes were produced for.

use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::platform::requirements::OptionalFeature;

use super::backend::PipelineCacheBackend;

/// A stable validation key attached to serialized pipeline-cache bytes.
///
/// The bytes are opaque to applications.  Equality is useful only to decide
/// whether a persisted blob was made for the same backend/device contract; it
/// must never be interpreted as a native driver identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineCacheValidationKey([u8; 32]);

impl PipelineCacheValidationKey {
    /// Returns the opaque key bytes for persistence beside cache data.
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Constructs a key from persisted bytes.
    ///
    /// The key does not assert that the corresponding cache data is valid; that
    /// judgement remains with [`PipelineCacheDescriptor::fallback`] at device
    /// creation time.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// What to do when supplied cache data cannot be restored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineCacheFallback {
    /// Refuse cache creation and preserve the caller's persisted data.
    RejectInvalidData,
    /// Start an empty cache.  The input bytes are never partially accepted.
    IgnoreInvalidData,
}

/// Input used to create one [`PipelineCache`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PipelineCacheDescriptor {
    /// Diagnostic-only label; excluded from cache compatibility.
    pub label: Label,
    /// Previously serialized opaque data, if any.
    pub initial_data: Option<Arc<[u8]>>,
    /// Validation key persisted with `initial_data`, if it was persisted.
    pub validation_key: Option<PipelineCacheValidationKey>,
    /// Behaviour for absent, mismatched, or corrupt data.
    pub fallback: PipelineCacheFallback,
}

impl PipelineCacheDescriptor {
    /// Describes a fresh empty cache.
    pub fn new() -> Self {
        Self {
            label: Label::default(),
            initial_data: None,
            validation_key: None,
            fallback: PipelineCacheFallback::RejectInvalidData,
        }
    }

    /// Supplies a persisted cache blob and the key stored beside it.
    pub fn with_initial_data(mut self, key: PipelineCacheValidationKey, data: Arc<[u8]>) -> Self {
        self.validation_key = Some(key);
        self.initial_data = Some(data);
        self
    }

    /// Selects the invalid-data behaviour.
    pub fn with_fallback(mut self, fallback: PipelineCacheFallback) -> Self {
        self.fallback = fallback;
        self
    }

    /// Sets a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}

impl Default for PipelineCacheDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

/// A device-scoped native pipeline-cache object.
#[derive(Clone)]
pub struct PipelineCache {
    inner: Arc<PipelineCacheInner>,
}

struct PipelineCacheInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: PipelineCacheDescriptor,
    validation_key: PipelineCacheValidationKey,
    serialization_enabled: bool,
    native: Box<dyn PipelineCacheBackend>,
}

impl PipelineCache {
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: PipelineCacheDescriptor,
        validation_key: PipelineCacheValidationKey,
        serialization_enabled: bool,
        native: Box<dyn PipelineCacheBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(PipelineCacheInner {
                id,
                device,
                descriptor,
                validation_key,
                serialization_enabled,
                native,
            }),
        }
    }

    /// The object identity.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }
    /// The owning device identity.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }
    /// The creation descriptor.
    pub fn descriptor(&self) -> &PipelineCacheDescriptor {
        &self.inner.descriptor
    }
    /// The key to persist beside [`Self::serialized_data`].
    pub fn validation_key(&self) -> PipelineCacheValidationKey {
        self.inner.validation_key
    }
    /// Serializes current cache contents when the backend supports persistence.
    pub fn serialized_data(&self) -> RhiResult<Vec<u8>> {
        if !self.inner.serialization_enabled {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device cannot serialize pipeline-cache data",
            )
            .at("PipelineCache::serialized_data"));
        }
        self.inner.native.serialized_data()
    }
    pub(crate) fn native(&self) -> &dyn PipelineCacheBackend {
        self.inner.native.as_ref()
    }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("id", &self.inner.id)
            .field("device", &self.inner.device)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Creates an empty or restored native pipeline cache.
    pub fn create_pipeline_cache(
        &self,
        descriptor: &PipelineCacheDescriptor,
    ) -> RhiResult<PipelineCache> {
        self.require_active()?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::PipelineCache)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "pipeline caches are not enabled on this device",
            )
            .at("Device::create_pipeline_cache"));
        }
        if descriptor.initial_data.is_some() != descriptor.validation_key.is_some() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "pipeline cache data and its validation key must be supplied together",
            )
            .at("Device::create_pipeline_cache"));
        }
        if descriptor.initial_data.is_some()
            && !self
                .capabilities()
                .supports_feature(OptionalFeature::PipelineCacheSerialization)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device cannot restore serialized pipeline-cache data",
            )
            .at("Device::create_pipeline_cache"));
        }
        let serialization_enabled = self
            .capabilities()
            .supports_feature(OptionalFeature::PipelineCacheSerialization);
        let (native, key) = self.native().create_pipeline_cache(descriptor)?;
        Ok(PipelineCache::new(
            ObjectId::next(),
            self.identity(),
            descriptor.clone(),
            key,
            serialization_enabled,
            native,
        ))
    }
}
