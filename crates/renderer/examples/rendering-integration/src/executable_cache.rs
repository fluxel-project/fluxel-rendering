//! Consumer-owned cache for device-local executable products.
//!
//! Logical artifact reuse remains outside this module. Its keys deliberately
//! include the selected RHI device and route ABI, so an executable product can
//! never be reused across a device replacement or a changed lowering target.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

use fluxel_rhi::api::{DeviceId, ShaderFormat, ShaderRoute};

/// A hashable copy of the RHI shader format, which intentionally is not Hash.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderFormatFingerprint {
    Glsl(u16),
    GlslEs(u16),
    SpirV,
    Wgsl,
    Hlsl(u16),
    Dxil,
    MetalSource,
    MetalLibrary,
}

impl From<ShaderFormat> for ShaderFormatFingerprint {
    fn from(format: ShaderFormat) -> Self {
        match format {
            ShaderFormat::Glsl(version) => Self::Glsl(version),
            ShaderFormat::GlslEs(version) => Self::GlslEs(version),
            ShaderFormat::SpirV => Self::SpirV,
            ShaderFormat::Wgsl => Self::Wgsl,
            ShaderFormat::Hlsl(version) => Self::Hlsl(version),
            ShaderFormat::Dxil => Self::Dxil,
            ShaderFormat::MetalSource => Self::MetalSource,
            ShaderFormat::MetalLibrary => Self::MetalLibrary,
        }
    }
}

/// The route and ABI selected by an RHI device for one complete shader description.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RouteFingerprint {
    kind: RouteKind,
    format: ShaderFormatFingerprint,
    environment: String,
    abi_key: String,
}

/// Lowering family selected by RHI.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RouteKind {
    Direct,
    Naga,
}

impl RouteFingerprint {
    /// Copies public RHI route facts into a hashable consumer key.
    pub(crate) fn from_route(route: &ShaderRoute) -> Result<Self, ExecutableCacheError> {
        let (kind, target) = match route {
            ShaderRoute::Direct(target) => (RouteKind::Direct, target),
            ShaderRoute::Naga(target) => (RouteKind::Naga, target),
            ShaderRoute::Unsupported => return Err(ExecutableCacheError::UnsupportedRoute),
        };
        Ok(Self {
            kind,
            format: target.format.into(),
            environment: target.environment.clone(),
            abi_key: target.abi_key.clone(),
        })
    }
}

/// Complete consumer key for a device-local executable.
///
/// `Logical`, `Capabilities`, and `Pipeline` are consumer-owned fingerprints:
/// callers must include every logical source/static input, capability/limit
/// fact actually consulted, and physical layout/state used to create the
/// executable. Dynamic values and concrete frame resources do not belong here.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ExecutableKey<Logical, Capabilities, Pipeline> {
    logical: Logical,
    device: DeviceId,
    capabilities: Capabilities,
    route: RouteFingerprint,
    pipeline: Pipeline,
}

impl<Logical, Capabilities, Pipeline> ExecutableKey<Logical, Capabilities, Pipeline> {
    /// Builds a key after the complete RHI shader description has selected a route.
    pub(crate) fn new(
        logical: Logical,
        device: DeviceId,
        capabilities: Capabilities,
        route: RouteFingerprint,
        pipeline: Pipeline,
    ) -> Self {
        Self {
            logical,
            device,
            capabilities,
            route,
            pipeline,
        }
    }

    /// Returns the device domain owning this executable.
    pub(crate) const fn device(&self) -> DeviceId {
        self.device
    }
}

/// Admission failures for the device-local executable cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ExecutableCacheError {
    UnsupportedRoute,
    DeviceLost(DeviceId),
}

impl fmt::Display for ExecutableCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRoute => formatter.write_str("shader route is unsupported"),
            Self::DeviceLost(device) => write!(formatter, "device {device:?} is unavailable"),
        }
    }
}

impl std::error::Error for ExecutableCacheError {}

/// Generic storage for consumer-created shader modules, layouts, and pipelines.
pub(crate) struct ExecutableCache<Logical, Capabilities, Pipeline, Value> {
    entries: HashMap<ExecutableKey<Logical, Capabilities, Pipeline>, Value>,
    unavailable_devices: HashSet<DeviceId>,
}

impl<Logical, Capabilities, Pipeline, Value> ExecutableCache<Logical, Capabilities, Pipeline, Value>
where
    Logical: Eq + Hash,
    Capabilities: Eq + Hash,
    Pipeline: Eq + Hash,
{
    /// Creates an empty executable cache.
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            unavailable_devices: HashSet::new(),
        }
    }

    /// Looks up a reusable executable unless its device has been lost.
    pub(crate) fn get(
        &self,
        key: &ExecutableKey<Logical, Capabilities, Pipeline>,
    ) -> Result<Option<&Value>, ExecutableCacheError> {
        self.ensure_available(key.device())?;
        Ok(self.entries.get(key))
    }

    /// Inserts a newly created executable unless its device has been lost.
    pub(crate) fn insert(
        &mut self,
        key: ExecutableKey<Logical, Capabilities, Pipeline>,
        value: Value,
    ) -> Result<Option<Value>, ExecutableCacheError> {
        self.ensure_available(key.device())?;
        Ok(self.entries.insert(key, value))
    }

    /// Removes reusable products for a terminally lost device and quarantines its id.
    ///
    /// Submitted owners must be retained by the submission/completion layer;
    /// this only discards values that would otherwise be reused for new work.
    pub(crate) fn invalidate_device(&mut self, device: DeviceId) {
        self.evict_device(device);
        self.unavailable_devices.insert(device);
    }

    /// Discards reusable products for a device being retired by this consumer.
    ///
    /// Unlike `invalidate_device`, replacement retirement does not claim a
    /// device loss: callers may still create new products for that device.
    /// Submitted owners remain with the submission/completion layer.
    pub(crate) fn evict_device(&mut self, device: DeviceId) {
        self.entries.retain(|key, _| key.device() != device);
    }

    /// Reports whether this cache holds a reusable product for the device.
    pub(crate) fn contains_device(&self, device: DeviceId) -> bool {
        self.entries.keys().any(|key| key.device() == device)
    }

    /// Returns how many reusable entries remain for diagnostics/tests.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    fn ensure_available(&self, device: DeviceId) -> Result<(), ExecutableCacheError> {
        if self.unavailable_devices.contains(&device) {
            Err(ExecutableCacheError::DeviceLost(device))
        } else {
            Ok(())
        }
    }
}
