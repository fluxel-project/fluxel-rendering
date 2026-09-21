//! GPU query objects and their portable recording vocabulary.

use core::fmt;
use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::backend::QuerySetBackend;

/// How an enabled device binds an occlusion query set to a raster pass.
///
/// The distinction is observable at recording time.  D3D12/Vulkan-style
/// devices can select a set while a pass is open; WebGPU binds one set in the
/// render-pass descriptor and can only begin/end indices from that set.  This
/// profile lets a backend expose the latter honestly instead of either
/// pretending it supports dynamic selection or disabling occlusion entirely.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OcclusionQueryBinding {
    /// An occlusion query set may be selected by `RasterScope::begin_query`.
    #[default]
    Dynamic,
    /// The one usable set is fixed when the raster scope begins.
    FixedAtRasterScope,
}

impl OcclusionQueryBinding {
    pub(crate) fn encode_into(self, out: &mut Vec<u8>) {
        out.push(match self {
            Self::Dynamic => 0,
            Self::FixedAtRasterScope => 1,
        });
    }
}

/// Timestamp conversion and resolve facts for one enabled device.
///
/// `period_nanos` converts a resolved native tick to nanoseconds. `valid_bits`
/// is `None` when native timestamps are full-width; otherwise callers mask a
/// resolved tick to the reported low-bit width before doing wrap-aware math.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampQueryCapabilities {
    /// Nanoseconds represented by one timestamp tick, when timestamps exist.
    pub period_nanos: Option<f64>,
    /// Meaningful native timestamp bits, if the backend has a narrower counter.
    pub valid_bits: Option<u8>,
    /// Whether query results may be resolved without waiting for availability.
    pub non_blocking_resolve: bool,
}

impl TimestampQueryCapabilities {
    /// No timestamp conversion or resolve facility.
    pub const NONE: Self = Self {
        period_nanos: None,
        valid_bits: None,
        non_blocking_resolve: false,
    };

    /// Builds valid timestamp facts, rejecting non-finite/zero periods and
    /// unusable bit widths before a backend can publish them.
    pub fn new(
        period_nanos: f64,
        valid_bits: Option<u8>,
        non_blocking_resolve: bool,
    ) -> Option<Self> {
        if !period_nanos.is_finite() || period_nanos <= 0.0 || matches!(valid_bits, Some(0)) {
            return None;
        }
        Some(Self {
            period_nanos: Some(period_nanos),
            valid_bits,
            non_blocking_resolve,
        })
    }

    /// Converts one native tick count to nanoseconds when conversion is known.
    pub fn ticks_to_nanos(self, ticks: u64) -> Option<f64> {
        self.period_nanos.map(|period| ticks as f64 * period)
    }

    pub(crate) fn encode_into(self, out: &mut Vec<u8>) {
        match self.period_nanos {
            Some(period) => {
                out.push(1);
                out.extend_from_slice(&period.to_bits().to_le_bytes());
            }
            None => out.push(0),
        }
        match self.valid_bits {
            Some(bits) => {
                out.push(1);
                out.push(bits);
            }
            None => out.push(0),
        }
        out.push(u8::from(self.non_blocking_resolve));
    }
}

/// Pipeline-statistics counters selected by a query set.
///
/// This is a selection mask, not decoded query output: native query results are
/// resolved into caller-owned buffer bytes. Backends publish their supported
/// subset through [`crate::api::capability::EnabledCapabilities::pipeline_statistics`]
/// so a descriptor never implies counters the device cannot produce.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PipelineStatistics(u32);

impl PipelineStatistics {
    /// Vertex-shader invocations.
    pub const VERTEX_SHADER_INVOCATIONS: Self = Self(1 << 0);
    /// Clipper invocations.
    pub const CLIPPER_INVOCATIONS: Self = Self(1 << 1);
    /// Primitives emitted by the clipper.
    pub const CLIPPER_PRIMITIVES_OUT: Self = Self(1 << 2);
    /// Fragment-shader invocations.
    pub const FRAGMENT_SHADER_INVOCATIONS: Self = Self(1 << 3);
    /// Compute-shader invocations.
    pub const COMPUTE_SHADER_INVOCATIONS: Self = Self(1 << 4);
    /// Every counter currently named by this portable vocabulary.
    pub const ALL: Self = Self(
        Self::VERTEX_SHADER_INVOCATIONS.0
            | Self::CLIPPER_INVOCATIONS.0
            | Self::CLIPPER_PRIMITIVES_OUT.0
            | Self::FRAGMENT_SHADER_INVOCATIONS.0
            | Self::COMPUTE_SHADER_INVOCATIONS.0,
    );
    /// No selected counter.
    pub const NONE: Self = Self(0);
    /// Combines two counter selections.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    /// Whether every counter in `other` is selected here.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    /// Whether the selection contains no counter.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    /// Number of 64-bit result words one pipeline-statistics query writes.
    ///
    /// Results are packed in the stable vocabulary order, containing exactly the
    /// selected counters.  Native APIs with wider fixed records are repacked by
    /// backend lowering; exposing their padding would make query-buffer layout a
    /// backend ABI.
    pub const fn result_words(self) -> u32 {
        self.0.count_ones()
    }
    pub(crate) fn encode_into(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
}

/// The kind of result stored by a query set.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryType {
    /// Counts samples that passed depth/stencil testing.
    Occlusion,
    /// Records a device timestamp.
    Timestamp,
    /// Records the selected pipeline counters.
    PipelineStatistics(PipelineStatistics),
}

impl QueryType {
    /// Number of 64-bit result words written for one query slot on resolve.
    pub const fn result_words(self) -> u32 {
        match self {
            Self::Occlusion | Self::Timestamp => 1,
            Self::PipelineStatistics(selection) => selection.result_words(),
        }
    }
}

/// Query-set creation parameters.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct QuerySetDescriptor {
    /// Diagnostic label.
    pub label: Label,
    /// Result type.
    pub ty: QueryType,
    /// Number of addressable query slots.
    pub count: u32,
}

impl QuerySetDescriptor {
    /// Creates an unlabelled query-set descriptor.
    pub fn new(ty: QueryType, count: u32) -> Self {
        Self {
            label: Label::default(),
            ty,
            count,
        }
    }

    /// Adds a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}

/// One GPU query-set handle.
#[derive(Clone)]
pub struct QuerySet {
    inner: Arc<QuerySetInner>,
}

struct QuerySetInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: QuerySetDescriptor,
    native: Box<dyn QuerySetBackend>,
}

impl QuerySet {
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: QuerySetDescriptor,
        native: Box<dyn QuerySetBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(QuerySetInner {
                id,
                device,
                descriptor,
                native,
            }),
        }
    }
    /// Process-local identity.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }
    /// Owning device identity.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }
    /// Immutable creation descriptor.
    pub fn descriptor(&self) -> &QuerySetDescriptor {
        &self.inner.descriptor
    }
    pub(crate) fn native(&self) -> &dyn QuerySetBackend {
        self.inner.native.as_ref()
    }
}

impl fmt::Debug for QuerySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuerySet")
            .field("id", &self.id())
            .field("device", &self.device_identity())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Creates a query set after checking the enabled query family.
    pub fn create_query_set(&self, descriptor: &QuerySetDescriptor) -> RhiResult<QuerySet> {
        self.require_active()
            .map_err(|error| error.at("Device::create_query_set"))?;
        if descriptor.count == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a query set must contain at least one slot",
            )
            .at("Device::create_query_set"));
        }
        let feature = match descriptor.ty {
            QueryType::Occlusion => OptionalFeature::OcclusionQuery,
            QueryType::Timestamp => OptionalFeature::TimestampQuery,
            QueryType::PipelineStatistics(_) => OptionalFeature::PipelineStatisticsQuery,
        };
        if !self.capabilities().supports_feature(feature) {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this query type was not enabled on the device",
            )
            .at("Device::create_query_set"));
        }
        if let QueryType::PipelineStatistics(selection) = descriptor.ty {
            if selection.is_empty() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a pipeline-statistics query must select at least one counter",
                )
                .at("Device::create_query_set"));
            }
            if !self
                .capabilities()
                .pipeline_statistics()
                .contains(selection)
            {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "this device does not support every selected pipeline-statistics counter",
                )
                .at("Device::create_query_set"));
            }
        }
        let Some(max_queries) = self.capabilities().limit(LimitKey::MaxQueriesPerQuerySet) else {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device did not report a query-set capacity",
            )
            .at("Device::create_query_set"));
        };
        if max_queries == 0 {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device reported an invalid zero query-set capacity",
            )
            .at("Device::create_query_set"));
        }
        if u64::from(descriptor.count) > max_queries {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "query-set count {} exceeds this device's maximum {max_queries}",
                    descriptor.count
                ),
            )
            .at("Device::create_query_set"));
        }
        let native = self.native().create_query_set(descriptor)?;
        Ok(QuerySet::new(
            ObjectId::next(),
            self.identity(),
            descriptor.clone(),
            native,
        ))
    }
}

/// Validates that `index` names one slot in `set` and that the set is local.
pub(crate) fn validate_query(
    set: &QuerySet,
    index: u32,
    device: DeviceIdentity,
    operation: &'static str,
) -> RhiResult<()> {
    if set.device_identity() != device {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "query set belongs to a different device",
        )
        .with_object(set.id())
        .at(operation));
    }
    if index >= set.descriptor().count {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "query index {index} is outside query-set count {}",
                set.descriptor().count
            ),
        )
        .at(operation));
    }
    Ok(())
}
