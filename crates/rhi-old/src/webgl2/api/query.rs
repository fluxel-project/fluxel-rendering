//! Typed query lifetime and result contracts, including browser disjoint time.

use super::{GlError, GlFamilyApi, QueryId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlQueryResult {
    Pending,
    Available(u64),
    /// A WebGL GPU-disjoint interval; numerical timer data must be discarded.
    Disjoint,
    Unknown,
}

/// Typed allocation, destruction, and result observation.
///
/// Each call preflights ownership/lifecycle and object context. Implementors
/// reject a non-live slot before any provider side effect; prior-epoch IDs are
/// stale and never become live merely because their numeric name is reused.
pub(crate) trait GlQueryObjectsApi: GlFamilyApi {
    fn create_query(&mut self) -> Result<QueryId, GlError>;
    fn destroy_query(&mut self, query: QueryId) -> Result<(), GlError>;
    fn query_result(&mut self, query: QueryId) -> Result<GlQueryResult, GlError>;
}

/// Occlusion is a begin/end query domain.
pub(crate) trait GlOcclusionQueryApi: GlFamilyApi {
    fn begin_occlusion_query(&mut self, query: QueryId) -> Result<(), GlError>;
    fn end_occlusion_query(&mut self) -> Result<(), GlError>;
}

/// Elapsed time is a distinct begin/end query domain.
pub(crate) trait GlElapsedQueryApi: GlFamilyApi {
    fn begin_elapsed_query(&mut self, query: QueryId) -> Result<(), GlError>;
    fn end_elapsed_query(&mut self) -> Result<(), GlError>;
}

/// Timestamp counters write one query value and have no begin/end pair.
pub(crate) trait GlTimestampQueryApi: GlFamilyApi {
    fn query_timestamp(&mut self, query: QueryId) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::GlQueryResult;
    #[test]
    fn disjoint_is_not_a_zero_elapsed_time() {
        assert_ne!(GlQueryResult::Disjoint, GlQueryResult::Available(0));
    }
}
