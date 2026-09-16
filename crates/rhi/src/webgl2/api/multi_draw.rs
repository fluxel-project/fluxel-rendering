//! WebGPU-aligned batch-draw vocabulary and its single-draw fallback rule.
//!
//! A batch is a nonempty run of draws that share one shape, so a provider has
//! exactly two correct ways to execute it: issue the whole batch as one command
//! when its context proved the combined command, or decompose it into the very
//! single draws the raster domain already issues. The batch therefore carries
//! the same fields a single draw carries, in submission order, and it never
//! reinterprets them (no merge of adjacent draws, no reordering of offsets).
//!
//! This module owns the batch shape and the rules decidable from the batch
//! alone. Every rule here is also enforced by the single-draw path, so a batch
//! rejected here is one the decomposed path would reject too, and a caller
//! never sees a batch that only one of the two routes can execute. Checks that
//! need provider state (a live index binding, its byte span, a buffer's
//! allocation) stay in the provider, which prepares every draw of a batch
//! before issuing the first one so a rejection cannot leave a partial
//! submission behind. Choosing the command, binding buffers, and running either
//! route are also the provider's.

use super::{GlDrawCommand, GlError, GlIndexedDraw, GlNonIndexedDraw, GlRasterCommandApi};

/// Why a batch cannot be issued as one command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlMultiDrawValidationError {
    /// An empty batch would be a driver no-op that hides the caller's mistake.
    EmptyBatch,
    /// The batch mixes indexed and non-indexed draws, which no one command
    /// covers: the two shapes need different per-draw parameter slices.
    MixedDrawShapes,
    /// One draw would read no vertex or no index.
    ZeroDrawCount,
    /// One draw would run no instance.
    ZeroInstanceCount,
    /// One per-draw value does not fit the signed 32-bit slot a per-draw
    /// parameter slice holds, so the combined command would silently truncate
    /// it into a different draw.
    ValueExceedsCommandRange,
}

/// A batch of draw payloads submitted as one command.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlMultiDraw {
    draws: Vec<GlDrawCommand>,
}

impl GlMultiDraw {
    /// Builds a batch, rejecting one this contract cannot issue as written.
    pub(crate) fn new(draws: Vec<GlDrawCommand>) -> Result<Self, GlMultiDrawValidationError> {
        let batch = Self { draws };
        batch.validate()?;
        Ok(batch)
    }

    /// Returns the draws in submission order.
    pub(crate) fn draws(&self) -> &[GlDrawCommand] {
        &self.draws
    }

    /// Returns whether the batch needs one instance count per draw.
    ///
    /// A batch whose draws all run a single instance is issued by the plain
    /// combined command; any draw running several instances needs the variant
    /// that carries one instance count per draw, which carries the
    /// single-instance draws of the same batch unchanged.
    pub(crate) fn needs_instance_counts(&self) -> bool {
        self.draws.iter().any(|draw| match draw {
            GlDrawCommand::NonIndexed(draw) => draw.instance_count != 1,
            GlDrawCommand::Indexed(draw) => draw.instance_count != 1,
        })
    }

    fn validate(&self) -> Result<(), GlMultiDrawValidationError> {
        let Some(first) = self.draws.first() else {
            return Err(GlMultiDrawValidationError::EmptyBatch);
        };
        let indexed = matches!(first, GlDrawCommand::Indexed(_));
        for draw in &self.draws {
            match draw {
                GlDrawCommand::NonIndexed(draw) if !indexed => validate_non_indexed(draw)?,
                GlDrawCommand::Indexed(draw) if indexed => validate_indexed(draw)?,
                _ => return Err(GlMultiDrawValidationError::MixedDrawShapes),
            }
        }
        Ok(())
    }
}

/// One non-indexed draw's own rules.
///
/// `first_vertex`, the draw count, and the instance count all enter a per-draw
/// parameter slice, so each must survive the 32-bit signed conversion intact.
fn validate_non_indexed(draw: &GlNonIndexedDraw) -> Result<(), GlMultiDrawValidationError> {
    if draw.vertex_count == 0 {
        return Err(GlMultiDrawValidationError::ZeroDrawCount);
    }
    if draw.instance_count == 0 {
        return Err(GlMultiDrawValidationError::ZeroInstanceCount);
    }
    if i32::try_from(draw.first_vertex).is_err()
        || i32::try_from(draw.vertex_count).is_err()
        || i32::try_from(draw.instance_count).is_err()
    {
        return Err(GlMultiDrawValidationError::ValueExceedsCommandRange);
    }
    Ok(())
}

/// One indexed draw's own rules.
///
/// `first_index` is deliberately not range-checked here: it never enters a
/// per-draw parameter slice, and the byte offset derived from it depends on the
/// bound index format, which is provider state. The provider applies the same
/// offset rule to a batch as it applies to a single draw.
fn validate_indexed(draw: &GlIndexedDraw) -> Result<(), GlMultiDrawValidationError> {
    if draw.index_count == 0 {
        return Err(GlMultiDrawValidationError::ZeroDrawCount);
    }
    if draw.instance_count == 0 {
        return Err(GlMultiDrawValidationError::ZeroInstanceCount);
    }
    if i32::try_from(draw.index_count).is_err() || i32::try_from(draw.instance_count).is_err() {
        return Err(GlMultiDrawValidationError::ValueExceedsCommandRange);
    }
    Ok(())
}

/// Issues a validated batch through the single-draw path, in submission order.
///
/// This is the one fallback every provider that cannot issue a combined
/// command uses, so the batch renders exactly what the same draws issued one by
/// one render and a rejected draw reports its own single-draw operation. A
/// provider must have readied the whole batch before calling it, so a rejection
/// cannot leave a partial submission behind.
pub(crate) fn issue_single_draws(
    provider: &mut impl GlRasterCommandApi,
    command: &GlMultiDraw,
) -> Result<(), GlError> {
    for draw in command.draws() {
        provider.draw_raster(*draw)?;
    }
    Ok(())
}

/// Executable batch-draw domain.
///
/// A provider that cannot issue the combined command decomposes the batch
/// through `GlRasterCommandApi::draw_raster`, one validated draw at a time, so
/// a caller never branches on which route its context proved and a batch
/// produces the same pixels on both. Every provider readies the whole batch
/// before issuing its first draw, so a rejected batch leaves no partial
/// submission behind.
pub(crate) trait GlMultiDrawApi: GlRasterCommandApi {
    /// Issues every draw of one batch in submission order.
    fn multi_draw(&mut self, command: &GlMultiDraw) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{
        GlDrawCommand, GlIndexedDraw, GlMultiDraw, GlMultiDrawValidationError, GlNonIndexedDraw,
    };

    fn non_indexed(first_vertex: u32, vertex_count: u32, instance_count: u32) -> GlDrawCommand {
        GlDrawCommand::NonIndexed(GlNonIndexedDraw {
            first_vertex,
            vertex_count,
            instance_count,
        })
    }

    fn indexed(first_index: u32, index_count: u32, instance_count: u32) -> GlDrawCommand {
        GlDrawCommand::Indexed(GlIndexedDraw {
            first_index,
            index_count,
            instance_count,
        })
    }

    #[test]
    fn a_batch_with_no_draw_is_not_a_batch() {
        assert_eq!(
            GlMultiDraw::new(vec![]).err(),
            Some(GlMultiDrawValidationError::EmptyBatch)
        );
    }

    /// One command carries one shape of per-draw parameter slice, so a mixed
    /// batch is refused instead of being silently split into two submissions.
    #[test]
    fn a_batch_mixing_shapes_is_rejected() {
        assert_eq!(
            GlMultiDraw::new(vec![non_indexed(0, 3, 1), indexed(0, 3, 1)]).err(),
            Some(GlMultiDrawValidationError::MixedDrawShapes)
        );
        assert_eq!(
            GlMultiDraw::new(vec![indexed(0, 3, 1), non_indexed(0, 3, 1)]).err(),
            Some(GlMultiDrawValidationError::MixedDrawShapes)
        );
    }

    /// A draw that reads nothing or runs nothing is refused for every draw of
    /// the batch, not only the first one, because a later zero would otherwise
    /// be found halfway through a decomposed submission.
    #[test]
    fn every_draw_of_the_batch_must_be_a_real_draw() {
        assert_eq!(
            GlMultiDraw::new(vec![non_indexed(0, 3, 1), non_indexed(3, 0, 1)]).err(),
            Some(GlMultiDrawValidationError::ZeroDrawCount)
        );
        assert_eq!(
            GlMultiDraw::new(vec![non_indexed(0, 3, 1), non_indexed(3, 3, 0)]).err(),
            Some(GlMultiDrawValidationError::ZeroInstanceCount)
        );
        assert_eq!(
            GlMultiDraw::new(vec![indexed(0, 6, 1), indexed(6, 0, 1)]).err(),
            Some(GlMultiDrawValidationError::ZeroDrawCount)
        );
        assert_eq!(
            GlMultiDraw::new(vec![indexed(0, 6, 1), indexed(6, 6, 0)]).err(),
            Some(GlMultiDrawValidationError::ZeroInstanceCount)
        );
    }

    #[test]
    fn per_draw_values_must_survive_one_parameter_slot() {
        assert_eq!(
            GlMultiDraw::new(vec![non_indexed(u32::MAX, 3, 1)]).err(),
            Some(GlMultiDrawValidationError::ValueExceedsCommandRange)
        );
        assert_eq!(
            GlMultiDraw::new(vec![non_indexed(0, u32::MAX, 1)]).err(),
            Some(GlMultiDrawValidationError::ValueExceedsCommandRange)
        );
        assert_eq!(
            GlMultiDraw::new(vec![indexed(0, 6, u32::MAX)]).err(),
            Some(GlMultiDrawValidationError::ValueExceedsCommandRange)
        );
        // The exact boundary of the signed slot is still one draw.
        assert!(GlMultiDraw::new(vec![non_indexed(i32::MAX as u32, i32::MAX as u32, 1)]).is_ok());
    }

    /// `first_index` is not a per-draw parameter: it is only ever turned into a
    /// byte offset against the bound index format, so its range belongs to the
    /// provider that knows the format.
    #[test]
    fn an_index_offset_beyond_the_parameter_slot_stays_the_providers_rule() {
        assert!(GlMultiDraw::new(vec![indexed(u32::MAX, 1, 1)]).is_ok());
    }

    #[test]
    fn a_batch_keeps_its_draws_in_submission_order() {
        let batch = GlMultiDraw::new(vec![indexed(6, 3, 1), indexed(0, 3, 1), indexed(9, 3, 1)])
            .expect("a homogeneous batch");
        assert_eq!(batch.draws().len(), 3);
        assert_eq!(batch.draws()[0], indexed(6, 3, 1));
        assert_eq!(batch.draws()[1], indexed(0, 3, 1));
        assert_eq!(batch.draws()[2], indexed(9, 3, 1));
    }

    /// The plain combined command carries no instance count, so it is chosen
    /// only when every draw already runs exactly one instance.
    #[test]
    fn instance_counts_are_needed_exactly_when_a_draw_runs_several_instances() {
        let single = |draws| GlMultiDraw::new(draws).expect("a valid batch");
        assert!(!single(vec![non_indexed(0, 3, 1), non_indexed(3, 3, 1)]).needs_instance_counts());
        assert!(single(vec![non_indexed(0, 3, 1), non_indexed(3, 3, 2)]).needs_instance_counts());
        assert!(!single(vec![indexed(0, 6, 1)]).needs_instance_counts());
        assert!(single(vec![indexed(0, 6, 1), indexed(6, 6, 4)]).needs_instance_counts());
    }
}
