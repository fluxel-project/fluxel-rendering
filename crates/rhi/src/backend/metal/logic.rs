//! Backend-private, platform-independent Metal lowering rules.
//!
//! This module deliberately has no Objective-C or Metal dependency.  It keeps
//! the small state machines that determine whether a native command buffer may
//! be committed, which binding indices a resource packet receives, and how a
//! drawable is retained.  Keeping those rules pure makes them executable on
//! non-Apple hosts and prevents an Objective-C callback from becoming the
//! authority for portable RHI state.

use core::fmt;

/// The result of preparing one command-buffer-sized batch during submit Phase A.
///
/// A plan is committable only when every batch is encoded.  In particular,
/// `Rejected` is never converted into a partially committed prefix: this is the
/// backend half of the public `submit(Err) => zero native work accepted`
/// contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BatchPreparation {
    /// Native objects were allocated and all commands were encoded successfully.
    Encoded,
    /// Allocation, validation, or encoding failed before commit.
    Rejected,
}

/// The immutable commit decision produced after Phase A.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CommitSchedule {
    batch_count: usize,
    committable: bool,
}

impl CommitSchedule {
    /// Builds the only legal Phase-B schedule from all Phase-A outcomes.
    pub(crate) fn from_phase_a(preparations: impl IntoIterator<Item = BatchPreparation>) -> Self {
        let mut batch_count = 0;
        let mut committable = true;
        for preparation in preparations {
            batch_count += 1;
            committable &= preparation == BatchPreparation::Encoded;
        }
        Self {
            batch_count,
            committable,
        }
    }

    /// Number of native command buffers prepared for this submission.
    pub(crate) const fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// Whether Phase B may call `commit` on any prepared command buffer.
    pub(crate) const fn may_commit(&self) -> bool {
        self.committable
    }

    /// Returns the commit order, or an empty iterator after any Phase-A failure.
    ///
    /// Metal command buffers belonging to one direct-queue plan preserve batch
    /// order.  Dependencies across future lanes are expressed by higher-level
    /// plan validation; this baseline intentionally has one queue.
    pub(crate) fn commit_order(&self) -> impl Iterator<Item = usize> {
        let end = if self.committable {
            self.batch_count
        } else {
            0
        };
        0..end
    }
}

/// Kinds that occupy separate Metal argument-table index namespaces.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum MetalBindingNamespace {
    Buffer,
    Texture,
    Sampler,
}

/// One logical resource field in a bind-group packet.
///
/// `logical_slot` is deliberately retained for deterministic diagnostics.  It
/// does not become a Metal index: Metal uses independent dense index spaces for
/// buffers, textures and samplers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BindingRequest {
    pub(crate) logical_slot: u32,
    pub(crate) namespace: MetalBindingNamespace,
}

/// One assigned Metal binding index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MetalBindingIndex {
    pub(crate) logical_slot: u32,
    pub(crate) namespace: MetalBindingNamespace,
    pub(crate) index: u32,
}

/// A deterministic binding-index plan for one pipeline stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingIndexPlan {
    assignments: Vec<MetalBindingIndex>,
    buffer_count: u32,
    texture_count: u32,
    sampler_count: u32,
}

/// A malformed logical packet cannot be mapped safely to Metal indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BindingPlanError {
    /// More than one resource attempted to occupy the same logical field.
    DuplicateLogicalSlot(u32),
}

impl fmt::Display for BindingPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLogicalSlot(slot) => {
                write!(f, "duplicate Metal binding logical slot {slot}")
            }
        }
    }
}

impl BindingIndexPlan {
    /// Assigns compact, stable indices to a canonicalized logical packet.
    ///
    /// Canonical layouts are normally already ordered by public validation, but
    /// sorting here keeps native lowering independent of caller iteration order.
    /// We reject duplicate logical slots even if their namespaces differ: one
    /// portable binding field owns exactly one resource class.
    pub(crate) fn build(
        requests: impl IntoIterator<Item = BindingRequest>,
    ) -> Result<Self, BindingPlanError> {
        let mut requests: Vec<_> = requests.into_iter().collect();
        requests.sort_unstable_by_key(|request| request.logical_slot);

        let mut assignments = Vec::with_capacity(requests.len());
        let mut previous_slot = None;
        let mut buffer_count = 0;
        let mut texture_count = 0;
        let mut sampler_count = 0;
        for request in requests {
            if previous_slot == Some(request.logical_slot) {
                return Err(BindingPlanError::DuplicateLogicalSlot(request.logical_slot));
            }
            previous_slot = Some(request.logical_slot);
            let index = match request.namespace {
                MetalBindingNamespace::Buffer => {
                    let value = buffer_count;
                    buffer_count += 1;
                    value
                }
                MetalBindingNamespace::Texture => {
                    let value = texture_count;
                    texture_count += 1;
                    value
                }
                MetalBindingNamespace::Sampler => {
                    let value = sampler_count;
                    sampler_count += 1;
                    value
                }
            };
            assignments.push(MetalBindingIndex {
                logical_slot: request.logical_slot,
                namespace: request.namespace,
                index,
            });
        }
        Ok(Self {
            assignments,
            buffer_count,
            texture_count,
            sampler_count,
        })
    }

    pub(crate) fn assignments(&self) -> &[MetalBindingIndex] {
        &self.assignments
    }

    pub(crate) const fn count(&self, namespace: MetalBindingNamespace) -> u32 {
        match namespace {
            MetalBindingNamespace::Buffer => self.buffer_count,
            MetalBindingNamespace::Texture => self.texture_count,
            MetalBindingNamespace::Sampler => self.sampler_count,
        }
    }
}

/// Backend-private state of a `CAMetalDrawable` lease.
///
/// The drawable is retained by the acquired frame from `nextDrawable` through
/// either a scheduled present or abandonment.  `presentDrawable` is scheduled
/// before the command buffer is committed; completed presentation is reported
/// separately by the public present receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrawableState {
    Idle,
    Acquired,
    Encoded,
    PresentScheduled,
    Committed,
    Abandoned,
    Lost,
}

/// Illegal drawable lifecycle transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DrawableTransitionError {
    pub(crate) state: DrawableState,
    pub(crate) operation: &'static str,
}

impl fmt::Display for DrawableTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot {} drawable while it is {:?}",
            self.operation, self.state
        )
    }
}

/// Pure state controller used by the Objective-C presentation wrapper.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DrawableLeaseState {
    state: DrawableState,
}

impl Default for DrawableLeaseState {
    fn default() -> Self {
        Self {
            state: DrawableState::Idle,
        }
    }
}

impl DrawableLeaseState {
    pub(crate) const fn state(&self) -> DrawableState {
        self.state
    }

    pub(crate) fn acquire(&mut self) -> Result<(), DrawableTransitionError> {
        self.transition(DrawableState::Idle, DrawableState::Acquired, "acquire")
    }

    pub(crate) fn encode(&mut self) -> Result<(), DrawableTransitionError> {
        self.transition(DrawableState::Acquired, DrawableState::Encoded, "encode")
    }

    pub(crate) fn schedule_present(&mut self) -> Result<(), DrawableTransitionError> {
        self.transition(
            DrawableState::Encoded,
            DrawableState::PresentScheduled,
            "schedule present",
        )
    }

    pub(crate) fn commit(&mut self) -> Result<(), DrawableTransitionError> {
        match self.state {
            DrawableState::Encoded | DrawableState::PresentScheduled => {
                self.state = DrawableState::Committed;
                Ok(())
            }
            state => Err(DrawableTransitionError {
                state,
                operation: "commit",
            }),
        }
    }

    /// Discards an acquired drawable that has not been accepted by a command
    /// buffer.  Once committed, Metal owns its presentation lifetime.
    pub(crate) fn abandon(&mut self) -> Result<(), DrawableTransitionError> {
        match self.state {
            DrawableState::Acquired | DrawableState::Encoded => {
                self.state = DrawableState::Abandoned;
                Ok(())
            }
            state => Err(DrawableTransitionError {
                state,
                operation: "abandon",
            }),
        }
    }

    /// Device loss is terminal and supersedes every pending drawable state.
    pub(crate) fn mark_lost(&mut self) {
        self.state = DrawableState::Lost;
    }

    fn transition(
        &mut self,
        expected: DrawableState,
        next: DrawableState,
        operation: &'static str,
    ) -> Result<(), DrawableTransitionError> {
        if self.state != expected {
            return Err(DrawableTransitionError {
                state: self.state,
                operation,
            });
        }
        self.state = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_a_failure_produces_no_commit_prefix() {
        let schedule = CommitSchedule::from_phase_a([
            BatchPreparation::Encoded,
            BatchPreparation::Rejected,
            BatchPreparation::Encoded,
        ]);
        assert_eq!(schedule.batch_count(), 3);
        assert!(!schedule.may_commit());
        assert_eq!(
            schedule.commit_order().collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn successful_phase_a_commits_every_batch_in_plan_order() {
        let schedule = CommitSchedule::from_phase_a([BatchPreparation::Encoded; 3]);
        assert!(schedule.may_commit());
        assert_eq!(schedule.commit_order().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn binding_plan_uses_independent_dense_metal_namespaces() {
        let plan = BindingIndexPlan::build([
            BindingRequest {
                logical_slot: 8,
                namespace: MetalBindingNamespace::Sampler,
            },
            BindingRequest {
                logical_slot: 2,
                namespace: MetalBindingNamespace::Texture,
            },
            BindingRequest {
                logical_slot: 1,
                namespace: MetalBindingNamespace::Buffer,
            },
            BindingRequest {
                logical_slot: 5,
                namespace: MetalBindingNamespace::Texture,
            },
        ])
        .unwrap();
        assert_eq!(
            plan.assignments(),
            &[
                MetalBindingIndex {
                    logical_slot: 1,
                    namespace: MetalBindingNamespace::Buffer,
                    index: 0
                },
                MetalBindingIndex {
                    logical_slot: 2,
                    namespace: MetalBindingNamespace::Texture,
                    index: 0
                },
                MetalBindingIndex {
                    logical_slot: 5,
                    namespace: MetalBindingNamespace::Texture,
                    index: 1
                },
                MetalBindingIndex {
                    logical_slot: 8,
                    namespace: MetalBindingNamespace::Sampler,
                    index: 0
                },
            ]
        );
        assert_eq!(plan.count(MetalBindingNamespace::Buffer), 1);
        assert_eq!(plan.count(MetalBindingNamespace::Texture), 2);
        assert_eq!(plan.count(MetalBindingNamespace::Sampler), 1);
    }

    #[test]
    fn binding_plan_refuses_ambiguous_logical_fields() {
        let error = BindingIndexPlan::build([
            BindingRequest {
                logical_slot: 4,
                namespace: MetalBindingNamespace::Buffer,
            },
            BindingRequest {
                logical_slot: 4,
                namespace: MetalBindingNamespace::Texture,
            },
        ])
        .unwrap_err();
        assert_eq!(error, BindingPlanError::DuplicateLogicalSlot(4));
    }

    #[test]
    fn drawable_is_scheduled_before_commit_and_cannot_be_abandoned_afterward() {
        let mut drawable = DrawableLeaseState::default();
        drawable.acquire().unwrap();
        drawable.encode().unwrap();
        drawable.schedule_present().unwrap();
        drawable.commit().unwrap();
        assert_eq!(drawable.state(), DrawableState::Committed);
        assert_eq!(
            drawable.abandon().unwrap_err().state,
            DrawableState::Committed
        );
    }

    #[test]
    fn loss_is_terminal_for_an_acquired_drawable() {
        let mut drawable = DrawableLeaseState::default();
        drawable.acquire().unwrap();
        drawable.mark_lost();
        assert_eq!(drawable.state(), DrawableState::Lost);
        assert_eq!(drawable.encode().unwrap_err().state, DrawableState::Lost);
    }
}
