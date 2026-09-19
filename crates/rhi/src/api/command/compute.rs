//! `ComputeScope` and the compute command set (specification section 33).
//!
//! This module owns the dispatch-shaped half of a recording: a scope with no
//! attachments, a pipeline, bind groups, and workgroup counts.
//!
//! # What this module does not own
//!
//! - Whether the device can compute at all. Section 33 gates the whole chapter on
//!   [`OptionalFeature::Compute`] and that is read from the snapshot the device
//!   interned, so [`CommandRecorder::begin_compute`] refuses rather than lowering
//!   when it is absent.
//! - Whether a workgroup count is within the device's
//!   `MaxComputeWorkgroupsPerDimension`. Also a device fact, read from the same
//!   snapshot, and also a refusal from [`ComputeScope::dispatch`] rather than a
//!   guessed bound.
//! - The workgroup *shape* rules — that a shader's `@workgroup_size` matches what a
//!   pipeline declares. Those belong to the shader artifact's own validator
//!   (section 19.7) and were already applied when the pipeline was created.
//!
//! # The invariant this module enforces
//!
//! **A dispatch is reachable only through a pipeline and a complete set of the
//! groups that pipeline's interface uses.** Both are portable facts, so both are
//! decided here: a dispatch that reached a backend without them would be a
//! backend discovering something the portable layer could have refused, which
//! section 4 forbids.

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::record::{BoundGroup, ComputeBegin, ComputeDispatch, RecordedPayload};
use crate::api::command::uses::{
    bound_group_uses, require_valid_dynamic_offsets, validate_bound_groups,
};
use crate::api::command::{CommandRecorder, RecorderPhase, require_device};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::graph_bridge::ResourceUse;
use crate::api::identity::Label;
use crate::api::pipeline::ComputePipeline;
use crate::api::platform::requirements::{LimitKey, OptionalFeature};

/// The domain bit every compute command contributes.
const COMPUTE_DOMAIN: crate::api::submission::LaneWorkDomains =
    crate::api::submission::LaneWorkDomains::COMPUTE;

/// Everything a caller states about one compute scope before it opens.
///
/// A descriptor rather than a bare `begin_compute()`, because a compute pass has no
/// attachment to carry a label and section 29.2's state machine still puts a
/// `ComputeScopeOpen` node in the recording. Without a descriptor that node would be
/// the one unnameable step in a capture or a log.
#[non_exhaustive]
#[derive(Clone)]
pub struct ComputeScopeDescriptor {
    /// Diagnostic label. It also establishes the scope's diagnostics nesting, which
    /// is separate from the debug-group stack (section 36).
    pub label: Label,
}

impl ComputeScopeDescriptor {
    /// States a scope with no label.
    pub fn new() -> Self {
        Self {
            label: Label::default(),
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}

impl Default for ComputeScopeDescriptor {
    /// The same unlabelled descriptor as [`ComputeScopeDescriptor::new`].
    ///
    /// `Default` is added because an unlabelled scope is a legal descriptor, not an
    /// incomplete one, and clippy's `new_without_default` lint is right that the two
    /// should agree.
    fn default() -> Self {
        Self::new()
    }
}

impl CommandRecorder {
    /// Opens a compute scope.
    ///
    /// Section 33's gate comes first: a device that has not enabled
    /// [`OptionalFeature::Compute`] must be refused with
    /// [`RhiErrorKind::Unsupported`] rather than lowered. The recorder's open-state
    /// check is taken first, because it is the one thing here that is decidable
    /// without the device and section 4 puts a portable refusal ahead of a
    /// capability question.
    ///
    /// [`OptionalFeature::Compute`]: crate::api::platform::requirements::OptionalFeature::Compute
    pub fn begin_compute<'a>(
        &'a mut self,
        desc: &ComputeScopeDescriptor,
    ) -> RhiResult<ComputeScope<'a>> {
        self.require_open("begin_compute")?;

        // Section 33's gate, and it is a refusal rather than a lowering: a device
        // that has not enabled the feature has no compute path at all, so
        // recording one would produce a recording that cannot be executed. The
        // answer comes from the snapshot the device interned, which is why the
        // recorder holds it — section 7.2 makes an enabled contract immutable, so
        // this cannot disagree with what the device reported at construction.
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::Compute)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device has not enabled compute, so no compute scope can be opened \
                 on it",
            )
            .at("CommandRecorder::begin_compute"));
        }

        let begin = ComputeBegin {
            label: desc.label.clone(),
        };
        self.record_command(
            RecordedPayload::ComputeBegin(begin),
            Vec::new(),
            COMPUTE_DOMAIN,
        );
        self.set_phase(RecorderPhase::ComputeScopeOpen);

        Ok(ComputeScope {
            recorder: self,
            pipeline: None,
            groups: Vec::new(),
            debug_stack: Vec::new(),
            ended: false,
        })
    }
}

/// A compute scope, open until it is ended.
///
/// The lifetime is the recorder's, which is section 29.2's "Rust mutable borrow
/// prevents the scope from directly issuing another type of command to the Recorder
/// while it is alive". As in the raster scope, the borrow is a real
/// `&mut CommandRecorder` rather than a `PhantomData`: a marker cannot record the
/// dispatch it exists to describe, and the borrow the specification documents is
/// the same one this field holds.
pub struct ComputeScope<'a> {
    /// The recorder this scope is writing into.
    recorder: &'a mut CommandRecorder,
    /// The bound pipeline, if any.
    pipeline: Option<ComputePipeline>,
    /// The bound bind groups, by index.
    groups: Vec<BoundGroup>,
    /// This scope's own debug-group stack, independent of the recorder's.
    debug_stack: Vec<String>,
    /// Whether `end` completed, which is what decides whether `Drop` poisons.
    ended: bool,
}

impl ComputeScope<'_> {
    /// Binds a compute pipeline.
    ///
    /// Only the device check, because a compute scope has no attachment set for a
    /// pipeline to be compatible *with*: the interface a compute pipeline declares
    /// is checked against the bound groups at dispatch, where the groups exist.
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) -> RhiResult<()> {
        require_device(
            pipeline.device_identity(),
            self.recorder.device_identity(),
            "the pipeline",
        )?;
        self.pipeline = Some(pipeline.clone());
        Ok(())
    }

    /// Binds a bind group at an index.
    ///
    /// The same rule as the raster path's, and deliberately the same function for
    /// the offset check: section 33 and section 32.3 state "dynamic offsets valid"
    /// once each, and two implementations of one sentence is how they drift. What
    /// that sentence costs is section 22.4's list, which
    /// `require_valid_dynamic_offsets` owns once for both scopes.
    pub fn set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        require_device(
            group.device_identity(),
            self.recorder.device_identity(),
            "the bind group",
        )?;
        require_device(
            group.layout().device_identity(),
            self.recorder.device_identity(),
            "the bind group's layout",
        )?;
        require_valid_dynamic_offsets(index, group, dynamic_offsets)?;

        let bound = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        match self
            .groups
            .iter_mut()
            .find(|existing| existing.index == index)
        {
            Some(existing) => *existing = bound,
            None => self.groups.push(bound),
        }
        Ok(())
    }

    /// Dispatches a workgroup grid.
    ///
    /// Section 33's validation list, in its own order: a pipeline must be bound, the
    /// groups its interface uses must be bound and compatible, and each count must
    /// be within the device's `MaxComputeWorkgroupsPerDimension`. A count of zero is
    /// legal and means what it says — a dispatch that launches nothing.
    ///
    /// The first two are portable and are decided here. The third is a device limit,
    /// so it is answered from the snapshot the device interned rather than from an
    /// assumed value — section 4 forbids inventing a bound the device has not
    /// stated.
    ///
    /// The limit is checked *before* the command is recorded, and the order is the
    /// contract rather than a preference: a dispatch that is refused must leave the
    /// recording exactly as it found it, or `finish` would hand back work containing
    /// a dispatch the caller was told did not happen.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> RhiResult<()> {
        let pipeline = self.bound_pipeline()?;
        validate_bound_groups(pipeline.interface(), &self.groups)?;

        // Section 33's ceiling. `None` is a legal device answer meaning the contract
        // states no such limit, which is not the same as zero — treating a missing
        // limit as a refusal would invent a bound, and treating it as unlimited is
        // what "the contract defines none" says.
        if let Some(max) = self
            .recorder
            .capabilities()
            .limit(LimitKey::MaxComputeWorkgroupsPerDimension)
        {
            for (axis, count) in [("x", x), ("y", y), ("z", z)] {
                if u64::from(count) > max {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        format!(
                            "a dispatch of {count} workgroups on the {axis} axis exceeds this \
                             device's limit of {max}"
                        ),
                    )
                    .at("ComputeScope::dispatch"));
                }
            }
        }

        let uses = self.dispatch_uses()?;
        let dispatch = ComputeDispatch {
            pipeline,
            groups: self.groups.clone(),
            workgroups: (x, y, z),
        };
        self.recorder.record_command(
            RecordedPayload::ComputeDispatch(Box::new(dispatch)),
            uses,
            COMPUTE_DOMAIN,
        );

        Ok(())
    }

    /// Pushes a label onto this scope's own debug-group stack.
    ///
    /// A separate stack from the recorder's, for the same reason the raster scope's
    /// is: the recorder's stack describes commands outside any pass, this one
    /// describes the pass's interior, and the two nest in the recording but never in
    /// each other.
    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.debug_stack.push(label.to_owned());
        self.recorder.record_command(
            RecordedPayload::DebugPush(Label(Some(label.to_owned()))),
            Vec::new(),
            COMPUTE_DOMAIN,
        );
        Ok(())
    }

    /// Pops this scope's own debug-group stack.
    ///
    /// An unmatched pop is a parameter error and poisons nothing: section 29.3 keeps
    /// parameter refusal out of the poisoning category.
    pub fn pop_debug_group(&mut self) -> RhiResult<()> {
        if self.debug_stack.pop().is_none() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "pop_debug_group has no matching push_debug_group on this compute scope",
            ));
        }
        self.recorder
            .record_command(RecordedPayload::DebugPop, Vec::new(), COMPUTE_DOMAIN);
        Ok(())
    }

    /// Inserts a marker without changing the stack.
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.recorder.record_command(
            RecordedPayload::DebugMarker(Label(Some(label.to_owned()))),
            Vec::new(),
            COMPUTE_DOMAIN,
        );
        Ok(())
    }

    /// Ends the scope and returns the recorder to the open state.
    ///
    /// A non-empty debug-group stack refuses the end, and the scope is deliberately
    /// left unterminated: section 29.3 puts a failed scope finalization in the
    /// poisoning category, and the `Drop` that follows this refusal carries it out.
    pub fn end(mut self) -> RhiResult<()> {
        if !self.debug_stack.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "this compute scope still has {} debug group(s) open at end()",
                    self.debug_stack.len()
                ),
            ));
        }

        self.recorder
            .record_command(RecordedPayload::ComputeEnd, Vec::new(), COMPUTE_DOMAIN);
        self.recorder.set_phase(RecorderPhase::Open);
        self.ended = true;
        Ok(())
    }

    /// The pipeline bound at this point, or a refusal.
    fn bound_pipeline(&self) -> RhiResult<ComputePipeline> {
        match &self.pipeline {
            Some(pipeline) => Ok(pipeline.clone()),
            None => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a dispatch needs a pipeline, and none is bound in this compute scope",
            )),
        }
    }

    /// The uses a dispatch produces.
    ///
    /// Section 33's table, applied through
    /// [`bound_group_uses`](crate::api::command::uses::bound_group_uses) so that the
    /// compute and raster paths cannot disagree: a `StorageBuffer { ReadWrite }`
    /// costs `SHADER_READ | SHADER_WRITE` in both, and a sampler costs nothing in
    /// either. A compute pass has no attachment and no vertex fetch, so the bound
    /// groups are the whole of its actual use.
    fn dispatch_uses(&self) -> RhiResult<Vec<ResourceUse>> {
        let mut uses = Vec::new();
        for group in &self.groups {
            uses.extend(bound_group_uses(&group.group)?);
        }
        Ok(uses)
    }
}

impl Drop for ComputeScope<'_> {
    /// Poisons the recorder when a scope did not reach `end`.
    ///
    /// Section 29.2's "drop without end -> Poisoned", and section 29.2's "Drop does
    /// not perform a backend finalize that may fail": with no native encoder in the
    /// recorder there is no finalize to attempt, so marking the recording unusable
    /// is the only thing this may do.
    fn drop(&mut self) {
        if !self.ended {
            self.recorder
                .poison("a compute scope was dropped without a successful end()");
        }
    }
}
