//! The submission plan builder and the validated plan it produces.
//!
//! This module owns rhi-design sections 39 and 40, and the plan-closure half of
//! section 45.3.
//!
//! # What a plan is
//!
//! A plan is a validated ordering: batches of recorded work on logical lanes,
//! the happens-before edges between them, and the frames they present. It names
//! no native queue, fence, semaphore, event, or timeline value.
//!
//! # Why the builder is the only way to make one
//!
//! `PlanPoint` has no public constructor, so a point from one builder cannot be
//! forged into another. The builder is device-scoped and holds the device's
//! declared submission capability set, which is what makes "is there a
//! happens-before route between these two lanes" answerable *before* a backend
//! is asked to lower anything.
//!
//! # Why validation happens in `build`
//!
//! `add_batch` and `add_dependency` answer local questions immediately, because a
//! caller can act on them. The whole-plan questions — cycles, unordered write
//! hazards, frame closure — can only be answered once the plan is complete, so
//! they run in `build` and refuse before any native submission exists.
//!
//! # What it deliberately does not own
//!
//! Native synchronization. The plan records that an edge exists and that the
//! device's declared routes can prove it; lowering that edge to a barrier, a
//! semaphore wait, or a single ordered domain is the backend's job. Nor does it
//! own completion: `build` produces an ordering, and `Device::submit` produces a
//! receipt.

use std::collections::BTreeMap;

use super::super::capability::EnabledCapabilities;
use super::super::command::RecordedWork;
use super::super::format::{LaneDependencyRoute, SubmissionCapabilities, SubmissionLaneId};
use super::super::graph_bridge::ResourceUse;
use super::super::platform::{Device, DeviceIdentity, RhiError, RhiErrorKind, RhiResult};
use super::super::presentation::{
    AcquiredFrame, AcquiredFrameId, AcquiredFrameState, PresentPlanId,
};
use super::{
    CompletionPoint, PlanPoint, SubmissionBatchId, SubmissionPlanId, next_plan_object_id,
};

/// One batch of recorded work on one logical lane.
#[derive(Debug)]
pub struct PlannedBatch {
    point: PlanPoint,
    lane: SubmissionLaneId,
    work: Vec<RecordedWork>,
}

impl PlannedBatch {
    /// This batch's point in the plan.
    pub fn point(&self) -> PlanPoint {
        self.point
    }

    /// The logical lane the batch runs on.
    pub fn lane(&self) -> SubmissionLaneId {
        self.lane
    }

    /// The recorded work, in logical order.
    pub fn work(&self) -> &[RecordedWork] {
        &self.work
    }
}

/// One frame this plan presents, and the point it presents after.
#[derive(Debug)]
pub struct PlannedPresent {
    id: PresentPlanId,
    frame: AcquiredFrame,
    after: PlanPoint,
}

impl PlannedPresent {
    /// This present's identity inside the plan.
    pub fn id(&self) -> PresentPlanId {
        self.id
    }

    /// The frame being presented.
    pub fn frame_id(&self) -> AcquiredFrameId {
        self.frame.id()
    }

    /// The point the present follows.
    pub fn after(&self) -> PlanPoint {
        self.after
    }
}

/// A validated submission plan.
///
/// It owns the frames it presents, so dropping it without submitting abandons
/// them through the ordinary frame drop path rather than leaving the
/// presentation system holding an image forever.
#[derive(Debug)]
pub struct SubmissionPlan {
    id: SubmissionPlanId,
    device: DeviceIdentity,
    batches: Vec<PlannedBatch>,
    dependencies: Vec<(PlanPoint, PlanPoint)>,
    external: Vec<(CompletionPoint, PlanPoint)>,
    presents: Vec<PlannedPresent>,
}

impl SubmissionPlan {
    /// This plan's identity.
    pub fn id(&self) -> SubmissionPlanId {
        self.id
    }

    /// The device identity this plan belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The plan's batches, in insertion order.
    pub fn batches(&self) -> &[PlannedBatch] {
        &self.batches
    }

    /// The explicit happens-before edges inside the plan.
    pub fn dependencies(&self) -> &[(PlanPoint, PlanPoint)] {
        &self.dependencies
    }

    /// The happens-before edges from previously submitted work.
    pub fn external_dependencies(&self) -> &[(CompletionPoint, PlanPoint)] {
        &self.external
    }

    /// The frames this plan presents.
    pub fn presents(&self) -> &[PlannedPresent] {
        &self.presents
    }
}

/// One node in the plan's ordering graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Node {
    Batch(usize),
    Present(usize),
}

/// Builds a validated [`SubmissionPlan`].
///
/// The builder owns every frame it consumes, so a builder dropped without
/// `build` abandons those frames instead of submitting or leaking them.
pub struct SubmissionPlanBuilder {
    id: SubmissionPlanId,
    device: DeviceIdentity,
    submission: SubmissionCapabilities,
    batches: Vec<PlannedBatch>,
    dependencies: Vec<(PlanPoint, PlanPoint)>,
    external: Vec<(CompletionPoint, PlanPoint)>,
    presents: Vec<PlannedPresent>,
    /// The lane each batch ran on, in insertion order; the implicit
    /// same-lane ordering is derived from this rather than stored as edges.
    lanes: Vec<SubmissionLaneId>,
}

impl SubmissionPlanBuilder {
    /// A builder for a new plan on `device`.
    pub fn new(device: &Device) -> Self {
        let device_identity = device.identity();
        Self::with_capabilities(device_identity, device.capabilities())
    }

    /// A builder over a device identity and its declared submission facts.
    ///
    /// This is the entry point a device implementation uses: it already holds
    /// its own identity and enabled capability set and must not have to rebuild
    /// a public `Device` handle to plan against them.
    pub(crate) fn with_capabilities(
        device: DeviceIdentity,
        capabilities: &EnabledCapabilities,
    ) -> Self {
        Self {
            id: SubmissionPlanId::new(device, next_plan_object_id().as_u64()),
            device,
            submission: capabilities.submission().clone(),
            batches: Vec::new(),
            dependencies: Vec::new(),
            external: Vec::new(),
            presents: Vec::new(),
            lanes: Vec::new(),
        }
    }

    /// This plan's identity, available before it is built.
    pub fn id(&self) -> SubmissionPlanId {
        self.id
    }

    /// Adds a logical ordered-lane batch.
    ///
    /// The vector's order is the batch's logical work order. It is not a memory
    /// barrier: the RHI still generates the synchronization the actual uses
    /// require, because submission order and memory dependency are separate
    /// concepts on every backend this library targets.
    pub fn add_batch(
        &mut self,
        lane: SubmissionLaneId,
        work: Vec<RecordedWork>,
    ) -> RhiResult<PlanPoint> {
        if work.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a batch must contain at least one recorded work",
            ));
        }
        let Some(lane_info) = self.submission.lane(lane) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("lane {} does not belong to this device", lane.as_u16()),
            ));
        };
        let lane_domains = lane_info.domains();
        for item in &work {
            if item.device_identity() != self.device {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "a batch may not mix recorded work from another device",
                ));
            }
            let domains = item.work_domains();
            if !lane_domains.contains(domains) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "lane {} does not accept every work domain in this batch",
                        lane.as_u16()
                    ),
                ));
            }
        }

        let batch = SubmissionBatchId::new(self.batches.len() as u32);
        let point = PlanPoint::new(self.id, batch);
        self.batches.push(PlannedBatch {
            point,
            lane,
            work,
        });
        self.lanes.push(lane);
        Ok(point)
    }

    /// Adds an explicit happens-before edge between two batches of this plan.
    ///
    /// Different lanes are unordered by default, so this is the only way to
    /// create portable happens-before between them.
    pub fn add_dependency(&mut self, before: PlanPoint, after: PlanPoint) -> RhiResult<()> {
        let before_index = self.batch_index(before)?;
        let after_index = self.batch_index(after)?;
        if before_index == after_index {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a batch cannot depend on itself",
            ));
        }
        let from = self.lanes[before_index];
        let to = self.lanes[after_index];
        let route = self.submission.dependency_route(from, to);
        if route == LaneDependencyRoute::Unsupported {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device proves no happens-before route from lane {} to lane {}",
                    from.as_u16(),
                    to.as_u16()
                ),
            ));
        }
        if self.dependencies.contains(&(before, after)) {
            return Ok(());
        }
        self.dependencies.push((before, after));
        Ok(())
    }

    /// Establishes happens-before from previously submitted work to a batch.
    ///
    /// This succeeds when the destination lane shares an ordered execution
    /// domain with every lane the device declares, or when the device proves a
    /// GPU-side or collapsing route into it: the builder cannot see which lane
    /// the completing work ran on, so it accepts exactly when *no* declared
    /// source lane could fail to reach `after`. A device with a single lane, and
    /// a device whose lanes are mutually ordered, therefore both accept, which is
    /// the "already ordered is not a reason to refuse" rule.
    ///
    /// A caller that knows the order already holds may instead observe
    /// completion as `Complete` before submitting.
    pub fn add_external_dependency(
        &mut self,
        before: CompletionPoint,
        after: PlanPoint,
    ) -> RhiResult<()> {
        if before.device_identity() != self.device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "a completion point from another device cannot order work on this one",
            ));
        }
        let after_index = self.batch_index(after)?;
        let destination = self.lanes[after_index];
        let unreachable = self
            .submission
            .lanes()
            .iter()
            .map(|info| info.id())
            .find(|source| {
                self.submission.dependency_route(*source, destination)
                    == LaneDependencyRoute::Unsupported
            });
        if let Some(source) = unreachable {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device proves no route from lane {} to lane {}; observe the prior \
                     completion as Complete instead of ordering against it",
                    source.as_u16(),
                    destination.as_u16()
                ),
            ));
        }
        self.external.push((before, after));
        Ok(())
    }

    /// Includes a frame's presentation in the plan.
    ///
    /// The frame is consumed and moves to `PlannedForPresent`. A frame may be
    /// presented at most once per plan.
    pub fn present_after(
        &mut self,
        frame: AcquiredFrame,
        after: PlanPoint,
    ) -> RhiResult<PresentPlanId> {
        if frame.device_identity() != self.device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the frame was acquired from another device",
            ));
        }
        let state = frame.state();
        if state != AcquiredFrameState::Acquired {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("the frame is {state:?}, so it cannot be planned for present"),
            ));
        }
        let after_index = self.batch_index(after)?;
        let frame_id = frame.id();
        if self
            .presents
            .iter()
            .any(|present| present.frame_id() == frame_id)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a frame may be presented at most once",
            ));
        }

        // The `after` point must be a batch the plan will give a frame use to, or
        // a batch that already precedes one; both are checked together in
        // `build`, where the whole ordering is known.
        let _ = after_index;
        let id = PresentPlanId {
            plan: self.id,
            local: self.presents.len() as u32,
        };
        frame.plan_for_present()?;
        self.presents.push(PlannedPresent { id, frame, after });
        Ok(id)
    }

    /// Validates the plan and produces it.
    ///
    /// Everything that can only be judged once the plan is complete runs here:
    /// the ordering graph must be acyclic, every pair of unordered batches that
    /// touch the same bytes with a write on either side must be ordered, and
    /// every frame the plan uses on the GPU must be presented by it.
    pub fn build(self) -> RhiResult<SubmissionPlan> {
        let nodes = self.nodes();
        let edges = self.edges();
        let reaches = reachability(&nodes, &edges);

        self.check_acyclic(&nodes, &edges)?;
        self.check_unordered_hazards(&reaches)?;
        self.check_frame_closure(&reaches)?;

        Ok(SubmissionPlan {
            id: self.id,
            device: self.device,
            batches: self.batches,
            dependencies: self.dependencies,
            external: self.external,
            presents: self.presents,
        })
    }

    /// The index of the batch a point names, refusing a foreign or unknown point.
    fn batch_index(&self, point: PlanPoint) -> RhiResult<usize> {
        if point.plan() != self.id {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the plan point belongs to a different plan",
            ));
        }
        let index = point.batch().get() as usize;
        if index >= self.batches.len() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the plan point names a batch this plan does not have",
            ));
        }
        Ok(index)
    }

    /// Every node of the ordering graph.
    fn nodes(&self) -> Vec<Node> {
        let mut nodes: Vec<Node> = (0..self.batches.len()).map(Node::Batch).collect();
        nodes.extend((0..self.presents.len()).map(Node::Present));
        nodes
    }

    /// Every ordering edge of the graph.
    ///
    /// Three sources produce portable order: the implicit order of batches
    /// inserted on one lane, the explicit batch dependencies, and a present's
    /// dependency on the point it follows.
    fn edges(&self) -> Vec<(Node, Node)> {
        let mut edges = Vec::new();

        // Same-lane batches are ordered by insertion, and only within a lane:
        // batching on one lane cannot order work on another.
        let mut last_on_lane: BTreeMap<u16, usize> = BTreeMap::new();
        for (index, lane) in self.lanes.iter().enumerate() {
            let key = lane.as_u16();
            if let Some(previous) = last_on_lane.insert(key, index) {
                edges.push((Node::Batch(previous), Node::Batch(index)));
            }
        }

        for (before, after) in &self.dependencies {
            if let (Ok(before), Ok(after)) = (self.batch_index(*before), self.batch_index(*after)) {
                edges.push((Node::Batch(before), Node::Batch(after)));
            }
        }

        for (index, present) in self.presents.iter().enumerate() {
            if let Ok(after) = self.batch_index(present.after) {
                edges.push((Node::Batch(after), Node::Present(index)));
            }
        }

        edges
    }

    /// Refuses a plan whose ordering contains a cycle.
    fn check_acyclic(&self, nodes: &[Node], edges: &[(Node, Node)]) -> RhiResult<()> {
        if has_cycle(nodes, edges) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the plan's dependencies form a cycle",
            ));
        }
        Ok(())
    }

    /// Refuses a pair of unordered batches that race on the same bytes.
    ///
    /// A read/read pair needs no order. Anything else — read/write, write/read,
    /// write/write — does, and a caller that forgot it would otherwise get a
    /// cross-queue data race that no backend is obliged to catch.
    fn check_unordered_hazards(&self, reaches: &Reachability) -> RhiResult<()> {
        for left in 0..self.batches.len() {
            for right in (left + 1)..self.batches.len() {
                if reaches.path(Node::Batch(left), Node::Batch(right))
                    || reaches.path(Node::Batch(right), Node::Batch(left))
                {
                    continue;
                }
                if let Some(hazard) = first_hazard(&self.batches[left], &self.batches[right]) {
                    return Err(RhiError::new(
                        RhiErrorKind::MissingDependency,
                        format!(
                            "batch {left} and batch {right} are unordered but both touch {hazard} \
                             with at least one write"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Refuses a plan that uses a frame on the GPU but never presents it, or
    /// that touches a frame after its present.
    fn check_frame_closure(&self, reaches: &Reachability) -> RhiResult<()> {
        let mut used: Vec<(AcquiredFrameId, usize)> = Vec::new();
        for (index, batch) in self.batches.iter().enumerate() {
            for frame in frames_used_by(batch) {
                used.push((frame, index));
            }
        }

        for (frame, batch) in &used {
            let Some(present_index) = self
                .presents
                .iter()
                .position(|present| present.frame_id() == *frame)
            else {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the plan uses a frame on the GPU but does not present it",
                ));
            };
            // Every frame use must happen-before its present, or be the point the
            // present follows. A use that is not ordered before the present is a
            // use after the image has been handed to the presentation engine.
            if !reaches.path(Node::Batch(*batch), Node::Present(present_index)) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the plan touches a frame that does not happen-before its present",
                ));
            }
        }
        Ok(())
    }
}

/// Whether any batch on any lane uses `frame`.
fn frames_used_by(batch: &PlannedBatch) -> Vec<AcquiredFrameId> {
    let mut frames = Vec::new();
    for work in &batch.work {
        for use_ in work.resource_uses() {
            if let ResourceUse::Frame(use_) = use_ {
                if !frames.contains(&use_.frame) {
                    frames.push(use_.frame);
                }
            }
        }
    }
    frames
}

/// The first resource two batches touch in a conflicting way, if any.
fn first_hazard(left: &PlannedBatch, right: &PlannedBatch) -> Option<String> {
    for left_work in &left.work {
        for right_work in &right.work {
            for left_use in left_work.resource_uses() {
                for right_use in right_work.resource_uses() {
                    if !use_writes(left_use) && !use_writes(right_use) {
                        continue;
                    }
                    if let Some(description) = use_conflict(left_use, right_use) {
                        return Some(description);
                    }
                }
            }
        }
    }
    None
}

/// Whether one use writes.
fn use_writes(use_: &ResourceUse) -> bool {
    match use_ {
        ResourceUse::Buffer(use_) => use_.access.writes(),
        ResourceUse::Texture(use_) => use_.access.writes(),
        ResourceUse::Frame(use_) => use_.access.writes(),
    }
}

/// The description of a conflict, when two uses touch overlapping bytes.
fn use_conflict(left: &ResourceUse, right: &ResourceUse) -> Option<String> {
    match (left, right) {
        (ResourceUse::Buffer(left), ResourceUse::Buffer(right)) => {
            if left.buffer.id() != right.buffer.id() {
                return None;
            }
            if !ranges_overlap(left.range, right.range) {
                return None;
            }
            Some(format!("buffer {}", left.buffer.id().as_u64()))
        }
        (ResourceUse::Texture(left), ResourceUse::Texture(right)) => {
            if left.texture.id() != right.texture.id() {
                return None;
            }
            if !subresources_overlap(left.subresources, right.subresources) {
                return None;
            }
            Some(format!("texture {}", left.texture.id().as_u64()))
        }
        (ResourceUse::Frame(left), ResourceUse::Frame(right)) => {
            if left.frame != right.frame {
                return None;
            }
            Some(format!("frame {}", left.frame.serial()))
        }
        _ => None,
    }
}

/// Whether two byte ranges overlap.
fn ranges_overlap(left: super::super::resource::BufferRange, right: super::super::resource::BufferRange) -> bool {
    let left_end = left.end();
    let right_end = right.end();
    match (left_end, right_end) {
        (Some(left_end), Some(right_end)) => left.offset < right_end && right.offset < left_end,
        // An unrepresentable end is a range that runs to the end of its buffer,
        // so it overlaps anything that starts at or after it.
        _ => true,
    }
}

/// Whether two subresource ranges select any texel in common.
fn subresources_overlap(
    left: super::super::resource::TextureSubresourceRange,
    right: super::super::resource::TextureSubresourceRange,
) -> bool {
    if left.aspects.bits() & right.aspects.bits() == 0 {
        return false;
    }
    let mip_overlap = left.base_mip < right.base_mip + right.mip_count
        && right.base_mip < left.base_mip + left.mip_count;
    let layer_overlap = left.base_layer < right.base_layer + right.layer_count
        && right.base_layer < left.base_layer + left.layer_count;
    mip_overlap && layer_overlap
}

/// Which nodes are reachable from which, in the plan's ordering.
struct Reachability {
    /// A square bit matrix indexed by node position.
    table: Vec<Vec<bool>>,
    positions: BTreeMap<Node, usize>,
}

impl Reachability {
    /// Whether `from` happens-before `to`.
    ///
    /// A node happens-before itself, which is what makes "be that point" in the
    /// present-closure rule expressible as one question.
    fn path(&self, from: Node, to: Node) -> bool {
        match (self.positions.get(&from), self.positions.get(&to)) {
            (Some(from), Some(to)) => self.table[*from][*to],
            _ => false,
        }
    }
}

/// Computes the transitive closure of the plan's ordering.
fn reachability(nodes: &[Node], edges: &[(Node, Node)]) -> Reachability {
    let positions: BTreeMap<Node, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect();
    let mut table = vec![vec![false; nodes.len()]; nodes.len()];
    for (index, _) in nodes.iter().enumerate() {
        table[index][index] = true;
    }
    for (from, to) in edges {
        if let (Some(from), Some(to)) = (positions.get(from), positions.get(to)) {
            table[*from][*to] = true;
        }
    }
    // Floyd–Warshall over a plan-sized graph: a plan has tens of batches, not
    // millions, and the closure is what both later checks read.
    for via in 0..nodes.len() {
        for from in 0..nodes.len() {
            if !table[from][via] {
                continue;
            }
            for to in 0..nodes.len() {
                if table[via][to] {
                    table[from][to] = true;
                }
            }
        }
    }
    Reachability { table, positions }
}

/// Whether the ordering graph contains a cycle.
fn has_cycle(nodes: &[Node], edges: &[(Node, Node)]) -> bool {
    let positions: BTreeMap<Node, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect();
    let mut outgoing = vec![Vec::new(); nodes.len()];
    for (from, to) in edges {
        if let (Some(from), Some(to)) = (positions.get(from), positions.get(to)) {
            outgoing[*from].push(*to);
        }
    }

    // 0 = unvisited, 1 = on the current path, 2 = finished.
    let mut state = vec![0u8; nodes.len()];
    fn visit(node: usize, outgoing: &[Vec<usize>], state: &mut [u8]) -> bool {
        if state[node] == 1 {
            return true;
        }
        if state[node] == 2 {
            return false;
        }
        state[node] = 1;
        for next in &outgoing[node] {
            if visit(*next, outgoing, state) {
                return true;
            }
        }
        state[node] = 2;
        false
    }

    (0..nodes.len()).any(|node| visit(node, &outgoing, &mut state))
}
