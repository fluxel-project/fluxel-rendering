# ADR-0004: Quarantine accepted-unknown GPU work

**Status:** Superseded by the normative [RHI API v1 submission acceptance
contract](../rhi-design/05-submission-completion-presentation.md#413-devicesubmit-acceptance-contract)
for 0.16.

**Supersession note.** This ADR preserves the evidence and safety concern from
the pre-v1 implementation history, but it does not define the 0.16 public API.
The accepted-unknown quarantine name and its generation-poisoning model are not
part of the v1 contract.

## Current RHI API v1 replacement

`Device::submit()` returns `Err` only when it guarantees that **zero native
work from the plan was accepted**. Once any native work has been accepted,
`Device::submit()` returns `Ok(SubmissionReceipt)`; a later submit failure,
device loss, presentation failure, or retirement outcome is reported through
the receipt's terminal `CompletionState` and, where applicable,
`PresentState`. Those terminal states exclusively own all post-acceptance
failure and retirement reporting. Accepted work remains retained until that
terminal lifecycle allows safe retirement.

## Context

A native submit error may occur after work has been accepted. Treating it as a
known rejection can release commands, staging memory, resources, or leases
while the GPU may still reference them.

## Historical decision

Distinguish known pre-submit rejection from accepted-unknown work and terminal
completion failure. Accepted-unknown work retains/quarantines all referenced
objects until safe retirement; it never publishes guessed outgoing state or a
ready snapshot.

## Alternatives

- Collapse all submit errors into one `Result` failure.
- Use queue-idle or a blocking drop as cleanup.

## Consequences

Pending operations are owning, non-blocking state machines. Renderer snapshot
reservations release before accepted work, but terminal failure or early drop
after acceptance poisons an affected generation when its state is unknown.

## Evidence

0.1.2 formalized structured completion and quarantine. 0.2.0 upload and
0.2.1–0.2.7 snapshot fault fixtures exercised partial acceptance, failure,
drop, and subsequent reuse behavior.

See [RHI design](../design-rhi.md) and [Renderer design](../design-renderer.md)
for the current lifecycle contract.
