# ADR-0004: Historical submission-uncertainty quarantine

**Status:** Superseded. The historical quarantine protocol does not define the
current RHI or renderer contract.

## Historical context

The pre-independent-RHI renderer used an `accepted-unknown` state after a
submission error whose native effect could not be determined. It retained the
objects that might still be referenced rather than treating every error as a
clean rejection.

## Durable lesson

A successful return is not completion, and a failed call is not necessarily a
proof that no work was issued. The current RHI distinguishes rejected,
accepted, partially accepted, and indeterminate outcomes. Renderer and Graph
must retain every resource possibly affected by the reported prefix or
uncertainty and must not reuse its state until current RHI evidence or recovery
makes it trustworthy.

The old name, generation poisoning, and local state machine are removed. Exact
submission, completion, and recovery behavior belongs to the current
[RHI Core design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md).

## Evidence

The old tests remain historical evidence that silent early retirement was
unsafe. New consumers test current RHI outcomes rather than reproducing this
protocol.
