# ADR-0014: Treat a lost device domain as terminal

**Status:** Accepted.

## Decision

After a terminal loss outcome, work and resources from that opened device domain
are not revived. Recovery obtains a new RHI device and rebuilds device-scoped
plans, pipelines, residency, and surface state from durable CPU inputs.

The selected RHI's `DeviceId`, loss outcome, and surface lifecycle define the
actual public facts. Renderer may keep private cache generations where needed,
but must not invent a competing public device-generation contract.

## Consequences

Old resources remain retained or are discarded only through the current RHI
terminal-outcome rules. Logical asset identity survives recovery; native objects
do not cross it.
