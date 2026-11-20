# 0008: Recover Extraction Submissions Within A Transient Store Lifetime

- Status: Accepted
- Date: 2026-11-18
- Supersedes: the expiry/admission and abandoned-work behavior in 0003 and 0005

## Context

A lost acceptance response makes retries ambiguous. Process-local admission does
not coordinate API workers, and abandoned background jobs can remain nonterminal.
Client polling budgets do not currently bound HTTP operations. Temporal is a
possible later executor, but is not part of this phase.

## Decision

Adopt the executor-independent HTTP contract in
`docs/specs/recoverable-extraction-jobs-v1.md`. Bind client keys to complete request
fingerprints and accepted release identities. Require `Idempotency-Key` on every
submission; older clients must update before submitting. Coordinate reservation
and finite ownership leases atomically in the shared job store before dispatch; retries
recover existing attempts before checking capacity. Fence terminal/expired owners
and fail abandoned attempts explicitly rather than automatically rescheduling.

Keep Redis transient by explicit user choice: a Redis restart may lose recovery
state and permit repeat work. Retain results for 24 hours from acceptance and key
tombstones for another 24 hours while the store survives. Keep background tasks
for this phase and defer durable execution to a possible Temporal transition.

Bound browser fetch/body reads and phase waits separately. Bound synchronous CPU
analysis in a cancellable process, with one independent analysis slot per API
worker, and distinguish transport deadlines from extraction processing deadlines in public failures and UI messages.

## Consequences

- Repeat submissions cannot duplicate dispatch while the shared acceptance record survives.
- Capacity is deployment-wide: four staging/production jobs and one development job.
- Worker restarts do not leave observable jobs active indefinitely, but do not resume extraction.
- Redis data loss is an accepted recovery boundary, not a promise of exactly-once execution.
- Public API types come from OpenAPI; base series and ownership data remain internal.
- Temporal must preserve this contract and route each accepted attempt to one executor during migration.
