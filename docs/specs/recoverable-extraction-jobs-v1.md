# Recoverable extraction jobs v1

This contract implements openskope/planning#43 for the current FastAPI executor.
It does not introduce Temporal or a durable execution queue. Temporal can later
replace the executor while preserving the HTTP surface and client ownership rules.

## Recovery boundary and identity

Clients generate a random UUID for each extraction attempt and send it as
`Idempotency-Key` on every retry. Keys are scoped to this API deployment's
extraction endpoint, across workers and origins. They are not authentication.
Do not reuse a key for another attempt. The header is required: submissions
without it receive 422 before any job is accepted. Older clients must update
before submitting extraction work.

The fingerprint covers the complete validated request (including defaults,
analysis options, processing budget and geometry) and the two optional source
headers, with sorted object keys and unchanged array order. A different
fingerprint with an existing key is `409 IDEMPOTENCY_CONFLICT`.

Clients can additionally send both `X-Skope-Release-Id` and
`X-Skope-Declaration-Digest`, taken from metadata. A new submission against a
different served release is `409 SOURCE_CHANGED`. Once accepted, a retry recovers
that job and its original release before checking the current registry or
capacity. The response declares the accepted release. Analysis still refuses
base series from a different currently served release (`410 SOURCE_CHANGED`).

Redis deliberately remains **transient**. Recovery and retention guarantees hold
while its records survive. A Redis restart/flush can lose keys, results and
admission records; retrying a lost key can then execute extraction again. The
system does not promise exactly-once execution across Redis loss. The filesystem
fallback is for local development and tests, not shared multi-host deployment.

## Acceptance and dispatch

`POST /timeseries/extract` returns `202 ExtractionAcceptedResponse` with `job_id`,
accepted release identity and lifecycle metadata. Replays return the same ID,
including for terminal jobs; they never schedule work again.

A shared store transaction checks the key, checks deployment-wide admission and
writes the pending record plus ownership lease. Redis uses `WATCH`/`MULTI` with
bounded contention retries and an expiring owner key watched through commit;
lease expiry between preparing an update and Redis `EXEC` invalidates that update; the filesystem fallback uses a cancellation-aware OS file-lock wait and
per-file atomic replacement. It serializes access but does not promise crash-atomic
multi-file writes; results are written before success metadata, and admission is
repaired when a terminal record is read. There are four deployment-wide slots in staging and
production, and one in development. Capacity refusal creates no key reservation
and returns `503 CAPACITY_EXHAUSTED`, `Retry-After: 5`. CORS exposes this header.

Only the worker holding the private reservation token may claim the pending job,
and it may claim it once. FastAPI scheduling happens after acceptance is recorded:
these cannot be made a durable transaction without a queue. A crash in this gap
leaves a finite pending lease, never permission for another retry to dispatch.
An observed scheduling failure records terminal `SCHEDULING_FAILED` and releases
admission. If that write fails, lease expiry provides recovery.

Pending work has 10 seconds to start, followed by 5 seconds of recovery grace.
At claim, `processing_deadline` becomes the actual start plus the requested
processing budget (at most 20 seconds). Claim/store response time consumes that
budget, and late computation or serialization cannot publish success merely
because a cancellation callback was delayed. The running ownership lease ends five
seconds after that deadline. Any read or recovery attempt reconciles an expired
lease to terminal `FAILED / WORKER_LOST`; admission also discards expired leases.
A delayed worker cannot claim after its dispatch deadline, and a stale owner
cannot publish results after lease expiry or terminal completion.

Extraction retains its cooperative raster-chunk cancellation. An OS-suspended
worker or uninterruptible native I/O is not physically killed by a Redis lease;
fencing prevents its late publication. There is no automatic re-execution of
abandoned work. A user may start a new attempt with a new key.

## Lifecycle, results and expiry

Lifecycle timestamps are UTC Unix seconds: `accepted_at`, `dispatch_deadline`,
`processing_deadline`, `expires_at`, `key_expires_at`, and optional `started_at`
and `finished_at`. Accepted responses provide the latest possible processing
deadline until a status response reports the actual start.

Results and base series expire **24 hours from acceptance**, without refreshing
on updates or reads. The lightweight key record remains for another 24 hours.
During this tombstone interval, matching submissions and status/analysis reads
return `410 RESULT_EXPIRED`; they do not dispatch new work. Conflicting payloads
still receive `409 IDEMPOTENCY_CONFLICT`. After 48 hours the key may be forgotten
and must not be reused. Unknown job IDs return `404 JOB_NOT_FOUND`.

Public status declares `PENDING`, `PROCESSING`, `SUCCESS`, `FAILED` or `TIMED_OUT`,
plus lifecycle and an optional structured failure code/message. `base_series`,
internal provenance, fingerprints and ownership tokens are never public. Initial
successful results preserve API precision and nulls. Legacy records can drain
through the existing read path, without gaining recovery guarantees retroactively.

## Analysis and errors

`POST /timeseries/analyze` declares `TimeseriesResponse` explicitly in OpenAPI.
It has a server-side 20-second budget, including waiting for a process slot.
Analysis runs in an AnyIO worker process; cancellation kills an executing process
rather than leaving a timed-out CPU task running. Each API worker has one analysis
slot, independent of extraction admission: four API workers may run up to four
analysis processes alongside four extractions. No extraction is launched by analysis.

Contract errors have `detail: {code, msg}`. Request validation retains FastAPI's
existing 422 representation. Important codes:

| Code | Meaning / client response |
| --- | --- |
| `CAPACITY_EXHAUSTED` | No new slot; retry the same key within the submission budget. |
| `COORDINATION_UNAVAILABLE` | Acceptance may be uncertain; retry the same key. |
| `IDEMPOTENCY_CONFLICT` | Key used with different input; report the conflict. |
| `SOURCE_CHANGED` | Reload metadata before a new extraction. |
| `RESULT_EXPIRED`, `JOB_NOT_FOUND`, `EXTRACTION_INCOMPATIBLE` | Stored work cannot serve the request. |
| `JOB_INCOMPLETE` | Poll the existing job. |
| `PROCESSING_TIMEOUT` | Extraction exceeded its processing budget; a smaller area may help. |
| `ANALYSIS_TIMEOUT` | Analysis exceeded its budget; retain the extraction. |
| `WORKER_LOST`, `SCHEDULING_FAILED` | Terminal attempt; require a new user attempt. |
| `EXTRACTION_FAILED`, `INVALID_EXTRACTION`, `UPSTREAM_FAILURE` | Terminal execution failure. |

## Client budgets and ownership

Submission/recovery and polling retain separate 60-second budgets. Each HTTP
operation is capped at 15 seconds or its phase's remaining time, whichever is
shorter. Phase budgets use a monotonic clock so wall-clock changes cannot extend
them. Fetch and response-body reads share that operation budget. Synchronous
analysis has a 30-second client budget, allowing time for the 20-second server
budget and transport. There is no additional end-to-end cap. Retry waits cannot
outlive their phase budget. `Retry-After` supports delay seconds and HTTP dates.

Network failures, HTTP-operation deadlines, phase deadlines and backend execution
timeouts have distinct client codes. A gateway 504 is connection trouble unless
the backend explicitly identifies a processing timeout. Never recommend a smaller
area merely because the browser or gateway stopped waiting.

The Pinia extraction owner retains a key and its original payload across uncertain
submissions. Changed analysis choices reuse the recovered base series through
`/analyze`. Known jobs remain available after polling/analysis connection failures.
Page disposal and variable changes do not abort shared work. Retiring the source
release/dataset or study area aborts obsolete HTTP requests and retry waits, but
does **not** imply server-side cancellation. Submission keys are currently kept
within the Pinia session; a browser reload begins a new attempt.

A terminal extraction failure does not automatically launch another expensive
job within the same request. A later explicit user attempt may use a new key.
Cached results that have expired or become incompatible may be replaced through
the existing extraction lifecycle.

## Deployment and Temporal transition

Deploy backend support before enabling the matching frontend's recovery retries.
Because the submission key is required, stale frontends must reload the updated
UI before submitting; coordinate this cutover. Drain old background work before the backend cutover so pre-contract process-local
admission does not overlap the new shared ceiling. Redis configuration stays
transient; no new volume or persistence mode is required.

A later Temporal phase must preserve key/payload conflict checks, source identity,
HTTP error semantics and client deadlines. Assign each accepted attempt to one
executor, drain old jobs and their key-retention windows, and only then retire the
old coordinator. Temporal activity retry policy must not silently reverse the
choice to avoid automatic duplicate expensive execution. Durable Temporal state
can eventually own identity/lifecycle while Redis remains a replaceable result
cache; losing that cache must not recreate the workflow.
