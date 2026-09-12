# 0005: Run Extraction Tasks In API Workers

- Status: Accepted
- Date: 2026-09-10

## Context

Extraction is asynchronous from the client's perspective, but the service does
not currently operate a separate task queue. Redis stores job state and results;
it does not claim, acknowledge, or resume work. Present extraction requests are
bounded by cell count, concurrency, and processing time.

## Decision

For the current release, execute extraction jobs as FastAPI background tasks in
the API worker that accepts them. Treat Redis as a 24-hour result and status
store, not as a durable execution queue.

## Consequences

- A worker restart can abandon an in-flight extraction, and abandoned work is
  not resumed automatically.
- Clients may resubmit missing or stale nonterminal jobs.
- A durable queue and separate worker service are required before promising
  restart-safe, resumable, or long-running extraction.
- Admission control remains process-local and the deployment-wide ceiling is
  the per-process limit multiplied by the number of API workers.
