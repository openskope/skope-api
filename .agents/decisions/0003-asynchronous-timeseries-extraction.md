# 0003: Separate Extraction Jobs From Analysis

- Status: Accepted
- Date: 2026-08-11

## Context

Reading and summarizing raster time series can exceed a normal request latency. Re-reading source data for every transform or smoothing choice would also waste storage and compute resources.

## Decision

Accept extraction requests asynchronously, return a job identifier, and expose status polling. Store job state in Redis for 24 hours in deployed environments, with a filesystem implementation available for tests and local fallback. Keep the extracted base series in internal job state so synchronous analysis can reuse it without another COG read.

## Consequences

- Clients follow the extract, poll, then analyze workflow.
- Missing or expired jobs return `404`; incomplete jobs cannot be analyzed.
- `base_series` must never be returned by the public status endpoint.
- Redis is part of the deployed service health contract.

