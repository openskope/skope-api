# Lightweight backend telemetry

The API and TiTiler emit single-line JSON events to stderr, collected by Docker
alongside existing application logs. No collector or external service is needed.
Sentry error reporting is independent; these events are not sent to Sentry.

## Volume controls

Set these environment variables when invoking the deployment Make target:

| Variable | Default | Behavior |
| --- | --- | --- |
| `SKOPE_TELEMETRY_ENABLED` | `true` | Set `false` to disable structured telemetry in both services. |
| `SKOPE_TELEMETRY_GET_SAMPLE_RATE` | `0.1` | Fraction of successful GET requests logged, from `0` to `1`. |

Changes require container recreation through the normal deployment process;
they do not require an image rebuild. Configuration is read at process startup.
The API and TiTiler should use the same sampling rate.

Health paths `/health` and `/healthz` are excluded. Successful GETs, including
tiles and polling, are sampled deterministically using the request ID. Errors,
incomplete requests and non-GET requests are always logged while enabled.
Job/analysis events are always logged while enabled and are not emitted per
pixel, timestep or series. Sustained error traffic can still generate substantial
logs; the switch disables these events without changing normal error reporting.
The TiTiler command and development API command disable duplicate Uvicorn access
logs; production/staging Gunicorn uses its default disabled access log. Normal
application diagnostics remain separate from these controls. Telemetry emission
is best-effort: formatting or sink failures must not break request/job handling.
Use Docker log rotation to bound disk usage.

## Events and interpretation

- `http_request`: service, method, route template, status, outcome and duration.
  Timing ends after the final response body is sent, before background tasks.
  A failed stream can have status 200 and outcome `error` because headers were
  already sent. Duration excludes time queued before the ASGI application runs.
- `extraction_submission`: job ID and whether the submission recovered an
  existing attempt, rather than scheduling new work.
- `raster_reads_finished`: raster extraction duration, cells, window cells,
  files, timesteps and bands per chunk. Includes waiting for read-thread capacity
  and calculating spatial means/medians; excludes subsequent temporal transforms
  and response assembly.
- `extraction_finished`: total claimed-job duration, intended terminal outcome,
  error code and whether the store accepted that terminal update (`recorded`).
  Includes completeness counts for the full extracted mean/median series and
  each returned series when available. This does not claim that a rejected/stale
  result was successfully published.
- `analysis_finished`: duration including waiting for process capacity, outcome,
  extraction ID, input timesteps and requested series count. `capacity_wait_ms`
  measures waiting for the per-worker analysis slot. `compute_ms` measures the
  admitted operation, including process startup, IPC and computation; it is not
  CPU time. A timeout while waiting reports zero compute time. Does not split
  individual transforms or smoothing operations.
  Completeness counts distinguish the full input mean/median series from each
  returned series after transformation, smoothing and range selection.
- `job_contract_error`: structured error code and HTTP status, including capacity,
  coordination, expiry and timeout errors. An HTTP completion event also records
  the failed request.

Events include UTC timestamp, PID and request ID. The API forwards the request ID
to TiTiler and returns it as `X-Request-ID`. Valid incoming IDs are 32 lowercase
hex characters; others are replaced. IDs correlate events but are not trusted
identities. Matching IDs keep sampled API and TiTiler requests together.

Structured events omit request bodies, geometry, scientific result values,
query strings and raster URLs. Route templates avoid a distinct route for every
tile coordinate or job ID. Existing diagnostics can still contain error details.

For worker-count comparisons, collect these logs alongside client-observed
request timings and Docker CPU/memory/I/O statistics. These events do not measure
individual GDAL HTTP transfers, internal cache hits or host resource use.

Completeness fields `base_series_counts` and `output_series_counts` report total,
valid and missing timesteps, with `all_missing` true only for a nonempty series
without valid values. Null and nonfinite values count as missing. Empty series
have zero counts and are not marked all-missing. Output counts are per series,
identified by response position; they must not be summed as unique timesteps.
Missing output can be expected at smoothing edges or from undefined transforms;
these fields are diagnostic, not a claim that a dataset is corrupt. Failed work
has no output counts rather than fabricated zero/missing results. Counting is
skipped when structured telemetry is disabled. No additional events are emitted.
