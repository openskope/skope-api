"""Executor-independent acceptance and lifecycle contract.

The ledger coordinates current background workers; Temporal can implement the
same contract later. Redis loss intentionally loses this recovery boundary.
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass

import anyio

from app.config import get_settings
from app.schemas.jobs import ExtractionProvenance
from app.store.jobs import JobStore, valid_job_id

RESULT_SECONDS = 86400
KEY_SECONDS = 2 * RESULT_SECONDS
ACTIVE = {"PENDING", "PROCESSING"}


class JobContractError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def submission_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "openskope:extract:" + key))


def request_fingerprint(payload, release_id: str | None, digest: str | None) -> str:
    raw = json.dumps(
        [payload.model_dump(mode="json"), release_id, digest],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def public_lifecycle(meta: dict) -> dict:
    return {
        name: meta[name]
        for name in (
            "accepted_at",
            "dispatch_deadline",
            "processing_deadline",
            "expires_at",
            "key_expires_at",
            "started_at",
            "finished_at",
        )
        if name in meta
    }


def accepted(meta: dict) -> dict:
    return {
        "job_id": meta["job_id"],
        "status": "accepted",
        "lifecycle": public_lifecycle(meta),
        "release_id": meta["provenance"]["release_id"],
        "declaration_digest": meta["provenance"]["declaration_digest"],
    }


@dataclass
class Reservation:
    record: dict
    created: bool


class RecoverableJobs:
    def __init__(self, store: JobStore):
        self.store = store
        self.settings = get_settings()

    async def _atomic(self, job_id, change):
        try:
            with anyio.fail_after(5):
                return await self.store.atomic_extraction(job_id, change)
        except JobContractError:
            raise
        except Exception as exc:
            raise JobContractError(
                503,
                "COORDINATION_UNAVAILABLE",
                "Job coordination is unavailable. Retry the same submission key.",
            ) from exc

    @staticmethod
    def _reconcile(state, now):
        state["active"] = {
            key: deadline for key, deadline in state["active"].items() if deadline > now
        }
        meta = state["meta"]
        if (
            meta
            and meta["status"] in ACTIVE
            and (meta["lease_expires_at"] <= now or not state.get("lease_valid", True))
        ):
            meta.update(
                status="FAILED",
                error_code="WORKER_LOST",
                error="The worker did not finish this job. Start a new attempt.",
                finished_at=now,
            )
        if meta and meta["status"] not in ACTIVE:
            # Repair admission after a partially written local completion.
            state["active"].pop(meta["job_id"], None)
        return meta

    @staticmethod
    def _check(meta, fingerprint, now):
        if meta["fingerprint"] != fingerprint:
            raise JobContractError(
                409,
                "IDEMPOTENCY_CONFLICT",
                "This submission key belongs to a different request.",
            )
        if now >= meta["expires_at"]:
            raise JobContractError(
                410,
                "RESULT_EXPIRED",
                "The extraction has expired. Start a new attempt with a new key.",
            )

    async def lookup(self, job_id, fingerprint):
        def change(state):
            now = time.time()
            meta = self._reconcile(state, now)
            if not meta or now >= meta["key_expires_at"]:
                return None
            self._check(meta, fingerprint, now)
            return Reservation(dict(meta), False)

        return await self._atomic(job_id, change)

    async def reserve(
        self, job_id, fingerprint, provenance: ExtractionProvenance, processing_ms
    ):
        def change(state):
            now = time.time()
            meta = self._reconcile(state, now)
            if meta and now < meta["key_expires_at"]:
                self._check(meta, fingerprint, now)
                return Reservation(dict(meta), False)
            if len(state["active"]) >= self.settings.max_concurrent_jobs:
                raise JobContractError(
                    503,
                    "CAPACITY_EXHAUSTED",
                    "The extraction service is at capacity. Retry later.",
                )
            dispatch = now + self.settings.extraction_dispatch_seconds
            lease = dispatch + self.settings.extraction_recovery_grace_seconds
            meta = dict(
                job_id=job_id,
                fingerprint=fingerprint,
                owner=str(uuid.uuid4()),
                provenance=provenance.model_dump(mode="json"),
                status="PENDING",
                accepted_at=now,
                dispatch_deadline=dispatch,
                processing_deadline=dispatch + processing_ms / 1000,
                processing_ms=processing_ms,
                lease_expires_at=lease,
                expires_at=now + RESULT_SECONDS,
                key_expires_at=now + KEY_SECONDS,
            )
            state.update(meta=meta, data=None)
            state["active"][job_id] = lease
            return Reservation(dict(meta), True)

        return await self._atomic(job_id, change)

    async def claim(self, job_id: str, owner: str) -> float | None:
        """Claim once and return the stored absolute processing deadline."""

        def change(state):
            now = time.time()
            meta = self._reconcile(state, now)
            if not meta or meta["owner"] != owner or meta["status"] != "PENDING":
                return None
            if now > meta["dispatch_deadline"]:
                meta.update(
                    status="FAILED",
                    error_code="WORKER_LOST",
                    error="The job did not start before its dispatch deadline.",
                    finished_at=now,
                )
                state["active"].pop(job_id, None)
                return None
            deadline = now + meta["processing_ms"] / 1000
            lease = deadline + self.settings.extraction_recovery_grace_seconds
            meta.update(
                status="PROCESSING",
                started_at=now,
                processing_deadline=deadline,
                lease_expires_at=lease,
            )
            state["active"][job_id] = lease
            return deadline

        return await self._atomic(job_id, change)

    async def finish(
        self,
        job_id,
        owner,
        status,
        *,
        result=None,
        base_series=None,
        error=None,
        error_code=None
    ):
        if status not in {"SUCCESS", "FAILED", "TIMED_OUT"}:
            raise ValueError("Completion requires a terminal job status.")

        def change(state):
            now = time.time()
            meta = self._reconcile(state, now)
            if not meta or meta["owner"] != owner or meta["status"] not in ACTIVE:
                return False
            meta.update(status=status, finished_at=now)
            if error is not None:
                meta.update(error=error, error_code=error_code)
            state["data"] = (
                {
                    "schema_version": "1.0.0",
                    "provenance": meta["provenance"],
                    "status": status,
                    "result": result,
                    "base_series": base_series,
                }
                if status == "SUCCESS"
                else None
            )
            state["active"].pop(job_id, None)
            return True

        return await self._atomic(job_id, change)

    async def get(self, job_id):
        if not valid_job_id(job_id):
            raise JobContractError(404, "JOB_NOT_FOUND", "Extraction not found.")

        def change(state):
            now = time.time()
            meta = self._reconcile(state, now)
            if meta is None or now >= meta["key_expires_at"]:
                if state["data"] and meta is None:
                    return state["data"]  # Pre-contract stored jobs drain normally.
                raise JobContractError(404, "JOB_NOT_FOUND", "Extraction not found.")
            if now >= meta["expires_at"]:
                raise JobContractError(
                    410, "RESULT_EXPIRED", "The extraction result has expired."
                )
            record = (
                dict(state["data"])
                if meta["status"] == "SUCCESS" and state["data"] is not None
                else {"schema_version": "1.0.0", "provenance": meta["provenance"]}
            )
            record.update(status=meta["status"], lifecycle=public_lifecycle(meta))
            for key in ("error", "error_code"):
                if key in meta:
                    record[key] = meta[key]
            if meta["status"] == "SUCCESS" and not record.get("result"):
                raise JobContractError(
                    410,
                    "RESULT_EXPIRED",
                    "The extraction result is no longer available.",
                )
            if meta["status"] == "SUCCESS":
                result = record["result"]
                provenance = meta["provenance"]
                if (
                    record.get("schema_version") != "1.0.0"
                    or record.get("provenance") != provenance
                    or not isinstance(result, dict)
                    or (result.get("dataset_id"), result.get("variable_id"))
                    != (provenance["dataset_id"], provenance["variable_id"])
                ):
                    raise JobContractError(
                        410,
                        "EXTRACTION_INCOMPATIBLE",
                        "The stored extraction is incompatible with its accepted source. Start a new attempt.",
                    )
            return record

        return await self._atomic(job_id, change)
