import asyncio
import time

import pytest

from app.core.job_control import RecoverableJobs, JobContractError, submission_id
from app.schemas.jobs import ExtractionProvenance
from app.store.jobs import FileSystemJobStore


@pytest.fixture(params=["filesystem", "redis"])
def store(request):
    return request.getfixturevalue(
        "fs_job_store" if request.param == "filesystem" else "redis_job_store"
    )


@pytest.fixture
def provenance(served_registry):
    return ExtractionProvenance.from_release(served_registry.get("annual"), "ppt")


async def reserve(jobs, provenance, key="same", fingerprint="payload"):
    return await jobs.reserve(submission_id(key), fingerprint, provenance, 20000)


async def test_concurrent_retries_share_one_reservation_at_capacity(store, provenance):
    # Separate service instances represent separate API workers.
    workers = [RecoverableJobs(store) for _ in range(8)]
    results = await asyncio.gather(*(reserve(worker, provenance) for worker in workers))
    assert sum(result.created for result in results) == 1
    assert len({result.record["owner"] for result in results}) == 1
    with pytest.raises(JobContractError) as exc:
        await reserve(workers[0], provenance, key="another")
    assert exc.value.code == "CAPACITY_EXHAUSTED"
    assert (await reserve(workers[1], provenance)).created is False


async def test_conflicting_payload_is_rejected_without_claiming_capacity(
    store, provenance
):
    jobs = RecoverableJobs(store)
    await reserve(jobs, provenance)
    with pytest.raises(JobContractError) as exc:
        await reserve(jobs, provenance, fingerprint="different")
    assert exc.value.code == "IDEMPOTENCY_CONFLICT"


async def test_only_one_dispatch_can_claim_and_terminal_owner_is_fenced(
    store, provenance
):
    jobs = RecoverableJobs(store)
    reservation = await reserve(jobs, provenance)
    job = reservation.record
    results = await asyncio.gather(
        *(jobs.claim(job["job_id"], job["owner"]) for _ in range(5))
    )
    assert sum(result is not None for result in results) == 1
    assert not await jobs.finish(job["job_id"], "wrong-owner", "FAILED")
    assert await jobs.finish(
        job["job_id"],
        job["owner"],
        "FAILED",
        error="stopped",
        error_code="EXTRACTION_FAILED",
    )
    assert not await jobs.finish(job["job_id"], job["owner"], "SUCCESS", result={})
    assert (await reserve(jobs, provenance, key="next")).created


async def test_abandoned_pending_job_expires_and_is_not_scheduled_again(
    store, provenance
):
    jobs = RecoverableJobs(store)
    job = (await reserve(jobs, provenance)).record

    def expire(state):
        state["meta"]["lease_expires_at"] = time.time() - 1
        state["active"][job["job_id"]] = time.time() - 1

    await store.atomic_extraction(job["job_id"], expire)
    assert (await jobs.get(job["job_id"]))["error_code"] == "WORKER_LOST"
    assert not (await reserve(jobs, provenance)).created
    assert not await jobs.claim(job["job_id"], job["owner"])
    assert (await reserve(jobs, provenance, key="next")).created


async def test_abandoned_running_job_cannot_publish_late_results(store, provenance):
    jobs = RecoverableJobs(store)
    job = (await reserve(jobs, provenance)).record
    assert await jobs.claim(job["job_id"], job["owner"])

    def expire(state):
        state["meta"]["lease_expires_at"] = time.time() - 1

    await store.atomic_extraction(job["job_id"], expire)
    assert not await jobs.finish(
        job["job_id"], job["owner"], "SUCCESS", result={"bad": "late"}
    )
    assert (await jobs.get(job["job_id"]))["error_code"] == "WORKER_LOST"


async def test_expired_key_is_a_tombstone_until_retention_ends(store, provenance):
    jobs = RecoverableJobs(store)
    job = (await reserve(jobs, provenance)).record

    def expire_result(state):
        state["meta"]["expires_at"] = time.time() - 1

    await store.atomic_extraction(job["job_id"], expire_result)
    for action in [
        lambda: jobs.lookup(job["job_id"], "payload"),
        lambda: jobs.get(job["job_id"]),
        lambda: reserve(jobs, provenance),
    ]:
        with pytest.raises(JobContractError) as exc:
            await action()
        assert exc.value.code == "RESULT_EXPIRED"

    def expire_key(state):
        state["meta"]["key_expires_at"] = time.time() - 1
        state["active"] = {}

    await store.atomic_extraction(job["job_id"], expire_key)
    replacement = await reserve(jobs, provenance)
    assert replacement.created
    assert replacement.record["owner"] != job["owner"]
    assert not await jobs.claim(job["job_id"], job["owner"])


async def test_filesystem_instances_share_admission(fs_job_store, provenance):
    other = FileSystemJobStore(fs_job_store.directory)
    first, second = await asyncio.gather(
        reserve(RecoverableJobs(fs_job_store), provenance),
        reserve(RecoverableJobs(other), provenance),
    )
    assert first.created != second.created


async def test_uncertain_reservation_reply_does_not_allow_another_dispatch(
    store, provenance, monkeypatch
):
    original = store.atomic_extraction
    lost = False

    async def lose_reply(job_id, change):
        nonlocal lost
        result = await original(job_id, change)
        if getattr(result, "created", False) and not lost:
            lost = True
            raise ConnectionError("reservation committed, response lost")
        return result

    monkeypatch.setattr(store, "atomic_extraction", lose_reply)
    jobs = RecoverableJobs(store)
    with pytest.raises(JobContractError) as exc:
        await reserve(jobs, provenance)
    assert exc.value.code == "COORDINATION_UNAVAILABLE"
    replay = await reserve(jobs, provenance)
    assert not replay.created
    assert replay.record["status"] == "PENDING"
    assert (await jobs.get(replay.record["job_id"]))["status"] == "PENDING"


async def test_independent_redis_clients_coordinate_admission(
    redis_job_store, provenance
):
    import os
    from app.store.jobs import RedisJobStore

    other = RedisJobStore(os.environ.get("REDIS_URL", "redis://redis:6379"))
    try:
        results = await asyncio.gather(
            reserve(RecoverableJobs(redis_job_store), provenance),
            reserve(RecoverableJobs(other), provenance),
        )
        assert sum(result.created for result in results) == 1
        with pytest.raises(JobContractError) as exc:
            await reserve(RecoverableJobs(other), provenance, key="second")
        assert exc.value.code == "CAPACITY_EXHAUSTED"
    finally:
        await other.close()


async def test_result_expiry_and_key_retention_are_fixed_from_acceptance(
    redis_job_store, provenance
):
    jobs = RecoverableJobs(redis_job_store)
    job = (await reserve(jobs, provenance)).record
    original_expiry = job["expires_at"]
    original_key_expiry = job["key_expires_at"]
    assert await jobs.claim(job["job_id"], job["owner"])
    current = await jobs.get(job["job_id"])
    assert current["lifecycle"]["expires_at"] == original_expiry
    assert current["lifecycle"]["key_expires_at"] == original_key_expiry
    assert original_key_expiry - original_expiry == 86400
    ttl = await redis_job_store._client.ttl(f"submission:{job['job_id']}")
    assert 172790 < ttl <= 172800


async def test_nonterminal_job_never_exposes_unpublished_result(store, provenance):
    jobs = RecoverableJobs(store)
    job = (await reserve(jobs, provenance)).record
    assert await jobs.claim(job["job_id"], job["owner"])

    def partial_completion(state):
        state["data"] = {"result": {"unpublished": True}, "base_series": {}}

    await store.atomic_extraction(job["job_id"], partial_completion)
    record = await jobs.get(job["job_id"])
    assert record["status"] == "PROCESSING"
    assert "result" not in record
    assert "base_series" not in record


async def test_terminal_record_repairs_stranded_admission(store, provenance):
    jobs = RecoverableJobs(store)
    job = (await reserve(jobs, provenance)).record

    def partial_completion(state):
        state["meta"]["status"] = "FAILED"

    await store.atomic_extraction(job["job_id"], partial_completion)
    assert (await jobs.get(job["job_id"]))["status"] == "FAILED"
    assert (await reserve(jobs, provenance, key="after-failure")).created


async def test_invalid_completion_does_not_release_a_live_job(store, provenance):
    jobs = RecoverableJobs(store)
    job = (await reserve(jobs, provenance)).record
    with pytest.raises(ValueError, match="terminal"):
        await jobs.finish(job["job_id"], job["owner"], "PROCESSING")
    with pytest.raises(JobContractError) as exc:
        await reserve(jobs, provenance, key="another")
    assert exc.value.code == "CAPACITY_EXHAUSTED"


async def test_job_lookup_cannot_read_files_outside_store(fs_job_store, tmp_path):
    import json

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"status": "PENDING"}))
    with pytest.raises(JobContractError) as exc:
        await RecoverableJobs(fs_job_store).get("../outside")
    assert exc.value.status == 404
    with pytest.raises(ValueError, match="identifier"):
        await fs_job_store.update_job("../outside", {"status": "FAILED"})
    assert json.loads(outside.read_text()) == {"status": "PENDING"}


@pytest.mark.parametrize("tamper", ["provenance", "result-source", "schema-version"])
async def test_successful_result_must_match_its_accepted_source(
    store, provenance, tamper
):
    jobs = RecoverableJobs(store)
    job = (await reserve(jobs, provenance)).record
    assert await jobs.claim(job["job_id"], job["owner"])
    result = dict(
        dataset_id=provenance.dataset_id,
        variable_id=provenance.variable_id,
        area=0,
        n_cells=1,
        summary_stats=[],
        series=[],
        transform={"type": "NoTransform"},
        zonal_statistic="mean",
    )
    await jobs.finish(
        job["job_id"],
        job["owner"],
        "SUCCESS",
        result=result,
        base_series={"timesteps": ["0001"], "mean": [1.0], "median": [1.0]},
    )
    assert (await jobs.get(job["job_id"]))["result"] == result

    def corrupt(state):
        if tamper == "provenance":
            state["data"]["provenance"]["declaration_digest"] = "0" * 64
        elif tamper == "result-source":
            state["data"]["result"]["dataset_id"] = "different-dataset"
        else:
            state["data"]["schema_version"] = "unsupported"

    await store.atomic_extraction(job["job_id"], corrupt)
    with pytest.raises(JobContractError) as exc:
        await jobs.get(job["job_id"])
    assert exc.value.code == "EXTRACTION_INCOMPATIBLE"


async def test_local_completion_survives_admission_write_failure(
    fs_job_store, provenance, monkeypatch
):
    import os

    jobs = RecoverableJobs(fs_job_store)
    job = (await reserve(jobs, provenance)).record
    assert await jobs.claim(job["job_id"], job["owner"])
    result = dict(
        dataset_id=provenance.dataset_id,
        variable_id=provenance.variable_id,
        area=0,
        n_cells=1,
        summary_stats=[],
        series=[],
        transform={"type": "NoTransform"},
        zonal_statistic="mean",
    )
    replace = os.replace

    def fail_admission(source, destination):
        if destination.endswith(".admission.json"):
            raise OSError("admission write failed")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr("app.store.jobs.os.replace", fail_admission)
        with pytest.raises(JobContractError) as exc:
            await jobs.finish(
                job["job_id"],
                job["owner"],
                "SUCCESS",
                result=result,
                base_series={"timesteps": ["0001"], "mean": [1.0], "median": [1.0]},
            )
        assert exc.value.code == "COORDINATION_UNAVAILABLE"
    # The result preceded the success marker. Reading repairs admission and
    # preserves completed work instead of launching another extraction.
    assert (await jobs.get(job["job_id"]))["result"] == result
    assert not (await reserve(jobs, provenance)).created
    assert (await reserve(jobs, provenance, key="next")).created


async def test_expired_redis_lease_fences_an_already_prepared_completion(
    redis_job_store, provenance, monkeypatch
):
    import anyio

    jobs = RecoverableJobs(redis_job_store)
    job = (await reserve(jobs, provenance)).record
    assert await jobs.claim(job["job_id"], job["owner"])
    lease = f"extraction:lease:{job['job_id']}"
    assert await redis_job_store._client.pexpire(lease, 50)
    pipeline = redis_job_store._client.pipeline

    def delayed_pipeline(*args, **kwargs):
        pipe = pipeline(*args, **kwargs)
        execute = pipe.execute

        async def delayed_execute(*args, **kwargs):
            await anyio.sleep(0.1)
            return await execute(*args, **kwargs)

        pipe.execute = delayed_execute
        return pipe

    monkeypatch.setattr(redis_job_store._client, "pipeline", delayed_pipeline)
    assert not await jobs.finish(
        job["job_id"], job["owner"], "SUCCESS", result={"late": True}
    )
    assert await redis_job_store.get_job_status(job["job_id"]) is None
    record = await jobs.get(job["job_id"])
    assert record["status"] == "FAILED"
    assert record["error_code"] == "WORKER_LOST"
    assert (await reserve(jobs, provenance, key="replacement")).created


async def test_delayed_result_write_cannot_extend_absolute_retention(
    redis_job_store, provenance, monkeypatch
):
    import anyio

    jobs = RecoverableJobs(redis_job_store)
    job = (await reserve(jobs, provenance)).record
    assert await jobs.claim(job["job_id"], job["owner"])

    def near_expiry(state):
        state["meta"]["expires_at"] = time.time() + 0.05

    await redis_job_store.atomic_extraction(job["job_id"], near_expiry)
    pipeline = redis_job_store._client.pipeline

    def delayed_pipeline(*args, **kwargs):
        pipe = pipeline(*args, **kwargs)
        execute = pipe.execute

        async def delayed_execute(*args, **kwargs):
            await anyio.sleep(0.1)
            return await execute(*args, **kwargs)

        pipe.execute = delayed_execute
        return pipe

    monkeypatch.setattr(redis_job_store._client, "pipeline", delayed_pipeline)
    assert await jobs.finish(
        job["job_id"], job["owner"], "SUCCESS", result={"completed": True}
    )
    assert await redis_job_store.get_job_status(job["job_id"]) is None
    with pytest.raises(JobContractError) as exc:
        await jobs.get(job["job_id"])
    assert exc.value.code == "RESULT_EXPIRED"
