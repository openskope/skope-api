"""Build, exercise, scan and retain the exact images used for deployment.

Called through the root Make targets. Images stay local until CI uploads the
archive. Deployments can download a CI bundle and invoke the root deploy target.
"""

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("RUNTIME_IMAGES_DIR", "runtime-images")).resolve()
REVISION_LABEL = "org.opencontainers.image.revision"


def now():
    return datetime.now(timezone.utc).isoformat()


def run(*args, capture=False):
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        text=True,
    )
    return result.stdout.strip() if capture else None


def inspect(image):
    return json.loads(run("docker", "image", "inspect", image, capture=True))[0]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def selection_images(selection, environment, revision):
    """Reject mismatched checkouts and anything other than exact image selection."""
    if set(selection) != {"x-skope", "services"}:
        raise ValueError(
            "expected a CI image selection, not an arbitrary Compose override"
        )
    if selection["x-skope"] != {"revision": revision, "environment": environment}:
        raise ValueError("image selection does not match the checkout and environment")
    if set(selection["services"]) != {"server", "titiler", "redis"}:
        raise ValueError("the selection must identify API, TiTiler and Redis")
    images = {}
    for service, config in selection["services"].items():
        if set(config) != {"image", "pull_policy"} or config["pull_policy"] != "never":
            raise ValueError("image selections may only select locally loaded images")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", config["image"]):
            raise ValueError("image selections must use immutable image IDs")
        images[service] = config["image"]
    return images


def clean_revision():
    if run("git", "status", "--porcelain", capture=True):
        raise ValueError(
            "deploy tested images from a clean checkout of their source commit"
        )
    return run("git", "rev-parse", "HEAD", capture=True)


def bundle_images(path, environment, revision):
    images = selection_images(json.loads(path.read_text()), environment, revision)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    approved = scanned_images(manifest)
    if manifest["revision"] != revision:
        raise ValueError("scan manifest was built from a different source revision")
    for service, image in images.items():
        if image != approved[environment if service == "server" else service]:
            raise ValueError(
                f"{service}: image selection does not match the scanned bundle"
            )
    return images


def check(path, environment):
    revision = clean_revision()
    images = bundle_images(path, environment, revision)
    for service, image in images.items():
        info = inspect(image)
        if info["Id"] != image:
            raise ValueError(f"{service}: loaded image does not match the selection")
        if info["Config"]["Labels"].get(REVISION_LABEL) != revision:
            raise ValueError(
                f"{service}: image was built from a different source revision"
            )
        if (
            service == "server"
            and f"ENVIRONMENT={environment}" not in info["Config"]["Env"]
        ):
            raise ValueError("API image was built for a different environment")
    print("Loaded images match the tested selection and source revision.")


def github_records(endpoint, collection):
    output = run(
        "gh",
        "api",
        "--paginate",
        endpoint,
        "--jq",
        f".{collection}[] | @json",
        capture=True,
    )
    return [json.loads(line) for line in output.splitlines() if line]


def ci_bundle(revision, repository, cache, environment):
    runs = github_records(
        f"repos/{repository}/actions/workflows/test.yml/runs"
        f"?head_sha={revision}&status=success&per_page=100",
        "workflow_runs",
    )
    for workflow in runs:
        if workflow["head_sha"] != revision or workflow["conclusion"] != "success":
            continue
        if workflow["event"] not in ("push", "schedule", "workflow_dispatch"):
            continue
        run_id = workflow["id"]
        name = f"runtime-images-{run_id}-{workflow['run_attempt']}"
        artifacts = github_records(
            f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100",
            "artifacts",
        )
        artifact = next(
            (
                item
                for item in artifacts
                if item["name"] == name and not item["expired"]
            ),
            None,
        )
        if artifact is None:
            continue
        destination = cache / repository / f"{name}-{artifact['id']}"
        if not destination.is_dir():
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Failed or interrupted downloads never become cached bundles.
            with tempfile.TemporaryDirectory(
                prefix=".download-", dir=destination.parent
            ) as directory:
                download = Path(directory) / "bundle"
                run(
                    "gh",
                    "run",
                    "download",
                    run_id,
                    "--repo",
                    repository,
                    "--name",
                    name,
                    "--dir",
                    download,
                )
                bundle_images(
                    download / f"images-{environment}.json", environment, revision
                )
                if not (download / "images.tar.gz").is_file():
                    raise ValueError("CI bundle is missing images.tar.gz")
                download.rename(destination)
        print(
            f"Using {repository} CI run {run_id}, attempt {workflow['run_attempt']}",
            flush=True,
        )
        return destination
    raise ValueError(
        f"no retained successful CI bundle for {revision} in {repository}; "
        "run test.yml for this commit, or supply DEPLOY_IMAGES for a retained local bundle"
    )


def deploy_ci(environment):
    revision = clean_revision()
    prefix = "STAGING" if environment == "staging" else "PRODUCTION"
    repository = os.environ.get(f"{prefix}_CI_REPOSITORY", "openskope/skope-api")
    default_cache = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "skope-api/image-bundles"
    )
    cache = (
        Path(os.environ.get(f"{prefix}_BUNDLES_DIR", default_cache))
        .expanduser()
        .resolve()
    )
    if cache == ROOT or ROOT in cache.parents:
        raise ValueError(f"{prefix}_BUNDLES_DIR must be outside the source checkout")
    bundle = ci_bundle(revision, repository, cache, environment)
    selection = bundle / f"images-{environment}.json"
    bundle_images(selection, environment, revision)
    run("docker", "load", "--input", bundle / "images.tar.gz")
    run("make", "deploy", f"ENVIRONMENT={environment}", f"DEPLOY_IMAGES={selection}")


def test():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # A failed local rerun must not leave an older bundle looking current.
    for name in (
        "manifest.json",
        "images.tar.gz",
        "images-staging.json",
        "images-prod.json",
    ):
        (OUTPUT / name).unlink(missing_ok=True)
    revision = run("git", "rev-parse", "HEAD", capture=True)
    if run("git", "status", "--porcelain", capture=True):
        revision += "-dirty"
    refresh = os.environ.get("RUNTIME_REFRESH", "0") == "1"
    build_flags = ["--pull", "--no-cache"] if refresh else ["--pull"]
    tags = {
        "titiler": "skope-runtime-titiler:tested",
        "redis": "skope-runtime-redis:tested",
    }
    for service in ("titiler", "redis"):
        run(
            "docker",
            "build",
            *build_flags,
            "-f",
            f"deploy/Dockerfile.{service}",
            "--label",
            f"{REVISION_LABEL}={revision}",
            "-t",
            tags[service],
            ".",
        )
    for environment in ("dev", "staging", "prod"):
        tag = f"skope-runtime-api:{environment}"
        tags[environment] = tag
        run(
            "docker",
            "build",
            *build_flags,
            "-f",
            "deploy/Dockerfile",
            "--build-arg",
            f"ENVIRONMENT={environment}",
            "--build-arg",
            f"REQUIREMENTS={'dev' if environment == 'dev' else 'production'}",
            "--label",
            f"{REVISION_LABEL}={revision}",
            "-t",
            tag,
            ".",
        )
    images = {name: inspect(tag)["Id"] for name, tag in tags.items()}

    with tempfile.TemporaryDirectory(prefix="skope-runtime-") as directory:
        fixture = Path(directory)
        fixture.chmod(0o755)
        mounts = [
            "-v",
            f"{ROOT / 'scripts/runtime-smoke.py'}:/smoke.py:ro",
            "-v",
            f"{ROOT / 'timeseries/app/tests/release_builder.py'}:/release_builder.py:ro",
        ]
        user = f"{os.getuid()}:{os.getgid()}"
        run(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            user,
            *mounts,
            "-v",
            f"{fixture}:/fixtures",
            images["dev"],
            "python",
            "/smoke.py",
            "prepare",
        )
        host_pin = fixture / "host-pin.yml"
        host_pin.write_text(f"release_root: {fixture / 'releases'}\n")
        for environment in ("staging", "prod"):
            # Exercise the deployed HTTP probe and startup allowance, rather
            # than a separate TCP-only approximation of readiness.
            deployed = json.loads(
                run(
                    "env",
                    f"SKOPE_RELEASE_ROOT={fixture / 'releases'}",
                    f"SKOPE_OPERATOR_UID={os.getuid()}",
                    f"SKOPE_OPERATOR_GID={os.getgid()}",
                    "docker",
                    "compose",
                    "-f",
                    "deploy/compose/base.yml",
                    "-f",
                    f"deploy/compose/{environment}.yml",
                    "config",
                    "--format",
                    "json",
                    capture=True,
                )
            )
            compose_path = fixture / f"{environment}.json"
            config = {
                "services": {
                    "server": {
                        "image": images[environment],
                        "healthcheck": deployed["services"]["server"]["healthcheck"],
                        "environment": {
                            "REDIS_URL": "redis://redis:6379",
                            "SENTRY_DSN": "",
                            "RELEASE_PIN_PATH": "/fixtures/pin.json",
                            "DISPLAY_PREFERENCES_PATH": "/fixtures/preferences.json",
                        },
                        "volumes": [
                            f"{fixture}:/fixtures:ro",
                            f"{fixture / 'releases'}:/releases:ro",
                        ],
                        "depends_on": {
                            name: {"condition": "service_healthy"}
                            for name in ("redis", "titiler")
                        },
                    },
                    "titiler": {
                        "image": images["titiler"],
                        "volumes": [f"{fixture / 'releases'}:/releases:ro"],
                        "healthcheck": deployed["services"]["titiler"]["healthcheck"],
                    },
                    "redis": {
                        "image": images["redis"],
                        "command": ["redis-server", "--save", "", "--appendonly", "no"],
                        "healthcheck": {
                            "test": ["CMD", "redis-cli", "ping"],
                            "interval": "1s",
                            "timeout": "3s",
                            "retries": 30,
                        },
                    },
                    "check": {
                        "image": images["dev"],
                        "user": user,
                        "volumes": [
                            f"{ROOT / 'scripts/runtime-smoke.py'}:/smoke.py:ro"
                        ],
                        "command": ["python", "/smoke.py", "check"],
                        "profiles": ["test"],
                    },
                },
                # No external access, host ports, credentials or real datasets.
                "networks": {"default": {"internal": True}},
            }
            write_json(compose_path, config)
            compose = [
                "docker",
                "compose",
                "-p",
                f"skope-runtime-{os.getpid()}-{environment}",
                "-f",
                compose_path,
            ]
            try:
                # Use the shared deploy recipe with synthetic releases and
                # already-built images. The dev preparation path needs no CI
                # bundle or clean checkout; this fixture has nothing to build.
                deploy_log = fixture / f"{environment}-deploy.jsonl"
                for skip_full, mode in (("0", "full"), ("1", "metadata")):
                    run(
                        "make",
                        "deploy-dev",
                        f"COMPOSE={shlex.join(str(arg) for arg in compose)}",
                        f"PIN_FILE={host_pin}",
                        f"DEPLOY_LOG={deploy_log}",
                        "DEPLOY_IMAGES=",
                        f"SKIP_FULL_VERIFICATION={skip_full}",
                    )
                    entry = json.loads(deploy_log.read_text().splitlines()[-1])
                    assert entry["images"] == {
                        "server": images[environment],
                        "titiler": images["titiler"],
                        "redis": images["redis"],
                    }, entry
                    assert entry["releases"][0]["verification"] == mode, entry
                    assert entry["releases"][0]["result"] == "passed", entry
                run(*compose, "--profile", "test", "run", "--rm", "--no-deps", "check")
            except subprocess.CalledProcessError:
                run(*compose, "logs", "--no-color")
                raise
            finally:
                run(
                    *compose,
                    "--profile",
                    "test",
                    "down",
                    "--volumes",
                    "--remove-orphans",
                )

    for environment in ("staging", "prod"):
        write_json(
            OUTPUT / f"images-{environment}.json",
            {
                "x-skope": {"revision": revision, "environment": environment},
                "services": {
                    service: {"image": image, "pull_policy": "never"}
                    for service, image in {
                        "server": images[environment],
                        "titiler": images["titiler"],
                        "redis": images["redis"],
                    }.items()
                },
            },
        )
    write_json(
        OUTPUT / "manifest.json",
        {
            "revision": revision,
            "built_at": now(),
            "maintenance_refresh": refresh,
            "images": {
                name: {
                    "id": images[name],
                    "platform": f"{inspect(images[name])['Os']}/{inspect(images[name])['Architecture']}",
                }
                for name in ("staging", "prod", "titiler", "redis")
            },
        },
    )
    print(f"Runtime checks passed; image selections are in {OUTPUT}")


def tested_images():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    return {name: value["id"] for name, value in manifest["images"].items()}


def scanned_images(manifest):
    if manifest.get("scan", {}).get("status") != "passed":
        raise ValueError(
            "the bundle must pass its vulnerability scan before export or deployment"
        )
    return {name: value["id"] for name, value in manifest["images"].items()}


def scan():
    # A report is retained even when a fixable vulnerability blocks promotion.
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    manifest["scan"] = {"status": "running", "policy": "fixable-high-critical"}
    write_json(OUTPUT / "manifest.json", manifest)
    run("docker", "pull", "aquasec/trivy:latest")
    failed = False
    for name, image in tested_images().items():
        with (OUTPUT / f"scan-{name}.txt").open("w") as report:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    "/var/run/docker.sock:/var/run/docker.sock",
                    "-v",
                    "skope-trivy-cache:/root/.cache/trivy",
                    "aquasec/trivy:latest",
                    "image",
                    "--scanners",
                    "vuln",
                    "--severity",
                    "HIGH,CRITICAL",
                    "--ignore-unfixed",
                    "--exit-code",
                    "1",
                    image,
                ],
                cwd=ROOT,
                stdout=report,
                stderr=subprocess.STDOUT,
            )
        print((OUTPUT / f"scan-{name}.txt").read_text())
        failed |= result.returncode != 0
    manifest["scan"].update(status="failed" if failed else "passed", checked_at=now())
    write_json(OUTPUT / "manifest.json", manifest)
    if failed:
        raise ValueError("runtime image vulnerability scan failed; see scan reports")


def export():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    images = sorted(set(scanned_images(manifest).values()))
    partial = OUTPUT / "images.tar.gz.partial"
    try:
        with gzip.open(partial, "wb", compresslevel=1) as archive:
            with subprocess.Popen(
                ["docker", "image", "save", *images], stdout=subprocess.PIPE
            ) as process:
                shutil.copyfileobj(process.stdout, archive)
                if process.wait() != 0:
                    raise ValueError("could not archive tested images")
        partial.replace(OUTPUT / "images.tar.gz")
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "test",
            "scan",
            "export",
            "check",
            "deploy-staging",
            "deploy-production",
        ],
    )
    parser.add_argument("selection", nargs="?", type=Path)
    parser.add_argument("environment", nargs="?", choices=["dev", "staging", "prod"])
    args = parser.parse_args()
    try:
        if args.command == "check":
            if not args.selection or not args.environment:
                parser.error("check needs a selection file and environment")
            check(args.selection, args.environment)
        else:
            {
                "test": test,
                "scan": scan,
                "export": export,
                "deploy-staging": lambda: deploy_ci("staging"),
                "deploy-production": lambda: deploy_ci("prod"),
            }[args.command]()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
