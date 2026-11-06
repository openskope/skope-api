"""The deployment log: what an environment served, from when (release spec TXN-011,
release consumption spec DISP-011).

    python -m app.registry.deployment PENDING LOG --environment ENV --commit COMMIT
        --deployed-by USER [--image SERVICE=IMAGE_ID ...]

`make deploy-<env>` runs this once the containers are up and healthy. It reads
the releases that `app.registry.verify --record PENDING` verified, adds when,
where, by whom, from which commit, and with which images the deploy went live,
appends that as one JSON line to LOG, and removes PENDING. A deploy that fails
verification or startup adds no line.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def record(
    pending: Path,
    log: Path,
    *,
    environment: str,
    commit: str,
    deployed_by: str,
    images: dict[str, str],
    now: datetime | None = None,
) -> dict:
    verified = json.loads(pending.read_text())
    entry = {
        "deployed_at": (now or datetime.now(timezone.utc)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "environment": environment,
        "deployed_by": deployed_by,
        # The images, and so the palettes file, were built from this commit
        # (DISP-011); "-dirty" marks uncommitted changes in the checkout.
        "commit": commit,
        "images": dict(sorted(images.items())),
        "releases": verified["releases"],
    }
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    pending.unlink()
    return entry


def _image(pair: str) -> tuple[str, str]:
    service, sep, image = pair.partition("=")
    if not sep or not service or not image:
        raise argparse.ArgumentTypeError(f"expected SERVICE=IMAGE_ID, got {pair!r}")
    return service, image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.registry.deployment")
    parser.add_argument("pending", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--deployed-by", required=True)
    parser.add_argument("--image", type=_image, action="append", default=[])
    args = parser.parse_args(argv)
    entry = record(
        args.pending,
        args.log,
        environment=args.environment,
        commit=args.commit,
        deployed_by=args.deployed_by,
        images=dict(args.image),
    )
    print(f"recorded the {entry['environment']} deploy in {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
