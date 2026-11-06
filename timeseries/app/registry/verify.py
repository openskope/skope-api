"""Promotion: fully verify the pinned releases before a deploy (release spec TXN-011).

    python -m app.registry.verify [--record PATH]

Runs every startup check (PIN-004) and also compares every file's checksum
with its manifest. Prints one JSON line per verified release and exits 0, or
logs each failed check and exits 1. With `--record`, it also writes the verified
releases to PATH, for `app.registry.deployment` to add to the deployment log
once the deploy has succeeded.
"""

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.registry.compose import ReleaseRefused, load_pin, log_refusals, verify_releases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.registry.verify")
    parser.add_argument(
        "--record", type=Path, help="also write the verified releases here"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    pin = load_pin(Path(settings.release_pin_path))
    try:
        registry = verify_releases(pin, Path(settings.release_root), full=True)
    except ReleaseRefused as exc:
        log_refusals(exc.refusals)
        print(f"verification failed: {len(exc.refusals)} check(s); see the log")
        return 1
    verified = []
    for pinned in pin.releases:
        release = registry.get(pinned.dataset)
        verified.append(
            {
                "dataset": pinned.dataset,
                "release_id": pinned.release_id,
                "declaration_digest": pinned.declaration_digest,
                "manifest_sha256": pinned.manifest_sha256,
                "path": str(release.path),
                "verification": "full",
                "result": "passed",
            }
        )
        print(json.dumps(verified[-1], sort_keys=True))
    if args.record is not None:
        args.record.write_text(json.dumps({"releases": verified}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
