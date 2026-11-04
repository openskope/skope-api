"""Command-line entry point: `skope-release <command>`.

Findings go to standard output; the exit status is non-zero while any error
remains (VAL-002). Paths default to the environment the `release` Compose
service sets.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import PRODUCER_NAME, __version__
from .findings import ReleaseError, Report
from .plan import Producer


def _env_path(name: str, default: str | None = None) -> Path | None:
    value = os.environ.get(name, default)
    return Path(value) if value else None


def _producer() -> Producer:
    return Producer(PRODUCER_NAME, __version__, os.environ.get("PRODUCER_REVISION", "unknown"))


def _print_report(title: str, report: Report, as_json: bool) -> None:
    print(f"== {title}")
    if as_json:
        print(report.to_json())
    elif report.findings:
        print(report.to_text())
    else:
        print("no findings")


def _release_ids(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs or []:
        dataset, sep, release_id = pair.partition("=")
        if not sep:
            raise SystemExit(f"--release expects DATASET=RELEASE_ID, got {pair!r}")
        out[dataset] = release_id
    return out


def _dataset_dirs(args) -> list[Path]:
    root = args.datasets
    return [root / name for name in args.dataset]


def cmd_preflight(args) -> int:
    from .preflight import preflight_many

    plans, reports = preflight_many(
        _dataset_dirs(args), release_ids=_release_ids(args.release), mirror=args.source_mirror, producer=_producer()
    )
    failed = []
    for dataset, report in reports.items():
        _print_report(f"preflight {dataset}", report, args.json)
        if dataset in plans:
            print(f"declaration_digest: {plans[dataset].declaration_digest}")
        else:
            failed.append(dataset)
    if failed:
        print(f"preflight failed for: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_fetch(args) -> int:
    from .fetch import fetch_dataset
    from .preflight import DatasetFiles

    if args.source_mirror is None:
        raise SystemExit("fetch needs a source mirror: --source-mirror or SKOPE_RELEASE_SOURCE_MIRROR (OBS-014)")
    failed = []
    for directory in _dataset_dirs(args):
        report = Report()
        outcomes = fetch_dataset(DatasetFiles(directory), mirror=args.source_mirror, report=report)
        _print_report(f"fetch {directory.name}", report, args.json)
        for variable, outcome in outcomes:
            print(f"{variable}: {outcome}")
        if not report.ok:
            failed.append(directory.name)
    if failed:
        print(f"fetch failed for: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_build(args) -> int:
    from .build import build_and_publish
    from .preflight import preflight_many

    release_ids = _release_ids(args.release)
    missing = [d for d in args.dataset if d not in release_ids]
    if missing:
        raise SystemExit(f"assign a release ID with --release DATASET=ID for: {', '.join(missing)} (REL-006)")
    # MIG-003: every dataset is preflighted before any transformation starts.
    plans, reports = preflight_many(_dataset_dirs(args), release_ids=release_ids, mirror=args.source_mirror, producer=_producer())
    failed = [d for d in args.dataset if d not in plans]
    for dataset in failed:
        _print_report(f"preflight {dataset}", reports[dataset], args.json)
    args.release_root.mkdir(parents=True, exist_ok=True)
    for dataset, plan in plans.items():
        try:
            published, report = build_and_publish(plan, args.release_root)
        except ReleaseError as exc:
            _print_report(f"build {dataset}: {exc.stage} failed", exc.report, args.json)
            failed.append(dataset)
            continue
        except Exception as exc:  # noqa: BLE001 - one dataset's failure must not stop the others (MIG-003)
            print(f"== build {dataset} failed: {exc!r}")
            failed.append(dataset)
            continue
        _print_report(f"build {dataset}", report, args.json)
        print("pin values (deploy/releases/<environment>.yml):")
        print(f"  - dataset: {dataset}")
        print(f"    release_id: {published.release_id}")
        print(f"    declaration_digest: {published.declaration_digest}")
        print(f"    manifest_sha256: {published.manifest_sha256}")
        print("ledger entry (datasets/<dataset>/releases.yml):")
        print(f"  - release_id: {published.release_id}")
        print(f"    declaration_digest: {published.declaration_digest}")
    if failed:
        print(f"failed datasets: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_verify(args) -> int:
    from .manifest import manifest_sha256, verify_release

    status = 0
    for path in args.release_path:
        report = Report()
        manifest = verify_release(path, report, full=not args.quick)
        _print_report(f"verify {path.name}", report, args.json)
        if manifest is None:
            status = 1
        else:
            print(f"release_id: {manifest.release_id}")
            print(f"declaration_digest: {manifest.declaration.digest}")
            print(f"manifest_sha256: {manifest_sha256(path)}")
    return status


def cmd_verify_reproducible(args) -> int:
    from .build import verify_reproducible
    from .documents import load_document
    from .models import ReleaseManifest
    from .preflight import DatasetFiles, preflight_dataset

    report = Report()
    manifest = load_document(ReleaseManifest, args.release_path / "release-manifest.json", requirement="MAN-001", report=report)
    if manifest is None:
        _print_report("verify-reproducible", report, args.json)
        return 1
    files = DatasetFiles(args.datasets / manifest.dataset.id)
    producer = Producer(manifest.producer.name, manifest.producer.version, manifest.producer.revision)
    plan = preflight_dataset(files, release_id=None, mirror=args.source_mirror, producer=producer, report=report)
    if plan is None:
        _print_report("verify-reproducible: preflight", report, args.json)
        return 1
    if plan.declaration_digest != manifest.declaration.digest:
        report.add("MAN-011", "the current declaration's digest differs from the release's; it is a different declaration")
        _print_report("verify-reproducible", report, args.json)
        return 1
    plan = preflight_dataset(files, release_id=manifest.release_id, mirror=args.source_mirror, producer=producer, report=Report())
    result = verify_reproducible(plan, args.release_path)
    _print_report(f"verify-reproducible {manifest.release_id}", result, args.json)
    return 0 if result.ok else 1


def cmd_export_schemas(args) -> int:
    from .documents import export_schemas

    for path in export_schemas():
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skope-release", description="Build, validate, and publish SKOPE dataset releases.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    parser.add_argument("--datasets", type=Path, default=_env_path("SKOPE_RELEASE_DATASETS", "datasets"), help="authoring root (datasets/<id>/)")
    parser.add_argument("--source-mirror", type=Path, default=_env_path("SKOPE_RELEASE_SOURCE_MIRROR"), help="local <bucket>/<key> copies of s3:// sources")
    parser.add_argument("--release-root", type=Path, default=_env_path("SKOPE_RELEASE_ROOT", "releases"))
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("preflight", help="validate declarations and sources; write nothing (VAL-004)")
    p.add_argument("dataset", nargs="+")
    p.add_argument("--release", action="append", metavar="DATASET=RELEASE_ID", help="also check the release ID")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("fetch", help="copy declared sources into the source mirror (OBS-014)")
    p.add_argument("dataset", nargs="+")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", help="build and publish one release per dataset")
    p.add_argument("dataset", nargs="+")
    p.add_argument("--release", action="append", metavar="DATASET=RELEASE_ID", required=True)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("verify", help="verify published releases against their manifests (TXN-011)")
    p.add_argument("release_path", nargs="+", type=Path)
    p.add_argument("--quick", action="store_true", help="presence and size only (the startup checks)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("verify-reproducible", help="rebuild a release in scratch and byte-compare it")
    p.add_argument("release_path", type=Path)
    p.set_defaults(func=cmd_verify_reproducible)

    p = sub.add_parser("export-schemas", help="regenerate the committed JSON Schemas")
    p.set_defaults(func=cmd_export_schemas)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command is None:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
