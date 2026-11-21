"""Report the benchmark results as Markdown, and with --select apply the selection rule.

Standard library only. Each setting and cache mode gets its own section.

Without --select (stage 1), the report holds measurements only, sorted by total
extraction time: what stage 2's candidates are chosen from.

With --select (stage 2), the rule:

1. Rank by extraction: the sum of the median extraction times of the four
   areas over every timestep the variants hold.
2. Ties: two layouts tie when their difference is within 2 standard errors of
   it, each median's standard error estimated from the median absolute
   deviation of its runs. A tie goes to the faster map views framed on the
   areas, then the smaller release, then fewer files.
3. Map-view limit: the first layout in that order whose map views framed on
   each area are at most --max-view-loss seconds per step slower than the
   fastest layout's on the same area.
4. Limits: release size relative to the smallest, and file count for a
   12-variable release.

Usage:
    python3 analyze.py results/*.jsonl
    python3 analyze.py --select results/stage2/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict

FULL_TIMESTEPS = 1898
VARIABLES = 12
AREAS = ("point", "small", "medium", "large")
HOST_MEMORY_GB = 16
CONCURRENT_EXTRACTIONS = 4  # one per API worker
SECTOR_BYTES = 512


def op_key(task: dict) -> str:
    if task["op"] == "extract":
        return f"extract:{task['area']}:{task['years'] or 'all'}"
    return f"view:{task['view']}:{task['steps']}:{task.get('tile_path', 'direct')}"


def median_se(values: list[float]) -> float:
    """Standard error of the median, from the median absolute deviation (robust to one-off outliers)."""
    if len(values) < 2:
        return 0.0
    m = statistics.median(values)
    sigma = 1.4826 * statistics.median(abs(v - m) for v in values)
    return 1.2533 * sigma / math.sqrt(len(values))


def load(paths: list[str]):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (setting, mode) -> variant -> op -> records
    metas, errors = {}, []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if "error" in r:
                    errors.append(r)
                    continue
                t = r["task"]
                groups[(r["setting"], t["mode"])][t["name"]][op_key(t)].append(r)
                metas[t["name"]] = r["variant_meta"]
    return groups, metas, errors


def seconds(ops: dict, key: str) -> tuple[float, float] | None:
    """Median seconds and its standard error."""
    records = ops.get(key)
    if not records:
        return None
    values = [r["seconds"] for r in records]
    return statistics.median(values), median_se(values)


def median_of(ops: dict, key: str, value) -> float | None:
    values = [v for r in ops.get(key, []) if (v := value(r)) is not None]
    return statistics.median(values) if values else None


def disk_per_step(r: dict, field: str) -> float | None:
    if not r.get("disk"):
        return None
    steps = len(r["step_seconds"]) if "step_seconds" in r else 1
    return r["disk"][field] / steps


def release_size(meta: dict) -> float:
    return meta["size_bytes"] * FULL_TIMESTEPS / meta["keys"] * VARIABLES


def release_files(meta: dict) -> int:
    return VARIABLES * math.ceil(FULL_TIMESTEPS / meta["chunk"])


def rows_for(variants: dict, metas: dict) -> dict:
    rows = {}
    for name, ops in variants.items():
        extract = {a: seconds(ops, f"extract:{a}:all") for a in AREAS}
        if any(e is None for e in extract.values()):
            continue
        views = {a: seconds(ops, f"view:area-{a}:12:vrt") for a in AREAS}
        meta = metas[name]
        rows[name] = {
            "extract": extract,
            "total": sum(m for m, _ in extract.values()),
            "total_se": math.sqrt(sum(se * se for _, se in extract.values())),
            "views": views if all(views.values()) else None,
            "fit": seconds(ops, "view:fit:12:vrt"),
            "native": seconds(ops, "view:native:12:vrt"),
            "peak": {a: median_of(ops, f"extract:{a}:all", lambda r: r["peak_rss_delta"]) for a in AREAS},
            "large_reads": median_of(ops, "extract:large:all", lambda r: disk_per_step(r, "reads")),
            "large_bytes": median_of(ops, "extract:large:all", lambda r: disk_per_step(r, "sectors")),
            "view_bytes": [median_of(ops, f"view:area-{a}:12:vrt", lambda r: disk_per_step(r, "sectors"))
                           for a in AREAS],
            "size": release_size(meta),
            "files": release_files(meta),
            "build": meta["build_seconds"],
        }
        if rows[name]["views"]:
            rows[name]["view_mean"] = statistics.mean(m for m, _ in rows[name]["views"].values())
    return rows


def ranked(rows: dict) -> list[str]:
    """Extraction order, with each tie group reordered by map views, then release size, then file count."""
    remaining, order = dict(rows), []
    while remaining:
        leader = min(remaining, key=lambda n: remaining[n]["total"])
        lead = remaining[leader]
        tied = [n for n, r in remaining.items()
                if r["total"] - lead["total"] <= 2 * math.sqrt(r["total_se"] ** 2 + lead["total_se"] ** 2)]
        tied.sort(key=lambda n: (remaining[n].get("view_mean", math.inf), remaining[n]["size"],
                                 remaining[n]["files"]))
        order += tied
        for n in tied:
            del remaining[n]
    return order


def select(rows: dict, limits: dict) -> tuple[str | None, list[str]]:
    """The selected layout, and the ranked order it was taken from."""
    smallest = min(r["size"] for r in rows.values())
    for r in rows.values():
        r["ok"] = r["size"] <= limits["size_ratio"] * smallest and r["files"] <= limits["files"]
    with_views = [r for r in rows.values() if r["views"]]
    best = {a: min(r["views"][a][0] for r in with_views) for a in AREAS} if with_views else {}
    for r in rows.values():
        r["view_loss"] = max(r["views"][a][0] - best[a] for a in AREAS) if r["views"] else None
    order = ranked(rows)
    for name in order:
        r = rows[name]
        if r["ok"] and r["view_loss"] is not None and r["view_loss"] <= limits["view_loss"]:
            return name, order
    return None, order


def fmt(value: tuple[float, float] | None, scale: float = 1, digits: int = 2) -> str:
    return "-" if value is None else f"{value[0] * scale:.{digits}f} ± {value[1] * scale:.{digits}f}"


def mb(value: float | None, scale: float = 1) -> str:
    return "-" if value is None else f"{value * scale / 1e6:.0f}"


def report(rows: dict, metas: dict, variants: dict, selection: tuple[str | None, list[str]] | None) -> None:
    order = selection[1] if selection else sorted(rows, key=lambda n: rows[n]["total"])
    years = {metas[n]["keys"] for n in rows}
    span = f"{years.pop()} years" if len(years) == 1 else "every timestep each variant holds"
    print(f"### Extraction\n\nMedian seconds ± standard error, over {span}. Sorted "
          + ("in the selection order." if selection else "by total.") + "\n")
    print("| Variant | Total | " + " | ".join(AREAS) + " |")
    print("| --- | --- |" + " --- |" * len(AREAS))
    for n in order:
        r = rows[n]
        print(f"| {n} | {r['total']:.2f} ± {r['total_se']:.2f} | "
              + " | ".join(fmt(r["extract"][a]) for a in AREAS) + " |")

    framed = AREAS if any(r["views"] for r in rows.values()) else ()
    print("\n### Map views\n\nMedian seconds per slider step, one-band tile reads. "
          + ("The views framed on each area are the visualize map (640 × 550 px); " if framed else "")
          + "`fit` and `native` show the whole region (1280 × 800 px).\n")
    print("| Variant | " + "".join(f"{a} | " for a in framed) + "fit | native |")
    print("| --- |" + " --- |" * (len(framed) + 2))
    for n in order:
        r = rows[n]
        area_cells = [fmt(r["views"][a]) if r["views"] else "-" for a in framed]
        print(f"| {n} | " + "".join(f"{c} | " for c in area_cells) + f"{fmt(r['fit'])} | {fmt(r['native'])} |")

    print(f"\n### Memory, disk and release\n\nPeak memory per extraction, median MB. "
          f"\"{CONCURRENT_EXTRACTIONS} × large\" is {CONCURRENT_EXTRACTIONS} large-area extractions at once, one per "
          f"API worker, against the host's {HOST_MEMORY_GB} GB. Disk reads and MB read come from the block device "
          f"(cold reads only); for map views, per step, averaged over the four areas. Release size and files are "
          f"for {VARIABLES} variables over {FULL_TIMESTEPS} years; build time is for this one variant.\n")
    print("| Variant | " + " | ".join(f"peak, {a}" for a in AREAS)
          + f" | {CONCURRENT_EXTRACTIONS} × large (GB) | reads, large | MB read, large | MB read per view step "
            "| Release (GB) | Files | Build (s) |")
    print("| --- |" + " --- |" * (len(AREAS) + 7))
    for n in order:
        r = rows[n]
        four = r["peak"]["large"]
        view_bytes = [b for b in r["view_bytes"] if b is not None]
        print(f"| {n} | " + " | ".join(mb(r["peak"][a]) for a in AREAS)
              + f" | {'-' if four is None else f'{CONCURRENT_EXTRACTIONS * four / 1e9:.1f}'}"
              + f" | {'-' if r['large_reads'] is None else f'{r['large_reads']:.0f}'}"
              + f" | {mb(r['large_bytes'], SECTOR_BYTES)}"
              + f" | {mb(statistics.mean(view_bytes) if view_bytes else None, SECTOR_BYTES)}"
              + f" | {r['size'] / 1e9:.1f} | {r['files']} | {r['build']:.0f} |")

    if selection:
        best, order = selection
        print("\n### Selection\n")
        print("| Order | Variant | Worst view loss per step (s) | Limits |")
        print("| --- | --- | --- | --- |")
        for i, n in enumerate(order, 1):
            r = rows[n]
            loss = "-" if r["view_loss"] is None else f"{r['view_loss']:.2f}"
            print(f"| {i} | {n} | {loss} | {'ok' if r['ok'] else 'FAILS'} |")
        print(f"\nSelected: **{best or 'none'}**\n")

    twins = defaultdict(dict)
    for n in rows:
        m = metas[n]
        twins[(m["block"], m["interleave"], m["chunk"])][m["dtype"]] = n
    pairs = [(t["uint16"], t["uint32"]) for t in twins.values() if {"uint16", "uint32"} <= t.keys()]
    if pairs:
        print("\n### UInt16 against UInt32\n\nThe same layout in both data types; ratios are UInt16 ÷ UInt32.\n")
        print("| Layout | Release size | Extraction total | Map views framed on the areas | Peak, large |")
        print("| --- | --- | --- | --- | --- |")
        for u16, u32 in sorted(pairs):
            a, b = rows[u16], rows[u32]
            views = f"{a['view_mean'] / b['view_mean']:.2f}" if "view_mean" in a and "view_mean" in b else "-"
            peak = (f"{a['peak']['large'] / b['peak']['large']:.2f}"
                    if a["peak"]["large"] and b["peak"]["large"] else "-")
            print(f"| {u16.removesuffix('-uint16')} | {a['size'] / b['size']:.2f} | {a['total'] / b['total']:.2f} "
                  f"| {views} | {peak} |")

    keys = sorted({k for ops in variants.values() for k in ops})
    print("\n### Identical outputs\n\nDistinct result hashes across variants, per operation:\n")
    for key in keys:
        by_dtype = defaultdict(set)
        for name, ops in variants.items():
            for r in ops.get(key, []):
                by_dtype["all"].add(r["digest"])
                by_dtype[metas[name]["dtype"]].add(r["digest"])
        flag = "" if len(by_dtype["all"]) == 1 else "  <-- differs"
        detail = ", ".join(f"{d}: {len(s)}" for d, s in sorted(by_dtype.items()) if d != "all")
        print(f"- `{key}`: {len(by_dtype['all'])} ({detail}){flag}")

    print("\n### Every extraction\n\nMedian seconds, by area and number of years.\n")
    ranges = sorted({k.split(":")[2] for k in keys if k.startswith("extract:")},
                    key=lambda y: math.inf if y == "all" else int(y))
    print("| Variant | Area | " + " | ".join(ranges) + " |")
    print("| --- | --- |" + " --- |" * len(ranges))
    for n in order:
        for a in AREAS:
            cells = [seconds(variants[n], f"extract:{a}:{y}") for y in ranges]
            print(f"| {n} | {a} | " + " | ".join("-" if c is None else f"{c[0]:.2f}" for c in cells) + " |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    p.add_argument("--select", action="store_true", help="apply the selection rule (stage 2)")
    p.add_argument("--max-view-loss", type=float, default=0.75, help="seconds per step, on any area")
    p.add_argument("--max-size-ratio", type=float, default=1.5)
    p.add_argument("--max-files", type=int, default=5000)
    p.add_argument("--include", nargs="*", help="report only variants matching these globs")
    args = p.parse_args()
    limits = {"view_loss": args.max_view_loss, "size_ratio": args.max_size_ratio, "files": args.max_files}

    groups, metas, errors = load(args.results)
    if args.include:
        import fnmatch

        for variants in groups.values():
            for name in [n for n in variants if not any(fnmatch.fnmatch(n, g) for g in args.include)]:
                del variants[name]
    print(f"# Encoding experiment results\n\n{sum(len(v) for v in groups.values())} variant runs, "
          f"{len(errors)} failed measurements.\n")
    for (setting, mode), variants in sorted(groups.items()):
        print(f"## Setting {setting}, {mode}\n")
        rows = rows_for(variants, metas)
        if not rows:
            print("No complete extraction measurements here (extraction and views run in separate images; "
                  "pass both result files).\n")
            continue
        report(rows, metas, variants, select(rows, limits) if args.select else None)
        print()
    if errors:
        print("## Failed measurements\n")
        for r in errors[:20]:
            print(f"- {r['setting']} {r['task']['name']} {op_key(r['task'])}: {r['error']}")


if __name__ == "__main__":
    main()
