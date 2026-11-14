"""The skope-release command (AT-020, AT-034; MIG-003, VAL-002, VAL-004)."""

from pathlib import Path

from skope_release.cli import main


def tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_batch_with_one_failing_dataset(temporal_dataset, tmp_path, monkeypatch, capsys):
    """Failures are reported and skipped; the other datasets publish."""
    temporal_dataset(dataset_id="good")
    bad = temporal_dataset(dataset_id="bad")
    (bad / "curated.yml").write_text((bad / "curated.yml").read_text().replace("end: '0104'", "end: '0105'"))
    monkeypatch.setenv("PRODUCER_REVISION", "abc123")
    root = tmp_path / "releases"
    code = main([
        "--datasets", str(tmp_path / "datasets"), "--release-root", str(root), "build", "good", "bad",
        "--release", "good=good-r-2026.10.07", "--release", "bad=bad-r-2026.10.07",
    ])
    out = capsys.readouterr().out
    assert code == 1 and "failed datasets: bad" in out
    assert "OBS-005" in out
    assert (root / "good-r-2026.10.07" / "release-manifest.json").is_file()
    assert not (root / "bad-r-2026.10.07").exists()
    assert "manifest_sha256:" in out and "declaration_digest:" in out


def test_build_needs_release_ids(temporal_dataset, tmp_path):
    temporal_dataset()
    try:
        main(["--datasets", str(tmp_path / "datasets"), "build", "synth", "--release", "other=other-r-2026.10.07"])
    except SystemExit as exc:
        assert "REL-006" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_preflight_writes_nothing(temporal_dataset, tmp_path, capsys):
    temporal_dataset()
    before = tree(tmp_path)
    code = main(["--datasets", str(tmp_path / "datasets"), "--release-root", str(tmp_path / "releases"), "preflight", "synth"])
    assert code == 0 and "declaration_digest:" in capsys.readouterr().out
    assert tree(tmp_path) == before


def test_verify_prints_pin_values(temporal_dataset, tmp_path, monkeypatch, capsys):
    temporal_dataset()
    monkeypatch.setenv("PRODUCER_REVISION", "abc123")
    root = tmp_path / "releases"
    main(["--datasets", str(tmp_path / "datasets"), "--release-root", str(root), "build", "synth", "--release", "synth=synth-r-2026.10.07"])
    capsys.readouterr()
    assert main(["verify", str(root / "synth-r-2026.10.07")]) == 0
    out = capsys.readouterr().out
    assert "release_id: synth-r-2026.10.07" in out and "manifest_sha256:" in out
    assert main(["--datasets", str(tmp_path / "datasets"), "verify-reproducible", str(root / "synth-r-2026.10.07")]) == 0
