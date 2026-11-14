"""Committed JSON Schemas (Section 20.3)."""

from skope_release.documents import RELEASE_SCHEMA_DIR, generated_schemas


def test_committed_schemas_match_the_models():
    """Section 20.3: the committed JSON Schemas are generated from the models."""
    for name, content in generated_schemas().items():
        committed = RELEASE_SCHEMA_DIR / name
        assert committed.is_file(), f"regenerate with `python -m skope_release.documents` ({name} missing)"
        assert committed.read_bytes() == content, f"{name} is stale; regenerate with `python -m skope_release.documents`"
