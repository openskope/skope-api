"""Guard the deployment boundary between a tested image set and a checkout."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "runtime_images", Path(__file__).resolve().parents[1] / "scripts/runtime-images.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class ImageSelectionTests(unittest.TestCase):
    def setUp(self):
        self.revision = "a" * 40
        self.image = "sha256:" + "b" * 64
        self.selection = {
            "x-skope": {"revision": self.revision, "environment": "prod"},
            "services": {
                name: {"image": self.image, "pull_policy": "never"}
                for name in ("server", "titiler", "redis")
            },
        }
        self.manifest = {
            "revision": self.revision,
            "scan": {"status": "passed"},
            "images": {
                name: {"id": self.image}
                for name in ("staging", "prod", "titiler", "redis")
            },
        }

    def test_exact_image_ids_are_accepted(self):
        self.assertEqual(
            runtime.selection_images(self.selection, "prod", self.revision),
            dict.fromkeys(("server", "titiler", "redis"), self.image),
        )

    def test_wrong_revision_or_environment_is_rejected(self):
        for environment, revision in (("staging", self.revision), ("prod", "c" * 40)):
            with self.assertRaises(ValueError):
                runtime.selection_images(self.selection, environment, revision)

    def test_mutable_tags_and_missing_services_are_rejected(self):
        changed = copy.deepcopy(self.selection)
        changed["services"]["server"]["image"] = "openskope/skope-api:latest"
        with self.assertRaises(ValueError):
            runtime.selection_images(changed, "prod", self.revision)
        del changed["services"]["server"]
        with self.assertRaises(ValueError):
            runtime.selection_images(changed, "prod", self.revision)

    def test_selection_cannot_override_mounts_or_pull_remote_images(self):
        for field, value in (("volumes", ["/:/host"]), ("pull_policy", "always")):
            changed = copy.deepcopy(self.selection)
            changed["services"]["server"][field] = value
            with self.assertRaises(ValueError):
                runtime.selection_images(changed, "prod", self.revision)

    def test_loaded_images_must_match_revision_and_api_environment(self):
        info = {
            "Id": self.image,
            "Config": {
                "Labels": {runtime.REVISION_LABEL: self.revision},
                "Env": ["ENVIRONMENT=prod"],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.json"
            path.write_text(json.dumps(self.selection))
            (path.parent / "manifest.json").write_text(json.dumps(self.manifest))
            with patch.object(
                runtime, "run", side_effect=["", self.revision]
            ), patch.object(runtime, "inspect", return_value=info):
                runtime.check(path, "prod")
            for labels, environment in (
                ({runtime.REVISION_LABEL: "c" * 40}, "prod"),
                ({runtime.REVISION_LABEL: self.revision}, "staging"),
            ):
                wrong = copy.deepcopy(info)
                wrong["Config"] = {
                    "Labels": labels,
                    "Env": [f"ENVIRONMENT={environment}"],
                }
                with patch.object(
                    runtime, "run", side_effect=["", self.revision]
                ), patch.object(runtime, "inspect", return_value=wrong):
                    with self.assertRaises(ValueError):
                        runtime.check(path, "prod")

    def test_dirty_checkout_is_refused_before_inspecting_images(self):
        with patch.object(runtime, "run", return_value=" M tracked.py"), patch.object(
            runtime, "inspect"
        ) as inspect:
            with self.assertRaises(ValueError):
                runtime.check(Path("unused.json"), "prod")
            inspect.assert_not_called()

    def test_export_requires_a_successful_scan(self):
        for status in (None, "running", "failed"):
            manifest = copy.deepcopy(self.manifest)
            manifest["scan"] = {} if status is None else {"status": status}
            with self.assertRaises(ValueError):
                runtime.scanned_images(manifest)

    def test_image_selection_must_match_the_scanned_bundle(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["images"]["prod"]["id"] = "sha256:" + "c" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.json"
            path.write_text(json.dumps(self.selection))
            (path.parent / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(
                runtime, "run", side_effect=["", self.revision]
            ), patch.object(runtime, "inspect") as inspect:
                with self.assertRaises(ValueError):
                    runtime.check(path, "prod")
                inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
