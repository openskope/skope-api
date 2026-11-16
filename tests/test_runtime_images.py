"""Guard the deployment boundary between a tested image set and a checkout."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
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


class StagingDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.revision = "a" * 40
        self.workflow = {
            "id": 123,
            "run_attempt": 2,
            "head_sha": self.revision,
            "conclusion": "success",
            "event": "push",
        }
        self.artifact = {
            "id": 456,
            "name": "runtime-images-123-2",
            "expired": False,
        }

    def test_successful_download_is_cached(self):
        def download(*args, **kwargs):
            directory = Path(args[-1])
            directory.mkdir()
            (directory / "images.tar.gz").touch()

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            records = [[self.workflow], [self.artifact]] * 2
            with patch.object(runtime, "github_records", side_effect=records), patch.object(
                runtime, "bundle_images"
            ), patch.object(runtime, "run", side_effect=download) as command:
                first = runtime.staging_bundle(self.revision, "openskope/skope-api", cache)
                second = runtime.staging_bundle(self.revision, "openskope/skope-api", cache)
            self.assertEqual(first, second)
            command.assert_called_once()
            self.assertTrue((first / "images.tar.gz").is_file())

    def test_failed_download_does_not_leave_a_cached_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(
                runtime, "github_records", side_effect=[[self.workflow], [self.artifact]]
            ), patch.object(
                runtime, "run", side_effect=subprocess.CalledProcessError(1, "gh")
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    runtime.staging_bundle(self.revision, "openskope/skope-api", cache)
            self.assertEqual(list(cache.rglob("runtime-images-*")), [])
            self.assertEqual(list(cache.rglob(".download-*")), [])

    def test_expired_or_previous_attempt_artifacts_are_not_selected(self):
        for artifact in (
            dict(self.artifact, expired=True),
            dict(self.artifact, name="runtime-images-123-1"),
        ):
            with patch.object(
                runtime, "github_records", side_effect=[[self.workflow], [artifact]]
            ), patch.object(runtime, "run") as command:
                with self.assertRaisesRegex(
                    ValueError, "no retained successful CI bundle"
                ):
                    runtime.staging_bundle(
                        self.revision, "openskope/skope-api", Path("unused")
                    )
                command.assert_not_called()

    def test_wrong_commit_failed_and_pull_request_runs_are_not_selected(self):
        runs = [
            dict(self.workflow, head_sha="b" * 40),
            dict(self.workflow, conclusion="failure"),
            dict(self.workflow, event="pull_request"),
        ]
        with patch.object(runtime, "github_records", return_value=runs) as records:
            with self.assertRaises(ValueError):
                runtime.staging_bundle(
                    self.revision, "openskope/skope-api", Path("unused")
                )
            records.assert_called_once()

    def test_dirty_checkout_stops_before_downloading_or_loading(self):
        with patch.object(runtime, "run", return_value=" M tracked.py") as command:
            with self.assertRaises(ValueError):
                runtime.deploy_staging()
            command.assert_called_once_with("git", "status", "--porcelain", capture=True)

    def test_deploy_validates_bundle_then_loads_and_uses_existing_target(self):
        bundle = Path("/tmp/test-staging-bundle")
        events = []
        with patch.object(
            runtime, "clean_revision", return_value=self.revision
        ), patch.object(runtime, "staging_bundle", return_value=bundle), patch.object(
            runtime, "bundle_images", side_effect=lambda *args: events.append("validate")
        ), patch.object(
            runtime, "run", side_effect=lambda *args: events.append(args)
        ):
            runtime.deploy_staging()
        self.assertEqual(
            events,
            [
                "validate",
                ("docker", "load", "--input", bundle / "images.tar.gz"),
                (
                    "make",
                    "deploy",
                    "ENVIRONMENT=staging",
                    f"DEPLOY_IMAGES={bundle / 'images-staging.json'}",
                ),
            ],
        )

    def test_invalid_bundle_stops_before_loading_or_deploying(self):
        with patch.object(
            runtime, "clean_revision", return_value=self.revision
        ), patch.object(
            runtime, "staging_bundle", return_value=Path("/tmp/test-staging-bundle")
        ), patch.object(
            runtime, "bundle_images", side_effect=ValueError("invalid bundle")
        ), patch.object(runtime, "run") as command:
            with self.assertRaisesRegex(ValueError, "invalid bundle"):
                runtime.deploy_staging()
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
