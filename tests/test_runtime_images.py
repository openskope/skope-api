"""Guard the deployment boundary between a tested image set and a checkout."""

import copy
import importlib.util
import io
import json
import os
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

    def test_pr_bundle_is_rejected_for_production_even_with_explicit_selection(self):
        manifest = dict(self.manifest, ci={"event": "pull_request"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images-prod.json"
            path.write_text(json.dumps(self.selection))
            (path.parent / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "only be deployed to staging"):
                runtime.bundle_images(path, "prod", self.revision)

    def test_ci_export_retains_pr_identity_with_the_scanned_images(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "manifest.json").write_text(json.dumps(self.manifest))
            event = output / "event.json"
            event.write_text(
                json.dumps(
                    {
                        "number": 69,
                        "pull_request": {
                            "head": {"sha": "c" * 40},
                            "base": {"sha": "d" * 40},
                        },
                    }
                )
            )
            environment = {
                "GITHUB_ACTIONS": "true",
                "GITHUB_EVENT_PATH": str(event),
                "GITHUB_REPOSITORY": "openskope/skope-api",
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "2",
                "GITHUB_EVENT_NAME": "pull_request",
            }
            with patch.object(runtime, "OUTPUT", output), patch.dict(
                os.environ, environment
            ), patch.object(runtime.subprocess, "Popen") as popen:
                process = popen.return_value.__enter__.return_value
                process.stdout = io.BytesIO(b"test image archive")
                process.wait.return_value = 0
                runtime.export()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["ci"],
                {
                    "repository": "openskope/skope-api",
                    "run_id": 123,
                    "run_attempt": 2,
                    "event": "pull_request",
                    "pr": 69,
                    "head_sha": "c" * 40,
                    "base_sha": "d" * 40,
                },
            )
            self.assertEqual(manifest["images"], self.manifest["images"])
            self.assertTrue((output / "images.tar.gz").is_file())


class CIDeploymentTests(unittest.TestCase):
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
            with patch.object(
                runtime, "github_records", side_effect=records
            ), patch.object(runtime, "bundle_images"), patch.object(
                runtime, "run", side_effect=download
            ) as command:
                first = runtime.ci_bundle(
                    self.revision, "openskope/skope-api", cache, "staging"
                )
                second = runtime.ci_bundle(
                    self.revision, "openskope/skope-api", cache, "staging"
                )
            self.assertEqual(first, second)
            command.assert_called_once()
            self.assertTrue((first / "images.tar.gz").is_file())

    def test_failed_download_does_not_leave_a_cached_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(
                runtime,
                "github_records",
                side_effect=[[self.workflow], [self.artifact]],
            ), patch.object(
                runtime, "run", side_effect=subprocess.CalledProcessError(1, "gh")
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    runtime.ci_bundle(
                        self.revision, "openskope/skope-api", cache, "staging"
                    )
            self.assertEqual(list(cache.rglob("runtime-images-*")), [])
            self.assertEqual(list(cache.rglob(".download-*")), [])

    def test_cached_bundle_without_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            cached = cache / "openskope/skope-api/runtime-images-123-2-456"
            cached.mkdir(parents=True)
            with patch.object(
                runtime, "github_records", return_value=[self.artifact]
            ), patch.object(runtime, "bundle_images"), patch.object(
                runtime, "run"
            ) as command:
                with self.assertRaisesRegex(ValueError, "cached CI bundle is missing"):
                    runtime.download_bundle(
                        self.workflow,
                        "openskope/skope-api",
                        cache,
                        "staging",
                        self.revision,
                    )
            command.assert_not_called()

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
                    runtime.ci_bundle(
                        self.revision, "openskope/skope-api", Path("unused"), "staging"
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
                runtime.ci_bundle(
                    self.revision, "openskope/skope-api", Path("unused"), "staging"
                )
            records.assert_called_once()

    def test_dirty_checkout_stops_before_downloading_or_loading(self):
        with patch.object(runtime, "run", return_value=" M tracked.py") as command:
            with self.assertRaises(ValueError):
                runtime.deploy_ci("staging")
            command.assert_called_once_with(
                "git", "status", "--porcelain", capture=True
            )

    def test_deploy_validates_bundle_then_loads_and_uses_existing_target(self):
        for environment in ("staging", "prod"):
            with self.subTest(environment=environment):
                bundle = Path("/tmp/test-ci-bundle")
                selection = bundle / f"images-{environment}.json"
                events = []
                with patch.object(
                    runtime, "clean_revision", return_value=self.revision
                ), patch.object(
                    runtime, "ci_bundle", return_value=bundle
                ), patch.object(
                    runtime,
                    "bundle_images",
                    side_effect=lambda *args: events.append(args),
                ), patch.object(
                    runtime, "run", side_effect=lambda *args: events.append(args)
                ):
                    runtime.deploy_ci(environment)
                self.assertEqual(
                    events,
                    [
                        (selection, environment, self.revision),
                        ("docker", "load", "--input", bundle / "images.tar.gz"),
                        (
                            "make",
                            "deploy",
                            f"ENVIRONMENT={environment}",
                            f"DEPLOY_IMAGES={selection}",
                        ),
                    ],
                )

    def test_production_download_validates_the_production_selection(self):
        def download(*args, **kwargs):
            directory = Path(args[-1])
            directory.mkdir()
            (directory / "images.tar.gz").touch()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                runtime,
                "github_records",
                side_effect=[[self.workflow], [self.artifact]],
            ), patch.object(runtime, "bundle_images") as validate, patch.object(
                runtime, "run", side_effect=download
            ):
                runtime.ci_bundle(
                    self.revision, "openskope/skope-api", Path(directory), "prod"
                )
            selection, environment, revision = validate.call_args.args
            self.assertEqual(selection.name, "images-prod.json")
            self.assertEqual((environment, revision), ("prod", self.revision))

    def test_invalid_bundle_stops_before_loading_or_deploying(self):
        with patch.object(
            runtime, "clean_revision", return_value=self.revision
        ), patch.object(
            runtime, "ci_bundle", return_value=Path("/tmp/test-staging-bundle")
        ), patch.object(
            runtime, "bundle_images", side_effect=ValueError("invalid bundle")
        ), patch.object(
            runtime, "run"
        ) as command:
            with self.assertRaisesRegex(ValueError, "invalid bundle"):
                runtime.deploy_ci("staging")
            command.assert_not_called()


class PRDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.revision = "a" * 40
        self.head = "b" * 40
        self.base = "c" * 40
        self.repository = "openskope/skope-api"
        self.pr = {
            "state": "open",
            "mergeable": True,
            "merge_commit_sha": self.revision,
            "head": {"sha": self.head},
            "base": {"sha": self.base, "repo": {"full_name": self.repository}},
        }
        # Fork PR runs can have an empty pull_requests collection.
        self.workflow = {
            "id": 123,
            "run_attempt": 2,
            "event": "pull_request",
            "head_sha": self.head,
            "status": "completed",
            "conclusion": "success",
            "pull_requests": [],
        }
        self.ci = {
            "repository": self.repository,
            "run_id": 123,
            "run_attempt": 2,
            "event": "pull_request",
            "pr": 69,
            "head_sha": self.head,
            "base_sha": self.base,
        }
        self.bundle = self.cache / "bundle"
        self.bundle.mkdir()
        (self.bundle / "manifest.json").write_text(json.dumps({"ci": self.ci}))
        self.checkout = self.cache / "checkouts" / self.repository / self.revision
        self.image = "sha256:" + "d" * 64
        self.images = dict.fromkeys(("server", "titiler", "redis"), self.image)
        self.log = self.cache / "deploy-state/staging.jsonl"
        self.commands = []
        self.pr_mock = self.enterContext(
            patch.object(runtime, "current_pr", return_value=self.pr)
        )
        self.enterContext(
            patch.object(runtime, "github_records", return_value=[self.workflow])
        )
        self.enterContext(
            patch.object(runtime, "download_bundle", return_value=self.bundle)
        )
        self.enterContext(
            patch.object(runtime, "bundle_images", return_value=self.images)
        )
        self.enterContext(patch.dict(os.environ, {"DEPLOY_LOG": str(self.log)}))

        def command(*args, **kwargs):
            self.commands.append((args, kwargs))
            if args[:2] == ("git", "rev-parse"):
                return self.revision
            if args[:2] == ("git", "status"):
                return ""

        self.command = self.enterContext(
            patch.object(runtime, "run", side_effect=command)
        )

    def test_fork_pr_uses_tested_merge_checkout_and_records_ci_after_success(self):
        runtime.deploy_pr("69", self.repository, self.cache)
        self.assertIn(
            (("git", "worktree", "add", "--detach", self.checkout, self.revision), {}),
            self.commands,
        )
        self.assertEqual(
            self.commands[-1],
            (
                (
                    "make",
                    "deploy",
                    "ENVIRONMENT=staging",
                    "PR=",
                    f"DEPLOY_IMAGES={self.bundle / 'images-staging.json'}",
                    f"DEPLOY_LOG={self.log}",
                ),
                {"cwd": self.checkout},
            ),
        )
        entry = json.loads(self.log.with_name("staging-ci.jsonl").read_text())
        self.assertEqual(
            (entry["pr"], entry["run_id"], entry["commit"], entry["head_sha"]),
            (69, 123, self.revision, self.head),
        )
        self.assertEqual(entry["images"], self.images)

    def test_latest_failed_or_running_run_does_not_fall_back_to_success(self):
        for status, conclusion in (("in_progress", None), ("completed", "failure")):
            latest = dict(self.workflow, status=status, conclusion=conclusion)
            with patch.object(
                runtime, "github_records", return_value=[latest, self.workflow]
            ):
                with self.assertRaisesRegex(ValueError, "latest CI run"):
                    runtime.deploy_pr("69", self.repository, self.cache)
        self.command.assert_not_called()

    def test_missing_bundle_or_mismatched_provenance_never_loads_images(self):
        with patch.object(runtime, "download_bundle", return_value=None):
            with self.assertRaisesRegex(ValueError, "no retained image bundle"):
                runtime.deploy_pr("69", self.repository, self.cache)
        for field, value in (
            ("pr", 68),
            ("head_sha", "e" * 40),
            ("run_attempt", 1),
            ("base_sha", "f" * 40),
            ("repository", "other/repo"),
        ):
            manifest = {"ci": dict(self.ci, **{field: value})}
            (self.bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "provenance"):
                runtime.deploy_pr("69", self.repository, self.cache)
        self.command.assert_not_called()

    def test_pr_change_during_download_stops_before_fetch_and_load(self):
        self.pr_mock.side_effect = [self.pr, dict(self.pr, merge_commit_sha="e" * 40)]
        with self.assertRaisesRegex(ValueError, "PR changed"):
            runtime.deploy_pr("69", self.repository, self.cache)
        self.command.assert_not_called()

    def test_ci_rerun_during_download_stops_before_fetch_and_load(self):
        latest = dict(self.workflow, run_attempt=3)
        with patch.object(
            runtime, "github_records", side_effect=[[self.workflow], [latest]]
        ):
            with self.assertRaisesRegex(ValueError, "PR CI changed"):
                runtime.deploy_pr("69", self.repository, self.cache)
        self.command.assert_not_called()

    def test_real_git_checkout_is_detached_reused_and_leaves_operator_branch_alone(
        self,
    ):
        source = self.cache / "source"
        source.mkdir()

        def git(*args, cwd=source):
            return subprocess.run(
                ["git", *map(str, args)],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        git("init", "--initial-branch=operator")
        git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        )
        revision = git("rev-parse", "HEAD")
        git("update-ref", "refs/pull/69/merge", revision)
        self.pr["merge_commit_sha"] = revision
        checkout = self.cache / "checkouts" / self.repository / revision

        def command(*args, **kwargs):
            if args[0] != "git":
                return
            arguments = list(args[1:])
            if arguments[0] == "fetch":
                arguments[2] = source  # Fetch the fixture instead of GitHub.
            return git(*arguments, cwd=kwargs.get("cwd", source))

        self.command.side_effect = command
        with patch.object(runtime, "ROOT", source):
            runtime.deploy_pr("69", self.repository, self.cache)
            runtime.deploy_pr("69", self.repository, self.cache)
        self.assertEqual(git("symbolic-ref", "--short", "HEAD"), "operator")
        self.assertEqual(git("rev-parse", "HEAD", cwd=checkout), revision)
        self.assertEqual(git("branch", "--show-current", cwd=checkout), "")
        self.assertEqual(git("status", "--porcelain", cwd=checkout), "")

    def test_changed_merge_ref_or_dirty_existing_checkout_never_loads_images(self):
        with patch.object(runtime, "run", side_effect=[None, "e" * 40]):
            with self.assertRaisesRegex(ValueError, "merge changed"):
                runtime.deploy_pr("69", self.repository, self.cache)
        self.checkout.mkdir(parents=True)
        with patch.object(runtime, "run", side_effect=[self.revision, " M file"]):
            with self.assertRaisesRegex(ValueError, "dirty or has the wrong revision"):
                runtime.deploy_pr("69", self.repository, self.cache)
        self.command.assert_not_called()

    def test_failed_deployment_does_not_record_success(self):
        def fail_deploy(*args, **kwargs):
            if args[0] == "make":
                raise subprocess.CalledProcessError(1, args)
            return self.revision if args[:2] == ("git", "rev-parse") else ""

        self.command.side_effect = fail_deploy
        with self.assertRaises(subprocess.CalledProcessError):
            runtime.deploy_pr("69", self.repository, self.cache)
        self.assertFalse(self.log.with_name("staging-ci.jsonl").exists())

    def test_pr_selection_rejects_production_or_explicit_images(self):
        for environment, selection in (
            ("prod", ""),
            ("staging", "/tmp/selection.json"),
        ):
            with patch.dict(os.environ, {"PR": "69", "DEPLOY_IMAGES": selection}):
                with self.assertRaisesRegex(ValueError, "staging-only"):
                    runtime.deploy_ci(environment)
        self.command.assert_not_called()

    def test_invalid_pr_number_is_rejected_before_network_calls(self):
        for number in ("-1", "0", "../69", "69;echo"):
            with self.assertRaisesRegex(ValueError, "positive pull request number"):
                runtime.deploy_pr(number, self.repository, self.cache)
        self.pr_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
