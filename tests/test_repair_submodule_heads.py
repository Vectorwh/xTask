"""Offline regressions for interrupted submodule clones and retry handling."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPAIR_SCRIPT = ROOT / "tools" / "repair_submodule_heads.py"
PREPARE_SCRIPT = ROOT / "tools" / "prepare_xllm.sh"


class RepairSubmoduleHeadsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="submodule-head-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.env = os.environ.copy()
        self.env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_COUNT="0",
            GIT_ALLOW_PROTOCOL="file",
            GIT_TERMINAL_PROMPT="0",
        )
        self.seed = self.base / "seed"
        self._init(self.seed)
        (self.seed / "source.txt").write_text("original source\n", encoding="utf-8")
        self.git(self.seed, "add", "source.txt")
        self._commit(self.seed)
        self.commit = self.git(self.seed, "rev-parse", "HEAD").stdout.strip()
        self.broken_remote = self.base / "broken-default.git"
        self.git(self.base, "clone", "--bare", str(self.seed), str(self.broken_remote))
        self.git(self.broken_remote, "symbolic-ref", "HEAD", "refs/heads/missing")
        self.parent = self.base / "parent"
        self._init(self.parent)

    def git(self, repo, *args, check=True):
        result = subprocess.run(
            [
                "git", "-c", "init.defaultBranch=master", "-c",
                "protocol.file.allow=always", "-C", str(repo), *args,
            ],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def _init(self, repo):
        repo.mkdir()
        self.git(repo, "init", "-q", "-b", "real-branch")

    def _commit(self, repo):
        self.git(
            repo, "-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.invalid", "commit", "-qm", "fixture",
        )

    def _add_links(self, repo, links):
        for index, (path, url, commit) in enumerate(links):
            key = f"submodule.fixture{index}"
            self.git(repo, "config", "-f", ".gitmodules", f"{key}.path", path)
            self.git(repo, "config", "-f", ".gitmodules", f"{key}.url", str(url))
            self.git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},{path}")
        self.git(repo, "add", ".gitmodules")
        self._commit(repo)

    def _interrupt(self, path="llvm"):
        self._add_links(self.parent, [
            (path, self.broken_remote, self.commit),
            ("z-network-failure", self.base / "unavailable.git", self.commit),
        ])
        result = self.git(self.parent, "submodule", "update", "--init", "--recursive", check=False)
        self.assertNotEqual(result.returncode, 0)
        child = self.parent / path
        self.assertEqual(self.git(child, "cat-file", "-t", self.commit).stdout.strip(), "commit")
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )
        return child

    def repair(self, *args):
        return subprocess.run(
            [sys.executable, str(REPAIR_SCRIPT), str(self.parent), *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_repairs_interrupted_clone_and_submodule_update_can_resume(self):
        child = self._interrupt()
        self.git(self.parent, "config", "submodule.fixture1.url", str(self.seed))
        failed_retry = self.git(
            self.parent, "submodule", "update", "--init", "--recursive", check=False
        )
        self.assertIn("Unable to find current revision", failed_retry.stderr)

        result = self.repair()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Recovered 1", result.stdout)
        self.assertEqual(self.git(child, "rev-parse", "HEAD").stdout.strip(), self.commit)
        self.assertEqual((child / "source.txt").read_text(), "original source\n")
        self.assertEqual(self.git(child, "symbolic-ref", "-q", "HEAD", check=False).returncode, 1)
        self.assertEqual(
            self.git(child, "remote", "get-url", "origin").stdout.strip(),
            str(self.broken_remote),
        )
        self.git(self.parent, "submodule", "update", "--init", "--recursive", "--no-fetch")
        self.assertEqual(self.repair().stdout, "")

    def test_dry_run_and_real_run_agree_on_unsafe_candidates(self):
        # The preflight decides for both modes, so a dry run must never promise
        # a recovery that the real run refuses.
        self._add_links(self.parent, [
            ("a-recoverable", self.broken_remote, self.commit),
            ("b-protected", self.broken_remote, self.commit),
            ("z-unavailable", self.base / "unavailable.git", self.commit),
        ])
        update = self.git(
            self.parent, "submodule", "update", "--init", "--recursive", check=False
        )
        self.assertNotEqual(update.returncode, 0)
        (self.parent / "b-protected/keep-me.txt").write_text("user data", encoding="utf-8")
        preview = self.repair("--dry-run")
        real = self.repair()
        self.assertEqual(preview.returncode, real.returncode)
        self.assertEqual(preview.stdout, real.stdout)
        self.assertEqual(preview.returncode, 2)
        self.assertEqual(preview.stdout, "")
        self.assertIn("worktree is not empty", preview.stderr)

    def test_dry_run_does_not_change_head_or_worktree(self):
        child = self._interrupt()
        result = self.repair("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Would recover 1", result.stdout)
        self.assertEqual([entry.name for entry in child.iterdir()], [".git"])
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_preserves_files_in_invalid_head_worktree(self):
        child = self._interrupt()
        sentinel = child / "keep-me.txt"
        sentinel.write_text("user data", encoding="utf-8")
        result = self.repair()
        self.assertEqual(result.returncode, 2)
        self.assertIn("worktree is not empty", result.stderr)
        self.assertEqual(sentinel.read_text(), "user data")
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_preserves_staged_entries_even_when_worktree_is_empty(self):
        child = self._interrupt()
        blob = self.git(self.seed, "rev-parse", f"{self.commit}:source.txt").stdout.strip()
        self.git(child, "update-index", "--add", "--cacheinfo", f"100644,{blob},staged.txt")
        before = self.git(child, "ls-files", "--stage").stdout
        result = self.repair()
        self.assertEqual(result.returncode, 2)
        self.assertIn("index contains entries", result.stderr)
        self.assertEqual(self.git(child, "ls-files", "--stage").stdout, before)

    def test_missing_pinned_commit_does_not_select_remote_branch(self):
        child = self._interrupt()
        missing = "f" * 40
        self.git(self.parent, "update-index", "--cacheinfo", f"160000,{missing},llvm")
        result = self.repair()
        self.assertEqual(result.returncode, 2)
        self.assertIn(f"required commit {missing}", result.stderr)
        self.assertIn("not available locally", result.stderr)
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_existing_git_lock_is_preserved(self):
        child = self._interrupt()
        git_dir = Path(self.git(child, "rev-parse", "--absolute-git-dir").stdout.strip())
        lock = git_dir / "index.lock"
        lock.write_text("busy", encoding="utf-8")
        result = self.repair()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Git lock exists", result.stderr)
        self.assertEqual(lock.read_text(), "busy")

    def test_healthy_dirty_checkout_is_not_changed(self):
        self._add_links(self.parent, [("llvm", self.seed, self.commit)])
        self.git(self.parent, "submodule", "update", "--init")
        child = self.parent / "llvm"
        (child / "source.txt").write_text("user changes", encoding="utf-8")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual((child / "source.txt").read_text(), "user changes")

    def test_update_none_submodule_is_not_repaired(self):
        child = self._interrupt()
        self.git(self.parent, "config", "submodule.fixture0.update", "none")
        self.git(self.parent, "config", "submodule.fixture1.url", str(self.seed))
        self.git(self.parent, "submodule", "update", "--init", "--recursive")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_inactive_submodule_is_not_repaired(self):
        child = self._interrupt()
        self.git(self.parent, "config", "submodule.fixture0.active", "false")
        self.git(self.parent, "config", "submodule.active", ".")
        self.git(self.parent, "config", "submodule.fixture1.url", str(self.seed))
        self.git(self.parent, "submodule", "update", "--init", "--recursive")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_submodule_active_pathspec_exclusion_is_respected(self):
        child = self._interrupt()
        self.git(self.parent, "config", "--unset", "submodule.fixture0.active")
        self.git(self.parent, "config", "--add", "submodule.active", ".")
        self.git(self.parent, "config", "--add", "submodule.active", ":(exclude)llvm")
        self.git(self.parent, "config", "submodule.fixture1.url", str(self.seed))
        self.git(self.parent, "submodule", "update", "--init", "--recursive")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_in_progress_pack_is_not_repaired(self):
        child = self._interrupt()
        git_dir = Path(self.git(child, "rev-parse", "--absolute-git-dir").stdout.strip())
        pack = git_dir / "objects/pack/tmp_pack_fixture"
        pack.write_bytes(b"in-progress download")
        result = self.repair()
        self.assertEqual(result.returncode, 2)
        self.assertIn("incomplete pack", result.stderr)
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )
        self.assertEqual(pack.read_bytes(), b"in-progress download")

    def test_partial_clone_is_not_automatically_repaired(self):
        child = self._interrupt()
        self.git(child, "config", "remote.origin.promisor", "true")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_all_candidates_are_checked_before_any_checkout(self):
        self._add_links(self.parent, [
            ("a-recoverable", self.broken_remote, self.commit),
            ("b-protected", self.broken_remote, self.commit),
            ("z-unavailable", self.base / "unavailable.git", self.commit),
        ])
        result = self.git(
            self.parent, "submodule", "update", "--init", "--recursive", check=False
        )
        self.assertNotEqual(result.returncode, 0)
        sentinel = self.parent / "b-protected/keep-me.txt"
        sentinel.write_text("user data", encoding="utf-8")
        result = self.repair()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        head = self.git(
            self.parent / "a-recoverable", "rev-parse", "--verify", "HEAD", check=False
        )
        self.assertNotEqual(head.returncode, 0)
        self.assertEqual(sentinel.read_text(), "user data")

    def test_gitmodules_update_none_is_respected(self):
        child = self._interrupt()
        self.git(
            self.parent, "config", "-f", ".gitmodules",
            "submodule.fixture0.update", "none",
        )
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_custom_update_strategy_is_left_to_git(self):
        child = self._interrupt()
        self.git(self.parent, "config", "submodule.fixture0.update", "rebase")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotEqual(
            self.git(child, "rev-parse", "--verify", "HEAD", check=False).returncode, 0
        )

    def test_explicit_active_overrides_pathspec_exclusion(self):
        child = self._interrupt()
        self.git(self.parent, "config", "submodule.active", "z-network-failure")
        self.git(self.parent, "config", "submodule.fixture0.active", "true")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(child, "rev-parse", "HEAD").stdout.strip(), self.commit)

    def test_update_init_without_pathspecs_reactivates_modules(self):
        child = self._interrupt()
        self.git(self.parent, "config", "submodule.fixture0.active", "false")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(child, "rev-parse", "HEAD").stdout.strip(), self.commit)
        # Recovery itself must not rewrite the caller's configuration.
        self.assertEqual(
            self.git(self.parent, "config", "submodule.fixture0.active").stdout.strip(),
            "false",
        )

    def test_path_escape_is_rejected_without_touching_other_repository(self):
        child = self._interrupt()
        (child / ".git").unlink()
        child.rmdir()
        child.symlink_to(self.seed, target_is_directory=True)
        result = self.repair()
        self.assertEqual(result.returncode, 2)
        self.assertIn("escapes its parent", result.stderr)
        self.assertEqual(self.git(self.seed, "rev-parse", "HEAD").stdout.strip(), self.commit)
        self.assertEqual((self.seed / "source.txt").read_text(), "original source\n")

    def test_uninitialized_submodules_are_left_for_normal_update(self):
        self._add_links(self.parent, [("llvm", self.broken_remote, self.commit)])
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.parent / "llvm").exists())

    def test_paths_with_spaces(self):
        child = self._interrupt("third party/llvm project")
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(child, "rev-parse", "HEAD").stdout.strip(), self.commit)

    def test_recovers_nested_submodule(self):
        nested_source = self.base / "nested-source"
        self._init(nested_source)
        self._add_links(nested_source, [
            ("llvm", self.broken_remote, self.commit),
            ("z-network-failure", self.base / "unavailable.git", self.commit),
        ])
        nested_commit = self.git(nested_source, "rev-parse", "HEAD").stdout.strip()
        self._add_links(self.parent, [("nested", nested_source, nested_commit)])
        result = self.git(self.parent, "submodule", "update", "--init", "--recursive", check=False)
        self.assertNotEqual(result.returncode, 0)
        result = self.repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        child = self.parent / "nested/llvm"
        self.assertEqual(self.git(child, "rev-parse", "HEAD").stdout.strip(), self.commit)


class PrepareRetryTests(unittest.TestCase):
    def _run_retry(self, function_body, stop_on_local_state=True):
        source = PREPARE_SCRIPT.read_text(encoding="utf-8")
        definitions = source[
            source.index("_apply_git_proxy_mode() {"):
            source.index("update_xllm_checkout() {")
        ]
        # The opt-in is per call, so the fixture must set it the same way the
        # script does instead of relying on the operation's function name.
        prefix = "RETRY_STOP_ON_LOCAL_STATE=1 " if stop_on_local_state else ""
        command = f"""
GIT_HTTP_PROXY=http://unused.invalid
GIT_HTTPS_PROXY=http://unused.invalid
GIT_MAX_RETRIES=3
GIT_RETRY_DELAY=0
{definitions}
calls=0
operation() {{
  calls=$((calls + 1))
  {function_body}
}}
{prefix}retry 'fixture update' operation
result=$?
printf 'result=%s calls=%s\\n' "$result" "$calls"
"""
        return subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=10)

    def test_local_recovery_failure_stops_retrying(self):
        result = self._run_retry("return 125")
        self.assertIn("result=125 calls=1", result.stdout)
        self.assertIn("requires manual attention", result.stderr)
        self.assertNotIn("retrying in", result.stderr)

    def test_network_errors_still_retry(self):
        result = self._run_retry("if (( calls < 3 )); then return 128; fi; return 0")
        self.assertIn("result=0 calls=3", result.stdout)
        self.assertEqual(result.stderr.count("retrying in"), 2)

    def test_exit_125_without_opt_in_keeps_retrying(self):
        # Callers that do not declare the contract must keep the network
        # retry behaviour, so an unrelated 125 is never treated as fatal.
        result = self._run_retry("return 125", stop_on_local_state=False)
        self.assertIn("result=1 calls=3", result.stdout)
        self.assertEqual(result.stderr.count("retrying in"), 2)
        self.assertNotIn("requires manual attention", result.stderr)

    def test_submodule_update_declares_the_local_state_contract(self):
        source = PREPARE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'RETRY_STOP_ON_LOCAL_STATE=1 retry "git submodule update" '
            "update_xllm_submodules",
            source,
        )
        self.assertIn('return "${RETRY_LOCAL_STATE_EXIT_CODE}"', source)

    def test_repair_is_in_the_retried_operation(self):
        source = PREPARE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('retry "git submodule update" update_xllm_submodules', source)
        self.assertIn('"${SCRIPT_DIR}/repair_submodule_heads.py" "${XLLM_DIR}"', source)
        self.assertIn('git -C "${XLLM_DIR}" submodule update --init --recursive --progress', source)


if __name__ == "__main__":
    unittest.main()
