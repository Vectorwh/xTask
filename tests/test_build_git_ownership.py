"""Git metadata ownership handoff on container build exit.

Root-only tests use temporary repositories and actually drop to the host UID.
They can be run in an isolated container without a network or NPU devices.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/build_xllm_in_container.sh"
BUILD_START = 'if [[ ! -f "${XLLM_DIR}/setup.py" ]]; then'


def _prefix():
    source = SCRIPT.read_text(encoding="utf-8")
    if BUILD_START not in source:
        raise AssertionError("Cannot locate build startup validation")
    return source.split(BUILD_START, 1)[0]


def _environment(project):
    env = os.environ.copy()
    env.update(
        WORK_ROOT=str(project), HOST_UID="1003", HOST_GID="1003",
        GIT_CONFIG_GLOBAL=str(project / "test.gitconfig"), GIT_CONFIG_NOSYSTEM="1",
    )
    return env


class BuildGitOwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="git-ownership-test-")
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.source = self.project / "xllm"
        self.git_dir = self.source / ".git"
        self.git_dir.mkdir(parents=True)
        self.fake_bin = self.project / "bin"
        self.fake_bin.mkdir()
        self.log = self.project / "find-arguments"
        self.env = _environment(self.project)
        self.env.update(
            PATH=f"{self.fake_bin}:{self.env['PATH']}",
            OWNERSHIP_FIND_LOG=str(self.log), OWNERSHIP_FIND_STATUS="0",
        )
        self._executable(
            "find",
            '#!/bin/bash\nprintf "%s\\n" "$@" > "$OWNERSHIP_FIND_LOG"\n'
            'exit "${OWNERSHIP_FIND_STATUS:-0}"\n',
        )

    def _executable(self, name, content):
        path = self.fake_bin / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def _run(self, status=0):
        return subprocess.run(
            ["bash", "-c", _prefix() + f"\nexit {status}\n"],
            env=self.env, text=True, capture_output=True, timeout=20,
        )

    def test_success_restores_only_root_owned_git_metadata(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text().splitlines(), [
            "-P", str(self.git_dir), "-xdev", "-uid", "0", "-exec",
            "chown", "-h", "--from=0", "--", "1003:1003", "{}", "+",
        ])
        self.assertIn("[git-ownership]", result.stdout)

    def test_build_failure_runs_cleanup_and_preserves_exit_code(self):
        result = self._run(37)
        self.assertEqual(result.returncode, 37)
        self.assertTrue(self.log.exists())

    def test_cleanup_failure_makes_successful_build_fail(self):
        self.env["OWNERSHIP_FIND_STATUS"] = "9"
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("cleanup failed", result.stderr)

    def test_cleanup_failure_does_not_hide_original_build_failure(self):
        self.env["OWNERSHIP_FIND_STATUS"] = "9"
        result = self._run(37)
        self.assertEqual(result.returncode, 37)
        self.assertIn("cleanup failed", result.stderr)

    def test_missing_host_identity_keeps_legacy_warning_behavior(self):
        self.env.pop("HOST_UID")
        result = self._run()
        self.assertEqual(result.returncode, 0)
        self.assertIn("HOST_UID/HOST_GID not set", result.stderr)
        self.assertFalse(self.log.exists())

    def test_invalid_identity_is_rejected_before_chown(self):
        self.env["HOST_UID"] = "not-a-uid"
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("must be numeric", result.stderr)
        self.assertFalse(self.log.exists())

    def test_absent_git_directory_is_not_created(self):
        self.git_dir.rmdir()
        result = self._run()
        self.assertEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.git_dir.exists())

    def test_git_directory_symlink_is_not_followed(self):
        self.git_dir.rmdir()
        external = self.project / "external"
        external.mkdir()
        self.git_dir.symlink_to(external, target_is_directory=True)
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("through a symlink", result.stderr)
        self.assertFalse(self.log.exists())

    def test_repository_symlink_is_not_followed(self):
        external = self.project / "external"
        self.source.rename(external)
        self.source.symlink_to(external, target_is_directory=True)
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.log.exists())

    def test_git_pointer_file_does_not_redirect_cleanup(self):
        self.git_dir.rmdir()
        self.git_dir.write_text("gitdir: /outside/repository\n", encoding="utf-8")
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("expected a .git directory", result.stderr)
        self.assertFalse(self.log.exists())

    def test_real_entrypoint_failure_before_compilation_runs_cleanup(self):
        (self.source / "setup.py").write_text("", encoding="utf-8")
        self._executable("git", "#!/bin/bash\nexit 0\n")
        self._executable("pip", "#!/bin/bash\nexit 37\n")
        result = subprocess.run(
            ["bash", str(SCRIPT)], env=self.env, text=True,
            capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertTrue(self.log.exists())


@unittest.skipUnless(os.geteuid() == 0, "requires isolated root container")
class RootGitOwnershipRoundTripTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="root-git-ownership-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.uid = int(os.environ.get("TEST_HOST_UID", "1003"))
        self.gid = int(os.environ.get("TEST_HOST_GID", "1003"))
        self.assertNotEqual(self.uid, 0)
        self.project = self.base / "project"
        self.source = self.project / "xllm"
        self.source.mkdir(parents=True)
        seed = self.base / "seed"
        seed.mkdir()
        self._git(seed, "init", "-q")
        (seed / "source.txt").write_text("fixture\n", encoding="utf-8")
        self._git(seed, "add", "source.txt")
        self._git(
            seed, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "fixture",
        )
        self._git(self.source, "init", "-q")
        self._git(
            self.source, "submodule", "add", str(seed), "third_party/Mooncake"
        )
        (self.source / "setup.py").write_text("", encoding="utf-8")
        self.child = self.source / "third_party/Mooncake"
        self.config = self.source / ".git/modules/third_party/Mooncake/config"
        self.config.chmod(0o640)
        # Initial checkout is owned by the host user, just like the real build.
        for directory, names, files in os.walk(self.base):
            os.chown(directory, self.uid, self.gid)
            for name in files:
                os.chown(Path(directory) / name, self.uid, self.gid)
        self._git(self.child, "config", "fixture.root-write", "yes")
        self.assertEqual(self.config.stat().st_uid, 0)
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o640)

        self.outside = self.project / "outside-git"
        self.outside.write_text("must remain root-owned\n", encoding="utf-8")
        self.link = self.source / ".git/external-link"
        self.link.symlink_to(self.outside)
        self.env = _environment(self.project)
        self.env.update(HOST_UID=str(self.uid), HOST_GID=str(self.gid))

    def _git(self, repo, *args, as_host=False, check=True):
        env = os.environ.copy()
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

        def drop_privileges():
            os.setgroups([])
            os.setgid(self.gid)
            os.setuid(self.uid)

        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "-c", "protocol.file.allow=always",
             "-C", str(repo), *args],
            env=env, text=True, capture_output=True, timeout=30,
            preexec_fn=drop_privileges if as_host else None,
        )
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def _verify_handoff(self, full_entrypoint):
        before = self._git(
            self.child, "rev-parse", "--show-toplevel", as_host=True, check=False
        )
        self.assertNotEqual(before.returncode, 0)
        self.assertIn("Permission denied", before.stderr)
        config_content = self.config.read_bytes()
        if full_entrypoint:
            fake_bin = self.base / "bin"
            fake_bin.mkdir()
            pip = fake_bin / "pip"
            pip.write_text("#!/bin/bash\nexit 37\n", encoding="utf-8")
            pip.chmod(0o755)
            self.env["PATH"] = f"{fake_bin}:{self.env['PATH']}"
            command = ["bash", str(SCRIPT)]
            expected_status = 37
        else:
            command = ["bash", "-c", _prefix() + "\nexit 0\n"]
            expected_status = 0
        result = subprocess.run(
            command, env=self.env, text=True, capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)
        self.assertEqual(self.config.stat().st_uid, self.uid)
        self.assertEqual(self.config.stat().st_gid, self.gid)
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o640)
        self.assertEqual(self.config.read_bytes(), config_content)
        self.assertEqual(self.outside.stat().st_uid, 0)
        self.assertEqual(self.link.lstat().st_uid, self.uid)
        self._git(self.child, "rev-parse", "--show-toplevel", as_host=True)
        print(f"Verified real ownership handoff: root:root 640 -> {self.uid}:{self.gid} 640; exit={expected_status}")

    def test_success_exit_restores_real_cross_uid_access(self):
        self._verify_handoff(full_entrypoint=False)

    def test_failed_build_restores_real_cross_uid_access(self):
        self._verify_handoff(full_entrypoint=True)


if __name__ == "__main__":
    unittest.main()
