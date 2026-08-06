import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.utils.workspace_sync import (
    resolve_workspace_root,
    sync_workspace_to_executor,
    sync_workspace_to_hosts,
)


class WorkspaceSyncTests(unittest.TestCase):
    def test_resolve_workspace_root(self):
        root = resolve_workspace_root()
        self.assertTrue((root / "main.py").is_file())
        self.assertTrue((root / "xtask").is_dir())

    def test_sync_workspace_to_executor_packs_and_uploads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "xtask").mkdir()
            (root / "xtask" / "__init__.py").write_text("", encoding="utf-8")
            (root / "tools").mkdir()
            (root / "tools" / "a.sh").write_text("echo a\n", encoding="utf-8")
            (root / "config").mkdir()
            (root / "config" / "rank.json").write_text("{}\n", encoding="utf-8")
            case_out = root / "outputs" / "case1"
            case_out.mkdir(parents=True)
            (case_out / "start.sh").write_text("echo start\n", encoding="utf-8")

            uploaded = []
            commands = []

            class FakeExecutor:
                host = "10.0.0.2"

                def run(self, command, timeout=300):
                    commands.append(command)
                    from xtask.utils.container_manager import CommandResult

                    marker = "ok"
                    if "EXTRA_OK" in command:
                        marker = "EXTRA_OK"
                    elif "SYNC_OK" in command:
                        marker = "SYNC_OK"
                    return CommandResult(
                        success=True,
                        exit_code=0,
                        stdout=marker + "\n",
                        stderr="",
                        execution_time=0.01,
                    )

                def upload(self, local_path, remote_path):
                    uploaded.append((local_path, remote_path))
                    return True

            ok = sync_workspace_to_executor(
                FakeExecutor(),
                workspace_root=str(root),
                remote_root=str(root),
                extra_paths=[str(case_out)],
            )
            self.assertTrue(ok)
            self.assertTrue(any(p.endswith(".tar.gz") for p, _ in uploaded))
            self.assertTrue(any("EXTRA_OK" in c or "SYNC_OK" in c for c in commands))
            self.assertTrue(any("tar xzf" in c for c in commands))
            extra_extract_commands = [c for c in commands if "EXTRA_OK" in c]
            self.assertEqual(len(extra_extract_commands), 1)
            self.assertIn("--unlink-first", extra_extract_commands[0])
            self.assertNotIn("--overwrite", extra_extract_commands[0])

    def test_sync_workspace_to_hosts_skips_local(self):
        with mock.patch(
            "xtask.utils.workspace_sync.create_host_executor"
        ) as create_ex, mock.patch(
            "xtask.utils.workspace_sync.sync_workspace_to_executor",
            return_value=True,
        ) as sync_one, mock.patch(
            "xtask.utils.workspace_sync.is_local_host",
            side_effect=lambda h: h in {"127.0.0.1", "local"},
        ):
            sync_workspace_to_hosts(["127.0.0.1", "10.0.0.2", "10.0.0.2"])
            create_ex.assert_called_once_with("10.0.0.2")
            self.assertEqual(sync_one.call_count, 1)

    def test_sync_workspace_to_hosts_raises_on_failure(self):
        with mock.patch(
            "xtask.utils.workspace_sync.create_host_executor"
        ), mock.patch(
            "xtask.utils.workspace_sync.sync_workspace_to_executor",
            return_value=False,
        ), mock.patch(
            "xtask.utils.workspace_sync.is_local_host",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError):
                sync_workspace_to_hosts(["10.0.0.9"])


if __name__ == "__main__":
    unittest.main()
