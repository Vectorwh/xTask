import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.utils.host_executor import SshHostExecutor


class SshHostExecutorUploadTests(unittest.TestCase):
    def test_directory_tar_upload_honors_remote_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "local_case"
            local.mkdir()
            (local / "marker.txt").write_text("ok", encoding="utf-8")
            remote = "/tmp/remote-case-run-123"

            executor = SshHostExecutor("127.0.0.1", user="user")
            executor.run = mock.Mock(return_value=None)
            executor.run.side_effect = None
            executor.upload = mock.Mock(return_value=True)

            def run(command, **kwargs):
                if command.startswith("rm -rf "):
                    return type("Result", (), {"success": True})()
                if command.startswith("tar xzf "):
                    return type("Result", (), {"success": True, "stdout": "DIR_OK"})()
                raise AssertionError(f"unexpected command: {command}")

            executor.run = mock.Mock(side_effect=run)
            self.assertTrue(executor.upload_dir_to_exact_path(str(local), remote))

            prep = executor.run.call_args_list[0].args[0]
            extract = executor.run.call_args_list[1].args[0]
            self.assertIn(f"mkdir -p {remote}", prep)
            self.assertIn(f"-C {remote}", extract)
            self.assertIn("--strip-components=1", extract)

    def test_regular_upload_preserves_existing_directory_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "local_case"
            local.mkdir()
            remote = "/tmp/remote-case-run-123"
            executor = SshHostExecutor("127.0.0.1", user="user")
            executor.upload = mock.Mock(wraps=executor.upload)
            executor._run_scp = mock.Mock(return_value=0)

            def run(command, **kwargs):
                return type("Result", (), {"success": True, "stdout": "DIR_OK"})()

            executor.run = mock.Mock(side_effect=run)
            self.assertTrue(executor.upload(str(local), remote))

            prep = executor.run.call_args_list[0].args[0]
            extract = executor.run.call_args_list[-1].args[0]
            self.assertIn(f"mkdir -p {Path(remote).parent}", prep)
            self.assertIn(f"-C {Path(remote).parent}", extract)
            self.assertNotIn("--strip-components", extract)


if __name__ == "__main__":
    unittest.main()
