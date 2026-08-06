"""SSH/SCP password authentication helpers without sshpass/paramiko.

Primary strategy: SSH_ASKPASS + setsid (most reliable in non-interactive agents).
Fallback: PTY auto-fill for edge cases.
"""

from __future__ import annotations

import logging
import os
import stat
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple


def run_ssh_with_password(
    argv: List[str],
    password: str,
    timeout: Optional[int] = 300,
) -> Tuple[int, str, str]:
    """Run ssh/scp argv with password via SSH_ASKPASS.

    argv should be a full command list, e.g.
      ["ssh", "-p", "22", "user@host", "bash", "-lc", "hostname"]
    """
    askpass_path = None
    start = time.time()
    try:
        fd, askpass_path = tempfile.mkstemp(prefix="xtask_askpass_", suffix=".sh")
        os.close(fd)
        # Write a tiny askpass script that prints the password once.
        with open(askpass_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n")
            f.write("printf '%s\\n' \"${XTASK_ASKPASS_PASSWORD}\"\n")
        os.chmod(askpass_path, stat.S_IRWXU)

        env = os.environ.copy()
        env["SSH_ASKPASS"] = askpass_path
        env["SSH_ASKPASS_REQUIRE"] = "force"
        env["XTASK_ASKPASS_PASSWORD"] = password
        # Some OpenSSH builds still require DISPLAY to trigger askpass.
        env.setdefault("DISPLAY", "xtask:0")
        # Prevent ssh from trying to read a real tty.
        env.pop("SSH_TTY", None)

        # setsid detaches from controlling tty so ssh uses ASKPASS.
        try:
            result = subprocess.run(
                ["setsid", *argv],
                text=True,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            return result.returncode, result.stdout or "", result.stderr or ""
        except FileNotFoundError:
            # setsid unavailable: fall back to start_new_session.
            result = subprocess.run(
                argv,
                text=True,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return result.returncode, result.stdout or "", result.stderr or ""
        except subprocess.TimeoutExpired:
            return -1, "", f"SSH/SCP timeout after {timeout} seconds"
    except Exception as exc:
        logging.error("SSH_ASKPASS run failed: %s", exc)
        return -1, "", str(exc)
    finally:
        if askpass_path and os.path.exists(askpass_path):
            try:
                os.remove(askpass_path)
            except OSError:
                pass
        _ = time.time() - start
