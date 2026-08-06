"""Pure-Python PTY-based runner for ssh/scp with password authentication.

Uses only the standard library (pty, select, os, subprocess) so it works in
environments where sshpass / paramiko / pexpect are unavailable.
"""

from __future__ import annotations

import logging
import os
import pty
import select
import subprocess
import time
from typing import List, Optional, Tuple


def run_with_pty_password(
    argv: List[str],
    password: Optional[str] = None,
    timeout: Optional[int] = 300,
    extra_prompts: Optional[dict] = None,
) -> Tuple[int, str, str]:
    """Run a command with a PTY, auto-feeding password when prompted.

    Args:
        argv: command list, e.g. ["ssh", "-p", "22", "user@host", "bash", "-lc", "..."]
        password: password to send when a password prompt is detected
        timeout: overall timeout in seconds
        extra_prompts: optional {prompt_substring: response} for host-key prompts etc.

    Returns:
        (exit_code, stdout, stderr) — stdout and stderr are merged from the PTY.
    """
    prompts = {
        "password": (password + "\n") if password else None,
        "are you sure you want to continue connecting": "yes\n",
    }
    if extra_prompts:
        prompts.update(extra_prompts)

    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
    finally:
        os.close(slave)

    output = b""
    sent_keys = set()
    start = time.time()

    try:
        while True:
            if proc.poll() is not None:
                # Drain any remaining output.
                _drain_fd(master, output_list=[output])
                break
            if timeout is not None and time.time() - start > timeout:
                proc.kill()
                logging.warning("PTY command timed out after %ss: %s", timeout, argv)
                break
            ready, _, _ = select.select([master], [], [], 0.5)
            if not ready:
                continue
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
            lower = chunk.lower()
            for key, response in prompts.items():
                if key in sent_keys or response is None:
                    continue
                if key.encode() in lower:
                    os.write(master, response.encode())
                    sent_keys.add(key)
                    logging.debug("PTY auto-responded to prompt: %s", key)
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        proc.wait()

    text = output.decode("utf-8", errors="replace")
    return proc.returncode, text, ""


def _drain_fd(fd: int, output_list: list, max_reads: int = 64) -> None:
    """Read remaining data from fd into output_list[0]."""
    for _ in range(max_reads):
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            break
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        output_list[0] += chunk
