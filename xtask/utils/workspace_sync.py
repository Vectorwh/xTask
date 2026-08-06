"""Sync local xTask workspace to remote hosts for multi-host cases.

Remote containers mount the same absolute host path as /workspace, so workers
need a local copy of the tree (tools, config/ranktable, generated scripts, etc.).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from xtask.utils.host_executor import HostExecutor, create_host_executor, is_local_host

# Paths relative to workspace root that remote hosts need for serving/benchmark.
_DEFAULT_REL_PATHS = (
    "main.py",
    "pyproject.toml",
    "setup.py",
    "requirement.txt",
    "README.md",
    "xtask",
    "tools",
    "config",
    "pkg",
    "templates",
    "tests",
)

_TAR_EXCLUDES = (
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=.git",
    "--exclude=venv",
    "--exclude=.venv",
    "--exclude=*.rpm",
    "--exclude=outputs",
    "--exclude=benchmark",
)


def resolve_workspace_root(start: Optional[str] = None) -> Path:
    """Locate xTask root (directory containing main.py and xtask/)."""
    cur = Path(start or __file__).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "main.py").is_file() and (candidate / "xtask").is_dir():
            return candidate
    raise FileNotFoundError("Unable to resolve xTask workspace root from %s" % start)


def _existing_rel_paths(root: Path, rel_paths: Sequence[str]) -> List[str]:
    found = []
    for rel in rel_paths:
        if (root / rel).exists():
            found.append(rel)
    return found


def sync_workspace_to_executor(
    executor: HostExecutor,
    workspace_root: Optional[str] = None,
    remote_root: Optional[str] = None,
    extra_paths: Optional[Iterable[str]] = None,
    rel_paths: Sequence[str] = _DEFAULT_REL_PATHS,
    timeout: int = 900,
) -> bool:
    """Pack and extract workspace onto a remote host via tar + upload.

    Args:
        executor: remote host executor
        workspace_root: local xTask root; auto-detected when None
        remote_root: remote absolute path (defaults to same as local root)
        extra_paths: additional absolute local paths to include (e.g. case outputs)
        rel_paths: relative entries under workspace_root to pack
        timeout: remote extract timeout seconds
    """
    root = Path(workspace_root) if workspace_root else resolve_workspace_root()
    root = root.resolve()
    remote = remote_root or str(root)
    include = _existing_rel_paths(root, rel_paths)
    if not include:
        logging.error("No workspace paths to sync under %s", root)
        return False

    tar_fd, tar_path = tempfile.mkstemp(prefix="xtask_ws_sync_", suffix=".tar.gz")
    os.close(tar_fd)
    try:
        cmd = ["tar", "czf", tar_path, *_TAR_EXCLUDES, "-C", str(root), *include]
        logging.info(
            "Packing workspace for %s: %s (%d entries)",
            getattr(executor, "host", "remote"),
            root,
            len(include),
        )
        subprocess.check_call(cmd)

        remote_tar = f"/tmp/xtask_ws_sync_{os.getpid()}.tar.gz"
        mkdir = executor.run(f"mkdir -p {remote}", timeout=60)
        if not mkdir.success:
            logging.error(
                "Failed to create remote workspace dir %s: %s",
                remote,
                mkdir.stderr or mkdir.stdout,
            )
            return False
        if not executor.upload(tar_path, remote_tar):
            logging.error("Failed to upload workspace tarball to %s", remote)
            return False

        # Prior syncs may leave read-only files under pkg/ (mode 550); make writable first.
        targets = " ".join(f"{remote}/{rel}" for rel in include)
        extract = executor.run(
            f"mkdir -p {remote} && "
            f"chmod -R u+w {targets} 2>/dev/null || true; "
            f"rm -rf {targets}; "
            f"tar xzf {remote_tar} -C {remote} && "
            f"rm -f {remote_tar} && test -f {remote}/main.py && echo SYNC_OK",
            timeout=timeout,
        )
        if not extract.success or "SYNC_OK" not in (extract.stdout or ""):
            logging.error(
                "Failed to extract workspace on remote: stdout=%s stderr=%s",
                extract.stdout,
                extract.stderr,
            )
            return False

        for abs_path in extra_paths or []:
            if not abs_path:
                continue
            if not _sync_extra_path(executor, abs_path, timeout=timeout):
                return False

        logging.info("Workspace synced to %s:%s", getattr(executor, "host", "?"), remote)
        return True
    except subprocess.CalledProcessError as exc:
        logging.error("Failed to pack workspace tarball: %s", exc)
        return False
    finally:
        try:
            os.remove(tar_path)
        except OSError:
            pass


_EXTRA_TAR_EXCLUDES = (
    "--exclude=kernel_meta",
    "--exclude=log",
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=core.*",
)


def _sync_extra_path(executor: HostExecutor, abs_path: str, timeout: int = 900) -> bool:
    """Best-effort tar-sync of case outputs (scripts); skips runtime junk.

    Returns True even on soft failure: instance startup still uploads output_dir.
    """
    local = Path(abs_path).resolve()
    if not local.exists():
        logging.warning("Skip missing extra sync path: %s", local)
        return True

    remote_extra = str(local)
    parent = str(local.parent)
    mk = executor.run(f"mkdir -p {parent}", timeout=60)
    if not mk.success:
        logging.warning("Remote mkdir failed for %s: %s", parent, mk.stderr)
        return True

    tar_fd, tar_path = tempfile.mkstemp(prefix="xtask_extra_sync_", suffix=".tar.gz")
    os.close(tar_fd)
    try:
        cmd = [
            "tar", "czf", tar_path, *_EXTRA_TAR_EXCLUDES,
            "-C", parent, local.name,
        ]
        subprocess.check_call(cmd)
        remote_tar = f"/tmp/xtask_extra_sync_{os.getpid()}.tar.gz"
        if not executor.upload(tar_path, remote_tar):
            logging.warning("Failed to upload extra tarball for %s (will retry at startup)", local)
            return True
        extract = executor.run(
            f"mkdir -p {parent} && "
            f"chmod -R u+w {remote_extra} 2>/dev/null || true; "
            f"rm -rf {remote_extra} 2>/dev/null || true; "
            f"tar xzf {remote_tar} -C {parent} --unlink-first; "
            f"rm -f {remote_tar}; "
            f"test -d {remote_extra} && echo EXTRA_OK",
            timeout=timeout,
        )
        if "EXTRA_OK" not in (extract.stdout or ""):
            logging.warning(
                "Best-effort extra sync incomplete for %s (startup upload will retry): %s",
                local,
                extract.stderr or extract.stdout,
            )
            return True
        logging.info("Synced extra path %s to remote", local)
        return True
    except subprocess.CalledProcessError as exc:
        logging.warning("Failed to pack extra path %s: %s", local, exc)
        return True
    finally:
        try:
            os.remove(tar_path)
        except OSError:
            pass


def sync_workspace_to_hosts(
    hosts: Sequence[str],
    workspace_root: Optional[str] = None,
    extra_paths: Optional[Iterable[str]] = None,
) -> None:
    """Sync workspace to each unique non-local host. Raises on failure."""
    root = str(resolve_workspace_root(workspace_root) if workspace_root is None else Path(workspace_root).resolve())
    seen = set()
    for host in hosts:
        if not host or is_local_host(host) or host in seen:
            continue
        seen.add(host)
        logging.info("Syncing xTask workspace to remote host %s ...", host)
        executor = create_host_executor(host)
        ok = sync_workspace_to_executor(
            executor,
            workspace_root=root,
            remote_root=root,
            extra_paths=extra_paths,
        )
        if not ok:
            raise RuntimeError(f"Failed to sync xTask workspace to host {host}")
