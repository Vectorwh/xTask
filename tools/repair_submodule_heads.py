#!/usr/bin/env python3
"""Recover empty submodule clones whose HEAD cannot resolve to a commit.

Only recover full clones using the checkout strategy, at the parent index commit.
Partial clones and explicitly excluded submodules are left to normal Git updates.
Never remove repositories, force a checkout, change remotes, or select a branch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GIT_NO_LAZY_FETCH"] = "1"
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, env=env
    )
    if check and result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return result


def _config(repo: Path, *args: str):
    result = _git(repo, "config", "-z", *args, check=False)
    if result.returncode == 1:
        return None
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Cannot read Git configuration in {repo}: {detail}")
    return result.stdout


def _setting(repo: Path, key: str, filename=None):
    args = ["--file", filename] if filename is not None else []
    value = _config(repo, *args, "--get", key)
    return None if value is None else os.fsdecode(value[:-1])


def _git_dir(repo: Path) -> Path:
    output = _git(repo, "rev-parse", "--absolute-git-dir").stdout
    return Path(os.fsdecode(output[:-1]))


def _is_partial_clone(repo: Path) -> bool:
    # Do not rely solely on GIT_NO_LAZY_FETCH on older Git versions. Inspecting
    # missing HEAD objects in promisor clones can itself trigger a fetch.
    if _setting(repo, "extensions.partialclone") is not None:
        return True
    promisor = _config(
        repo, "--type=bool", "--get-regexp", r"^remote\..*\.promisor$"
    )
    if promisor and any(
        entry.endswith(b"\ntrue") for entry in promisor.split(b"\0")
    ):
        return True
    return any((_git_dir(repo) / "objects" / "pack").glob("*.promisor"))


def _gitlinks(repo: Path):
    # Leaf repositories (notably LLVM) can have very large indexes.
    if not (repo / ".gitmodules").is_file():
        return
    declared = _config(
        repo, "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"
    )
    names = {}
    for entry in (declared or b"").split(b"\0"):
        if entry:
            key, path = entry.split(b"\n", 1)
            names[os.fsdecode(path)] = os.fsdecode(key)[len("submodule."):-len(".path")]

    active_specs = _config(repo, "--get-all", "submodule.active")
    active_paths = None
    if active_specs is not None:
        # Let Git match pathspecs, including :(exclude), rather than treating
        # these settings as shell globs. Explicit per-module active wins.
        specs = [os.fsdecode(value) for value in active_specs[:-1].split(b"\0")]
        active_paths = set(_git(repo, "ls-files", "-z", "--", *specs).stdout.split(b"\0"))

    entries = _git(repo, "ls-files", "--stage", "-z").stdout.split(b"\0")
    for entry in entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, commit, stage = metadata.split()
        if mode != b"160000":
            continue
        path = os.fsdecode(raw_path)
        name = names.get(path)
        if name is None:
            # Let submodule update report missing .gitmodules metadata.
            continue
        if active_paths is not None:
            active = _config(repo, "--type=bool", "--get", f"submodule.{name}.active")
            if active == b"false\0" or (active is None and raw_path not in active_paths):
                continue
        # Without submodule.active pathspecs, update --init initializes all
        # modules, including ones with a previous per-module active=false.
        strategy = _setting(repo, f"submodule.{name}.update")
        if strategy is None:
            strategy = _setting(repo, f"submodule.{name}.update", ".gitmodules")
        if strategy not in (None, "checkout"):
            continue
        if stage != b"0":
            raise RuntimeError(f"Unmerged submodule entry: {repo / path}")
        yield path, commit.decode("ascii")


def _validate_empty_clone(repo: Path, commit: str) -> None:
    # A missing HEAD can also indicate damaged metadata in a real worktree.
    # Recover only the narrow interrupted-clone case; preserve all user files.
    if any(entry.name != ".git" for entry in repo.iterdir()):
        raise RuntimeError(
            f"Invalid HEAD in {repo}, but the worktree is not empty; "
            "refusing automatic recovery. Inspect and preserve local files first."
        )
    if _git(repo, "ls-files", "-z").stdout:
        raise RuntimeError(
            f"Invalid HEAD in {repo}, but the index contains entries; "
            "refusing automatic recovery. Inspect and preserve staged changes first."
        )
    git_dir = _git_dir(repo)
    for name in (
        "index.lock", "HEAD.lock", "config.lock", "packed-refs.lock",
        "shallow.lock", "FETCH_HEAD.lock",
    ):
        if (git_dir / name).exists():
            raise RuntimeError(
                f"Git lock exists: {git_dir / name}; stop concurrent Git operations "
                "and inspect the lock before retrying."
            )
    incomplete_pack = next((git_dir / "objects" / "pack").glob("tmp_pack_*"), None)
    if incomplete_pack is not None:
        raise RuntimeError(
            f"In-progress or incomplete pack exists: {incomplete_pack}; "
            "inspect the interrupted/concurrent download before retrying."
        )
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode:
        raise RuntimeError(
            f"Invalid HEAD in {repo}; required commit {commit} is not available "
            "locally. Fetch that exact commit from the configured origin, then retry."
        )


def _recovery_candidates(repo: Path):
    for relative_path, commit in _gitlinks(repo):
        child = (repo / relative_path).resolve()
        try:
            child.relative_to(repo)
        except ValueError as error:
            raise RuntimeError(f"Submodule path escapes its parent: {child}") from error
        if child == repo:
            raise RuntimeError(f"Submodule path points to its parent: {child}")
        if not (child / ".git").exists():
            # Not cloned yet: leave normal initialization to submodule update.
            continue
        actual_root = Path(
            os.fsdecode(_git(child, "rev-parse", "--show-toplevel").stdout[:-1])
        ).resolve()
        if actual_root != child:
            raise RuntimeError(f"Invalid submodule Git directory: {child}")
        if _is_partial_clone(child):
            continue
        head = _git(child, "rev-parse", "--verify", "HEAD^{commit}", check=False)
        if head.returncode:
            _validate_empty_clone(child, commit)
            yield repo, relative_path, child, commit
        elif head.stdout.decode("ascii").strip() == commit:
            # Only inspect descendants of the version actually selected by
            # the parent. Normal update must switch an outdated parent first.
            yield from _recovery_candidates(child)


def repair_submodule_heads(repo: Path, dry_run: bool = False) -> int:
    # Complete the read-only preflight before modifying any candidate. A later
    # unsafe checkout must not cause an earlier candidate to be changed first.
    candidates = list(_recovery_candidates(repo))
    for parent, relative_path, child, commit in candidates:
        if not dry_run:
            # Re-check immediately before writing: the preflight above ran for
            # every candidate, so concurrent Git activity may have invalidated
            # this one. A dry run writes nothing and needs no re-check.
            if (relative_path, commit) not in _gitlinks(parent):
                raise RuntimeError(f"Parent submodule entry changed: {child}")
            head = _git(child, "rev-parse", "--verify", "HEAD^{commit}", check=False)
            if head.returncode == 0:
                raise RuntimeError(f"HEAD changed during recovery: {child}")
            _validate_empty_clone(child, commit)
        action = "Would recover" if dry_run else "Recovering"
        print(
            f"[submodule-repair] {action} {child} at pinned commit {commit}",
            flush=True,
        )
        if not dry_run:
            _git(child, "checkout", "--detach", "--no-overwrite-ignore", commit)
    return len(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        repo = args.repository.resolve()
        _git(repo, "rev-parse", "--verify", "HEAD^{commit}")
        repaired = repair_submodule_heads(repo, args.dry_run)
        if repaired:
            action = "Would recover" if args.dry_run else "Recovered"
            print(f"[submodule-repair] {action} {repaired} empty submodule checkout(s).")
    except (OSError, RuntimeError) as error:
        print(f"[submodule-repair] ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
