#!/usr/bin/env python3
"""HCCL / xLLM ranktable query, build, merge and split helpers."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


class RanktableError(RuntimeError):
    """Raised when NPU / ranktable generation fails."""


def _run(cmd: Sequence[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RanktableError(f"command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RanktableError(f"command timed out: {' '.join(cmd)}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RanktableError(
            f"command failed ({result.returncode}): {' '.join(cmd)}"
            + (f"\n{detail}" if detail else "")
        )
    return result.stdout


def detect_local_ip() -> str:
    """Best-effort local IP used as server_id/host_ip."""
    try:
        from xtask.utils import get_host_ip

        return get_host_ip()
    except Exception:
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except Exception:
        pass

    try:
        out = _run(["hostname", "-I"])
        for token in out.split():
            if re.match(r"^\d+\.\d+\.\d+\.\d+$", token) and not token.startswith("127."):
                return token
    except Exception:
        pass

    raise RanktableError("unable to detect local IP; pass --server-ip explicitly")


def parse_device_ids(spec: Optional[str], available: Sequence[int]) -> List[int]:
    """Parse --device-ids like '0-15' or '0,1,2,3'."""
    if not spec:
        return list(available)

    selected: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise RanktableError(f"invalid device range: {part}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))

    unknown = [item for item in selected if item not in available]
    if unknown:
        raise RanktableError(
            f"device ids not present on this host: {unknown}; available={list(available)}"
        )
    return list(dict.fromkeys(selected))


def format_device_ids(device_ids: Sequence[int]) -> str:
    """Compact device id list for CLI, e.g. 0-15 or 0,1,2."""
    ids = list(dict.fromkeys(int(item) for item in device_ids))
    if not ids:
        return ""
    if ids == list(range(ids[0], ids[-1] + 1)):
        return f"{ids[0]}-{ids[-1]}" if ids[0] != ids[-1] else str(ids[0])
    return ",".join(str(item) for item in ids)


def parse_npu_logic_mapping(mapping_text: Optional[str] = None) -> Dict[int, Tuple[int, int]]:
    """Parse `npu-smi info -m` into logic_id -> (npu_id, chip_id)."""
    text = mapping_text if mapping_text is not None else _run(["npu-smi", "info", "-m"])
    mapping: Dict[int, Tuple[int, int]] = {}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) < 5:
            continue
        if not cols[0].isdigit() or not cols[1].isdigit():
            continue
        if cols[2] == "-" or not cols[2].isdigit():
            continue
        mapping[int(cols[2])] = (int(cols[0]), int(cols[1]))

    if not mapping:
        raise RanktableError("no Ascend chips found in `npu-smi info -m` output")
    return dict(sorted(mapping.items()))


def query_vnic_ip(logic_id: int) -> str:
    out = _run(["hccn_tool", "-i", str(logic_id), "-vnic", "-g"])
    match = re.search(r"vnic\s+ipaddr:\s*([0-9.]+)", out, re.IGNORECASE)
    if not match:
        raise RanktableError(f"failed to parse vnic ip for logic device {logic_id}:\n{out}")
    return match.group(1)


def query_super_device_id(npu_id: int, chip_id: int) -> str:
    out = _run(
        ["npu-smi", "info", "-t", "spod-info", "-i", str(npu_id), "-c", str(chip_id)]
    )
    match = re.search(r"SDID\s*:\s*(\d+)", out)
    if not match:
        raise RanktableError(
            f"failed to parse SDID for npu={npu_id} chip={chip_id}:\n{out}"
        )
    return match.group(1)


def query_super_pod_id(npu_id: int, chip_id: int) -> Optional[str]:
    try:
        out = _run(
            ["npu-smi", "info", "-t", "spod-info", "-i", str(npu_id), "-c", str(chip_id)]
        )
    except RanktableError:
        return None
    match = re.search(r"Super Pod ID\s*:\s*(\d+)", out)
    return match.group(1) if match else None


def collect_devices(
    device_ids: Sequence[int], mapping: Dict[int, Tuple[int, int]]
) -> List[dict]:
    devices = []
    for rank_index, logic_id in enumerate(device_ids):
        npu_id, chip_id = mapping[logic_id]
        devices.append(
            {
                "device_id": str(logic_id),
                "device_ip": query_vnic_ip(logic_id),
                "rank_id": str(rank_index),
                "super_device_id": query_super_device_id(npu_id, chip_id),
            }
        )
    return devices


def build_hccl_ranktable(
    server_ip: str,
    devices: List[dict],
    *,
    include_super_pod: bool = False,
    super_pod_id: Optional[str] = None,
) -> dict:
    """Build single-host HCCL ranktable compatible with xLLM."""
    ranktable: Dict[str, Any] = {
        "version": "1.2",
        "server_count": "1",
        "server_list": [
            {
                "server_id": server_ip,
                "host_ip": server_ip,
                "container_ip": server_ip,
                "device": devices,
                "host_nic_ip": "reserve",
            }
        ],
        "status": "completed",
    }
    if include_super_pod and super_pod_id is not None:
        ranktable["super_pod_list"] = [
            {
                "super_pod_id": str(super_pod_id),
                "server_list": [{"server_id": server_ip}],
            }
        ]
    return ranktable


def write_ranktable(path: str, ranktable: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ranktable, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.chmod(path, 0o640)


def load_ranktable(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_ranktables(
    host_tables: Sequence[dict],
    *,
    super_pod_id: Optional[str] = None,
) -> dict:
    """Merge single-host ranktables into one interconnected multi-host table.

    rank_id is renumbered globally in server_list order; device_id stays local.
    """
    if not host_tables:
        raise RanktableError("merge_ranktables requires at least one host table")

    merged_servers: List[dict] = []
    rank_offset = 0
    server_ids: List[str] = []
    resolved_pod = super_pod_id

    for table in host_tables:
        servers = table.get("server_list") or []
        if not servers:
            raise RanktableError("host ranktable missing server_list")
        if resolved_pod is None:
            pods = table.get("super_pod_list") or []
            if pods and pods[0].get("super_pod_id") is not None:
                resolved_pod = str(pods[0]["super_pod_id"])

        for server in servers:
            server_copy = copy.deepcopy(server)
            devices = []
            for device in server_copy.get("device") or []:
                device_copy = copy.deepcopy(device)
                device_copy["rank_id"] = str(rank_offset)
                rank_offset += 1
                devices.append(device_copy)
            server_copy["device"] = devices
            merged_servers.append(server_copy)
            server_ids.append(str(server_copy.get("server_id") or server_copy.get("host_ip")))

    merged: Dict[str, Any] = {
        "version": "1.2",
        "server_count": str(len(merged_servers)),
        "server_list": merged_servers,
        "status": "completed",
    }
    if resolved_pod is not None and server_ids:
        merged["super_pod_list"] = [
            {
                "super_pod_id": str(resolved_pod),
                "server_list": [{"server_id": sid} for sid in dict.fromkeys(server_ids)],
            }
        ]
    return merged


def split_ranktable(merged: dict) -> List[dict]:
    """Split a multi-host ranktable back into per-host tables with local rank_id."""
    result: List[dict] = []
    pods = merged.get("super_pod_list") or []
    pod_id = None
    if pods and pods[0].get("super_pod_id") is not None:
        pod_id = str(pods[0]["super_pod_id"])

    for server in merged.get("server_list") or []:
        server_copy = copy.deepcopy(server)
        devices = []
        for local_rank, device in enumerate(server_copy.get("device") or []):
            device_copy = copy.deepcopy(device)
            device_copy["rank_id"] = str(local_rank)
            devices.append(device_copy)
        server_copy["device"] = devices
        server_ip = str(server_copy.get("server_id") or server_copy.get("host_ip"))
        table: Dict[str, Any] = {
            "version": "1.2",
            "server_count": "1",
            "server_list": [server_copy],
            "status": "completed",
        }
        if pod_id is not None:
            table["super_pod_list"] = [
                {
                    "super_pod_id": pod_id,
                    "server_list": [{"server_id": server_ip}],
                }
            ]
        result.append(table)
    return result


def generate_ranktable_for_host(
    server_ip: Optional[str] = None,
    device_ids_spec: Optional[str] = None,
    output_path: Optional[str] = None,
    include_super_pod: bool = False,
) -> dict:
    mapping = parse_npu_logic_mapping()
    available = list(mapping.keys())
    selected = parse_device_ids(device_ids_spec, available)
    server_ip = server_ip or detect_local_ip()
    devices = collect_devices(selected, mapping)

    super_pod_id = None
    if include_super_pod and selected:
        npu_id, chip_id = mapping[selected[0]]
        super_pod_id = query_super_pod_id(npu_id, chip_id)

    ranktable = build_hccl_ranktable(
        server_ip,
        devices,
        include_super_pod=include_super_pod,
        super_pod_id=super_pod_id,
    )

    if output_path:
        write_ranktable(output_path, ranktable)
        print(f"RANKTABLE JSON 文件已生成：{output_path}")
        print(f"  server_ip={server_ip}, devices={len(devices)}, ids={selected}")
    return ranktable


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query local NPU vnic/SDID and generate an xLLM HCCL ranktable "
            "(same schema as config/ranktable/ranktable_75_16.json)."
        )
    )
    parser.add_argument(
        "--server-ip",
        default=None,
        help="Host IP written into server_id/host_ip/container_ip (default: auto-detect)",
    )
    parser.add_argument(
        "--device-ids",
        default=None,
        help="Logic device ids, e.g. '0-15' or '0,1,2,3' (default: all Ascend chips)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON path (default: print to stdout)",
    )
    parser.add_argument(
        "--include-super-pod",
        action="store_true",
        help="Also write super_pod_list from npu-smi spod-info",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        ranktable = generate_ranktable_for_host(
            server_ip=args.server_ip,
            device_ids_spec=args.device_ids,
            output_path=args.output,
            include_super_pod=args.include_super_pod,
        )
    except RanktableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.output:
        print(json.dumps(ranktable, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
