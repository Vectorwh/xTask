#!/usr/bin/env python3
"""Create a container-only HCCN copy using read-only physical-device queries."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def prepare(source, devices, output, mode="auto", query=None):
    if mode not in ("auto", "strict", "disabled"):
        raise ValueError("HCCN mode must be auto, strict or disabled")
    if mode == "disabled":
        return {"path": None, "reason": "disabled"}
    devices = list(dict.fromkeys(int(device) for device in devices))
    if not devices or any(device < 0 for device in devices):
        raise ValueError("HCCN requires non-negative physical device IDs")
    source = Path(source)
    if source.exists() and not source.is_file():
        raise ValueError("HCCN source is not a regular file")
    text = source.read_text() if source.is_file() else ""
    addresses = {}
    for device, address in re.findall(r"^\s*address_(\d+)\s*=\s*([^\s#]+)", text, re.M):
        device = int(device)
        address = str(ipaddress.ip_address(address))
        if device in addresses and addresses[device] != address:
            raise ValueError("Conflicting HCCN address entries for device %s" % device)
        addresses[device] = address
    if query is None:
        tool = next((path for path in (
            shutil.which("hccn_tool"), "/usr/local/Ascend/driver/tools/hccn_tool",
            "/usr/local/Ascend/driver/bin/hccn_tool", "/usr/local/sbin/hccn_tool"
        ) if path and os.path.isfile(path) and os.access(path, os.X_OK)), None)

        def query(device):
            if tool is None:
                return None
            result = subprocess.run([tool, "-i", str(device), "-ip", "-g"],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode:
                return None
            match = re.search(r"(?:ipaddr|ipaddress|ip_address|ip)\s*[:=]\s*(\S+)",
                              result.stdout, re.I)
            return str(ipaddress.ip_address(match.group(1))) if match else None
    missing = []
    additions = []
    for device in devices:
        try:
            address = query(device)
        except subprocess.TimeoutExpired:
            address = None
        if address:
            address = str(ipaddress.ip_address(address))
            if device in addresses and addresses[device] != address:
                raise ValueError("HCCN address conflicts with device %s query" % device)
            if device not in addresses:
                additions.append("address_%s=%s\n" % (device, address))
            addresses[device] = address
        if device not in addresses:
            missing.append(device)
    if missing:
        if mode == "strict":
            raise ValueError("Unable to resolve HCCN addresses for devices %s" % missing)
        # Never mount an incomplete TLS-only file: intentional tool fallback is
        # preferable to changing the host's configuration or inventing an IP.
        return {"path": None, "reason": "unresolved device addresses", "devices": missing}
    output = Path(output)
    if output.resolve() == source.resolve():
        raise ValueError("HCCN output must not replace the host source")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".hccn-", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text + ("\n" if text and not text.endswith("\n") else ""))
            stream.writelines(additions)
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(output), "devices": devices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/etc/hccn.conf")
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="auto", choices=("auto", "strict", "disabled"))
    args = parser.parse_args()
    try:
        devices = args.devices.split(",")
        if args.devices == "all":
            devices = [path.name[len("davinci"):] for path in Path("/dev").glob("davinci*")
                       if path.name[len("davinci"):].isdigit()]
            if not devices and args.mode == "auto":
                print(json.dumps({"path": None, "reason": "no physical NPU devices"}))
                return
        print(json.dumps(prepare(args.source, devices, args.output, args.mode)))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, "HCCN preparation failed: %s\n" % exc)


if __name__ == "__main__":
    main()
