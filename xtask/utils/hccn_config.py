"""Prepare HCCN on each Engine host before creating its PD container."""
import json
import logging
import os
import shlex
import uuid


def prepare_runtime_hccn(plan, config, executor):
    deployment = plan.instance_config.get("deployment") or {}
    if deployment.get("mode") != "pd_disaggregated":
        return
    settings = deployment.get("hccn") or {}
    if not isinstance(settings, dict):
        raise ValueError("deployment.hccn must be a mapping")
    mode = settings.get("mode", "auto")
    if mode == "disabled":
        return
    # Refuse a caller-provided writable host bind even if querying fails.
    for host_path, destination in list(config.volumes.items()):
        if str(destination).split(":")[0] == "/etc/hccn.conf":
            raise ValueError("PD HCCN host bind is unsafe; remove it and use deployment.hccn.source")
    from xtask.utils.workspace_sync import resolve_workspace_root

    helper = str(resolve_workspace_root() / "tools/prepare_hccn_config.py")
    # Remote Engine script sync wipes plan.output_dir just before launch. Keep
    # the bind source in its sibling, otherwise the live mount gets unlinked.
    output = os.path.join(os.path.dirname(plan.output_dir), ".hccn", uuid.uuid4().hex + ".conf")
    command = shlex.join([
        "python3", helper, "--source", settings.get("source", "/etc/hccn.conf"),
        "--devices", ",".join(str(device) for device in plan.device_ids),
        "--output", output, "--mode", mode,
    ])
    result = executor.run(command, timeout=max(30, len(plan.device_ids) * 12))
    if not result.success:
        raise ValueError("HCCN preparation failed on %s: %s" % (plan.host, result.stderr))
    try:
        probe = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid HCCN preparation result on %s" % plan.host) from exc
    if not probe.get("path"):
        logging.warning("HCCN on %s: %s; retain device-tool fallback", plan.host, probe.get("reason"))
        return
    if probe["path"] != output:
        raise ValueError("Unexpected HCCN output path")
    config.volumes[output] = "/etc/hccn.conf:ro"
    logging.info("Prepared read-only HCCN copy for %s, devices %s", plan.host, plan.device_ids)
