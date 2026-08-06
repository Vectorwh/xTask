"""Named PD topology choices shared by exhaustive and TPE searches."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


ASSESS_TOPOLOGY_PARAMETER = "deployment_topology"


def normalize_assess_topologies(
    base_config: Dict[str, Any],
    parameter_space: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Validate named PD topology choices used by ``para_set``.

    A topology deliberately owns only ``deployment.pd_instances``. Shared
    deployment settings such as the control service, workspace sync, and
    benchmark runner remain in the base case.
    """
    raw_topologies = base_config.get("assess_topologies")
    if raw_topologies is None:
        topologies = {}
    else:
        if not isinstance(raw_topologies, dict) or not raw_topologies:
            raise TypeError("assess_topologies must be a non-empty mapping")

        deployment = base_config.get("deployment")
        if not isinstance(deployment, dict):
            raise ValueError("assess_topologies requires a deployment mapping")
        if deployment.get("instances") not in (None, []):
            raise ValueError(
                "assess_topologies is only supported for hierarchical PD YAML; "
                "use deployment.pd_instances instead of deployment.instances"
            )

        from xtask.core.service_group import (
            normalize_pd_deployment,
            validate_pd_deployment_authoring,
        )

        topologies = {}
        for raw_name, raw_topology in raw_topologies.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("assess_topologies names must be non-empty strings")
            topology_name = raw_name.strip()
            if topology_name in topologies:
                raise ValueError(f"Duplicate assess_topologies name: {topology_name}")
            if not isinstance(raw_topology, dict):
                raise TypeError(
                    f"assess_topologies.{topology_name} must be a mapping"
                )

            unknown_fields = set(raw_topology) - {"pd_instances"}
            if unknown_fields:
                raise ValueError(
                    f"Unsupported fields in assess_topologies.{topology_name}: "
                    f"{sorted(unknown_fields)}; only pd_instances is supported"
                )
            pd_instances = raw_topology.get("pd_instances")
            if not isinstance(pd_instances, list) or not pd_instances:
                raise ValueError(
                    f"assess_topologies.{topology_name}.pd_instances must be "
                    "a non-empty list"
                )

            normalized_topology = {"pd_instances": copy.deepcopy(pd_instances)}
            candidate = copy.deepcopy(base_config)
            candidate["deployment"].pop("instances", None)
            candidate["deployment"]["pd_instances"] = copy.deepcopy(pd_instances)
            validate_pd_deployment_authoring(candidate)
            # Reuse the detailed logical-instance/node validation performed
            # immediately before scheduling, without changing the retained
            # authoring topology above.
            normalize_pd_deployment(candidate)
            topologies[topology_name] = normalized_topology

    if parameter_space is not None:
        choices = parameter_space.get(ASSESS_TOPOLOGY_PARAMETER)
        if choices is not None:
            if not topologies:
                raise ValueError(
                    f"assess_test_param.{ASSESS_TOPOLOGY_PARAMETER} requires "
                    "assess_topologies"
                )
            for choice in choices:
                if not isinstance(choice, str) or not choice.strip():
                    raise TypeError(
                        f"assess_test_param.{ASSESS_TOPOLOGY_PARAMETER} values "
                        "must be non-empty topology names"
                    )
                if choice.strip() not in topologies:
                    raise KeyError(
                        f"Unknown deployment topology {choice!r}; expected one of "
                        f"{sorted(topologies)}"
                    )
    return topologies


def apply_assess_topology(
    config: Dict[str, Any],
    topology_name: Any,
    topologies: Dict[str, Dict[str, Any]],
) -> list:
    """Replace the candidate's hierarchical PD layout with one named choice."""
    if not isinstance(topology_name, str) or topology_name.strip() not in topologies:
        raise KeyError(f"Unknown deployment topology: {topology_name!r}")
    topology_name = topology_name.strip()
    deployment = config.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("deployment_topology requires a deployment mapping")
    deployment.pop("instances", None)
    deployment["pd_instances"] = copy.deepcopy(
        topologies[topology_name]["pd_instances"]
    )
    return ["deployment.pd_instances"]


def resolved_topology_metadata(
    parameters: Dict[str, Any],
    topologies: Dict[str, Dict[str, Any]],
    candidate_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return the selected name and final effective PD layout for a trial."""
    topology_name = parameters.get(ASSESS_TOPOLOGY_PARAMETER)
    if topology_name is None:
        return None
    topology_name = topology_name.strip()
    if candidate_config is not None:
        pd_instances = (
            (candidate_config.get("deployment") or {}).get("pd_instances")
        )
    else:
        pd_instances = topologies[topology_name]["pd_instances"]
    return {
        "name": topology_name,
        "pd_instances": copy.deepcopy(pd_instances),
    }
