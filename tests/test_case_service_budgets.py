from types import SimpleNamespace
import pytest
from xtask.core.xllm_service_cluster import XllmServiceCluster


def cluster(control):
    value = XllmServiceCluster.__new__(XllmServiceCluster)
    value.control = control
    value.topology = SimpleNamespace(etcd_host="127.0.0.1", etcd_client_port=3389)
    value.config = SimpleNamespace(internal_api_token="test")
    value.task_config = {"model_path": "/models/test"}
    return value


ITEM = SimpleNamespace(host="127.0.0.1", http_server_port=38888,
                       rpc_server_port=38889, role="ROUTER")


def test_unconfigured_cases_keep_existing_arguments():
    argv = cluster({})._service_argv(ITEM)
    assert not any("deadline" in value or "grant_units" in value or "queue_wait" in value for value in argv)


@pytest.mark.parametrize("role", ["COORDINATOR", "ROUTER"])
def test_explicit_case_budgets_reach_each_service_role(role):
    options = {"default_request_deadline_ms": 3600000,
               "max_request_deadline_ms": 3600000,
               "batch_credit_max_grant_units": 16384,
               "batch_credit_max_grant_units_per_router_tenant": 16384}
    item = SimpleNamespace(**{**vars(ITEM), "role": role})
    argv = cluster(options)._service_argv(item)
    for name, value in options.items():
        assert f"--{name}={value}" in argv
    assert not any("flow_max_queue_wait_ms" in value for value in argv)


@pytest.mark.parametrize("value", [True, "3600", 0, -1])
def test_invalid_case_budget_is_rejected(value):
    with pytest.raises(ValueError):
        cluster({"max_request_deadline_ms": value})._service_argv(ITEM)
