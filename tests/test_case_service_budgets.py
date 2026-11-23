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
                       rpc_server_port=38889, role="ROUTER",
                       extra_service_args=None)


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
    argv = cluster({"extra_service_args": options})._service_argv(item)
    for name, value in options.items():
        assert f"--{name}={value}" in argv


def test_new_flag_passes_without_code_change():
    argv = cluster({"extra_service_args": {"some_new_flag": 123, "str_flag": "abc"}})._service_argv(ITEM)
    assert "--some_new_flag=123" in argv
    assert "--str_flag=abc" in argv


def test_tokenizer_path_override_is_appended_after_default():
    override = "/export/home/models/joyo_v2_diffusers_stage3p5_it526400/tokenizer"
    argv = cluster({"extra_service_args": {"tokenizer_path": override}})._service_argv(ITEM)
    # gflags accepts a repeated flag and keeps the last value.
    assert argv.count(f"--tokenizer_path={override}") == 1
    assert argv.index(f"--tokenizer_path={override}") > argv.index(
        "--tokenizer_path=/models/test"
    )


@pytest.mark.parametrize("name", ["with-dash", "with space", "9leading", "", "--inject=1"])
def test_invalid_flag_name_is_rejected(name):
    with pytest.raises(ValueError):
        cluster({"extra_service_args": {name: 1}})._service_argv(ITEM)


@pytest.mark.parametrize("value", [True, None, {"a": 1}, ["x"]])
def test_non_scalar_value_is_rejected(value):
    with pytest.raises(TypeError):
        cluster({"extra_service_args": {"max_request_deadline_ms": value}})._service_argv(ITEM)
