from pathlib import Path
from unittest import mock
from types import SimpleNamespace
import subprocess
import copy
import yaml
import pytest
from xtask.core.service_group import build_service_plans, get_benchmark_endpoint
from xtask.core.xllm_service_config import build_xllm_service_topology
from xtask.core.xllm_service_cluster import XllmServiceCluster
from xtask.core.result_processor import parse_perf_baseline_rules, _perf_rule_holds
from xtask.serv_fmk.xllm_serv.generate_xllm_serv import _build_script_params, _format_startup_environment
from xtask.benchmarks.aisbench import generate_aisbench_script as generator

CASES = Path(__file__).resolve().parents[1] / "tests/e2e/xllm/deepseek/deepseek_v41_1017"


@pytest.mark.parametrize("hosts", [("11.87.191.75", "11.87.191.121"), ("11.87.191.66", "11.87.191.68"), ("10.20.30.11", "10.20.30.12")])
def test_case_host_anchors_update_whole_topology_and_hccl(hosts, tmp_path):
    for path in CASES.glob("*ep16.yaml"):
        text = path.read_text().replace("&prefill_host 11.87.191.75", "&prefill_host " + hosts[0]).replace("&decode_host 11.87.191.121", "&decode_host " + hosts[1])
        case = yaml.safe_load(text)
        deployment = copy.deepcopy(case["deployment"])
        assert case["hccl_if_port"] == 43432
        assert case["progress_conn_port"] == 8773
        assert deployment["xllm_service"]["etcd"]["client_port"] == 22389
        assert deployment["xllm_service"]["etcd"]["peer_port"] == 22380
        assert deployment["xllm_service"]["transfer_port_base"] == 26000
        topology = build_xllm_service_topology(case)
        assert deployment["master_host"] == topology.etcd_host == hosts[0]
        assert topology.router.host == hosts[1]
        assert topology.router.container_instance == "decode-0-node-0"
        assert get_benchmark_endpoint(case, build_service_plans(case, str(tmp_path))) == f"http://{hosts[1]}:38888"
        for instance, host in zip(deployment["pd_instances"], hosts):
            assert instance["group_master_host"] == instance["nodes"][0]["host"] == host
            assert instance["serving_extensiable_param"]["disagg_pd_port"] == 7777
            assert instance["nodes"][0]["local_port"] == 18004
            config = {**case, **instance, "host_addr": host}
            params = _build_script_params(config)
            env = _format_startup_environment(config)
            assert 'export HCCL_IF_IP="$LOCAL_IP"' in env
            response = subprocess.check_output(["bash", "-c", "LOCAL_IP=" + params["LOCAL_IP"] + '; export HCCL_IF_IP="$LOCAL_IP"; printf %s "$HCCL_IF_IP"'], text=True)
            assert response == host
        budgets = deployment["xllm_service"]
        cluster = XllmServiceCluster.__new__(XllmServiceCluster)
        cluster.topology = topology
        cluster.control = budgets
        cluster.config = SimpleNamespace(internal_api_token="test")
        cluster.task_config = case
        for service in topology.instances:
            argv = cluster._service_argv(service)
            assert "--batch_credit_max_grant_units=16384" in argv
            assert "--batch_credit_max_grant_units_per_router_tenant=16384" in argv
            assert not any("flow_max_queue_wait_ms" in argument for argument in argv)
            if case["metrics"] == "acc":
                assert "--max_request_deadline_ms=3600000" in argv
                assert "--default_request_deadline_ms=3600000" in argv


def test_performance_yaml_rejects_failed_requests_and_allows_improvement():
    for path in CASES.glob("*perf*ep16.yaml"):
        case = yaml.safe_load(path.read_text())
        config = case["benchmark"]["perf"]
        rules = parse_perf_baseline_rules(config)
        expected = config["num_prompts"]
        measurements = dict(output_tps_avg=config["baseline"], failed_requests=0, total_requests=expected)
        assert all(_perf_rule_holds(measurements[rule["metric"]], rule) for rule in rules)
        measurements["output_tps_avg"] *= 2
        assert all(_perf_rule_holds(measurements[rule["metric"]], rule) for rule in rules)
        measurements.update(failed_requests=1, total_requests=expected-1)
        assert not all(_perf_rule_holds(measurements[rule["metric"]], rule) for rule in rules)


def test_gpqa_generated_timeout_and_thinking_config(tmp_path):
    case = yaml.safe_load(next(CASES.glob("*acc*ep16.yaml")).read_text())
    case.update(host_addr="11.87.191.68", port=38888)
    with mock.patch.object(generator, "path_mapping_in_container", side_effect=lambda x: x), mock.patch.object(generator, "download_dataset", return_value=[]), mock.patch.object(generator, "create_dataset_config_override", return_value="gpqa_gen_0_shot_cot_chat_prompt"):
        generator.generate_aisbench_script(case, str(tmp_path))
    script = (tmp_path / "run_aisbench_test.sh").read_text()
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "request_timeout" in script and "3600" in script
    assert "thinking_mode" in script and "thinking" in script
    assert "after_think" not in script and "130000" in script
