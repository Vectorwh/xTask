"""xtask.core.gedit_service 单元测试。

覆盖 gedit 自服务模式（deployment.encoder/dit/judge，与
deployment.benchmark 等同级兄弟键）的：
- 激活门槛（仅 image_edit / gedit aisbench 且 external_service 关闭）
- spec 构建（模型路径缺省、端口/设备冲突校验、port 单值按 rank 自增）
- 启动脚本渲染（路径映射、per-rank 端口、ext 参数展开）
- task_config 地址注入（host_addr/port、encoder_url、judge_endpoint）
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.core.gedit_service import (
    GeditServiceSpec,
    apply_gedit_services_to_task_config,
    build_gedit_service_specs,
    gedit_health_url,
    is_gedit_services_enabled,
    render_gedit_startup_script,
)


def _encoder_service():
    return {
        "device_ids": [7],
        "port": 18007,
        "master_node_addr": "127.0.0.1:29798",
        "serving_extensiable_param": {
            "max_memory_utilization": 0.86,
            "task": "embed",
            "enable_prefix_cache": False,
        },
    }


def _dit_service():
    return {
        "device_ids": [0, 1, 2, 3, 4, 5],
        "port": 19000,
        "master_node_addr": "127.0.0.1:19757",
        "serving_extensiable_param": {
            "backend": "dit",
            "cfg_size": 2,
            "sp_size": 3,
        },
    }


def _image_edit_config(services=None):
    return {
        "case_name": "case_gedit",
        "model_path": "/export/home/models/Qwen-Image-Edit-2509",
        "host_addr": "placeholder",
        "port": 14830,
        "deployment": (
            services
            if services is not None
            else {"encoder": _encoder_service(), "dit": _dit_service()}
        ),
        "benchmark": {
            "benchmark_type": "image_edit",
            "perf": {
                "num_prompts": 20,
            },
        },
    }


def _aisbench_gedit_config(services=None):
    return {
        "case_name": "case_gedit_acc",
        "model_path": "/export/home/models/Qwen-Image-Edit-2509",
        "host_addr": "placeholder",
        "port": 14830,
        "metrics": ["acc"],
        "deployment": (
            services
            if services is not None
            else {"encoder": _encoder_service(), "dit": _dit_service()}
        ),
        "benchmark": {
            "benchmark_type": "aisbench",
            "acc": {
                "api_type": "gedit_image_edit",
                "dataset_conf": "gedit_gen_0_shot_llmjudge",
                "generation_kwargs": {"temperature": 0.01},
            },
        },
    }


class GeditServiceActivationTests(unittest.TestCase):
    def test_image_edit_with_services_is_enabled(self):
        self.assertTrue(is_gedit_services_enabled(_image_edit_config()))

    def test_aisbench_gedit_api_is_enabled(self):
        self.assertTrue(is_gedit_services_enabled(_aisbench_gedit_config()))

    def test_non_gedit_benchmark_type_is_disabled(self):
        config = _image_edit_config()
        config["benchmark"]["benchmark_type"] = "evalscope"
        self.assertFalse(is_gedit_services_enabled(config))

    def test_aisbench_non_gedit_api_is_disabled(self):
        config = _aisbench_gedit_config()
        config["benchmark"]["acc"]["api_type"] = "vllm_api_general_chat"
        self.assertFalse(is_gedit_services_enabled(config))

    def test_missing_services_block_is_disabled(self):
        config = _image_edit_config()
        config.pop("deployment")
        self.assertFalse(is_gedit_services_enabled(config))

    def test_deployment_without_gedit_keys_is_disabled(self):
        # deployment 存在但只含多机实例键：不得误触发 gedit 自服务。
        config = _image_edit_config()
        config["deployment"] = {"instances": [{"host": "10.0.0.1"}]}
        self.assertFalse(is_gedit_services_enabled(config))

    def test_legacy_benchmark_gedit_services_is_ignored(self):
        # 旧字段 benchmark.gedit_services 已删除，不得再激活。
        config = _image_edit_config()
        config.pop("deployment")
        config["benchmark"]["gedit_services"] = {
            "encoder": _encoder_service(),
        }
        self.assertFalse(is_gedit_services_enabled(config))

    def test_external_service_overrides_gedit_services(self):
        config = _image_edit_config()
        config["external_service"] = True
        self.assertFalse(is_gedit_services_enabled(config))


class GeditServiceSpecTests(unittest.TestCase):
    def test_encoder_model_path_derives_from_task_model(self):
        specs = build_gedit_service_specs(_image_edit_config())
        encoder = next(spec for spec in specs if spec.role == "encoder")
        self.assertEqual(
            encoder.model_path,
            "/export/home/models/Qwen-Image-Edit-2509/text_encoder",
        )

    def test_dit_is_default_benchmark_runner(self):
        specs = build_gedit_service_specs(_image_edit_config())
        dit = next(spec for spec in specs if spec.role == "dit")
        self.assertTrue(dit.benchmark_runner)
        encoder = next(spec for spec in specs if spec.role == "encoder")
        self.assertFalse(encoder.benchmark_runner)

    def test_consecutive_ports_from_single_port(self):
        # DiT 6 卡、起始端口 19000：按 rank 连续自增 -> 19000..19005。
        specs = build_gedit_service_specs(_image_edit_config())
        dit = next(spec for spec in specs if spec.role == "dit")
        self.assertEqual(
            dit.ports, [19000, 19001, 19002, 19003, 19004, 19005]
        )

    def test_ports_list_is_rejected(self):
        # ports 列表已删除，只支持 port 单值 + 按 rank 自增。
        services = {"dit": _dit_service()}
        services["dit"] = dict(services["dit"])
        services["dit"]["ports"] = [19000, 19001, 19002, 19003, 19004, 19012]
        with self.assertRaises(ValueError) as ctx:
            build_gedit_service_specs(_image_edit_config(services))
        self.assertIn("no longer supported", str(ctx.exception))

    def test_missing_port_rejected(self):
        services = {"dit": _dit_service()}
        services["dit"] = dict(services["dit"])
        del services["dit"]["port"]
        with self.assertRaises(ValueError) as ctx:
            build_gedit_service_specs(_image_edit_config(services))
        self.assertIn("requires port", str(ctx.exception))

    def test_device_conflict_rejected(self):
        # dit 复用 encoder 的 device 7：同设备分配给两个服务必须报错。
        services = {"encoder": _encoder_service()}
        conflict = _dit_service()
        conflict["device_ids"] = [7, 8, 9, 10, 11, 13]
        services["dit"] = conflict
        with self.assertRaises(ValueError) as ctx:
            build_gedit_service_specs(_image_edit_config(services))
        self.assertIn("Device 7", str(ctx.exception))

    def test_port_conflict_rejected(self):
        services = {"encoder": _encoder_service()}
        conflict = _dit_service()
        conflict["device_ids"] = [0]
        conflict["port"] = 18007
        services["dit"] = conflict
        with self.assertRaises(ValueError) as ctx:
            build_gedit_service_specs(_image_edit_config(services))
        self.assertIn("18007", str(ctx.exception))

    def test_judge_requires_model_path(self):
        services = {"encoder": _encoder_service(), "dit": _dit_service()}
        services["judge"] = {"device_ids": [12], "port": 18008}
        with self.assertRaises(ValueError) as ctx:
            build_gedit_service_specs(_image_edit_config(services))
        self.assertIn("model_path", str(ctx.exception))

    def test_judge_with_model_path_builds(self):
        services = {"encoder": _encoder_service(), "dit": _dit_service()}
        services["judge"] = {
            "device_ids": [12],
            "port": 18008,
            "model_path": "/export/home/models/Qwen2.5-VL-7B-Instruct",
        }
        specs = build_gedit_service_specs(_image_edit_config(services))
        judge = next(spec for spec in specs if spec.role == "judge")
        self.assertEqual(judge.ports, [18008])
        self.assertEqual(
            judge.model_path, "/export/home/models/Qwen2.5-VL-7B-Instruct"
        )

    def test_empty_services_rejected(self):
        with self.assertRaises(ValueError):
            build_gedit_service_specs(
                _image_edit_config(services={})
            )

    def test_deployment_instances_coexist_with_gedit_keys(self):
        # deployment.encoder/dit/judge 与多机实例键共存时只取 gedit 三键。
        config = _image_edit_config()
        config["deployment"] = dict(config["deployment"])
        config["deployment"]["instances"] = [{"host": "10.0.0.1"}]
        config["deployment"]["mode"] = "service_routed"
        specs = build_gedit_service_specs(config)
        self.assertEqual(
            sorted(spec.role for spec in specs), ["dit", "encoder"]
        )


class GeditStartupScriptTests(unittest.TestCase):
    def _render(self, spec, output_dir):
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            return render_gedit_startup_script(spec, output_dir)

    def test_script_renders_per_rank_commands(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            specs = build_gedit_service_specs(_image_edit_config())
            dit = next(spec for spec in specs if spec.role == "dit")
            script_path = self._render(dit, temporary_directory)
            content = Path(script_path).read_text(encoding="utf-8")

        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=0", content)
        self.assertIn("--node_rank=0", content)
        self.assertIn("--node_rank=5", content)
        self.assertIn("--port 19000", content)
        self.assertIn("--port 19005", content)
        self.assertIn("--backend=dit", content)
        self.assertIn("--cfg_size=2", content)
        self.assertIn("> $LOG_DIR/node_5.log 2>&1 &", content)

    def test_script_maps_paths_into_container(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            specs = build_gedit_service_specs(_image_edit_config())
            encoder = next(spec for spec in specs if spec.role == "encoder")
            with mock.patch(
                "xtask.utils.container_manager.path_mapping_in_container",
                side_effect=lambda path: (
                    "/workspace" + path.replace(
                        "/export/home/models", "/export/home/models"
                    )
                    if path.startswith("/export/home/models")
                    else path
                ),
            ):
                script_path = render_gedit_startup_script(
                    encoder, temporary_directory
                )
            content = Path(script_path).read_text(encoding="utf-8")
        self.assertIn(
            'MODEL_PATH="/workspace/export/home/models', content
        )

    def test_bool_flag_rendering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            specs = build_gedit_service_specs(_image_edit_config())
            encoder = next(spec for spec in specs if spec.role == "encoder")
            script_path = self._render(encoder, temporary_directory)
            content = Path(script_path).read_text(encoding="utf-8")
        self.assertIn("--enable_prefix_cache=false", content)
        self.assertIn("--task=embed", content)

    def test_script_exports_runtime_lib_dirs(self):
        """LD_LIBRARY_PATH must point at the xllm build libs (libetcd_wrapper)."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            specs = build_gedit_service_specs(_image_edit_config())
            dit = next(spec for spec in specs if spec.role == "dit")
            script_path = self._render(dit, temporary_directory)
            content = Path(script_path).read_text(encoding="utf-8")
        self.assertIn(
            "export LD_LIBRARY_PATH=", content,
        )
        # Default runtime lib dir is the workspace xllm build tree.
        self.assertIn(
            "xllm/build/lib.linux-aarch64-cpython-312/xllm", content,
        )

    def test_script_sources_vendors_env(self):
        """custom_xllm_math set_env.bash provides libcust_opapi.so."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            specs = build_gedit_service_specs(_image_edit_config())
            encoder = next(spec for spec in specs if spec.role == "encoder")
            script_path = self._render(encoder, temporary_directory)
            content = Path(script_path).read_text(encoding="utf-8")
        self.assertIn(
            "opp/vendors/custom_xllm_math/bin/set_env.bash", content,
        )

    def test_runtime_lib_dirs_yaml_override(self):
        services = {"dit": _dit_service()}
        services["dit"] = dict(services["dit"])
        services["dit"]["runtime_lib_dirs"] = ["/opt/custom/libs"]
        specs = build_gedit_service_specs(_image_edit_config(services))
        self.assertEqual(specs[0].runtime_lib_dirs, ["/opt/custom/libs"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            script_path = self._render(specs[0], temporary_directory)
            content = Path(script_path).read_text(encoding="utf-8")
        self.assertIn('export LD_LIBRARY_PATH="/opt/custom/libs:', content)


class GeditTaskConfigApplyTests(unittest.TestCase):
    def test_runner_endpoint_and_encoder_url_applied(self):
        config = _image_edit_config()
        specs = build_gedit_service_specs(config)
        apply_gedit_services_to_task_config(config, specs)
        self.assertEqual(config["host_addr"], "127.0.0.1")
        self.assertEqual(config["port"], 19000)
        self.assertEqual(
            config["benchmark"]["perf"]["encoder_url"],
            "http://127.0.0.1:18007",
        )

    def test_judge_endpoint_applied_for_aisbench(self):
        config = _aisbench_gedit_config()
        config["deployment"]["judge"] = {
            "device_ids": [12],
            "port": 18008,
            "model_path": "/export/home/models/Qwen2.5-VL-7B-Instruct",
        }
        specs = build_gedit_service_specs(config)
        apply_gedit_services_to_task_config(config, specs)
        self.assertEqual(config["host_addr"], "127.0.0.1")
        self.assertEqual(config["port"], 19000)
        self.assertEqual(
            config["benchmark"]["acc"]["judge_endpoint"],
            "127.0.0.1:18008",
        )
        self.assertEqual(
            config["benchmark"]["acc"]["generation_kwargs"]["encoder_url"],
            "http://127.0.0.1:18007",
        )

    def test_explicit_yaml_values_are_preserved(self):
        config = _image_edit_config()
        config["benchmark"]["perf"]["encoder_url"] = "http://10.0.0.9:28007"
        specs = build_gedit_service_specs(config)
        apply_gedit_services_to_task_config(config, specs)
        self.assertEqual(
            config["benchmark"]["perf"]["encoder_url"],
            "http://10.0.0.9:28007",
        )

    def test_health_url(self):
        specs = build_gedit_service_specs(_image_edit_config())
        encoder = next(spec for spec in specs if spec.role == "encoder")
        self.assertEqual(
            gedit_health_url(encoder), "http://127.0.0.1:18007/health"
        )


if __name__ == "__main__":
    unittest.main()
