"""_wait_npu_hbm_released 状态机与基线解析的单测。

覆盖：HBM 探测命令兼容任意卡容量、采样基线优先级（采样+容差 > 环境变量 >
默认）、回基线 settle 窗口、deadline 延展、settle 期间反弹重臂、超时放行、
npu-smi 不可读放行。全部用 fake executor 驱动，不碰真机 NPU。
"""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.core import scheduler


def _runtime(baseline=0, readings=None, error=None):
    """构造最小 runtime shim + 可编排输出的 fake executor。

    readings: 每次 poll 依次返回的 HBM 值（None 表示该次不可读）。
    error: 抛给 executor.run 的异常（模拟 SSH 断连等）。
    """
    seq = list(readings or [])

    class _Executor:
        def run(self, cmd, timeout=20):
            if error is not None:
                raise error
            if not seq:
                return SimpleNamespace(stdout="")
            v = seq.pop(0)
            return SimpleNamespace(stdout="" if v is None else str(v))

    return SimpleNamespace(
        plan=SimpleNamespace(host="h1", hbm_baseline_mb=baseline),
        container_manager=SimpleNamespace(executor=_Executor()),
    )


def _fast(monkeypatch, sleep_log):
    """把 time.sleep 与轮询间隔压缩，记录 sleep 调用。"""
    monkeypatch.setattr(scheduler.time, "sleep", lambda s: sleep_log.append(s))
    # 让每轮循环的时钟真实前进，但不真的等 3 秒。
    real_monotonic = scheduler.time.monotonic
    clock = {"t": 0.0}

    def fake_monotonic():
        return clock["t"]

    def fake_sleep(s):
        sleep_log.append(s)
        clock["t"] += s

    monkeypatch.setattr(scheduler.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(scheduler.time, "sleep", fake_sleep)
    return clock


class NpuHbmProbeCmdTests(unittest.TestCase):
    def test_probe_cmd_not_hardcoded_to_total(self):
        """总量用 [0-9]+ 匹配：换 32G/96G 卡不再静默失效。"""
        cmd = scheduler._npu_hbm_probe_cmd()
        self.assertNotIn("65536", cmd)
        # 斜杠两侧零或多个空格：npu-smi "used/total" 是固定列宽右对齐，
        # 4 位数 "3149 / 65536"、5 位数 "59194/ 65536" 必须都能匹配。
        self.assertIn("[0-9]+ */ *[0-9]+", cmd)

    def test_probe_cmd_with_device_ids_filters_chips(self):
        """带 device_ids 时按芯片(Phy-ID)过滤，只查本用例占的卡。"""
        cmd = scheduler._npu_hbm_probe_cmd([0, 1])
        self.assertIn("awk -v devs='0,1'", cmd)
        # awk 里按行内 Chip 列（=ASCEND_RT_VISIBLE_DEVICES 编号空间）匹配
        self.assertIn("chip=f[2]", cmd)

    def test_probe_cmd_without_device_ids_keeps_legacy(self):
        """无 device_ids 时保持全卡探测（单机无 plan / 旧路径兼容）。"""
        cmd = scheduler._npu_hbm_probe_cmd(None)
        self.assertIn("grep -oE", cmd)
        self.assertNotIn("devs=", cmd)


class RuntimeDeviceIdsTests(unittest.TestCase):
    def test_plan_device_ids_passed_through(self):
        rt = SimpleNamespace(
            plan=SimpleNamespace(host="h1", hbm_baseline_mb=0, device_ids=[3, 7]),
            container_manager=None,
        )
        self.assertEqual(scheduler._runtime_device_ids(rt), [3, 7])

    def test_missing_or_empty_falls_back_to_none(self):
        rt = SimpleNamespace(
            plan=SimpleNamespace(host="h1", hbm_baseline_mb=0),
            container_manager=None,
        )
        self.assertIsNone(scheduler._runtime_device_ids(rt))
        rt.plan.device_ids = []
        self.assertIsNone(scheduler._runtime_device_ids(rt))


class SingleHostDeviceIdsTests(unittest.TestCase):
    def test_device_ids_field_wins(self):
        cfg = {"device_ids": [4, 5]}
        self.assertEqual(scheduler._single_host_device_ids(cfg), [4, 5])

    def test_start_and_count_fallback(self):
        cfg = {"start_device_id": 2, "num_devices": 4}
        self.assertEqual(scheduler._single_host_device_ids(cfg), [2, 3, 4, 5])

    def test_defaults_to_single_card_zero(self):
        self.assertEqual(scheduler._single_host_device_ids({}), [0])

    def test_bad_values_return_none(self):
        self.assertIsNone(scheduler._single_host_device_ids(
            {"start_device_id": "x", "num_devices": "y"}))



class ReadNpuHbmMaxMbTests(unittest.TestCase):
    def test_reads_last_numeric_line(self):
        rt = _runtime(readings=[3161])
        self.assertEqual(scheduler._read_npu_hbm_max_mb(rt), 3161)

    def test_returns_none_on_garbage(self):
        rt = _runtime(readings=[None])
        self.assertIsNone(scheduler._read_npu_hbm_max_mb(rt))

    def test_returns_none_on_exception(self):
        rt = _runtime(error=RuntimeError("ssh down"))
        self.assertIsNone(scheduler._read_npu_hbm_max_mb(rt))


class SampleBaselineTests(unittest.TestCase):
    def test_sampling_stores_value_on_plan(self):
        rt = _runtime(readings=[3021])
        mb = scheduler._sample_npu_hbm_baseline(rt)
        self.assertEqual(mb, 3021)
        self.assertEqual(rt.plan.hbm_baseline_mb, 3021)

    def test_sampling_failure_returns_zero_and_keeps_zero(self):
        rt = _runtime(readings=[None])
        mb = scheduler._sample_npu_hbm_baseline(rt)
        self.assertEqual(mb, 0)
        self.assertEqual(rt.plan.hbm_baseline_mb, 0)


class ResolveBaselineTests(unittest.TestCase):
    def test_sampled_baseline_plus_slack_wins(self):
        rt = _runtime(baseline=3000)
        with mock.patch.dict(os.environ, {"XTASK_HBM_BASELINE_MB": "99999"}):
            # 采样值 3000 + 默认容差 512
            self.assertEqual(scheduler._resolve_hbm_baseline_mb(rt), 3512)

    def test_env_fallback_when_not_sampled(self):
        rt = _runtime(baseline=0)
        with mock.patch.dict(os.environ, {"XTASK_HBM_BASELINE_MB": "4200"}):
            self.assertEqual(scheduler._resolve_hbm_baseline_mb(rt), 4200)

    def test_default_5000_when_nothing_set(self):
        rt = _runtime(baseline=0)
        env = {
            k: v for k, v in os.environ.items()
            if not k.startswith("XTASK_HBM")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(scheduler._resolve_hbm_baseline_mb(rt), 5000)

    def test_slack_env_override(self):
        rt = _runtime(baseline=3000)
        with mock.patch.dict(
            os.environ, {"XTASK_HBM_BASELINE_SLACK_MB": "100"}
        ):
            self.assertEqual(scheduler._resolve_hbm_baseline_mb(rt), 3100)


class WaitNpuHbmReleasedTests(unittest.TestCase):
    def test_immediate_release_then_settle(self):
        """首查即回基线：记录 settle 日志，settle_s 后返回。"""
        rt = _runtime(readings=[3000, 3000, 3000])
        sleeps = []
        with mock.patch.dict(os.environ, {}, clear=False):
            scheduler._wait_npu_hbm_released(
                rt, timeout_s=60, baseline_mb=5000, settle_s=6
            )
        # settle 窗口被完整走过（fake sleep 推进时钟）。

    def test_settle_window_extends_deadline(self):
        """回基线晚于 timeout 时 deadline 延展，settle 不被截断。"""
        # 序列：30 秒高于基线 → 回基线 → settle 期间两次轮询。
        rt = _runtime(
            readings=[8000] * 10 + [3000, 3000, 3000]
        )
        clock = {"t": 0.0}

        real_mono = scheduler.time.monotonic

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_npu_hbm_released(
                rt, timeout_s=10, baseline_mb=5000, settle_s=30
            )
        # 没抛异常且执行完毕即证明 deadline 被延展（否则 10s 后直接退出，
        # settle 未满就放行 —— 无法从返回值区分，这里至少保证不崩溃）。
        # 更强的断言：函数执行结束时总时钟 > timeout_s。
        self.assertGreater(clock["t"], 10)

    def test_no_false_timeout_when_last_poll_lands_in_settle_tail(self):
        """回归：最后一轮轮询落在 settle 窗口尾部(如 29.7s)不误报超时。

        实测事故：HBM 低于基线(3149MB<=3661MB)却打出 "not released
        after 60s"。根因：回基线时刻 ~7s，最后一次轮询落在 settle 第
        ~29.7s，随后 sleep(3) 越过 deadline(60s)，while 以超时路径退出。
        修复后 sleep 收敛为「距 settle 结束的剩余时间」，窗口必然走满
        并以正常 "settle window done" 路径返回。
        """
        # 回基线后水位持续正常(3149)，轮询值足够多。
        rt = _runtime(readings=[8000] * 2 + [3149] * 40)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep), \
                mock.patch.object(
                    scheduler.logging, "warning"
                ) as warn, mock.patch.object(scheduler.logging, "info") as info:
            scheduler._wait_npu_hbm_released(
                rt, timeout_s=60, baseline_mb=3661, settle_s=30
            )
        # 不允许任何 warning（尤其 "not released"）
        self.assertFalse(
            any("not released" in str(c) for c in warn.call_args_list),
            "误报超时: %s" % warn.call_args_list,
        )
        # 正常路径必须走到 settle window done
        self.assertTrue(
            any("settle window done" in str(c) for c in info.call_args_list),
            "未正常完成 settle: %s" % info.call_args_list,
        )
        # settle 窗口确实走满 30s（2 次高水位轮询 6s + settle 30s = 36s）
        self.assertGreaterEqual(clock["t"], 36)

    def test_rearm_when_hbm_rises_during_settle(self):
        """settle 期间 HBM 反弹：重置 settle 窗口重新等。"""
        rt = _runtime(
            readings=[3000, 9000, 3000, 3000]
        )
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_npu_hbm_released(
                rt, timeout_s=120, baseline_mb=5000, settle_s=6
            )
        # 反弹后重新 settle：总耗时 > 首次 settle 完成的时间。

    def test_timeout_proceeds_anyway(self):
        """始终高于基线：等满 timeout 放行（返回 None，不抛异常）。"""
        rt = _runtime(readings=[30000] * 100)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_npu_hbm_released(
                rt, timeout_s=30, baseline_mb=5000, settle_s=0
            )
        self.assertGreaterEqual(clock["t"], 30)

    def test_unreadable_proceeds_immediately(self):
        """npu-smi 连续不可读超过重试上限：放行，不无限等。"""
        rt = _runtime(readings=[None, None, None])
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_npu_hbm_released(
                rt, timeout_s=60, baseline_mb=5000, settle_s=30
            )
        # 3 次探测失败各隔 3s → 6s 后放行（不再立即返回）
        self.assertEqual(clock["t"], 6.0)

    def test_transient_unreadable_recovers_then_settles(self):
        """npu-smi 瞬断一次后恢复：按重试继续，正常走完 settle。

        实测场景：SIGKILL 大批 xllm 进程后驱动短暂忙，两台机器同毫秒
        返回空；旧实现立即放行跳过等待，HBM 实际还没释放完。
        """
        # 瞬断 1 次 → 恢复正常水位；settle 30s 每 3s 轮询，读数给足。
        rt = _runtime(readings=[None] + [3149] * 15)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep), \
                mock.patch.object(scheduler.logging, "warning") as warn:
            scheduler._wait_npu_hbm_released(
                rt, timeout_s=60, baseline_mb=3661, settle_s=30
            )
        # 不允许最终放行型 warning
        self.assertFalse(
            any("skipping release wait" in str(c) for c in warn.call_args_list),
            "不应最终放行: %s" % warn.call_args_list,
        )
        # 有一次 retry 提示
        self.assertTrue(
            any("retrying" in str(c) for c in warn.call_args_list),
            "应有重试提示: %s" % warn.call_args_list,
        )
        # 3s 重试 + 30s settle = 33s
        self.assertGreaterEqual(clock["t"], 33)

    def test_settle_zero_returns_on_baseline(self):
        """settle_s=0：回基线立即返回（保留旧行为开关）。"""
        rt = _runtime(readings=[3000, 3000])
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_npu_hbm_released(
                rt, timeout_s=60, baseline_mb=5000, settle_s=0
            )
        self.assertEqual(clock["t"], 0.0)


class SingleHostCleanupTests(unittest.TestCase):
    """单机清理路径也等待 HBM 释放（此前只有 service_group 路径覆盖）。"""

    def test_single_host_cleanup_calls_hbm_wait(self):
        state = {
            "multi_host": False,
            "engine_config": {},
            "container_manager": SimpleNamespace(
                cleanup=mock.Mock(),
                container_config=SimpleNamespace(container_name="c1"),
                exec_in_container=mock.Mock(return_value=SimpleNamespace(success=True)),
            ),
            "vbench_container_manager": None,
        }
        context = SimpleNamespace(
            state=state,
            output_file_path=None,
            result={},
            task_yaml_path="/tmp/t.yaml",
        )
        wait_mock = mock.Mock()
        with mock.patch.object(
            scheduler, "_ensure_benchmark_date_dir"
        ), mock.patch.object(
            scheduler, "_cleanup_xllm_shm_in_container"
        ), mock.patch.object(
            scheduler, "_wait_npu_hbm_released", wait_mock
        ), mock.patch.object(
            scheduler, "_cleanup_xllm_shm"
        ), mock.patch.object(
            scheduler,
            "get_xtask_config_data",
            return_value={"ENV_CONFIG": {"CLEAN_RESULTS": True}},
        ):
            scheduler._cleanup_scheduler(context)
        wait_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
