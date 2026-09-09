"""PD listen-port wait: derive ports from the case plan and poll until free.

Does not start xLLM. Fake executor + fake clock; no host ss/npu-smi.
"""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.core import scheduler


def _plan(**kwargs):
    defaults = dict(
        instance_name="prefill-0",
        host="h1",
        role="prefill",
        rank_start=0,
        device_ids=[0, 1],
        instance_config={
            "enable_disagg_pd": True,
            "transfer_port_base": 26000,
            "disagg_pd_port": 7777,
        },
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _runtime(plan=None, readings=None, error=None):
    seq = list(readings or [])

    class _Executor:
        def run(self, cmd, timeout=20):
            if error is not None:
                raise error
            if not seq:
                return SimpleNamespace(stdout="")
            v = seq.pop(0)
            if v is None:
                return SimpleNamespace(stdout="")
            return SimpleNamespace(stdout=str(v))

    return SimpleNamespace(
        plan=plan or _plan(),
        container_manager=SimpleNamespace(executor=_Executor()),
    )


class PdListenPortsTests(unittest.TestCase):
    def test_non_pd_plan_returns_no_ports(self):
        plan = _plan(
            role="worker",
            instance_config={"enable_service_routing": True},
        )
        self.assertEqual(scheduler._pd_listen_ports(plan), [])

    def test_missing_plan_or_config_returns_no_ports(self):
        self.assertEqual(scheduler._pd_listen_ports(None), [])
        self.assertEqual(
            scheduler._pd_listen_ports(SimpleNamespace(role="worker")),
            [],
        )

    def test_instance_disagg_pd_port_wins_over_ext_param(self):
        plan = _plan(
            instance_config={
                "enable_disagg_pd": True,
                "disagg_pd_port": 7787,
                "transfer_port_base": 27000,
                "serving_extensiable_param": {"disagg_pd_port": 7777},
            },
            rank_start=0,
            device_ids=[0],
        )
        self.assertEqual(scheduler._pd_listen_ports(plan), [7787, 27000])

    def test_ext_param_used_when_instance_field_missing(self):
        plan = _plan(
            instance_config={
                "enable_disagg_pd": True,
                "transfer_port_base": 26000,
                "serving_extensiable_param": {"disagg_pd_port": 7797},
            },
            rank_start=0,
            device_ids=[0],
        )
        self.assertEqual(scheduler._pd_listen_ports(plan), [7797, 26000])

    def test_transfer_ports_use_base_plus_rank_and_device_count(self):
        plan = _plan(
            instance_config={
                "enable_disagg_pd": True,
                "transfer_port_base": 27000,
                "disagg_pd_port": 7787,
            },
            rank_start=4,
            device_ids=[0, 1, 2, 3],
        )
        self.assertEqual(
            scheduler._pd_listen_ports(plan),
            [7787, 27004, 27005, 27006, 27007],
        )
        self.assertNotIn(26000, scheduler._pd_listen_ports(plan))
        self.assertNotIn(7777, scheduler._pd_listen_ports(plan))

    def test_prefill_role_without_enable_flag_still_collects_ports(self):
        plan = _plan(
            role="decode",
            instance_config={"transfer_port_base": 26000, "disagg_pd_port": 7777},
            device_ids=[0],
        )
        self.assertEqual(scheduler._pd_listen_ports(plan), [7777, 26000])

    def test_probe_cmd_contains_case_ports_not_hardcoded_defaults(self):
        cmd = scheduler._pd_listen_ports_probe_cmd([7787, 27000, 27001])
        self.assertIn("7787,27000,27001", cmd)
        self.assertIn("command -v ss", cmd)
        self.assertIn("ss -lnt", cmd)
        self.assertNotIn("7777", cmd)
        self.assertNotIn("26000", cmd)


class ReadPdListenPortsTests(unittest.TestCase):
    def test_parses_busy_ports(self):
        rt = _runtime(readings=["7777\n26001\n"])
        self.assertEqual(
            scheduler._read_pd_listen_ports_in_use(rt, [7777, 26000, 26001]),
            [7777, 26001],
        )

    def test_empty_stdout_means_free(self):
        rt = _runtime(readings=[""])
        self.assertEqual(
            scheduler._read_pd_listen_ports_in_use(rt, [7777, 26000]),
            [],
        )

    def test_unreadable_returns_none(self):
        rt = _runtime(readings=["UNREADABLE"])
        self.assertIsNone(
            scheduler._read_pd_listen_ports_in_use(rt, [7777]),
        )

    def test_probe_exception_returns_none(self):
        rt = _runtime(error=RuntimeError("ssh down"))
        self.assertIsNone(
            scheduler._read_pd_listen_ports_in_use(rt, [7777]),
        )

    def test_empty_port_list_does_not_probe(self):
        rt = _runtime(error=RuntimeError("should not run"))
        self.assertEqual(scheduler._read_pd_listen_ports_in_use(rt, []), [])


class WaitPdListenPortsReleasedTests(unittest.TestCase):
    def test_non_pd_runtime_does_not_probe(self):
        rt = _runtime(
            plan=_plan(role="worker", instance_config={}),
            error=RuntimeError("should not run"),
        )
        scheduler._wait_pd_listen_ports_released(rt, timeout_s=30)

    def test_already_free_returns_without_sleep(self):
        rt = _runtime(readings=[""])
        clock = {"t": 0.0}

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_pd_listen_ports_released(rt, timeout_s=30)
        self.assertEqual(clock["t"], 0.0)

    def test_waits_until_ports_free(self):
        rt = _runtime(readings=["7777", "7777", ""])
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_pd_listen_ports_released(
                rt, timeout_s=30, interval_s=1
            )
        self.assertEqual(clock["t"], 2)

    def test_timeout_proceeds_anyway(self):
        rt = _runtime(readings=["7777"] * 100)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep), \
                mock.patch.object(scheduler.logging, "warning") as warn:
            scheduler._wait_pd_listen_ports_released(
                rt, timeout_s=5, interval_s=1
            )
        self.assertGreaterEqual(clock["t"], 5)
        self.assertTrue(
            any("proceeding anyway" in str(call) for call in warn.call_args_list),
            warn.call_args_list,
        )

    def test_unreadable_skips_wait(self):
        rt = _runtime(readings=["UNREADABLE"])
        clock = {"t": 0.0}

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_pd_listen_ports_released(rt, timeout_s=30)
        self.assertEqual(clock["t"], 0.0)

    def test_default_timeout_comes_from_env(self):
        rt = _runtime(readings=["7777"] * 100)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.dict(os.environ, {"XTASK_PD_PORT_RELEASE_TIMEOUT": "3"}), \
                mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_pd_listen_ports_released(rt, interval_s=1)
        self.assertGreaterEqual(clock["t"], 3)
        self.assertLess(clock["t"], 30)


class WaitPdListenPortsBeforeStartTests(unittest.TestCase):
    def test_non_pd_does_not_wait(self):
        with mock.patch(
            "xtask.core.service_group.run_on_runtimes_parallel"
        ) as parallel:
            scheduler._wait_pd_listen_ports_before_start(
                {"pd_disaggregated": False}, [_runtime()]
            )
            scheduler._wait_pd_listen_ports_before_start(
                {"service_routed": True}, [_runtime()]
            )
        parallel.assert_not_called()

    def test_pd_waits_on_each_runtime(self):
        runtimes = [_runtime(), _runtime()]
        with mock.patch(
            "xtask.core.service_group.run_on_runtimes_parallel"
        ) as parallel:
            scheduler._wait_pd_listen_ports_before_start(
                {"pd_disaggregated": True}, runtimes
            )
        parallel.assert_called_once_with(
            runtimes, scheduler._wait_pd_listen_ports_released
        )
