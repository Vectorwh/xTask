"""HCCL listen-port wait: derive ports from the case plan and poll until free.

Does not start xLLM. Fake executor + fake clock; no host ss/npu-smi.
"""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.core import scheduler


def _plan(**kwargs):
    defaults = dict(
        instance_name="decode-0",
        host="h1",
        role="decode",
        rank_start=0,
        device_ids=[0, 1, 12],
        instance_config={
            "hccl_if_port": 43432,
            "communication_backend": "hccl",
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


class HcclListenPortsTests(unittest.TestCase):
    def test_missing_plan_returns_no_ports(self):
        self.assertEqual(scheduler._hccl_listen_ports(None), [])

    def test_non_hccl_backend_returns_no_ports(self):
        plan = _plan(instance_config={"hccl_if_port": 43432, "communication_backend": "nccl"})
        self.assertEqual(scheduler._hccl_listen_ports(plan), [])

    def test_ports_use_base_plus_device_id(self):
        plan = _plan(
            instance_config={"hccl_if_port": 43432, "communication_backend": "hccl"},
            device_ids=[0, 1, 12],
        )
        self.assertEqual(scheduler._hccl_listen_ports(plan), [43432, 43433, 43444])

    def test_default_base_when_field_missing(self):
        plan = _plan(
            instance_config={"communication_backend": "hccl"},
            device_ids=[0],
        )
        self.assertEqual(scheduler._hccl_listen_ports(plan), [43439])

    def test_empty_device_ids_returns_no_ports(self):
        plan = _plan(device_ids=[])
        self.assertEqual(scheduler._hccl_listen_ports(plan), [])


class WaitHcclListenPortsReleasedTests(unittest.TestCase):
    def test_no_hccl_ports_does_not_probe(self):
        rt = _runtime(
            plan=_plan(device_ids=[]),
            error=RuntimeError("should not run"),
        )
        scheduler._wait_hccl_listen_ports_released(rt, timeout_s=30)

    def test_already_free_returns_without_sleep(self):
        rt = _runtime(readings=[""])
        clock = {"t": 0.0}

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_hccl_listen_ports_released(rt, timeout_s=30)
        self.assertEqual(clock["t"], 0.0)

    def test_waits_until_ports_free(self):
        rt = _runtime(readings=["43444", "43444", ""])
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_hccl_listen_ports_released(
                rt, timeout_s=30, interval_s=1
            )
        self.assertEqual(clock["t"], 2)

    def test_timeout_proceeds_anyway(self):
        rt = _runtime(readings=["43444"] * 100)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep), \
                mock.patch.object(scheduler.logging, "warning") as warn:
            scheduler._wait_hccl_listen_ports_released(
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
            scheduler._wait_hccl_listen_ports_released(rt, timeout_s=30)
        self.assertEqual(clock["t"], 0.0)

    def test_default_timeout_comes_from_env(self):
        rt = _runtime(readings=["43444"] * 100)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.dict(os.environ, {"XTASK_HCCL_PORT_RELEASE_TIMEOUT": "3"}), \
                mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_hccl_listen_ports_released(rt, interval_s=1)
        self.assertGreaterEqual(clock["t"], 3)
        self.assertLess(clock["t"], 30)


class WaitHcclListenPortsBeforeStartTests(unittest.TestCase):
    def test_non_multi_host_does_not_wait(self):
        with mock.patch(
            "xtask.core.service_group.run_on_runtimes_parallel"
        ) as parallel:
            scheduler._wait_hccl_listen_ports_before_start(
                {"multi_host": False}, [_runtime()]
            )
        parallel.assert_not_called()

    def test_multi_host_waits_on_each_runtime(self):
        runtimes = [_runtime(), _runtime()]
        with mock.patch(
            "xtask.core.service_group.run_on_runtimes_parallel"
        ) as parallel:
            scheduler._wait_hccl_listen_ports_before_start(
                {"multi_host": True}, runtimes
            )
        parallel.assert_called_once_with(
            runtimes, scheduler._wait_hccl_listen_ports_released
        )


if __name__ == "__main__":
    unittest.main()
