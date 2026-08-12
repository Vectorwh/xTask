import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.utils.openme import OpenMeClient


class OpenMeClientStatsTest(unittest.TestCase):
    def setUp(self):
        owners = {
            "glm": {
                "owner": [
                    ["沈运恒", "shenyunheng3"],
                    ["沈效龙", "ext.shenxiaolong5"],
                ],
                "model_name": ["glm_5", "glm_4.7"],
            },
            "Qwen3": {
                "owner": [["冯艳", "fengyan.119"]],
                "model_name": ["qwen3_8b"],
            },
        }
        self._tmpdir = tempfile.TemporaryDirectory()
        self.owners_path = Path(self._tmpdir.name) / "model_owners.json"
        self.owners_path.write_text(
            json.dumps(owners), encoding="utf-8"
        )
        self.client = OpenMeClient(
            "main",
            "A3-ARM",
            "10.0.0.1",
            model_owners_path=self.owners_path,
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_add_case_result_aggregates_by_yaml_parent_model(self):
        glm_yaml = Path(self._tmpdir.name) / "glm_5" / "case_a.yaml"
        glm_yaml.parent.mkdir(parents=True)
        glm_yaml.write_text("case_name: a\n", encoding="utf-8")
        qwen_yaml = Path(self._tmpdir.name) / "qwen3_8b" / "case_b.yaml"
        qwen_yaml.parent.mkdir(parents=True)
        qwen_yaml.write_text("case_name: b\n", encoding="utf-8")

        self.client.AddCaseResult(
            {"success": True, "task_config": {"case_name": "a"}},
            task_path=str(glm_yaml),
        )
        self.client.AddCaseResult(
            {"success": False, "task_config": {"case_name": "a2"}},
            task_path=str(glm_yaml),
        )
        self.client.AddCaseResult(
            {"success": True, "task_config": {"case_name": "b"}},
            task_path=str(qwen_yaml),
        )

        self.assertEqual(self.client.all_case_num, 3)
        self.assertEqual(self.client.fail_case_num, 1)
        self.assertEqual(
            self.client.model_stats,
            {
                "glm_5": {"success": 1, "failed": 1},
                "qwen3_8b": {"success": 1, "failed": 0},
            },
        )

    def test_send_builds_template_rows_and_ats_failed_owners(self):
        self.client.model_stats = {
            "glm_5": {"success": 2, "failed": 1},
            "qwen3_8b": {"success": 3, "failed": 0},
        }
        self.client.all_case_num = 6
        self.client.fail_case_num = 1
        self.client.teamAccessToken = "token"
        self.client.appExpireIn = int(1e12)
        self.client.teamExpireIn = int(1e12)

        with mock.patch.object(
            OpenMeClient, "_validate_token"
        ), mock.patch("xtask.utils.openme.requests.post") as post:
            post.return_value.json.return_value = {"code": 0}
            self.client.SendJUEMsgCard()

        payload = json.loads(post.call_args.kwargs["data"])
        variables = payload["params"]["data"]["cardData"]["templateCardVariable"]
        self.assertEqual(variables["all_case_num"], "6")
        self.assertEqual(
            variables["fail_case_num"], "**<font color='red'>1</font>**"
        )
        self.assertEqual(
            variables["case_details"],
            [
                {
                    "model_name": "glm_5",
                    "exec_status": "3 / **<font color='red'>1</font>**",
                    "model_owner": (
                        "[@沈运恒](shenyunheng3/ee)\n"
                        "[@沈效龙](ext.shenxiaolong5/ee)"
                    ),
                },
                {
                    "model_name": "qwen3_8b",
                    "exec_status": "3 / 0",
                    "model_owner": "NA",
                },
            ],
        )
        self.assertEqual(
            payload["params"]["data"]["at"],
            {
                "atAll": False,
                "users": [
                    {"app": "ee", "pin": "shenyunheng3"},
                    {"app": "ee", "pin": "ext.shenxiaolong5"},
                ],
            },
        )

    def test_send_uses_timestamped_case_framework_log_target(self):
        outputs_root = Path(self._tmpdir.name) / "outputs"
        log_dir = outputs_root / "log"
        log_dir.mkdir(parents=True)
        timestamped_log = log_dir / "case_framework_20260811_180035.log"
        timestamped_log.write_text("framework log\n", encoding="utf-8")
        (outputs_root / "case_framework.log").symlink_to(
            Path("log") / timestamped_log.name
        )
        self.client.teamAccessToken = "token"
        self.client.appExpireIn = int(1e12)
        self.client.teamExpireIn = int(1e12)

        with mock.patch.object(
            OpenMeClient, "OUTPUTS_ROOT", outputs_root
        ), mock.patch.object(
            OpenMeClient, "_validate_token"
        ), mock.patch("xtask.utils.openme.requests.post") as post:
            post.return_value.json.return_value = {"code": 0}
            self.client.SendJUEMsgCard()

        payload = json.loads(post.call_args.kwargs["data"])
        variables = payload["params"]["data"]["cardData"][
            "templateCardVariable"
        ]
        self.assertEqual(
            variables["case_framework_log"],
            (
                "[测试框架执行日志](http://10.0.0.1/log/"
                "case_framework_20260811_180035.log)"
            ),
        )

    def test_case_framework_log_falls_back_when_alias_is_missing(self):
        outputs_root = Path(self._tmpdir.name) / "missing_outputs"
        with mock.patch.object(
            OpenMeClient, "OUTPUTS_ROOT", outputs_root
        ):
            link = self.client._case_framework_log_markdown()

        self.assertEqual(
            link,
            "[测试框架执行日志](http://10.0.0.1/case_framework.log)",
        )

    def test_case_framework_log_falls_back_for_external_target(self):
        outputs_root = Path(self._tmpdir.name) / "outputs_external"
        outputs_root.mkdir()
        external_log = Path(self._tmpdir.name) / "external.log"
        external_log.write_text("external\n", encoding="utf-8")
        (outputs_root / "case_framework.log").symlink_to(external_log)

        with mock.patch.object(
            OpenMeClient, "OUTPUTS_ROOT", outputs_root
        ):
            link = self.client._case_framework_log_markdown()

        self.assertEqual(
            link,
            "[测试框架执行日志](http://10.0.0.1/case_framework.log)",
        )

    def test_send_omits_at_when_no_failures(self):
        self.client.model_stats = {
            "qwen3_8b": {"success": 2, "failed": 0},
        }
        self.client.all_case_num = 2
        self.client.fail_case_num = 0
        self.client.teamAccessToken = "token"
        self.client.appExpireIn = int(1e12)
        self.client.teamExpireIn = int(1e12)

        with mock.patch.object(
            OpenMeClient, "_validate_token"
        ), mock.patch("xtask.utils.openme.requests.post") as post:
            post.return_value.json.return_value = {"code": 0}
            self.client.SendJUEMsgCard()

        payload = json.loads(post.call_args.kwargs["data"])
        self.assertNotIn("at", payload["params"]["data"])
        variables = payload["params"]["data"]["cardData"]["templateCardVariable"]
        self.assertEqual(variables["fail_case_num"], "0")
        self.assertEqual(variables["model_summary"], "")

    def test_send_sets_model_summary_for_case_group_workbook(self):
        workbook = (
            Path.cwd()
            / "outputs"
            / "daily_group"
            / "daily_group_20260811_101112.xlsx"
        )
        args = SimpleNamespace(
            case_group="daily_group",
            **{
                OpenMeClient.CASE_GROUP_WORKBOOK_PATH_ATTR: str(workbook),
            },
        )
        self.client.teamAccessToken = "token"
        self.client.appExpireIn = int(1e12)
        self.client.teamExpireIn = int(1e12)

        with mock.patch.object(
            OpenMeClient, "_validate_token"
        ), mock.patch("xtask.utils.openme.requests.post") as post:
            post.return_value.json.return_value = {"code": 0}
            self.client.SendJUEMsgCard(args=args)

        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(
            payload["params"]["data"]["cardData"]["templateCardVariable"][
                "model_summary"
            ],
            (
                "[测试报表链接](http://10.0.0.1/"
                "daily_group/daily_group_20260811_101112.xlsx)"
            ),
        )

    def test_send_skips_model_summary_without_case_group(self):
        args = SimpleNamespace(case_group=None)
        self.client.teamAccessToken = "token"
        self.client.appExpireIn = int(1e12)
        self.client.teamExpireIn = int(1e12)

        with mock.patch.object(
            OpenMeClient, "_validate_token"
        ), mock.patch("xtask.utils.openme.requests.post") as post:
            post.return_value.json.return_value = {"code": 0}
            self.client.SendJUEMsgCard(args=args)

        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(
            payload["params"]["data"]["cardData"]["templateCardVariable"][
                "model_summary"
            ],
            "",
        )


if __name__ == "__main__":
    unittest.main()
