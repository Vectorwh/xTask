import json
import logging
import sys
import time
import uuid
from pathlib import Path

import requests

# Support both `python -m xtask.utils.openme` and direct file execution.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xtask.core.model_identity import resolve_model_name


class OpenMeClient:
    """OpenMe Message Sender"""

    # Keep in sync with xtask.core.result_processor.CASE_GROUP_WORKBOOK_PATH_ATTR
    CASE_GROUP_WORKBOOK_PATH_ATTR = "_xtask_case_group_result_path"
    OUTPUTS_ROOT = _REPO_ROOT / "outputs"

    OPENME_CONFIG = {
        "AppId": "lw5xdL5HsIHbXdssEMaV",  # 算法CICD应用ID
        "ERP": "kongshuaikang1",  # 消息发送人
        "TenantId": "CN.JD.GROUP",  # 租户，国内
        "GroupId": "10231269663",  # 发送京me群ID
        "RobotId": "00_df87bfb3cfd340ec",  # 机器人ID
        "TemplateCardId": "sFJl6O7bjk2DYO4a863nk",  # 卡片ID
        "TemplateCardVersion": "0.0.5",  # 卡片版本
    }

    URL_CONFIG = {
        "AppAccessTokenUrl": "http://openme.jd.local/open-api/auth/v1/app_access_token",
        "TeamAccessTokenUrl": "http://openme.jd.local/open-api/auth/v1/team_access_token",
        "JUEMsgUrl": "http://openme.jd.local/open-api/suite/v1/timline/sendJUEMsg",  # 互动卡片URL
    }

    def __init__(self, version, device, host, model_owners_path=None):
        self.appAccessToken = ""  # app token
        self.appExpireIn = 0  # 有效期
        self.teamAccessToken = ""  # team token
        self.teamExpireIn = 0  # 有效期

        self.host = host
        self.all_case_num = 0
        self.fail_case_num = 0
        # model_name -> {"success": int, "failed": int}
        self.model_stats = {}
        self.model_summary = ""
        self._owner_index = self._load_owner_index(model_owners_path)

        timestamp = time.time()
        local_time = time.localtime(timestamp)
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

        self.dataMap = {
            "appId": OpenMeClient.OPENME_CONFIG["AppId"],
            "erp": OpenMeClient.OPENME_CONFIG["ERP"],
            "tenantId": OpenMeClient.OPENME_CONFIG["TenantId"],
            "requestId": "",
            "dateTime": "",
            "groupId": OpenMeClient.OPENME_CONFIG["GroupId"],
            "params": {
                "robotId": OpenMeClient.OPENME_CONFIG["RobotId"],
                "data": {
                    "cardData": {
                        "templateCardId": OpenMeClient.OPENME_CONFIG[
                            "TemplateCardId"
                        ],
                        "templateCardVersion": OpenMeClient.OPENME_CONFIG[
                            "TemplateCardVersion"
                        ],
                        "templateCardVariable": {
                            "version": version,
                            "device": device,
                            "host": host,
                            "execution_time": formatted_time,
                            "all_case_num": "0",
                            "fail_case_num": "0",
                            "case_details": [],
                            "case_framework_log": (
                                f"[测试框架执行日志](http://{host}/case_framework.log)"
                            ),
                            "model_summary": "",
                        },
                    },
                    "forward": {
                        "reload": False,
                    },
                    "reload": False,
                },
            },
        }

    @staticmethod
    def _default_model_owners_path():
        return (
            Path(__file__).resolve().parents[2] / "config" / "model_owners.json"
        )

    @classmethod
    def _load_owner_index(cls, model_owners_path=None):
        path = Path(model_owners_path or cls._default_model_owners_path())
        index = {}
        if not path.is_file():
            logging.warning("model owners file not found: %s", path)
            return index
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Failed to load model owners from %s", path)
            return index

        for group in payload.values():
            owners = []
            for owner in group.get("owner") or []:
                if (
                    isinstance(owner, (list, tuple))
                    and len(owner) >= 2
                    and owner[0]
                    and owner[1]
                ):
                    owners.append((str(owner[0]), str(owner[1])))
            if not owners:
                continue
            for model_name in group.get("model_name") or []:
                key = str(model_name).strip()
                if key:
                    index[key] = list(owners)
        return index

    def _owners_for_model(self, model_name):
        key = str(model_name or "").strip()
        if key in self._owner_index:
            return list(self._owner_index[key])
        lowered = key.lower()
        for candidate, owners in self._owner_index.items():
            if candidate.lower() == lowered:
                return list(owners)
        return []

    @staticmethod
    def _format_owner_mentions(owners):
        mentions = [
            f"[@{name}]({erp}/ee)" for name, erp in owners if name and erp
        ]
        return "\n".join(mentions)

    @staticmethod
    def _format_fail_count_md(fail_count):
        """Format overview fail count for me_md substitution.

        JingME me_md expects bold wrapping font:
        **<font color='red'>N</font>**
        """
        fail_count = int(fail_count or 0)
        text = str(fail_count)
        if fail_count:
            return f"**<font color='red'>{text}</font>**"
        return text

    @staticmethod
    def _format_exec_status(success_count, fail_count):
        """Format table exec_status as '{total} / {failed}' me_md text."""
        success_count = int(success_count or 0)
        fail_count = int(fail_count or 0)
        total = success_count + fail_count
        if fail_count:
            fail_text = f"**<font color='red'>{fail_count}</font>**"
        else:
            fail_text = str(fail_count)
        return f"{total} / {fail_text}"

    def _build_case_details(self):
        details = []
        for model_name in sorted(self.model_stats):
            stats = self.model_stats[model_name]
            owners = self._owners_for_model(model_name)
            if stats.get("failed", 0) > 0:
                model_owner = self._format_owner_mentions(owners) or "NA"
            else:
                model_owner = "NA"
            details.append(
                {
                    "model_name": model_name,
                    "exec_status": self._format_exec_status(
                        stats["success"], stats["failed"]
                    ),
                    "model_owner": model_owner,
                }
            )
        return details

    def _build_at_users(self):
        users = []
        seen = set()
        for model_name, stats in self.model_stats.items():
            if stats.get("failed", 0) <= 0:
                continue
            for _name, erp in self._owners_for_model(model_name):
                pin = str(erp).strip()
                if not pin or pin in seen:
                    continue
                seen.add(pin)
                users.append({"app": "ee", "pin": pin})
        return users

    @staticmethod
    def _report_path_under_outputs(workbook_path):
        path = Path(workbook_path)
        outputs_root = (Path.cwd() / "outputs").resolve()
        try:
            return path.resolve().relative_to(outputs_root).as_posix()
        except Exception:
            parts = path.parts
            try:
                outputs_index = len(parts) - 1 - tuple(reversed(parts)).index(
                    "outputs"
                )
            except ValueError:
                return path.as_posix().lstrip("/")
            return Path(*parts[outputs_index + 1 :]).as_posix()

    def _case_framework_log_markdown(self):
        outputs_root = self.OUTPUTS_ROOT.resolve()
        log_alias = outputs_root / "case_framework.log"
        try:
            log_path = log_alias.resolve(strict=True)
            relative_path = log_path.relative_to(outputs_root).as_posix()
        except (OSError, ValueError):
            relative_path = log_alias.name
        return (
            f"[测试框架执行日志](http://{self.host}/{relative_path})"
        )

    def configure_case_group_report(self, args=None):
        """Fill model_summary only for --case_group runs with a workbook path."""
        self.model_summary = ""
        if args is None or not getattr(args, "case_group", None):
            return

        workbook_path = getattr(
            args, self.CASE_GROUP_WORKBOOK_PATH_ATTR, None
        )
        if not workbook_path:
            logging.warning(
                "case_group=%s but no workbook path found for OpenMe report link",
                getattr(args, "case_group", None),
            )
            return

        report_path = self._report_path_under_outputs(workbook_path)
        self.model_summary = (
            f"[测试报表链接](http://{self.host}/{report_path})"
        )

    def SendJUEMsgCard(self, args=None):
        """exec send action"""
        if args is not None:
            self.configure_case_group_report(args)

        variables = self.dataMap["params"]["data"]["cardData"][
            "templateCardVariable"
        ]
        self.dataMap["requestId"] = str(uuid.uuid4())
        self.dataMap["dateTime"] = int(time.time())
        variables["all_case_num"] = str(self.all_case_num)
        variables["fail_case_num"] = self._format_fail_count_md(
            self.fail_case_num
        )
        variables["case_details"] = self._build_case_details()
        variables["case_framework_log"] = (
            self._case_framework_log_markdown()
        )
        variables["model_summary"] = self.model_summary or ""

        payload_data = self.dataMap["params"]["data"]
        at_users = self._build_at_users()
        if at_users:
            payload_data["at"] = {"atAll": False, "users": at_users}
        else:
            payload_data.pop("at", None)

        self._validate_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.teamAccessToken,
        }

        response = requests.post(
            OpenMeClient.URL_CONFIG["JUEMsgUrl"],
            headers=headers,
            data=json.dumps(self.dataMap),
        )

        logging.info(response.json())

    def AddCaseResult(self, result, task_path=None):
        """Accumulate one case into per-model summary for the card template."""
        result = result or {}
        task_config = result.get("task_config") or {}
        if not task_path:
            metadata = task_config.get("_xtask") or {}
            task_path = metadata.get("source_case_file")

        model_name = resolve_model_name(task_path, task_config)
        success = bool(result.get("success"))
        stats = self.model_stats.setdefault(
            model_name, {"success": 0, "failed": 0}
        )

        self.all_case_num += 1
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            self.fail_case_num += 1

    def _validate_token(self):
        # 获取当前时间戳（秒）
        timestamp = int(time.time())

        # 过期时间30天
        if timestamp - self.appExpireIn >= 30 * 24 * 3600:
            dataMap = {
                "appKey": "lw5xdL5HsIHbXdssEMaV",
                "appSecret": "fIE2YfLqIUCaDB3fIGYz",
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                OpenMeClient.URL_CONFIG["AppAccessTokenUrl"],
                headers=headers,
                data=json.dumps(dataMap),
            )

            logging.info(response.json())

            self.appAccessToken = response.json()["data"]["appAccessToken"]
            self.appExpireIn = response.json()["data"]["expireIn"]

        # 过期时间2小时
        if timestamp - self.teamExpireIn >= 2 * 3600:
            dataMap = {
                "appAccessToken": self.appAccessToken,
                "openTeamId": "ce9f90ee26807da7661a6743ff20f426",
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                OpenMeClient.URL_CONFIG["TeamAccessTokenUrl"],
                headers=headers,
                data=json.dumps(dataMap),
            )

            logging.info(response.json())

            self.teamAccessToken = response.json()["data"]["teamAccessToken"]
            self.teamExpireIn = response.json()["data"]["expireIn"]


if __name__ == "__main__":
    openme_client = OpenMeClient("v0.8.0", "A2", "11.87.191.101")
    openme_client.SendJUEMsgCard()
