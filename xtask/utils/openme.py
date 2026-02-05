import os
import sys
import uuid
import logging
import json
import time
import requests


class OpenMeClient:
    """OpenMe Message Sender"""

    OPENME_CONFIG = {
            "AppId": "lw5xdL5HsIHbXdssEMaV",  # 算法CICD应用ID
            "ERP": "kongshuaikang1",          # 消息发送人
            "TenantId": "CN.JD.GROUP",        # 租户，国内
            "GroupId": "10224622479",         # 发送京me群ID
            "RobotId": "00_df87bfb3cfd340ec",   # 机器人ID
            "TemplateCardId": "YWYOs4nghsVy1VWPHngjS",  # 卡片ID
            "TemplateCardVersion": "0.0.3"              # 卡片版本
            }

    URL_CONFIG = {
            "AppAccessTokenUrl": "http://openme.jd.local/open-api/auth/v1/app_access_token",
            "TeamAccessTokenUrl": "http://openme.jd.local/open-api/auth/v1/team_access_token",
            "JUEMsgUrl": "http://openme.jd.local/open-api/suite/v1/timline/sendJUEMsg" # 互动卡片URL
            }

    def __init__(self, version, device, host):
        self.appAccessToken = ""       # app token
        self.appExpireIn = 0           # 有效期
        self.teamAccessToken = ""      # team token
        self.teamExpireIn = 0          # 有效期

        self.all_case_num = 0
        self.fail_case_num = 0
        # 获取当前时间戳
        timestamp = time.time()
        # 将时间戳转换为本地时间结构
        local_time = time.localtime(timestamp)
        # 格式化时间
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        
        self.dataMap = {
		    "appId":     OpenMeClient.OPENME_CONFIG["AppId"],
		    "erp":       OpenMeClient.OPENME_CONFIG["ERP"],
		    "tenantId":  OpenMeClient.OPENME_CONFIG["TenantId"],
		    "requestId": "",
		    "dateTime":  "",
		    "groupId":   OpenMeClient.OPENME_CONFIG["GroupId"],
		    "params": {
			    "robotId": OpenMeClient.OPENME_CONFIG["RobotId"],
			    "data": {
				    "cardData": {
					    "templateCardId":       OpenMeClient.OPENME_CONFIG["TemplateCardId"],
					    "templateCardVersion":  OpenMeClient.OPENME_CONFIG["TemplateCardVersion"],
					    "templateCardVariable": {
                            "version": version,
                            "device": device,
                            "host": host,
                            "execution_time": formatted_time,
                            "all_case_num": "0",
                            "fail_case_num": "0",
                            "case_details": []
                        }
				    },
				    "forward": {
					    "reload": False,
				    },
				   "reload": False
			    }
		    }
	    }

    def SendJUEMsgCard(self):
        """exec send action """
        self.dataMap["requestId"] = str(uuid.uuid4())
        self.dataMap["dateTime"] = int(time.time())
        self.dataMap["params"]["data"]["cardData"]["templateCardVariable"]["all_case_num"] = str(self.all_case_num)
        self.dataMap["params"]["data"]["cardData"]["templateCardVariable"]["fail_case_num"] = str(self.fail_case_num)

        self._validate_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': "Bearer " + self.teamAccessToken
        }

        response = requests.post(OpenMeClient.URL_CONFIG["JUEMsgUrl"], headers=headers, data=json.dumps(self.dataMap))
        
        # 打印结果
        logging.info(response.status_code)
        logging.info(response.json())

    def AddCaseResult(self, success, case_name, model, report):
        """case result handle """
        case_detail = {
            "case_name": case_name,
            "model": model.split("/")[-1],
            "report": "[summary](http://xllm-ci-hud.jd.com/" + case_name + "/" + case_name + ".json)"
        }

        self.all_case_num = self.all_case_num + 1
        if success:
            case_detail["status"] = [
                {
                    "color": "green",
                    "text": "成功"
                }
            ]
        else:
            self.fail_case_num = self.fail_case_num + 1
            case_detail["status"] = [
                {
                    "color": "red",
                    "text": "失败"
                }
            ]

        self.dataMap["params"]["data"]["cardData"]["templateCardVariable"]["case_details"].append(case_detail)

    def _validate_token(self):
        # 获取当前时间戳（秒）
        timestamp = int(time.time())

        # 过期时间30天
        if timestamp-self.appExpireIn >= 30*24*3600:
            dataMap = {
			    "appKey":    "lw5xdL5HsIHbXdssEMaV",
			    "appSecret": "fIE2YfLqIUCaDB3fIGYz"
		    }
            headers = {
                'Content-Type': 'application/json'
            }
            response = requests.post(OpenMeClient.URL_CONFIG["AppAccessTokenUrl"], headers=headers, data=json.dumps(dataMap))
        
            # 打印结果
            logging.info(response.status_code)
            logging.info(response.json())
            
            self.appAccessToken = response.json()["data"]["appAccessToken"]
            self.appExpireIn = response.json()["data"]["expireIn"]

        # 过期时间2小时
        if timestamp-self.teamExpireIn >= 2*3600:
            dataMap = {
		    	"appAccessToken": self.appAccessToken,
			    "openTeamId":     "ce9f90ee26807da7661a6743ff20f426"
		    }
            headers = {
                'Content-Type': 'application/json'
            }
            response = requests.post(OpenMeClient.URL_CONFIG["TeamAccessTokenUrl"], headers=headers, data=json.dumps(dataMap))
        
            # 打印结果
            logging.info(response.status_code)
            logging.info(response.json())

            self.teamAccessToken = response.json()["data"]["teamAccessToken"]
            self.teamExpireIn = response.json()["data"]["expireIn"]


if __name__ == "__main__":
    openme_client = OpenMeClient("v0.8.0", "A2", "11.87.191.101")
    openme_client.SendJUEMsgCard()
