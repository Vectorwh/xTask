import os, json
from xtask.utils import get_host_ip


def generate_ranktable(num_prefill: int, 
                       num_decode: int, 
                       prefill_tp: int, 
                       decode_tp: int, 
                       init_device_id: int, 
                       device_info: list, 
                       init_port: int,
                       server_ip: str = None,
                       ranktable_path: str = None,
                       is_A3: bool = True):
    if server_ip is None:
        server_ip = get_host_ip()
    server_list = []
    server_group_list = []
    port_offset = 0
    rank_offset = 0
    for i in range(num_prefill+num_decode):
        # Set Port
        predict_port = init_port + port_offset
        mgmt_port = init_port + port_offset + 1
        metric_port = init_port + port_offset + 2
        inter_comm_port = init_port + port_offset + 3
        port_offset += 4
        
        # Set Device
        device = []
        
        tp_range = prefill_tp if i < num_prefill else decode_tp
        for j in range(tp_range):
            cur_rank_id = init_device_id + rank_offset
            device_json = {
                "device_id": str(cur_rank_id),
                "device_ip": str(device_info[cur_rank_id]['ip']),
                "rank_id": str(rank_offset),
                "device_logical_id": str(cur_rank_id)
            }
            if is_A3:
                device_json["super_device_id"] = str(device_info[cur_rank_id]['sdid'])
            rank_offset += 1
            device.append(device_json)
            
        server_list.append(
            {
                "server_id": "127.0.0.1",
                "server_ip": "11.87.48.243",
                "predict_port": str(predict_port),
                "mgmt_port": str(mgmt_port),
                "metric_port": str(metric_port),
                "inter_comm_port": str(inter_comm_port),
                "device": device
            }
        )

    group_id_2_dict = {
        "group_id": "2",
        "server_count": num_prefill + num_decode,
        "server_list": server_list
    }
    
    if is_A3:
        group_id_2_dict['super_pod_list'] = [
            {
                "super_pod_id": "1",
                "server_list": [
                    {
                        "server_id": "127.0.0.1"
                    }
                ]
            }
        ]
    server_group_list.append(group_id_2_dict)

    server_group_list.append(
        {
            "group_id": "1",
            "server_count": 1,
            "server_list": [
                {
                    "server_ip": "11.87.48.243"
                }
            ]
        }
    )
    
    server_group_list.append(
        {
            "group_id": "0",
            "server_count": 1,
            "server_list": [
                {
                    "server_ip": server_ip
                }
            ]
        }
    )
    
    global_rank_table = {
        "version": "1.0",
        "server_group_list": server_group_list
    }
    
    with open(ranktable_path, "w", encoding="utf-8") as f:
        json.dump(global_rank_table, f, indent=4, ensure_ascii=False)

    os.chmod(ranktable_path, 0o640)
    print(f"RANKTABLE JSON 文件已生成：{ranktable_path}")

    
if __name__ == "__main__":
    num_prefill = 2
    num_decode = 3
    prefill_tp = 2
    decode_tp = 2
    init_device_id = 0
    init_port = 2000

    # server_ip = "141.61.57.125"
    server_ip = get_host_ip()
    
    device_info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'device_info.json')
    
    with open(device_info_path, "r", encoding="utf-8") as f:
        device_info = json.load(f)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'global_rank_table_file.json')
    
    generate_ranktable(
        num_prefill = num_prefill,
        num_decode = num_decode,
        prefill_tp = prefill_tp,
        decode_tp = decode_tp,
        init_device_id = init_device_id,
        device_info = device_info,
        init_port = init_port,
        server_ip = server_ip,
        ranktable_path = output_path
    )