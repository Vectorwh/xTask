import socket
import json


def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def load_json_file(file_path) -> bool:
    try:
        with open(file_path, 'r') as f:
            json_file = json.load(f)
    except FileNotFoundError:
        print(f"Error: Template Config file not found at {file_path}")
        raise ValueError
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        raise ValueError
    return json_file
