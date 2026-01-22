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


def check_port_available(port: int, host: str = 'localhost') -> bool:
    """
    检查指定端口是否可用（未被占用）
    
    参数:
        port: 要检查的端口号
        host: 要检查的主机地址，默认为'localhost'
    
    返回:
        bool: True表示端口可用，False表示已被占用
    """
    try:
        # 创建套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 尝试绑定到端口
        sock.bind((host, port))
        sock.close()
        return True
    except OSError as e:
        # 如果端口已被占用，会抛出 OSError (通常是 errno 98 或 10048)
        return False


def find_available_port(start_port: int = 8000, 
                       end_port: int = 9000, 
                       host: str = 'localhost') -> int:
    """
    在指定范围内查找可用的端口
    
    参数:
        start_port: 起始端口号
        end_port: 结束端口号
        host: 主机地址
    
    返回:
        int: 找到的可用端口号，如果没找到则返回None
    """
    for port in range(start_port, end_port + 1):
        if check_port_available(port, host):
            return port
    return None


def find_available_ports(count: int = 1, 
                        start_port: int = 8000, 
                        host: str = 'localhost') -> list:
    """
    查找多个可用的连续端口
    
    参数:
        count: 需要查找的端口数量
        start_port: 起始端口号
        host: 主机地址
    
    返回:
        list: 可用的端口列表
    """
    available_ports = []
    current_port = start_port
    
    while len(available_ports) < count and current_port <= 65535:
        if check_port_available(current_port, host):
            available_ports.append(current_port)
        current_port += 1
    
    return available_ports if len(available_ports) == count else []
