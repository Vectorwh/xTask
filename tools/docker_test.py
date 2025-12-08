import json
import subprocess


def run_command(command):
    """运行命令并返回输出和错误"""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output.decode('utf-8'), error.decode('utf-8')


def check_image_exists(image):
    """检查镜像是否存在"""
    output, error = run_command(f"docker images -q {image}")
    return output.strip() != ""


def download_image(image):
    """下载镜像"""
    print(f"Downloading image: {image}")
    output, error = run_command(f"docker pull {image}")
    if error:
        print(f"Error downloading image: {error}")
        return False
    return True


def create_container(image, container_name, cmd):
    """创建并运行容器"""
    docker_cmd = f"docker run -dit --ipc=host -u 0 --name {container_name} -v /home:/home {image} /bin/sh -c 'sleep infinity'"
    print(f"Creating container: {container_name}, cmd is: {docker_cmd}")
    output, error = run_command(f"{docker_cmd}")
    if error:
        print(f"Error creating container: {error}")
        return False
    return True


def execute_command_in_container(container_name, cmd):
    """在容器中执行命令"""
    print(f"Executing command in container: {cmd}")
    output, error = run_command(
        f"docker exec {container_name} /bin/sh -c '{cmd}'")
    if error:
        print(f"Error executing command: {error}")
        return False
    print(f"Command output: {output}")
    return True


def remove_container(container_name):
    """删除容器"""
    print(f"Removing container: {container_name}")
    output, error = run_command(f"docker rm -f {container_name}")
    if error:
        print(f"Error removing container: {error}")
        return False
    return True


def main():
    # 读取配置文件
    with open('config.json', 'r') as file:
        config = json.load(file)

    image = config.get('image')
    container_name = config.get('container_name')
    cmd = config.get('cmd')

    if not image or not container_name or not cmd:
        print("Invalid configuration file")
        return

    # 检查镜像是否存在
    if not check_image_exists(image):
        if not download_image(image):
            print("Failed to download image")
            return

    # 创建容器
    if not create_container(image, container_name, cmd):
        print("Failed to create container")
        return

    # 在容器中执行命令
    if not execute_command_in_container(container_name, cmd):
        print("Failed to execute command in container")
        return

    # 删除容器
    if not remove_container(container_name):
        print("Failed to remove container")
        return

    print("All operations completed successfully")


if __name__ == "__main__":
    main()
