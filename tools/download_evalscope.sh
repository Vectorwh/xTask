#!/bin/bash
set -e

# evalscope 离线安装脚本（1.14.0）
# 依赖包目录: /xllm_test_060/evalscope_offline/ （127 个 wheel 的完整闭包）
SOURCE_DIR="/xllm_test_060/evalscope_offline"
TARGET_DIR="/workspace"

# 检查离线包目录
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 离线包目录 $SOURCE_DIR 不存在"
    exit 1
fi

if [ ! -f "$SOURCE_DIR/evalscope-1.14.0-py3-none-any.whl" ]; then
    echo "错误: $SOURCE_DIR 下缺少 evalscope-1.14.0-py3-none-any.whl"
    exit 1
fi

# 复制 wheel 到容器 /workspace 下临时目录（避免直接在挂载目录安装）
OFFLINE_TMP="$TARGET_DIR/evalscope_offline_tmp"
rm -rf "$OFFLINE_TMP"
mkdir -p "$OFFLINE_TMP"
cp -f "$SOURCE_DIR"/*.whl "$OFFLINE_TMP/"

# 纯离线安装，不访问 PyPI。
#
# 重要：这里不能用 `pip install --no-index --find-links=... 'evalscope[perf]==1.14.0'`
# 的完整解析安装——litellm 1.106.2 要求 openai<3.0.0，会把容器内预装的
# openai 3.20.0 降级，可能影响 aisbench 等其他评测链路（aisbench 依赖 openai）。
#
# 因此采用"缺什么补什么"的 --no-deps 策略：
#   1) 先卸载旧的 evalscope（若有），再 --no-deps 安装 evalscope 1.14.0 主包
#   2) 对闭包中的每个依赖 wheel，仅当环境中"未安装同名包"时才安装；
#      已安装（无论版本）一律跳过，保证 openai/aiohttp/pandas/requests 等
#      共享库版本绝对不动，aisbench 等链路不受影响。
pip uninstall -y evalscope >/dev/null 2>&1 || true
pip install --no-index --no-deps "$OFFLINE_TMP/evalscope-1.14.0-py3-none-any.whl"

# 从 wheel 文件名解析发行名（将下划线归一化为连字符）
whl_dist_name() {
    basename "$1" | cut -d- -f1 | tr '_' '-'
}

for whl in "$OFFLINE_TMP"/*.whl; do
    [ -e "$whl" ] || continue
    dist="$(whl_dist_name "$whl")"
    # evalscope 主包上面已装，跳过
    [ "$dist" = "evalscope" ] && continue
    if pip show "$dist" >/dev/null 2>&1; then
        # 已安装（任意版本）：跳过，避免抬升/降级容器内共享库
        echo "skip $dist (already installed)"
    else
        pip install --no-index --no-deps "$whl"
    fi
done

# 清理临时目录
rm -rf "$OFFLINE_TMP"

# 验证安装
evalscope --version

echo "安装完成！"
