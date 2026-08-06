#!/bin/bash
# ais_bench编译与部署
#cd /workspace
SOURCE_DIR="${AISBENCH_SOURCE_DIR:-/xllm_test_060/benchmark}"
TARGET_DIR="${AISBENCH_TARGET_DIR:-/workspace/benchmark}"
REQUIRED_FILES=(
    "setup.py"
    "requirements.txt"
    "ais_bench/benchmark/cli/main.py"
    "ais_bench/datasets/__init__.py"
)

has_required_files() {
    local root="$1"
    local relative_path
    for relative_path in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$root/$relative_path" ]; then
            return 1
        fi
    done
    return 0
}

pip uninstall ais-bench-benchmark -y
#git clone https://github.com/AISBench/benchmark.git
if [ $? -ne 0 ]; then
    echo "Error: Failed to uninstall ais_bench"
    exit 1
fi

if ! has_required_files "$SOURCE_DIR"; then
    echo "Error: AISBench source is incomplete: $SOURCE_DIR"
    exit 1
fi

if has_required_files "$TARGET_DIR"; then
    echo "AISBench source already exists and is complete, skip copy: $TARGET_DIR"
else
    echo "AISBench source is missing or incomplete, copying to: $TARGET_DIR"
    rm -rf "$TARGET_DIR"
    mkdir -p "$(dirname "$TARGET_DIR")"
    if cp --help 2>/dev/null | grep -q -- '--reflink'; then
        cp -a --reflink=auto "$SOURCE_DIR" "$TARGET_DIR"
    else
        cp -a "$SOURCE_DIR" "$TARGET_DIR"
    fi
    if ! has_required_files "$TARGET_DIR"; then
        echo "Error: AISBench copy verification failed: $TARGET_DIR"
        exit 1
    fi
fi

cd "$TARGET_DIR/"
pip3 install -e ./ --use-pep517
if [ $? -ne 0 ]; then
    echo "Error: Failed to install ais_bench Python package"
    exit 1
fi

cd /workspace
echo "Installation ais_bench successfully!"
