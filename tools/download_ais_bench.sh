#!/bin/bash
# ais_bench编译与部署
#cd /workspace
SOURCE_DIR="/xllm_test_060/benchmark"
TARGET_DIR="${AISBENCH_TARGET_DIR:-/workspace/benchmark}"

has_directory_content() {
    local root="$1"
    [ -d "$root" ] && find "$root" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

pip uninstall ais-bench-benchmark -y
#git clone https://github.com/AISBench/benchmark.git
if [ $? -ne 0 ]; then
    echo "Error: Failed to uninstall ais_bench"
    exit 1
fi

if ! has_directory_content "$SOURCE_DIR"; then
    echo "Error: AISBench source directory is missing or empty: $SOURCE_DIR"
    exit 1
fi

if [ -z "$TARGET_DIR" ] || [ "$TARGET_DIR" = "/" ]; then
    echo "Error: Invalid AISBench target directory: $TARGET_DIR"
    exit 1
fi

echo "Synchronizing AISBench directory: $SOURCE_DIR -> $TARGET_DIR"
mkdir -p "$(dirname "$TARGET_DIR")"
if command -v rsync >/dev/null 2>&1; then
    if ! rsync -a --delete "$SOURCE_DIR/" "$TARGET_DIR/"; then
        echo "Error: Failed to synchronize AISBench directory: $TARGET_DIR"
        exit 1
    fi
else
    echo "Warning: rsync is unavailable, replacing AISBench target directory"
    rm -rf "$TARGET_DIR"
    if cp --help 2>/dev/null | grep -q -- '--reflink'; then
        cp -a --reflink=auto "$SOURCE_DIR" "$TARGET_DIR"
    else
        cp -a "$SOURCE_DIR" "$TARGET_DIR"
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy AISBench directory: $TARGET_DIR"
        exit 1
    fi
fi

if ! has_directory_content "$TARGET_DIR"; then
    echo "Error: AISBench target directory is missing or empty: $TARGET_DIR"
    exit 1
fi

cd "$TARGET_DIR/"
pip3 install -e ./ --use-pep517
if [ $? -ne 0 ]; then
    echo "Error: Failed to install ais_bench Python package"
    exit 1
fi

cd /workspace
echo "Installation ais_bench successfully!"
