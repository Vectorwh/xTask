#!/usr/bin/env bash
# Install AISBench + mini-swe-agent + SWE-bench harness on the x86 runner host.
# This is host-side only. It must not run inside the NPU/xLLM container.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_ROOT="${INSTALL_ROOT:-/export/home/xtask_swe/runtime}"
AISBENCH_DIR="${AISBENCH_DIR:-}"
MINI_SWE_DIR="${MINI_SWE_DIR:-}"
SWE_HARNESS_DIR="${SWE_HARNESS_DIR:-}"
MINI_SWE_GIT_URL="${MINI_SWE_GIT_URL:-https://github.com/AISBench/mini-swe-agent.git}"
SWE_HARNESS_GIT_URL="${SWE_HARNESS_GIT_URL:-https://github.com/SWE-bench/SWE-bench.git}"
CLONE_IF_MISSING="${CLONE_IF_MISSING:-1}"

has_directory_content() {
    local root="$1"
    [ -d "$root" ] && find "$root" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

python_importable() {
    "$PYTHON_BIN" -c "$1" >/dev/null 2>&1
}

install_editable() {
    local src="$1"
    local name="$2"
    if ! has_directory_content "$src"; then
        echo "Error: $name source is missing or empty: $src" >&2
        exit 1
    fi
    echo "Installing $name from $src"
    if ! "$PYTHON_BIN" -m pip install -e "$src" --use-pep517; then
        echo "Error: pip install -e failed for $name at $src" >&2
        exit 1
    fi
}

clone_if_needed() {
    local dest="$1"
    local url="$2"
    local name="$3"
    if has_directory_content "$dest"; then
        echo "$name already present at $dest"
        return 0
    fi
    if [ "$CLONE_IF_MISSING" != "1" ]; then
        echo "Error: $name is not installed and clone is disabled. Provide a local source directory." >&2
        exit 1
    fi
    echo "Cloning $name from $url into $dest"
    mkdir -p "$(dirname "$dest")"
    if ! git clone --depth 1 "$url" "$dest"; then
        echo "Error: git clone failed for $name ($url)." >&2
        echo "Put a local checkout on the control node and set benchmark.swebench.setup source fields." >&2
        exit 1
    fi
}

echo "===== xTask SWE-bench x86 runtime setup ====="
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is required on the x86 runner and cannot be installed by xTask." >&2
    exit 1
fi
if ! docker ps >/dev/null 2>&1; then
    echo "Error: docker ps failed on the x86 runner. Start Docker and retry." >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN is not available on the x86 runner." >&2
    exit 1
fi

mkdir -p "$INSTALL_ROOT"

if [ -z "$AISBENCH_DIR" ]; then
    echo "Error: AISBENCH_DIR is empty. xTask must sync AISBench source first." >&2
    exit 1
fi
if ! python_importable "from ais_bench.benchmark.tasks import SWEBenchInferTask, SWEBenchEvalTask"; then
    install_editable "$AISBENCH_DIR" "AISBench"
else
    echo "AISBench SWE tasks already importable; skip pip"
fi

if [ -z "$MINI_SWE_DIR" ]; then
    MINI_SWE_DIR="$INSTALL_ROOT/mini-swe-agent"
fi
if ! python_importable "import minisweagent"; then
    if ! has_directory_content "$MINI_SWE_DIR"; then
        clone_if_needed "$MINI_SWE_DIR" "$MINI_SWE_GIT_URL" "mini-swe-agent"
    fi
    install_editable "$MINI_SWE_DIR" "mini-swe-agent"
else
    echo "mini-swe-agent already importable; skip pip"
fi

if [ -z "$SWE_HARNESS_DIR" ]; then
    SWE_HARNESS_DIR="$INSTALL_ROOT/SWE-bench"
fi
if ! python_importable "import swebench"; then
    if ! has_directory_content "$SWE_HARNESS_DIR"; then
        clone_if_needed "$SWE_HARNESS_DIR" "$SWE_HARNESS_GIT_URL" "SWE-bench harness"
    fi
    install_editable "$SWE_HARNESS_DIR" "SWE-bench harness"
else
    echo "SWE-bench harness already importable; skip pip"
fi

AIS_BENCH_BIN="$("$PYTHON_BIN" -c "import shutil; print(shutil.which('ais_bench') or '')" || true)"
if [ -z "$AIS_BENCH_BIN" ]; then
    AIS_BENCH_BIN="ais_bench"
fi
mkdir -p "$INSTALL_ROOT"
printf '%s\n' "$AIS_BENCH_BIN" > "$INSTALL_ROOT/ais_bench_bin.txt"
echo "AIS_BENCH_BIN=$AIS_BENCH_BIN"
echo "===== xTask SWE-bench x86 runtime setup done ====="
