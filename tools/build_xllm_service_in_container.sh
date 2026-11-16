#!/bin/bash
# xLLM-Service 容器内编译与版本化归档。
set -euo pipefail

# CMake/vcpkg/Cargo may still need network access inside the build container.
export http_proxy="${http_proxy:-http://bamboo-proxy.jd.com:80}"
export https_proxy="${https_proxy:-http://bamboo-proxy.jd.com:80}"
export no_proxy="${no_proxy:-.jd.com,.jd.local,.local}"

if [[ -z "${WORK_ROOT:-}" ]]; then
  echo "ERROR: WORK_ROOT is not set. Use prepare_xllm_service.sh." >&2
  exit 1
fi

SERVICE_SOURCE_DIR="${XLLM_SERVICE_SOURCE_DIR:-${WORK_ROOT}/xllm-service}"
XLLM_PKG_DIR="${WORK_ROOT}/pkg/xllm"
SERVICE_ARCHIVE_DIR="${XLLM_PKG_DIR}/service_binary"
BUILD_DIR="${SERVICE_SOURCE_DIR}/build"
BUILD_JOBS="${XLLM_SERVICE_BUILD_JOBS:-8}"
BUILD_TYPE="${XLLM_SERVICE_BUILD_TYPE:-Release}"
CXX_STANDARD="${XLLM_SERVICE_CXX_STANDARD:-20}"
BUILD_TESTING="${XLLM_SERVICE_BUILD_TESTING:-1}"
case "$BUILD_TESTING" in
  1) CMAKE_TESTING=ON ;;
  0) CMAKE_TESTING=OFF ;;
  *) echo "ERROR: XLLM_SERVICE_BUILD_TESTING must be 0 or 1." >&2; exit 1 ;;
esac

restore_host_ownership() {
  if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
    [[ -d "${BUILD_DIR}" ]] && chown -R "${HOST_UID}:${HOST_GID}" "${BUILD_DIR}"
    [[ -d "${SERVICE_ARCHIVE_DIR}" ]] && \
      chown -R "${HOST_UID}:${HOST_GID}" "${SERVICE_ARCHIVE_DIR}"
  fi
}
trap restore_host_ownership EXIT

if [[ ! -f "${SERVICE_SOURCE_DIR}/CMakeLists.txt" ]]; then
  echo "ERROR: xLLM-Service source is incomplete: ${SERVICE_SOURCE_DIR}" >&2
  exit 1
fi

cd "${SERVICE_SOURCE_DIR}"
git config --global --add safe.directory '*' 2>/dev/null || true

# prepare.sh only applies the repository's required cpprest patch. Applying it
# repeatedly is not safe, so treat an already-applied patch as a valid state.
# Newer service versions (for example coding.jd.com main) already carry the
# static-link cpprest default and ship no prepare.sh at all, so it is optional.
if [[ ! -f "${SERVICE_SOURCE_DIR}/prepare.sh" ]]; then
  echo "xLLM-Service has no prepare.sh; skipping the cpprest patch step."
elif ! bash prepare.sh; then
  PATCH_FILE="${SERVICE_SOURCE_DIR}/third_party/custom_cache/cpprestsdk.patch"
  if [[ ! -f "${PATCH_FILE}" ]] || ! git -C "${SERVICE_SOURCE_DIR}/third_party/cpprestsdk" apply --reverse --check "../custom_cache/cpprestsdk.patch"; then
    echo "ERROR: xLLM-Service prepare.sh failed." >&2
    exit 1
  fi
  echo "xLLM-Service prepare.sh reported an already-applied/local cpprest patch; continuing."
fi

mkdir -p "${BUILD_DIR}"

# xLLM-Service's Ascend branch still adds legacy vcpkg paths such as
# /vcpkg-src/packages/boost-locale_arm64-linux. The current image uses
# manifest-mode vcpkg, whose installed tree is under build/vcpkg_installed.
# Provide ephemeral compatibility links inside this disposable container so
# those include/link directories exist without modifying the Service source or
# the host's vcpkg cache.
#
# USE_ASCEND defaults to ON in the Service CMakeLists and CPPREST builds with
# -Werror=missing-include-dirs, so a missing link fails the build outright
# rather than degrading to a warning. The links are created before configure
# because a warm build tree already has vcpkg_installed; a fresh tree only
# produces it during configure, so the same helper runs again afterwards.
case "$(uname -m)" in
  aarch64) VCPKG_TRIPLET="${VCPKG_TARGET_TRIPLET:-arm64-linux}" ;;
  x86_64) VCPKG_TRIPLET="${VCPKG_TARGET_TRIPLET:-x64-linux}" ;;
  *) VCPKG_TRIPLET="${VCPKG_TARGET_TRIPLET:-}" ;;
esac
VCPKG_INSTALL_ROOT="${BUILD_DIR}/vcpkg_installed/${VCPKG_TRIPLET}"

ensure_vcpkg_compat_links() {
  if [[ -z "${VCPKG_ROOT:-}" || ! -d "${VCPKG_ROOT}" || -z "${VCPKG_TRIPLET}" ]]; then
    return 0
  fi
  mkdir -p "${VCPKG_ROOT}/packages"
  for package_name in boost-locale boost-interprocess zlib; do
    package_link="${VCPKG_ROOT}/packages/${package_name}_${VCPKG_TRIPLET}"
    # -e misses a dangling link, so -L has to be checked as well.
    if [[ ! -e "${package_link}" && ! -L "${package_link}" ]]; then
      ln -s "${VCPKG_INSTALL_ROOT}" "${package_link}"
    fi
  done
}

# Wheel-installed Torch is not in CMake's default prefix search. Resolve its
# config with the same Python used by setup.py, without importing torch/NPU or
# installing/replacing any package in the image. Explicit Torch_DIR takes
# precedence; keep the caller's remaining CMake prefixes intact.
ensure_torch_cmake_prefix() {
  local torch_dir="${Torch_DIR:-}"
  if [[ -z "$torch_dir" ]]; then
    if ! torch_dir="$(python3 -c 'import importlib.util,pathlib; spec=importlib.util.find_spec("torch"); print(pathlib.Path(spec.origin).resolve().parent / "share/cmake/Torch" if spec and spec.origin else "")')"; then
      echo "ERROR: failed to discover Torch CMake directory with python3." >&2
      return 1
    fi
  fi
  if [[ -z "$torch_dir" ]]; then
    echo "[service-build] Python Torch is absent; leave CMake discovery unchanged for legacy/no-Torch builds."
    return 0
  fi
  if [[ ! -f "$torch_dir/TorchConfig.cmake" ]]; then
    echo "ERROR: Torch CMake config is missing: $torch_dir/TorchConfig.cmake" >&2
    return 1
  fi
  export Torch_DIR="$torch_dir"
  export CMAKE_PREFIX_PATH="$torch_dir${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  echo "[service-build] Torch_DIR=$Torch_DIR"
}

ensure_vcpkg_compat_links
ensure_torch_cmake_prefix

# Newer Service versions (coding.jd.com main) build through setup.py, which
# checks every direct submodule against the indexed gitlink before invoking
# CMake and prefers Ninja over a stale Makefile cache. It compiles all targets,
# not just the service binary, and it always builds Release: CMakeLists hard
# codes CMAKE_CXX_STANDARD 20 and defaults CMAKE_BUILD_TYPE to Release, so the
# previous -DWERROR=OFF and CXX_STANDARD injection are gone (WERROR no longer
# exists upstream; cpprestsdk supplies its own -Werror). Older GitHub-based
# versions have no setup.py and keep the explicit CMake path, which is also the
# fallback whenever a non-default build type is requested.
if [[ -f "${SERVICE_SOURCE_DIR}/setup.py" && "${BUILD_TYPE}" == "Release" ]]; then
  echo "[service-build] python3 setup.py build (jobs=${BUILD_JOBS})"
  TESTING_ARGS=()
  if [[ "$BUILD_TESTING" == 0 ]]; then
    if ! python3 setup.py build --help | grep -q -- '--no-testing'; then
      echo "ERROR: this Service setup.py does not support --no-testing." >&2
      exit 1
    fi
    TESTING_ARGS+=(--no-testing)
  fi
  python3 setup.py build --jobs "${BUILD_JOBS}" --build-dir "${BUILD_DIR}" "${TESTING_ARGS[@]}"
  ensure_vcpkg_compat_links
else
  if [[ -f "${SERVICE_SOURCE_DIR}/setup.py" ]]; then
    echo "[service-build] build_type=${BUILD_TYPE} requested; using explicit CMake instead of setup.py."
  fi
  CMAKE_COMPAT_FILE="${BUILD_DIR}/xtask_service_compat.cmake"
  printf 'set(CMAKE_CXX_STANDARD "%s")\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n' \
    "${CXX_STANDARD}" > "${CMAKE_COMPAT_FILE}"
  cmake -S "${SERVICE_SOURCE_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_PROJECT_xllm_service_INCLUDE="${CMAKE_COMPAT_FILE}" \
    -DBUILD_TESTING="${CMAKE_TESTING}" \
    -DWERROR=OFF
  ensure_vcpkg_compat_links
  cmake --build "${BUILD_DIR}" --target xllm_master_serving --parallel "${BUILD_JOBS}"
fi

SERVICE_BINARY="${BUILD_DIR}/xllm_service/xllm_master_serving"
if [[ ! -x "${SERVICE_BINARY}" ]]; then
  echo "ERROR: xLLM-Service build did not produce executable: ${SERVICE_BINARY}" >&2
  exit 1
fi

# Inspect the executable, not a stale Cargo output directory. New tokenizer
# staticlib builds need no tokenizer .so; legacy DT_NEEDED contracts still do.
if ! DYNAMIC_INFO="$(readelf -d "${SERVICE_BINARY}" 2>&1)"; then
  echo "ERROR: cannot inspect Service ELF runtime dependencies." >&2
  exit 1
fi
TOKENIZER_LINKAGE=static
TOKENIZER_LIB=""
if grep -q 'Shared library: \[librust_tokenizers.so\]' <<< "$DYNAMIC_INFO"; then
  TOKENIZER_LINKAGE=dynamic
  TOKENIZER_LIB="$(find "${BUILD_DIR}/xllm_service/tokenizer" \
    -type f -path '*/release/librust_tokenizers.so' \
    ! -path '*/release/deps/*' -print -quit)"
  if [[ -z "$TOKENIZER_LIB" || ! -f "$TOKENIZER_LIB" ]]; then
    echo "ERROR: Service requires librust_tokenizers.so, but it was not produced." >&2
    exit 1
  fi
fi
# etcd/etcdctl are independent runtime tools, not Service build outputs.

SERVICE_COMMIT="unknown"
XLLM_SUBMODULE_COMMIT="unknown"
if git -C "${SERVICE_SOURCE_DIR}" rev-parse HEAD >/dev/null 2>&1; then
  SERVICE_COMMIT="$(git -C "${SERVICE_SOURCE_DIR}" rev-parse HEAD)"
fi
if git -C "${SERVICE_SOURCE_DIR}/third_party/xllm" rev-parse HEAD >/dev/null 2>&1; then
  XLLM_SUBMODULE_COMMIT="$(git -C "${SERVICE_SOURCE_DIR}/third_party/xllm" rev-parse HEAD)"
fi

mkdir -p "${SERVICE_ARCHIVE_DIR}/releases"
RELEASE_ID="${SERVICE_COMMIT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RELEASE_DIR="${SERVICE_ARCHIVE_DIR}/releases/${RELEASE_ID}"
STAGE_DIR="${SERVICE_ARCHIVE_DIR}/.release-${RELEASE_ID}.tmp"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"
install -m 0755 "${SERVICE_BINARY}" "${STAGE_DIR}/xllm_master_serving"
if [[ "$TOKENIZER_LINKAGE" == dynamic ]]; then
  install -m 0755 "${TOKENIZER_LIB}" "${STAGE_DIR}/librust_tokenizers.so"
fi
{
  printf 'service_commit=%s\n' "${SERVICE_COMMIT}"
  printf 'service_version=%s\n' "$(cat "${SERVICE_SOURCE_DIR}/version.txt" 2>/dev/null || echo unknown)"
  printf 'xllm_submodule_commit=%s\n' "${XLLM_SUBMODULE_COMMIT}"
  printf 'tokenizer_linkage=%s\n' "$TOKENIZER_LINKAGE"
  printf 'build_testing=%s\n' "$BUILD_TESTING"
  printf 'build_type=%s\n' "${BUILD_TYPE}"
  printf 'build_jobs=%s\n' "${BUILD_JOBS}"
  printf 'cxx_standard=%s\n' "${CXX_STANDARD}"
  printf 'build_image=%s\n' "${XLLM_SERVICE_BUILD_IMAGE:-unknown}"
  printf 'built_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${STAGE_DIR}/manifest.txt"

(cd "$STAGE_DIR" && sha256sum xllm_master_serving manifest.txt \
  ${TOKENIZER_LIB:+librust_tokenizers.so} > SHA256SUMS)

# Keep previous usable releases until the complete new release is staged, then
# atomically switch the current pointer. Legacy direct files are left alone so
# older workspaces/loaders remain usable.
mv "${STAGE_DIR}" "${RELEASE_DIR}"
CURRENT_LINK="${SERVICE_ARCHIVE_DIR}/current"
TMP_LINK="${SERVICE_ARCHIVE_DIR}/.current-${RELEASE_ID}.tmp"
ln -s "releases/${RELEASE_ID}" "${TMP_LINK}"
mv -Tf "${TMP_LINK}" "${CURRENT_LINK}"

echo "Archived xLLM-Service release: ${RELEASE_DIR}"
echo "Current xLLM-Service release: ${CURRENT_LINK}"

# Publish the complete optional pack after a successful Service build. This
# keeps the legacy independent artifacts usable and only updates the pack that
# cases may select explicitly via deployment.xllm_package.release_pack.
XLLM_SERVICE_APPEND_PACK="${XLLM_SERVICE_APPEND_PACK:-}"
XLLM_SERVICE_AUTO_APPEND_PACK="${XLLM_SERVICE_AUTO_APPEND_PACK:-1}"
case "$XLLM_SERVICE_AUTO_APPEND_PACK" in
  1|true|yes) XLLM_SERVICE_AUTO_APPEND_PACK=1 ;;
  0|false|no) XLLM_SERVICE_AUTO_APPEND_PACK=0 ;;
  *)
    echo "ERROR: XLLM_SERVICE_AUTO_APPEND_PACK must be 0/1 or true/false." >&2
    exit 1
    ;;
esac
RELEASE_PACK_CURRENT="${XLLM_PKG_DIR}/release_pack/current/xllm_pack.tar.gz"
if [[ -z "$XLLM_SERVICE_APPEND_PACK" &&
      "$XLLM_SERVICE_AUTO_APPEND_PACK" == 1 &&
      -f "$RELEASE_PACK_CURRENT" ]]; then
  XLLM_SERVICE_APPEND_PACK="$RELEASE_PACK_CURRENT"
  echo "[service-pack] Auto-appending Service to the current xLLM release pack."
fi
if [[ -n "$XLLM_SERVICE_APPEND_PACK" ]]; then
  if [[ ! -f "$XLLM_SERVICE_APPEND_PACK" ]]; then
    echo "ERROR: XLLM_SERVICE_APPEND_PACK does not exist: $XLLM_SERVICE_APPEND_PACK" >&2
    exit 1
  fi
  python3 "${WORK_ROOT}/tools/xllm_release_pack.py" publish \
    "$XLLM_SERVICE_APPEND_PACK" "${XLLM_PKG_DIR}/release_pack" --service "$RELEASE_DIR"
else
  echo "[service-pack] No complete xLLM release pack was selected; keep independent artifacts only."
fi
