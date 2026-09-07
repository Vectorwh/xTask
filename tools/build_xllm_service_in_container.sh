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

restore_host_ownership() {
  if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
    [[ -d "${BUILD_DIR}" ]] && chown -R "${HOST_UID}:${HOST_GID}" "${BUILD_DIR}"
    [[ -d "${SERVICE_ARCHIVE_DIR}" ]] && \
      chown -R "${HOST_UID}:${HOST_GID}" "${SERVICE_ARCHIVE_DIR}"
  fi
}
trap restore_host_ownership EXIT

if [[ ! -f "${SERVICE_SOURCE_DIR}/CMakeLists.txt" || ! -f "${SERVICE_SOURCE_DIR}/prepare.sh" ]]; then
  echo "ERROR: xLLM-Service source is incomplete: ${SERVICE_SOURCE_DIR}" >&2
  exit 1
fi

cd "${SERVICE_SOURCE_DIR}"
git config --global --add safe.directory '*' 2>/dev/null || true

# prepare.sh only applies the repository's required cpprest patch. Applying it
# repeatedly is not safe, so treat an already-applied patch as a valid state.
if ! bash prepare.sh; then
  PATCH_FILE="${SERVICE_SOURCE_DIR}/third_party/custom_cache/cpprestsdk.patch"
  if [[ ! -f "${PATCH_FILE}" ]] || ! git -C "${SERVICE_SOURCE_DIR}/third_party/cpprestsdk" apply --reverse --check "../custom_cache/cpprestsdk.patch"; then
    echo "ERROR: xLLM-Service prepare.sh failed." >&2
    exit 1
  fi
  echo "xLLM-Service prepare.sh reported an already-applied/local cpprest patch; continuing."
fi

mkdir -p "${BUILD_DIR}"
CMAKE_COMPAT_FILE="${BUILD_DIR}/xtask_service_compat.cmake"
printf 'set(CMAKE_CXX_STANDARD "%s")\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n' \
  "${CXX_STANDARD}" > "${CMAKE_COMPAT_FILE}"
cmake -S "${SERVICE_SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_PROJECT_xllm_service_INCLUDE="${CMAKE_COMPAT_FILE}" \
  -DBUILD_TESTING=ON \
  -DWERROR=OFF

# xLLM-Service's Ascend branch still adds legacy vcpkg paths such as
# /vcpkg-src/packages/boost-locale_arm64-linux. The current image uses
# manifest-mode vcpkg, whose installed tree is under build/vcpkg_installed.
# Provide ephemeral compatibility links inside this disposable container so
# those include/link directories exist without modifying the Service source or
# the host's vcpkg cache.
if [[ -n "${VCPKG_ROOT:-}" && -d "${VCPKG_ROOT}" ]]; then
  case "$(uname -m)" in
    aarch64) VCPKG_TRIPLET="${VCPKG_TARGET_TRIPLET:-arm64-linux}" ;;
    x86_64) VCPKG_TRIPLET="${VCPKG_TARGET_TRIPLET:-x64-linux}" ;;
    *) VCPKG_TRIPLET="${VCPKG_TARGET_TRIPLET:-}" ;;
  esac
  VCPKG_INSTALL_ROOT="${BUILD_DIR}/vcpkg_installed/${VCPKG_TRIPLET}"
  if [[ -n "${VCPKG_TRIPLET}" && -d "${VCPKG_INSTALL_ROOT}/include" ]]; then
    mkdir -p "${VCPKG_ROOT}/packages"
    for package_name in boost-locale boost-interprocess zlib; do
      package_link="${VCPKG_ROOT}/packages/${package_name}_${VCPKG_TRIPLET}"
      if [[ ! -e "${package_link}" && ! -L "${package_link}" ]]; then
        ln -s "${VCPKG_INSTALL_ROOT}" "${package_link}"
      fi
    done
  fi
fi

cmake --build "${BUILD_DIR}" --target xllm_master_serving --parallel "${BUILD_JOBS}"

SERVICE_BINARY="${BUILD_DIR}/xllm_service/xllm_master_serving"
if [[ ! -x "${SERVICE_BINARY}" ]]; then
  echo "ERROR: xLLM-Service build did not produce executable: ${SERVICE_BINARY}" >&2
  exit 1
fi

# The service executable has a runtime dependency on the Rust tokenizer shared
# object. It must travel with the service archive; etcd/etcdctl are independent
# runtime tools and are intentionally not built here.
TOKENIZER_LIB="$(
  find "${BUILD_DIR}/xllm_service/tokenizer" \
    -type f -path '*/release/librust_tokenizers.so' \
    ! -path '*/release/deps/*' -print -quit
)"
if [[ -z "${TOKENIZER_LIB:-}" || ! -f "${TOKENIZER_LIB}" ]]; then
  echo "ERROR: librust_tokenizers.so was not produced by xLLM-Service build." >&2
  exit 1
fi

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
install -m 0755 "${TOKENIZER_LIB}" "${STAGE_DIR}/librust_tokenizers.so"
{
  printf 'service_commit=%s\n' "${SERVICE_COMMIT}"
  printf 'service_version=%s\n' "$(cat "${SERVICE_SOURCE_DIR}/version.txt" 2>/dev/null || echo unknown)"
  printf 'xllm_submodule_commit=%s\n' "${XLLM_SUBMODULE_COMMIT}"
  printf 'build_type=%s\n' "${BUILD_TYPE}"
  printf 'build_jobs=%s\n' "${BUILD_JOBS}"
  printf 'cxx_standard=%s\n' "${CXX_STANDARD}"
  printf 'build_image=%s\n' "${XLLM_SERVICE_BUILD_IMAGE:-unknown}"
  printf 'built_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${STAGE_DIR}/manifest.txt"

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
