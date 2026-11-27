#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/.tools"
MICROMAMBA="${TOOLS_DIR}/micromamba"
SIM_PREFIX="${REPO_ROOT}/.envs/simulation"
PERCEPTION_PREFIX="${REPO_ROOT}/.venvs/perception"
CONTROLLER_DIR="${REPO_ROOT}/.deps/go2-convex-mpc"
CONTROLLER_COMMIT="1c63c6a762779887ab0431fd60db681dede6cb32"
CONTROLLER_URL="https://github.com/elijah-waichong-chan/go2-convex-mpc.git"
PATCH_PATH="${REPO_ROOT}/controller/patches/simulation-hardening.patch"
PATCH_SHA256="8b14f03e0a95d0a21e9699fc77b4629396fc906e63e07d860752cb3df413e364"
MICROMAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-osx-arm64"
MICROMAMBA_SHA256="ec2a072f028e1a7cf20f3e2e74d5a8127cf5a5f27636375b5359811565f4e5be"
HF_REVISION="a2bb814dd30d776dcf7e30523b00659f4f141c71"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "bootstrap_macos.sh requires arm64 macOS" >&2
  exit 1
fi

mkdir -p "${TOOLS_DIR}" "${REPO_ROOT}/.deps" "${REPO_ROOT}/.envs" "${REPO_ROOT}/.venvs" "${REPO_ROOT}/.cache/downloads"

verify_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: expected ${expected}, got ${actual}" >&2
    exit 1
  fi
}

if [[ ! -x "${MICROMAMBA}" ]]; then
  curl --fail --location --retry 3 "${MICROMAMBA_URL}" --output "${MICROMAMBA}.download"
  verify_sha256 "${MICROMAMBA_SHA256}" "${MICROMAMBA}.download"
  mv "${MICROMAMBA}.download" "${MICROMAMBA}"
  chmod 0755 "${MICROMAMBA}"
else
  verify_sha256 "${MICROMAMBA_SHA256}" "${MICROMAMBA}"
fi

verify_sha256 "${PATCH_SHA256}" "${PATCH_PATH}"
if [[ ! -d "${CONTROLLER_DIR}/.git" ]]; then
  git clone "${CONTROLLER_URL}" "${CONTROLLER_DIR}"
fi
if [[ "$(git -C "${CONTROLLER_DIR}" remote get-url origin)" != "${CONTROLLER_URL}" ]]; then
  echo "controller origin mismatch" >&2
  exit 1
fi
if [[ "$(git -C "${CONTROLLER_DIR}" rev-parse HEAD)" != "${CONTROLLER_COMMIT}" ]]; then
  echo "controller commit mismatch" >&2
  exit 1
fi
if git -C "${CONTROLLER_DIR}" apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
  echo "controller patch already applied"
elif [[ -n "$(git -C "${CONTROLLER_DIR}" status --porcelain)" ]]; then
  echo "controller dependency has unexpected changes; refusing to overwrite it" >&2
  exit 1
else
  git -C "${CONTROLLER_DIR}" apply --check "${PATCH_PATH}"
  git -C "${CONTROLLER_DIR}" apply "${PATCH_PATH}"
fi

"${MICROMAMBA}" create --yes --prefix "${SIM_PREFIX}" --file "${REPO_ROOT}/environments/simulation.macos-arm64.yml"
"${MICROMAMBA}" run --prefix "${SIM_PREFIX}" python -m pip install --no-deps --editable "${CONTROLLER_DIR}"
"${MICROMAMBA}" run --prefix "${SIM_PREFIX}" python -m pip install --no-deps --editable "${REPO_ROOT}"

download_ycb() {
  local object_id="$1"
  local url="$2"
  local archive_sha="$3"
  local mesh_sha="$4"
  local archive="${REPO_ROOT}/.cache/downloads/${object_id}_google_16k.tgz"
  local destination="${REPO_ROOT}/.cache/assets/ycb"
  if [[ ! -f "${archive}" ]]; then
    curl --fail --location --retry 3 "${url}" --output "${archive}.download"
    verify_sha256 "${archive_sha}" "${archive}.download"
    mv "${archive}.download" "${archive}"
  else
    verify_sha256 "${archive_sha}" "${archive}"
  fi
  mkdir -p "${destination}/${object_id}/google_16k"
  tar -xzf "${archive}" -C "${destination}" \
    "${object_id}/google_16k/textured.obj" \
    "${object_id}/google_16k/textured.mtl" \
    "${object_id}/google_16k/texture_map.png"
  verify_sha256 "${mesh_sha}" "${destination}/${object_id}/google_16k/textured.obj"
}

download_ycb \
  "054_softball" \
  "https://ycb-benchmarks.s3.amazonaws.com/data/google/054_softball_google_16k.tgz" \
  "c8d87659ebc2f6fa7111c455cceb0e8c918a50bd55d1a5201202d19493ce85cc" \
  "53202bf47b68af01356e0795e32c8bd85d44e7853e55786f6b050c32c7e6c98a"
download_ycb \
  "055_baseball" \
  "https://ycb-benchmarks.s3.amazonaws.com/data/google/055_baseball_google_16k.tgz" \
  "46d0fcd1b8b10622b0adde9fe4eb064dcd37dd236af87fc095dfaf1d85419bd7" \
  "f9b719a13329aafdd5cd9e42717a9420af0b8e0d609670e0132529d4210c8696"

"${MICROMAMBA}" run --prefix "${SIM_PREFIX}" python -m unittest discover -s "${CONTROLLER_DIR}/tests" -v
"${MICROMAMBA}" run --prefix "${SIM_PREFIX}" go2-ovd validate-controller --repeats 1
"${MICROMAMBA}" run --prefix "${SIM_PREFIX}" go2-ovd validate-camera
"${MICROMAMBA}" run --prefix "${SIM_PREFIX}" go2-ovd capture \
  --config "${REPO_ROOT}/configs/episodes/smoke.yaml" \
  --output "${REPO_ROOT}/runs/bootstrap_smoke"

PYTHON311="$(command -v python3.11 || true)"
if [[ -z "${PYTHON311}" && -x /opt/homebrew/bin/python3.11 ]]; then
  PYTHON311=/opt/homebrew/bin/python3.11
fi
if [[ -z "${PYTHON311}" ]]; then
  echo "Python 3.11 is required for the perception environment" >&2
  exit 1
fi
if [[ ! -x "${PERCEPTION_PREFIX}/bin/python" ]]; then
  "${PYTHON311}" -m venv "${PERCEPTION_PREFIX}"
fi
"${PERCEPTION_PREFIX}/bin/python" -m pip install --upgrade pip setuptools wheel
"${PERCEPTION_PREFIX}/bin/python" -m pip install --require-hashes --only-binary=:all: -r "${REPO_ROOT}/requirements-perception.lock.txt"
"${PERCEPTION_PREFIX}/bin/python" -m pip install --no-deps --editable "${REPO_ROOT}"

HF_HOME="${REPO_ROOT}/.cache/huggingface" "${PERCEPTION_PREFIX}/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="IDEA-Research/grounding-dino-tiny",
    revision="${HF_REVISION}",
)
PY

HF_HOME="${REPO_ROOT}/.cache/huggingface" HF_HUB_OFFLINE=1 \
  "${PERCEPTION_PREFIX}/bin/go2-ovd" detect \
  --config "${REPO_ROOT}/configs/detectors/grounding_dino_tiny.yaml" \
  --frames "${REPO_ROOT}/runs/bootstrap_smoke" \
  --device cpu
cp "${REPO_ROOT}/runs/bootstrap_smoke/detections/grounding_dino_tiny.jsonl" \
  "${REPO_ROOT}/runs/bootstrap_smoke/detections/grounding_dino_tiny_cpu_smoke.jsonl"

HF_HOME="${REPO_ROOT}/.cache/huggingface" HF_HUB_OFFLINE=1 \
  "${PERCEPTION_PREFIX}/bin/go2-ovd" detect \
  --config "${REPO_ROOT}/configs/detectors/grounding_dino_tiny.yaml" \
  --frames "${REPO_ROOT}/runs/bootstrap_smoke" \
  --device mps
cp "${REPO_ROOT}/runs/bootstrap_smoke/detections/grounding_dino_tiny.jsonl" \
  "${REPO_ROOT}/runs/bootstrap_smoke/detections/grounding_dino_tiny_mps_smoke.jsonl"

echo "Bootstrap complete. No physical robot or ROS 2 command was invoked."
