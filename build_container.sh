#!/bin/bash
set -euo pipefail

# Build an Apptainer SIF for Berzelius while keeping ALL build-time
# temporary files and OCI cache on the node-local /tmp filesystem.
# This avoids creating large numbers of temporary files on /proj.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${DEF_FILE:-berzelius-aiml.def}"
SIF_FILE="${SIF_FILE:-berzelius-aiml.sif}"

DEF_PATH="${SCRIPT_DIR}/${DEF_FILE}"
SIF_PATH="${SCRIPT_DIR}/${SIF_FILE}"

LOCAL_WORK="/tmp/${USER:-user}_apptainer_build_${SLURM_JOB_ID:-$$}"
LOCAL_TMP="${LOCAL_WORK}/tmp"
LOCAL_CACHE="${LOCAL_WORK}/cache"

mkdir -p "${LOCAL_TMP}" "${LOCAL_CACHE}"

export APPTAINER_TMPDIR="${LOCAL_TMP}"
export APPTAINER_CACHEDIR="${LOCAL_CACHE}"
export TMPDIR="${LOCAL_TMP}"

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    echo
    echo "=============================================="
    echo "Cleaning Apptainer temporary files and cache"
    echo "=============================================="
    echo "Local work directory: ${LOCAL_WORK}"

    # Cache is already inside LOCAL_WORK, but this is harmless and may
    # clean Apptainer-managed entries before removing the directory.
    apptainer cache clean --force >/dev/null 2>&1 || true
    rm -rf "${LOCAL_WORK}" || true

    echo "Cleanup finished."
    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

echo "=============================================="
echo "Apptainer build configuration"
echo "=============================================="
echo "Definition : ${DEF_PATH}"
echo "Output SIF : ${SIF_PATH}"
echo "Cache      : ${APPTAINER_CACHEDIR}"
echo "Build TMP  : ${APPTAINER_TMPDIR}"
echo "Host       : $(hostname)"
echo "=============================================="

command -v apptainer >/dev/null 2>&1 || {
    echo "ERROR: apptainer is not available in PATH."
    exit 1
}

if [[ ! -f "${DEF_PATH}" ]]; then
    echo "ERROR: Definition file not found: ${DEF_PATH}"
    exit 1
fi

echo
echo "Node-local /tmp disk space:"
df -h /tmp

if [[ -e "${SIF_PATH}" ]]; then
    if [[ "${OVERWRITE:-0}" == "1" ]]; then
        echo
echo "OVERWRITE=1: removing existing image: ${SIF_PATH}"
        rm -f "${SIF_PATH}"
    else
        echo
        echo "ERROR: Output image already exists: ${SIF_PATH}"
        echo "Remove it first, choose another SIF_FILE, or run with OVERWRITE=1."
        exit 1
    fi
fi

echo
echo "Starting Apptainer build..."
echo

apptainer build --fakeroot "${SIF_PATH}" "${DEF_PATH}"

echo
echo "=============================================="
echo "Build completed successfully"
echo "=============================================="
ls -lh "${SIF_PATH}"
