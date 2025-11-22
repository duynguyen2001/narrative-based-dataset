#!/bin/bash
# Shell script to host vLLM server with Qwen2.5-7B-Instruct model
# This script starts vLLM as an OpenAI-compatible API server

set -e  # Exit on error

# Configuration variables
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PORT=22003
HOST="0.0.0.0"
GPU_DEVICES="6"  # Modify this to match your available GPUs
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096
DOWNLOAD_DIR="${HOME}/.cache/huggingface/hub"

# Project directory
PROJECT_DIR="/shared/nas2/knguye71/conterfactual_dataset"
VENV_PATH="${PROJECT_DIR}/.venv"

# Log file
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vllm_server_$(date +%Y%m%d_%H%M%S).log"

# Check if virtual environment exists
if [ ! -d "${VENV_PATH}" ]; then
    echo "Error: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

# Check if port is already in use
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Error: Port ${PORT} is already in use!"
    exit 1
fi

# Run vLLM server with logging
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --download-dir "${DOWNLOAD_DIR}" \
    --served-model-name "qwen2.5-7b" \
    2>&1 | tee "${LOG_FILE}"

# Note: The server runs in foreground. Press Ctrl+C to stop.
# If you want to run in background, add '&' at the end and redirect logs properly

