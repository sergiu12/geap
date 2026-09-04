#!/usr/bin/env bash
set -e

# ==============================================================================
# Custom Container Entrypoint for Hugging Face LLMs on Agent Platform (fka Vertex AI)
# Orchestrates vLLM high-throughput engine and FastAPI translation gateway
# ==============================================================================

# Port configuration (Vertex AI default is 8080)
PORT="${PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="127.0.0.1"

# Model and Engine Configurations
MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

echo "==========================================================="
echo " Starting LLM Serving Container on Agent Platform"
echo " Model ID:                ${MODEL_ID}"
echo " Reasoning Parser:        ${REASONING_PARSER:-None}"
echo " Tool Call Parser:        ${TOOL_CALL_PARSER:-None}"
echo " Tensor Parallel Size:    ${TENSOR_PARALLEL_SIZE}"
echo " Max Model Length:        ${MAX_MODEL_LEN}"
echo " GPU Memory Utilization:  ${GPU_MEMORY_UTILIZATION}"
echo " Max Sequences / Batched: ${MAX_NUM_SEQS} / ${MAX_NUM_BATCHED_TOKENS}"
echo " Enforce Eager Mode:      ${ENFORCE_EAGER}"
echo " Dtype:                   ${DTYPE}"
echo " Gateway Port:            ${PORT} | vLLM Port: ${VLLM_PORT}"
echo "==========================================================="

# Function to clean up background processes on SIGTERM / SIGINT
cleanup() {
    echo "Caught termination signal. Cleaning up background services..."
    if [ -n "${VLLM_PID:-}" ]; then
        echo "Terminating vLLM (PID: $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
    fi
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Terminating Adapter Server (PID: $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi
    wait
    echo "All processes exited cleanly."
    exit 0
}

trap cleanup SIGTERM SIGINT

# 1. Build vLLM command arguments dynamically
VLLM_ARGS=(
    serve "${MODEL_ID}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --dtype "${DTYPE}"
    --max-num-seqs "${MAX_NUM_SEQS}"
)

if [ -n "${TOOL_CALL_PARSER:-}" ]; then
    VLLM_ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER}" --enable-auto-tool-choice)
fi

if [ -n "${REASONING_PARSER:-}" ]; then
    VLLM_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

if [ "${ENFORCE_EAGER:-false}" = "true" ]; then
    VLLM_ARGS+=(--enforce-eager)
fi

if [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    VLLM_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

# Append any extra user-defined flags
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2086
    VLLM_ARGS+=(${VLLM_EXTRA_ARGS})
fi

# Launch vLLM in background
echo "Launching vLLM serving engine..."
vllm "${VLLM_ARGS[@]}" &
VLLM_PID=$!
echo "vLLM started with PID ${VLLM_PID}"

# 2. Launch FastAPI Adapter on $PORT
echo "Starting FastAPI Agent Platform Inference Gateway on port ${PORT}..."
export VLLM_HOST="${VLLM_HOST}"
export VLLM_PORT="${VLLM_PORT}"
export MODEL_ID="${MODEL_ID}"

python3 -m uvicorn server:app --host 0.0.0.0 --port "${PORT}" --workers 2 &
SERVER_PID=$!
echo "Gateway server started with PID ${SERVER_PID}"

# Wait for either process to terminate
wait -n "$VLLM_PID" "$SERVER_PID"
cleanup
