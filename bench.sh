#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Endpoint Benchmark Runner (vLLM-Bench for Agent Platform)
# Defaults to high-performance vllm-bench package (downloading prebuilt binary).
# Automatically resolves the live Endpoint via ENDPOINT_NAME in .env
# Supports model presets via --config <preset>
# ==============================================================================

# --- Helper Functions ---
usage() {
    cat <<EOF
Usage: ./bench.sh [OPTIONS]

Options:
  -c, --config PRESET         Specify model preset from configurations/ (e.g. gpt-oss-120b, gemma-4)
  --endpoint-id ENDPOINT_ID   Explicit Agent Platform numeric endpoint ID
  --project PROJECT_ID        Override Google Cloud Project ID
  --region REGION             Override Google Cloud Region
  --local-url URL             Benchmark against local container URL (e.g. http://localhost:8080)
  --num-prompts NUM           Total benchmark requests to send (default: 10)
  --concurrency NUM           Concurrent requests (default: 2)
  --prompt-tokens NUM         Approximate input prompt token count (default: 128)
  --max-tokens NUM            Maximum output generation tokens per request (default: 256)
  --temperature TEMP          Sampling temperature (default: 0.7)
  --use-python                Force using Python benchmark suite (tests/benchmark.py) instead of vllm-bench
  -h, --help                  Show this help message

Default Engine:
  vllm-bench (Rust). If not installed locally, the prebuilt binary is downloaded
  automatically from:
  https://github.com/vllm-project/vllm-bench/releases/latest/download/vllm-bench-\$(uname -m)-linux-musl

Examples:
  ./bench.sh
  ./bench.sh --config gpt-oss-120b --num-prompts 20 --concurrency 4
  ./bench.sh --local-url http://localhost:8080 --num-prompts 50 --concurrency 8
  ./bench.sh --use-python
EOF
    exit 0
}

# --- Parse Arguments ---
CLI_CONFIG=""
USE_PYTHON=false
NUM_PROMPTS=10
CONCURRENCY=2
PROMPT_TOKENS=128
MAX_TOKENS=256
TEMPERATURE=0.7
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CLI_CONFIG="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --prompt-tokens)
            PROMPT_TOKENS="$2"
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --use-python)
            USE_PYTHON=true
            shift
            ;;
        --use-rust)
            # Retained for backwards compatibility
            USE_PYTHON=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- 1. Load Baseline Defaults ---
if [ -f ".env.defaults" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env.defaults"
    set +a
fi

# Determine CONFIG selection (CLI flag > ENV var > default)
CONFIG="${CLI_CONFIG:-${CONFIG:-gpt-oss-120b}}"

# --- 2. Load Preset Configuration from configurations/ ---
PRESET_FILE=""
if [ -f "configurations/${CONFIG}.env" ]; then
    PRESET_FILE="configurations/${CONFIG}.env"
elif [ -f "configurations/${CONFIG}" ]; then
    PRESET_FILE="configurations/${CONFIG}"
fi

if [ -n "${PRESET_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${PRESET_FILE}"
    set +a
fi

# --- 3. Load User Overrides from .env ---
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

# --- 4. Load Secrets from .env.secrets ---
if [ -f ".env.secrets" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env.secrets"
    set +a
fi

# Apply explicit CLI config if provided
if [ -n "${CLI_CONFIG}" ]; then CONFIG="${CLI_CONFIG}"; fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo '')}"
REGION="${REGION:-us-central1}"
ENDPOINT_NAME="${ENDPOINT_NAME:-gpt-oss-120b-endpoint}"
ENDPOINT_ID="${ENDPOINT_ID:-${ENDPOINT_NUMERIC_ID:-}}"
MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"

# Ensure vllm-bench binary is available
VLLM_BENCH_BIN=""
CAN_RUN_VLLM_BENCH=false

ensure_vllm_bench() {
    if [ "${USE_PYTHON}" = true ]; then
        return 0
    fi

    # 1. Check existing PATH / local installs
    if command -v vllm-bench &>/dev/null; then
        VLLM_BENCH_BIN="$(command -v vllm-bench)"
    elif [ -f "./vllm-bench" ]; then
        VLLM_BENCH_BIN="./vllm-bench"
    elif [ -f "$HOME/.local/bin/vllm-bench" ]; then
        VLLM_BENCH_BIN="$HOME/.local/bin/vllm-bench"
    elif [ -f "$HOME/.cargo/bin/vllm-bench" ]; then
        VLLM_BENCH_BIN="$HOME/.cargo/bin/vllm-bench"
    else
        # 2. Download prebuilt binary per upstream instructions
        # https://github.com/vllm-project/vllm/tree/main/rust/src/bench#prebuilt-binaries-linux
        local raw_arch arch
        raw_arch="$(uname -m)"
        case "${raw_arch}" in
            arm64|aarch64) arch="aarch64" ;;
            x86_64|amd64) arch="x86_64" ;;
            *) arch="${raw_arch}" ;;
        esac
        echo "vllm-bench binary not found locally."
        echo "Downloading prebuilt vllm-bench binary (arch: ${arch})..."
        local download_url="https://github.com/vllm-project/vllm-bench/releases/latest/download/vllm-bench-${arch}-linux-musl"
        if curl -fsSL "${download_url}" -o ./vllm-bench && chmod +x ./vllm-bench; then
            echo "Successfully downloaded vllm-bench to ./vllm-bench"
            VLLM_BENCH_BIN="./vllm-bench"
        else
            echo "Warning: Could not download prebuilt vllm-bench binary from ${download_url}."
        fi
    fi

    # 3. Test if binary is executable on the current host OS
    if [ -n "${VLLM_BENCH_BIN}" ]; then
        if "${VLLM_BENCH_BIN}" --version &>/dev/null || "${VLLM_BENCH_BIN}" --help &>/dev/null; then
            CAN_RUN_VLLM_BENCH=true
        else
            echo "Notice: '${VLLM_BENCH_BIN}' is not executable on this OS ($(uname -s) $(uname -m))."
            echo "Prebuilt binaries are built for Linux. Falling back to Python benchmark engine..."
            echo "(To build native macOS binary: cargo install --git https://github.com/vllm-project/vllm-bench vllm-bench)"
            CAN_RUN_VLLM_BENCH=false
        fi
    fi
}

ensure_vllm_bench

# Function to execute vllm-bench
run_vllm_bench() {
    local target_url="$1"
    local auth_header="${2:-}"

    echo "=================================================================="
    echo " Running benchmark with native vllm-bench (${VLLM_BENCH_BIN})"
    echo " Target:       ${target_url}"
    echo " Model:        ${MODEL_ID}"
    echo " Prompts:      ${NUM_PROMPTS}"
    echo " Concurrency:  ${CONCURRENCY}"
    echo "=================================================================="

    local cmd=(
        "${VLLM_BENCH_BIN}"
        --backend openai-chat
        --base-url "${target_url}"
        --model "${MODEL_ID}"
        --dataset-name random
        --random-input-len "${PROMPT_TOKENS}"
        --random-output-len "${MAX_TOKENS}"
        --num-prompts "${NUM_PROMPTS}"
        --max-concurrency "${CONCURRENCY}"
    )

    if [ -n "${auth_header}" ]; then
        OPENAI_API_KEY="${auth_header}" "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

# Function to run Python benchmark fallback
run_python_bench() {
    local extra_args=("$@")
    python3 tests/benchmark.py "${extra_args[@]}"
}

# 1. Forward custom arguments directly if explicitly provided on the CLI
if [[ " ${PASSTHROUGH_ARGS[*]:-} " =~ " --endpoint-id " ]] || [[ " ${PASSTHROUGH_ARGS[*]:-} " =~ " --local-url " ]]; then
    run_python_bench "${PASSTHROUGH_ARGS[@]}"
    exit $?
fi

# 2. If a specific numeric ENDPOINT_ID is defined in environment
if [ -n "${ENDPOINT_ID}" ] && [ -n "${PROJECT_ID}" ]; then
    echo "Benchmarking Agent Platform endpoint ${ENDPOINT_ID} (Project: ${PROJECT_ID}, Region: ${REGION})..."
    if [ "${CAN_RUN_VLLM_BENCH}" = true ]; then
        AUTH_TOKEN="$(gcloud auth print-access-token 2>/dev/null || echo '')"
        RAW_URL="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:rawPredict"
        run_vllm_bench "${RAW_URL}" "${AUTH_TOKEN}"
    else
        run_python_bench \
            --endpoint-id "${ENDPOINT_ID}" \
            --project "${PROJECT_ID}" \
            --region "${REGION}" \
            "${PASSTHROUGH_ARGS[@]}"
    fi
    exit $?
fi

# 3. Dynamic lookup: Query Google Cloud for the endpoint with display name ENDPOINT_NAME
if [ -n "${PROJECT_ID}" ]; then
    echo "Looking up endpoint '${ENDPOINT_NAME}' (Preset: ${CONFIG}) in region '${REGION}'..."
    ENDPOINT_RESOURCE_ID=$(gcloud ai endpoints list \
        --region="${REGION}" \
        --filter="displayName=${ENDPOINT_NAME}" \
        --format="value(name)" \
        --limit=1 \
        --project="${PROJECT_ID}" 2>/dev/null || true)

    if [ -n "${ENDPOINT_RESOURCE_ID}" ]; then
        RESOLVED_ID=$(echo "${ENDPOINT_RESOURCE_ID}" | awk -F'/' '{print $NF}')
        echo "Found active endpoint: ${ENDPOINT_RESOURCE_ID} (ID: ${RESOLVED_ID})"
        if [ "${CAN_RUN_VLLM_BENCH}" = true ]; then
            AUTH_TOKEN="$(gcloud auth print-access-token 2>/dev/null || echo '')"
            RAW_URL="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${RESOLVED_ID}:rawPredict"
            run_vllm_bench "${RAW_URL}" "${AUTH_TOKEN}"
        else
            run_python_bench \
                --endpoint-id "${RESOLVED_ID}" \
                --project="${PROJECT_ID}" \
                --region="${REGION}" \
                "${PASSTHROUGH_ARGS[@]}"
        fi
        exit $?
    else
        echo "No active endpoint found with display name '${ENDPOINT_NAME}' in project ${PROJECT_ID}."
    fi
fi

# 4. Fallback to benchmarking local container gateway
echo "Falling back to benchmarking local gateway at http://localhost:8080..."
if [ "${CAN_RUN_VLLM_BENCH}" = true ]; then
    run_vllm_bench "http://localhost:8080/v1" ""
else
    run_python_bench --local-url "http://localhost:8080" "${PASSTHROUGH_ARGS[@]}"
fi
