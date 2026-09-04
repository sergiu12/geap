#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Endpoint Test & Verification Runner
# Runs tests/test_endpoint.py resolving the live Endpoint via ENDPOINT_NAME in .env
# Supports model presets via --config <preset>
# ==============================================================================

# --- Helper Functions ---
usage() {
    cat <<EOF
Usage: ./test.sh [OPTIONS]

Options:
  -c, --config PRESET         Specify model preset from configurations/ (e.g. gpt-oss-120b, gemma-4)
  --endpoint-id ENDPOINT_ID   Explicit Agent Platform numeric endpoint ID
  --project PROJECT_ID        Override Google Cloud Project ID
  --region REGION             Override Google Cloud Region
  --local-url URL             Test against local container URL (e.g. http://localhost:8080)
  --mock-test                 Run mock test
  --query QUERY               Custom test prompt query
  -h, --help                  Show this help message

Examples:
  ./test.sh
  ./test.sh --config gemma-4
  ./test.sh --local-url http://localhost:8080
EOF
    exit 0
}

# --- Parse Arguments for --config ---
CLI_CONFIG=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CLI_CONFIG="$2"
            shift 2
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

# 1. Forward custom arguments directly if explicitly provided on the CLI
if [ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]; then
    python3 tests/test_endpoint.py "${PASSTHROUGH_ARGS[@]}"
    exit $?
fi

# 2. If a specific numeric ENDPOINT_ID is already defined in environment
if [ -n "${ENDPOINT_ID}" ] && [ -n "${PROJECT_ID}" ]; then
    echo "Testing Agent Platform endpoint ${ENDPOINT_ID} (Project: ${PROJECT_ID}, Region: ${REGION})..."
    python3 tests/test_endpoint.py \
        --endpoint-id "${ENDPOINT_ID}" \
        --project "${PROJECT_ID}" \
        --region "${REGION}"
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
        python3 tests/test_endpoint.py \
            --endpoint-id "${RESOLVED_ID}" \
            --project="${PROJECT_ID}" \
            --region="${REGION}"
        exit $?
    else
        echo "No active endpoint found with display name '${ENDPOINT_NAME}' in project ${PROJECT_ID}."
    fi
fi

# 4. Fallback to testing local container gateway
echo "Falling back to testing local gateway at http://localhost:8080..."
python3 tests/test_endpoint.py --local-url "http://localhost:8080"
