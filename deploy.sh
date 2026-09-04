#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Model Serving Deployment Script for Agent Platform (fka Vertex AI)
# Deploys Open-Source Hugging Face LLMs (GPT-OSS-120B, Gemma 4, Llama, etc.)
#
# Features:
#   1. Model configuration presets via configurations/<CONFIG>.env
#   2. Builds and pushes custom container (vLLM + FastAPI proxy) via Cloud Build
#   3. Uploads model with required custom container flags & shared memory (16GB)
#   4. Deploys to dedicated GPU endpoints (H100, RTX PRO 6000 G4, A100, L4)
# ==============================================================================

# --- Helper Functions ---
usage() {
    cat <<EOF
Usage: ./deploy.sh [OPTIONS]

Options:
  -c, --config PRESET     Specify model preset from configurations/ (e.g. gpt-oss-120b, gemma-4, generic)
  -p, --project PROJECT   Override Google Cloud Project ID
  -r, --region REGION     Override Google Cloud Region (e.g. us-central1)
  --skip-build            Skip container image build and use existing Artifact Registry image
  -h, --help              Show this help message

Examples:
  ./deploy.sh --config gpt-oss-120b
  ./deploy.sh --config gemma-4
  CONFIG=gemma-4 ./deploy.sh
EOF
    exit 0
}

# --- Parse Command-Line Arguments ---
CLI_CONFIG=""
CLI_PROJECT=""
CLI_REGION=""
CLI_SKIP_BUILD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CLI_CONFIG="$2"
            shift 2
            ;;
        -p|--project)
            CLI_PROJECT="$2"
            shift 2
            ;;
        -r|--region)
            CLI_REGION="$2"
            shift 2
            ;;
        --skip-build)
            CLI_SKIP_BUILD="true"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# --- 1. Load Baseline Defaults ---
if [ -f ".env.defaults" ]; then
    echo "Loading baseline defaults from .env.defaults..."
    set -a
    # shellcheck disable=SC1091
    source ".env.defaults"
    set +a
fi

# Determine initial CONFIG selection (CLI flag > ENV var > defaults)
CONFIG="${CLI_CONFIG:-${CONFIG:-gpt-oss-120b}}"

# --- 2. Load Model Configuration Preset from configurations/ ---
PRESET_FILE=""
if [ -f "configurations/${CONFIG}.env" ]; then
    PRESET_FILE="configurations/${CONFIG}.env"
elif [ -f "configurations/${CONFIG}" ]; then
    PRESET_FILE="configurations/${CONFIG}"
elif [ -f "${CONFIG}" ]; then
    PRESET_FILE="${CONFIG}"
fi

if [ -n "${PRESET_FILE}" ]; then
    echo "Loading model configuration preset from ${PRESET_FILE}..."
    set -a
    # shellcheck disable=SC1090
    source "${PRESET_FILE}"
    set +a
else
    echo "Note: No preset file found for '${CONFIG}'. Using direct environment settings."
fi

# --- 3. Load User Overrides from .env (Highest User Precedence) ---
if [ -f ".env" ]; then
    echo "Loading user overrides from .env..."
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

# --- 4. Load Optional Secrets from .env.secrets ---
if [ -f ".env.secrets" ]; then
    echo "Loading secrets from .env.secrets..."
    set -a
    # shellcheck disable=SC1091
    source ".env.secrets"
    set +a
fi

# Apply explicit CLI overrides if provided
if [ -n "${CLI_CONFIG}" ]; then CONFIG="${CLI_CONFIG}"; fi
if [ -n "${CLI_PROJECT}" ]; then PROJECT_ID="${CLI_PROJECT}"; fi
if [ -n "${CLI_REGION}" ]; then REGION="${CLI_REGION}"; fi
if [ -n "${CLI_SKIP_BUILD}" ]; then SKIP_BUILD="${CLI_SKIP_BUILD}"; fi

# --- Configuration & Defaults ---
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo '')}"
REGION="${REGION:-us-central1}"
ARTIFACT_REGISTRY_REPOSITORY_NAME="${ARTIFACT_REGISTRY_REPOSITORY_NAME:-geap-models}"
IMAGE_NAME="${IMAGE_NAME:-custom-llm}"
TAG="${TAG:-latest}"
MODEL_NAME="${MODEL_NAME:-custom-llm-geap}"
ENDPOINT_NAME="${ENDPOINT_NAME:-custom-llm-endpoint}"

# Model & Engine Parameters
MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
REASONING_PARSER="${REASONING_PARSER:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
DTYPE="${DTYPE:-auto}"
SHARED_MEMORY_MB="${SHARED_MEMORY_MB:-16384}"

# Hardware Configuration (Default: 1x NVIDIA H100 80GB)
MACHINE_TYPE="${MACHINE_TYPE:-a3-highgpu-1g}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-h100-80gb}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
MIN_REPLICA_COUNT="${MIN_REPLICA_COUNT:-1}"
MAX_REPLICA_COUNT="${MAX_REPLICA_COUNT:-1}"

# Deployment Control Flags
SKIP_BUILD="${SKIP_BUILD:-false}"
BUILD_METHOD="${BUILD_METHOD:-cloudbuild}" # 'cloudbuild' or 'docker'
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
ENABLE_APIS="${ENABLE_APIS:-true}"

# Validation
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: PROJECT_ID is not set."
    echo "Please set PROJECT_ID environment variable or configure gcloud:"
    echo "  export PROJECT_ID=your-gcp-project-id"
    echo "  gcloud config set project your-gcp-project-id"
    exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPOSITORY_NAME}/${IMAGE_NAME}:${TAG}"

echo "================================================================================"
echo " Model Serving Deployment for Agent Platform (fka Vertex AI)"
echo "================================================================================"
echo " Config Preset:          ${CONFIG}"
echo " Model ID:               ${MODEL_ID}"
echo " Project ID:             ${PROJECT_ID}"
echo " Region:                 ${REGION}"
echo " Artifact Registry URI:  ${IMAGE_URI}"
echo " Model Display Name:     ${MODEL_NAME}"
echo " Endpoint Display Name:  ${ENDPOINT_NAME}"
echo " Machine Type:           ${MACHINE_TYPE}"
echo " Accelerator:            ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}"
echo " Tensor Parallel Size:   ${TENSOR_PARALLEL_SIZE}"
echo " Max Context Length:     ${MAX_MODEL_LEN}"
echo " Enforce Eager Mode:     ${ENFORCE_EAGER}"
echo " Reasoning / Tool Parser:${REASONING_PARSER:-None} / ${TOOL_CALL_PARSER:-None}"
echo "================================================================================"

# ------------------------------------------------------------------------------
# Step 0: Ensure Google Cloud APIs are Enabled
# ------------------------------------------------------------------------------
if [ "$ENABLE_APIS" = "true" ]; then
    echo "Step 0: Ensuring required Google Cloud APIs are enabled..."
    gcloud services enable \
        aiplatform.googleapis.com \
        artifactregistry.googleapis.com \
        cloudbuild.googleapis.com \
        compute.googleapis.com \
        --project="${PROJECT_ID}"
fi

# ------------------------------------------------------------------------------
# Step 1: Ensure Artifact Registry Repository Exists
# ------------------------------------------------------------------------------
echo "Step 1: Checking Artifact Registry repository '${ARTIFACT_REGISTRY_REPOSITORY_NAME}'..."
if ! gcloud artifacts repositories describe "${ARTIFACT_REGISTRY_REPOSITORY_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" &>/dev/null; then
    echo "Creating Artifact Registry repository '${ARTIFACT_REGISTRY_REPOSITORY_NAME}'..."
    gcloud artifacts repositories create "${ARTIFACT_REGISTRY_REPOSITORY_NAME}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="Docker repository for Agent Platform LLM custom containers" \
        --project="${PROJECT_ID}"
else
    echo "Artifact Registry repository already exists."
fi

# ------------------------------------------------------------------------------
# Step 2: Build and Push Container Image
# ------------------------------------------------------------------------------
if [ "$SKIP_BUILD" != "true" ]; then
    echo "Step 2: Building and pushing container image (${BUILD_METHOD})..."
    if [ "$BUILD_METHOD" = "cloudbuild" ]; then
        echo "Submitting build to Google Cloud Build (context scoped from root)..."
        gcloud builds submit . \
            --tag="${IMAGE_URI}" \
            --project="${PROJECT_ID}" \
            --timeout=2400s
    elif [ "$BUILD_METHOD" = "docker" ]; then
        echo "Building and pushing locally using Docker..."
        gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
        docker build -t "${IMAGE_URI}" .
        docker push "${IMAGE_URI}"
    else
        echo "ERROR: Unknown BUILD_METHOD '${BUILD_METHOD}'. Must be 'cloudbuild' or 'docker'."
        exit 1
    fi
    echo "Container image successfully built and pushed: ${IMAGE_URI}"
else
    echo "Step 2: Skipping container build (SKIP_BUILD=true). Using: ${IMAGE_URI}"
fi

# ------------------------------------------------------------------------------
# Step 3: Register Custom Model in Agent Platform Model Registry
# ------------------------------------------------------------------------------
echo "Step 3: Registering Model in Model Registry..."

CONTAINER_ENV_VARS="MODEL_ID=${MODEL_ID}"
CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},MAX_MODEL_LEN=${MAX_MODEL_LEN}"
CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},MAX_NUM_SEQS=${MAX_NUM_SEQS}"
CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},DTYPE=${DTYPE}"
CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},ENFORCE_EAGER=${ENFORCE_EAGER}"

if [ -n "${REASONING_PARSER:-}" ]; then
    CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},REASONING_PARSER=${REASONING_PARSER}"
fi
if [ -n "${TOOL_CALL_PARSER:-}" ]; then
    CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},TOOL_CALL_PARSER=${TOOL_CALL_PARSER}"
fi
if [ -n "${HF_TOKEN:-}" ]; then
    CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS},HF_TOKEN=${HF_TOKEN}"
fi

echo "Uploading model resource with health probes and shared memory (${SHARED_MEMORY_MB}MB)..."
gcloud ai models upload \
    --region="${REGION}" \
    --display-name="${MODEL_NAME}" \
    --container-image-uri="${IMAGE_URI}" \
    --container-predict-route="/predict" \
    --container-health-route="/health" \
    --container-ports=8080 \
    --container-shared-memory-size-mb="${SHARED_MEMORY_MB}" \
    --container-env-vars="${CONTAINER_ENV_VARS}" \
    --project="${PROJECT_ID}"

# Retrieve Model Resource ID
MODEL_RESOURCE_ID=$(gcloud ai models list \
    --region="${REGION}" \
    --filter="displayName=${MODEL_NAME}" \
    --format="value(name)" \
    --sort-by="~createTime" \
    --limit=1 \
    --project="${PROJECT_ID}")

echo "Model successfully registered. Model Resource ID: ${MODEL_RESOURCE_ID}"

# ------------------------------------------------------------------------------
# Step 4: Create or Resolve Inference Endpoint
# ------------------------------------------------------------------------------
echo "Step 4: Checking / Creating Inference Endpoint..."

ENDPOINT_RESOURCE_ID=$(gcloud ai endpoints list \
    --region="${REGION}" \
    --filter="displayName=${ENDPOINT_NAME}" \
    --format="value(name)" \
    --limit=1 \
    --project="${PROJECT_ID}" || true)

if [ -z "$ENDPOINT_RESOURCE_ID" ]; then
    echo "Creating new Endpoint '${ENDPOINT_NAME}' in region '${REGION}'..."
    ENDPOINT_RESOURCE_ID=$(gcloud ai endpoints create \
        --region="${REGION}" \
        --display-name="${ENDPOINT_NAME}" \
        --format="value(name)" \
        --project="${PROJECT_ID}")
    echo "Endpoint created: ${ENDPOINT_RESOURCE_ID}"
else
    echo "Using existing Endpoint: ${ENDPOINT_RESOURCE_ID}"
fi

# Extract numeric ID for convenient CLI queries
ENDPOINT_NUMERIC_ID=$(echo "${ENDPOINT_RESOURCE_ID}" | awk -F'/' '{print $NF}')

# ------------------------------------------------------------------------------
# Step 5: Deploy Model to Dedicated GPU Endpoint
# ------------------------------------------------------------------------------
echo "Step 5: Deploying ${MODEL_ID} to Endpoint..."
echo "Note: Deployment typically takes 10-15 minutes while GPU nodes provision and weights load into VRAM."

DEPLOY_ARGS=(
    ai endpoints deploy-model "${ENDPOINT_NUMERIC_ID}"
    --region="${REGION}"
    --model="${MODEL_RESOURCE_ID}"
    --display-name="${MODEL_NAME}-deployment"
    --machine-type="${MACHINE_TYPE}"
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}"
    --min-replica-count="${MIN_REPLICA_COUNT}"
    --max-replica-count="${MAX_REPLICA_COUNT}"
    --traffic-split="0=100"
    --project="${PROJECT_ID}"
)

if [ -n "${SERVICE_ACCOUNT}" ]; then
    DEPLOY_ARGS+=(--service-account="${SERVICE_ACCOUNT}")
fi

gcloud "${DEPLOY_ARGS[@]}"

echo "================================================================================"
echo " Deployment Successfully Completed!"
echo "================================================================================"
echo " Model ID:             ${MODEL_ID}"
echo " Endpoint Resource ID: ${ENDPOINT_RESOURCE_ID}"
echo " Endpoint Numeric ID:  ${ENDPOINT_NUMERIC_ID}"
echo " Region:               ${REGION}"
echo " Project:              ${PROJECT_ID}"
echo ""
echo " To test standard Agent Platform prediction (/predict):"
echo "   ./test.sh --config ${CONFIG}"
echo "   # Or: ./test.sh --endpoint-id ${ENDPOINT_NUMERIC_ID} --project ${PROJECT_ID} --region ${REGION}"
echo ""
echo " To send a raw cURL request via REST:"
echo "   curl -X POST \\"
echo "     -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_NUMERIC_ID}:predict \\"
echo "     -d @request.json"
echo "================================================================================"
