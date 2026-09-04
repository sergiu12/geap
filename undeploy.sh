#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Model Serving Undeployment Script for Agent Platform (fka Vertex AI)
# Undeploys the model from the inference endpoint to stop GPU compute charges.
#
# IMPORTANT:
#   This script ONLY undeploys the active model from the endpoint to stop
#   the main GPU serving cost. It does NOT delete other assets that may have
#   been created, such as:
#     - The Agent Platform endpoint resource itself
#     - The registered model in the Agent Platform Model Registry
#     - Container images in Artifact Registry
#     - Cloud Storage buckets or IAM service accounts
# ==============================================================================

# --- Helper Functions ---
usage() {
    cat <<EOF
Usage: ./undeploy.sh [OPTIONS]

Options:
  -c, --config PRESET             Specify model preset from configurations/ (e.g. gpt-oss-120b, gemma-4, generic)
  --endpoint-id ENDPOINT_ID       Explicit Agent Platform numeric endpoint ID
  --deployed-model-id MODEL_ID    Explicit deployed model ID on the endpoint
  -p, --project PROJECT           Override Google Cloud Project ID
  -r, --region REGION             Override Google Cloud Region (e.g. us-central1)
  -y, --yes                       Skip interactive confirmation prompt
  -h, --help                      Show this help message

Examples:
  ./undeploy.sh                               # Default: reads ENDPOINT_NAME from .env, discovers models & undeploys
  ./undeploy.sh --config gpt-oss-120b         # Undeploy using preset configuration
  ./undeploy.sh --config gemma-4              # Undeploy using gemma-4 preset
  ./undeploy.sh --endpoint-id 4456140857224986624
  ./undeploy.sh -y                            # Non-interactive mode (auto-confirm)
EOF
    exit 0
}

# --- Parse Command-Line Arguments ---
CLI_CONFIG=""
CLI_PROJECT=""
CLI_REGION=""
CLI_ENDPOINT_ID=""
CLI_DEPLOYED_MODEL_ID=""
AUTO_CONFIRM="false"

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
        --endpoint-id)
            CLI_ENDPOINT_ID="$2"
            shift 2
            ;;
        --deployed-model-id)
            CLI_DEPLOYED_MODEL_ID="$2"
            shift 2
            ;;
        -y|--yes)
            AUTO_CONFIRM="true"
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
    set -a
    # shellcheck disable=SC1090
    source "${PRESET_FILE}"
    set +a
fi

# --- 3. Load User Overrides from .env (Highest User Precedence) ---
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

# Apply explicit CLI overrides if provided
if [ -n "${CLI_CONFIG}" ]; then CONFIG="${CLI_CONFIG}"; fi
if [ -n "${CLI_PROJECT}" ]; then PROJECT_ID="${CLI_PROJECT}"; fi
if [ -n "${CLI_REGION}" ]; then REGION="${CLI_REGION}"; fi
if [ -n "${CLI_ENDPOINT_ID}" ]; then ENDPOINT_ID="${CLI_ENDPOINT_ID}"; fi
if [ -n "${CLI_DEPLOYED_MODEL_ID}" ]; then DEPLOYED_MODEL_ID="${CLI_DEPLOYED_MODEL_ID}"; fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo '')}"
REGION="${REGION:-us-central1}"
ENDPOINT_NAME="${ENDPOINT_NAME:-custom-llm-endpoint}"
ENDPOINT_ID="${ENDPOINT_ID:-${ENDPOINT_NUMERIC_ID:-}}"
DEPLOYED_MODEL_ID="${DEPLOYED_MODEL_ID:-}"

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: PROJECT_ID is not set."
    echo "Please set PROJECT_ID environment variable or configure gcloud:"
    echo "  export PROJECT_ID=your-gcp-project-id"
    exit 1
fi

echo "================================================================================"
echo " Model Serving Undeployment for Agent Platform (fka Vertex AI)"
echo "================================================================================"
echo " Config Preset:          ${CONFIG}"
echo " Project ID:             ${PROJECT_ID}"
echo " Region:                 ${REGION}"
echo " Endpoint Display Name:  ${ENDPOINT_NAME}"

# Step 1: Resolve Endpoint ID
RESOLVED_ENDPOINT_ID="${ENDPOINT_ID}"
if [ -z "${RESOLVED_ENDPOINT_ID}" ]; then
    echo ""
    echo "Searching for active endpoint named '${ENDPOINT_NAME}'..."
    ENDPOINT_RESOURCE_NAME=$(gcloud ai endpoints list \
        --region="${REGION}" \
        --filter="displayName=${ENDPOINT_NAME}" \
        --format="value(name)" \
        --limit=1 \
        --project="${PROJECT_ID}" 2>/dev/null || true)

    if [ -n "${ENDPOINT_RESOURCE_NAME}" ]; then
        RESOLVED_ENDPOINT_ID=$(echo "${ENDPOINT_RESOURCE_NAME}" | awk -F'/' '{print $NF}')
        echo "Found Endpoint: ${ENDPOINT_RESOURCE_NAME} (ID: ${RESOLVED_ENDPOINT_ID})"
    else
        echo "ERROR: No endpoint found with display name '${ENDPOINT_NAME}' in region '${REGION}'."
        echo "You can specify the numeric endpoint ID manually via: ./undeploy.sh --endpoint-id <ID>"
        exit 1
    fi
else
    echo "Using Specified Endpoint ID: ${RESOLVED_ENDPOINT_ID}"
fi

# Step 2: Retrieve Deployed Models on this Endpoint
if [ -z "${DEPLOYED_MODEL_ID}" ]; then
    echo "Querying deployed models on endpoint ${RESOLVED_ENDPOINT_ID}..."
    DEPLOYED_MODELS_RAW=$(gcloud ai endpoints describe "${RESOLVED_ENDPOINT_ID}" \
        --region="${REGION}" \
        --project="${PROJECT_ID}" \
        --format="value(deployedModels[].id)" 2>/dev/null || true)

    if [ -z "${DEPLOYED_MODELS_RAW}" ]; then
        echo "No models are currently deployed on endpoint ${RESOLVED_ENDPOINT_ID}."
        echo "Endpoint is already idle (no active GPU compute charges)."
        exit 0
    fi
    # Convert space/tab separated values to array
    read -r -a DEPLOYED_MODEL_IDS <<< "${DEPLOYED_MODELS_RAW}"
else
    DEPLOYED_MODEL_IDS=("${DEPLOYED_MODEL_ID}")
fi

echo "Found deployed model ID(s): ${DEPLOYED_MODEL_IDS[*]}"

# Step 3: Confirmation
echo ""
echo "--------------------------------------------------------------------------------"
echo " IMPORTANT NOTICE:"
echo " This action will undeploy the model(s) from endpoint '${RESOLVED_ENDPOINT_ID}'."
echo " This will STOP the active GPU and compute instance charges."
echo ""
echo " Note: This does NOT delete the endpoint itself, registered models in the"
echo " Model Registry, or images in Artifact Registry."
echo "--------------------------------------------------------------------------------"

if [ "${AUTO_CONFIRM}" != "true" ]; then
    read -r -p "Do you want to proceed with undeployment? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            ;;
        *)
            echo "Undeployment cancelled."
            exit 0
            ;;
    esac
fi

# Step 4: Execute Undeploy for each deployed model
for DM_ID in "${DEPLOYED_MODEL_IDS[@]}"; do
    echo ""
    echo "Undeploying model ID '${DM_ID}' from endpoint '${RESOLVED_ENDPOINT_ID}'..."
    gcloud ai endpoints undeploy-model "${RESOLVED_ENDPOINT_ID}" \
        --deployed-model-id="${DM_ID}" \
        --region="${REGION}" \
        --project="${PROJECT_ID}"
done

echo ""
echo "================================================================================"
echo " Model Undeployment Completed Successfully!"
echo "================================================================================"
echo " Active GPU compute instances for endpoint ${RESOLVED_ENDPOINT_ID} have been stopped."
echo ""
echo " To manage remaining assets if desired:"
echo "   - Delete endpoint:       gcloud ai endpoints delete ${RESOLVED_ENDPOINT_ID} --region=${REGION} --project=${PROJECT_ID}"
echo "   - List models:           gcloud ai models list --region=${REGION} --project=${PROJECT_ID}"
echo "   - List container images: gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPOSITORY_NAME:-geap-models}"
echo "================================================================================"
