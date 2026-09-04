#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Setup Dedicated Least-Privilege Service Account for Agent Platform (fka Vertex AI)
# ==============================================================================

# --- Environment Files Loading ---
# 1. Load baseline non-sensitive defaults if present
if [ -f ".env.defaults" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env.defaults"
    set +a
fi

# 2. Load user environment configuration (.env takes precedence over defaults)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

# 3. Load optional sensitive secrets if present
if [ -f ".env.secrets" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env.secrets"
    set +a
fi

# --- Configuration Defaults ---
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo '')}"
REGION="${REGION:-us-central1}"
ARTIFACT_REGISTRY_REPOSITORY_NAME="${ARTIFACT_REGISTRY_REPOSITORY_NAME:-geap-models}"
SA_NAME="${SA_NAME:-geap-endpoint-runner}"
SA_DISPLAY_NAME="GEAP LLM Endpoint Runtime Runner"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "================================================================="
echo " Setting up Least-Privilege Service Account for GEAP Endpoint"
echo "================================================================="
echo " Project:              ${PROJECT_ID}"
echo " Service Account Name: ${SA_NAME}"
echo " Service Account Email:${SA_EMAIL}"
echo " Region:               ${REGION}"
echo " Artifact Repo:        ${ARTIFACT_REGISTRY_REPOSITORY_NAME}"
echo "================================================================="

# 1. Create the Service Account if it does not already exist
echo "Step 1: Checking / Creating Service Account '${SA_NAME}'..."
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="${SA_DISPLAY_NAME}" \
        --description="Least-privilege runtime identity for LLM inference endpoints on Agent Platform" \
        --project="${PROJECT_ID}"
    echo "Created Service Account: ${SA_EMAIL}"
else
    echo "Service Account '${SA_EMAIL}' already exists."
fi

# 2. Grant Repository-Level Artifact Registry Reader (Resource-Scoped Least Privilege)
echo "Step 2: Granting repository-scoped Artifact Registry Reader on '${ARTIFACT_REGISTRY_REPOSITORY_NAME}'..."
gcloud artifacts repositories add-iam-policy-binding "${ARTIFACT_REGISTRY_REPOSITORY_NAME}" \
    --location="${REGION}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.reader" \
    --project="${PROJECT_ID}"

# 3. Grant Cloud Logging Writer (Project-Level)
echo "Step 3: Granting Logging Log Writer role..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter" \
    --condition=None

# 4. Grant Monitoring Metric Writer role (for container health metrics)
echo "Step 4: Granting Monitoring Metric Writer role..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/monitoring.metricWriter" \
    --condition=None

# 5. Grant Service Account User to current caller so they can attach it during deploy
CURRENT_USER=$(gcloud config get-value account 2>/dev/null || echo '')
if [ -n "${CURRENT_USER}" ]; then
    echo "Step 5: Granting 'roles/iam.serviceAccountUser' on the SA to deployer '${CURRENT_USER}'..."
    gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
        --member="user:${CURRENT_USER}" \
        --role="roles/iam.serviceAccountUser" \
        --project="${PROJECT_ID}"
fi

echo "================================================================="
echo " Service Account setup completed successfully!"
echo " Email: ${SA_EMAIL}"
echo ""
echo " To use in deployment, pass:"
echo "   --service-account=${SA_EMAIL}"
echo "================================================================="
