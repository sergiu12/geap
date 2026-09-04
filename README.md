# Serving Open-Source LLMs on Google Cloud Agent Platform (fka Vertex AI)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-Supported-purple.svg)](https://github.com/vllm-project/vllm)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Agent%20Platform-4285F4.svg)](https://cloud.google.com/vertex-ai)
[![Version](https://img.shields.io/badge/Version-v0.1.0-orange.svg)](https://github.com/sergiu12/geap)

> [!CAUTION]
> **Disclaimer:** This code is provided "as-is" as a demonstration only to illustrate a potential solution. The code does not constitute a Google product or service of any kind, and Google offers no support, warranties, or liability of any kind with its regard. Whoever chooses to use this code accepts all responsibility related to it, including for its implementation, use, and ongoing maintenance. For the avoidance of doubt, this code is not eligible for the Google Open Source Software Vulnerability Rewards Program.

This repository provides an example of how to deploy and serve open-source Large Language Models from **Hugging Face** (including **OpenAI GPT-OSS-120B**, **Google Gemma 4**, **Meta Llama 3.3**, and other models) on Google Cloud's **Agent Platform (fka Vertex AI)** inference endpoints.

It packages demonstration assets including a container built on **vLLM** and a **FastAPI adapter** to provide:
* **Dual API Support:** OpenAI-compatible standard routes (`POST /v1/chat/completions`, streaming, `:rawPredict`) and standard Agent Platform prediction routes (`POST /predict`).
* **Agentic Tool Calling & Multi-Turn Conversations:** Native function-calling loops, auto tool choice, and context preservation.
* **Modular Model Presets:** Pre-configured presets in [`configurations/`](configurations/) for one-command deployment across model families.
* **Optimized Hardware Profiles:** Sizing for **NVIDIA H100**, **NVIDIA RTX PRO 6000 (G4 Blackwell)**, **NVIDIA A100**, and **NVIDIA L4**.
* **Performance Benchmarking:** Built-in benchmarking suite and native Rust **[`vllm-bench`](https://github.com/vllm-project/vllm/tree/main/rust/src/bench)** integration.

---

## TL;DR — Quickstart (Deploy GPT-OSS-120B in 3 Steps)

```bash
# 1. Clone the repository
git clone https://github.com/sergiu12/geap.git
cd geap

# 2. Configure your Google Cloud Project ID
cp .env.example .env
# Edit .env and set PROJECT_ID=your-gcp-project-id (or run):
sed -i.bak 's/your-gcp-project-id/'"$(gcloud config get-value project)"'/' .env && rm -f .env.bak

# 3. Deploy GPT-OSS-120B to Agent Platform (fka Vertex AI)
./deploy.sh --config gpt-oss-120b

# 4. Test the deployed endpoint
./test.sh --config gpt-oss-120b
```

---

## Table of Contents

* [TL;DR — Quickstart](#tldr--quickstart-deploy-gpt-oss-120b-in-3-steps)
* [1. Architecture Overview](#1-architecture-overview)
* [2. Model Configuration Presets](#2-model-configuration-presets-configurations)
* [3. Dedicated Model Guides](#3-dedicated-model-guides)
* [4. Directory Structure](#4-directory-structure)
* [5. Quickstart & Deployment](#5-quickstart--deployment)
* [6. Testing & Verifying Endpoint](#6-testing--verifying-endpoint)
* [7. Undeploying the Model (`./undeploy.sh`)](#7-undeploying-the-model-undeploysh)
* [8. Performance Benchmarking (`vllm-bench`)](#8-performance-benchmarking-vllm-bench)
* [9. API Request Examples](#9-api-request-examples)
* [10. Security & Best Practices](#10-security--best-practices)


---

## 1. Architecture Overview

```
                       +-------------------------------------------------------------------+
                       |                  Agent Platform (fka Vertex AI)                   |
                       |                     Online Inference Endpoint                     |
                       |                                                                   |
                       |  +-------------------------------------------------------------+  |
Client Requests ------>|  | FastAPI Gateway Adapter (Port 8080)                         |  |
                       |  |                                                             |  |
(OpenAI SDK / REST) -> |  |  - POST /v1/chat/completions (OpenAI Compatible & Streaming)|  |
(OpenAI SDK / REST) -> |  |  - GET  /v1/models           (List Models)                  |  |
(Agent Platform SDK)-> |  |  - POST /predict             (Unpacks instances & params)   |  |
(Platform Probes) ---> |  |  - GET  /health              (Readiness/Liveness probe)     |  |
                       |  +------------------------------+------------------------------+  |
                       |                                 | HTTP localhost:8000             |
                       |  +------------------------------v------------------------------+  |
                       |  | vLLM Serving Engine (Port 8000)                             |  |
                       |  |  - Supported Models: GPT-OSS-120B, Gemma 4, Llama, Mistral  |  |
                       |  |  - Features: PagedAttention, Continuous Batching, KV Cache  |  |
                       |  |  - Tool Calling: Dynamic parser per model family            |  |
                       |  |  - GPU Backend: CUDA 12 / FlashAttention / Triton           |  |
                       |  +------------------------------+------------------------------+  |
                       |                                 | Tensor Parallel IPC             |
                       |  +------------------------------v------------------------------+  |
                       |  | Hardware Acceleration                                       |  |
                       |  |  - 1x NVIDIA H100 80GB (a3-highgpu-1g) OR                   |  |
                       |  |  - 1x NVIDIA RTX PRO 6000 96GB (g4-standard-48) OR          |  |
                       |  |  - 1x NVIDIA A100 80GB (a2-ultragpu-1g) OR                  |  |
                       |  |  - 2x/4x/8x NVIDIA L4 (g2-standard-48/96)                  |  |
                       |  |  - Shared Memory /dev/shm: 16 GB                            |  |
                       |  +-------------------------------------------------------------+  |
                       +-------------------------------------------------------------------+
```

### Dual-Interface Architecture
1. **OpenAI-Compatible Standard Routes (`/v1/chat/completions`, `/v1/models`):** Direct compatibility with OpenAI client libraries, streaming responses (`text/event-stream`), tool calls, and standard JSON chat payloads via Agent Platform's arbitrary custom routes and direct proxying.
2. **Agent Platform Standard Route (`/predict`):** Compatibility with Google Cloud SDKs (`google-cloud-aiplatform`) and services expecting the `{"instances": [...], "parameters": {...}}` contract.
3. **Platform Health Probes (`/health`):** Verifies that vLLM model weights and CUDA graphs are fully loaded before Agent Platform begins routing online traffic.

---

## 2. Model Configuration Presets (`configurations/`)

Model-specific parameters, parser flags, and recommended hardware are organized in the [`configurations/`](file:///Users/sergiulupu/Documents/dev/sergiu-demos/antigravity/gptoss-geap/configurations) directory:

| Preset File | Model ID | Default Hardware | Tool / Reasoning Parsers |
|---|---|---|---|
| **[`configurations/gpt-oss-120b.env`](file:///Users/sergiulupu/Documents/dev/sergiu-demos/antigravity/gptoss-geap/configurations/gpt-oss-120b.env)** | `openai/gpt-oss-120b` | `a3-highgpu-1g` (H100) / `g4-standard-48` (G4) | `openai` / `openai_gptoss` |
| **[`configurations/gemma-4.env`](file:///Users/sergiulupu/Documents/dev/sergiu-demos/antigravity/gptoss-geap/configurations/gemma-4.env)** | `google/gemma-4-27b-it` | `a3-highgpu-1g` (H100) / `g4-standard-48` (G4) / `g2-standard-48` (2x L4) | `hermes` / standard |
| **[`configurations/generic.env`](file:///Users/sergiulupu/Documents/dev/sergiu-demos/antigravity/gptoss-geap/configurations/generic.env)** | Any Hugging Face model | Configurable | Configurable |

### Selecting a Preset
You can select a preset via the `--config` flag or in `.env`:
```bash
# Deploy Google Gemma 4
./deploy.sh --config gemma-4

# Deploy OpenAI GPT-OSS-120B
./deploy.sh --config gpt-oss-120b

# Or set CONFIG in .env:
# CONFIG=gemma-4
```

---

## 3. Dedicated Model Guides

### A. OpenAI GPT-OSS-120B

[`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) is an open-weights Mixture-of-Experts (MoE) model released natively with **MXFP4 quantization** (~4.25 bits/weight).

* **Memory Footprint:** ~60–65 GB VRAM weight footprint.
* **Single GPU Deployment:** Fits on **1x NVIDIA H100 80GB**, **1x NVIDIA RTX PRO 6000 96GB (G4)**, or **1x A100 80GB**, leaving ample headroom for KV cache (32k–128k context).
* **vLLM Engine Flags:**
  * `--reasoning-parser openai_gptoss`: Parses internal reasoning tokens into separate `reasoning` message attributes.
  * `--tool-call-parser openai`: Translates internal token emissions into standard OpenAI `tool_calls`.
  * `--enforce-eager`: Eliminates CUDA graph token drop issues on MXFP4 kernels.
  * `--enable-auto-tool-choice`: Dynamically routes between natural language thought and tool execution.

#### GPT-OSS-120B Sizing:
| GPU Type | Count | Machine Type | Context Window | Recommended Use Case |
|---|---|---|---|---|
| **NVIDIA H100 (80GB)** | 1 | `a3-highgpu-1g` | 32k - 64k tokens | Standard single-GPU deployment; lowest latency |
| **NVIDIA RTX PRO 6000 (96GB)** | 1 | `g4-standard-48` | 64k - 128k tokens | G4 Blackwell GPU with native FP4 acceleration & 96GB VRAM |
| **NVIDIA A100 (80GB)** | 1 | `a2-ultragpu-1g` | 32k tokens | Cost-effective single GPU option |
| **NVIDIA H100 (80GB)** | 2 or 4 | `a3-megagpu-8g` | 128k tokens | High concurrency and extended context length |
| **NVIDIA L4 (24GB)** | 8 (192GB total) | `g2-standard-96` | 64k tokens | Multi-GPU Tensor Parallelism (`--tensor-parallel-size 8`) |

---

### B. Google Gemma 4 & Gemma Family

Google's **Gemma** family (including **Gemma 4**, **Gemma 2 27B/9B**, and **Gemma 3**) delivers strong reasoning and agentic performance in open-weights formats.

* **Precision & Dtype:** Uses `bfloat16` or `float16` with native CUDA graph support (`ENFORCE_EAGER=false`).
* **Tool Calling:** Configured with `--tool-call-parser hermes` or `openai` for structured function calling.
* **Multi-Turn Context:** Leverages vLLM automatic prefix caching to reduce Time-To-First-Token (TTFT) by up to 80% on multi-turn conversations.

#### Gemma Sizing:
| Model Variant | GPU Type | Count | Machine Type | Tensor Parallel |
|---|---|---|---|---|
| **Gemma 4 / 2 (27B)** | **NVIDIA H100 (80GB)** | 1 | `a3-highgpu-1g` | `TENSOR_PARALLEL_SIZE=1` |
| **Gemma 4 / 2 (27B)** | **NVIDIA RTX PRO 6000 (96GB)** | 1 | `g4-standard-48` | `TENSOR_PARALLEL_SIZE=1` |
| **Gemma 4 / 2 (27B)** | **NVIDIA A100 (80GB)** | 1 | `a2-ultragpu-1g` | `TENSOR_PARALLEL_SIZE=1` |
| **Gemma 4 / 2 (27B)** | **NVIDIA L4 (24GB)** | 2 | `g2-standard-48` | `TENSOR_PARALLEL_SIZE=2` |
| **Gemma 4 / 2 (9B)** | **NVIDIA L4 (24GB)** | 1 | `g2-standard-24` | `TENSOR_PARALLEL_SIZE=1` |

---

### C. Deploying Any Custom Hugging Face Model

To serve any other model (such as `meta-llama/Llama-3.3-70B-Instruct`, `mistralai/Mistral-Small-24B-Instruct`, or `Qwen/Qwen2.5-72B-Instruct`):

1. Create or copy a configuration in `configurations/` (e.g. `configurations/my-model.env`):
   ```bash
   MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
   IMAGE_NAME=llama-70b
   MODEL_NAME=llama-70b-geap
   ENDPOINT_NAME=llama-70b-endpoint
   MACHINE_TYPE=a3-highgpu-1g
   ACCELERATOR_TYPE=nvidia-h100-80gb
   ACCELERATOR_COUNT=1
   ```
2. Deploy directly:
   ```bash
   ./deploy.sh --config my-model
   ```

---

## 4. Directory Structure

```
.
├── configurations/             # Model configuration presets
│   ├── gpt-oss-120b.env        # OpenAI GPT-OSS-120B preset (MXFP4 MoE)
│   ├── gemma-4.env             # Google Gemma 4 preset
│   └── generic.env             # Generic Hugging Face model template
├── src/                        # Container runtime code
│   ├── server.py               # FastAPI adapter for Agent Platform (fka Vertex AI) & OpenAI routes
│   ├── entrypoint.sh           # Dynamic container entrypoint managing vLLM and proxy processes
│   └── requirements.txt        # Container python dependencies
├── tests/                      # Testing & verification suite
│   ├── test_endpoint.py        # End-to-end multi-turn & tool calling verification client
│   └── test_adapter.py         # Unit tests for FastAPI adapter
├── Dockerfile                  # Builds custom container packaging src/
├── deploy.sh                   # Deployment automation script with preset support
├── undeploy.sh                 # Model undeployment script to stop active GPU compute charges
├── test.sh                     # Helper script to test deployed or local endpoints
├── bench.sh                    # Performance benchmarking runner (vllm-bench)
├── setup_service_account.sh    # Script to create least-privilege runtime service account
├── request.json                # Sample prediction payload
├── .env.defaults               # Non-sensitive configuration defaults (committed to Git)
├── .env.example                # Configuration template
└── .gitignore                  # Ignores secrets (.env, .env.secrets) and local caches
```

---

## 5. Quickstart & Deployment

### Prerequisites
1. [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and configured:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. Enable required Google Cloud APIs:
   ```bash
   gcloud services enable \
     aiplatform.googleapis.com \
     artifactregistry.googleapis.com \
     cloudbuild.googleapis.com \
     compute.googleapis.com
   ```

---

### Step 1: Configure Environment Variables

Copy the configuration template [.env.example](file:///Users/sergiulupu/Documents/dev/sergiu-demos/antigravity/gptoss-geap/.env.example) to `.env` (which is git-ignored):

```bash
cp .env.example .env
```

Open `.env` in your editor and configure your environment:
* **`PROJECT_ID`** (Required): Your target Google Cloud Project ID.
* **`CONFIG`**: Choose the model preset (`gpt-oss-120b`, `gemma-4`, or custom).
* **`HF_TOKEN`** (Optional): Hugging Face token if accessing gated model weights (e.g. Gemma or Llama).

> [!NOTE]
> All variables configured in `.env` automatically take precedence over model presets and baseline defaults.

---

### Step 2: Deploy to Agent Platform (fka Vertex AI)

Execute `deploy.sh`:

```bash
# Deploy default configured model:
./deploy.sh

# Or explicitly choose a model preset:
./deploy.sh --config gemma-4
./deploy.sh --config gpt-oss-120b
```

The script will automatically:
1. Load `.env.defaults`, the chosen `configurations/<CONFIG>.env` preset, and `.env`.
2. Ensure the Artifact Registry repository exists.
3. Build and push the container image via Cloud Build.
4. Register the model in Agent Platform (fka Vertex AI) Model Registry with custom routes and shared memory.
5. Create or resolve the Endpoint and deploy to the specified GPU machine type.

---

## 6. Testing & Verifying Endpoint

### A. Run the Test Script (`./test.sh`)
You can quickly run multi-turn and tool-calling tests using the provided test runner:

```bash
# Tests deployed endpoint (automatically looks up active endpoint using ENDPOINT_NAME & PROJECT_ID from .env)
./test.sh

# Test a specific model preset endpoint:
./test.sh --config gemma-4
./test.sh --config gpt-oss-120b

# Or explicitly specify parameters:
./test.sh --project="your-project-id" --region="us-central1" --endpoint-id="YOUR_ENDPOINT_NUMERIC_ID"
```

### B. Testing Locally with Docker
You can run and test the container locally on a GPU-enabled machine:
```bash
docker build -t llm-geap .
docker run --gpus all -p 8080:8080 --shm-size=16g llm-geap

# Test health check:
curl http://localhost:8080/health

# Test prediction locally:
./test.sh --local-url http://localhost:8080
```

### C. Run Unit Tests
```bash
python3 -m unittest discover tests/
```

---

## 7. Undeploying the Model (`./undeploy.sh`)

When you are finished testing or want to stop active GPU compute charges, run the undeploy script. **No parameters are required**:

```bash
# 1. Zero-parameter execution (reads ENDPOINT_NAME from .env, discovers deployed models, and undeploys):
./undeploy.sh

# 2. Or undeploy a specific preset or explicit endpoint ID:
./undeploy.sh --config gpt-oss-120b
./undeploy.sh --endpoint-id 4456140857224986624
```

### How `./undeploy.sh` Works:
1. **Reads Configuration:** Automatically loads your `.env` (or preset / `.env.defaults`) to obtain `PROJECT_ID`, `REGION`, and `ENDPOINT_NAME`.
2. **Discovers Deployed Models via `gcloud`:** Queries Agent Platform using `ENDPOINT_NAME` to find the endpoint, then inspects the endpoint to discover which model ID(s) are currently deployed and consuming GPU resources.
3. **Undeploys Models:** Executes `gcloud ai endpoints undeploy-model` to tear down the GPU instances.

> [!IMPORTANT]
> **What `./undeploy.sh` does and does NOT do:**
> * **Stops Active GPU Compute Costs:** The script undeploys the active model replica from the Agent Platform endpoint, immediately terminating the underlying GPU virtual machines (`a3-highgpu-1g`, `g4-standard-48`, etc.) and stopping ongoing compute charges.
> * **Preserves Other Assets:** It does **NOT** delete the endpoint resource itself, registered models in Agent Platform Model Registry, container images in Artifact Registry, or Cloud Storage assets.
>
> To manage or permanently delete the remaining static assets:
> ```bash
> # Delete the idle endpoint resource (if no longer needed):
> gcloud ai endpoints delete <ENDPOINT_ID> --region=us-central1
>
> # Delete the model entry from Model Registry:
> gcloud ai models delete <MODEL_ID> --region=us-central1
> ```

---

## 8. Performance Benchmarking (`vllm-bench`)

You can measure throughput, request rate, TTFT/TPOT, and latency percentiles on the deployed Agent Platform endpoint using **[`vllm-bench`](https://github.com/vllm-project/vllm/tree/main/rust/src/bench)**.

By default, `./bench.sh` uses `vllm-bench`. If the binary is not installed locally, the script automatically downloads the prebuilt binary:
```bash
curl -fsSL https://github.com/vllm-project/vllm-bench/releases/latest/download/vllm-bench-$(uname -m)-linux-musl -o vllm-bench && chmod +x vllm-bench
```

### A. Run Benchmark Runner (`./bench.sh`)
```bash
# Benchmark live deployed endpoint (concurrency=4, 20 requests):
./bench.sh --config gpt-oss-120b --num-prompts 20 --concurrency 4

# Benchmark local container:
./bench.sh --local-url http://localhost:8080 --num-prompts 50 --concurrency 8

# Force using the Python benchmarking engine instead:
./bench.sh --use-python
```

The benchmark reports:
* **Request Throughput:** requests per second (req/s)
* **Token Throughput:** output tokens/sec & total tokens/sec
* **Latency Percentiles:** Mean, Min, P50 (median), P90, P95, P99, Max (seconds)
* **Time Per Output Token (TPOT):** ms/token

### B. Direct Execution with `vllm-bench`
You can also invoke `vllm-bench` directly against local or remote endpoints:

```bash
# Benchmark local container:
./vllm-bench \
  --backend openai-chat \
  --base-url http://localhost:8080/v1 \
  --model openai/gpt-oss-120b \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 256 \
  --num-prompts 100 \
  --max-concurrency 16
```

---

## 9. API Request Examples

### A. Agent Platform Standard Route (`POST /predict`)
Send standard `{"instances": [...], "parameters": {...}}` payload via `gcloud` or cURL:

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/YOUR_ENDPOINT_ID:predict \
  -d '{
    "instances": [
      {
        "messages": [
          {"role": "user", "content": "Explain the architecture of Mixture of Experts in 2 sentences."}
        ]
      }
    ],
    "parameters": {
      "temperature": 0.2,
      "max_tokens": 512
    }
  }'
```

---

### B. OpenAI-Compatible Route (`POST /v1/chat/completions`)
Directly invoke OpenAI chat format using Agent Platform's `:rawPredict` endpoint:

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/YOUR_ENDPOINT_ID:rawPredict \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "List 3 advantages of vLLM serving on Google Cloud."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

---

## 10. Security & Best Practices

1. **Service Accounts & Least Privilege:**
   * Never deploy inference endpoints using project owner or editor service accounts.
   * Use [`setup_service_account.sh`](setup_service_account.sh) to provision a dedicated runtime service account with minimal IAM roles (`roles/aiplatform.user`, `roles/artifactregistry.reader`, `roles/logging.logWriter`).

2. **Secret Management:**
   * Do not commit `.env` or `.env.secrets` files. Keep secrets in git-ignored `.env` files.
   * Hugging Face access tokens (`HF_TOKEN`) and GCP credentials should be managed via environment variables, Secret Manager, or IAM Workload Identity Federation.

3. **Network Isolation:**
   * When working in restricted environments, consider deploying endpoints inside Virtual Private Cloud (VPC) Service Controls or private Service Attachments.

