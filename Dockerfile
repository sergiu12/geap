# syntax=docker/dockerfile:1.4
FROM vllm/vllm-openai:latest

LABEL maintainer="Google Cloud Partner & Solutions Engineering"
LABEL description="Hugging Face LLM Inference Container for Agent Platform (fka Vertex AI)"

WORKDIR /app

# Ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8080 \
    VLLM_PORT=8000 \
    MODEL_ID=openai/gpt-oss-120b \
    REASONING_PARSER=openai_gptoss \
    ENFORCE_EAGER=true \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.92 \
    MAX_MODEL_LEN=32768

# Install system utilities and Python dependencies
COPY src/requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy server code and entrypoint from src/
COPY src/ /app/
RUN chmod +x /app/entrypoint.sh

# Expose standard inference port for Vertex AI / Agent Platform
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
