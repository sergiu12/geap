# Contributing to GEAP Model Serving

Thank you for your interest in contributing to these open-source examples and demonstration assets for serving Large Language Models on Google Cloud's **Agent Platform (fka Vertex AI)**!

---

## Code of Conduct

Please help us keep this project open, welcoming, and productive. Be respectful, constructive, and collaborative.

---

## How to Contribute

### 1. Adding a New Model Preset
To add support for a new model from Hugging Face:
1. Create a new configuration file in `configurations/<model-name>.env`.
2. Configure the required parameters:
   * `MODEL_ID`: Hugging Face repository ID (e.g. `google/gemma-4-27b-it`, `meta-llama/Llama-3.3-70B-Instruct`).
   * `TOOL_CALL_PARSER`: Tool call parser name supported by vLLM (e.g., `openai`, `hermes`, `llama3_json`, `mistral`).
   * `REASONING_PARSER`: Optional reasoning token parser.
   * `MACHINE_TYPE` & `ACCELERATOR_TYPE`: Recommended Google Cloud GPU machine type.
   * `TENSOR_PARALLEL_SIZE`: Number of GPUs required.
3. Update [README.md](README.md) to document the new model preset.

---

### 2. Code Style & Standards

* **Python:** Follow PEP 8 guidelines. Include type annotations for public methods and Pydantic schemas.
* **Shell Scripts:** Use `set -euo pipefail` at the top of all bash scripts. Ensure compatibility with bash 4+ and zsh.
* **Security:** Never commit API keys, service account keys, or tokens. Ensure `.gitignore` and `.gcloudignore` remain comprehensive.

---

### 3. Testing Changes

* Run unit tests:
  ```bash
  python3 -m unittest discover tests/
  ```
* Test deployment and endpoint verification:
  ```bash
  ./deploy.sh --config <preset>
  ./test.sh --config <preset>
  ```
