"""
Client verification and demonstration script for Hugging Face LLMs (GPT-OSS-120B, Gemma 4, Llama, etc.) on Agent Platform (fka Vertex AI).
Demonstrates:
  1. Standard Agent Platform (fka Vertex AI) Endpoint prediction (/predict)
  2. Multi-turn conversation context retention
  3. Agentic Tool Calling loop (Tool definition -> Model invocation -> Tool execution -> Final response)
  4. Local mock/direct FastAPI gateway testing (OpenAI /predict routes)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import httpx


# --- Simulated Tools for Agent Execution ---

def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Mock weather lookup tool."""
    mock_db = {
        "tokyo": "18°C, Partly Cloudy with mild breeze",
        "san francisco": "15°C, Foggy / Marine Layer",
        "london": "12°C, Light Rain",
        "new york": "22°C, Clear and Sunny",
    }
    normalized = location.lower().strip()
    return mock_db.get(normalized, f"20°C, Sunny in {location}")


def calculate_portfolio_value(ticker: str, shares: float) -> str:
    """Mock financial calculation tool."""
    prices = {"GOOGL": 185.50, "MSFT": 420.00, "AAPL": 225.00, "NVDA": 125.00}
    price = prices.get(ticker.upper(), 100.0)
    total = price * shares
    return json.dumps({"ticker": ticker, "shares": shares, "unit_price": price, "total_value": total})


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather and conditions for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g. Tokyo, Japan",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio_value",
            "description": "Calculate total valuation of stock holdings given ticker and quantity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock symbol (e.g. GOOGL, NVDA)"},
                    "shares": {"type": "number", "description": "Number of shares owned"},
                },
                "required": ["ticker", "shares"],
            },
        },
    },
]

TOOL_CALL_MAP = {
    "get_current_weather": get_current_weather,
    "calculate_portfolio_value": calculate_portfolio_value,
}


# --- Prediction Invokers ---

def invoke_via_vertex_rest(endpoint_id: str, project_id: str, region: str, instance: Dict[str, Any]) -> Dict[str, Any]:
    """Invokes Vertex AI prediction REST endpoint directly using gcloud access token."""
    import subprocess
    clean_id = endpoint_id.split("/")[-1]
    token_out = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    token = [line.strip() for line in token_out.splitlines() if line.strip() and not line.startswith("WARNING")][-1]
    
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/{clean_id}:predict"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(url, headers=headers, json={"instances": [instance]})
        if resp.status_code != 200:
            raise RuntimeError(f"Vertex AI API error ({resp.status_code}): {resp.text}")
        return resp.json()["predictions"][0]


def invoke_via_vertex_sdk(endpoint_resource: str, project_id: str, region: str, instance: Dict[str, Any]) -> Dict[str, Any]:
    """Invokes prediction using the official google-cloud-aiplatform SDK."""
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)
    if not endpoint_resource.startswith("projects/"):
        endpoint_resource = f"projects/{project_id}/locations/{region}/endpoints/{endpoint_resource}"
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_resource)
    response = endpoint.predict(instances=[instance])
    return response.predictions[0]


def invoke_via_local_gateway(gateway_url: str, instance: Dict[str, Any]) -> Dict[str, Any]:
    """Invokes prediction via local or forwarded FastAPI gateway."""
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{gateway_url}/predict",
            json={"instances": [instance]},
        )
        resp.raise_for_status()
        return resp.json()["predictions"][0]


# --- Multi-Turn & Tool Calling Agent Workflow ---

def run_agentic_workflow(invoker_fn, prompt_query: str):
    """
    Executes a complete multi-turn conversation loop with dynamic tool calling.
    """
    print(f"\n===========================================================")
    print(f" User Query: '{prompt_query}'")
    print(f"===========================================================")

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are an intelligent AI agent powered by GPT-OSS-120B on Google Cloud Agent Platform. "
                       "You have access to real-time tools. ALWAYS invoke tools when requested to retrieve real data.",
        },
        {"role": "user", "content": prompt_query},
    ]

    iteration = 1
    max_iterations = 5

    while iteration <= max_iterations:
        print(f"\n--- Turn {iteration}: Sending request to model ---")
        instance = {
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
            "temperature": 0.2,
            "reasoning_effort": "medium",
        }

        result = invoker_fn(instance)
        choice = result["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")

        # Append assistant response to chat history
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if tool_calls:
            print(f" Model invoked {len(tool_calls)} tool call(s):")
            for tc in tool_calls:
                call_id = tc["id"]
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                print(f"   -> Function: {fn_name}({fn_args}) [Call ID: {call_id}]")

                # Execute tool locally
                if fn_name in TOOL_CALL_MAP:
                    tool_output = TOOL_CALL_MAP[fn_name](**fn_args)
                else:
                    tool_output = f"Error: Tool {fn_name} not recognized."

                print(f"   <- Tool Output: {tool_output}")

                # Return tool result back to model in multi-turn format
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(tool_output),
                })

            iteration += 1
            continue

        # If no tool calls, model gave final answer
        print(f"\n Final Assistant Response:")
        print(f"{message.get('content')}")
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            print(f"\n Chain of Thought / Reasoning:")
            print(f"{reasoning}")
        break


def main():
    parser = argparse.ArgumentParser(description="Test LLM Endpoint on Agent Platform (fka Vertex AI)")
    parser.add_argument("--endpoint-id", help="Agent Platform (fka Vertex AI) Endpoint Resource Name / ID")
    parser.add_argument("--project", help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--local-url", default="http://127.0.0.1:8080", help="Local adapter server URL")
    parser.add_argument("--mock-test", action="store_true", help="Run simulated test against local adapter")
    parser.add_argument(
        "--query",
        default="What's the weather in Tokyo right now, and what is the current value of 50 shares of GOOGL?",
        help="Test query",
    )

    args = parser.parse_args()

    if args.endpoint_id and args.project:
        print(f"Connecting to live Agent Platform (fka Vertex AI) Endpoint: {args.endpoint_id}")
        invoker = lambda inst: invoke_via_vertex_rest(args.endpoint_id, args.project, args.region, inst)
    else:
        print(f"Connecting to local/forwarded gateway at: {args.local_url}")
        invoker = lambda inst: invoke_via_local_gateway(args.local_url, inst)

    run_agentic_workflow(invoker, args.query)


if __name__ == "__main__":
    main()
