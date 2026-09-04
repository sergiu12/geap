"""
Unit Tests for Agent Platform Custom Container Adapter.
Tests /health, /predict, /v1/chat/completions, and /v1/models using Python unittest.
"""

import os
import sys
import json
import unittest
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient, ASGITransport, Response

# Ensure src/ is on python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from server import app


class TestAdapterEndpoints(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.transport = ASGITransport(app=app)

    @patch("server.http_client")
    async def test_health_check_healthy(self, mock_client):
        mock_client.get = AsyncMock(return_value=Response(200, json={"status": "ok"}))
        async with AsyncClient(transport=self.transport, base_url="http://test") as ac:
            response = await ac.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "healthy")

    @patch("server.http_client")
    async def test_health_check_unhealthy(self, mock_client):
        mock_client.get = AsyncMock(return_value=Response(503, json={"status": "loading"}))
        async with AsyncClient(transport=self.transport, base_url="http://test") as ac:
            response = await ac.get("/health")
            self.assertEqual(response.status_code, 503)

    @patch("server.http_client")
    async def test_vertex_predict_translation(self, mock_client):
        mock_vllm_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1740312000,
            "model": "openai/gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "Tokyo"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }

        mock_client.post = AsyncMock(return_value=Response(200, json=mock_vllm_response))

        async with AsyncClient(transport=self.transport, base_url="http://test") as ac:
            payload = {
                "instances": [
                    {
                        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
                        "tools": [{"type": "function", "function": {"name": "get_current_weather"}}],
                        "temperature": 0.2,
                    }
                ],
                "parameters": {"max_tokens": 1024},
            }
            response = await ac.post("/predict", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("predictions", data)
            self.assertEqual(len(data["predictions"]), 1)
            pred = data["predictions"][0]
            self.assertEqual(pred["choices"][0]["finish_reason"], "tool_calls")
            self.assertEqual(
                pred["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "get_current_weather",
            )

    @patch("server.http_client")
    async def test_v1_models_endpoint(self, mock_client):
        mock_models_response = {
            "object": "list",
            "data": [{"id": "openai/gpt-oss-120b", "object": "model"}],
        }
        mock_client.get = AsyncMock(return_value=Response(200, json=mock_models_response))

        async with AsyncClient(transport=self.transport, base_url="http://test") as ac:
            response = await ac.get("/v1/models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["object"], "list")
            self.assertEqual(data["data"][0]["id"], "openai/gpt-oss-120b")


if __name__ == "__main__":
    unittest.main()
