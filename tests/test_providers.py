"""Offline adapter tests using real SDKs and mocked HTTP transports. No API calls."""

import json
import io
import os
import unittest
from unittest.mock import patch

import anthropic
import httpx
import openai
from google import genai
from google.oauth2.credentials import Credentials

from course_game.narrator import NarrationError, narrate


class ProviderTests(unittest.TestCase):
    def test_vertex_contract_and_metadata(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
                    {"text": '{"text":"A hall.","fact_ids":["room"]}'}]}}],
                "modelVersion": "resolved-test-model",
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 8, "totalTokenCount": 20},
            })

        original = genai.Client

        def client(**kwargs):
            options = kwargs.pop("http_options")
            self.assertEqual(options.timeout, 20_000)
            self.assertEqual(options.retry_options.attempts, 1)
            options = options.model_copy(update={"client_args": {"transport": httpx.MockTransport(respond)}})
            return original(**kwargs, credentials=Credentials(token="test-only"), http_options=options)

        env = {"COURSE_MODEL": "student-selected-model", "GOOGLE_CLOUD_PROJECT": "test-project",
               "GOOGLE_CLOUD_LOCATION": "us-central1"}
        with patch.dict(os.environ, env, clear=True), patch("google.genai.Client", side_effect=client):
            result = narrate({"room": "A hall."}, "look", "vertex")
        self.assertEqual(result.provider, "vertex")
        self.assertEqual(result.model, "resolved-test-model")
        self.assertEqual(result.total_tokens, 20)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["generationConfig"]["maxOutputTokens"], 1024)

    def invoke(self, provider, body, status=200):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(status, json=body)

        sdk = openai.OpenAI if provider == "openai" else anthropic.Anthropic
        target = "openai.OpenAI" if provider == "openai" else "anthropic.Anthropic"
        env = {"COURSE_MODEL": "student-selected-model", f"{provider.upper()}_API_KEY": "test-only"}
        if provider == "anthropic":
            # Short-circuits the SDK's auto-discovery check before it falls back to
            # pathlib.Path.home(), which raises on Windows once clear=True below
            # strips USERPROFILE/APPDATA from the environment.
            env["ANTHROPIC_CONFIG_DIR"] = "."
        with patch.dict(os.environ, env, clear=True), patch(target) as factory:
            factory.side_effect = lambda **kw: sdk(
                **kw, http_client=httpx.Client(transport=httpx.MockTransport(respond))
            )
            result = narrate({"room": "A hall."}, "look", provider)
            self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
            self.assertEqual(factory.call_args.kwargs["timeout"], 20.0)
        return result, requests

    def openai_body(self, raw=None):
        return {
            "id": "resp_test", "object": "response", "created_at": 0,
            "status": "completed", "model": "resolved-test-model",
            "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                        "text": raw or '{"text":"A hall.","fact_ids":["room"]}', "annotations": []}]}],
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        }

    def anthropic_body(self, raw=None):
        return {
            "id": "msg_test", "type": "message", "role": "assistant",
            "model": "resolved-test-model", "stop_reason": "end_turn", "stop_sequence": None,
            "content": [{"type": "text", "text": raw or '{"text":"A hall.","fact_ids":["room"]}'}],
            "usage": {"input_tokens": 12, "output_tokens": 8,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    def test_openai_contract_and_metadata(self):
        result, requests = self.invoke("openai", self.openai_body())
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.model, "resolved-test-model")
        self.assertEqual(result.total_tokens, 20)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["model"], "student-selected-model")
        self.assertFalse(requests[0]["store"])
        self.assertTrue(requests[0]["text"]["format"]["strict"])
        self.assertEqual(requests[0]["max_output_tokens"], 1024)

    def test_anthropic_contract_and_metadata(self):
        result, requests = self.invoke("anthropic", self.anthropic_body())
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.model, "resolved-test-model")
        self.assertEqual(result.total_tokens, 20)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["model"], "student-selected-model")
        self.assertEqual(requests[0]["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(requests[0]["max_tokens"], 1024)

    def test_all_live_providers_require_only_their_own_configuration(self):
        for provider in ("vertex", "anthropic", "openai"):
            with self.subTest(provider=provider), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(NarrationError):
                    narrate({}, "look", provider)

    def test_invalid_facts_and_json_rejected_for_both_adapters(self):
        for provider, body in (("openai", self.openai_body), ("anthropic", self.anthropic_body)):
            for raw in ('not JSON', '{"text":"A key.","fact_ids":["hidden_key"]}'):
                with self.subTest(provider=provider, raw=raw), self.assertRaises(NarrationError):
                    self.invoke(provider, body(raw))

    def test_truncation_does_not_pass_even_with_valid_json(self):
        body = self.openai_body()
        body["status"] = "incomplete"
        with self.assertRaises(NarrationError):
            self.invoke("openai", body)
        body = self.anthropic_body()
        body["stop_reason"] = "max_tokens"
        with self.assertRaises(NarrationError):
            self.invoke("anthropic", body)

    def test_provider_error_never_falls_back_to_fixture(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider), self.assertRaises(Exception) as caught:
                self.invoke(provider, {"error": {"type": "rate_limit_error", "message": "test"}}, 429)
            self.assertEqual(type(caught.exception).__name__, "RateLimitError")

    def test_setup_requires_explicit_mode_before_diagnostics(self):
        from course_game.setup_check import main
        with patch("sys.argv", ["setup_check"]), patch("subprocess.run") as run, patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as caught:
                main()
            self.assertEqual(caught.exception.code, 2)
            run.assert_not_called()

    def test_interactive_attempt_budget_applies_to_every_live_provider(self):
        from course_game.__main__ import main
        for provider in ("vertex", "anthropic", "openai"):
            with self.subTest(provider=provider), patch("sys.argv", ["game", "--provider", provider]):
                with patch("builtins.input", side_effect=["ask look"] * 7 + ["quit"]):
                    with patch("course_game.__main__.narrate", side_effect=NarrationError("test")) as call:
                        with patch("sys.stdout", new_callable=io.StringIO) as output:
                            main()
                        self.assertEqual(call.call_count, 5)
                        self.assertIn("Live-call budget reached", output.getvalue())


if __name__ == "__main__":
    unittest.main()
