"""Credential-free tests at requests' serialized HTTP transport boundary."""
import ast
import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests


_MODULE_PATH = Path(__file__).with_name("api_caller.py")
_SPEC = importlib.util.spec_from_file_location("api_caller_under_test", _MODULE_PATH)
_API_CALLER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_API_CALLER)
GLOBAL = "https://api.minimax.io/v1/chat/completions"
CHINA = "https://api.minimaxi.com/v1/chat/completions"
FLASH = "MiniMax-M3.1-Flash"
PARAMS = {"temperature": 0.2, "top_p": 1.0, "max_tokens": 128}
MESSAGES = [
    {"role": "assistant", "content": "previous", "reasoning_content": "saved reasoning"},
    {"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]},
]
TOOLS = [{"type": "function", "function": {"name": "lookup"}}]


class MiniMaxApiCallerTest(unittest.TestCase):
    def request(self, model, endpoint=GLOBAL, controls=None, fallback=False, message=None):
        """Let requests prepare real JSON; intercept before any network I/O."""
        captured = []
        env = {
            "REASONING_API_KEY": "credential-free-test",
            "REASONING_END_POINT": endpoint,
            **(controls or {}),
        }
        if fallback:
            env.update(REASONING_API_KEY_2="fallback-test", REASONING_END_POINT_2=CHINA)
        response_message = message or {"content": "answer"}
        original_messages = copy.deepcopy(MESSAGES)

        def send(adapter, request, **kwargs):
            captured.append(request)
            response = requests.Response()
            response.status_code = 500 if fallback and len(captured) == 1 else 200
            body = {"choices": [{"message": response_message}]}
            if response.status_code == 500:
                body = {"error": {"message": "mock temporary failure"}}
            response._content = json.dumps(body).encode()
            response.request = request
            return response

        with patch.dict(os.environ, env, clear=True), patch.object(
            requests.adapters.HTTPAdapter, "send", send
        ):
            result = _API_CALLER.call_vision_api(
                model, MESSAGES, PARAMS, max_retries=1, tools=TOOLS
            )
        self.assertEqual(MESSAGES, original_messages)
        self.assertEqual(result, response_message if len(response_message) > 1 else "answer")
        self.assertEqual(len(captured), 2 if fallback else 1)
        payloads = []
        for request in captured:
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.headers["Content-Type"], "application/json")
            self.assertIsInstance(request.body, bytes)
            payload = json.loads(request.body)
            self.assertEqual(payload["messages"], MESSAGES)
            self.assertEqual(payload["tools"], TOOLS)
            self.assertEqual(payload["max_tokens"], 128)
            payloads.append(payload)
        return captured, payloads

    def test_flash_default_only_for_direct_minimax_endpoints(self):
        for endpoint in (GLOBAL, CHINA, GLOBAL + "/"):
            for model in (None, ""):
                with self.subTest(endpoint=endpoint, model=model):
                    requests_sent, payloads = self.request(model, endpoint)
                    self.assertEqual(payloads[0]["model"], FLASH)
                    self.assertEqual(requests_sent[0].url, endpoint)
        for endpoint in (None, "https://other.invalid/v1/chat/completions",
                         "https://api.minimax.io.evil.invalid/v1/chat/completions",
                         "https://api.minimax.io/v1/images/generations"):
            self.assertIsNone(_API_CALLER.resolve_model_name(None, endpoint))

    def test_all_flash_efforts_are_serialized_without_m3_thinking(self):
        for endpoint in (GLOBAL, CHINA):
            for effort in ("low", "medium", "high", "xhigh", "max"):
                with self.subTest(endpoint=endpoint, effort=effort):
                    _, payloads = self.request(FLASH, endpoint, {
                        "REASONING_EFFORT": effort.upper(), "REASONING_THINKING": "adaptive"
                    })
                    self.assertEqual(payloads[0]["reasoning_effort"], effort)
                    self.assertNotIn("thinking", payloads[0])

    def test_flash_omits_unset_and_invalid_efforts(self):
        for effort in (None, "", "none", "false", "minimal", "unsupported"):
            with self.subTest(effort=effort):
                controls = {} if effort is None else {"REASONING_EFFORT": effort}
                _, payloads = self.request(FLASH, controls=controls)
                self.assertNotIn("reasoning_effort", payloads[0])
                self.assertNotIn("thinking", payloads[0])

    def test_explicit_m3_retains_supported_thinking_modes(self):
        for endpoint in (GLOBAL, CHINA):
            for mode in ("adaptive", "disabled"):
                with self.subTest(endpoint=endpoint, mode=mode):
                    _, payloads = self.request("MiniMax-M3", endpoint, {
                        "REASONING_THINKING": mode.upper(), "REASONING_EFFORT": "high"
                    })
                    self.assertEqual(payloads[0]["model"], "MiniMax-M3")
                    self.assertEqual(payloads[0]["thinking"], {"type": mode})
                    self.assertNotIn("reasoning_effort", payloads[0])
        _, payloads = self.request("MiniMax-M3", controls={"REASONING_THINKING": "enabled"})
        self.assertNotIn("thinking", payloads[0])

    def test_explicit_legacy_custom_and_media_ids_are_preserved(self):
        for model in ("MiniMax-M2.7", "custom-minimax-m3.1-flash", "MiniMax-M3-custom",
                      "music-2.6", "image-01", "saved-provider-model"):
            with self.subTest(model=model):
                _, payloads = self.request(model, controls={
                    "REASONING_EFFORT": "high", "REASONING_THINKING": "adaptive"
                })
                self.assertEqual(payloads[0]["model"], model)
                self.assertNotIn("reasoning_effort", payloads[0])
                self.assertNotIn("thinking", payloads[0])

    def test_other_provider_and_router_behavior_is_preserved(self):
        _, payloads = self.request("gpt-5", "https://other.invalid/v1/chat/completions", {
            "REASONING_EFFORT": "high"
        })
        self.assertEqual(payloads[0]["model"], "gpt-5")
        self.assertEqual(payloads[0]["reasoning_effort"], "high")
        for model in (FLASH, "minimax/minimax-m3.1-flash", "MiniMax-M3"):
            _, payloads = self.request(model, "https://router.invalid/v1/chat/completions", {
                "REASONING_EFFORT": "high", "REASONING_THINKING": "adaptive"
            })
            self.assertEqual(payloads[0]["model"], model)
            self.assertNotIn("reasoning_effort", payloads[0])
            self.assertNotIn("thinking", payloads[0])

    def test_round_robin_preserves_same_selected_model(self):
        for model in (None, "MiniMax-M3", "saved-custom"):
            with self.subTest(model=model):
                captured, payloads = self.request(model, fallback=True)
                self.assertEqual([r.url for r in captured], [GLOBAL, CHINA])
                self.assertEqual([p["model"] for p in payloads], [model or FLASH] * 2)

    def test_response_reasoning_and_tool_calls_are_preserved(self):
        self.request(FLASH, message={"content": "answer", "reasoning_content": "reasoning"})
        self.request(FLASH, message={"content": "answer", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]})

    def test_cli_initialization_uses_same_default_and_preserves_other_validation(self):
        # Execute the real main initialization up to the first progress print,
        # avoiding unrelated image/torch dependencies and inference execution.
        path = _MODULE_PATH.parents[1] / "infer_api.py"
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        prefix = []
        for node in main.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
                    break
            prefix.append(node)
        code = compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec")
        for endpoint, model, expected in ((GLOBAL, "", FLASH), (CHINA, "MiniMax-M3", "MiniMax-M3"),
                                          (GLOBAL, "legacy", "legacy")):
            args = SimpleNamespace()
            with patch.dict(os.environ, {"REASONING_END_POINT": endpoint,
                                        "REASONING_MODEL_NAME": model}, clear=True):
                exec(code, {"os": os, "args": args, "resolve_model_name": _API_CALLER.resolve_model_name})
            self.assertEqual(args.model_name, expected)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            exec(code, {"os": os, "args": SimpleNamespace(),
                        "resolve_model_name": _API_CALLER.resolve_model_name})


if __name__ == "__main__":
    unittest.main()
