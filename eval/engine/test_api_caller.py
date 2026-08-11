import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


_MODULE_PATH = Path(__file__).with_name("api_caller.py")
_SPEC = importlib.util.spec_from_file_location("api_caller_under_test", _MODULE_PATH)
_API_CALLER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_API_CALLER)
_add_reasoning_param = _API_CALLER._add_reasoning_param
_parse_api_response = _API_CALLER._parse_api_response


class _Response:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class MiniMaxApiCallerTest(unittest.TestCase):
    def test_adds_supported_thinking_modes_for_minimax_m3(self):
        for mode in ("adaptive", "disabled"):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"REASONING_THINKING": mode.upper()}
            ):
                payload = {}
                _add_reasoning_param(payload, "MiniMax-M3")

                self.assertEqual(payload["thinking"], {"type": mode})

    def test_does_not_override_always_on_thinking_for_minimax_m27(self):
        with patch.dict(os.environ, {"REASONING_THINKING": "disabled"}):
            payload = {}
            _add_reasoning_param(payload, "MiniMax-M2.7")

        self.assertNotIn("thinking", payload)

    def test_ignores_unsupported_minimax_m3_thinking_mode(self):
        with patch.dict(os.environ, {"REASONING_THINKING": "high"}):
            payload = {}
            _add_reasoning_param(payload, "MiniMax-M3")

        self.assertNotIn("thinking", payload)

    def test_preserves_reasoning_content_in_response_message(self):
        message = {"content": "answer", "reasoning_content": "reasoning"}
        response = _Response({"choices": [{"message": message}]})

        result, is_429, error_type = _parse_api_response(response, "MiniMax")

        self.assertEqual(result, message)
        self.assertFalse(is_429)
        self.assertIsNone(error_type)


if __name__ == "__main__":
    unittest.main()
