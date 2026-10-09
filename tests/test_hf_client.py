import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from rag.config import Settings
from rag.hf_client import HuggingFaceClient


CHAT_URL = "https://chat.endpoints.huggingface.cloud"
EMBED_URL = "https://embed.endpoints.huggingface.cloud"
TOKEN = "hf_test_token_do_not_print"


def make_settings() -> Settings:
    return Settings(CHAT_URL, EMBED_URL, TOKEN)


class FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None, json_error=None):
        self.status_code = status_code
        self.body = {} if body is None else body
        self.headers = {} if headers is None else headers
        self.json_error = json_error
        self.closed = False

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.body

    def close(self):
        self.closed = True


class RequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = Mock(spec=requests.Session)
        self.client = HuggingFaceClient(make_settings(), session=self.session)

    def test_get_request_uses_bearer_token_and_disables_redirects(self) -> None:
        self.session.request.return_value = FakeResponse(body={"data": []})

        result = self.client.request(CHAT_URL, "/v1/models")

        self.assertEqual(result, {"data": []})
        self.session.request.assert_called_once()
        args, kwargs = self.session.request.call_args
        self.assertEqual(args[:2], ("GET", CHAT_URL + "/v1/models"))
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {TOKEN}")
        self.assertFalse(kwargs["allow_redirects"])

    def test_post_request_passes_json_payload(self) -> None:
        self.session.request.return_value = FakeResponse(body=[[0.1, 0.2]])
        payload = {"inputs": ["hello"], "truncate": False, "normalize": True}

        result = self.client.request(EMBED_URL, "/embed", payload)

        self.assertEqual(result, [[0.1, 0.2]])
        args, kwargs = self.session.request.call_args
        self.assertEqual(args[:2], ("POST", EMBED_URL + "/embed"))
        self.assertEqual(kwargs["json"], payload)

    def test_rejects_unknown_endpoint_before_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown workshop endpoint"):
            self.client.request("https://other.endpoints.huggingface.cloud", "/embed")
        self.session.request.assert_not_called()

    def test_retries_temporary_failure_then_returns_json(self) -> None:
        busy = FakeResponse(status_code=503)
        success = FakeResponse(body={"ready": True})
        self.session.request.side_effect = [busy, success]

        with patch("rag.hf_client.sleep") as sleep, patch(
            "rag.hf_client.random.uniform", return_value=0
        ):
            result = self.client.request(CHAT_URL, "/v1/models")

        self.assertEqual(result, {"ready": True})
        self.assertEqual(self.session.request.call_count, 2)
        self.assertTrue(busy.closed)
        sleep.assert_called_once_with(5)

    def test_uses_numeric_retry_after_as_lower_bound(self) -> None:
        self.session.request.side_effect = [
            FakeResponse(status_code=429, headers={"Retry-After": "11"}),
            FakeResponse(body={"ready": True}),
        ]

        with patch("rag.hf_client.sleep") as sleep, patch(
            "rag.hf_client.random.uniform", return_value=0
        ):
            self.client.request(EMBED_URL, "/info")

        sleep.assert_called_once_with(11)

    def test_cold_start_can_succeed_after_more_than_six_attempts(self) -> None:
        clock = [0.0]
        self.session.request.side_effect = [FakeResponse(status_code=503) for _ in range(7)] + [
            FakeResponse(body={"ready": True})
        ]

        def advance(seconds):
            clock[0] += seconds

        with patch("rag.hf_client.monotonic", side_effect=lambda: clock[0]), patch(
            "rag.hf_client.sleep", side_effect=advance
        ), patch("rag.hf_client.random.uniform", return_value=0):
            result = self.client.request(CHAT_URL, "/v1/models")

        self.assertEqual(result, {"ready": True})
        self.assertEqual(self.session.request.call_count, 8)
        self.assertGreater(clock[0], 120)

    def test_stops_at_retry_deadline_and_names_the_unavailable_endpoint(self) -> None:
        clock = [0.0]
        self.session.request.return_value = FakeResponse(status_code=503)

        def advance(seconds):
            clock[0] += seconds

        with patch("rag.hf_client.monotonic", side_effect=lambda: clock[0]), patch(
            "rag.hf_client.sleep", side_effect=advance
        ) as sleep, patch("rag.hf_client.random.uniform", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "Chat endpoint unavailable.*HTTP 503"):
                self.client.request(CHAT_URL, "/v1/models", retry_window=12)

        self.assertEqual(self.session.request.call_count, 2)
        self.assertEqual(clock[0], 12)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 7])

    def test_rejects_unbounded_retry_windows_without_sending_a_request(self) -> None:
        for window in (0, -1, float("inf"), float("nan")):
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.client.request(CHAT_URL, "/v1/models", retry_window=window)
        self.session.request.assert_not_called()

    def test_authentication_failure_does_not_disclose_token(self) -> None:
        self.session.request.return_value = FakeResponse(status_code=401)

        with self.assertRaises(RuntimeError) as raised:
            self.client.request(CHAT_URL, "/v1/models")

        self.assertNotIn(TOKEN, str(raised.exception))

    def test_invalid_json_is_reported_without_response_body(self) -> None:
        self.session.request.return_value = FakeResponse(
            json_error=ValueError("body includes secrets")
        )

        with self.assertRaisesRegex(RuntimeError, "invalid JSON") as raised:
            self.client.request(CHAT_URL, "/v1/models")

        self.assertNotIn("body includes secrets", str(raised.exception))

    def test_connection_failure_is_reported_without_details(self) -> None:
        self.session.request.side_effect = requests.Timeout("private transport detail")

        with self.assertRaises(RuntimeError) as raised:
            self.client.request(CHAT_URL, "/v1/models")

        self.assertNotIn("private transport detail", str(raised.exception))
        self.assertNotIn(TOKEN, str(raised.exception))

    def test_embed_validates_and_returns_float32_matrix(self) -> None:
        self.session.request.return_value = FakeResponse(body=[[1, 0], [0, 1]])

        result = self.client.embed(["first", "second"])

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(self.session.request.call_args.kwargs["json"], {
            "inputs": ["first", "second"],
            "truncate": False,
            "normalize": True,
        })

    def test_embed_rejects_invalid_shape(self) -> None:
        self.session.request.return_value = FakeResponse(body=[[1, 0]])

        with self.assertRaises(ValueError):
            self.client.embed(["first", "second"])

    def test_check_models_confirms_both_expected_models(self) -> None:
        self.session.request.side_effect = [
            FakeResponse(body={"data": [{"id": "Qwen/Qwen3-1.7B"}]}),
            FakeResponse(body={
                "model_id": "google/embeddinggemma-300m",
                "model_sha": "revision-123",
            }),
        ]

        result = self.client.check_models()

        self.assertEqual(result["embedding_revision"], "revision-123")
        self.assertEqual(result["embedding_model"], "google/embeddinggemma-300m")
        self.assertTrue(result["embedding_model_verified"])
        self.assertEqual(self.session.request.call_count, 2)

    def test_managed_tei_mount_is_connected_but_identity_is_not_verified(self) -> None:
        self.session.request.side_effect = [
            FakeResponse(body={"data": [{"id": "Qwen/Qwen3-1.7B"}]}),
            FakeResponse(body={
                "model_id": "/repository",
                "model_sha": None,
                "model_type": {"embedding": {"pooling": "mean"}},
            }),
        ]

        result = self.client.check_models()

        self.assertEqual(result["embedding_model"], "/repository")
        self.assertFalse(result["embedding_model_verified"])
        self.assertIsNone(result["embedding_revision"])

    def test_check_models_still_rejects_wrong_or_unidentified_embedding_models(self) -> None:
        for info in (
            {"model_id": "another/model", "model_type": {"embedding": {}}},
            {"model_id": "/repository", "model_type": {"reranker": {}}},
            {"model_id": "/repository"},
            {"model_type": {"embedding": {}}},
        ):
            with self.subTest(info=info):
                self.session.request.side_effect = [
                    FakeResponse(body={"data": [{"id": "Qwen/Qwen3-1.7B"}]}),
                    FakeResponse(body=info),
                ]
                with self.assertRaisesRegex(RuntimeError, "different embedding model"):
                    self.client.check_models()

    def test_check_models_rejects_wrong_generation_model(self) -> None:
        self.session.request.return_value = FakeResponse(
            body={"data": [{"id": "another/model"}]}
        )

        with self.assertRaisesRegex(RuntimeError, "different generation model"):
            self.client.check_models()

    def test_generate_returns_answer_and_usage(self) -> None:
        self.session.request.return_value = FakeResponse(body={
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "The lab closes at 5 p.m. [hours_0000]"},
            }],
            "usage": {"completion_tokens": 12, "prompt_tokens": 80},
        })
        passages = [{
            "id": "hours_0000",
            "title": "Opening hours",
            "text": "Open until 5 p.m.",
        }]

        with patch("rag.hf_client.perf_counter", side_effect=[10.0, 11.5]):
            result = self.client.generate("When does it close?", passages)

        self.assertEqual(result["answer"], "The lab closes at 5 p.m. [hours_0000]")
        self.assertEqual(result["generation_s"], 1.5)
        self.assertEqual(result["tokens_s"], 8.0)
        self.assertEqual(result["generated_tokens"], 12)
        self.assertEqual(result["prompt_tokens"], 80)
        payload = self.session.request.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "Qwen/Qwen3-1.7B")
        self.assertIn("[hours_0000]", payload["messages"][1]["content"])

    def test_generate_rejects_truncated_answers(self) -> None:
        self.session.request.return_value = FakeResponse(body={
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "partial"},
            }]
        })

        with self.assertRaisesRegex(RuntimeError, "Truncated answer"):
            self.client.generate("question", [])


if __name__ == "__main__":
    unittest.main()
