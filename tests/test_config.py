import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rag.config import Settings, load_settings, validate_endpoint_url


CHAT_URL = "https://chat.endpoints.huggingface.cloud"
EMBED_URL = "https://embed.endpoints.huggingface.cloud"
TOKEN = "hf_test_token_do_not_print"


class LoadSettingsTests(unittest.TestCase):
    def write_env(self, directory: str, contents: str) -> Path:
        env_file = Path(directory) / ".env"
        env_file.write_text(contents, encoding="utf-8")
        return env_file

    def test_loads_settings_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = self.write_env(
                directory,
                f"HF_CHAT_ENDPOINT={CHAT_URL}\n"
                f"HF_EMBED_ENDPOINT={EMBED_URL}\n"
                f"HF_TOKEN={TOKEN}\n",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(env_file)

        self.assertEqual(
            settings,
            Settings(
                chat_endpoint=CHAT_URL,
                embed_endpoint=EMBED_URL,
                hf_token=TOKEN,
            ),
        )

    def test_process_environment_overrides_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = self.write_env(
                directory,
                f"HF_CHAT_ENDPOINT={CHAT_URL}\n"
                f"HF_EMBED_ENDPOINT={EMBED_URL}\n"
                f"HF_TOKEN={TOKEN}\n",
            )

            with patch.dict(
                os.environ,
                {"HF_CHAT_ENDPOINT": "https://override.endpoints.huggingface.cloud"},
                clear=True,
            ):
                settings = load_settings(env_file)

        self.assertEqual(
            settings.chat_endpoint,
            "https://override.endpoints.huggingface.cloud",
        )

    def test_missing_settings_name_variables_without_echoing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = self.write_env(directory, f"HF_TOKEN={TOKEN}\n")

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError) as raised:
                    load_settings(env_file)

        message = str(raised.exception)
        self.assertIn("HF_CHAT_ENDPOINT", message)
        self.assertIn("HF_EMBED_ENDPOINT", message)
        self.assertNotIn(TOKEN, message)

    def test_rejects_token_without_hugging_face_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = self.write_env(
                directory,
                f"HF_CHAT_ENDPOINT={CHAT_URL}\n"
                f"HF_EMBED_ENDPOINT={EMBED_URL}\n"
                "HF_TOKEN=not-a-valid-token\n",
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
                    load_settings(env_file)

    def test_loading_does_not_print_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = self.write_env(
                directory,
                f"HF_CHAT_ENDPOINT={CHAT_URL}\n"
                f"HF_EMBED_ENDPOINT={EMBED_URL}\n"
                f"HF_TOKEN={TOKEN}\n",
            )
            output = StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
                load_settings(env_file)

        self.assertNotIn(TOKEN, output.getvalue())


class ValidateEndpointUrlTests(unittest.TestCase):
    def test_accepts_hugging_face_endpoint_base_url(self) -> None:
        self.assertEqual(validate_endpoint_url(CHAT_URL, "HF_CHAT_ENDPOINT"), CHAT_URL)
        self.assertEqual(
            validate_endpoint_url(CHAT_URL + "/", "HF_CHAT_ENDPOINT"),
            CHAT_URL,
        )

    def test_rejects_unsafe_or_non_base_urls(self) -> None:
        invalid_urls = (
            "http://chat.endpoints.huggingface.cloud",
            "https://not-huggingface.example",
            "https://user:pass@chat.endpoints.huggingface.cloud",
            CHAT_URL + "?token=secret",
            CHAT_URL + "/embed",
            CHAT_URL + "/v1",
        )

        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_endpoint_url(url, "HF_CHAT_ENDPOINT")


if __name__ == "__main__":
    unittest.main()
