"""Configuration loading and validation for the RAG workshop."""

# dataclass gives us one typed object for all settings; frozen=True prevents
# accidental changes to endpoints or credentials after the client is created.
from dataclasses import dataclass
# os.environ lets deployments override the local .env file without editing code.
import os
# Path makes file paths work consistently across Windows, macOS, and Linux.
from pathlib import Path
# urlsplit separates URL components so we can validate host, scheme, and routes.
from urllib.parse import urlsplit

# dotenv_values reads key/value pairs without printing them or changing the
# process-wide environment. This makes settings loading explicit and testable.
from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    """Validated endpoint and model configuration passed to the RAG client."""

    # The chat-compatible endpoint generates natural-language answers.
    chat_endpoint: str
    # The embedding endpoint converts passages and questions to vectors.
    embed_endpoint: str
    # Kept in memory only and sent as an Authorization header by the HTTP client.
    hf_token: str
    # These identifiers must agree with the models deployed behind the URLs.
    chat_model: str = "Qwen/Qwen3-1.7B"
    embed_model: str = "google/embeddinggemma-300m"


def validate_endpoint_url(value: str, variable_name: str) -> str:
    """Validate and normalize a Hugging Face dedicated endpoint base URL."""
    # Reject blank or non-text configuration before attempting URL parsing.
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{variable_name} must be a Hugging Face endpoint base URL.")

    # Remove surrounding whitespace and a trailing slash so URL joining later
    # produces one consistent base URL.
    value = value.strip().rstrip("/")
    try:
        # Parse rather than inspect strings: this lets us reject credentials,
        # query parameters, and unsupported ports explicitly.
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise ValueError(
            f"{variable_name} must be a valid HTTPS endpoint URL."
        ) from None

    # Dedicated endpoints must use HTTPS and Hugging Face's endpoint host.
    # User-info, query strings, fragments, and non-default ports are rejected
    # because they can leak credentials or point to unintended destinations.
    if (
        parts.scheme != "https"
        or not parts.hostname
        or not parts.hostname.endswith(".endpoints.huggingface.cloud")
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or port not in (None, 443)
    ):
        raise ValueError(
            f"{variable_name} must be the HTTPS endpoint URL copied from "
            "Hugging Face, without credentials or query parameters."
        )

    # The setting must be the endpoint root; API routes are added by hf_client.
    if parts.path.rstrip("/").endswith(
        ("/v1", "/chat/completions", "/embed", "/embeddings")
    ):
        raise ValueError(
            f"{variable_name} must be the base endpoint URL, without the API route."
        )

    return value


def load_settings(env_file: Path | None = None) -> Settings:
    """Load required settings, preferring process environment over the .env file."""
    # By default find .env relative to this package, not to the notebook's current
    # working directory. Tests and other callers can inject another path.
    if env_file is None:
        env_file = Path(__file__).resolve().parents[1] / ".env"
    else:
        env_file = Path(env_file)

    # Read the local file if present; an absent file is handled by the same clear
    # missing-setting validation as an empty file.
    file_values = dotenv_values(env_file) if env_file.is_file() else {}
    # Process-level values take precedence, which supports notebooks, CI, and
    # hosted deployments without requiring a local .env file.
    values = {
        key: os.environ.get(key, file_values.get(key))
        for key in ("HF_CHAT_ENDPOINT", "HF_EMBED_ENDPOINT", "HF_TOKEN")
    }
    # Report only variable names. Never include the configured token or URL value
    # in a configuration error.
    missing = [key for key, value in values.items() if value is None or not value.strip()]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Missing required settings: {names}. Copy .env.example to .env "
            "and fill in the values supplied by the instructor."
        )

    # Keep whitespace copied around the token out of the actual request header.
    token = values["HF_TOKEN"].strip()
    if not token.startswith("hf_"):
        raise ValueError("HF_TOKEN must be the Hugging Face token supplied by the instructor.")

    # Validate URLs at startup so a misconfigured base URL fails before any
    # network request is attempted.
    return Settings(
        chat_endpoint=validate_endpoint_url(values["HF_CHAT_ENDPOINT"], "HF_CHAT_ENDPOINT"),
        embed_endpoint=validate_endpoint_url(values["HF_EMBED_ENDPOINT"], "HF_EMBED_ENDPOINT"),
        hf_token=token,
    )
