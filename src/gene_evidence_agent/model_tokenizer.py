"""Prompt token estimates with a pinned profile and a generic conservative fallback."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib.resources import files
from typing import Protocol

from tokenizers import Tokenizer

from gene_evidence_agent.vendor.deepseek_v4_0e1a0e5.encoding_dsv4 import encode_messages


MODEL_REPOSITORY = "deepseek-ai/DeepSeek-V4-Pro"
MODEL_REPOSITORY_REVISION = "0e1a0e5e52aea73055f50fef6f2423db370265b6"
KNOWN_MODEL_IDENTITIES = {
    "ali-dashscope/qwen3.8-flash": "Qwen/Qwen3.8-Flash-Next",
}
SUPPORTED_MODEL_ALIASES = frozenset(
    {
        "deepseek-v4-pro",
        "bytedance-volcengine/deepseek-v4-pro",
    }
)
_TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
_UNKNOWN_MODEL_TEMPLATE_RESERVE = 1024
_UNKNOWN_MODEL_MESSAGE_RESERVE = 64


class MessageLike(Protocol):
    role: str
    content: str


class ModelTokenizerError(ValueError):
    """Raised when no verified local tokenizer profile can count a prompt."""


def has_verified_tokenizer_profile(model_name: str) -> bool:
    """Return whether ``model_name`` has a locally pinned tokenizer profile."""

    return isinstance(model_name, str) and model_name.strip().casefold() in SUPPORTED_MODEL_ALIASES


def canonical_model_identity(model_name: str) -> str | None:
    """Return a known public identity without implying a verified tokenizer."""

    if not isinstance(model_name, str):
        return None
    return KNOWN_MODEL_IDENTITIES.get(model_name.strip().casefold())


@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer:
    tokenizer_path = files("gene_evidence_agent.vendor.deepseek_v4_0e1a0e5").joinpath("tokenizer.json")
    tokenizer_bytes = tokenizer_path.read_bytes()
    digest = hashlib.sha256(tokenizer_bytes).hexdigest()
    if digest != _TOKENIZER_SHA256:
        raise ModelTokenizerError("Bundled DeepSeek-V4-Pro tokenizer failed its SHA-256 integrity check")
    try:
        return Tokenizer.from_file(str(tokenizer_path))
    except Exception as error:
        raise ModelTokenizerError("Bundled DeepSeek-V4-Pro tokenizer could not be loaded") from error


def estimate_prompt_input_tokens(model_name: str, messages: list[MessageLike]) -> int:
    """Estimate visible chat input conservatively for any configured model alias.

    The pinned V4 profile counts the exact prompt in both supported modes and
    reserves the larger count, even if the caller configures one mode. For a
    model without a locally pinned tokenizer profile, the UTF-8 byte length of
    visible roles and content is used as an upper bound on text tokens, with
    additional fixed headroom for chat-template framing. This fallback is
    intentionally an estimate, not a verified model tokenizer. Undisclosed
    gateway-injected system text cannot be counted locally in either mode.
    """

    if not messages or any(message.role not in {"system", "user", "assistant"} for message in messages):
        raise ModelTokenizerError("Prompt preflight only supports non-empty text chat messages")
    if any(not isinstance(message.content, str) for message in messages):
        raise ModelTokenizerError("Prompt preflight only supports text chat messages")

    if not has_verified_tokenizer_profile(model_name):
        return (
            _UNKNOWN_MODEL_TEMPLATE_RESERVE
            + sum(
                len(message.role.encode("utf-8"))
                + len(message.content.encode("utf-8"))
                + _UNKNOWN_MODEL_MESSAGE_RESERVE
                for message in messages
            )
        )

    prompt_messages = [{"role": message.role, "content": message.content} for message in messages]
    tokenizer = _load_tokenizer()
    counts: list[int] = []
    try:
        for thinking_mode in ("chat", "thinking"):
            prompt = encode_messages(prompt_messages, thinking_mode=thinking_mode)
            encoding = tokenizer.encode(prompt, add_special_tokens=False)
            if not encoding.ids:
                raise ValueError("rendered prompt produced no token IDs")
            counts.append(len(encoding.ids))
    except Exception as error:
        raise ModelTokenizerError("DeepSeek-V4-Pro prompt could not be encoded") from error
    return max(counts)
