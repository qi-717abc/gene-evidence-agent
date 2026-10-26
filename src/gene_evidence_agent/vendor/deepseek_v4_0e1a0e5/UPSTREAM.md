# Pinned upstream assets

These assets are copied from DeepSeek's official `deepseek-ai/DeepSeek-V4-Pro`
Hugging Face repository at immutable revision
`0e1a0e5e52aea73055f50fef6f2423db370265b6` (Release DeepSeek-V4).

- `tokenizer.json` — official V4-Pro tokenizer.
- `encoding_dsv4.py` — official message encoder for DeepSeek-V4 chat prompts.
- `LICENSE` — upstream MIT license and copyright notice.

SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `tokenizer.json` | `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf` |
| `encoding_dsv4.py` | `bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0` |
| `LICENSE` | `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` |

The adapter counts locally rendered DeepSeek-V4 prompts in both `chat` and
`thinking` modes and reserves the larger count because SDU has not documented
the default mode. This cannot account for undisclosed text injected by a
gateway; provider-reported usage remains authoritative after a response.

Upstream references:

- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/0e1a0e5e52aea73055f50fef6f2423db370265b6/encoding>
- <https://github.com/deepseek-ai/deepseek-recipe/blob/main/static/tokenizers/v4/README.md>
