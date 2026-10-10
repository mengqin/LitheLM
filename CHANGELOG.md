# Changelog

## Unreleased

- Reject composite-loss updates without a globally present, nonzero-weight component before model or optimizer execution.
- Verify configured frozen extension file contents on every rank during initialization and resume preflight.

## 0.1.0 - 2026-10-07

Initial release.

### Initial feature set

- Configurable causal language models with attention, custom Mixers, and optional FLA GatedDeltaNet layers.
- PT/SFT stages, dataset mixtures and replay, WSD schedules, curriculum, and checkpoint recovery.
- Shared corpora and tokenizers, optional self-contained runs, and environment provenance.
- Shifted and next-token targets, response masking, and token-weighted loss/evaluation.
- Custom scalar statistics with microbatch and distributed aggregation; stdout, JSONL, CSV, TensorBoard, and W&B sinks.
- CPU and single-GPU examples, a 7.4M full training course, and CPU/CUDA regression tests.
- Apache-2.0 licensing and source-installation documentation.

### Incremental inference

- Prompt prefill and single-token cached decoding with SDPA, Flash, and Triton attention.
- MHA/GQA/MQA caches that store unexpanded KV heads, with learned-position and RoPE offsets.
- Preallocated contiguous K/V buffers with in-place appends and configurable total context capacity through `inference.max_context_length` or `lithelm infer --max-context-length`.
- Per-layer Attention and FLA state through `ModelCache`, including hybrid models.
- Greedy and sampled generation with temperature, top-k/top-p filtering, and EOS or context-limit stopping.

### Data preparation and training controls

- Byte-level BPE tokenizer training from plain text, prompt/response pairs, messages, and TinyStoriesInstruct records.
- Shared token caches with source metadata tracking and file locks for concurrent preparation.
- Explicit EOS/BOS/PAD roles and optional SFT EOS supervision.
- Separate input-token, target-token, and sample throughput metrics that exclude evaluation, checkpointing, and other maintenance work.
- Bounded retries for AMP-skipped optimizer updates, configuration validation, and deterministic CUDA execution.
