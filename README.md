# LitheLM

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

LitheLM is a configuration-driven framework for training causal language models and testing model architectures. You define model layers, datasets, and training stages in YAML. PyTorch runs the model; Hugging Face Accelerate handles device placement, mixed precision, and distributed execution.

Use LitheLM to train a model from scratch, continue pretraining, run supervised fine-tuning with pretraining replay, or compare a custom Mixer against an attention baseline. The same training loop handles these workflows. The repository includes a small CPU example and a complete [7.4M-parameter, single-GPU training example](examples/tlm_7m/README.md).

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Data and target protocols](#data-and-target-protocols)
- [Runs and checkpoints](#runs-and-checkpoints)
- [Reference inference](#reference-inference)
- [Custom Mixers and statistics](#custom-mixers-and-statistics)
- [GPU and distributed execution](#gpu-and-distributed-execution)
- [Project structure](#project-structure)
- [Testing and contributing](#testing-and-contributing)
- [License](#license)

## Features

| Area | Support |
|---|---|
| Model architecture | Per-layer configuration, width changes with projections, pre/post normalization, LayerNorm/RMSNorm, standard/gated FFNs, tied embeddings, learned positions, and RoPE |
| Mixers | PyTorch attention, custom Python modules, and optional FLA GatedDeltaNet layers in the same model |
| Attention operators | SDPA, a forced PyTorch Flash backend, a Triton research implementation, and custom operators |
| Training | PT and SFT stages, weighted dataset mixtures, PT replay, gradient accumulation and clipping, AdamW, and warmup-stable-decay schedules |
| Curriculum | Changes to sequence length and dataset weights at configured stage steps |
| Data | Text, JSON/JSONL, and Hugging Face datasets; byte-level BPE or an existing HF fast tokenizer; memory-mapped token caches |
| Loss and evaluation | Two target protocols, response masking, token-weighted NLL/PPL, token accuracy, and teacher-forced exact match |
| Execution | CPU, single GPU, Accelerate DDP, fp32/fp16/bf16, and optional `torch.compile` |
| Inference | Prompt prefill and cached single-token decoding for built-in attention and FLA, including hybrid models |
| Experiment records | Checkpoints with optimizer/scheduler/RNG state, shared input references, environment metadata, JSONL/CSV logs, and optional TensorBoard/W&B |

## Installation

Use Python 3.10 or newer in a virtual environment. Install a [PyTorch build for your hardware](https://pytorch.org/get-started/locally/) first, then run these commands from the repository root, which contains `pyproject.toml`:

```bash
python -m pip install -e ".[test,examples]"
lithelm doctor
```

For a runtime installation, use `python -m pip install -e .`. You can also install the runtime dependencies through `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Package metadata reads the same requirements file. It declares minimum versions; pin your installed versions or use a lockfile for experiments that require reproducible environments.

| Extra | Install for |
|---|---|
| `test` | Pytest and the regression suite |
| `examples` | Matplotlib for the TLM training chart |
| `triton-linux` | The research attention operator on Linux/CUDA |
| `fla` | FLA GatedDeltaNet on Linux |
| `tensorboard`, `wandb` | The corresponding logging sink |

For example, `python -m pip install -e ".[triton-linux,fla]"` adds both optional GPU dependencies. Choose versions that work with your PyTorch/CUDA environment. LitheLM records their versions and passes FLA options to FLA without a compatibility matrix.

`python -m lithelm` exposes the same commands as `lithelm`. Run `lithelm --help` or `lithelm train --help` for command syntax. The shell examples below use Bash.

## Quick start

### Train on CPU

The synthetic example creates its own small text and instruction datasets, trains a 280-token BPE tokenizer, and runs six PT updates followed by four SFT updates with PT replay. It requires no corpus download.

From the repository root:

```bash
LITHELM_REPO="$PWD"
mkdir -p ../lithelm-experiments
cd ../lithelm-experiments

RUN="$(python "$LITHELM_REPO/examples/synthetic/prepare.py")"
lithelm train "$RUN/resolved_config.yaml"
lithelm eval "$RUN/resolved_config.yaml" --checkpoint "$RUN/sft/final"
lithelm infer "$RUN/resolved_config.yaml" --checkpoint "$RUN/sft/final" \
  --prompt "A happy cat" --max-new-tokens 20
```

The preparation script prints the new run path. It places shared inputs in `corpora/` and `tokenizers/`, and training outputs in `outputs/synthetic_26.7K_YYYY-MM-DD/`, under your experiment directory.

### Train the 7.4M model

From an experiment directory, run:

```bash
CUDA_VISIBLE_DEVICES=0 python "$LITHELM_REPO/scripts/run_tlm.py"
```

The default course uses TinyStories and TinyStoriesInstruct: 60,000 PT updates followed by 4,000 SFT updates on one GPU. The runner prepares data, trains, exercises checkpoint recovery, evaluates four checkpoints, generates text, and plots the training curve.

Use `--smoke` for a 20 PT + 20 SFT update check with the same model architecture and bounded corpus slices. See the [TLM guide](examples/tlm_7m/README.md) for data requirements, resume commands, and memory settings.

## Configuration

Start with [configs/base.yaml](configs/base.yaml) or [configs/examples/tlm_7m.yaml](configs/examples/tlm_7m.yaml). [lithelm/config.py](lithelm/config.py) defines the defaults and validation rules.

| Section | Purpose |
|---|---|
| `experiment` | Run name, seed, optional size/date labels, and an existing output directory |
| `paths`, `workspace` | Shared input roots, output root, and optional input copying |
| `runtime`, `distributed` | Device, precision, compilation, and launch strategy |
| `tokenizer`, `datasets` | Tokenizer artifact and the dataset catalog |
| `model` | Embeddings, layer defaults, layer overrides, and output head |
| `stages` | Objectives, batches, mixtures, curriculum, optimizer, schedule, evaluation, and checkpoints |
| `targets`, `statistics`, `logging` | Target alignment, scalar aggregation, and log sinks |
| `inference` | Context capacity, generation length, sampling, temperature, top-k, and top-p |

### Model layers

Set shared values in `model.layer_defaults`, then override them on individual layers. `repeat` expands a layer specification. This fragment defines eight transformer layers with grouped-query attention and gated FFNs:

```yaml
model:
  vocab_size: 4096
  hidden_size: 256
  max_position_embeddings: 256
  position_embedding: {type: rope}
  layer_defaults:
    norm: {type: rmsnorm}
    mixer:
      attention: {impl: sdpa, num_heads: 8, num_kv_heads: 2}
    ffn: {type: gated, hidden_size: 1024, activation: silu}
  layers:
    - {type: transformer, repeat: 8}
```

Use `auto_project_width_changes` to insert projections between layers with different widths. A custom Mixer or GDN layer can replace attention in a layer while retaining the block's normalization, residual path, and optional FFN.

### Training stages

Define a dataset once in `datasets`, then reference its name from a stage. This fragment shows PT followed by a 90/10 SFT and PT mixture:

```yaml
stages:
  - name: pt
    type: pt
    max_steps: 1000
    data:
      datasets: [{name: stories, weight: 1}]
    lr_schedule: {warmup_steps: 100, decay_steps: 100}
  - name: sft
    type: sft
    max_steps: 100
    data:
      datasets:
        - {name: instructions, weight: 0.9}
        - {name: stories, objective: pt, weight: 0.1}
```

These are configuration fragments; supply dataset definitions and the remaining model/tokenizer settings through a complete config.

At a stage boundary, training continues with the current model weights and a fresh optimizer/scheduler. `stage_step` resets; `global_step` continues. Use `init_from` to select a preceding stage's `best`, `final`, or numbered checkpoint and to control optimizer/scheduler reuse.

Validation rejects a `final` dependency if its source disables `export_final`, and a `best` dependency without a matching evaluation dataset/metric. Optimizer and scheduler reuse require their saved state. Keep any numbered checkpoint referenced by a later stage available through that transition.

WSD supports linear or cosine decay. The warmup, stable, and decay lengths must sum to `max_steps`; omit `stable_steps` to derive it. Curriculum entries use `until_step` to change length or mixture weights within a stage. Update these boundaries with the schedule if you change the course length.

An AMP-skipped optimizer step retries the same sampled batches without advancing the step or scheduler. `runtime.max_optimizer_step_retries` defaults to `8`: the eighth consecutive skipped attempt raises an error with stage, step, retry count, and precision. A successful update resets the counter.

For BF16 and FP32, the trainer checks the gradient norm after clipping and before updating parameters. If any rank reports NaN or Inf, all ranks raise `FloatingPointError` with the stage, stage/global steps, precision, and local gradient norm. FP16 retains GradScaler's skipped-step retry behavior.

### Command-line overrides

Use repeated `--set KEY=VALUE` arguments, including numeric list indices:

```bash
lithelm train experiment.yaml \
  --set runtime.precision=bf16 \
  --set stages.0.batch.micro_batch_size=4 \
  --set stages.0.batch.grad_accum_steps=8
```

Bare `KEY=VALUE` arguments also work. LitheLM resolves OmegaConf interpolation, validates the result, and writes `resolved_config.yaml` to the run. Unknown core fields raise an error; extension `kwargs` remain open to module-specific options. Existing runs with checkpoints require an explicit `--resume`.

## Data and target protocols

Use `source: text`, `json`, or `hf` in the dataset catalog. Text inputs support record separators and bounded preprocessing; Hugging Face Datasets handles structured files and remote datasets, including streaming reads during cache preparation. Training samples from local memory-mapped token, mask, and record-boundary caches.

Train a byte-level BPE tokenizer with `tokenizer.train_from`, or set `tokenizer.type: hf` and provide a fast tokenizer artifact or model ID. Tokenizer and model vocabulary sizes must match. Existing artifacts take precedence over BPE training options; choose a new artifact path to train a different tokenizer.

BPE training reads records through the dataset's preprocessing limits and uses the same text formatter as model training. Plain text, prompt/response, messages, and TinyStoriesInstruct sources follow their configured dataset format. `tokenizer.train_from.sample_bytes` bounds the UTF-8 text supplied to tokenizer training.

For SFT, choose prompt/response records, role-tagged messages, or the TinyStoriesInstruct format. The default response mask excludes prompt tokens and padding from the loss. Messages use a `role:\ncontent\n` layout; provide your own formatting if you need a model-specific chat template. The TLM example sets `loss_mask: all_tokens` for its instruction corpus.

To train an explicit answer terminator, add these fields to your tokenizer and SFT dataset definitions:

```yaml
tokenizer:
  eos_token: "<|endoftext|>"
  bos_token: null
  pad_token: null
datasets:
  instructions:
    objective: sft
    append_eos: true
```

New BPE training reserves configured role tokens within `vocab_size`. Existing BPE/HF artifacts must contain them; LitheLM rejects missing tokens instead of expanding the model vocabulary. Saved tokenizers retain these roles. `append_eos` defaults to `false`; when enabled for SFT, preprocessing adds EOS after each supervised response and includes it in the loss. With `loss_mask: all_tokens`, it adds one EOS at the end of the record. PT preprocessing does not add EOS. BOS and PAD roles do not add tokens to encoded text. Generation stops when all sequences produce the configured EOS.

Dataset `kwargs` cannot repeat `path`, `split`, `streaming`, `cache_dir`, or `download_config`. For `source: json`, `data_files` is also reserved; set the top-level `path`. For `source: hf`, `kwargs.data_files` remains available for file-based builders.

Packing joins records into a token stream and permits attention across document boundaries. Without packing, a sample stays within one record. Each microbatch draws from one dataset; mixture weights describe sampling probabilities, not exact per-update token ratios.

Dense batches omit `attention_mask`, allowing causal SDPA to use `attn_mask=None, is_causal=True`. Padded batches retain an explicit mask under both target protocols.

### Target alignment

For context length `T`:

| Mode | Tokens read | Input | Labels used by the loss | Maximum targets per sequence |
|---|---:|---|---|---:|
| `shifted` (default) | T | `tokens[:T]` | The model aligns `logits[:, :-1]` with `labels[:, 1:]` | T − 1 |
| `next_token` | T + 1 | `tokens[:T]` | The data layer supplies `tokens[1:T+1]`; the loss uses matching positions | T |

```yaml
targets: {mode: shifted}
stages:
  - name: pt
    targets: {mode: next_token}
    # Other stage fields go here.
```

A stage inherits the global mode unless it declares an override. Both modes preserve response masks. Loss, gradient normalization, evaluation, and exposure counts use the number of valid targets after masking, summed across microbatches and ranks. Start a new experiment to change target mode; exact resume checks the protocol.

Evaluation reports NLL/PPL by dataset and objective, plus optional token accuracy and teacher-forced sequence exact match. Exact match measures predicted target tokens under teacher forcing. Use generated samples for a separate inspection of free-running output.

## Runs and checkpoints

### Shared inputs

```yaml
paths:
  corpus_root: ./corpora
  tokenizer_root: ./tokenizers
  run_root: ./outputs
workspace:
  copy_inputs: false
tokenizer:
  artifact: ${paths.tokenizer_root}/shared/tokenizer.json
datasets:
  stories:
    source: text
    path: ${paths.corpus_root}/stories/train.txt
```

Relative roots resolve against the directory where you launch the command. The default workspace references shared corpora and tokenizer files. HF download caches use `corpus_root/hf/`; a dataset's `cache_dir` can override that location. Tokenized corpora use the shared `corpus_root/.lithelm-cache/` directory. Cache keys include source metadata, dataset/preprocessing settings, and the tokenizer fingerprint; changing the model, learning rate, or run name reuses the cache.

For a local directory source, cache identity includes each file's relative path, size, and nanosecond modification time, collected recursively in sorted order. Adding, removing, or modifying files changes the identity without reading their contents for hashing.

Tokenizer and token-cache preparation use per-artifact OS file locks through `filelock`, with a 600-second wait timeout. Each writer rechecks the asset after acquiring its lock. Process exit releases the lock; persistent `.lock` files do not indicate an active writer. Do not delete them while processes may be waiting. Deleting a run leaves shared assets intact. Keep input files unchanged during preparation and training: cache identity uses local file size and modification time, not a full corpus hash, and remote datasets should use a pinned revision.

New runs use `<model-name>_<parameter-size>_<UTC-start-date>`. A second run with the same name on the same date gets a `-02` suffix. Omit `experiment.output_dir` for a new run; use the saved config or `--run-dir` to enter an existing run.

```text
experiment-directory/
├── corpora/                   # Raw inputs, .lithelm-cache/, and HF downloads
├── tokenizers/
└── outputs/<model>_<size>_<date>/
    ├── run.json
    ├── resolved_config.yaml
    ├── inputs.json
    ├── environment.json
    ├── environments/          # Records from later invocations
    ├── cache/                 # Runtime caches; token caches in copy mode
    ├── pt/                    # Stage checkpoints: best, final, checkpoint-N
    ├── sft/
    ├── metrics.jsonl
    ├── samples.jsonl          # If training-time generation is enabled
    ├── evaluations/
    └── inference/
```

`inputs.json` records paths and available file metadata. `environment.json` adds package versions, source commit/dirty state if available, device information, precision, world size, and tokenizer fingerprint. Later invocations write separate records under `environments/`. Versions serve as provenance; a dependency version change does not itself block resume.

### Resume or move a run

For a fresh prepared run, stop at a global optimizer step and resume:

```bash
lithelm train "$RUN/resolved_config.yaml" --stop-after 3
lithelm train "$RUN/resolved_config.yaml" --resume latest
```

Checkpoints contain model weights, optimizer/scheduler state when enabled, RNG state, training counters, configuration, and tokenizer fingerprint. LitheLM publishes a checkpoint after writing its `COMPLETE` marker. Numbered checkpoint retention leaves `best` and `final` intact.

All ranks finish saving state before rank zero writes `COMPLETE` and publishes the directory. On checkpoint lookup, a complete canonical directory takes priority. If it is absent, LitheLM promotes a complete `.writing` directory, or falls back to a complete `.previous` directory. It then removes stale sibling directories. Incomplete snapshots cannot become checkpoints. Distributed recovery runs on rank zero before the other ranks read state.

Exact resume requires compatible model/data/stage/runtime settings, the same tokenizer, seed, and world size, plus saved optimizer and scheduler state. Keep any earlier checkpoint referenced by `init_from`. Keep shared inputs unchanged throughout a run and its resumes.

To make a run self-contained with local inputs, set `workspace.copy_inputs=true` at creation. LitheLM copies those inputs into the run, stores token caches under its `cache/`, and includes a tokenizer snapshot in checkpoints. Prepare remote HF datasets as local files first if you need an offline bundle. In the default reference mode, shared input paths must remain accessible after you move the run.

### Throughput metrics

Training logs report `perf/input_tokens_per_sec` for non-padding tokens sent to forward, `perf/target_tokens_per_sec` for valid loss labels, and `perf/samples_per_sec`. SFT prompt tokens count toward input throughput even when the loss masks them. Shifted dense PT has `T` inputs and `T−1` targets per sequence.

The denominator sums training-update time, including batch reads, forward/backward, optimizer work, and enabled in-update statistics. It excludes curriculum/cache preparation, evaluation, checkpointing, generation, and logging. CUDA timing synchronizes at each boundary; distributed timing uses the slowest rank per attempt with global token counts. Skipped attempts count their work and time; exposure counters advance only on successful updates. Match hardware, precision, sequence lengths, batch sizes, target protocol, padding, attention backend, and statistics settings when comparing runs.

## Reference inference

Training uses full-sequence execution with `use_cache=False`, the default. Generation caches are temporary inference objects; LitheLM does not save them in model, optimizer, or Trainer state.

`infer.py` runs one prompt prefill, then forwards only the latest token at each decode step. Choose greedy decoding or sampling with temperature and top-k/top-p filtering. Generation stops at EOS or when prompt tokens plus generated tokens reach the requested context capacity.

Inference logs checkpoint paths relative to their output run when possible, and as resolved absolute paths for checkpoints outside it. Evaluation and inference can both load an external checkpoint with the matching model and tokenizer configuration.

Set the capacity in your configuration or override it for one invocation:

```yaml
inference:
  max_context_length: null
  max_new_tokens: 256
```

```bash
lithelm infer "$RUN/resolved_config.yaml" --checkpoint "$RUN/sft/final" \
  --prompt "A happy cat" --max-context-length 128 --max-new-tokens 64
```

The priority is `--max-context-length` > `inference.max_context_length` > `model.max_position_embeddings`. A smaller `model.position_embedding.max_positions` also bounds the default and any explicit capacity. Choose a positive integer within both model limits. LitheLM allocates the requested capacity, even when the model supports a longer context.

A prompt longer than the capacity raises an error. A prompt that fills it receives one prefill and returns without new tokens. Setting `max_new_tokens=0` returns a valid prompt without a forward pass. A generation request can stop before its `max_new_tokens` budget: a 1,500-token prompt with capacity 2,048 has room for 548 new tokens. LitheLM leaves the configured budget unchanged.

For direct model use, enter evaluation mode and disable gradients:

```python
model.eval()
with torch.inference_mode():
    out = model(prompt_ids, use_cache=True, max_context_length=128)
    cache = out.cache
    if cache.position < cache.max_length:
        next_ids = out.logits[:, -1].argmax(-1, keepdim=True)
        out = model(next_ids, cache=cache, use_cache=True)
```

`ModelCache.position` counts processed token slots, including padding; learned positions and RoPE start there on the next forward. `max_length` records the fixed capacity, `batch_size` records the fixed batch, and `attention_mask` retains the padding mask. `layers` holds one entry per model layer.

Each `AttentionCache` contains contiguous `key` and `value` buffers shaped `[batch, num_kv_heads, max_length, head_dim]`. Prefill allocates these buffers once. Decode writes the new K/V with `copy_()` into the next slice and reads only the initialized prefix. The buffer addresses stay fixed. Keys already include RoPE; historical keys keep their original rotation. GQA/MQA expands KV heads only for attention computation, so cache storage uses `num_kv_heads`.

Per Attention layer, K/V storage requires `2 × batch × num_kv_heads × max_context_length × head_dim × element_size` bytes. The two buffers account for the factor of 2; a BF16 element takes 2 bytes. This estimate excludes weights, activations, masks, and FLA state. Direct cached forwards raise an error on overflow or an attempt to change capacity. LitheLM does not grow, truncate, or slide the buffers.

Each FLA wrapper owns a separate FLA cache at its global `ModelCache.layers[i]` slot. Within that independent cache, its GatedDeltaNet uses `layer_idx=0`. The wrapper passes the opaque object through `past_key_values` and retains the returned state. FLA manages its recurrent and convolution state; the model still enforces the total context capacity. Low-precision prefill and decode kernels can produce small differences from full-sequence logits; compare with a tolerance appropriate to the dtype and kernel.

Cached forwards require `eval()`, disabled gradients, and no labels. Reuse a cache only with the same model, batch order, device, and precision. Prefill accepts a 2D 0/1 padding mask with at least one valid token per row. Decode accepts exactly one unmasked token per row; omit its mask or supply an all-one mask shaped `[batch, 1]`. The model retains the prefill mask. `generate()` accepts unpadded prompts of equal length within a fixed batch; it has no padding-mask argument.

Built-in SDPA, Flash, and Triton attention support caching. Custom Mixers and custom attention operators keep their full-sequence interfaces and report an error if asked for cached inference. Cache inference uses eager execution as its baseline; training retains its existing `torch.compile` path. This reference interface does not provide cache reordering, dynamic batching, sliding windows, or production serving. Run one experiment per process.

## Custom Mixers and statistics

Register a module in `CUSTOM_LAYERS` before constructing the model. A Mixer receives `[batch, length, hidden_size]` and returns a tensor with the same shape. It owns its initialization: LitheLM calls an optional `init_weights()` once after construction and excludes the module's subtree from core initialization and residual scaling.

Save this example as a Python launcher such as `research.py`:

```python
from torch import nn
from lithelm.cli import main
from lithelm.model.builders import CUSTOM_LAYERS
from lithelm.statistics import StatSpec


class LinearProbe(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)

    def init_weights(self):
        nn.init.constant_(self.projection.bias, 0.1)

    def stat_specs(self):
        return {"output_norm": StatSpec(aggregation="mean", distributed="max", format=".4f")}

    def forward(self, x, attention_mask=None, collect_stats=False):
        y = self.projection(x)
        if collect_stats:
            return y, {"output_norm": y.detach().float().norm()}
        return y


CUSTOM_LAYERS["linear_probe"] = LinearProbe

if __name__ == "__main__":
    main()
```

Select it in a layer with `mixer: {type: custom, custom: {name: linear_probe}}`, then use `python research.py train experiment.yaml`. Use the same registration launcher for evaluation and inference. A checkpoint stores weights, not the Python implementation of your module.

`init_weights`, `stat_specs`, and `collect_stats` are optional. A Mixer can return a tensor alone or `(tensor, stats)`, where values are finite scalar tensors or numbers. Core adds a namespace such as `layer/0/mixer/output_norm` and handles aggregation and logging.

```yaml
statistics:
  every: 50
  microbatch_aggregation: mean
  overrides:
    "*/output_norm": last
    "layer/3/mixer/output_norm": max
logging:
  csv: true
  csv_file: metrics.csv
```

Microbatch aggregation supports `last`, `mean`, `sum`, `min`, and `max`. Precedence is exact field override, first matching `fnmatchcase` wildcard, `StatSpec.aggregation`, then the global default. Leave the spec's aggregation as `None` to inherit the global value. Mean gives equal weight to reported microbatch values.

`StatSpec.distributed` applies a separate reduction across ranks: `mean`, `sum`, `min`, `max`, or `local`. Local reports rank zero's value. Ranks must report matching field names and specs at a collection step. Rank means average the values after each rank's microbatch reduction.

Set `statistics.every` to a positive step interval to collect stats, or `0` to disable them. The default `null` follows the stage's `stats.model_internal.enabled` and `intervals.stats` settings. Logger writes `model_stats` events to enabled sinks. `format` controls stdout display; JSONL and CSV retain numeric values. CSV uses one row per metric with fixed columns: timestamp, global step, stage, event, metric, value.

Other extension points are plain dictionaries:

| Registry | Contract |
|---|---|
| `lithelm.ops.attention.CUSTOM_OPS` | Attention over Q/K/V tensors shaped `[batch, heads, length, head_dim]`; accepts causal, mask, and dropout arguments |
| `lithelm.metrics.STATS` | Model-level scalar functions for configured training statistics |
| `lithelm.metrics.METRICS` | Evaluation functions returning additive numerator/denominator pairs |

For a metric used with `next_token`, accept `target_mode` and use `lithelm.metrics.loss_inputs` to align logits and labels. Existing two-argument metrics remain usable with shifted targets.

## GPU and distributed execution

Choose visible devices through `CUDA_VISIBLE_DEVICES` or Docker's `--gpus` option. On shared infrastructure, use the operator's resource guard before borrowing a GPU.

For an untrained prepared run, launch two ranks with Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  -m lithelm train "$RUN/resolved_config.yaml" \
  --set runtime.device=cuda --set runtime.precision=bf16 \
  --set distributed.strategy=ddp
```

Global batch size is `micro_batch_size × grad_accum_steps × world_size`. Adjust per-stage batches for the desired global batch before training. The TLM convenience runner requires `strategy: single`; use the core CLI for distributed experiments.

| Backend | Use and constraints |
|---|---|
| `attention.impl: sdpa` | Default PyTorch scaled dot-product attention |
| `attention.impl: flash` | Forces the PyTorch Flash backend; unsupported device/mask combinations raise errors |
| `attention.impl: triton` | Research operator: PyTorch GEMMs and Triton masked-softmax forward/backward; CUDA fp16/bf16/fp32, head dimensions 16/32/64/128, length ≤ 1024, attention dropout 0 |
| `attention.impl: custom` | Calls the named function in `CUSTOM_OPS` |

The Triton operator materializes an attention matrix with O(batch × heads × length²) memory and supports first-order gradients. Use it for operator experiments; benchmark your workload before choosing it for performance.

FLA layers preserve FLA initialization and no-weight-decay markers. Set `mixer.type: gdn` and pass options through `mixer.gdn.mode` and `mixer.gdn.kwargs`. FLA reports unsupported configurations.

Linux CUDA launches use NCCL where available; CPU and Windows use Gloo. CPU CI covers Linux and Windows. Install optional GPU libraries that support your platform. The Python run context changes process-wide working-directory and cache settings, so run concurrent experiments in separate processes.

CPU execution rejects `runtime.precision: fp16`, including when `device: auto` resolves to CPU; use `fp32` or a supported CPU `bf16` path. With `runtime.deterministic: true`, LitheLM sets a missing `CUBLAS_WORKSPACE_CONFIG` to `:4096:8` before probing CUDA and enables PyTorch deterministic algorithms. Set this environment variable before initializing CUDA yourself. Determinism is best-effort: third-party and custom kernels can impose their own limits or raise errors.

## Project structure

```text
lithelm/
├── cli.py, config.py           # CLI and validated configuration
├── workspace.py               # Run allocation and input references
├── tokenizer.py, data.py       # Tokenization, masks, caches, sampling
├── model/                     # Embeddings, blocks, Mixers, FFNs, output head, inference caches
├── ops/                       # Attention dispatch and Triton kernels
├── train.py, stages.py         # Runtime composition and stage sequencing
├── trainer.py, scheduler.py    # Update loop, optimizer groups, WSD
├── checkpoint.py               # Save, restore, and retention
├── evaluate.py, infer.py       # Evaluation and text generation
└── metrics.py, statistics.py, logging_utils.py, provenance.py
configs/                       # Complete example configurations
examples/                      # CPU preparation and TLM guide
scripts/                       # TLM preparation and full workflow
tests/                         # Unit and subprocess integration tests
```

The CLI loads and validates the config, enters a run directory, and prepares the tokenizer and model. `stages.py` sequences the course through one `Trainer`. During an update, the trainer samples microbatches, counts valid targets across ranks, computes normalized gradients, and advances the optimizer and WSD schedule. Evaluation, checkpointing, and logging run at their configured intervals. Model modules own the forward computation and optional scalar measurements; the trainer owns their scheduling and reduction.

## Testing and contributing

```bash
python -m pytest tests -q
```

The suite covers model construction, target/mask alignment, data sampling, WSD, scalar statistics, checkpoint recovery, and run portability. CPU integration tests launch two ranks; CUDA tests skip if CUDA is unavailable. To run the distributed tests on two visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 LITHELM_TEST_DEVICE=cuda \
  python -m pytest tests/test_distributed.py -q
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, [SECURITY.md](SECURITY.md) for security reporting and input trust, and [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

LitheLM is licensed under [Apache-2.0](LICENSE). Dependencies and downloaded corpora retain their own licenses.
