# TLM 7.4M: single-GPU training

This example trains a causal language model on TinyStories, then fine-tunes it on TinyStoriesInstruct with pretraining replay. The [configuration](../../configs/examples/tlm_7m.yaml) defines the model and course. The [runner](../../scripts/run_tlm.py) handles preparation, checkpoint recovery, evaluation, text generation, and plotting.

## Model and course

| Setting | Value |
|---|---|
| Parameters | 7,432,704 |
| Decoder blocks | 8 |
| Hidden size / attention heads | 256 / 8 |
| FFN | GELU, hidden size 1,024 |
| Context / position embeddings | 256 / learned absolute |
| Normalization / dropout | Pre-LayerNorm / 0.05 |
| Vocabulary | Byte-level BPE, 4,096 tokens, tied input/output embeddings |
| Execution | One process on one GPU, bf16 by default |
| Batch | 32 sequences per update, accumulation 1 |
| PT | 60,000 updates on TinyStories |
| SFT | 4,000 updates; 90% instruction batches, 10% PT replay batches |
| Optimizer and schedule | AdamW; warmup, stable, cosine decay |
| Target protocol | `shifted`: up to 255 valid targets per 256-token sequence |

SFT starts from the PT `best` checkpoint and resets the optimizer and scheduler. The instruction datasets use `loss_mask: all_tokens`. Mixture weights are sampling probabilities; token proportions can differ from the 90/10 batch mixture.

## Setup

Create a Python 3.10+ environment and install PyTorch for your GPU. From the repository root:

```bash
python -m pip install -e ".[examples]"
LITHELM_REPO="$PWD"
mkdir -p ../tlm-experiments
cd ../tlm-experiments
```

The commands in this guide use Bash. `LITHELM_REPO` points to the checkout; the current directory holds shared inputs and run outputs.

## Run the complete course

```bash
CUDA_VISIBLE_DEVICES=0 python "$LITHELM_REPO/scripts/run_tlm.py"
```

The runner downloads the four source files if they are absent. Allow several GB for the corpora, plus space for token caches and checkpoints. Use an infrastructure guard before borrowing a GPU that hosts other workloads.

The workflow performs these steps:

1. Prepare shared corpora, BPE, and token caches.
2. Train PT through step 1,000, save a checkpoint, and resume in a new process.
3. Finish PT and SFT, for 64,000 optimizer updates in total.
4. Evaluate PT/SFT `best` and `final` on both validation datasets, and generate a sample from each checkpoint.
5. Write a training chart and `WORKFLOW_COMPLETE` after successful completion.

Periodic training evaluation uses at most 50 batches per dataset. The final evaluation passes use the full validation datasets. Read `logs/workflow.log` for subprocess output if a step fails.

## Check the workflow with a short run

```bash
CUDA_VISIBLE_DEVICES=0 python "$LITHELM_REPO/scripts/run_tlm.py" --smoke
```

Smoke mode keeps the 7.4M architecture and uses 20 PT + 20 SFT updates. It sets batch size 4, accumulation 2, and a length curriculum from 128 to 256. It stops at PT step 10 to exercise resume, limits evaluation to two batches per dataset, and uses bounded corpus slices.

Smoke inputs live under `corpora/tlm-smoke/` and `tokenizers/tlm-smoke/`. The run name starts with `tlm-smoke_7.4M_`; the completion marker records `smoke: true`. Use this mode to check execution and resource requirements. Assess model quality after a full training course.

## Data and tokenizer

The preparation script reads [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct). It records file sizes and SHA256 digests in `corpus-manifest.json`, plus the repository revision and URL for files it downloads.

BPE training reads the first 10 MiB of PT training text. PT and SFT use the same tokenizer. The full instruction preprocessing pass reads at most 419,430,400 characters from its training file; the preparation step retains the complete downloaded source file.

To use files you have downloaded, provide this layout:

```text
existing-corpora/
├── tinystories/
│   ├── TinyStories-train.txt
│   └── TinyStories-valid.txt
└── tinystories_instruct/
    ├── TinyStories-Instruct-train.txt
    └── TinyStories-Instruct-valid.txt
```

```bash
python "$LITHELM_REPO/scripts/prepare_tlm.py" --data-dir /path/to/existing-corpora
```

`--data-dir` populates the configured shared corpus directory when a target file is missing. Omit the flag to download instead. `prepare_tlm.py` prepares inputs without training and prints the run path on its last output line. The same preparation arguments work with `run_tlm.py`.

Choose shared roots and a run root at creation:

```bash
python "$LITHELM_REPO/scripts/prepare_tlm.py" \
  --set paths.corpus_root=/path/to/corpora \
  --set paths.tokenizer_root=/path/to/tokenizers \
  --set paths.run_root=/path/to/runs
```

Preparation locks shared tokenizer and token-cache writes. Tokenized corpora live in `corpus_root/.lithelm-cache/`; subsequent runs reuse them. Keep corpus files and tokenizer artifacts unchanged for the duration of those experiments.

## Outputs

With the default paths:

```text
corpora/tinystories/
corpora/tinystories_instruct/
tokenizers/tlm_4096/tokenizer.json
outputs/tlm_7.4M_YYYY-MM-DD/
├── run.json
├── resolved_config.yaml
├── configs/examples/tlm_7m.yaml
├── scripts/
├── inputs.json
├── corpus-manifest.json
├── environment.json
├── environments/
├── cache/
├── pt/                          # best, final, checkpoint-N
├── sft/
├── metrics.jsonl
├── samples.jsonl
├── evaluations/
├── inference/
├── logs/workflow.log
├── charts/training.png
└── WORKFLOW_COMPLETE
```

Names use the UTC start date. A second run with the same name on the same date gets a suffix such as `-02`; resume retains the original identity.

## Resume, evaluate, or move a run

Set `RUN` to the path printed during preparation:

```bash
RUN=/path/to/experiments/outputs/tlm_7.4M_YYYY-MM-DD
CUDA_VISIBLE_DEVICES=0 python "$LITHELM_REPO/scripts/run_tlm.py" --run-dir "$RUN"
```

The runner resumes the latest published checkpoint and completes any remaining postprocessing. A run with `WORKFLOW_COMPLETE` returns without another training pass.

You can also use the core CLI:

```bash
# Start a prepared run that has no checkpoints.
lithelm train "$RUN/resolved_config.yaml"

# Continue a run after an interruption.
lithelm train "$RUN/resolved_config.yaml" --resume latest

lithelm eval "$RUN/resolved_config.yaml" --checkpoint "$RUN/sft/final"
lithelm infer "$RUN/resolved_config.yaml" --checkpoint "$RUN/sft/best" \
  --prompt "Once upon a time" --max-new-tokens 40
```

The default run references shared inputs. After moving a run, keep those inputs accessible at their saved paths. To create a bundle with local input copies, add `--set workspace.copy_inputs=true` at creation. Preserve preceding `best`/`final` checkpoints that later stages reference. Install LitheLM and its dependencies in the destination environment.

## Adjust memory use or target exposure

For a new full-course run, keep global batch 32 with smaller microbatches:

```bash
CUDA_VISIBLE_DEVICES=0 python "$LITHELM_REPO/scripts/run_tlm.py" \
  --set stages.0.batch.micro_batch_size=8 \
  --set stages.0.batch.grad_accum_steps=4 \
  --set stages.1.batch.micro_batch_size=8 \
  --set stages.1.batch.grad_accum_steps=4 \
  --set stages.0.eval.batch_size=8 \
  --set stages.1.eval.batch_size=8
```

Smoke mode supplies its own batch and schedule settings after applying `--set`. The memory adjustment above applies to the full-course runner. Changes to batch shape can alter the training trajectory; use a new run rather than a checkpoint that requires exact resume.

Set `--set targets.mode=next_token` on a new experiment to read 257 tokens for each 256-position input. This yields up to 256 targets per sequence. The default remains shifted, with at most 255. Response masking can reduce either count.

The convenience runner enforces single-process execution. For DDP experiments, prepare the run and use the [core distributed CLI](../../README.md#gpu-and-distributed-execution) with an explicit strategy and batch configuration.
