# Explicit training extensions

From the repository root, create synthetic assets, interrupt a two-stage run,
and resume it:

```bash
python -m examples.extensions.run --init work/extension-demo
python -m examples.extensions.run work/extension-demo/config.yaml --stop-after 2
python -m examples.extensions.run work/extension-demo/config.yaml --resume latest
```

The initializer refuses to overwrite an existing directory. The example registers
all three hooks, uses StepPlan + LossComposer + built-in values-file LR, then
inherits Adam moments into SFT with a new LR schedule. Default CE validation
continues independently of the training components.

For DDP, initialize once before launching. Every worker imports the registration
module:

```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 \
  -m examples.extensions.run work/extension-demo/config.yaml \
  --set runtime.device=cuda --set runtime.precision=bf16 --set distributed.strategy=ddp
```

Use the Windows launcher instructions in the main README for a CPU DDP launch.
Invoke this module in place of `lithelm`; the ordinary CLI does not import external
hooks from YAML.

Each extension works independently. Keep `data.strategy: default` for BatchStream,
`loss.type: default` for token-weighted CE, or `lr_schedule.type: wsd` for WSD.
To use the registered external LR provider, replace a stage's `lr_schedule` with:

```yaml
lr_schedule:
  type: custom
  name: sequence
  kwargs:
    values: [0.0002, 0.0005, 0.0001, 0.0]
```

This fragment matches the four-update PT stage. All optimizer param groups use the
same absolute LR; the framework does not apply relative group multipliers.

## Python contracts

Register factories in `lithelm.extensions.STEP_PLANS`, `LOSS_COMPOSERS`, or
`LR_PROVIDERS`. StepPlan factories receive `stage`, `tokenizer`, `path`, and
configured `kwargs`. Composer and custom LR factories receive `stage` and
`kwargs`. No YAML-driven imports or plugin discovery run.

All hooks expose `identity()`, returning finite JSON data that describes their
implementation revision and assets. The framework preflights configured stage
extensions once before the first update. It hashes configured files and agrees on
identities across ranks. Plugins must include hashes for additional frozen files
they manage through `kwargs`; the core does not discover private assets.

A plan's `batch_at(stage_step, rank, microbatch_index)` returns `PlannedBatch`.
The stage instance provides `stage_name`; step and microbatch indices start at zero.
Retries reuse the same indices, and resume starts at the saved successful step.
Inputs must have exactly `micro_batch_size` rows and a length within `data.max_length`.
Default CE requires int64 labels, with `-100` for unsupervised positions.

Labels and loss masks have the input shape. In `shifted` mode, labels at position
`j+1` supervise logits at `j`; `next_token` labels already supervise the corresponding
logit position. Plans supply the final tensors, and the framework does not collate
or shift them again. LossComposer owns alignment; `metrics.loss_inputs` implements
both protocols. Tensor metadata moves with the batch but never enters model kwargs.
Only `input_ids`, `attention_mask`, and default-CE `labels` reach the model.

`StepContext` contains `stage_name`, `stage_step`, `rank`, and `target_mode`.
`components(context)` returns fixed named `ComponentSpec(weight, required=True)`
entries; weights may vary by step but must agree across ranks.
`denominators(batch, context)` supplies nonnegative scalar counts without gradients
before forward. `numerators(output, batch, context)` supplies finite differentiable
scalar sums after forward. A locally absent component may omit its entries or return
zeros. A required component with zero global denominator fails; an optional one
contributes zero and logs zero.
Each update needs at least one component with a positive global denominator and
nonzero weight; otherwise training fails before forward or optimizer steps.
Compute numerators from model output tensors; creating a scalar from `.item()` and
setting `requires_grad=True` loses the model graph and fails validation.

For each component, the update objective is its weighted global numerator sum
divided by its own global denominator sum. The framework reduces the denominator
vector once, then backpropagates one microbatch graph at a time, multiplying by
`world_size * grad_accum_steps` to compensate DDP averaging and Accelerate scaling.
Plugins must not perform collectives, optimizer/scheduler steps, checkpoint I/O,
dynamic imports, or logging. The logger records successful-update
`train/loss_total`, `train/loss/<name>`, component weights and denominators;
these metrics are not NLL/PPL. Target exposure counts use the union of loss masks,
intersected with valid labels when present.
Custom-loss updates also log actual `data/source_samples/<name>`,
`data/source_targets/<name>` and `data/source_objective/<name>` provenance.

LR providers expose `value_at(k)` for the zero-based successful update index and
return a finite nonnegative absolute LR. Built-in `values_file` accepts exactly
`max_steps` headerless little-endian float64 values (`<f8`). It loads immutable
values once and checks the SHA-256 on resume. A skipped optimizer step retains its
index and LR. The final scheduler position holds the last LR without reading past
the file. `optim/lr` reports the applied LR, and `optim/lr_phase` identifies the
provider type.

Loading optimizer state without scheduler state preserves moments and applies the
new stage's LR at its starting index. Continuing a provider scheduler requires
identical configuration, asset identity and `max_steps`, plus
`init_from.reset_stage_step: false`. Loading at an index beyond the target stage
fails. WSD retains its previous behavior.

## Files and resume

YAML extension paths resolve relative to the YAML directory (the run root when
loading a saved run config). Programmatic configs resolve extension paths against
the caller's cwd during normalization. Private paths inside `kwargs` belong to the
plugin; anchor them explicitly in your launch script.

`extensions.json` and checkpoint trainer state record config/implementation
identities, configured file paths, sizes and SHA-256 hashes. Content changes reject
exact resume before backward, including changes to assets for other stages.
Each rank streams and hashes its local configured file during preflight. The file
check compares size and SHA-256, and retains rank-zero paths for provenance;
unreadable or different contents fail on all ranks. Hashing does not run per update.
The default signature omits the new inactive fields, preserving v0.1 checkpoint
compatibility while retaining existing strict checks.

`copy_inputs: true` does not bundle extension assets. Runs record this external
dependency in an `external_extension_assets` event; retain those files separately.
The first version uses fixed microbatch and accumulation counts. It does not add
variable backward counts, plugin discovery, an event bus, or external scoring.
