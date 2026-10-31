# GoTrain — train, measure, improve

An autonomous geospatial model training agent for the AUB ACM × RASID hackathon.
Give it a GEO-Bench task and a compute budget. A single Strands agent on Amazon
Bedrock inspects the dataset, chooses a recipe, runs training tools, compares
validation experiments, and publishes the best version with an auditable history.

**Status:** implementation includes classification, segmentation, agent tools,
artifact storage, validation reports, MCP adapter and tests. Real GPU runs and
the four final test evaluations must be performed on the team's SageMaker instance.
No benchmark results are fabricated. The earlier manually run EuroSAT CNN reached
**75% validation accuracy** in five epochs (user-reported; not a final test score).
Beating the instructor's reference requires its exact scores, metrics and protocol;
neither this implementation nor pretrained weights guarantee that outcome.

## The four tasks

| Task | Input | Output | Default validation selection |
|---|---|---|---|
| `m-eurosat` | 13-band satellite image | 10 land-cover classes | accuracy |
| `m-pv4ger` | RGB aerial image | solar panel present/absent | accuracy |
| `m-pv4ger-seg` | RGB aerial image | panel foreground mask | mean IoU |
| `m-nz-cattle` | RGB aerial image | cattle foreground mask | mean IoU |

For binary segmentation we also record foreground IoU, per-class IoU, accuracy,
macro F1, balanced accuracy and the global confusion matrix. Mean IoU includes
both background and foreground when their union is nonzero. Absent classes with
zero union are omitted from mean IoU. Do not compare different metric definitions.
Cattle masks in GEO-Bench are rasterized annotations, not precise animal outlines.

## Quick start on the existing SageMaker environment

Python 3.12 and a CUDA-capable GPU are expected. The existing environment already
has GPU PyTorch 2.7.1/torchvision 0.22.1 and working Bedrock access.

```bash
source ~/gotrain-env/bin/activate
cd ~/GoTrain-Hackathon
git pull --rebase origin main
python -m pip install -e '.[dev]'
python -m pip check
python -m gotrain doctor
python -m pytest -q
python -m gotrain smoke
```

The smoke test uses synthetic tensors; it does not access any benchmark test data.
If Git refuses a rebase because your working tree has changes, commit those changes
first. Do not discard the local baseline or reset the repository.

For a fresh Linux GPU environment, install the tested CUDA wheel pair before the package:

```bash
python -m venv ~/gotrain-env
source ~/gotrain-env/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[dev]'
```

For a CPU-only development machine use `https://download.pytorch.org/whl/cpu`
instead. Full GEO-Bench training is intended for the supplied GPU, not CPU CI.
Pretrained ResNet weights download on the first training run. An unavailable
download fails explicitly; there is no silent fallback to random initialization.

## One task, end to end

The existing EuroSAT download at `~/gotrain-data/classification_v1.0/m-eurosat`
is reused. `prepare` downloads only the named task, not the full benchmark.

```bash
python -m gotrain prepare m-eurosat
python -m gotrain inspect m-eurosat
python -m gotrain train m-eurosat
python -m gotrain compare m-eurosat
python -m gotrain report m-eurosat
```

`train` runs a manual pretrained baseline (15 epochs, early stopping, 30-minute
soft budget). `agent` runs the autonomous workflow:

```bash
python -m gotrain agent m-eurosat --experiments 3 --minutes 60
```

The agent can inspect data, run bounded experiments, compare results, publish the
best model, and verify a download. It **cannot** access a final test evaluation tool.
Failed training attempts consume slots. Models are trained sequentially; a filesystem
lock prevents simultaneous GoTrain training/evaluation processes on the same run root.
Limits are soft wall-clock budgets: preprocessing, downloads and an in-progress
validation pass can exceed them. Claude API retries/latency are not a hard process timeout.

Complete the other tasks **sequentially** on the single T4:

```bash
python -m gotrain prepare m-pv4ger
python -m gotrain agent m-pv4ger --experiments 3 --minutes 60
python -m gotrain prepare m-pv4ger-seg
python -m gotrain agent m-pv4ger-seg --experiments 3 --minutes 90
python -m gotrain prepare m-nz-cattle
python -m gotrain agent m-nz-cattle --experiments 3 --minutes 90
```

If memory is limited, try batch size 4 and image size 128 or 256 for segmentation.
Preserving small objects may need larger images; confirm the tradeoff on validation.

## What the agent actually controls

One tool call is one experiment with an explicit hypothesis. The recipe can change
model (`small`, `resnet18`, `resnet34`), pretrained initialization, learning rate,
weight decay, image size, batch size, augmentation, class-balanced loss, Dice loss,
epochs and patience. Classification uses a ResNet head; segmentation uses a ResNet
encoder with U-Net-style skip connections and a newly initialized decoder.

Use a recipe file for a manual experiment:

The optional `warm_start` field takes a completed run ID from the same task. It
fine-tunes that run's best validation checkpoint, preserving its normalization,
model architecture and image size. A new run records the parent checkpoint hash.
The optimizer restarts with the new recipe; this is fine-tuning rather than exact
optimizer-state resumption.

```json
{
  "model": "resnet18",
  "pretrained": true,
  "epochs": 25,
  "batch_size": 8,
  "image_size": 256,
  "learning_rate": 0.0003,
  "augmentation": true,
  "balanced_loss": true,
  "dice": true,
  "max_minutes": 45
}
```

```bash
python -m gotrain train m-pv4ger-seg --recipe recipe.json --hypothesis "Test balanced CE plus Dice for sparse panel foreground"
```

The loss setting is a hypothesis, not proof that class imbalance caused errors.
Use measured comparisons and per-class recall to decide. A larger backbone can
cost more time without improving validation. Keep the seed and evaluation protocol
fixed where possible. Scores should eventually be checked across seeds if time permits.

## Correctness and evaluation protocol

* Training uses `train`; tuning uses **`valid`**, not `val`.
* Mean/std and class balance are estimated from a seeded sample of training images
  only, then stored with exact sampled indices. No benchmark-wide normalization
  statistics are used (they may reflect a different band selection or protocol).
* All input channels use physical stored order. EuroSAT's published band-name
  metadata is incorrect; the code never selects its bands by those names.
* For non-RGB inputs, pretrained input weights are initialized from averaged RGB
  weights and scaled for the channel count. This is a documented initialization
  strategy, not a geospatial foundation model.
* Image and mask augmentations remain aligned. Masks use nearest-neighbor resizing.
* Resized development images use a versioned, memory-mapped float16 cache when the
  estimated image payload is at most 2 GiB. Normalization and random augmentation
  still happen at sample access. Validation masks retain their native resolution.
  Larger recipes stream images instead. Set `cache: false` to disable this
  optimization or to check sensitivity to cache quantization.
* Validation/test segmentation predictions are resized to **native label resolution**
  before metrics. Training may use smaller masks to fit the GPU.
* IoU is calculated from a dataset-wide confusion matrix, not averaged batch IoUs.
* Only validation scores choose checkpoints. Ties retain the earlier run/epoch.
* Seeds and package/source versions are recorded. Bitwise GPU determinism is not claimed.

### Final test: only after development is finished

Do not run these during tuning. There is no test command in the agent/MCP tools.
Each command freezes the best validation run, verifies its checkpoint and dataset
metadata, creates an exclusive test-access marker, and then evaluates that test split.
Further training for that task is blocked after freezing.

```bash
python -m gotrain finalize m-eurosat --confirm-test-once
python -m gotrain finalize m-pv4ger --confirm-test-once
python -m gotrain finalize m-pv4ger-seg --confirm-test-once
python -m gotrain finalize m-nz-cattle --confirm-test-once
python -m gotrain export --out submission
```

A second finalization fails before opening test samples. If a process crashes after
claiming test access, it remains locked: inspect logs and discuss with the organizers;
do not delete markers or create new run roots to evade the once-only rule. These are
application safeguards, not filesystem isolation against someone manually reading files.

## Extra points: artifact storage and versions

Run layout (outside Git):

```text
~/gotrain-runs/v1/<task>/<timestamp_uuid>/
    config.json          # hypothesis, recipe, seed, source/package provenance
    normalization.json   # train-only stats and selected sample indices
    history.json         # every epoch's loss and validation confusion metrics
    events.jsonl         # timestamped execution events
    best.pt              # weights, preprocessing, recipe, selected epoch
    summary.json         # selected score, runtime, peak GPU allocation, SHA-256
~/gotrain-runs/v1/sessions/<session_id>/
    events.jsonl         # agent tool inputs, outputs, errors and decisions
    messages.json        # observable agent conversation / tool calls
    session.json         # session limits and completed run IDs
```

The conversation records visible explanations and tool evidence; it does not expose
private model reasoning. Run IDs are immutable versions. Original manual baseline
files in `~/gotrain-runs/eurosat-baseline` are left intact and separate from v1 runs.

Publish and restore model versions with checksummed manifests:

```bash
python -m gotrain publish-model m-eurosat RUN_ID
python -m gotrain restore-model m-eurosat RUN_ID
python -m gotrain publish-data m-eurosat
python -m gotrain restore-data m-eurosat
```

Without cloud settings these use `~/gotrain-artifacts`; model and dataset restores
go into separate directories and cannot overwrite active experiment history.
Dataset archives include the benchmark splits, but archive copying does not open
test samples or compute test scores. Active datasets are reused from Hugging Face.

For S3, ask the organizers for an approved bucket and execution-role access to
`s3:PutObject` and `s3:GetObject` under its `gotrain/` prefix. Set:

```bash
export GOTRAIN_S3_BUCKET=THE_PROVIDED_BUCKET
```

The same commands then also upload/download through S3. No AWS keys in code,
no bucket creation, no permission changes. S3 integration is implemented but must
be exercised against the actual authorized bucket. It is not marked verified here.
Model keys contain task and run ID. Dataset manifests record archive hashes.

## Demo: show evidence

`python -m gotrain report TASK` creates a self-contained `report.html` under the task's
run directory: validation curves, experiment hypotheses, runtime, measured metrics,
and fixed validation prediction examples. Open/download it from JupyterLab's file
browser. It needs no hosted dashboard, public port, or separate JavaScript app.
Show the agent in the terminal, then the report. No synthetic scores are presented
as real results. A score improvement should appear in both the report and the logs.

## GoPilot / MCP integration

```bash
python -m gotrain.mcp_server
```

This starts a stdio MCP server exposing inspection, training, comparison and model
upload/download. GoPilot can launch it with the same Python executable/environment
on SageMaker. Transport and deployment authentication must be agreed with RASID;
the adapter is not a claim of completed GoPilot deployment. Final test access is
intentionally absent. Direct Strands tools are the default hackathon runtime.

## Submission and presentation

`export` bundles real scores, development histories, configs, agent session logs,
and reports. It explicitly lists any missing final scores; it does not fill them in.
Checkpoints remain separate artifacts referenced by run ID and hash. Review logs
before sharing; never commit credentials. Commit the small submission bundle if
the organizers want it in GitHub; keep datasets and checkpoint weights out of Git.

Suggested presentation: problem → tool/agent architecture → a real baseline and
validation experiment → segmentation prediction example → four final test results
and comparison protocol → artifact round-trip/MCP plan → measured limitations.

### Tests

```bash
python -m pytest -q
```

Tests cover classification/segmentation forwards, tiny training runs, aligned masks,
train-only normalization, global IoU, checkpoint round trips and corruption, reports,
and once-only final evaluation on **synthetic fixtures**. GitHub Actions runs CPU
checks. SageMaker must validate real data, pretrained downloads, CUDA, Bedrock tool
execution, S3 permissions, and the instructor's baseline comparison separately.

### References

* [Challenge benchmark](https://github.com/ServiceNow/geo-bench)
* [GEO-Bench data](https://huggingface.co/datasets/recursix/geo-bench-1.0)
* [Strands Python quickstart](https://strandsagents.com/docs/user-guide/sdk/quickstart/python/)
* [Torchvision ResNet18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
