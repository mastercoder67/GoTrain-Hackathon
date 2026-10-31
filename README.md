# GoTrain — train, measure, improve

An autonomous geospatial model training agent for the AUB ACM × RASID hackathon.
Give it a GEO-Bench task and a compute budget. A single Strands agent on Amazon
Bedrock inspects the dataset, chooses a recipe, runs training tools, compares
validation experiments, and publishes the best version with an auditable history.

**Status (31 October 2026):** the core implementation is pushed to `main`.
The repository contains a five-epoch manual EuroSAT baseline (75.00% validation
accuracy) and three GPU validation experiments for `m-eurosat`; the best recorded
validation accuracy is **97.10%**. These are validation results, not final test
scores. The current GitHub tree does not contain final `test-result.json` files
for all four tasks or a complete submission bundle. If final tests have already
been run in SageMaker, recover and verify their original outputs rather than
running them again. Never present validation metrics as test results.

## The AI-scientist loop

GoTrain does not only tune numbers; it investigates why a model fails and tests a fix:

```text
inspect → baseline → diagnose errors → one targeted hypothesis → train → kept / rejected → diagnose again …
```

| Step | Tool | What it produces |
|---|---|---|
| Diagnose | `diagnose_errors` | Measured validation failures: object detection rate by object size (small/medium/large terciles), predicted vs true foreground area, false positives on empty images, most-confused class pairs, worst examples as images, and candidate recipe levers |
| Hypothesize | `run_experiment` | One change that must quote the number it targets, e.g. *"small panels are 31% detected vs 90% for large; test focus_crops"* |
| Targeted fix | `focus_crops` recipe option | Half of the segmentation training samples are zoomed 2× onto a random foreground object (aligned image/mask) |
| Verdict | `run_experiment` result | `kept` only if strictly better on validation than the previous best, otherwise `rejected`; plus minutes and attempts left |
| Show | live diary | `~/gotrain-runs/v1/live-diary.html` refreshes every 10 s with each observation, hypothesis, verdict, epoch progress and the budget |
| Before → after | showcase | At session end: the first model's worst validation images, shown as input · first model · best model · true mask |

Rejected hypotheses stay in the diary and logs; they are evidence too. All of this
uses the validation split only.

```bash
python -m gotrain agent m-pv4ger-seg --experiments 3 --minutes 30
# In JupyterLab: right-click ~/gotrain-runs/v1/live-diary.html → Open in New Browser Tab
python -m gotrain diagnose m-pv4ger-seg            # analyse the best run by hand
python -m gotrain showcase m-pv4ger-seg            # first run vs best run, same images
python -m gotrain diary                            # rebuild the latest session's diary
```

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
foreground-focused crops (segmentation), epochs and patience. Classification uses a ResNet head; segmentation uses a ResNet
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

## Validation-only inference comparison

After training and fixed refinement, `python -m gotrain tta TASK` compares four-flip
probability averaging on validation data and records a new immutable version.
The selected recipe preserves its inference policy and input precision in final
evaluation and prediction reports. This uses four forward passes per image.
A variant is retained only if its measured validation score wins selection.

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

For the live demo, open `live-diary.html` (see *The AI-scientist loop*) beside the
terminal while the agent runs. Each session also keeps its own `diary.html` and
`showcase/` images under `sessions/<session_id>/`, which `export` includes.

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


## Team handoff — 31 October 2026

This section is the working handoff for the remaining hackathon work. The code and
experiment metadata already committed to `main` should be preserved; continue from
the existing implementation rather than rebuilding the project.

### What is already implemented and pushed

- GEO-Bench task support for EuroSAT classification, PV4GER classification,
  PV4GER segmentation and NZ cattle segmentation.
- A Strands-based autonomous agent with bounded experiment tools, explicit
  hypotheses, experiment comparison and best-validation-checkpoint publishing.
- Training and validation pipelines, ResNet-based classification, U-Net-style
  segmentation, aligned image/mask augmentation, train-only normalization,
  dataset-level confusion-matrix metrics and reproducibility metadata.
- Experiment history, immutable run IDs, checksummed checkpoints, warm-start
  fine-tuning, validation-only TTA comparison, HTML report generation and
  submission export.
- Filesystem artifact storage, optional S3 upload/download integration, an MCP
  stdio adapter for GoPilot integration, test-once finalization safeguards,
  automated tests and a GitHub Actions CPU workflow.
- The latest code commit on `main` at the time of this handoff is
  `fb640e8b9f8af2776e52fbfe3665dee0c352bcf2`
  (`Validate inference averaging and preserve input precision through final evaluation`).

### Results that are actually present in GitHub

All entries below are **validation** results. None is a final benchmark test score.

| Evidence | Run / setup | Validation accuracy | Macro F1 | Mean IoU |
|---|---|---:|---:|---:|
| Manual EuroSAT baseline | 5 epochs | 75.00% | — | — |
| Pretrained ResNet18 | `20261031T120129_a50b2160`, best epoch 13 | 96.80% | 96.79% | 93.86% |
| Longer ResNet18 | `20261031T120752_e6ef7de9`, best epoch 20 | **97.10%** | **97.09%** | **94.42%** |
| Pretrained ResNet34 | `20261031T121704_3f24ccc8`, best epoch 13 | 95.90% | 95.88% | 92.20% |

The three experiment summaries, configs, histories and normalization metadata are
under `results/validation-eurosat/`. The manual baseline history is
`results/eurosat-baseline.json`. The 97.10% result is the best recorded EuroSAT
validation result in this repository; it is not evidence of a final test score or
proof that the instructor's reference has been beaten.

### What is not yet published in the repository

The current GitHub tree does not include the following deliverables:

- Final test result files for all four GEO-Bench tasks.
- A complete `submission/` export containing the final scores and agent-session logs.
- The generated HTML reports and prediction examples for the final selected runs.
- The final presentation/slide deck.
- Real S3 round-trip evidence or a confirmed GoPilot deployment through RASID.

Some of these may exist in the SageMaker workspace. Export and verify them before
marking them complete. Checkpoint weights, datasets, credentials and large archives
must remain outside Git; publish their checksummed manifests and approved small
reports/logs instead. The current `.gitignore` intentionally excludes model weights
and datasets.

### Ownership: two workstreams

**Workstream A — Charbel and Marwa: benchmark experiments**

- **Charbel — classification:** sync the SageMaker environment; review the three
  committed EuroSAT validation runs and their configs; confirm the instructor's
  reference score, metric definition, data split and allowed pretraining; then run
  and compare the `m-pv4ger` classification task. Do not rerun EuroSAT final test
  evaluation if its once-only test marker already exists.
- **Marwa — segmentation:** run and compare `m-pv4ger-seg` and `m-nz-cattle`
  sequentially on the shared T4. Check foreground IoU, mean IoU, per-class metrics
  and visual prediction examples. Confirm that masks are aligned and that reported
  IoU is computed from the dataset-wide confusion matrix.

**Workstream B — Rabih: integration, verification and release**

- Pull the latest `main`, run the environment/QA commands below, and resolve any
  integration failures without deleting existing runs or test-once markers.
- Verify run IDs, checkpoint SHA-256 hashes, metrics and confusion matrices against
  the original SageMaker artifacts. Recover any already-completed final test outputs;
  do not repeat a test evaluation just to recreate a missing report.
- Once all development and validation choices are frozen, coordinate the one-time
  finalization of any task that has not yet been tested. Generate reports, run
  `export`, check the submission's missing-task list, and assemble the final
  presentation from verified results only.
- Confirm the real S3 bucket/role with RASID before testing S3. Confirm the required
  transport and authentication with RASID before describing the MCP adapter as a
  completed GoPilot deployment.

The single T4 must be used **sequentially** for model training/evaluation. Coordinate
GPU runs rather than launching multiple training jobs at once.

### Exact continuation sequence

1. In SageMaker, activate the existing environment and sync the repository:

   ```bash
   source ~/gotrain-env/bin/activate
   cd ~/GoTrain-Hackathon
   git pull --rebase origin main
   python -m pip install -e '.[dev]'
   python -m pip check
   ```

2. Verify the environment and current code before starting new experiments:

   ```bash
   python -m gotrain doctor
   python -m pytest -q
   python -m gotrain smoke
   ```

   `tests/test_workflow.py` currently contains 16 test functions. The earlier
   verification record in `VALIDATION.md` documents 14 passing checks for the
   initial implementation; rerun the current suite and record its actual result.
   Do not report 18 passing tests unless that exact run is available.

3. Inspect the existing EuroSAT runs before changing anything:

   ```bash
   python -m gotrain inspect m-eurosat
   python -m gotrain compare m-eurosat
   python -m gotrain report m-eurosat
   ```

   The best recorded validation run is currently
   `20261031T120752_e6ef7de9` at 97.10%. Keep its original run and checkpoint
   intact. Compare any refinement on validation data only.

4. Prepare and run the remaining tasks one at a time:

   ```bash
   python -m gotrain prepare m-pv4ger
   python -m gotrain agent m-pv4ger --experiments 3 --minutes 60
   python -m gotrain compare m-pv4ger
   python -m gotrain report m-pv4ger

   python -m gotrain prepare m-pv4ger-seg
   python -m gotrain agent m-pv4ger-seg --experiments 3 --minutes 90
   python -m gotrain compare m-pv4ger-seg
   python -m gotrain report m-pv4ger-seg

   python -m gotrain prepare m-nz-cattle
   python -m gotrain agent m-nz-cattle --experiments 3 --minutes 90
   python -m gotrain compare m-nz-cattle
   python -m gotrain report m-nz-cattle
   ```

   Adjust batch size/image size only when GPU memory requires it, and record the
   exact recipe and reason. Do not tune against test data.

5. Before final evaluation, confirm with the team whether a task's test split has
   already been opened. The application intentionally blocks a second finalization.
   If a final test has already run, recover its original result and evidence; do not
   remove markers, create a new run root, or rerun it. For tasks not yet finalized,
   freeze the selected validation checkpoint first, then run each exactly once:

   ```bash
   python -m gotrain finalize m-eurosat --confirm-test-once
   python -m gotrain finalize m-pv4ger --confirm-test-once
   python -m gotrain finalize m-pv4ger-seg --confirm-test-once
   python -m gotrain finalize m-nz-cattle --confirm-test-once
   python -m gotrain export --out submission
   ```

6. Inspect `submission/scores.json`. It must list all four task results and have
   `"complete": true` before calling the benchmark submission complete. Keep the
   reports, configs, histories, session logs and checksums together. Review them
   for credentials or private information before sharing.

7. Finish the presentation with: challenge/problem, agent architecture, experiment
   workflow, classification evidence, segmentation examples, the four final test
   results and exact metrics, artifact/versioning evidence, MCP/GoPilot integration
   status, and limitations. Use only results found in the exported evidence.

### Important safeguards

- Training/tuning uses `train` and `valid`; never use `test` to choose a model.
- Do not compare accuracy, foreground IoU or mean IoU as if they were the same metric.
- Do not claim S3 or GoPilot production integration until tested with RASID's actual
  approved bucket, role, transport and authentication.
- Never commit AWS credentials, API keys, datasets, checkpoints or large archives.
- Preserve the existing baseline, run directories, checkpoints and final-test markers.
