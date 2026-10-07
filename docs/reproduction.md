# Reproduction

## Environment

Use Linux, Python 3.11 and GNU screen. Install a hardware-compatible PyTorch build, then `pip install -e '.[reproduction]'`. Model and dataset access credentials, if required by their providers, belong in the standard Hugging Face configuration outside this source tree. Standard Hugging Face cache variables are supported.

The reference experiments used PyTorch 2.12.1, Transformers 4.45.1 and Accelerate 0.34.2. GPU kernel, dtype and optimizer changes can affect numerical results. The supplied commands use one process with `device_map=balanced`, BF16 and two visible GPUs (`0,1`). MUSE uses `paged_adamw_32bit`; WMDP uses `adamw_torch`.

The runner checks actual cgroup memory and free disk before submission. The 7B setup requires two GPUs with at least 80,000 MiB each and a cgroup memory limit of at least 200 GiB. It waits for 120 GiB memory headroom and 75,000 MiB free on each GPU. Mask search reserves 70 GiB plus missing packed masks; Stage 2 preflight reserves 40 GiB. Pinned models, datasets and caches need additional disk space. Only one method runs at a time. Cache reclamation occurs after subprocesses exit.

## Configure the two stages

`configs/stage1.yaml` and `configs/stage2.yaml` point to each other. To create a separate experiment, copy both files into a directory, keep those relative references, and pass `--config /path/to/stage1.yaml` or `--config /path/to/stage2.yaml` to the corresponding script. Relative command-line paths are resolved from the caller's directory.

Both files support:

- `budgets: headline` or `budgets: all`.
- `seeds: [3]` or `seeds: [3, 7, 11]`.
- `datasets`: any nonempty subset of `wmdpall`, `muse-news`, `muse-books`.

Stage 1 selects `gec`, `wagle`, `tcus-gd`, and/or `tcus-npo`. Its `tcus`, `gec` and `wagle` sections control scoring hyperparameters. Stage 2 selects `gec`, `wagle`, and/or `tcus`, plus `gd` and/or `npo` updaters. Its `training` section controls each dataset's learning rates, loss coefficients, scheduler, optimizer, clipping and update count. Both stages resolve those same numerical settings so TCUS observes the downstream optimizer's actual trajectory.

The supported pinned data protocol fixes batch size 1 and gradient accumulation 4. `max_steps` can be reduced but must cover the requested TCUS search and remain at most 500. Weight decay must stay zero to preserve unselected coordinates. Unknown settings, invalid ranges and incompatible settings fail before submission.

Stage 2's `mask_seed: 3` reuses seed-3 masks for every downstream seed. Set `mask_seed: match` to use a separately searched mask for each seed, and run Stage 1 with the corresponding seeds first. A missing mask causes an error unless its selected Stage 1 campaign has a verified live predecessor session.

Budgets are 1%, 5%, 10%, 20% and 40%. Headline budgets are 10% for WMDP and MUSE News, and 1% for MUSE Books. TCUS selects `round(total_parameters * budget)` coordinates (at least one); GEC/WAGLE use floor. Ties are resolved by parameter order and flat coordinate index.

Default Stage 1 runs 12 scoring processes and produces 60 masks. Default Stage 2 runs 54 training cells plus original-model evaluations. All budgets with seed 3 gives 90 Stage 2 cells; all budgets with three seeds gives 270. The paper's full seed-3 sweep and three-seed headlines can be reproduced by selecting those two Stage 2 matrices in turn; completed shared cells are validated and reused.

## Prepare and execute

```bash
bash mask_search.sh --dry-run
bash stage2.sh --dry-run
bash mask_search.sh --prepare
bash mask_search.sh
bash stage2.sh
```

`--dry-run` composes the selected Hydra recipes and prints the matrix and representative commands. It neither loads models nor downloads inputs or submits jobs. `--prepare` explicitly permits downloading the exact revisions in `configs/reproduction/paper.yaml`, validates their content, creates the immutable pair streams, and exits. Normal submission consumes the cached pinned inputs offline.

To use a different work directory:

```bash
export SPARSE_UNLEARN_WORK_ROOT=/path/outside/checkout/work
bash mask_search.sh --prepare
bash mask_search.sh
```

Use a new work root when changing numerical hyperparameters or source code. Changing only the budget/seed selection reuses compatible masks and completed cells; mask keys include the search seed. The effective numerical configuration, source hash, asset revisions and input hashes are recorded for every campaign.

TCUS uses five actual updates by default, consuming the first 20 downstream forget/retain pairs in training mode. WMDP uses the prefix of its 500-step linear schedule, including its zero-learning-rate first step; MUSE uses a constant schedule. Search scores use the observed parameter displacement, with a per-update positive retain penalty. Stage 2 reloads the original checkpoint and creates a fresh optimizer before applying the fixed Boolean support.

GEC and WAGLE score the original model in evaluation mode with complete independent forget and retain passes. The paired streams preserve each benchmark's published ordering and preprocessing. Official evaluation data is used only after optimization. WMDP evaluates bio, cyber and MMLU; MUSE evaluates knowledge, verbatim memorization and privacy leakage against pinned retain-model logs.

## Outputs and recovery

The work root contains `assets/`, `scores/`, `masks/`, `cells/`, `original/` and `jobs/`. Each job records its request, state and screen log. The shell reports a verified screen session and its log when submission succeeds. Inspect it with `screen -ls` and read the returned log path.

States distinguish planned, queued, running, complete and failed. A queued job has a live runner waiting for masks or resources. Completion requires validated result receipts. Scores and dense checkpoints are deleted only after their corresponding masks or evaluations are sealed; packed masks, metrics and provenance remain available.

Repeat a completed invocation to validate and reuse its receipts. A failed, interrupted or stale public job stops with its state path for inspection; it is not silently restarted. Train, evaluation and mask-materialization boundaries support validated recovery internally. Do not edit receipts or manufacture completion markers. Use a fresh work root to rerun a failed campaign when recovery has not been independently established.

## Configuration checks and source distribution

Validate both stages before submission:

```bash
bash mask_search.sh --dry-run
bash stage2.sh --dry-run
```

These commands check the selected matrix and Hydra configurations without loading models or submitting jobs.

Build an anonymous source ZIP outside the checkout with:

```bash
python -m src.reproduction.distribution --output /path/outside/checkout/reproduction.zip
```

The explicit allowlist excludes Git metadata, caches, runtime artifacts, paper files and local development records. ZIP timestamps and permissions are normalized and every member is checksum-verified. The builder scans machine paths, email addresses and common access-token forms. Optional `--private-terms /path/to/private.json` adds a local JSON list of identifying terms; keep that list outside the source tree. Inspect the final archive before submitting it. Required dependency license notices are retained.
