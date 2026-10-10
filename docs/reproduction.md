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

The supported pinned data protocol fixes batch size 1 and gradient accumulation 4. `max_steps` can be reduced but must cover the requested TCUS search and remain at most 500. Weight decay must stay zero to preserve unselected coordinates. 

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
