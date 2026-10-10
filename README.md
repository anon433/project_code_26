# Sparse unlearning reproduction

Code for trajectory-conditioned utility scoring (TCUS), GEC and WAGLE mask selection, followed by sparse GD or NPO on WMDP and MUSE.

Use Python 3.11 on Linux. Install a PyTorch build compatible with your hardware, then:

```bash
pip install -e '.[reproduction]'
bash mask_search.sh --dry-run
bash stage2.sh --dry-run
```

The two entry scripts read separate YAML files:

| Entry script | Configuration | Default |
| --- | --- | --- |
| `mask_search.sh` | `configs/stage1.yaml` | All five budgets, mask seed 3 |
| `stage2.sh` | `configs/stage2.yaml` | Headline budget, seeds 3, 7, 11 |

Set `budgets: headline` or `budgets: all`, and `seeds: [3]` or `seeds: [3, 7, 11]` in either file. Stage 2 uses `mask_seed: 3` by default, so its three training seeds share one mask. TCUS search uses the optimizer and objective specified by Stage 2.

Prepare the pinned models and data, generate masks, then train and evaluate:

```bash
bash mask_search.sh --prepare
bash mask_search.sh
bash stage2.sh
```

Runs are serial and owned by verified GNU screen sessions. Generated files go to the sibling `<checkout-name>-work` directory, configurable with `SPARSE_UNLEARN_WORK_ROOT`. The supplied 7B setup requires two GPUs with at least 80,000 MiB each, a memory limit of at least 200 GiB, and sufficient disk space. Dry runs require no GPU or downloads.

See [reproduction instructions](docs/reproduction.md) for configuration, pinned inputs, output layout, resume behavior and verification.

This distribution contains code, configurations and reproduction instructions. Third-party code retains its required notices in `LICENSE` and `third_party/WAGLE/LICENSE`.
