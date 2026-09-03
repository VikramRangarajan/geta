# GETA baseline runs — 2026-09-03

All jobs use `run.sh` (`-q standby`, 4h) with `GETA_JOB` selecting the baseline.
Slurm log per job: `logs/<job-name>-<jobid>.out`. Training outputs per job:
`outputs/<GETA_JOB>/` (checkpoints, subnet, per-epoch log). Shared data cache:
`data/` (CIFAR-10 auto-downloaded once). Each job runs in
`outputs/<GETA_JOB>/work` so CWD-relative writes (`outputs/`, `projection_qm_*.txt`)
never collide. Monitor with `squeue -u rangarav` / `tail -f logs/<name>-<id>.out`.

## Submitted (all PENDING at submit time, standby QoS)

| JobID | Slurm name | GETA_JOB | Partition | Paper ref |
|---|---|---|---|---|
| 11671060 | geta-resnet20-cifar10 | resnet20-cifar10 | a10 | Tab 2 (sparsity 0.35, 350ep, SGD 0.1) |
| 11671061 | geta-vgg7-cifar10 | vgg7-cifar10 | a10 | Tab 4 (sparsity 0.7, 200ep, Adam 1e-3, wt+act quant) |
| 11671062 | geta-resnet56-cifar10 | resnet56-cifar10 | a10 | Fig 4 ctx (sparsity 0.5, 200ep) |
| 11671063 | geta-simplevit-cifar10 | simplevit-cifar10 | a10 | Tab 6 (100ep) |
| 11671065 | geta-vit-imagenet | vit-imagenet | a100-80gb | Tab 6 ViT-Small (100ep, sparsity 0.3 assumed) |
| 11671066 | geta-deit-imagenet | deit-imagenet | a100-80gb | Tab 6 DeiT-Tiny (same) |
| 11671067 | geta-swin-imagenet | swin-imagenet | a100-80gb | Tab 6 Swin-Tiny (same) |

ImageNet path: `/scratch/gilbreth/wang4094/imagenet/{train,val}` (ImageFolder layout).
Exact commands per job: `case "${GETA_JOB}"` block in `run.sh`.
Submit example: `sbatch --job-name=geta-vgg7-cifar10
--output=logs/geta-vgg7-cifar10-%j.out --export=ALL,GETA_JOB=vgg7-cifar10 run.sh`

## Path fixes applied (no more dev-machine hardcodes)

- `test_scripts/geta_common.py` (new): repo-root bootstrap, `--output_dir/--data_dir`
  (`$GETA_OUTPUT_DIR`/`$GETA_DATA_DIR`), `check_accuracy` file-based fallback loader,
  `create_exp_dir` fallback (the `common.utils` module never existed in-repo).
- Removed `sys.path.append("/home/xiaoyi/...")` and
  `sys.path.append("/home/davidaponte/...")` from all 7 `Qtest_*.py`; replaced with
  `bootstrap_paths()`. Fixed `Qtest_app.py` default `output_dir` (was an absolute
  `/home/davidaponte/...` path) and `test_scripts/run_script.sh` (was Windows
  `C:/Users/...` paths) — rewritten as a local sequential paper-args runner.
- All file outputs (logs, `*.pt`, `subnet/`, bitwidth PDFs, `metrics_info.txt`,
  CIFAR root, `visualize` out_dir) now live under `--output_dir/--data_dir`.
- `run.sh` rewritten: was missing the actual training command (`wait $!` with
  nothing running); now env-driven (`GETA_JOB`), labeled logs, isolated workdirs,
  keeps `-q standby` and the SIGTERM requeue handler (also fixed its
  `Restarts=[0-9]*****` grep).
- Paper fidelity: VGG7 now uses `WEIGHT_AND_ACTIVATION` quant mode (Tab 4 has Act✓;
  scripts defaulted to weight-only); `lr_quant` set to 1e-4 per Appendix C.
- Env: added missing deps to pixi `geta` feature (`tqdm typer transformers datasets
  h5py einops timm`).

## Caveats / not covered by existing scripts

- Standby jobs are preemptible and scripts have no checkpoint-resume: a preempted
  long run restarts from scratch via the requeue handler. CIFAR jobs may fit in 4h;
  100-epoch ImageNet jobs almost certainly need re-submission under a non-standby
  QoS (or resume support added to the scripts).
- Paper Tab 5 (ResNet50/ImageNet): no ResNet50 backend or script in repo — skipped.
- Paper Tab 3 (BERT/SQuAD) and Fig 3 (Phi2): backends exist (`hf_phi2`) but there is
  no training harness in `test_scripts/` — skipped, needs a new script.
- Paper Tab 6 PVTv2-B2: backend only has `pvt_v2_b0` — skipped, same CLI pattern as
  the other ViT jobs applies.
- ViT/ImageNet sparsity set to 0.3 (paper does not report per-model sparsity there).
