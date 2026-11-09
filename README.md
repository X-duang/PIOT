# PIOT: Prior-Informed Optimal Transport

PIOT adapts an entity matcher from labeled source pairs to unlabeled target pairs. The package contains PIOT with automatic constraint selection, the fixed variants PIOT-B/U, source-only, MMD, InvGAN+KD (AdvDA), and the joint feature-label DeepJDOT implementation used in the manuscript.

## Status and limitations

- Dataset CSV files are fetched from the DADER upstream and checksum-verified; their terms are separate from the code license. See `data/README.md`.
- WDC resampling and aggregation scripts are provided for the controlled prior-shift study.
- The supplied server environment is recorded in `environment-server-record.yml`.
- Raw result logs, model checkpoints, caches, and paper source are not included. `results_summary/benchmark_per_run.csv` records the per-seed test F1 values for the two twelve-task benchmark tables; it does not contain the controlled WDC study or the other ablations. `results_summary/deepjdot_faithful.csv` summarizes DeepJDOT runs. These summaries do not replace the full run artifacts.
- DADER-derived files retain their upstream license and attribution; see `THIRD_PARTY_NOTICES.md`.

## Layout

`main/`, `train/`, `metrics/`, `modules/`: training implementation; `manifests/`: experiment configurations; `scripts/`: launchers; `tests/`: available focused tests.

## Dependencies

The supplied Linux server environment used Python 3.10.21, PyTorch 2.5.1, Transformers 4.57.6 and POT 0.9.7.post1. `environment-server-record.yml` preserves its exported packages and channels, with the machine-specific prefix removed. This is an environment record, not a guarantee of identical results across machines or of successful installation on non-Linux systems. The accompanying `requirements.txt` pins the core non-PyTorch packages recorded there.

Create a Python 3.10 environment, install a PyTorch 2.5.1 build suitable for your NVIDIA driver, then run `python -m pip install -r requirements.txt`. For example, a compatible CUDA 12.4 setup can use:

```bash
conda create -n piot python=3.10.21 pip
conda activate piot
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The encoder `bert-base-multilingual-cased` is downloaded by default or loaded from `Models/bert-base-multilingual-cased`.

## Data and launching

Run `python scripts/download_datasets.py` to fetch and verify the exact paper CSVs from the DADER upstream (after line-ending normalization).

From the repository root, after preparing dependencies and data, launch the paper's PIOT benchmark via `bash scripts/run_bench_ot300.sh` or run one task using `python run_experiments.py --manifest manifests/experiment_A_ot300.json --methods ot_balanced ot_unbalanced ot_auto --tasks T1 --results-root results_check`. The batch launcher uses four GPU IDs (override `GPUS`) and runs all five seeds. Other launchers cover 20-epoch and faithful DeepJDOT protocols. `bash scripts/run_wdc_prior_shift.sh` schedules the controlled WDC grid after its prerequisites are available. Review their GPU defaults and output paths before running.

The target pool is unlabeled for adaptation. Main-benchmark checkpoint selection and the optional PIOT retry use target-validation labels; the 20-epoch last-checkpoint protocol disables those interventions. Dataset splits are stratified using labels for evaluation construction.

## Paper configuration notes

The main benchmark manifest sets `ot_max_iter=300` and `ot_tol=1e-7`; direct training entry points have different defaults (2000 and 1e-9). Use the manifest or pass the paper values explicitly. The optional validation-triggered collapse retry uses a best validation F1 threshold of 0.05. These settings are distinct from the fixed-epoch protocol, which disables validation-based checkpoint selection and retry.

The probe samples 20% of each domain without replacement, capped at 2,000 pairs, with stratified source sampling and uniform target sampling. It uses 50 bootstrap resamples, `w=1.96*sd`, `alpha=0.5`, `tau_max=20`, `tau_ref=tau_0=1`, prior clipping at 1e-4, and a 1e-12 surcharge floor. These values are also specified in Section IV-D of the manuscript.

## Benchmark result summary

`results_summary/benchmark_per_run.csv` contains five training seeds per task and method under `target_validation` and `fixed_epoch`. The former uses target-validation checkpoint selection and includes the validation-triggered PIOT retry for FZ→ZY seed 0, marked in the `retry` column. The latter uses the fixed-epoch, final-checkpoint protocol without that retry. All rows report held-out test F1; data splits are fixed across the five training seeds.

Run `python scripts/summarize_benchmark_results.py` to check the seed sets and print each task/method mean and sample standard deviation. For each task, average the five seeds to obtain a method's mean F1; use sample standard deviation for the table's spread. The reported ΔF1 is the difference between source-only and the post-hoc better of PIOT-B and PIOT-U, calculated from unrounded per-task means; group averages are computed across tasks. This file supports the two benchmark tables only; probe decisions, controlled shifts, and other study artifacts are not included.

The optional Experiment B diagnostic aggregation references `analysis.discrepancy` and `analysis.transport_stats`, which are not included in this draft. Its CSV can therefore omit discrepancy and transport-statistic fields; inspect its error columns. F1, precision, and recall fields come from the run summaries and do not require those modules.
