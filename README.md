# PIOT: Prior-Informed Optimal Transport

PIOT adapts an entity matcher from labeled source pairs to unlabeled target pairs. The package contains PIOT-B/U/A, source-only, MMD, InvGAN+KD (AdvDA), and the joint feature-label DeepJDOT implementation used in the manuscript.

## Status and limitations

- Dataset CSV files are fetched from the DADER upstream and checksum-verified; their terms are separate from the code license. See `data/README.md`.
- WDC resampling and aggregation scripts are provided for the controlled prior-shift study.
- The supplied server environment is recorded in `environment-server-record.yml`.
- Raw result logs, model checkpoints, caches, and paper source are not included. `results_summary/deepjdot_faithful.csv` summarizes the new DeepJDOT runs; it is not a replacement for the full run artifacts.
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

The target pool is unlabeled for adaptation. Main-benchmark checkpoint selection and the optional PIOT-A retry use target-validation labels; the 20-epoch last-checkpoint protocol disables those interventions. Dataset splits are stratified using labels for evaluation construction.
