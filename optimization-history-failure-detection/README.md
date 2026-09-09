# Optimization History Failure Detection

Associative optimization-history memory for medical-image failure detection: train a ResNet-18 classifier while recording delta-rule memory matrices, query them label-independent at deployment, and compare a calibration-fitted memory score `z_combined` against normalized predictive entropy `H`.

## Setup

From the repo root:

```bash
pip install -e .
python -c "from optimizer import research_nrm_v2_observe; from research.common import clinical_training"
pytest tests/ -q
```

MedMNIST datasets download automatically on first use into `data/clinical/`.

Optional cheap sanity check before a full DermaMNIST run:

```bash
python research/associative_memory_fft/diagnose_associative_memory.py
```

---



## Notebook pipeline (recommended order)


| Step | Notebook                                                                                   | Role                                                                                    |
| ---- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| 1    | [dermamnist_full_validation.ipynb](notebooks/dermamnist_full_validation.ipynb)             | **Build** frozen classifier + associative memory on DermaMNIST                          |
| 2    | [memory_vector_vs_magnitude_probe.ipynb](notebooks/memory_vector_vs_magnitude_probe.ipynb) | **Primary evaluation** — `z_combined` vs `H`, magnitudes vs full vectors, paper metrics |
| 3    | [bloodmnist_organcmnist_probe.ipynb](notebooks/bloodmnist_organcmnist_probe.ipynb)         | **Cross-dataset** — same probe protocol on BloodMNIST and OrganCMNIST                   |


Step 2 requires artifacts from step 1. Step 3 is independent (trains its own models under `cross_dataset/`).

Retired notebooks live under `notebooks/_archived/` for reference only.

---



## 1. `dermamnist_full_validation.ipynb`

**What it does**

Primary DermaMNIST notebook. Trains ResNet-18 with an associative-memory side channel (four delta-rule levels, FFT off / mean-pool aggregation), saves frozen checkpoints, runs label-free deployment, and optionally runs legacy downstream analyses (memory ID, NRO, deferral ablations).


| Part                     | Sections                    | Purpose                                                                |
| ------------------------ | --------------------------- | ---------------------------------------------------------------------- |
| **I — Build**            | §1 Train, §2 Reconstruction | Train model + `{M^j}`; verify `M^j k` vs training residual `v = y - p` |
| **II — Deploy**          | §D.1                        | Label-free inference: `x → h_T → k_query → z_j = M^j k_query`          |
| **II — Eval (optional)** | §3–§8                       | Memory identification, NRO, deferral comparisons                       |


**Key configuration flags**


| Flag             | Default                             | Meaning                                   |
| ---------------- | ----------------------------------- | ----------------------------------------- |
| `RUN_BUILD`      | `False`                             | Part I: train + reconstruction            |
| `RUN_DEPLOYMENT` | `True`                              | Part II §D.1: deployment pipeline         |
| `RUN_EVALUATION` | `False`                             | Part II §3–§8: deferral / NRO / ablations |
| `FORCE_RETRAIN`  | `False`                             | Delete and retrain existing checkpoints   |
| `ASSOC_CFG`      | `use_fft=True, use_attention=False` | Associative memory toggles                |


**How to run**

1. Open the notebook from `notebooks/` (or set kernel cwd to repo root).
2. Run **§0 Setup** → **Configuration** → **Shared dataset bundle**.
3. **First-time / full build:** set `RUN_BUILD=True`, run §1 and §2 (~1–2 h per seed on CPU for 3 seeds).
4. **Deploy only (cached artifacts):** keep `RUN_BUILD=False`, `RUN_DEPLOYMENT=True`, run **§D.1**.
5. **Optional legacy eval:** set `RUN_EVALUATION=True`, run §3–§8.

**Outputs**


| Path                                                                              | Content                                    |
| --------------------------------------------------------------------------------- | ------------------------------------------ |
| `research/associative_memory_fft/artifacts/checkpoints/`                          | ResNet checkpoints `{task}_seed{seed}.pkl` |
| `research/associative_memory_fft/artifacts/memory/`                               | `*_associative.npz`, `*_checkpoints.npz`   |
| `research/associative_memory_fft/artifacts/memory_reconstruction_diagnostics.csv` | Reconstruction gate (§2)                   |
| `research/associative_memory_fft/deployment/dermamnist/`                          | Deployment records (§D.1)                  |


**Smoke test (no notebook):** `python notebooks/_execute_derma_pipeline.py` (1 seed, 2 epochs, subsampled data).

---



## 2. `memory_vector_vs_magnitude_probe.ipynb`

**What it does**

Primary **failure-detection** evaluation for the paper. Uses trained DermaMNIST artifacts (no retraining). Deploys label-free memory features, fits logistic probes on ID calibration only, and reports held-out test AUROC/AUPRC.


| Phase       | Question                                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** | Do full vectors `[z_1,…,z_4]` beat magnitudes `[‖z_j‖]`? Select `X_best` by **calibration AUROC** (no test leakage). |
| **Phase 3** | Does `z_combined` (probe on `X_best`) beat normalized entropy `H`?                                                   |


Also produces level ablation, ROC/PR, risk–coverage, and confident-wrong summaries (via cached features + `probe_analysis.py`).

**Prerequisites**

- Completed `dermamnist_full_validation.ipynb` §1 (or restored artifacts under `research/associative_memory_fft/artifacts/`).

**Key configuration**


| Variable           | Typical value                               | Meaning                                          |
| ------------------ | ------------------------------------------- | ------------------------------------------------ |
| `ARTIFACT_DIR`     | `research/associative_memory_fft/artifacts` | Trained model + memory                           |
| `TEST_SAMPLE_SIZE` | `2000`                                      | Test subsample per seed (`2000/seed`, not total) |
| `SEEDS`            | `(42, 123, 456)`                            | Must match training seeds                        |
| `FORCE_REDEPLOY`   | `False`                                     | Set `True` to re-run slow deployment inference   |


**How to run**


| Run once                                                    | Re-run when changing test subsample or probes      |
| ----------------------------------------------------------- | -------------------------------------------------- |
| **Configuration** → **Setup** → **Deploy & cache features** | **Eval configuration** → **Phase 1** → **Phase 3** |


1. Run path/config cell, then Setup (loads bundle, defines probe helpers).
2. Run **Deploy & cache features** (writes per-seed CSVs; skip if cache exists and `FORCE_REDEPLOY=False`).
3. Set `TEST_SAMPLE_SIZE` in **Eval configuration**, then run **Phase 1** and **Phase 3**.

Changing `TEST_SAMPLE_SIZE` only subsamples cached test features — seconds, not minutes.

**Outputs**


| Path                                                                     | Content                                                     |
| ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `research/associative_memory_fft/probe_experiments/feature_cache/seed*/` | `cal_features.csv`, `test_features_full.csv`                |
| `research/associative_memory_fft/probe_experiments/probe_results.json`   | Phase 1/3 summaries                                         |
| `research/associative_memory_fft/probe_experiments/*.csv`                | Per-seed metrics                                            |
| `publication/images/`                                                    | Figures (also via `python publication/generate_figures.py`) |


**Headless analysis:** `python research/associative_memory_fft/probe_experiments/probe_analysis.py`

---



## 3. `bloodmnist_organcmnist_probe.ipynb`

**What it does**

Cross-dataset generalization: repeat the DermaMNIST probe protocol on **BloodMNIST** (8 classes) and **OrganCMNIST** (11 classes). For each dataset it (1) trains ResNet-18 + associative memory, (2) deploys cal/test features, (3) compares `z_combined` vs `H` on held-out test.

**Sample caps (matched to DermaMNIST):** train `6408`, cal `1602`, test `2005`; probe eval uses `n=2000` per seed.

**Key configuration**


| Variable         | Typical value                   | Meaning                                    |
| ---------------- | ------------------------------- | ------------------------------------------ |
| `TASKS`          | `("bloodmnist", "organcmnist")` | Datasets to evaluate                       |
| `RUN_BUILD`      | `True`                          | Train if checkpoints missing               |
| `FORCE_RETRAIN`  | `False`                         | Rebuild after changing caps or config      |
| `FORCE_REDEPLOY` | `False`                         | Recompute deployment feature cache         |
| `FAST_MODE`      | `False`                         | `True` → 1 seed, 2 epochs, subsampled data |


**How to run**

1. **§0 Setup** → **Configuration** (verify caps line: `train=6408 cal=1602 test=2005 probe_n=2000`).
2. **§1 Train** — one seed ≈ 1.5–2 h on CPU at full caps; 3 seeds × 2 datasets is an overnight job.
3. **§2 Deploy features + probe** — caches features, runs Phase 1-style representation selection and `z_combined` vs `H`.
4. **§3 Summary** — writes markdown summary.

If you already trained with full (uncapped) MedMNIST splits, set `FORCE_RETRAIN=True` and `FORCE_REDEPLOY=True`.

**Outputs**


| Path                                                            | Content                                                             |
| --------------------------------------------------------------- | ------------------------------------------------------------------- |
| `research/associative_memory_fft/cross_dataset/checkpoints/`    | Per-task checkpoints                                                |
| `research/associative_memory_fft/cross_dataset/probe_features/` | Deployed cal/test features                                          |
| `research/associative_memory_fft/cross_dataset/probe_results/`  | `cross_dataset_probe_summary.csv`, `phase3_per_task_seed.csv`, JSON |


**Smoke test:** `python notebooks/_execute_cross_dataset_pipeline.py`

---



## End-to-end reproduction checklist

```text
[ ] pip install -e . && pytest tests/ -q
[ ] dermamnist_full_validation: RUN_BUILD=True  → §1, §2  (once)
[ ] memory_vector_vs_magnitude_probe: deploy cache → Phase 1 → Phase 3
[ ] bloodmnist_organcmnist_probe: §1 train → §2 probe → §3 summary
[ ] publication/generate_figures.py  (optional, refreshes paper figures)
```

**Runtime (rough, CPU):** DermaMNIST train ~1 h/seed; probe deploy ~minutes/seed once cached; cross-dataset train ~1.5 h/seed/dataset at matched caps.

---



## Further reading

- Method spec and artifact layout: [research/associative_memory_fft/README.md](research/associative_memory_fft/README.md)
- Paper: [publication/main.tex](publication/main.tex)

