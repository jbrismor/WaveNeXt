# 🌊 WaveNeXt

**Can Foundation Models Replace Feature Engineering?** A comparative study benchmarking **N-HiTS** and **PatchTST** (exploring **iTransformer** for variable correlation) against the "Window & Flatten" Random Forest and LSTM baselines. This study evaluates performance on **Significant Wave Height ($H_s$)**, **Peak Wave Period ($T_p$)**, and **Wave Direction ($Dir$)**, with a specific focus on **physics-consistent losses** and **zero-shot generalization** across heterogeneous sea states.

## 📌 Overview

This repository implements the research code for the study **"Evaluating the Efficacy of Modern Time-Series Architectures in Operational Wave Forecasting."**

The recent study by *Portillo Juan et al. (2026)* investigated the trade-offs between deep learning and ensemble methods for wave forecasting in the Mediterranean Sea. Their key findings established that:

* **Random Forest (RF)**, when combined with their novel **Window & Flatten (WF)** technique, surprisingly outperformed LSTMs in **short-term forecasts** for Wave Height ($H_s$) and Peak Period ($T_p$).
* **LSTMs** retained superiority for **long-term trends** and were the only viable option for **Wave Direction ($Dir$)**, where RFs failed to capture variability and converged to the mean.
* While feature engineering (WF) improved results for complex variables like $T_p$, both models struggled to accurately capture the full energy of extreme storm events (Dataset K2).

**Our Goal:** To determine if modern, specialized time-series architectures (**N-HiTS**, **PatchTST**) can achieve superior performance across *all* horizons and variables—including the "easy" Significant Wave Height ($H_s$) and the "hard" Period ($T_p$) and Direction ($Dir$)—**using only raw sequential data**, effectively rendering manual windowing techniques and hybrid physics-derived variables obsolete.

## 🔬 Research Hypotheses

1.  **Native Learning vs. Manual Flattening:** The *Portillo Juan et al.* study required manually flattening time windows to make Random Forests competitive. We hypothesize that **PatchTST**, with its native internal patching mechanism, will capture both the short-term accuracy of the WF-RF and the long-term stability of the LSTM without manual data restructuring.
2.  **The Directional Challenge:** The 2026 study found that RF models completely failed at predicting Wave Direction ($Dir$). We hypothesize that **N-HiTS**, utilizing harmonic basis expansion, will naturally model the circular periodicity of direction, outperforming the LSTM baseline which showed degradation in high-energy directional shifts.
3.  **Robustness to Extreme Events:** Both RF and LSTM models in the original study showed significant performance drops during high-energy wave events (Dataset K2). We hypothesize that modern architectures trained on raw sequences will exhibit higher robustness to these "storm shocks," reducing the error gap between calm and energetic sea states.

## 📂 Repository Structure

```
DeepWave-Bench/
├── data/
│   ├── raw/                 # Original Buoy Data (Valencia)
│   ├── processed/           # Standardized .parquet files
│   └── loaders/             # DataLoaders for both "Raw" (Sequential) and "Flattened" (Tabular) formats
│
├── models/
│   ├── baselines/           # The "Defenders" (from 2026 Paper)
│   │   ├── rf_flatten.py    # Random Forest + Window Flattening (The SOTA to beat)
│   │   ├── lstm_standard.py # Standard LSTM (The weak baseline)
│   │   └── physics_eqs.py   # Implementation of Eqs 6-10 (Derived Variables)
│   ├── contenders/          # The "Challengers" (Modern Archs)
│   │   ├── nhits.py         # N-HiTS (Basis Expansion)
│   │   ├── patchtst.py      # PatchTST (Transformer)
│   │   └── dlinear.py       # DLinear (Simple benchmark)
│   └── components/
│       └── circular_loss.py # Von Mises loss implementation
│
├── experiments/
│   ├── 01_baseline_replication/  # Validate we can match Portillo Juan et al.'s numbers
│   ├── 02_raw_modern_comparison/ # Modern models (Raw Input) vs RF (WF Input)
│   └── 03_ablation_windowing/    # Do Modern models improve if we ALSO give them WF? (Optional)
│
├── analysis/
│   ├── comparisons.py       # Stat tests (Diebold-Mariano) to prove significance
│   ├── error_analysis.py    # Analyze where models fail (e.g., during "K3" shifts)
│   ├── polar_plots.py       # Directional accuracy visualization
│   └── seasonality.py       # Extraction of N-HiTS stacks vs Swell Period
│
├── scripts/
│   ├── run_benchmark.py     # Main execution script
│   └── preprocess_paper.py  # Recreate the exact 2026 paper preprocessing
│
├── environment.yml
└── README.md
```

## 🛠️ Experimental Design

The study is structured into four progressive phases, moving from baseline replication to advanced generalization across heterogeneous sea states.

### Phase 1: Replication (The Control)
We first replicate the results of *Portillo Juan et al. (2026) to establish a valid baseline and ensure fair comparison.

* **Models:** Random Forest (RF) & Standard LSTM.
* **Input Engineering:**
    * **Window & Flatten (WF):** Window size $n=6$ flattened into tabular vectors.
    * **Physics-Derived Variables:** Implementation of Eqs 6-10 from the original paper (Wave Power, Wind-Wave Interaction, etc.).

* **Test Sets:** Evaluation on the three specific datasets defined in the paper:
    * **K1:** Calm/Low energy conditions.
    * **K2:** High energy/Storm conditions (The stress test).
    * **K3:** Abrupt transitions (Simulated shock).

### Phase 2: The Challenge (Modern Architectures)

We train the "Challengers" using **standard sequential input**, removing the manual feature engineering step entirely.

* **Models:** **N-HiTS** (Neural Hierarchical Interpolation for Time Series) and **PatchTST** (Patch Time Series Transformer).
* **Input:** Raw time-series window $(H_s, W_s, Dir, T_p)_{t-n...t}$ without derived physics variables.
* **Hypothesis Test:** If `MAE(PatchTST_Raw) < MAE(RF_Flattened)`, we demonstrate that architectural sophistication (native patching/hierarchical stacking) supersedes manual data restructuring.

### Phase 3: The Hard Variables ($T_p$ & $Dir$)
Special focus is placed on advanced techniques for complex oceanographic variables that the baseline RF failed to capture:

* **Wave Direction ($Dir$) with Von Mises Loss:**
    * Standard regression (MSE) fails at the $0^\circ/360^\circ$ discontinuity. We implement a **Von Mises Probabilistic Loss** within N-HiTS, specifically designed for circular data distributions. This allows the model to "wrap around" North correctly, penalizing errors based on angular distance rather than Euclidean distance.

* **Interpretability of Period ($T_p$):**
    * To validate if the model "learns the physics," we extract the **"Seasonality Stack" output from N-HiTS** and regress it against the observed Swell Period. A strong correlation here would prove that the model internally disentangles long-period swell components from noisy wind-sea data (Trend Stack).

* **Physical Consistency Constraint (Provisional):**
    * We explore hard-coding a physical constraint into the loss function: $$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \cdot \text{ReLU}(H_s/L - 0.142)$$. This penalizes predictions where wave steepness exceeds the physical breaking limit ($\frac{1}{10}$), ensuring the model respects fluid dynamics boundaries.

### Phase 4: Generalization (Tier 1 Validation)

To prove that our findings are not specific to the local bathymetry of Valencia (a common criticism of ML wave studies), we extend the benchmark to **three heterogeneous wave regimes**:

1.  **Valencia (Mediterranean):** Short fetch, wind-sea dominated, complex local bathymetry.
2. other
3. other

**Goal:** Demonstrate that **N-HiTS** and **PatchTST** generalize across these regimes *without* requiring site-specific feature engineering (e.g., re-tuning window sizes or physics equations for each port).
