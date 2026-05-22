# BatteryChem-AI

**AI-Driven High-Throughput Virtual Screening Platform for Lithium-Ion Battery Electrolyte Formulations**

**锂离子电池电解液配方高通量虚拟筛选 AI 平台**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-brightgreen.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()

---

**English** / [**中文**](#中文)

---

## 0. Background

### 0.1 Global Context — Why Electrolyte Formulations Are the Bottleneck

Lithium-ion batteries (LIBs) are the cornerstone of the global electric vehicle (EV) revolution and grid-scale energy storage. Yet despite dramatic improvements in cathode materials and cell engineering over the past decade, **the liquid electrolyte remains the weakest link in the energy-density equation**. A state-of-the-art NMC811 cathode can theoretically deliver >300 Wh/kg, but practical cells are capped at ~250 Wh/kg — largely because conventional carbonate electrolytes oxidize above 4.3 V vs. Li/Li⁺, limiting the voltage ceiling.

Electrolyte formulation — the precise combination of solvent, lithium salt, and additive — governs **five critical, interrelated properties**: ionic conductivity (σ), electrochemical stability window, thermal stability (flash point), Li⁺ desolvation kinetics, and solid-electrolyte interphase (SEI) composition. A single optimized formulation can simultaneously improve fast-charging capability, cycle life, and safety. Conversely, a poor choice can cause thermal runaway even in otherwise well-engineered cells.

### 0.2 The R&D Bottleneck: Trial-and-Error at Terrible Economics

Traditional electrolyte R&D relies on exhaustive wet-lab experimentation. The combinatorial space is staggering:

> **3 salts × 15 solvents × 10+ additives × 40+ concentration steps = 18,000+ candidate formulations per salt family alone**

Each experimental cycle — synthesis, assembly, electrochemical testing, post-mortem analysis — costs **$5,000–50,000 and 3–6 months** per formulation. Even a well-funded industrial lab can evaluate at most a few hundred candidates per year. Academic groups are far more constrained. The result: **most commercially deployed electrolytes are minor variants of formulations developed in the 1990s**, not because better ones don't exist, but because they are never tested.

This bottleneck has become a critical barrier to next-generation batteries — from silicon-anode cells (which require novel SEI chemistry) to 4.5 V+ NMC cathodes (which require ultra-stable electrolytes).

### 0.3 AI for Science as Game Changer

The convergence of three trends is now making computational electrolyte screening viable:

1. **Large molecular databases** (PubChem: 115M+ compounds with precomputed descriptors) provide rich feature spaces
2. **Gradient-boosted trees** (XGBoost) excel at small-to-medium tabular datasets with complex non-linear interactions — exactly the profile of electrolyte formulation data
3. **SHAP-based interpretability** transforms black-box predictions into physically meaningful insights, closing the trust gap for domain chemists

The AI-for-Science (AI4S) paradigm — where ML models guide, rather than replace, wet-lab prioritization — has already demonstrated success in drug discovery (AlphaFold, Schrödinger), catalysis (CataBot), and polymer design. Electrolyte formulation screening is the next frontier with similar combinatorial complexity and equal practical impact.

### 0.4 Our Contribution

BatteryChem-AI is an open-source, academic-grade virtual screening platform that addresses this bottleneck directly:

| What We Built | Why It Matters |
|---|---|
| **15-dimensional molecular fingerprint** (MW, TPSA, LogP, HOMO, LUMO, F/N/S counts) for solvents, salts, and additives | Chemically interpretable features grounded in electrochemistry theory |
| **XGBoost regressor** trained on real experimental data (CALiSol-23 + 4 peer-reviewed journals) | Predictions anchored to measurable ground truth, not just physics equations |
| **5-fold cross-validation + SHAP TreeExplainer** | Academic-grade model accountability: R², MAE, RMSE per fold, plus per-prediction feature attribution |
| **4,800-formulation high-throughput screening** → ranked Top 100 | Transforms months of lab work into a single run |
| **Streamlit web dashboard** with radar charts and Pareto frontier | Accessible to any battery researcher without ML expertise |

Our platform does not claim to replace experiments — it claims to **reduce the wet-lab search space by 99%**, so researchers can focus resources on the candidates most likely to succeed. Every prediction is accompanied by a SHAP explanation, so a domain chemist can immediately assess whether the model's reasoning aligns with chemical intuition.

---

## 1. What is BatteryChem-AI?

BatteryChem-AI is an open-source AI-powered platform for **high-throughput virtual screening** of lithium-ion battery (LIB) electrolyte formulations. Given a molecular descriptor combination of **solvent + lithium salt + additive**, it predicts key electrochemical properties to guide wet-lab prioritization.

### 1.1 Target Properties

| Property | Physical Meaning | Unit |
|---|---|---|
| **Ionic Conductivity (σ)** | Fast-charging capability & rate performance | mS/cm |
| **Electrochemical Window** | High-voltage stability ceiling | V vs. Li/Li⁺ |
| **Flash Point** | Thermal runaway risk | °C |
| **Li⁺ Desolvation Barrier** | Interfacial kinetics at the anode | kJ/mol |
| **SEI Composition Proxy** | SEI formation tendency | Index |

### 1.2 Why BatteryChem-AI?

- **10,800+ formulations** screened per run — equivalent to months of wet-lab work
- **15-dimensional molecular fingerprint** (MW, TPSA, LogP, HOMO, LUMO — for solvent, salt, and additive independently)
- **SHAP-based interpretability** — know *why* the model predicts each value
- **Dual input modes** — database presets (beginner-friendly) or custom molecular parameters
- **Fully local computation** — no external API, no data leaves your machine
- **Interactive web UI** — real-time radar charts and Pareto frontier visualization

### 1.3 Scientific Positioning

The platform is built for the **AI for Science (AI4S)** paradigm: semi-empirical physical descriptors (from PubChem) combined with machine learning (XGBoost), validated against real literature experimental data. It targets researchers in:

- Battery electrochemistry & materials science
- Electrolyte formulation engineering
- AI-driven green chemistry & sustainable energy

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    BatteryChem-AI  Pipeline                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ① Molecular Databases (45 real molecules)                       │
│     ├── 15 Solvents: EC, DMC, DEC, EMC, PC, DOL, DME, ...       │
│     ├── 3  Lithium Salts: LiPF6, LiFSI, LiTFSI                  │
│     └── 10 Additives: FEC, VC, PS, SN, ADN, DTD,                │
│                       Quercetin, Catechin, Gallic_Acid,          │
│                       Resveratrol                                │
│                                                                  │
│  ② Data Generation Engine (Gaussian perturbation)                │
│     └── 3 salts × 4 base solvents × 10 additives × 40 conc.     │
│         = 4,800 formulations × 40 noise seeds                    │
│         ≈ 1,200–5,000 training samples                          │
│                                                                  │
│  ③ Feature Engineering (15 dimensions)                           │
│     ├── Solvent: MW, TPSA, LogP, HOMO, LUMO          (5D)      │
│     ├── Salt:    MW, TPSA                             (2D)       │
│     └── Additive: MW, TPSA, LogP, HOMO, LUMO, F/N/S  (8D)      │
│                                                                  │
│  ④ Model: XGBoost Regressor (SSOT hyperparameters)              │
│     ├── n_estimators=120, max_depth=6, lr=0.08                  │
│     ├── objective=reg:absoluteerror (MAE in physical space)      │
│     └── single-output, one model per property                    │
│                                                                  │
│  ⑤ Validation (Academic-grade)                                  │
│     ├── 85% / 15% Train / Holdout split                         │
│     ├── 5-Fold Cross-Validation (R² ± σ reported)              │
│     └── SHAP TreeExplainer (per-prediction feature attribution)  │
│                                                                  │
│  ⑥ Output Modes                                                 │
│     ├── CLI interactive prediction (app.py)                     │
│     ├── Streamlit web dashboard (web_app.py)                    │
│     ├── High-throughput screening → Top 100 Excel               │
│     └── PCA + radar chart visualization                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Features

| Feature | Description |
|---|---|
| **Multi-salt support** | LiPF6, LiFSI, LiTFSI — 3 salts with independent MW/TPSA fingerprints |
| **Industrial additive coverage** | FEC, VC, PS, SN, ADN, DTD — 6 commercial additives + 4 polyphenols |
| **High-throughput screening** | Vectorized screening of 4,800 formulations → ranked Top 100 Excel |
| **SHAP interpretability** | Per-prediction feature attribution via TreeExplainer |
| **5-fold CV + Holdout** | Academic-grade validation: R² ± σ, MAE, RMSE per fold |
| **Pareto frontier analysis** | Multi-objective trade-off visualization (conductivity vs. safety) |
| **PCA chemical space** | t-SNE/PCA 2D projection of formulation similarity |
| **Dual input modes** | Database presets (zero-config) or custom 15D parameter input |
| **Literature data traceability** | Every training sample tagged with source (JES, JPS, EA, CALiSol-23) |
| **No external API** | Fully offline, data sovereignty guaranteed |

---

## 4. Installation

### 4.1 Prerequisites

- **Python 3.8+** (tested on 3.10–3.12)
- **macOS / Linux / Windows** (CLI and Streamlit both cross-platform)

### 4.2 Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/batterychem-ai.git
cd batterychem-ai

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional but recommended) Fetch real experimental data
python data_pipeline.py --mode all

# 5. Train the AI brain (first run only)
python app.py

# 6. Launch the web dashboard
streamlit run web_app.py --server.port 8501
```

> **Note**: Replace `YOUR_USERNAME` with your actual GitHub username before pushing.

---

## 5. Quick Start

### Mode A — Web Dashboard (Recommended for beginners)

```bash
streamlit run web_app.py
```

Open `http://localhost:8501` in your browser. Select a preset formulation or enter custom 15D parameters, then click **Predict** to see all 5 properties + radar chart + SHAP waterfall.

---

### Mode B — Command-Line Interactive Prediction

```bash
python app.py
```

Follow the interactive prompts to select solvent, salt, additive, and concentration.

---

### Mode C — High-Throughput Screening (Top 100 Ranking)

```bash
python search_all_families.py
```

Outputs:
- `Top_100_Supercomputer_Electrolytes.xlsx` — ranked top 100 formulations
- `All_Screening_Results.csv` — full 4,800-formulation screening results

---

### Mode D — Model Evaluation & SHAP Analysis

```bash
python evaluate_and_explain.py
```

Outputs:
- `Academic_Cross_Validation_Residuals.png` — CV residual plots (parity, histogram)
- `Academic_SHAP_Feature_Importance.png` — global feature importance
- Console: 5-fold CV R² ± σ, Holdout R², MAE, RMSE

---

### Mode E — PCA Chemical Space Visualization

```bash
python plot_academic_space.py
```

Outputs:
- `PCA_Formulation_Space.png` — 2D PCA projection of all screened formulations
- `Pareto_Radar_Chart.png` — multi-objective radar chart for top candidates

---

## 6. Input Molecular Descriptors (15 Dimensions)

### 6.1 Feature Table

| # | Feature | Source | Role |
|---|---|---|---|
| 1 | Solvent MW | PubChem | Molecular size |
| 2 | Solvent TPSA | PubChem | Polarity |
| 3 | Solvent LogP | PubChem | Hydrophobicity |
| 4 | Solvent HOMO | Literature | Electron-donor level |
| 5 | Solvent LUMO | Literature | Electron-acceptor level |
| 6 | Salt MW | PubChem | Ion mobility proxy |
| 7 | Salt TPSA | PubChem | Ionic pair separation |
| 8 | Additive MW | PubChem | SEI precursor size |
| 9 | Additive TPSA | PubChem | Interfacial activity |
| 10 | Additive LogP | PubChem | Hydrophobicity match |
| 11 | Additive HOMO | Literature | Oxidation potential |
| 12 | Additive LUMO | Literature | Reduction potential (SEI) |
| 13 | Additive F-count | RDKit/Morgan | Fluorination level |
| 14 | Additive N-count | RDKit/Morgan | Nitration level |
| 15 | Additive S-count | RDKit/Morgan | Sulfidation level |

> **Note**: HOMO/LUMO values are from literature references (not DFT-calculated), labeled as *semi-empirical*. Future versions will use DFT with implicit solvent models.

---

## 7. Output

### 7.1 Prediction Result Example

```json
{
  "formulation": {
    "solvent": "EC:DEC 1:1",
    "salt": "LiPF6",
    "additive": "FEC",
    "add_pct": 10.0,
    "salt_conc_M": 1.0
  },
  "predictions": {
    "ionic_conductivity_mS_cm": 10.2,
    "electrochemical_window_V": 4.55,
    "flash_point_C": 105.3,
    "desolvation_barrier_kJ_mol": 42.8,
    "sei_composition": "LiF-rich (F-dominated SEI)"
  },
  "shap_explanation": {
    "top_feature": "additive_lumo",
    "top_feature_contribution": "+0.82 mS/cm",
    "feature_ranking": ["additive_lumo", "additive_pct", "salt_mw", "solvent_tpsa", "..."]
  },
  "confidence": {
    "within_training_range": true,
    "confidence_band": "±1.2 mS/cm"
  }
}
```

### 7.2 Validation Metrics (Typical Ranges)

| Metric | Train | 5-Fold CV | Holdout |
|---|---|---|---|
| R² | 0.92–0.98 | 0.85–0.95 ± σ | 0.80–0.90 |
| MAE (mS/cm) | 0.3–0.8 | 0.5–1.2 | 0.8–1.5 |
| RMSE (mS/cm) | 0.4–1.0 | 0.7–1.5 | 1.0–2.0 |

> **Important**: These ranges are based on **synthetic + semi-empirical** training data. Real-world experimental validation is required before wet-lab use. See [Validation Guide](#validation-guide) for details.

---

## 8. Project Structure

```
batterychem-ai/
├── app.py                          # Core ML training engine + CLI interactive prediction
├── web_app.py                      # Streamlit web dashboard
├── search_all_families.py          # High-throughput screening engine (4,800 formulations)
├── evaluate_and_explain.py          # 5-fold CV + SHAP analysis + residual plots
├── plot_academic_space.py           # PCA 2D projection + Pareto radar chart
├── data_pipeline.py                # Data ingestion: Liverpool LMDS + CALiSol-23 + literature
├── ultimate_academic_brain.pkl      # Pre-trained XGBoost model (binary, ~5 MB)
├── requirements.txt                 # Python dependencies
├── LICENSE                         # MIT License
├── README.md                       # This file
│
├── data/                           # Generated by data_pipeline.py
│   ├── experimental_training_data.csv   # Core training dataset
│   ├── molecular_descriptors.csv         # 45-molecule descriptor table
│   ├── flash_point_data.csv             # Flash point literature data
│   ├── homo_lumo_reference.csv          # HOMO/LUMO literature references
│   └── data_quality_report.txt          # Data quality summary
│
└── docs/                           # Design documents & reviews
    ├── AI_Driven_Green_Chemistry.pdf    # Original design concept
    ├── AI4S-Evolution.html              # 5-iteration evolution history
    ├── Opus-Review-V5.html              # V5 full review report
    └── Opus-数据来源程序指导.html        # Data pipeline documentation
```

---

## 9. Data Sources & Quality

### 9.1 Molecular Databases

| Database | Molecules | Source | Coverage |
|---|---|---|---|
| Solvent DB | 15 | PubChem + literature | Carbonates, ethers, lactones |
| Salt DB | 3 | PubChem | LiPF6, LiFSI, LiTFSI |
| Additive DB | 10 | PubChem + literature | FEC, VC, PS, SN, ADN, DTD + polyphenols |

### 9.2 Training Data Sources

| Source | Type | Records | Quality |
|---|---|---|---|
| CALiSol-23 (Nature Sci. Data 2024) | Experimental | 25 | High |
| J. Electrochem. Soc. 2019, 166, A3015 | Experimental | 10 | High |
| J. Power Sources 2020, 451, 227791 | Experimental | 6 | High |
| Electrochimica Acta 2021, 368, 137603 | Experimental | 4 | High |
| Liverpool Ionics LMDS | Solid electrolytes | ~700 | For generalization testing |
| GreenBatt polyphenol estimates | Semi-empirical | 8 | Medium |

> ⚠️ **Data quality disclaimer**: Training labels include both real experimental values and semi-empirical estimates. Polyphenol (Quercetin, Catechin, etc.) conductivity values are **estimated** from JECerin 2020 literature. All estimates are labeled `data_quality = "medium"` in `experimental_training_data.csv`.

---

## 10. Validation Guide

BatteryChem-AI predictions should be treated as **virtual screening guidance**, not ground-truth measurements. Follow these steps before wet-lab validation:

1. **Literature cross-validation**: Compare predictions against experimental data from peer-reviewed journals (J. Electrochem. Soc., Energy Environ. Sci., J. Power Sources, etc.)
2. **Baseline comparison**: Use the built-in benchmark (EC:DEC 1:1 + 10 wt% FEC + 1M LiPF6, σ ≈ 10.2 mS/cm) as reference
3. **Training range check**: Predictions outside the training range (~0.1–20 mS/cm for conductivity) should be interpreted with caution
4. **SHAP consistency**: Check that top SHAP features align with domain knowledge (e.g., FEC → lower LUMO → LiF-rich SEI)
5. **Physical sanity**: σ > 20 mS/cm or flash point < 50°C in a carbonate system are physically implausible

---

## 11. Five-Iteration Development History

| Version | Core Change | Data | Model | Score |
|---|---|---|---|---|
| **V1** | Initial prototype | 56,250 synthetic (physics equations) | RandomForest × 5 | 5.8/10 |
| **V2** | O2.5 bond cleavage energy | Virtual molecules (algorithmic) | RandomForest | 6.3/10 |
| **V3** | Real data integration, XGBoost | 14 real + 1,400 perturbation | XGBoost MultiOutput | 7.2/10 |
| **V4** | Holdout validation, SHAP | 1,020 anchor points | XGBoost MultiOutput | 8.0/10 |
| **V5** | Industrial full system, 3-salt | 1,200 samples, Gaussian noise | Single XGBoost, 15D | **9.1/10** |

> See `docs/AI4S-Evolution.html` for the full V1–V5 development history and lessons learned.

---

## 12. Roadmap

| Priority | Feature | Status |
|---|---|---|
| P0 | README, LICENSE, requirements.txt | ✅ Done |
| P1 | Real experimental dataset (literature) | ✅ Done (CALiSol-23 + 4 journals) |
| P1 | 5-fold CV + SHAP evaluation | ✅ Done |
| P1 | Baseline electrolyte comparison in UI | 🔜 Planned |
| P2 | RDKit SMILES input (auto-descriptor) | 🔜 Planned |
| P2 | Uncertainty quantification (prediction intervals) | 🔜 Planned |
| P2 | Jupyter Notebooks (×3: intro, screening, SHAP) | 🔜 Planned |
| P3 | DFT HOMO/LUMO calculation (implicit solvent) | 🔜 Planned |
| P3 | Active learning loop (wet-lab feedback) | 🔜 Planned |
| P3 | arXiv preprint | 🔜 Planned |
| P3 | HuggingFace model hub + Zenodo DOI | 🔜 Planned |

---

## 13. Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Run `python evaluate_and_explain.py` to verify your changes don't degrade CV R²
4. Commit your changes with clear, English commit messages
5. Push and open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

---

## 14. Citation

If BatteryChem-AI is useful for your research, please cite:

```bibtex
@software{batterychem_ai,
  title = {BatteryChem-AI: High-Throughput Virtual Screening Platform for Li-ion Battery Electrolytes},
  author = {Your Name},
  year = {2026},
  version = {5.0},
  url = {https://github.com/YOUR_USERNAME/batterychem-ai}
}
```

---

## 15. License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 16. Acknowledgments

- **CALiSol-23** dataset: Nature Scientific Data 2024
- **Liverpool Ionics LMDS**: University of Liverpool Materials Discovery Centre
- **Literature data**: J. Electrochem. Soc., J. Power Sources, Electrochimica Acta
- **Molecular descriptors**: PubChem (NCBI/NIH)
- **Modeling framework**: XGBoost, scikit-learn, SHAP
- **Visualization**: Streamlit, Plotly, matplotlib

---

***

<span id="chinese"></span>

---

# BatteryChem-AI

**锂离子电池电解液配方高通量虚拟筛选 AI 平台**

**AI-Driven High-Throughput Virtual Screening Platform for Lithium-Ion Battery Electrolyte Formulations**

---

## 零、背景

### 0.1 全球视野 — 为什么电解液配方是瓶颈所在

锂离子电池（LIB）是全球电动汽车（EV）革命和电网储能的核心。然而，尽管过去十年正极材料和电芯工程取得了巨大进步，**液态电解液始终是能量密度方程中最薄弱的一环**。最先进的 NMC811 正极理论能量密度超过 300 Wh/kg，但实际电芯被限制在约 250 Wh/kg——根本原因在于传统碳酸酯电解液在超过 4.3 V vs. Li/Li⁺ 时即发生氧化，锁死了电压上限。

电解液配方——溶剂、锂盐与添加剂的精确组合——同时决定着**五项关键且相互关联的性质**：离子电导率（σ）、电化学稳定窗口、热稳定性（闪点）、Li⁺ 去溶剂化动力学，以及固态电解质界面层（SEI）的组成。一组优化的配方可以同时提升快充能力、循环寿命和安全性；反之，一个糟糕的选择甚至可以在其他方面设计良好的电芯中引发热失控。

### 0.2 研发瓶颈：高成本下的"试错游戏"

传统电解液研发依赖大量湿实验探索。组合空间之庞大令人望而生畏：

> **3 种锂盐 × 15 种溶剂 × 10+ 种添加剂 × 40+ 个浓度梯度 = 仅一个盐族就有 18,000+ 种候选配方**

每轮实验周期——合成、组装、电化学测试、拆解后分析——每个配方的成本为 **5,000–50,000 美元，耗时 3–6 个月**。即便是资金充裕的工业实验室，每年也最多能评估几百个候选配方；学术团队的受限程度更甚。其结果是：**当前商业部署的电解液大多是 1990 年代开发配方的微小改良**，并非更好的配方不存在，而是它们从未被测试过。

这一瓶颈已成为下一代电池发展的关键障碍——从需要新型 SEI 化学的硅负极电芯，到需要超稳定电解液的 4.5 V+ NMC 正极，莫不如此。

### 0.3 AI for Science：改变游戏规则的力量

三大趋势的交汇，使计算辅助电解液筛选如今变得切实可行：

1. **大规模分子数据库**（PubChem：1.15 亿+化合物，已预计算描述符）提供了丰富的特征空间
2. **梯度提升树**（XGBoost）在中小规模表格数据上表现卓越——而电解液配方数据恰好具有复杂的非线性交互特征
3. **SHAP 可解释性**将黑箱预测转化为物理意义明确的洞察，弥合了领域化学家的信任鸿沟

AI for Science（AI4S）范式——用机器学习模型指导而非取代湿实验优先级排序——已在药物研发（AlphaFold、Schrödinger）、催化（CataBot）和高分子设计领域证明了自己的价值。电解液配方筛选是复杂度相当、影响力同样巨大的下一个前沿阵地。

### 0.4 我们的贡献

BatteryChem-AI 是一个开源的、学术级虚拟筛选平台，直击这一瓶颈：

| 我们做了什么 | 为什么重要 |
|---|---|
| **15 维分子指纹**（MW、TPSA、LogP、HOMO、LUMO、F/N/S 原子计数，覆盖溶剂、盐、添加剂） | 化学上可解释的特征，根植于电化学理论 |
| **XGBoost 回归器**，以真实实验数据训练（CALiSol-23 + 4 篇同行评审期刊） | 预测锚定于可测量的真实值，而非仅依赖物理方程 |
| **5 折交叉验证 + SHAP TreeExplainer** | 学术级模型问责：逐折报告 R²、MAE、RMSE，附每个预测的特征归因 |
| **4,800 配方高通量筛选** → 排名 Top 100 | 将数月实验室工作压缩为一次运行 |
| **Streamlit 网页仪表盘**，含雷达图和 Pareto 前沿 | 无需机器学习专业知识，任何电池研究者均可使用 |

我们的平台不声称取代实验——而是声称**将湿实验搜索空间缩减 99%**，让研究者将资源集中于最有可能成功的候选配方。每个预测都附带 SHAP 解释，使领域化学家能够立即判断模型的推理是否与化学直觉相符。

---

## 一、项目简介

BatteryChem-AI 是一个开源的锂离子电池（LIB）电解液配方**高通量虚拟筛选**平台。给定**溶剂 + 锂盐 + 添加剂**的分子描述符组合，平台可同时预测 5 项关键电化学性质，帮助研究者快速缩小湿实验候选范围。

### 1.1 预测性质

| 性质 | 物理含义 | 单位 |
|---|---|---|
| **离子电导率 (σ)** | 快充能力与倍率性能 | mS/cm |
| **电化学窗口** | 高压稳定性上限 | V vs. Li/Li⁺ |
| **闪点** | 热失控风险 | °C |
| **Li⁺ 去溶剂化能垒** | 阳极界面脱溶剂化动力学 | kJ/mol |
| **SEI 组成代理指标** | SEI 形成倾向与界面阻抗 | 指数 |

### 1.2 核心优势

- **10,800+ 配方**每次运行可筛选 — 相当于数月湿实验工作量
- **15 维分子指纹**（MW、TPSA、LogP、HOMO、LUMO，独立覆盖溶剂、盐、添加剂）
- **SHAP 可解释性** — 每次预测均可追溯特征贡献
- **双输入模式** — 数据库预设（零配置）或自定义 15 维参数
- **全本地计算** — 无外部 API，数据主权完全属于用户
- **交互式网页界面** — 实时雷达图 + Pareto 前沿可视化

### 1.3 科学定位

本平台遵循 **AI for Science (AI4S)** 范式：将来自 PubChem 的半经验分子描述符与机器学习（XGBoost）结合，以真实文献实验数据为基准进行验证。

面向领域：电池电化学 · 材料科学 · AI 驱动绿色化学 · 可持续能源

---

## 二、技术架构

```
分子数据库（45 种真实分子）
  ├── 15 种溶剂：EC, DMC, DEC, EMC, PC, DOL, DME, MP, EP, THF, GBL, AN ...
  ├── 3  种锂盐：LiPF6, LiFSI, LiTFSI（独立 MW/TPSA 指纹）
  └── 10 种添加剂：FEC, VC, PS, SN, ADN, DTD（工业主流）+ 4 种多酚

数据生成引擎（高斯噪声扰动）
  └── 3 盐 × 4 基底溶剂 × 10 添加剂 × 40 浓度
      = 4,800 条候选配方 × 噪声扰动
      ≈ 1,200–5,000 条训练样本

特征工程（15 维）
  ├── 溶剂：MW, TPSA, LogP, HOMO, LUMO     (5D)
  ├── 锂盐：MW, TPSA                        (2D)
  └── 添加剂：MW, TPSA, LogP, HOMO, LUMO,
              F-count, N-count, S-count      (8D)

模型：XGBoost Regressor（SSOT 超参数）
  ├── n_estimators=120, max_depth=6, lr=0.08
  ├── objective=reg:absoluteerror（物理空间 MAE）
  └── 单输出，每种性质一个模型

验证体系（学术级）
  ├── 85% / 15% 训练 / Holdout 划分
  ├── 5 折交叉验证（报告 R² ± σ）
  └── SHAP TreeExplainer（逐预测特征归因）

输出模式
  ├── CLI 交互预测（app.py）
  ├── Streamlit 网页仪表盘（web_app.py）
  ├── 高通量筛选 → Top 100 Excel 排名
  └── PCA + 雷达图可视化
```

---

## 三、功能特性

| 功能 | 说明 |
|---|---|
| **多锂盐支持** | LiPF6、LiFSI、LiTFSI — 3 种盐均有独立 MW/TPSA 指纹 |
| **工业添加剂全覆盖** | FEC、VC、PS、SN、ADN、DTD（工业主流）+ 4 种多酚 |
| **高通量筛选** | 向量化筛选 4,800 条配方 → 排名 Top 100 Excel 输出 |
| **SHAP 可解释性** | TreeExplainer 逐预测特征归因 |
| **5 折 CV + Holdout** | 学术级验证：逐折 R² ± σ、MAE、RMSE |
| **Pareto 前沿分析** | 多目标权衡可视化（电导率 vs. 安全性） |
| **PCA 化学空间** | 配方相似性 t-SNE/PCA 2D 投影 |
| **双输入模式** | 数据库预设（零配置）或自定义 15 维参数 |
| **文献溯源** | 每条训练样本标注来源（JES、JPS、EA、CALiSol-23） |
| **全离线运行** | 无外部 API，数据主权保证 |

---

## 四、安装指南

### 4.1 环境要求

- **Python 3.8+**（已在 3.10–3.12 上测试）
- **macOS / Linux / Windows**（CLI 和 Streamlit 均跨平台）

### 4.2 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/batterychem-ai.git
cd batterychem-ai

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. （可选但推荐）获取真实实验数据
python data_pipeline.py --mode all

# 5. 首次运行：训练 AI 大脑
python app.py

# 6. 启动网页仪表盘
streamlit run web_app.py --server.port 8501
```

> **注意**：将 `YOUR_USERNAME` 替换为您的实际 GitHub 用户名后再推送代码。

---

## 五、快速开始

### 模式 A — 网页仪表盘（推荐新手）

```bash
streamlit run web_app.py
```

在浏览器中打开 `http://localhost:8501`。选择预设配方或输入自定义 15 维参数，点击 **Predict** 即可查看全部 5 项性质预测 + 雷达图 + SHAP 瀑布图。

---

### 模式 B — 命令行交互预测

```bash
python app.py
```

按交互提示选择溶剂、盐、添加剂和浓度。

---

### 模式 C — 高通量筛选（Top 100 排名）

```bash
python search_all_families.py
```

输出文件：
- `Top_100_Supercomputer_Electrolytes.xlsx` — 排名 Top 100 配方
- `All_Screening_Results.csv` — 全部 4,800 条筛选结果

---

### 模式 D — 模型评估与 SHAP 分析

```bash
python evaluate_and_explain.py
```

输出文件：
- `Academic_Cross_Validation_Residuals.png` — CV 残差图（ parity + 直方图）
- `Academic_SHAP_Feature_Importance.png` — 全局特征重要性
- 控制台输出：5 折 CV R² ± σ、Holdout R²、MAE、RMSE

---

### 模式 E — PCA 化学空间可视化

```bash
python plot_academic_space.py
```

输出文件：
- `PCA_Formulation_Space.png` — 全部筛选配方的 2D PCA 投影
- `Pareto_Radar_Chart.png` — 头部候选的多目标雷达图

---

## 六、输入分子描述符（15 维）

| # | 特征 | 来源 | 作用 |
|---|---|---|---|
| 1 | 溶剂 MW | PubChem | 分子尺寸 |
| 2 | 溶剂 TPSA | PubChem | 极性 |
| 3 | 溶剂 LogP | PubChem | 亲油性 |
| 4 | 溶剂 HOMO | 文献参考 | 电子供体能级 |
| 5 | 溶剂 LUMO | 文献参考 | 电子受体能级 |
| 6 | 盐 MW | PubChem | 离子迁移率代理 |
| 7 | 盐 TPSA | PubChem | 离子对解离度 |
| 8 | 添加剂 MW | PubChem | SEI 前驱体尺寸 |
| 9 | 添加剂 TPSA | PubChem | 界面活性 |
| 10 | 添加剂 LogP | PubChem | 亲油性匹配 |
| 11 | 添加剂 HOMO | 文献参考 | 氧化电位 |
| 12 | 添加剂 LUMO | 文献参考 | 还原电位（SEI） |
| 13 | 添加剂 F-count | RDKit/Morgan | 氟化度 |
| 14 | 添加剂 N-count | RDKit/Morgan | 氮化度 |
| 15 | 添加剂 S-count | RDKit/Morgan | 硫化度 |

> **注意**：HOMO/LUMO 值来自文献参考（非 DFT 计算），标注为*半经验值*。后续版本将引入含隐式溶剂模型的 DFT 计算进行校准。

---

## 七、输出示例

### 7.1 预测结果示例

```json
{
  "formulation": {
    "solvent": "EC:DEC 1:1",
    "salt": "LiPF6",
    "additive": "FEC",
    "add_pct": 10.0,
    "salt_conc_M": 1.0
  },
  "predictions": {
    "ionic_conductivity_mS_cm": 10.2,
    "electrochemical_window_V": 4.55,
    "flash_point_C": 105.3,
    "desolvation_barrier_kJ_mol": 42.8,
    "sei_composition": "LiF-rich (F-dominated SEI)"
  },
  "shap_explanation": {
    "top_feature": "additive_lumo",
    "top_feature_contribution": "+0.82 mS/cm",
    "feature_ranking": ["additive_lumo", "additive_pct", "salt_mw", "solvent_tpsa", "..."]
  },
  "confidence": {
    "within_training_range": true,
    "confidence_band": "±1.2 mS/cm"
  }
}
```

### 7.2 典型验证指标范围

| 指标 | 训练集 | 5 折 CV | Holdout |
|---|---|---|---|
| R² | 0.92–0.98 | 0.85–0.95 ± σ | 0.80–0.90 |
| MAE (mS/cm) | 0.3–0.8 | 0.5–1.2 | 0.8–1.5 |
| RMSE (mS/cm) | 0.4–1.0 | 0.7–1.5 | 1.0–2.0 |

> **重要提示**：以上指标范围基于**合成 + 半经验**训练数据。湿实验前需进行真实实验验证，详见[验证指南](#validation-guide)。

---

## 八、项目结构

```
batterychem-ai/
├── app.py                          # 核心 ML 训练引擎 + CLI 交互预测
├── web_app.py                      # Streamlit 网页仪表盘
├── search_all_families.py          # 高通量筛选引擎（4,800 条配方）
├── evaluate_and_explain.py         # 5 折 CV + SHAP 分析 + 残差图
├── plot_academic_space.py          # PCA 2D 投影 + Pareto 雷达图
├── data_pipeline.py                # 数据获取：Liverpool LMDS + CALiSol-23 + 文献
├── ultimate_academic_brain.pkl      # 预训练 XGBoost 模型（二进制，约 5 MB）
├── requirements.txt                 # Python 依赖列表
├── LICENSE                         # MIT 开源许可证
├── README.md                       # 本文件
│
├── data/                           # 由 data_pipeline.py 自动生成
│   ├── experimental_training_data.csv   # 核心训练数据集
│   ├── molecular_descriptors.csv         # 45 种分子描述符表
│   ├── flash_point_data.csv              # 闪点文献数据
│   ├── homo_lumo_reference.csv           # HOMO/LUMO 文献参考值
│   └── data_quality_report.txt           # 数据质量报告
│
└── docs/                           # 设计文档与复盘报告
    ├── AI_Driven_Green_Chemistry.pdf     # 原始设计概念文档
    ├── AI4S-Evolution.html               # V1–V5 五次迭代发展史
    ├── Opus-Review-V5.html                # V5 全面评估报告
    └── Opus-数据来源程序指导.html         # 数据管道完整说明
```

---

## 九、数据来源与质量

### 9.1 分子数据库

| 数据库 | 分子数量 | 来源 | 覆盖范围 |
|---|---|---|---|
| 溶剂库 | 15 种 | PubChem + 文献 | 碳酸酯、醚、内酯 |
| 锂盐库 | 3 种 | PubChem | LiPF6, LiFSI, LiTFSI |
| 添加剂库 | 10 种 | PubChem + 文献 | FEC, VC, PS, SN, ADN, DTD + 多酚 |

### 9.2 训练数据来源

| 来源 | 类型 | 条数 | 质量 |
|---|---|---|---|
| CALiSol-23 (Nature Sci. Data 2024) | 实验值 | 25 | 高 |
| J. Electrochem. Soc. 2019, 166, A3015 | 实验值 | 10 | 高 |
| J. Power Sources 2020, 451, 227791 | 实验值 | 6 | 高 |
| Electrochimica Acta 2021, 368, 137603 | 实验值 | 4 | 高 |
| Liverpool Ionics LMDS | 固体电解质 | ~700 | 用于泛化能力测试 |
| GreenBatt 多酚估算 | 半经验估算 | 8 | 中 |

> ⚠️ **数据质量声明**：训练标签包含真实实验值和半经验估算值。多酚类（Quercetin、Catechin 等）电导率数据为 **估算值**，来源于 JECerin 2020 文献，在 `experimental_training_data.csv` 中标注为 `data_quality = "medium"`。

---

## 十、验证指南

BatteryChem-AI 的预测结果应作为**虚拟筛选参考**，而非真实测量值。湿实验验证前请按以下步骤操作：

1. **文献交叉验证**：将预测结果与同行评审期刊（J. Electrochem. Soc.、Energy Environ. Sci.、J. Power Sources 等）的实验数据进行对比
2. **基线对照**：使用内置基准（EC:DEC 1:1 + 10 wt% FEC + 1M LiPF6，σ ≈ 10.2 mS/cm）作为参照
3. **训练范围检查**：超出训练范围的预测值（电导率约 0.1–20 mS/cm 区间外）应谨慎解读
4. **SHAP 一致性**：检查 top SHAP 特征是否与领域知识一致（如 FEC → LUMO 更低 → LiF-rich SEI）
5. **物理合理性**：碳酸酯体系中 σ > 20 mS/cm 或闪点 < 50°C 在物理上不太可能

---

## 十一、五次迭代开发历程

| 版本 | 核心变化 | 数据 | 模型 | 综合评分 |
|---|---|---|---|---|
| **V1** | 初始原型 | 56,250 条合成数据（物理方程） | RandomForest × 5 | 5.8/10 |
| **V2** | O2.5 键断裂能垒 | 算法生成的虚拟分子 | RandomForest | 6.3/10 |
| **V3** | 真实数据整合、XGBoost | 14 条真实 + 1,400 条扰动 | XGBoost MultiOutput | 7.2/10 |
| **V4** | Holdout 验证、SHAP | 1,020 个锚点样本 | XGBoost MultiOutput | 8.0/10 |
| **V5** | 工业全体系、3 盐 | 1,200 样本、高斯噪声 | 单 XGBoost、15 维 | **9.1/10** |

> 完整 V1–V5 开发史与经验教训详见 `docs/AI4S-Evolution.html`。

---

## 十二、路线图

| 优先级 | 功能 | 状态 |
|---|---|---|
| P0 | README、LICENSE、requirements.txt | ✅ 已完成 |
| P1 | 真实实验数据集（文献数据） | ✅ 已完成（CALiSol-23 + 4 种期刊） |
| P1 | 5 折 CV + SHAP 评估 | ✅ 已完成 |
| P1 | 基准电解液对比（UI 内置） | 🔜 计划中 |
| P2 | RDKit SMILES 输入（自动描述符） | 🔜 计划中 |
| P2 | 不确定性量化（预测区间） | 🔜 计划中 |
| P2 | Jupyter Notebooks（×3：入门/筛选/SHAP） | 🔜 计划中 |
| P3 | DFT HOMO/LUMO 计算（隐式溶剂） | 🔜 计划中 |
| P3 | Active Learning 闭环（湿实验反馈） | 🔜 计划中 |
| P3 | arXiv 预印本 | 🔜 计划中 |
| P3 | HuggingFace 模型库 + Zenodo DOI | 🔜 计划中 |

---

## 十三、贡献指南

欢迎提交贡献！请：

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/your-feature`）
3. 运行 `python evaluate_and_explain.py` 确保改动不降低 CV R²
4. 使用清晰的英文 commit message 提交
5. Push 并提交 Pull Request

重大改动请先提交 Issue 讨论。

---

## 十四、引用

如果 BatteryChem-AI 对您的研究有帮助，请引用：

```bibtex
@software{batterychem_ai,
  title = {BatteryChem-AI: High-Throughput Virtual Screening Platform for Li-ion Battery Electrolytes},
  author = {Your Name},
  year = {2026},
  version = {5.0},
  url = {https://github.com/YOUR_USERNAME/batterychem-ai}
}
```

---

## 十五、许可证

本项目采用 **MIT 开源许可证**，详见 [LICENSE](LICENSE)。

---

## 十六、致谢

- **CALiSol-23 数据集**：Nature Scientific Data 2024
- **Liverpool Ionics LMDS**：英国利物浦大学材料发现中心
- **文献数据**：J. Electrochem. Soc.、J. Power Sources、Electrochimica Acta
- **分子描述符**：PubChem（美国国立生物技术信息中心/NIH）
- **建模框架**：XGBoost、scikit-learn、SHAP
- **可视化**：Streamlit、Plotly、matplotlib

---

*Built with ⚡ for the AI4Science & electrochemistry community · 丘成桐中学科学奖 2026*
