# Thai Capital Market & Corporate Credit XAI System (Project LIME)

> **Explainable AI (LIME & SHAP) Early Warning System for Corporate Bond Defaults & Debt Restructuring**  
> Comprehensive 30-Determinant Risk Explanation with Dual-Dataset Adapter, Compact SQLite Database, & Three-Layer Review Queue.

---

## 📋 Overview (ภาพรวมโครงการ)

This repository contains the complete, production-ready implementation of the **Explainable AI (XAI) Early Warning System (EWS)** for Thai corporate bonds and listed companies in the capital market.

### ✨ Key Features:
1. **Dual-Dataset DataAdapter**:
   - `Dataset 1 (Legacy)`: `ibond_33features_panel` (16,986 firm-months, 219–293 bond issuers)
   - `Dataset 2 (New Panel)`: `ibond_33features_panel_941firm` (187,007 firm-months, 941 firms across SET/mai)
2. **Compact & Dedicated Database (`lime_credit.db`)**:
   - Clean SQLite database containing **ONLY** the necessary tables and indexes.
   - Includes `lime_credit.db.zip` (23.8 MB) for instant GitHub cloning and automatic extraction.
3. **Explainable AI (Repeated LIME + Exact Tree SHAP)**:
   - **Repeated-Seed LIME**: 5,000 perturbation samples across 8 random seeds with confidence intervals.
   - **Exact Additive Tree SHAP**: True game-theoretic Shapley additive feature decomposition matching model PD.
   - **Distance from Median**: Visualizing each firm's standing against peer population in Standard Deviations (SD).
4. **Three-Layer Review Queue (Methods A, B, C)**:
   - **Method A (Level Rule)**: $\text{PD} \ge \text{Threshold}_{5\%}$
   - **Method B (Fragility Rule)**: Sensitivity to single shock $\le 1.0\,\text{SD}$
   - **Method C (Masked Distress Rule)**: Bottom 10% decile in Retained Earnings (`REtoTA`) & Cash Ratio (`CashRatio`)
   - **Detection Accuracy**: Catches **8 / 8 (100%)** defaults on Dataset 1, and **15 / 31 (48.4%)** distress cases on Dataset 2.
5. **Full Documentation & Manual (50 Pages)**: Full LaTeX manual in `lime33.tex` and compiled publication-grade PDF in `lime33.pdf`.

---

## 📂 Project Structure (โครงสร้างไฟล์)

```
lime/
├── lime33_adapter_panel.py    # Main program (Dual DataAdapter + Interactive Menu + CLI)
├── lime_feature33.py          # Standalone legacy program
├── data_adapter.py            # Unified DataAdapter with auto-unzip for lime_credit.db.zip
├── a_approach.py              # Three-layer Methods A, B, C evaluation engine
├── evaluate_methods_abc.py    # Detection accuracy benchmark script
├── batch_run_20_issuers.py    # Batch graphic generator for 20 sample issuers
├── firm_shock_panel.py        # Shock fragility engine
├── cmdf_tree_classify.py      # CatBoost/RandomForest classifier wrapper
├── lime_credit.db.zip         # Compact SQLite database (23.8 MB, auto-unpacked on first run)
├── lime_credit.db             # Unpacked SQLite database (contains 16k & 187k tables)
├── lime33.tex                 # Master Thai LaTeX documentation (50 pages)
├── lime33.pdf                 # Compiled publication-grade PDF manual
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Standard git ignore rules
├── README.md                  # Project overview and user guide
└── tex_out/
    ├── lime_jpg/              # High-resolution JPG explanation figures for 20+ firms
    ├── lime_figs/             # PNG explanation figures
    └── summary_941_firms.csv  # Complete directory & stats of 941 companies
```

---

## 🗄️ Database Schema (`lime_credit.db`)

The clean SQLite database contains exclusively the tables used by the XAI engine:

| Table Name | Rows | Columns | Description |
| :--- | :---: | :---: | :--- |
| **`ibond_33features_panel`** | 16,986 | 61 | Legacy panel covering 219–293 ThaiBMA bond issuers |
| **`ibond_33features_panel_941firm`** | 187,007 | 95 | Comprehensive panel covering 941 SET/mai firms (Jan 2007 – Aug 2026) |
| **`ibond_default_payment`** | 50 | 12 | Official ThaiBMA bond default register |
| **`firm_issuer_mapping`** | 985 | 6 | Securities ticker to ThaiBMA issuer code mapping |
| **`ibond_issuer`** | 678 | 8 | Issuer metadata and industry classifications |

---

## 🚀 Quick Start (การติดตั้งและใช้งาน)

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/<YOUR_USERNAME>/lime.git
cd lime
pip install -r requirements.txt
```

### 2. Interactive Menu Mode
Run the main program to choose between Dataset 1 or Dataset 2 interactively:
```bash
python lime33_adapter_panel.py
```

### 3. Direct Command-Line Interface (CLI)
* **Explain specific firm on Dataset 2 (941 firms):**
  ```bash
  python lime33_adapter_panel.py -d 2 --issuer PTT
  python lime33_adapter_panel.py -d 2 --issuer A
  ```

* **Explain all High-Risk firms above Review Threshold:**
  ```bash
  python lime33_adapter_panel.py -d 2 --all-high-risk
  ```

* **Explain specific firm on Legacy Dataset (Dataset 1):**
  ```bash
  python lime33_adapter_panel.py -d 1 --issuer A
  ```

* **Run Methods A, B, C Detection Accuracy Benchmark:**
  ```bash
  python evaluate_methods_abc.py
  ```

---

## 📊 30 Financial, Liquidity, Macro & ESG Determinants

| Category | Determinants / Column Names |
| :--- | :--- |
| **1. Liquidity & Trading** | `amihud_monthly`, `adj_illiq_kz`, `percent_zero_days`, `zero_days`, `n_days` |
| **2. Financial Ratios** | `ROA`, `ROE`, `DE`, `CurrentRatio`, `QuickRatio`, `CashRatio`, `EBITtoTA`, `REtoTA`, `WorkingCapitaltoTA`, `TDTA`, `LTDtoTA`, `STDtoTA`, `cf_Interestcoverageratio`, `acc_DebtServiceCoverageRatio` |
| **3. Scale & Age** | `lnTotalAssets`, `lnAge` |
| **4. Macroeconomic** | `Policyrate`, `GDPgrowth`, `UnemploymentratemodeledILOe` |
| **5. ESG & Governance** | `ESGScore`, `GovernancePillarScore`, `EnvironmentalPillarScore`, `SocialPillarScore`, `IndependentBoardMembers`, `AverageBoardTenure` |

---

## 🛡️ Three-Layer Detection Benchmark (Accuracy)

| Review Method | Flagged Firms | Caught Events | Recall (%) | Precision (%) | Market Capacity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **[Dataset 1: Legacy 293 Firms (8 Defaults)]** | | | | | |
| Method A (Level Rule: $\text{PD} \ge \text{Thr}$) | 15 | **4 / 8** | 50.0% | 26.7% | 5.1% |
| Method B (Fragility: Shock $\le 1.0\,\text{SD}$) | 8 | **2 / 8** | 25.0% | 25.0% | 2.7% |
| Method C (Masked Distress: Bottom 10%) | 12 | **4 / 8** | 50.0% | 33.3% | 4.1% |
| **Combined Methods A + B + C** | **28** | **8 / 8** | **100.0%** | **28.6%** | **9.6%** |
| | | | | | |
| **[Dataset 2: New 941 Firms (31 Distress/RS)]** | | | | | |
| Method A (Level Rule: $\text{PD} \ge \text{Thr}$) | 48 | **8 / 31** | 25.8% | 16.7% | 5.1% |
| Method B (Fragility: Shock $\le 1.0\,\text{SD}$) | 21 | **3 / 31** | 9.7% | 14.3% | 2.2% |
| Method C (Masked Distress: Bottom 10%) | 39 | **5 / 31** | 16.1% | 12.8% | 4.1% |
| **Combined Methods A + B + C** | **106** | **15 / 31** | **48.4%** | **14.2%** | **11.3%** |

---

## 📄 Full Manual & LaTeX Source

The complete 50-page manual is provided in:
- 📖 [`lime33.pdf`](lime33.pdf) (Compiled PDF Manual)
- 📝 [`lime33.tex`](lime33.tex) (XeLaTeX Source Document)
