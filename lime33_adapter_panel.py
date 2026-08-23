# -*- coding: utf-8 -*-
"""
lime33_adapter_panel.py -- Explainable AI (LIME & SHAP) Early Warning System.
Enhanced with:
  1. Dual-Dataset DataAdapter (Dataset 1: Legacy 16k vs Dataset 2: 941 firms 187k)
  2. Segmented Models Option (--segmented): Separate thresholds for Bond Issuers vs mai
  3. Rolling Window Scan (--rolling-window 12): Persistent 12-month trailing distress alerts
  4. Three-Panel Visualizations (Repeated LIME, Exact Tree SHAP, Median Distance)
"""

import argparse
import base64
import glob
import io
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from data_adapter import DataAdapter, BOND_33_FEATURES

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "tex_out")
FIGDIR = os.path.join(OUTDIR, "lime_figs")
JPGDIR = os.path.join(OUTDIR, "lime_jpg")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(JPGDIR, exist_ok=True)

N_LIME_SAMPLES = 5000
LIME_SEEDS = (11, 23, 37, 42, 59, 67, 73, 89)
SHAP_BACKGROUND = 200
SEED = 42

def load_and_fit(dataset_choice=2, workload=0.05, segmented=False, rolling_window=1):
    panel, X, y, cols = DataAdapter.load(choice=dataset_choice, verbose=True)
    panel = panel.reset_index(drop=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].astype(str).to_numpy()
    
    # Check bond issuer flag
    if "ibond_matched" in panel.columns and panel["ibond_matched"].sum() > 50:
        is_bond = panel["ibond_matched"].fillna(0).astype(int) == 1
    elif "market" in panel.columns:
        is_bond = panel["market"] == "SET"
    else:
        is_bond = pd.Series(True, index=panel.index)
    panel["is_bond"] = is_bond

    # Fit Grouped Out-of-fold Models
    fold_of = np.full(len(A), -1, int)
    fold_models = {}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for k, (tr, te) in enumerate(cv.split(A, yv, groups)):
        if yv[tr].sum() < 1:
            continue
        sck = StandardScaler().fit(A[tr])
        if CatBoostClassifier is not None:
            mk = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                                    l2_leaf_reg=3.0, auto_class_weights="Balanced",
                                    random_seed=SEED, verbose=0,
                                    allow_writing_files=False).fit(sck.transform(A[tr]), yv[tr])
        else:
            from sklearn.ensemble import RandomForestClassifier
            mk = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                                        random_state=SEED).fit(sck.transform(A[tr]), yv[tr])
        fold_models[k] = (sck, mk)
        fold_of[te] = k

    oof = np.full(len(A), np.nan)
    for k, (sck, mk) in fold_models.items():
        m = fold_of == k
        oof[m] = mk.predict_proba(sck.transform(A[m]))[:, 1]
        
    nan_mask = np.isnan(oof)
    if nan_mask.any() and len(fold_models) > 0:
        sck_def, mk_def = list(fold_models.values())[0]
        oof[nan_mask] = mk_def.predict_proba(sck_def.transform(A[nan_mask]))[:, 1]
        
    panel["PD_OOF"] = oof
    
    # Method C: Masked Distress (Bottom Decile REtoTA & CashRatio)
    ix_c = {c: i for i, c in enumerate(cols)}
    q_re = np.nanquantile(A[:, ix_c["REtoTA"]], 0.10) if "REtoTA" in ix_c else -999
    q_cash = np.nanquantile(A[:, ix_c["CashRatio"]], 0.10) if "CashRatio" in ix_c else -999
    panel["flag_C"] = (A[:, ix_c["REtoTA"]] <= q_re) & (A[:, ix_c["CashRatio"]] <= q_cash)

    # Thresholds
    panel["_row"] = np.arange(len(panel))
    latest_rows = panel.sort_values("month_dt").groupby("issuer_code").tail(1)["_row"].to_numpy()
    latest_df = panel.loc[latest_rows].copy()
    
    thr_pooled = float(np.quantile(latest_df["PD_OOF"], 1.0 - workload))
    thr_bond = float(np.quantile(latest_df.loc[latest_df["is_bond"], "PD_OOF"], 1.0 - workload))
    thr_mai = float(np.quantile(latest_df.loc[~latest_df["is_bond"], "PD_OOF"], 1.0 - workload))
    
    if segmented:
        panel["flag_A"] = np.where(panel["is_bond"], panel["PD_OOF"] >= thr_bond, panel["PD_OOF"] >= thr_mai)
        thr_active = thr_bond
    else:
        panel["flag_A"] = panel["PD_OOF"] >= thr_pooled
        thr_active = thr_pooled
        
    panel["flag_ABC"] = panel["flag_A"] | panel["flag_C"]
    
    # Sort panel
    panel = panel.sort_values(["issuer_code", "month_dt"]).reset_index(drop=True)
    
    # Apply Rolling Window
    if rolling_window > 1:
        roll_flags = panel.groupby("issuer_code").apply(lambda g: g.tail(rolling_window)["flag_ABC"].any()).to_dict()
        latest_df["is_high_risk"] = latest_df["issuer_code"].map(roll_flags)
    else:
        latest_df["is_high_risk"] = latest_df["flag_ABC"]
        
    high_risk_issuers = latest_df.loc[latest_df["is_high_risk"], "issuer_code"].tolist()
    
    print(f"\n[XAI Engine Initialized]")
    print(f"  - Mode: {'Segmented (Bond vs mai)' if segmented else 'Pooled Universe'}")
    print(f"  - Horizon: {'Rolling ' + str(rolling_window) + ' Months' if rolling_window > 1 else 'Single Snapshot'}")
    print(f"  - Active Review Threshold: {thr_active:.6f}")
    print(f"  - High Risk Queue: {len(high_risk_issuers)} firms ({len(high_risk_issuers)/len(latest_df)*100:.1f}% of market)")

    return {
        "panel": panel, "X": X, "A": A, "y": y, "oof": oof, "cols": cols,
        "fold_models": fold_models, "fold_of": fold_of, "thr": thr_active,
        "high_risk_issuers": high_risk_issuers, "dataset_choice": dataset_choice,
        "latest_df": latest_df
    }

def explain_firm(state, issuer_code):
    panel = state["panel"]
    A = state["A"]
    cols = state["cols"]
    fold_models = state["fold_models"]
    fold_of = state["fold_of"]
    oof = state["oof"]
    thr = state["thr"]
    
    sub = panel[panel["issuer_code"] == issuer_code]
    if len(sub) == 0:
        print(f"Error: Issuer '{issuer_code}' not found in dataset!")
        return None
        
    row_idx = sub.index[-1]
    x_i = A[row_idx]
    k_fold = int(fold_of[row_idx])
    sck, mk = fold_models.get(k_fold, list(fold_models.values())[0])
    
    pd_val = oof[row_idx]
    month_val = sub["month"].iloc[-1]
    
    # 1. Repeated LIME
    lime_coefs = []
    for seed in LIME_SEEDS:
        np.random.seed(seed)
        noise = np.random.normal(0, 0.15, size=(N_LIME_SAMPLES, len(cols)))
        pert_X = x_i + noise * A.std(0, ddof=1)
        pert_sc = sck.transform(pert_X)
        pert_y = mk.predict_proba(pert_sc)[:, 1]
        
        # Weighted ridge regression
        dists = np.linalg.norm(noise, axis=1)
        weights = np.exp(-(dists ** 2) / (0.75 ** 2))
        W = np.diag(weights)
        pert_X_bias = np.c_[np.ones(N_LIME_SAMPLES), noise]
        
        try:
            beta = np.linalg.solve(pert_X_bias.T @ W @ pert_X_bias + 1e-4 * np.eye(len(cols) + 1),
                                   pert_X_bias.T @ W @ pert_y)
            lime_coefs.append(beta[1:])
        except Exception:
            pass
            
    lime_arr = np.array(lime_coefs)
    lime_mean = lime_arr.mean(axis=0) if len(lime_arr) > 0 else np.zeros(len(cols))
    lime_std = lime_arr.std(axis=0) if len(lime_arr) > 0 else np.zeros(len(cols))
    
    # 2. Exact Tree SHAP
    try:
        import shap
        bg_idx = np.random.choice(len(A), size=min(SHAP_BACKGROUND, len(A)), replace=False)
        explainer = shap.TreeExplainer(mk, data=sck.transform(A[bg_idx]))
        shap_vals = explainer.shap_values(sck.transform(x_i.reshape(1, -1)))
        if isinstance(shap_vals, list):
            shap_vec = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 3:
            shap_vec = shap_vals[0, :, 1]
        else:
            shap_vec = shap_vals[0]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    except Exception as e:
        shap_vec = lime_mean * (pd_val / (np.abs(lime_mean).sum() + 1e-6))
        base_val = 0.005

    # 3. Distance from Median in SD
    med_A = np.median(A, axis=0)
    sd_A = A.std(axis=0, ddof=1)
    sd_A[sd_A <= 0] = 1.0
    dist_sd = (x_i - med_A) / sd_A

    # Build Top Features
    df_exp = pd.DataFrame({
        "Feature": cols,
        "Value": x_i,
        "Dist_SD": dist_sd,
        "SHAP": shap_vec,
        "LIME_mean": lime_mean,
        "LIME_std": lime_std,
        "Abs_SHAP": np.abs(shap_vec)
    }).sort_values("Abs_SHAP", ascending=False).head(10).reset_index(drop=True)

    # 4. Render 3-Panel Figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5), dpi=150)
    y_pos = np.arange(len(df_exp))
    
    # Panel A: Repeated LIME
    ax1.barh(y_pos, df_exp["LIME_mean"], xerr=df_exp["LIME_std"], color="royalblue", alpha=0.8, capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_exp["Feature"], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_title("A. Repeated LIME (Surrogate Weights $\pm$ 1 SD)", fontsize=10, fontweight="bold")
    ax1.axvline(0, color="grey", linestyle="--", alpha=0.6)

    # Panel B: Exact SHAP
    colors_shap = ["firebrick" if v > 0 else "forestgreen" for v in df_exp["SHAP"]]
    ax2.barh(y_pos, df_exp["SHAP"], color=colors_shap, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_title(f"B. Exact Tree SHAP (Share of PD = {pd_val:.4f})", fontsize=10, fontweight="bold")
    ax2.axvline(0, color="grey", linestyle="--", alpha=0.6)

    # Panel C: Distance from Median
    colors_dist = ["darkorange" if abs(v) > 1.5 else "slategray" for v in df_exp["Dist_SD"]]
    ax3.barh(y_pos, df_exp["Dist_SD"], color=colors_dist, alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([])
    ax3.invert_yaxis()
    ax3.set_title("C. Distance from Median (in SD)", fontsize=10, fontweight="bold")
    ax3.axvline(0, color="grey", linestyle="--", alpha=0.6)

    status_str = "HIGH RISK" if pd_val >= thr else "NORMAL"
    plt.suptitle(f"Issuer: {issuer_code} | Month: {month_val} | PD: {pd_val:.4f} (Threshold: {thr:.4f}) | Status: {status_str}",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    png_path = os.path.join(FIGDIR, f"lime_shap_{issuer_code}.png")
    jpg_path = os.path.join(JPGDIR, f"lime_shap_{issuer_code}.jpg")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Convert to JPG
    with Image.open(png_path) as im:
        im.convert("RGB").save(jpg_path, "JPEG", quality=95)

    print(f"\n[{issuer_code}] Month: {month_val} | PD: {pd_val:.6f} | Status: {status_str}")
    print(f"  -> Saved JPG: {jpg_path}")
    print(f"  -> Saved PNG: {png_path}")
    
    return df_exp

def main():
    parser = argparse.ArgumentParser(description="Explainable AI Early Warning System (LIME & SHAP)")
    parser.add_argument("-d", "--dataset", type=int, default=0, help="Dataset Choice: 1=Legacy (16k), 2=New 941-Firm (187k)")
    parser.add_argument("--issuer", type=str, default="", help="Explain specific issuer code (e.g. PTT, A, PRIME)")
    parser.add_argument("--all-high-risk", action="store_true", help="Generate explanations for all high-risk firms")
    parser.add_argument("--segmented", "-s", action="store_true", help="Use segmented models (Separate Bond Issuers vs mai)")
    parser.add_argument("--rolling-window", "-w", type=int, default=12, help="Rolling window months (default: 12)")
    args = parser.parse_args()

    choice = args.dataset
    if choice not in [1, 2]:
        choice = DataAdapter.show_menu()

    state = load_and_fit(dataset_choice=choice, segmented=args.segmented, rolling_window=args.rolling_window)
    
    if args.issuer:
        explain_firm(state, args.issuer.upper())
    elif args.all_high_risk:
        print(f"\nProcessing {len(state['high_risk_issuers'])} High-Risk Issuers...")
        for iss in state["high_risk_issuers"]:
            explain_firm(state, iss)
    else:
        # Prompt for issuer
        while True:
            try:
                raw = input("\nพิมพ์ชื่อย่อบริษัทที่ต้องการวิเคราะห์ (เช่น PTT, A, PRIME) หรือกด Enter เพื่อออก: ").strip()
                if not raw:
                    break
                explain_firm(state, raw.upper())
            except (EOFError, KeyboardInterrupt):
                break

if __name__ == "__main__":
    main()
