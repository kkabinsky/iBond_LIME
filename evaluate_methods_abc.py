# -*- coding: utf-8 -*-
"""
evaluate_methods_abc.py - Evaluates Methods A, B, C on Dataset 1 vs Dataset 2.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

SUB_DIR = r"d:\tadgan_gaf\cmdf_credit_app\thaibma\dataset\datasets_bond"
sys.path.insert(0, SUB_DIR)
from data_adapter import DataAdapter, BOND_33_FEATURES

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

PAIR = ("REtoTA", "CashRatio")
FRAG_FEATS = ("ROA", "ROE", "TDTA", "REtoTA", "CashRatio", "amihud_monthly",
              "WorkingCapitaltoTA", "DE")
GRID = np.arange(-3.0, 3.01, 0.25)
SEED = 42

def evaluate_dataset(choice=1, workload=0.05, cut_pct=10.0, frag_sd_limit=1.0):
    print(f"\n================================================================================")
    print(f"EVALUATING DATASET {choice}: {DataAdapter.DATASET_CHOICES[choice]['name']}")
    print(f"================================================================================")
    
    panel, X, y, cols = DataAdapter.load(choice=choice, verbose=False)
    panel = panel.reset_index(drop=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].astype(str).to_numpy()
    
    # Identify defaulted/distressed firms
    ev_firms = set(panel.loc[yv == 1, "issuer_code"].unique())
    print(f"Total Rows: {len(panel):,} | Total Firms: {panel['issuer_code'].nunique():,}")
    print(f"Total Positive Months: {int(yv.sum()):,} | Total Distress/Default Firms: {len(ev_firms):,}")
    
    # Fit Grouped Out-of-fold Models
    fold_of = np.full(len(A), -1, int)
    fold_models = {}
    n_splits = min(5, max(2, int(yv.sum()))) if int(yv.sum()) >= 2 else 2
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
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

    # Latest cross-section per issuer
    last_rows = (panel.assign(_i=np.arange(len(panel)))
                 .sort_values("month_dt").groupby("issuer_code").tail(1)["_i"]
                 .to_numpy())
    
    cross_oof = oof[last_rows]
    valid_cross = cross_oof[np.isfinite(cross_oof)]
    thr = float(np.quantile(valid_cross, 1.0 - workload)) if len(valid_cross) > 0 else 0.05
    
    # -------------------------------------------------------------
    # METHOD A: Level Rule (PD >= Threshold)
    # -------------------------------------------------------------
    flag_A = cross_oof >= thr
    
    # -------------------------------------------------------------
    # METHOD C: Masked Distress (Bottom Decile in REtoTA and CashRatio)
    # -------------------------------------------------------------
    ix_c = {c: i for i, c in enumerate(cols)}
    q_re = np.nanquantile(A[:, ix_c["REtoTA"]], cut_pct / 100.0) if "REtoTA" in ix_c else -999
    q_cash = np.nanquantile(A[:, ix_c["CashRatio"]], cut_pct / 100.0) if "CashRatio" in ix_c else -999
    
    flag_C = (A[last_rows, ix_c["REtoTA"]] <= q_re) & (A[last_rows, ix_c["CashRatio"]] <= q_cash)
    
    # -------------------------------------------------------------
    # METHOD B: Fragility / Shock Sensitivity (<= 1.0 SD to cross threshold)
    # -------------------------------------------------------------
    sd_arr = A.std(0, ddof=1)
    sd_arr[sd_arr <= 0] = 1.0
    
    frag_list = []
    sc_full = StandardScaler().fit(A)
    sck_def, mk_def = list(fold_models.values())[0] if len(fold_models) > 0 else (sc_full, None)
    
    scan_features = [f for f in FRAG_FEATS if f in ix_c]
    for idx_row in last_rows:
        x_orig = A[idx_row].copy()
        k_fold = int(fold_of[idx_row])
        sck_i, mk_i = fold_models.get(k_fold, (sck_def, mk_def))
        
        min_frag = 999.0
        # If already above threshold, fragility is 0
        if oof[idx_row] >= thr:
            frag_list.append(0.0)
            continue
            
        # Scan shocks on key features
        for f in scan_features:
            j = ix_c[f]
            sd_j = sd_arr[j]
            x_test = np.tile(x_orig, (len(GRID), 1))
            x_test[:, j] = x_orig[j] + GRID * sd_j
            probs = mk_i.predict_proba(sck_i.transform(x_test))[:, 1]
            cross_mask = probs >= thr
            if cross_mask.any():
                step_min = np.abs(GRID[cross_mask]).min()
                if step_min < min_frag:
                    min_frag = step_min
        frag_list.append(min_frag)
        
    frag_arr = np.array(frag_list)
    flag_B = (frag_arr <= frag_sd_limit) & (~flag_A) # Fragile but not already in A
    
    # Combined Flags
    flag_AB = flag_A | flag_B
    flag_AC = flag_A | flag_C
    flag_ABC = flag_A | flag_B | flag_C
    
    issuers = panel.loc[last_rows, "issuer_code"].to_numpy()
    is_event = np.array([iss in ev_firms for iss in issuers])
    
    total_ev = int(is_event.sum())
    
    def get_stats(flag):
        n_flag = int(flag.sum())
        n_caught = int((flag & is_event).sum())
        recall = (n_caught / total_ev) * 100.0 if total_ev > 0 else 0.0
        prec = (n_caught / n_flag) * 100.0 if n_flag > 0 else 0.0
        pct_mkt = (n_flag / len(last_rows)) * 100.0
        return n_flag, n_caught, total_ev, recall, prec, pct_mkt
    
    stats_A = get_stats(flag_A)
    stats_B = get_stats(flag_B)
    stats_C = get_stats(flag_C)
    stats_AC = get_stats(flag_AC)
    stats_ABC = get_stats(flag_ABC)
    
    df_res = pd.DataFrame([
        {"Method": "วิธี A (Level Rule: PD >= Threshold)", "Flagged_Firms": stats_A[0], "Caught_Events": f"{stats_A[1]}/{stats_A[2]}", "Recall_Pct": f"{stats_A[3]:.1f}%", "Precision_Pct": f"{stats_A[4]:.1f}%", "Market_Share": f"{stats_A[5]:.1f}%"},
        {"Method": "วิธี B (Fragility: Shock <= 1.0 SD)", "Flagged_Firms": stats_B[0], "Caught_Events": f"{stats_B[1]}/{stats_B[2]}", "Recall_Pct": f"{stats_B[3]:.1f}%", "Precision_Pct": f"{stats_B[4]:.1f}%", "Market_Share": f"{stats_B[5]:.1f}%"},
        {"Method": "วิธี C (Masked Distress: Bottom Decile)", "Flagged_Firms": stats_C[0], "Caught_Events": f"{stats_C[1]}/{stats_C[2]}", "Recall_Pct": f"{stats_C[3]:.1f}%", "Precision_Pct": f"{stats_C[4]:.1f}%", "Market_Share": f"{stats_C[5]:.1f}%"},
        {"Method": "รวมวิธี A + C (Level + Masked)", "Flagged_Firms": stats_AC[0], "Caught_Events": f"{stats_AC[1]}/{stats_AC[2]}", "Recall_Pct": f"{stats_AC[3]:.1f}%", "Precision_Pct": f"{stats_AC[4]:.1f}%", "Market_Share": f"{stats_AC[5]:.1f}%"},
        {"Method": "รวมทุกวิธี A + B + C (Three-Layer Combined)", "Flagged_Firms": stats_ABC[0], "Caught_Events": f"{stats_ABC[1]}/{stats_ABC[2]}", "Recall_Pct": f"{stats_ABC[3]:.1f}%", "Precision_Pct": f"{stats_ABC[4]:.1f}%", "Market_Share": f"{stats_ABC[5]:.1f}%"},
    ])
    
    print(df_res.to_string(index=False))
    
    # Detail of caught issuers
    caught_firms_ABC = issuers[flag_ABC & is_event]
    missed_firms_ABC = issuers[(~flag_ABC) & is_event]
    print(f"\nCaught Distress/Default Issuers ({len(caught_firms_ABC)}/{total_ev}): {list(caught_firms_ABC)}")
    if len(missed_firms_ABC) > 0:
        print(f"Missed Distress/Default Issuers ({len(missed_firms_ABC)}/{total_ev}): {list(missed_firms_ABC)}")
        
    return df_res, issuers, flag_ABC, is_event

def main():
    df1, _, _, _ = evaluate_dataset(choice=1, workload=0.05)
    df2, _, _, _ = evaluate_dataset(choice=2, workload=0.05)

if __name__ == "__main__":
    main()
