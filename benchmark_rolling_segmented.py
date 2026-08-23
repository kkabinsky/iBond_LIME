# -*- coding: utf-8 -*-
"""
benchmark_rolling_segmented.py - Comprehensive benchmark with y_target attached.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

SUB_DIR = r"d:\tadgan_gaf\cmdf_credit_app\lime"
sys.path.insert(0, SUB_DIR)
from data_adapter import DataAdapter, BOND_33_FEATURES

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

SEED = 42

def main():
    print("================================================================================")
    print("RUNNING COMPREHENSIVE EXPERIMENT: ROLLING WINDOW vs SEGMENTED MODELS")
    print("================================================================================")
    
    panel, X, y, cols = DataAdapter.load(choice=2, verbose=False)
    panel = panel.reset_index(drop=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].astype(str).to_numpy()
    panel["y_target"] = yv
    
    # 1. Segment Definition (Bond Issuers vs Non-Bond / mai)
    if "ibond_matched" in panel.columns and panel["ibond_matched"].sum() > 50:
        is_bond = panel["ibond_matched"].fillna(0).astype(int) == 1
    elif "market" in panel.columns:
        is_bond = panel["market"] == "SET"
    else:
        is_bond = pd.Series(True, index=panel.index)
        
    panel["is_bond"] = is_bond
    
    ev_firms = set(panel.loc[yv == 1, "issuer_code"].unique())
    bond_ev_firms = set(panel.loc[(yv == 1) & is_bond, "issuer_code"].unique())
    mai_ev_firms = ev_firms - bond_ev_firms
    
    print(f"Total Universe: {panel['issuer_code'].nunique():,} firms | Total Distress Firms: {len(ev_firms):,}")
    print(f"  - Group 1: Bond Issuers (SET/iBond): {panel.loc[is_bond, 'issuer_code'].nunique():,} firms ({len(bond_ev_firms)} distress firms)")
    print(f"  - Group 2: Non-Bond / Small-Cap mai: {panel.loc[~is_bond, 'issuer_code'].nunique():,} firms ({len(mai_ev_firms)} distress firms)")
    
    # 2. Train Out-of-fold Models
    print("\n[Step 1] Fitting Out-of-fold ML Engine across 187,007 firm-months...")
    fold_of = np.full(len(A), -1, int)
    fold_models = {}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for k, (tr, te) in enumerate(cv.split(A, yv, groups)):
        if yv[tr].sum() < 1:
            continue
        sck = StandardScaler().fit(A[tr])
        mk = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                                l2_leaf_reg=3.0, auto_class_weights="Balanced",
                                random_seed=SEED, verbose=0,
                                allow_writing_files=False).fit(sck.transform(A[tr]), yv[tr])
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
    
    # 3. Method C: Masked Distress (Bottom Decile REtoTA & CashRatio)
    ix_c = {c: i for i, c in enumerate(cols)}
    q_re = np.nanquantile(A[:, ix_c["REtoTA"]], 0.10) if "REtoTA" in ix_c else -999
    q_cash = np.nanquantile(A[:, ix_c["CashRatio"]], 0.10) if "CashRatio" in ix_c else -999
    panel["flag_C"] = (A[:, ix_c["REtoTA"]] <= q_re) & (A[:, ix_c["CashRatio"]] <= q_cash)
    
    # Latest observation per firm
    panel["_row"] = np.arange(len(panel))
    latest_rows = panel.sort_values("month_dt").groupby("issuer_code").tail(1)["_row"].to_numpy()
    
    # Thresholds at 5% capacity
    thr_pooled = float(np.quantile(panel.loc[latest_rows, "PD_OOF"], 0.95))
    thr_bond = float(np.quantile(panel.loc[latest_rows[panel.loc[latest_rows, "is_bond"]], "PD_OOF"], 0.95))
    thr_mai = float(np.quantile(panel.loc[latest_rows[~panel.loc[latest_rows, "is_bond"]], "PD_OOF"], 0.95))
    
    print(f"\n[Step 2] Calibrating 5% Review Thresholds:")
    print(f"  - Pooled Threshold (Top 5%): {thr_pooled:.6f}")
    print(f"  - Segment 1 (Bond Issuers) Threshold: {thr_bond:.6f}")
    print(f"  - Segment 2 (Non-Bond / mai) Threshold: {thr_mai:.6f}")
    
    # Flags for all rows in panel
    panel["flag_A_pooled"] = panel["PD_OOF"] >= thr_pooled
    panel["flag_A_seg"] = np.where(panel["is_bond"], panel["PD_OOF"] >= thr_bond, panel["PD_OOF"] >= thr_mai)
    
    panel["flag_ABC_pooled"] = panel["flag_A_pooled"] | panel["flag_C"]
    panel["flag_ABC_seg"] = panel["flag_A_seg"] | panel["flag_C"]
    
    # Slicing latest_df AFTER adding columns
    latest_df = panel.loc[latest_rows].copy()
    
    # Sort panel
    panel = panel.sort_values(["issuer_code", "month_dt"]).reset_index(drop=True)
    
    # 5. Evaluate the Strategies
    all_firms = sorted(panel["issuer_code"].unique())
    is_ev_firm = {f: (f in ev_firms) for f in all_firms}
    is_bond_firm = {f: (f in bond_ev_firms or panel[panel["issuer_code"]==f]["is_bond"].iloc[-1]) for f in all_firms}
    
    # Strategy 1: Baseline Single Month Snapshot (Pooled)
    s1_flags = {r["issuer_code"]: bool(r["flag_ABC_pooled"]) for _, r in latest_df.iterrows()}
    
    # Strategy 2: Segmented Single Month Snapshot
    s2_flags = {r["issuer_code"]: bool(r["flag_ABC_seg"]) for _, r in latest_df.iterrows()}
    
    # Strategy 3: Rolling 12-Month Window (Pooled)
    s3_flags = {}
    s4_flags = {}
    s5_flags = {}
    s6_flags = {}
    
    for f, g in panel.groupby("issuer_code"):
        # Rolling 12M pooled
        s3_flags[f] = bool(g.tail(12)["flag_ABC_pooled"].any())
        # Rolling 12M segmented
        s4_flags[f] = bool(g.tail(12)["flag_ABC_seg"].any())
        # Rolling 24M segmented
        s5_flags[f] = bool(g.tail(24)["flag_ABC_seg"].any())
        
        # Lead-Time Horizon (12 Months Leading to Event)
        if f not in ev_firms:
            s6_flags[f] = bool(g.tail(12)["flag_ABC_seg"].any())
        else:
            ev_matches = g[g["y_target"] == 1]
            if len(ev_matches) > 0:
                first_idx = ev_matches.index[0]
                first_loc = g.index.get_loc(first_idx)
                start_loc = max(0, first_loc - 12)
                end_loc = min(len(g), first_loc + 3)
                s6_flags[f] = bool(g.iloc[start_loc:end_loc]["flag_ABC_seg"].any())
            else:
                s6_flags[f] = bool(g["flag_ABC_seg"].any())

    strategies = [
        ("1. Baseline Snapshot (Single Month, Pooled 941)", s1_flags),
        ("2. Segmented Snapshot (Separate Bond vs mai Thresholds)", s2_flags),
        ("3. Rolling 12-Month Window (Pooled 941)", s3_flags),
        ("4. Segmented + Rolling 12-Month Window (Recommended)", s4_flags),
        ("5. Segmented + Rolling 24-Month Window (Full Debt Cycle)", s5_flags),
        ("6. Dynamic Lead-Time Horizon (12 Months Before Event)", s6_flags),
    ]
    
    results = []
    for name, f_dict in strategies:
        flagged_firms = [f for f in all_firms if f_dict.get(f, False)]
        caught_all = [f for f in flagged_firms if is_ev_firm[f]]
        missed_all = [f for f in ev_firms if not f_dict.get(f, False)]
        
        caught_bond = [f for f in caught_all if is_bond_firm[f]]
        missed_bond = [f for f in bond_ev_firms if not f_dict.get(f, False)]
        
        recall_all = len(caught_all) / len(ev_firms) * 100.0
        prec_all = len(caught_all) / len(flagged_firms) * 100.0 if len(flagged_firms) > 0 else 0.0
        recall_bond = len(caught_bond) / len(bond_ev_firms) * 100.0 if len(bond_ev_firms) > 0 else 0.0
        
        results.append({
            "Strategy": name,
            "Total_Flagged": len(flagged_firms),
            "Overall_Caught": f"{len(caught_all)}/{len(ev_firms)}",
            "Recall_Overall": f"{recall_all:.1f}%",
            "Precision": f"{prec_all:.1f}%",
            "Bond_Caught": f"{len(caught_bond)}/{len(bond_ev_firms)}",
            "Bond_Recall": f"{recall_bond:.1f}%",
            "Missed_Count": len(missed_all),
            "Missed_Firms": missed_all,
            "Caught_Firms": caught_all
        })
        
    df_res = pd.DataFrame(results)
    
    print("\n" + "=" * 115)
    print("FINAL BENCHMARK RESULTS: ENHANCED DETECTION FRAMEWORKS ON DATASET 2 (941 FIRMS)")
    print("=" * 115)
    cols_show = ["Strategy", "Total_Flagged", "Overall_Caught", "Recall_Overall", "Precision", "Bond_Caught", "Bond_Recall"]
    print(df_res[cols_show].to_string(index=False))
    
    print("\n" + "=" * 115)
    print("DETAILED DETECTED ISSUERS BREAKDOWN:")
    print("=" * 115)
    for idx, r in df_res.iterrows():
        print(f"\n[{r['Strategy']}]")
        print(f"  -> Total Caught ({r['Overall_Caught']}, {r['Recall_Overall']}): {sorted(r['Caught_Firms'])}")
        if r["Missed_Count"] > 0:
            print(f"  -> Missed ({r['Missed_Count']}): {sorted(r['Missed_Firms'])}")
            
    df_res.to_csv(os.path.join(SUB_DIR, "tex_out", "comparison_rolling_segmented.csv"), index=False)
    print("\nSaved benchmark results to tex_out/comparison_rolling_segmented.csv!")

if __name__ == "__main__":
    main()
