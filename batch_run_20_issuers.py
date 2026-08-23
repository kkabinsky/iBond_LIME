# -*- coding: utf-8 -*-
"""
batch_run_20_issuers.py - Generate JPG and PNG explanation assets for 20 issuers.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, r"d:\tadgan_gaf\cmdf_credit_app\thaibma\dataset\datasets_bond")
from lime33feature import load_state, explain, render_figure, save_figure_jpg_png, report

def main():
    print("Loading Dataset 1 & 2 state...")
    S = load_state(choice=1, workload=0.05)
    tab = S["tab"]
    
    # Pick top 20 issuers by PD (high risk & prominent issuers)
    top_issuers = tab.head(20).issuer.tolist()
    print(f"Top 20 issuers to generate: {top_issuers}")
    
    generated = []
    for i, iss in enumerate(top_issuers, 1):
        print(f"\n[{i}/20] Processing issuer: {iss}...")
        res = explain(S, iss, samples=2500, repeats=6)
        if res is None:
            print(f"Skipping {iss} (no data)")
            continue
        fig = render_figure(res, top=10)
        jpg_p, png_p = save_figure_jpg_png(fig, iss)
        print(f"  Saved JPG: {jpg_p}")
        print(f"  Saved PNG: {png_p}")
        generated.append({
            "issuer": iss,
            "month": res["month"],
            "pd": res["pd_now"],
            "thr": res["thr"],
            "status": "HIGH RISK" if res["pd_now"] >= res["thr"] else "NORMAL",
            "jpg": jpg_p,
            "png": png_p
        })
        
    print(f"\nSuccessfully generated {len(generated)} issuer explanation figures!")
    df_gen = pd.DataFrame(generated)
    df_gen.to_csv(r"d:\tadgan_gaf\cmdf_credit_app\thaibma\dataset\datasets_bond\tex_out\generated_20_issuers.csv", index=False)
    print(df_gen)

if __name__ == "__main__":
    main()
