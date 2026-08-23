# -*- coding: utf-8 -*-
"""
lime33feature.py -- Per-issuer explanation of default probability with LIME and SHAP
side by side, supporting both Legacy and New 941-Firm SQLite tables with JPG export.

Supports:
  - Menu selection: [1] Legacy (ibond_33features_panel) or [2] New (ibond_33features_panel_941firm)
  - CLI switch: --dataset-choice 1 / --dataset-choice 2
  - Output figures as JPG directly into `tex_out/lime_jpg/` (and PNG in `tex_out/lime_figs/`)
  - Full LIME stability bars & Exact SHAP decomposition & Distance from median.

USAGE:
    python lime33feature.py                             # Interactive dataset menu + Top HIGH RISK issuer
    python lime33feature.py --dataset-choice 1 --issuer A
    python lime33feature.py --dataset-choice 2 --issuer PTT
    python lime33feature.py --dataset-choice 2 --all-high-risk
    python lime33feature.py --dataset-choice 2 --issuer MUD --repeats 10 --samples 5000
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

import shap

from data_adapter import DataAdapter, BOND_33_FEATURES

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR_FIGS = os.path.join(HERE, "tex_out", "lime_figs")
OUTDIR_JPG = os.path.join(HERE, "tex_out", "lime_jpg")
os.makedirs(OUTDIR_FIGS, exist_ok=True)
os.makedirs(OUTDIR_JPG, exist_ok=True)

SEED = 42
N_SAMPLES = 5000
N_REPEATS = 8
TOP = 10
DEFAULT_WORKLOAD = 0.05

_STATE_CACHE = {}

def load_state(choice: int = 1, workload: float = DEFAULT_WORKLOAD, force: bool = False):
    cache_key = (choice, workload)
    if not force and cache_key in _STATE_CACHE:
        return _STATE_CACHE[cache_key]

    panel, X, y, cols = DataAdapter.load(choice=choice, verbose=True)
    panel = panel.reset_index(drop=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].astype(str).to_numpy()

    # StratifiedGroupKFold out-of-fold models
    fold_of = np.full(len(A), -1, int)
    fold_models = {}
    
    n_pos = int(yv.sum())
    n_splits = min(5, max(2, n_pos)) if n_pos >= 2 else 2
    
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

    last_rows = (panel.assign(_i=np.arange(len(panel)))
                 .sort_values("month_dt").groupby("issuer_code").tail(1)["_i"]
                 .to_numpy())
    cross = oof[last_rows]
    cross = cross[np.isfinite(cross)]
    thr = float(np.quantile(cross, 1.0 - workload)) if len(cross) > 0 else 0.05

    tab = pd.DataFrame({
        "issuer": panel.loc[last_rows, "issuer_code"].to_numpy(),
        "month": panel.loc[last_rows, "month"].to_numpy(),
        "pd": oof[last_rows],
        "status": np.where(oof[last_rows] >= thr, "HIGH RISK", "NORMAL")
    }).sort_values("pd", ascending=False).reset_index(drop=True)

    sc_full = StandardScaler().fit(A)
    if CatBoostClassifier is not None:
        cb_full = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                                     l2_leaf_reg=3.0, auto_class_weights="Balanced",
                                     random_seed=SEED, verbose=0,
                                     allow_writing_files=False).fit(sc_full.transform(A), yv)
    else:
        from sklearn.ensemble import RandomForestClassifier
        cb_full = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                                         random_state=SEED).fit(sc_full.transform(A), yv)

    state = {
        "panel": panel, "X": X, "y": y, "cols": cols, "A": A, "yv": yv,
        "fold_of": fold_of, "fold_models": fold_models, "oof": oof,
        "thr": thr, "tab": tab, "sc": sc_full, "model": cb_full,
        "dataset_choice": choice
    }
    _STATE_CACHE[cache_key] = state
    return state

def issuer_row(S, issuer):
    p = S["panel"]
    rows = p.index[p["issuer_code"] == issuer].to_numpy()
    if len(rows) == 0:
        return None, None, None
    last = rows[np.argmax(p.loc[rows, "month_dt"].to_numpy())]
    k = int(S["fold_of"][last])
    sck, mk = S["fold_models"].get(k, (S["sc"], S["model"]))
    return last, sck, mk

def predict_fn(sck, mk):
    def f(B):
        B = np.asarray(B, float)
        if B.ndim == 1:
            B = B[None, :]
        return mk.predict_proba(sck.transform(B))
    return f

def run_lime(S, last, sck, mk, samples=N_SAMPLES, repeats=N_REPEATS, seed=SEED):
    A, cols = S["A"], S["cols"]
    x0 = A[last]
    f = predict_fn(sck, mk)

    runs = []
    for r in range(repeats):
        ex = LimeTabularExplainer(
            A, feature_names=cols, class_names=["no event", "event"],
            mode="classification", discretize_continuous=True,
            random_state=seed + r)
        e = ex.explain_instance(x0, f, num_features=len(cols), num_samples=samples)
        w = dict(e.as_map()[1])
        runs.append([w.get(j, 0.0) for j in range(len(cols))])

    M = np.array(runs)
    return pd.DataFrame(dict(
        feature=cols,
        lime=M.mean(0), lime_sd=M.std(0, ddof=1),
        lime_lo=M.min(0), lime_hi=M.max(0)
    ))

def run_shap(S, last, sck, mk, background=200, seed=SEED):
    A, cols = S["A"], S["cols"]
    rng = np.random.default_rng(seed)
    bg = A[rng.choice(len(A), size=min(background, len(A)), replace=False)]

    try:
        ex = shap.TreeExplainer(mk, shap.maskers.Independent(
            sck.transform(bg), max_samples=background),
            model_output="probability")
        sv = ex.shap_values(sck.transform(A[last][None, :]))
    except Exception:
        ex = shap.Explainer(mk.predict_proba, sck.transform(bg))
        sv = ex(sck.transform(A[last][None, :])).values[:, :, 1]

    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[0, :, -1]
    elif sv.ndim == 2:
        sv = sv[0]
    base = float(np.ravel(getattr(ex, "expected_value", [0.05]))[-1]) if hasattr(ex, "expected_value") else 0.05
    return pd.DataFrame(dict(feature=cols, shap=np.ravel(sv)[:len(cols)])), base

def explain(S, issuer, samples=N_SAMPLES, repeats=N_REPEATS, seed=SEED):
    last, sck, mk = issuer_row(S, issuer)
    if last is None:
        return None
    A, cols = S["A"], S["cols"]
    x0 = A[last]
    pd_now = float(predict_fn(sck, mk)(x0[None, :])[0, 1])

    L = run_lime(S, last, sck, mk, samples, repeats, seed)
    Sh, base = run_shap(S, last, sck, mk, seed=seed)
    d = L.merge(Sh, on="feature")

    med = np.median(A, axis=0)
    sd = A.std(0, ddof=1)
    pct = np.array([100.0 * (A[:, j] < x0[j]).mean() for j in range(len(cols))])
    d["value"] = x0
    d["median"] = med
    d["pct_of_median"] = np.where(np.abs(med) > 1e-12,
                                  100.0 * (x0 - med) / np.abs(med), np.nan)
    d["sd_from_median"] = np.where(sd > 0, (x0 - med) / sd, 0.0)
    d["percentile"] = pct

    tot = np.abs(d.shap).sum()
    d["shap_share"] = 100.0 * d.shap / tot if tot > 0 else 0.0
    d["shap_pct_of_pd"] = 100.0 * d.shap / pd_now if pd_now > 0 else np.nan
    ltot = np.abs(d.lime).sum()
    d["lime_share"] = 100.0 * d.lime / ltot if ltot > 0 else 0.0

    d["dir_lime"] = np.where(d.lime > 0, "เพิ่ม", np.where(d.lime < 0, "ลด", "-"))
    d["dir_shap"] = np.where(d.shap > 0, "เพิ่ม", np.where(d.shap < 0, "ลด", "-"))
    d["agree"] = np.sign(d.lime) == np.sign(d.shap)
    d["lime_stable"] = np.sign(d.lime_lo) == np.sign(d.lime_hi)

    d = d.sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True)
    return dict(issuer=issuer, row=int(last), pd_now=pd_now, base=base,
                thr=S["thr"], month=str(S["panel"].loc[last, "month"]),
                table=d, dataset_choice=S["dataset_choice"])

def render_figure(res, top=TOP, dpi=130):
    d = res["table"].head(top).iloc[::-1]
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 0.46 * len(d) + 3.1),
                             gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]})

    # A. LIME
    ax = axes[0]
    col = ["#b91c1c" if v > 0 else "#1d4ed8" for v in d.lime]
    err = np.vstack([(d.lime - d.lime_lo).to_numpy(),
                     (d.lime_hi - d.lime).to_numpy()])
    ax.barh(y, d.lime, color=col, alpha=0.88,
            xerr=np.abs(err), error_kw=dict(ecolor="#334155", lw=1.0, capsize=2.5))
    for i, (v, ok) in enumerate(zip(d.lime, d.lime_stable)):
        if not ok:
            ax.text(0, i, "  unstable", va="center", fontsize=7.5,
                    color="#92400e", fontweight="bold")
    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d.feature, fontsize=8.5)
    ax.set_xlabel("LIME surrogate weight", fontsize=9)
    ax.set_title("A. LIME\nwhiskers span the range over seeds",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # B. SHAP
    ax = axes[1]
    col = ["#b91c1c" if v > 0 else "#1d4ed8" for v in d.shap]
    ax.barh(y, d.shap, color=col, alpha=0.88)
    for i, (v, s) in enumerate(zip(d.shap, d.shap_pct_of_pd)):
        ax.text(v, i, f"  {v:+.5f} ({s:+.1f}%)", va="center", fontsize=7.6,
                color="#111827")
    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("SHAP contribution (probability units)", fontsize=9)
    ax.set_title(f"B. SHAP\nadds back to the PD exactly, base {res['base']:.5f}",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # C. Position vs Median
    ax = axes[2]
    v = d.sd_from_median.to_numpy()
    col = ["#ea580c" if abs(t) >= 1 else "#94a3b8" for t in v]
    ax.barh(y, v, color=col, alpha=0.88)
    for i, (t, p, pm) in enumerate(zip(v, d.percentile, d.pct_of_median)):
        lab = f"  p{p:.0f}"
        if np.isfinite(pm) and abs(pm) < 1e5:
            lab += f", {pm:+.0f}% vs median"
        ax.text(t, i, lab, va="center", fontsize=7.4, color="#111827")
    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("distance from the panel median (SD)", fontsize=9)
    ax.set_title("C. where this issuer sits on each determinant",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    agree = int(res["table"].head(top).agree.sum())
    ds_label = "Dataset 2 (941 firms)" if res.get("dataset_choice") == 2 else "Dataset 1 (Legacy 33 features)"
    fig.suptitle(
        f"[{ds_label}]  {res['issuer']}   month {res['month']}   "
        f"PD = {res['pd_now']:.6f}   review threshold = {res['thr']:.6f}   "
        f"agree = {agree}/{top}\n"
        f"red = pushes PD up    blue = pulls PD down",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig

def save_figure_jpg_png(fig, issuer: str):
    jpg_path = os.path.join(OUTDIR_JPG, f"lime_shap_{issuer}.jpg")
    png_path = os.path.join(OUTDIR_FIGS, f"lime_shap_{issuer}.png")
    
    fig.savefig(png_path, dpi=130, bbox_inches="tight", pad_inches=0.10)
    im = Image.open(png_path).convert("RGB")
    im.save(jpg_path, "JPEG", quality=95)
    plt.close(fig)
    return jpg_path, png_path

def report(res, top=TOP):
    d = res["table"]
    ds_name = "Dataset 2: 941 firms" if res.get("dataset_choice") == 2 else "Dataset 1: Legacy"
    bar = "=" * 100
    print(bar)
    print(f"[{ds_name}] {res['issuer']}   เดือน {res['month']}   PD = {res['pd_now']:.6f}   "
          f"เส้นเตือนภัย = {res['thr']:.6f}   "
          f"{'เหนือเส้น' if res['pd_now'] >= res['thr'] else 'ใต้เส้น'}")
    print(f"ฐานของ SHAP (ค่าเฉลี่ยของโมเดล) = {res['base']:.6f}   "
          f"ผลรวม SHAP = {d.shap.sum():+.6f}   "
          f"ฐาน + ผลรวม = {res['base'] + d.shap.sum():.6f}")
    print(bar)
    print(f"{'ตัวแปร':<28} {'ค่าปัจจุบัน':>12} {'%จากกลาง':>10} {'SD':>7} "
          f"{'SHAP':>11} {'%ของPD':>9} {'LIME':>10} {'ทิศ':>6} {'ตรงกัน':>7}")
    for _, r in d.head(top).iterrows():
        pm = f"{r.pct_of_median:+.0f}%" if np.isfinite(r.pct_of_median) and abs(r.pct_of_median) < 1e5 else "-"
        flag = "ใช่" if r.agree else "ไม่"
        note = "" if r.lime_stable else " *"
        print(f"{r.feature:<28} {r.value:>12.4f} {pm:>10} "
              f"{r.sd_from_median:>+7.2f} {r.shap:>+11.6f} "
              f"{r.shap_pct_of_pd:>+8.1f}% {r.lime:>+10.5f} "
              f"{r.dir_shap:>6} {flag:>7}{note}")
    n_bad = int((~d.head(top).lime_stable).sum())
    if n_bad:
        print(f"\n  * มี {n_bad} ตัวที่ค่า LIME เปลี่ยนเครื่องหมายเมื่อรันคนละ seed ไม่ควรนำไปตีความ")
    agree = int(d.head(top).agree.sum())
    print(f"\n  สองวิธีเห็นตรงกันเรื่องทิศทาง {agree} จาก {top} ตัว")
    up = d.head(top)[d.head(top).shap > 0]
    if len(up):
        print(f"  ตัวที่ดัน PD ขึ้นมากสุด: {up.iloc[0].feature} "
              f"({up.iloc[0].shap:+.6f} คิดเป็น {up.iloc[0].shap_pct_of_pd:+.1f}% ของ PD)")
    dn = d.head(top)[d.head(top).shap < 0]
    if len(dn):
        print(f"  ตัวที่ดึง PD ลงมากสุด : {dn.iloc[0].feature} "
              f"({dn.iloc[0].shap:+.6f} คิดเป็น {dn.iloc[0].shap_pct_of_pd:+.1f}% ของ PD)")

def main():
    ap = argparse.ArgumentParser(
        description="LIME กับ SHAP เทียบกัน บนแผง 33 ตัวแปรจริง (รองรับทั้งชุดเก่าและชุดใหม่ 941 บริษัท)")
    ap.add_argument("--dataset-choice", "-d", type=int, choices=[1, 2], default=None,
                    help="เลือกชุดข้อมูล: 1 = ของเก่า (ibond_33features_panel), 2 = ชุดใหม่ (ibond_33features_panel_941firm)")
    ap.add_argument("--issuer", type=str, default=None)
    ap.add_argument("--all-high-risk", action="store_true", help="ทำทุกรายที่อยู่เหนือเส้นเตือนภัย")
    ap.add_argument("--workload", type=float, default=DEFAULT_WORKLOAD)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()

    choice = a.dataset_choice
    if choice is None:
        if sys.stdin.isatty():
            choice = DataAdapter.show_menu()
        else:
            choice = 1

    print(f"\n--- เริ่มการทำงานด้วยชุดข้อมูล: [{choice}] {DataAdapter.DATASET_CHOICES[choice]['name']} ---")
    S = load_state(choice=choice, workload=a.workload)
    tab = S["tab"]
    high = tab[tab.status == "HIGH RISK"].issuer.tolist()
    
    if a.all_high_risk:
        targets = high
    elif a.issuer:
        targets = [a.issuer]
    else:
        targets = high[:1] if len(high) > 0 else tab.issuer.head(1).tolist()

    print(f"เส้นเตือนภัยที่กำลังคน {a.workload:.0%} = {S['thr']:.6f} | HIGH RISK {len(high)} ราย")
    print(f"LIME: สุ่ม {a.samples:,} จุด ทำซ้ำ {a.repeats} seed | SHAP: TreeExplainer\n")

    allrows = []
    for iss in targets:
        res = explain(S, iss, a.samples, a.repeats, a.seed)
        if res is None:
            print(f"{iss}: ไม่มีข้อมูลในแผง")
            continue
        report(res, a.top)
        t = res["table"].copy()
        t.insert(0, "issuer", iss)
        t.insert(1, "pd_now", res["pd_now"])
        allrows.append(t)
        if not a.no_figure:
            fig = render_figure(res, a.top)
            jpg_f, png_f = save_figure_jpg_png(fig, iss)
            print(f"  [Output JPG] -> {jpg_f}")
            print(f"  [Output PNG] -> {png_f}")
        print()

    if allrows:
        out = pd.concat(allrows, ignore_index=True)
        csv_fn = os.path.join(HERE, "tex_out", f"lime_shap_ds{choice}.csv")
        out.to_csv(csv_fn, index=False)
        print(f"เขียนตารางสรุป {csv_fn} ({len(out):,} แถว จาก {len(allrows)} บริษัท)")

if __name__ == "__main__":
    main()
