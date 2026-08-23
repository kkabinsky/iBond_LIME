# -*- coding: utf-8 -*-
"""
data_adapter.py -- Unified Data Adapter for LIME & Explainable AI (XAI) System.
Supports:
  Dataset 1: 'ibond_33features_panel' (Legacy 219-293 bond issuers, 16,986 firm-months)
  Dataset 2: 'ibond_33features_panel_941firm' (New 941 firms panel, 187,007 firm-months)

Automatically connects to `lime_credit.db` (or unzips `lime_credit.db.zip` if needed),
falls back to `cmdf_credit.db` or `bond_financials.db`.
"""

import os
import sqlite3
import zipfile
import numpy as np
import pandas as pd

BOND_33_FEATURES = [
    # 1. Liquidity & Trading (5)
    "amihud_monthly", "adj_illiq_kz", "percent_zero_days", "zero_days", "n_days",
    # 2. Financial Ratios & Solvency (14)
    "ROA", "ROE", "DE", "CurrentRatio", "QuickRatio", "CashRatio",
    "EBITtoTA", "REtoTA", "WorkingCapitaltoTA", "TDTA", "LTDtoTA", "STDtoTA",
    "cf_Interestcoverageratio", "acc_DebtServiceCoverageRatio",
    # 3. Scale & Age (2)
    "lnTotalAssets", "lnAge",
    # 4. Macroeconomic (3)
    "Policyrate", "GDPgrowth", "UnemploymentratemodeledILOe",
    # 5. ESG & Governance (6)
    "ESGScore", "GovernancePillarScore", "EnvironmentalPillarScore",
    "SocialPillarScore", "IndependentBoardMembers", "AverageBoardTenure",
]

class DataAdapter:
    """Unified Data Adapter supporting Dataset 1 (Legacy) and Dataset 2 (New 941 Firms)."""

    DATASET_CHOICES = {
        1: {
            "name": "ชุดข้อมูลเดิม: ibond_33features_panel (16,986 แถว, 219-293 บริษัท)",
            "table": "ibond_33features_panel",
            "target": "d_DP_RS",
            "desc": "Legacy panel focused on bond issuers with 33 features"
        },
        2: {
            "name": "ชุดข้อมูลใหม่: ibond_33features_panel_941firm (187,007 แถว, 941 บริษัท)",
            "table": "ibond_33features_panel_941firm",
            "target": "y_pre3m",
            "target_fallback": "d_DP_RS",
            "desc": "New comprehensive panel covering 941 SET/mai firms through Aug 2026"
        }
    }

    @classmethod
    def get_db_path(cls) -> str:
        """Finds or unpacks the SQLite database path."""
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "lime_credit.db"),
            os.path.join(here, "cmdf_credit.db"),
            os.path.join(here, "..", "cmdf_credit.db"),
            os.path.join(here, "..", "lime_credit.db"),
            os.path.join(here, "bond_financials.db"),
            os.path.join(here, "..", "bond_financials.db"),
        ]
        for p in candidates:
            if os.path.exists(p) and os.path.getsize(p) > 1024:
                return os.path.abspath(p)
                
        # Check for zip file
        zip_candidates = [
            os.path.join(here, "lime_credit.db.zip"),
            os.path.join(here, "..", "lime_credit.db.zip"),
        ]
        for zp in zip_candidates:
            if os.path.exists(zp):
                print(f"[DataAdapter] Extracting compact database from {zp}...")
                target_dir = os.path.dirname(zp)
                with zipfile.ZipFile(zp, 'r') as zf:
                    zf.extractall(target_dir)
                target_db = os.path.join(target_dir, "lime_credit.db")
                if os.path.exists(target_db):
                    return os.path.abspath(target_db)

        return os.path.abspath(candidates[0])

    @classmethod
    def show_menu(cls) -> int:
        """Displays interactive menu to select dataset."""
        print("=" * 80)
        print("     ระบบเลือกชุดข้อมูลวิเคราะห์ความเสี่ยงเครดิต (Thai Credit XAI EWS)")
        print("=" * 80)
        for k, v in cls.DATASET_CHOICES.items():
            print(f"  [{k}] {v['name']}")
            print(f"      -> {v['desc']}")
        print("=" * 80)
        
        while True:
            try:
                raw = input("กรุณาเลือกชุดข้อมูล [1 หรือ 2] (กด Enter เพื่อเลือก 2): ").strip()
                if raw == "":
                    return 2
                choice = int(raw)
                if choice in cls.DATASET_CHOICES:
                    return choice
                print("ตัวเลือกไม่ถูกต้อง กรุณาใส่ 1 หรือ 2")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\nใช้ค่าเริ่มต้น: [2]")
                return 2

    @classmethod
    def load(cls, choice: int = 2, verbose: bool = True):
        """Loads and prepares the chosen dataset from SQLite."""
        if choice not in cls.DATASET_CHOICES:
            raise ValueError(f"Invalid dataset choice: {choice}. Expected 1 or 2.")

        info = cls.DATASET_CHOICES[choice]
        table_name = info["table"]
        db_path = cls.get_db_path()

        if verbose:
            print(f"\n--- การทำงานบนชุดข้อมูล: [{choice}] {info['name']} ---")
            print(f"Connecting to DB: {db_path}")

        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]

        if table_name not in tables:
            con.close()
            raise RuntimeError(f"Table '{table_name}' not found in {db_path}. Available: {tables}")

        if verbose:
            print(f"Loading Table: '{table_name}' ({info['name']})")

        df = pd.read_sql(f"SELECT * FROM {table_name}", con)
        con.close()

        # Unify column naming
        if "issuer_code" not in df.columns:
            if "symbol" in df.columns:
                df["issuer_code"] = df["symbol"].astype(str)
            else:
                raise ValueError("Missing 'issuer_code' or 'symbol' in dataset!")
        else:
            df["issuer_code"] = df["issuer_code"].astype(str)

        if "month" in df.columns:
            df["month_dt"] = pd.to_datetime(df["month"], errors="coerce")
        elif "year_month" in df.columns:
            df["month_dt"] = pd.to_datetime(df["year_month"], errors="coerce")
            df["month"] = df["year_month"]
        else:
            df["month_dt"] = pd.to_datetime("2026-08-01")
            df["month"] = "2026-08"

        # Check target column
        target_col = info["target"]
        if target_col not in df.columns:
            fallback = info.get("target_fallback")
            if fallback and fallback in df.columns:
                target_col = fallback
            else:
                for cand in ["y_pre3m", "d_DP_RS", "d_Default_Payment", "default_event"]:
                    if cand in df.columns:
                        target_col = cand
                        break
        
        if target_col in df.columns:
            y = df[target_col].fillna(0).astype(int)
        else:
            y = pd.Series(0, index=df.index)

        # Select Available Features
        avail_features = [f for f in BOND_33_FEATURES if f in df.columns]
        X = df[avail_features].copy()

        # Clean Missing Values
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            med = X[col].median()
            if pd.isna(med):
                med = 0.0
            X[col] = X[col].fillna(med)

        # Retain metadata columns
        meta_cols = ["issuer_code", "month", "month_dt"]
        for c in ["name", "issuer_name", "issuer_name_th", "market", "sector", "industry"]:
            if c in df.columns:
                meta_cols.append(c)

        panel = pd.concat([df[meta_cols], X], axis=1)

        if verbose:
            print(f"  Dataset loaded: {len(df):,} rows | {df['issuer_code'].nunique():,} firms | {len(avail_features)}/30 Features")
            n_events = int(y.sum())
            n_ev_firms = df.loc[y == 1, 'issuer_code'].nunique() if n_events > 0 else 0
            print(f"  Distress/Default events: {n_events:,} months ({n_events/len(df)*100:.3f}%) from {n_ev_firms} firms")

        return panel, X, y, avail_features
