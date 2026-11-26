"""src/preprocessing.py için birim testleri (Aşama 2)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import extract_nasa_cycle_df, extract_stanford_diag_df, load_nasa_mat, load_stanford_diagnostic
from src.preprocessing import (
    clean_nasa_cycle,
    clean_stanford_cycling,
    compute_soc_charge_cycle,
    compute_soc_from_capacity_test,
    segment_stanford_cycles,
)

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"


def test_clean_stanford_cycling_synthetic():
    df = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 2.0, 3.0, 4.0],
            "current_A": [0.0, 2.4, 2.4, 2.4, np.nan, 2.4],
            "voltage_V": [0.0, 3.6, 3.7, 3.7, 3.7, 4.30],
            "step_index": [0, 7, 7, 7, 7, 8],
        }
    )
    df_clean, rep = clean_stanford_cycling(df)
    assert rep["removed_step0"] == 1
    assert rep["removed_v_zero"] == 1
    assert rep["removed_v_oob"] == 2  # 0.0 V ve 4.30 V
    assert rep["removed_non_monotonic"] == 1
    assert rep["removed_nan"] == 1
    assert len(df_clean) == 2
    assert np.all(np.diff(df_clean["time_s"]) > 0)


def test_clean_nasa_removes_spikes_and_nans():
    b5 = load_nasa_mat(DATA_RAW / "nasa" / "B0005.mat")
    spike_found = 0
    for idx, cyc in enumerate(b5["cycles"]):
        if cyc["type"] in ("charge", "discharge"):
            df_c = extract_nasa_cycle_df(b5, idx)
            df_clean, rep = clean_nasa_cycle(df_c)
            spike_found += rep["removed_v_oob"]
            assert df_clean["voltage_V"].between(1.8, 4.35).all()
    assert spike_found >= 1

    b18 = load_nasa_mat(DATA_RAW / "nasa" / "B0018.mat")
    nans_removed = 0
    for idx, cyc in enumerate(b18["cycles"]):
        if cyc["type"] in ("charge", "discharge"):
            df_c = extract_nasa_cycle_df(b18, idx)
            df_clean, rep = clean_nasa_cycle(df_c)
            nans_removed += rep["removed_nan"]
            assert df_clean.isna().sum().sum() == 0
    assert nans_removed >= 1


def test_c20_soc_verification_real():
    cap_diag = load_stanford_diagnostic(DATA_RAW / "stanford" / "diagnostic" / "capacity_test.mat")
    df_c = extract_stanford_diag_df(cap_diag, "W8", 1)
    df_soc, ver = compute_soc_from_capacity_test(df_c)
    assert ver["max_abs_soc_diff"] < 0.005
    assert df_soc["soc"].between(0.0, 1.0).all()


def test_segment_and_charge_soc_synthetic():
    # Basit iki çevrimlik sentetik veri
    t = np.arange(0.0, 20.0, 1.0)
    steps = np.array([2, 3, 7, 7, 8, 9, 10, 11, 13, 14, 7, 7, 8, 9, 10, 11, 13, 14, 14, 14])
    df = pd.DataFrame(
        {
            "time_s": t,
            "current_A": np.where(np.isin(steps, [13, 14]), -1.2, np.where(steps == 11, 0.0, 2.4)),
            "voltage_V": np.full(20, 3.7),
            "step_index": steps,
        }
    )
    tagged, summary = segment_stanford_cycles(df)
    assert len(summary) == 2
    assert set(tagged["phase"].unique()).issubset({"charge", "rest", "discharge"})

    chg1 = tagged[(tagged["cycle_num"] == 1) & (tagged["step_index"].isin([7, 8, 9, 10]))].copy()
    chg1["time_s"] = chg1["time_s"] - chg1["time_s"].iloc[0]
    df_soc, stats = compute_soc_charge_cycle(chg1, capacity_Ah=4.85, soc_end=1.0)
    assert df_soc["soc"].between(0.0, 1.0).all()
    assert np.isclose(stats["soc_end"], 1.0)


def test_phase2_processed_outputs_exist_and_valid():
    therm_path = DATA_PROC / "stanford_w10_thermal_sample.csv"
    rep_chg_path = DATA_PROC / "stanford_rep_charge_profiles.csv"
    cyc_sum_path = DATA_PROC / "stanford_charge_cycles_summary.csv"

    assert therm_path.exists()
    assert rep_chg_path.exists()
    assert cyc_sum_path.exists()

    therm_df = pd.read_csv(therm_path)
    assert therm_df["temperature_C"].between(15.0, 55.0).all()
    assert (therm_df["temperature_C"].max() - therm_df["temperature_C"].min()) > 1.0

    rep_df = pd.read_csv(rep_chg_path)
    assert rep_df["soc"].between(0.0, 1.0).all()
    # %20-%80 UDDS sonrası şarj başlangıç SOC değerinin ~%20 civarında olması beklenir
    for cid in ("W4", "W8", "W9", "W10"):
        s0 = float(rep_df[rep_df["cell_id"] == cid]["soc"].iloc[0])
        assert 0.10 <= s0 <= 0.30

    csum = pd.read_csv(cyc_sum_path)
    for cid in ("W4", "W8", "W9", "W10"):
        assert len(csum[csum["cell_id"] == cid]) >= 20
