"""src/data_loader.py için birim testleri (Aşama 1)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    NASA_DATASET_ID,
    STANFORD_DATASET_ID,
    extract_nasa_cycle_df,
    extract_stanford_diag_df,
    inspect_timeseries,
    load_nasa_mat,
    load_stanford_cycling,
    load_stanford_diagnostic,
)

DATA_RAW = PROJECT_ROOT / "data" / "raw"


def test_load_stanford_diagnostics_real():
    cap_path = DATA_RAW / "stanford" / "diagnostic" / "capacity_test.mat"
    hppc_path = DATA_RAW / "stanford" / "diagnostic" / "HPPC_test.mat"
    eis_path = DATA_RAW / "stanford" / "diagnostic" / "EIS_test.mat"

    cap_data = load_stanford_diagnostic(cap_path)
    assert cap_data["dataset"] == STANFORD_DATASET_ID
    assert "W8" in cap_data["cell_labels"]
    df_cap = extract_stanford_diag_df(cap_data, "W8", 1)
    assert set(["time_s", "voltage_V", "current_A", "capacity_Ah"]).issubset(df_cap.columns)
    assert len(df_cap) > 50000
    assert df_cap["current_A"].max() <= 0.001  # C/20 deşarj testi

    hppc_data = load_stanford_diagnostic(hppc_path)
    df_hppc = extract_stanford_diag_df(hppc_data, "W8", 1)
    assert df_hppc["current_A"].min() < -4.0  # -1C darbe
    assert df_hppc["current_A"].max() > 1.5   # +0.5C şarj darbesi

    eis_data = load_stanford_diagnostic(eis_path)
    df_eis = extract_stanford_diag_df(eis_data, "W8", 1)
    assert set(["soc_pct", "freq_Hz", "re_z_Ohm", "im_z_Ohm"]).issubset(df_eis.columns)
    assert set(df_eis["soc_pct"].unique()) == {20, 50, 80}


def test_load_stanford_cycling_real_and_zero_spike():
    w10_path = DATA_RAW / "stanford" / "cycling" / "Cycling_1" / "W10.mat"
    cyc = load_stanford_cycling(w10_path)
    assert cyc["dataset"] == STANFORD_DATASET_ID
    assert cyc["cell_id"] == "W10"
    assert cyc["c_rate"] == "3C"
    assert cyc["current_sign_convention"] == "positive_charge_negative_discharge"

    df = cyc["dataframe"]
    assert len(df) > 6_000_000
    # 3C şarj akımı pozitif (~14.55 A)
    assert df["current_A"].max() > 14.0
    # Step 0 ilk satırdaki 0V birleştirme/başlangıç artifaktı tespit edilmeli
    audit = inspect_timeseries(df.iloc[:10000], v_bounds=(2.5, 4.25))
    assert audit["voltage_zero_count"] >= 1
    assert audit["time_monotonic_inc"] is True


def test_load_nasa_real_and_separation():
    b0005_path = DATA_RAW / "nasa" / "B0005.mat"
    nasa = load_nasa_mat(b0005_path)
    assert nasa["dataset"] == NASA_DATASET_ID
    assert nasa["dataset"] != STANFORD_DATASET_ID
    assert nasa["nominal_capacity_Ah"] < 2.5  # Stanford 4.85 Ah ile karıştırılmamalı

    assert not nasa["capacity_fade"].empty
    assert nasa["capacity_fade"]["capacity_Ah"].iloc[0] > nasa["capacity_fade"]["capacity_Ah"].iloc[-1]

    chg_idx = int(nasa["cycle_summary"].query("type == 'charge'")["cycle_index"].iloc[0])
    df_chg = extract_nasa_cycle_df(nasa, chg_idx)
    assert set(["time_s", "voltage_V", "current_A", "temperature_C"]).issubset(df_chg.columns)
    assert df_chg["temperature_C"].mean() > 20.0


def test_inspect_timeseries_synthetic_anomalies():
    df = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 1.0, 3.0],  # monoton değil (dt=0)
            "voltage_V": [0.0, 3.7, np.nan, 4.30],  # 0V, NaN ve >4.25V
            "current_A": [1.0, 1.0, 1.0, 1.0],
            "temperature_C": [25.0, 26.0, 65.0, 27.0],  # >60C
        }
    )
    audit = inspect_timeseries(df, temp_col="temperature_C", v_bounds=(2.5, 4.25), t_bounds=(0.0, 60.0))
    assert audit["n_rows"] == 4
    assert audit["nan_total"] == 1
    assert audit["time_monotonic_inc"] is False
    assert audit["non_pos_dt_count"] == 1
    assert audit["voltage_zero_count"] == 1
    assert audit["voltage_oob_count"] == 2
    assert audit["temp_oob_count"] == 1
