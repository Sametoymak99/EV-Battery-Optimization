"""src/parameter_estimation.py için birim testleri (Aşama 4)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.battery_model import BatteryParameters, OCVInterpolator
from src.parameter_estimation import estimate_electrical_params_hppc, simulate_1rc_voltage

PROC_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def test_synthetic_hppc_parameter_recovery():
    t = np.arange(0.0, 2000.0, 1.0)
    curr = np.zeros_like(t)
    curr[100:200] = -4.5
    curr[500:550] = 2.0
    curr[900:1100] = -4.5
    curr[1400:1500] = -3.0

    ocv = OCVInterpolator(np.linspace(0.0, 1.0, 21), 3.0 + 1.2 * np.linspace(0.0, 1.0, 21), cell_id="synth")
    true_r0, true_r1, true_c1 = 0.028, 0.014, 1800.0
    q_ah = 4.85
    soc = np.clip(1.0 + np.cumsum(curr) / (3600.0 * q_ah), 0.0, 1.0)
    vt, _ = simulate_1rc_voltage(t, curr, ocv(soc), true_r0, true_r1, true_c1)

    df_synth = pd.DataFrame({"time_s": t, "current_A": curr, "voltage_V": vt})
    res = estimate_electrical_params_hppc(df_synth, ocv, q_ah=q_ah, soc_window=(0.10, 0.999))
    assert abs(res["R0_Ohm"] - true_r0) / true_r0 < 0.02
    assert abs(res["R1_Ohm"] - true_r1) / true_r1 < 0.02
    assert abs(res["C1_F"] - true_c1) / true_c1 < 0.02


def test_calibrated_parameters_physical_bounds_and_json_load():
    json_path = PROC_DIR / "calibrated_parameters.json"
    assert json_path.exists()

    params = BatteryParameters.from_calibrated_json(json_path)
    ocv = OCVInterpolator.from_calibrated_json(json_path)

    assert params.is_calibrated is True
    assert 4.70 <= params.Q_Ah <= 5.00
    assert 0.010 <= params.R0_Ohm <= 0.060
    assert 0.002 <= params.R1_Ohm <= 0.080
    assert 100.0 <= params.C1_F <= 15000.0
    assert 3.0 <= params.Rth_K_per_W <= 30.0
    assert 30.0 <= params.Cth_J_per_K <= 150.0
    assert 2.49 <= ocv(0.0) <= 2.55
    assert 4.15 <= ocv(1.0) <= 4.21


def test_eis_agreement_and_independent_validation_metrics():
    param_tbl = pd.read_csv(TABLES_DIR / "phase4_parameter_estimation.csv")
    r0_hppc = float(param_tbl.loc[param_tbl["parameter"] == "R0_Ohm", "value"].iloc[0])
    r0_eis = float(param_tbl.loc[param_tbl["parameter"] == "R0_EIS_Ohm", "value"].iloc[0])
    assert abs(r0_hppc - r0_eis) < 0.015

    val_tbl = pd.read_csv(TABLES_DIR / "phase4_validation_metrics.csv")
    indep = val_tbl[val_tbl["split"] == "independent_validation"]
    assert set(indep["cell_id"]) == {"W5", "V4"}
    assert (indep["rmse_mV"] < 35.0).all()

    import json
    with open(PROC_DIR / "calibrated_parameters.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["thermal_fit_metrics"]["rmse_C"] < 1.5
