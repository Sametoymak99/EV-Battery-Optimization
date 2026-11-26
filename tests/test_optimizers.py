"""src/optimizers.py, objective_functions.py ve constraints.py için birim testleri (Aşama 7–9)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.battery_model import BatteryLimits, BatteryParameters, OCVInterpolator
from src.objective_functions import ReferenceScales
from src.optimizers import optimize_adam, optimize_gradient_descent, optimize_slsqp

PROC_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
OUT_DIR = PROJECT_ROOT / "results" / "simulation_outputs"


def _load_ctx():
    json_path = PROC_DIR / "calibrated_parameters.json"
    params = BatteryParameters.from_calibrated_json(json_path)
    ocv = OCVInterpolator.from_calibrated_json(json_path)
    limits = BatteryLimits()
    ref = ReferenceScales.from_json(PROC_DIR / "reference_scales.json")
    return params, ocv, limits, ref


def test_all_three_optimizers_feasible_and_beat_cccv():
    params, ocv, limits, ref = _load_ctx()
    for fn in (optimize_gradient_descent, optimize_adam, optimize_slsqp):
        res = fn(params, ocv, limits, ref, weights=(0.4, 0.2, 0.4), seed=42)
        cons = res["constraints"]
        assert cons["is_feasible"] is True
        assert cons["max_voltage_V"] <= limits.v_max_V + 2e-3
        assert cons["max_temperature_C"] <= limits.t_max_C + 5e-2
        assert cons["soc_target_err"] < 1e-6
        assert 0.50 - 1e-6 <= cons["current_min_A"]
        assert cons["current_max_A"] <= limits.i_max_A + 1e-6
        assert res["objectives"]["J_total"] < 1.0


def test_same_seed_reproducibility():
    params, ocv, limits, ref = _load_ctx()
    r1 = optimize_adam(params, ocv, limits, ref, weights=(0.4, 0.2, 0.4), seed=42, max_iters_per_stage=15)
    r2 = optimize_adam(params, ocv, limits, ref, weights=(0.4, 0.2, 0.4), seed=42, max_iters_per_stage=15)
    assert np.allclose(r1["u_opt_A"], r2["u_opt_A"])
    assert np.isclose(r1["objectives"]["J_total"], r2["objectives"]["J_total"])


def test_phase9_artifacts_exist():
    assert (TABLES_DIR / "phase9_optimizer_comparison.csv").exists()
    assert (TABLES_DIR / "phase9_multiseed_sensitivity.csv").exists()
    assert (OUT_DIR / "phase9_optimal_profiles.json").exists()
    comp = pd.read_csv(TABLES_DIR / "phase9_optimizer_comparison.csv")
    assert len(comp) == 4
    for fn in ("fig07_gradient_descent.png", "fig08_adam_optimization.png", "fig09_optimizer_comparison.png"):
        assert (FIGURES_DIR / fn).exists()
