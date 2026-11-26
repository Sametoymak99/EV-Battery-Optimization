"""src/numerical_analysis.py için birim testleri (Aşama 5)."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.battery_model import BatteryParameters, OCVInterpolator
from src.numerical_analysis import benchmark_finite_difference_gradients, benchmark_ode_solvers

PROC_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def test_ode_convergence_orders_and_euler_instability():
    json_path = PROC_DIR / "calibrated_parameters.json"
    params = BatteryParameters.from_calibrated_json(json_path)
    ocv = OCVInterpolator.from_calibrated_json(json_path)

    df, orders, instab = benchmark_ode_solvers(params, ocv, n_repeats=2)
    assert 0.85 <= orders["euler_order_vp"] <= 1.15
    assert 3.70 <= orders["rk4_order_vp"] <= 4.30
    assert instab["err_vp_euler_max_V"] > 0.30
    assert instab["err_vp_euler_max_V"] > 10.0 * instab["err_vp_rk4_max_V"]
    assert not df.empty


def test_finite_difference_gradient_accuracy():
    json_path = PROC_DIR / "calibrated_parameters.json"
    params = BatteryParameters.from_calibrated_json(json_path)
    ocv = OCVInterpolator.from_calibrated_json(json_path)

    fd_df = benchmark_finite_difference_gradients(params, ocv)
    row_1e5 = fd_df[fd_df["h_A"] == 1e-5].iloc[0]
    row_1e3 = fd_df[fd_df["h_A"] == 1e-3].iloc[0]

    assert row_1e5["central_rel_err_tf"] < 1e-9
    assert row_1e3["central_rel_err_tf"] < row_1e3["forward_rel_err_tf"]


def test_phase5_outputs_exist():
    for fn in (
        "phase5_ode_solver_comparison.csv",
        "phase5_soc_dt_sensitivity.csv",
        "phase5_fd_gradient_sensitivity.csv",
    ):
        p = TABLES_DIR / fn
        assert p.exists()
        assert len(pd.read_csv(p)) > 0

    for fn in (
        "fig05_ode_convergence_and_runtime.png",
        "fig05_fd_step_sensitivity.png",
    ):
        assert (FIGURES_DIR / fn).exists()
