"""src/evaluation.py için birim testleri (Aşama 10–11)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.battery_model import BatteryParameters
from src.evaluation import identify_pareto_front, sample_uncertain_parameters

PROC_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
OUT_DIR = PROJECT_ROOT / "results" / "simulation_outputs"


def test_identify_pareto_front_synthetic():
    df = pd.DataFrame(
        {
            "tf_min": [30.0, 45.0, 50.0, 40.0],
            "energy_loss_Wh": [1.00, 0.60, 0.80, 0.70],
            "aging_D": [2.0e-4, 1.1e-4, 1.5e-4, 1.2e-4],
        }
    )
    mask = identify_pareto_front(df)
    # 0 (hızlı), 1 (verimli/uzun ömür), 3 (ara denge) Pareto; 2 ise 1 ve 3 tarafından baskılanır
    assert list(mask) == [True, True, False, True]


def test_sample_uncertain_parameters_deterministic_and_positive():
    params = BatteryParameters.from_calibrated_json(PROC_DIR / "calibrated_parameters.json")
    stds = {"Q_Ah": 0.025, "R0_Ohm": 0.0037, "R1_Ohm": 0.0004, "C1_F": 128.0, "Rth_K_per_W": 0.64, "Cth_J_per_K": 13.5}

    s1 = sample_uncertain_parameters(params, stds, n_samples=50, seed=2026)
    s2 = sample_uncertain_parameters(params, stds, n_samples=50, seed=2026)
    pd.testing.assert_frame_equal(s1, s2)

    for col in ("Q_Ah", "R0_Ohm", "R1_Ohm", "C1_F", "Rth_K_per_W", "Cth_J_per_K"):
        assert (s1[col] > 0.0).all()


def test_phase10_and_11_outputs_valid():
    p10 = pd.read_csv(TABLES_DIR / "phase10_pareto_solutions.csv")
    assert p10["feasible"].sum() >= 10
    assert p10["is_pareto_optimal"].sum() >= 3

    p11_sum = pd.read_csv(TABLES_DIR / "phase11_monte_carlo_summary.csv")
    assert set(p11_sum["strategy"]) == {"SLSQP Optimal", "CC-CV (1.0C)"}
    assert (p11_sum["n_samples"] == 200).all()

    p11_raw = pd.read_csv(OUT_DIR / "phase11_monte_carlo_raw.csv")
    assert len(p11_raw) == 400
    assert p11_raw.isna().sum().sum() == 0

    assert (FIGURES_DIR / "fig10_pareto_analysis.png").exists()
    assert (FIGURES_DIR / "fig11_monte_carlo_uncertainty.png").exists()
