"""src/cc_cv.py için birim testleri (Aşama 6)."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.battery_model import BatteryLimits, BatteryParameters, OCVInterpolator
from src.cc_cv import simulate_cc_cv
from src.objective_functions import ReferenceScales

PROC_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def test_cc_cv_voltage_clamp_and_soc_target():
    json_path = PROC_DIR / "calibrated_parameters.json"
    params = BatteryParameters.from_calibrated_json(json_path)
    ocv = OCVInterpolator.from_calibrated_json(json_path)
    limits = BatteryLimits()

    sim_05 = simulate_cc_cv(params, ocv, limits, i_cc_A=0.5 * params.Q_Ah)
    sim_10 = simulate_cc_cv(params, ocv, limits, i_cc_A=1.0 * params.Q_Ah)
    sim_20 = simulate_cc_cv(params, ocv, limits, i_cc_A=2.0 * params.Q_Ah)

    for sim in (sim_05, sim_10, sim_20):
        assert abs(sim["final_soc"] - 0.90) < 1e-5
        assert sim["max_voltage_V"] <= limits.v_max_V + 5e-4
        assert sim["total_energy_loss_J"] > 0.0
        assert sim["total_aging_D"] > 0.0

    # Daha yüksek CC oranı daha kısa toplam şarj süresi üretmelidir
    assert sim_20["final_time_s"] < sim_10["final_time_s"] < sim_05["final_time_s"]
    assert sim_10["cv_duration_s"] > 0.0


def test_phase6_reference_scales_and_artifacts():
    ref = ReferenceScales.from_json(PROC_DIR / "reference_scales.json")
    assert ref.t_ref_s > 1000.0
    assert ref.E_ref_J > 500.0
    assert ref.D_ref > 0.0
    assert (TABLES_DIR / "phase6_cc_cv_comparison.csv").exists()
    assert len(pd.read_csv(TABLES_DIR / "phase6_cc_cv_comparison.csv")) == 5
    assert (FIGURES_DIR / "fig06_cc_cv_reference.png").exists()
