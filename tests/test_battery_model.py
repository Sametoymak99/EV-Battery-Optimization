"""src/battery_model.py için analitik doğrulama ve birim testleri (Aşama 3)."""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.battery_model import (
    BatteryLimits,
    BatteryParameters,
    OCVInterpolator,
    euler_stability_limit,
    simulate_current_profile,
    simulate_soc_profile,
)

OCV_CSV = PROJECT_ROOT / "data" / "processed" / "stanford_ocv_c20_diag1.csv"


def test_analytical_rc_polarization_and_soc():
    params = BatteryParameters(Q_Ah=4.85, eta=1.0, R0_Ohm=0.025, R1_Ohm=0.02, C1_F=1500.0)
    ocv = lambda s: 3.5 + 0.6 * s
    t_eval = np.linspace(0.0, 180.0, 181)  # dt = 1.0 s
    i_const = 4.0

    res_rk4 = simulate_current_profile(t_eval, np.full_like(t_eval, i_const), params, ocv, method="rk4")
    res_ivp = simulate_current_profile(t_eval, np.full_like(t_eval, i_const), params, ocv, method="solve_ivp")

    tau = params.R1_Ohm * params.C1_F
    vp_exact = i_const * params.R1_Ohm * (1.0 - np.exp(-t_eval / tau))
    soc_exact = 0.20 + (params.eta * i_const / (3600.0 * params.Q_Ah)) * t_eval

    assert np.max(np.abs(res_rk4["vp_V"] - vp_exact)) < 1e-6
    assert np.max(np.abs(res_ivp["vp_V"] - vp_exact)) < 1e-6
    assert np.max(np.abs(res_rk4["soc"] - soc_exact)) < 1e-10
    assert np.max(np.abs(res_ivp["soc"] - soc_exact)) < 1e-10


def test_analytical_thermal_cooling():
    params = BatteryParameters(Rth_K_per_W=10.0, Cth_J_per_K=60.0, T_amb_C=23.0)
    ocv = lambda s: 3.7 + 0.0 * s
    t_eval = np.linspace(0.0, 1200.0, 601)  # dt = 2.0 s
    x0 = np.array([0.50, 0.0, 40.0, 0.0, 0.0])

    res_rk4 = simulate_current_profile(t_eval, np.zeros_like(t_eval), params, ocv, x0=x0, method="rk4")
    res_ivp = simulate_current_profile(t_eval, np.zeros_like(t_eval), params, ocv, x0=x0, method="solve_ivp")

    tau_th = params.Rth_K_per_W * params.Cth_J_per_K
    t_exact = params.T_amb_C + (40.0 - params.T_amb_C) * np.exp(-t_eval / tau_th)

    assert np.max(np.abs(res_rk4["temperature_C"] - t_exact)) < 1e-6
    assert np.max(np.abs(res_ivp["temperature_C"] - t_exact)) < 1e-6


def test_ocv_interpolator_real_w8():
    ocv = OCVInterpolator.from_processed_csv(OCV_CSV, cell_id="W8")
    grid = np.linspace(0.0, 1.0, 101)
    vals = ocv(grid)
    assert vals.shape == (101,)
    assert np.all(np.diff(vals) >= 0.0)
    assert 2.49 <= float(vals[0]) <= 2.55
    assert 4.15 <= float(vals[-1]) <= 4.21
    assert isinstance(ocv(0.5), float)


def test_simulate_soc_profile_and_solver_agreement():
    ocv = OCVInterpolator.from_processed_csv(OCV_CSV, cell_id="W8")
    params = BatteryParameters()
    currents = [6.0, 5.5, 5.0, 4.5, 4.0, 3.0, 2.0]

    res_eu = simulate_soc_profile(currents, params, ocv, method="euler", dt_max_s=0.5)
    res_rk4 = simulate_soc_profile(currents, params, ocv, method="rk4", dt_max_s=0.5)
    res_ivp = simulate_soc_profile(currents, params, ocv, method="solve_ivp", dt_max_s=0.5)

    for res in (res_eu, res_rk4, res_ivp):
        assert abs(res["final_soc"] - 0.90) < 1e-10
        assert res["total_energy_loss_J"] > 0.0
        assert np.all(np.diff(res["aging_D"]) >= 0.0)

    assert abs(res_rk4["max_voltage_V"] - res_ivp["max_voltage_V"]) < 1e-4
    assert abs(res_eu["max_voltage_V"] - res_ivp["max_voltage_V"]) < 5e-3


def test_euler_stability_limit_and_coarse_accuracy():
    params = BatteryParameters(R1_Ohm=0.02, C1_F=500.0, Rth_K_per_W=10.0, Cth_J_per_K=60.0)
    assert np.isclose(euler_stability_limit(params), 20.0)  # 2 * min(10, 600)

    ocv = lambda s: 3.6 + 0.5 * s
    t_coarse = np.linspace(0.0, 100.0, 11)  # dt = 10 s (büyük adım)
    tau = params.R1_Ohm * params.C1_F
    vp_exact = 4.0 * params.R1_Ohm * (1.0 - np.exp(-t_coarse / tau))

    res_eu = simulate_current_profile(t_coarse, np.full_like(t_coarse, 4.0), params, ocv, method="euler")
    res_rk4 = simulate_current_profile(t_coarse, np.full_like(t_coarse, 4.0), params, ocv, method="rk4")

    err_eu = np.max(np.abs(res_eu["vp_V"] - vp_exact))
    err_rk4 = np.max(np.abs(res_rk4["vp_V"] - vp_exact))
    assert err_rk4 < err_eu * 0.05
    assert BatteryLimits().v_max_V == 4.20
