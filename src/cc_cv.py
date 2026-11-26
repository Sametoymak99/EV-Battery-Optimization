"""Aşama 6: Geleneksel CC-CV referans şarj stratejisi simülatörü."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.battery_model import (
    BatteryLimits,
    BatteryParameters,
    OCVInterpolator,
    instantaneous_power_loss_W,
    ode_rhs,
    terminal_voltage,
)


def _cc_cv_current(
    y: np.ndarray,
    i_cc_A: float,
    v_max_V: float,
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    i_cutoff_A: float,
) -> tuple[float, str]:
    """Duruma bağlı anlık CC-CV akımını ve faz etiketini ('CC' veya 'CV') döndürür."""
    soc, vp_v = float(y[0]), float(y[1])
    i_limit_v = (v_max_V - float(ocv_fn(soc)) - vp_v) / params.R0_Ohm
    if i_limit_v >= i_cc_A:
        return float(i_cc_A), "CC"
    return max(float(i_cutoff_A), float(i_limit_v)), "CV"


def simulate_cc_cv(
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    limits: BatteryLimits | None = None,
    i_cc_A: float = 4.86,
    soc_init: float = 0.20,
    soc_target: float = 0.90,
    T_init_C: float | None = None,
    dt_s: float = 1.0,
    i_cutoff_A: float = 0.05,
    max_time_s: float = 36000.0,
) -> dict[str, Any]:
    """RK4 kullanarak Vmax sınırlı geleneksel CC-CV şarj simülasyonu yapar."""
    lim = limits or BatteryLimits()
    if i_cc_A <= 0.0:
        raise ValueError("CC akımı pozitif olmalıdır.")

    t_init = params.T_amb_C if T_init_C is None else float(T_init_C)
    y = np.array([soc_init, 0.0, t_init, 0.0, 0.0], dtype=np.float64)

    t_list = [0.0]
    y_list = [y.copy()]
    i0, ph0 = _cc_cv_current(y, i_cc_A, lim.v_max_V, params, ocv_fn, i_cutoff_A)
    i_list = [i0]
    ph_list = [ph0]

    t_curr = 0.0
    t_wall0 = time.perf_counter()

    def _rhs_cc_cv(t_val: float, y_val: np.ndarray) -> np.ndarray:
        i_val, _ = _cc_cv_current(y_val, i_cc_A, lim.v_max_V, params, ocv_fn, i_cutoff_A)
        return ode_rhs(t_val, y_val, i_val, params)

    while y[0] < soc_target and t_curr < max_time_s:
        k1 = _rhs_cc_cv(t_curr, y)
        k2 = _rhs_cc_cv(t_curr + 0.5 * dt_s, y + 0.5 * dt_s * k1)
        k3 = _rhs_cc_cv(t_curr + 0.5 * dt_s, y + 0.5 * dt_s * k2)
        k4 = _rhs_cc_cv(t_curr + dt_s, y + dt_s * k3)
        y_next = y + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if y_next[0] >= soc_target:
            # Tam hedef SOC anına doğrusal/RK4 adım enterpolasyonu
            frac = float((soc_target - y[0]) / max(y_next[0] - y[0], 1e-12))
            dt_rem = max(1e-6, frac * dt_s)
            k1r = _rhs_cc_cv(t_curr, y)
            k2r = _rhs_cc_cv(t_curr + 0.5 * dt_rem, y + 0.5 * dt_rem * k1r)
            k3r = _rhs_cc_cv(t_curr + 0.5 * dt_rem, y + 0.5 * dt_rem * k2r)
            k4r = _rhs_cc_cv(t_curr + dt_rem, y + dt_rem * k3r)
            y = y + (dt_rem / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
            t_curr += dt_rem
        else:
            y = y_next
            t_curr += dt_s

        i_now, ph_now = _cc_cv_current(y, i_cc_A, lim.v_max_V, params, ocv_fn, i_cutoff_A)
        t_list.append(t_curr)
        y_list.append(y.copy())
        i_list.append(i_now)
        ph_list.append(ph_now)

        if ph_now == "CV" and i_now <= i_cutoff_A + 1e-9:
            break

    runtime_s = float(time.perf_counter() - t_wall0)
    t_arr = np.asarray(t_list, dtype=np.float64)
    states = np.vstack(y_list)
    i_arr = np.asarray(i_list, dtype=np.float64)
    ph_arr = np.asarray(ph_list, dtype=object)

    soc = states[:, 0]
    vp_v = states[:, 1]
    temp_c = states[:, 2]
    aging_d = states[:, 3]
    e_loss_j = states[:, 4]
    vt_v = np.asarray(terminal_voltage(soc, vp_v, i_arr, params, ocv_fn), dtype=np.float64)
    p_loss_w = np.asarray(instantaneous_power_loss_W(vp_v, i_arr, params), dtype=np.float64)

    cc_mask = ph_arr == "CC"
    cv_mask = ph_arr == "CV"
    cc_dur = float(t_arr[cc_mask][-1] - t_arr[cc_mask][0]) if np.any(cc_mask) else 0.0
    cv_dur = float(t_arr[-1] - t_arr[cc_mask][-1]) if np.any(cc_mask) and np.any(cv_mask) else (float(t_arr[-1]) if np.any(cv_mask) else 0.0)

    return {
        "time_s": t_arr,
        "current_A": i_arr,
        "soc": soc,
        "vp_V": vp_v,
        "temperature_C": temp_c,
        "aging_D": aging_d,
        "terminal_voltage_V": vt_v,
        "power_loss_W": p_loss_w,
        "energy_loss_J": e_loss_j,
        "energy_loss_Wh": e_loss_j / 3600.0,
        "phase": ph_arr,
        "cc_duration_s": cc_dur,
        "cv_duration_s": cv_dur,
        "final_time_s": float(t_arr[-1]),
        "final_soc": float(soc[-1]),
        "max_voltage_V": float(np.max(vt_v)),
        "max_temperature_C": float(np.max(temp_c)),
        "total_energy_loss_J": float(e_loss_j[-1]),
        "total_aging_D": float(aging_d[-1] - aging_d[0]),
        "i_cc_A": float(i_cc_A),
        "runtime_s": runtime_s,
    }


def run_phase6_cc_cv_benchmark(
    processed_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    c_rates: tuple[float, ...] = (0.5, 0.7, 1.0, 1.5, 2.0),
) -> dict[str, Any]:
    """Farklı C-oranlarında CC-CV simülasyonları yapar, 1C referans ölçeklerini ve grafikleri kaydeder."""
    proc_path = Path(processed_dir)
    tbl_path = Path(tables_dir)
    fig_path = Path(figures_dir)
    tbl_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    json_path = proc_path / "calibrated_parameters.json"
    params = BatteryParameters.from_calibrated_json(json_path)
    ocv_fn = OCVInterpolator.from_calibrated_json(json_path)
    limits = BatteryLimits()

    rows: list[dict[str, Any]] = []
    sims: dict[float, dict[str, Any]] = {}
    for cr in c_rates:
        i_cc = cr * params.Q_Ah
        sim = simulate_cc_cv(params, ocv_fn, limits=limits, i_cc_A=i_cc, soc_init=0.20, soc_target=0.90, dt_s=1.0)
        sims[cr] = sim
        rows.append(
            {
                "c_rate": cr,
                "i_cc_A": i_cc,
                "cc_duration_min": sim["cc_duration_s"] / 60.0,
                "cv_duration_min": sim["cv_duration_s"] / 60.0,
                "final_time_s": sim["final_time_s"],
                "final_time_min": sim["final_time_s"] / 60.0,
                "energy_loss_J": sim["total_energy_loss_J"],
                "energy_loss_Wh": sim["total_energy_loss_J"] / 3600.0,
                "aging_D": sim["total_aging_D"],
                "max_voltage_V": sim["max_voltage_V"],
                "max_temperature_C": sim["max_temperature_C"],
                "final_soc": sim["final_soc"],
            }
        )

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(tbl_path / "phase6_cc_cv_comparison.csv", index=False)

    # 1.0C CC-CV baz stratejisinden normalizasyon referans değerleri
    ref_sim = sims[1.0]
    ref_scales = {
        "baseline_strategy": "CC-CV 1.0C (SOC 0.20 -> 0.90)",
        "t_ref_s": ref_sim["final_time_s"],
        "E_ref_J": ref_sim["total_energy_loss_J"],
        "D_ref": ref_sim["total_aging_D"],
        "i_cc_ref_A": ref_sim["i_cc_A"],
    }
    with open(proc_path / "reference_scales.json", "w", encoding="utf-8") as f:
        json.dump(ref_scales, f, indent=2)

    # 2x2 Grafik
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    palette = {0.5: "#1f77b4", 0.7: "#17becf", 1.0: "#2ca02c", 1.5: "#ff7f0e", 2.0: "#d62728"}

    for cr in c_rates:
        sim = sims[cr]
        t_m = sim["time_s"] / 60.0
        lbl = f"{cr:.1f}C ({sim['i_cc_A']:.2f} A)" + (" [Referans]" if cr == 1.0 else "")
        col = palette.get(cr, "k")
        axes[0, 0].plot(t_m, sim["current_A"], linewidth=1.7, color=col, label=lbl)
        axes[0, 1].plot(t_m, sim["terminal_voltage_V"], linewidth=1.7, color=col, label=lbl)
        axes[1, 0].plot(t_m, sim["temperature_C"], linewidth=1.7, color=col, label=lbl)
        axes[1, 1].plot(t_m, sim["aging_D"], linewidth=1.7, color=col, label=lbl)

    axes[0, 0].set_title("Şarj Akımı Profili")
    axes[0, 0].set_ylabel("Akım [A]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Terminal Gerilimi (V_max = 4.20 V)")
    axes[0, 1].set_ylabel("Gerilim [V]")
    axes[0, 1].axhline(limits.v_max_V, color="gray", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Hücre Sıcaklığı")
    axes[1, 0].set_ylabel("Sıcaklık [°C]")
    axes[1, 0].set_xlabel("Süre [dk]")
    axes[1, 0].axhline(limits.t_max_C, color="gray", linestyle="--", linewidth=1.0, label="T_max = 45 °C")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_title("Göreli Yaşlanma Göstergesi D(t)")
    axes[1, 1].set_ylabel("Göreli Yaşlanma D [-]")
    axes[1, 1].set_xlabel("Süre [dk]")

    fig.suptitle("Geleneksel CC-CV Referans Şarj Stratejisi Karşılaştırması (SOC: 0.20 → 0.90)", fontsize=13)
    fig.tight_layout()
    fig_out = fig_path / "fig06_cc_cv_reference.png"
    fig.savefig(fig_out, dpi=300)
    plt.close(fig)

    return {
        "comparison_table": comp_df,
        "reference_scales": ref_scales,
        "simulations": sims,
        "figure_path": str(fig_out),
    }
