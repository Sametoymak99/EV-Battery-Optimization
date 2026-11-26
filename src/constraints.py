"""Optimizasyon kısıt denetimleri, projeksiyon ve dış ceza fonksiyonları."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.battery_model import BatteryLimits


def project_currents(
    u: np.ndarray,
    i_min_A: float = 0.50,
    i_max_A: float = 9.72,
) -> np.ndarray:
    """Akım karar vektörünü [i_min_A, i_max_A] kutu sınırlarına projekte eder."""
    return np.clip(np.asarray(u, dtype=np.float64), float(i_min_A), float(i_max_A))


def evaluate_constraints(
    sim_result: dict[str, Any],
    limits: BatteryLimits,
    soc_target: float = 0.90,
    i_min_A: float = 0.50,
    i_max_A: float | None = None,
    tol_v: float = 2e-3,
    tol_t: float = 5e-2,
    tol_soc: float = 1e-4,
) -> dict[str, Any]:
    """Simülasyon yörüngesi boyunca tüm gerilim, sıcaklık, SOC ve akım kısıtlarını denetler."""
    upper_i = limits.i_max_A if i_max_A is None else float(i_max_A)
    vt = np.asarray(sim_result["terminal_voltage_V"], dtype=np.float64)
    temp = np.asarray(sim_result["temperature_C"], dtype=np.float64)
    soc = np.asarray(sim_result["soc"], dtype=np.float64)
    curr = np.asarray(sim_result["current_A"], dtype=np.float64)

    v_excess = float(np.max(vt) - limits.v_max_V)
    t_excess = float(np.max(temp) - limits.t_max_C)
    soc_min_viol = float(limits.soc_min - np.min(soc))
    soc_max_viol = float(np.max(soc) - limits.soc_max)
    soc_target_err = float(abs(soc[-1] - soc_target))
    i_low_viol = float(i_min_A - np.min(curr))
    i_high_viol = float(np.max(curr) - upper_i)

    is_feasible = bool(
        v_excess <= tol_v
        and t_excess <= tol_t
        and soc_min_viol <= tol_soc
        and soc_max_viol <= tol_soc
        and soc_target_err <= tol_soc
        and i_low_viol <= 1e-6
        and i_high_viol <= 1e-6
    )

    return {
        "is_feasible": is_feasible,
        "max_voltage_V": float(np.max(vt)),
        "voltage_excess_V": max(0.0, v_excess),
        "max_temperature_C": float(np.max(temp)),
        "temperature_excess_C": max(0.0, t_excess),
        "soc_min": float(np.min(soc)),
        "soc_max": float(np.max(soc)),
        "soc_target_err": soc_target_err,
        "current_min_A": float(np.min(curr)),
        "current_max_A": float(np.max(curr)),
    }


def compute_stage_margins(
    sim_result: dict[str, Any],
    limits: BatteryLimits,
    n_stages: int,
    soc_init: float = 0.20,
    soc_target: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """SLSQP için aşama bazlı düzgün (V_max - max Vt_k) ve (T_max - max T_k) kısıt vektörlerini hesaplar."""
    soc = np.asarray(sim_result["soc"], dtype=np.float64)
    vt = np.asarray(sim_result["terminal_voltage_V"], dtype=np.float64)
    temp = np.asarray(sim_result["temperature_C"], dtype=np.float64)

    edges = np.linspace(soc_init, soc_target, n_stages + 1)
    v_margins = np.zeros(n_stages, dtype=np.float64)
    t_margins = np.zeros(n_stages, dtype=np.float64)

    for k in range(n_stages):
        if k < n_stages - 1:
            mask = (soc >= edges[k] - 1e-9) & (soc <= edges[k + 1] + 1e-9)
        else:
            mask = (soc >= edges[k] - 1e-9) & (soc <= edges[k + 1] + 1e-6)
        v_margins[k] = limits.v_max_V - float(np.max(vt[mask]))
        t_margins[k] = limits.t_max_C - float(np.max(temp[mask]))

    return v_margins, t_margins


def compute_exterior_penalty(
    sim_result: dict[str, Any],
    limits: BatteryLimits,
    rho_v: float = 2000.0,
    rho_t: float = 50.0,
) -> float:
    """GD ve Adam için dış karesel ceza terimi: tepe + ortalama aşım cezası."""
    vt = np.asarray(sim_result["terminal_voltage_V"], dtype=np.float64)
    temp = np.asarray(sim_result["temperature_C"], dtype=np.float64)

    dv = np.maximum(0.0, vt - limits.v_max_V)
    dt = np.maximum(0.0, temp - limits.t_max_C)

    v_pen = rho_v * (float(np.max(dv) ** 2) + float(np.mean(dv ** 2)))
    t_pen = rho_t * (float(np.max(dt) ** 2) + float(np.mean(dt ** 2)))
    return float(v_pen + t_pen)
