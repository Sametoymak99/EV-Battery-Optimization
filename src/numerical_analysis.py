"""Aşama 5: Sayısal ODE çözücüleri (Euler, RK4, solve_ivp) ve sonlu fark gradyan analizi."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.battery_model import (
    BatteryParameters,
    OCVInterpolator,
    euler_stability_limit,
    instantaneous_power_loss_W,
    ode_rhs,
    simulate_current_profile,
    simulate_soc_profile,
    terminal_voltage,
)


def compute_reference_solution_dop853(
    t_eval: np.ndarray,
    i_fn: Callable[[float], float],
    params: BatteryParameters,
    x0: np.ndarray,
) -> np.ndarray:
    """Yüksek doğruluklu DOP853 (rtol=1e-12, atol=1e-14) referans ODE çözümünü hesaplar."""
    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, float(i_fn(float(t))), params),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(x0, dtype=np.float64),
        t_eval=np.asarray(t_eval, dtype=np.float64),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    if not sol.success:
        raise RuntimeError(f"DOP853 referans çözümü başarısız: {sol.message}")
    return sol.y.T


def benchmark_ode_solvers(
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    dt_values: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    t_end: float = 1800.0,
    n_repeats: int = 5,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    """Euler, RK4 ve solve_ivp çözücülerini DOP853 referansına göre doğruluk ve süre açısından kıyaslar."""
    i_fn = lambda t: 6.5 * np.exp(-t / 900.0) + 1.5
    x0 = np.array([0.20, 0.0, params.T_amb_C, 0.0, 0.0], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for dt in dt_values:
        t_eval = np.arange(0.0, t_end + 1e-9, float(dt))
        i_arr = np.array([float(i_fn(float(t))) for t in t_eval], dtype=np.float64)
        ref = compute_reference_solution_dop853(t_eval, i_fn, params, x0)
        vt_ref = np.asarray(terminal_voltage(ref[:, 0], ref[:, 1], i_arr, params, ocv_fn), dtype=np.float64)

        for method in ("euler", "rk4", "solve_ivp"):
            runtimes: list[float] = []
            sim: dict[str, Any] | None = None
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                sim = simulate_current_profile(t_eval, i_fn, params, ocv_fn, x0=x0, method=method)
                runtimes.append((time.perf_counter() - t0) * 1000.0)
            assert sim is not None

            err_vp = float(np.max(np.abs(sim["vp_V"] - ref[:, 1])))
            err_vt = float(np.max(np.abs(sim["terminal_voltage_V"] - vt_ref)))
            err_temp = float(np.max(np.abs(sim["temperature_C"] - ref[:, 2])))
            err_eloss_rel = float(abs(sim["total_energy_loss_J"] - ref[-1, 4]) / max(abs(ref[-1, 4]), 1e-12))
            err_aging_rel = float(abs(sim["total_aging_D"] - ref[-1, 3]) / max(abs(ref[-1, 3]), 1e-12))

            rows.append(
                {
                    "dt_s": float(dt),
                    "n_steps": int(len(t_eval) - 1),
                    "method": method,
                    "err_vp_max_V": err_vp,
                    "err_vt_max_V": err_vt,
                    "err_temp_max_C": err_temp,
                    "err_eloss_rel": err_eloss_rel,
                    "err_aging_rel": err_aging_rel,
                    "runtime_ms_mean": float(np.mean(runtimes)),
                    "runtime_ms_std": float(np.std(runtimes)),
                }
            )

    df = pd.DataFrame(rows)

    # Ampirik yakınsama mertebesi (log-log eğim, dt in [1.0, 20.0])
    orders: dict[str, float] = {}
    for method in ("euler", "rk4"):
        sub = df[(df["method"] == method) & (df["dt_s"] >= 1.0) & (df["dt_s"] <= 20.0)]
        slope, _ = np.polyfit(np.log(sub["dt_s"].to_numpy()), np.log(np.maximum(sub["err_vp_max_V"].to_numpy(), 1e-16)), 1)
        orders[f"{method}_order_vp"] = float(slope)
        slope_t, _ = np.polyfit(np.log(sub["dt_s"].to_numpy()), np.log(np.maximum(sub["err_temp_max_C"].to_numpy(), 1e-16)), 1)
        orders[f"{method}_order_temp"] = float(slope_t)

    # Kararlılık sınırı üstünde (dt = 130 s > 2*R1*C1 ≈ 120.3 s) Euler vs RK4 testi
    stab_lim = euler_stability_limit(params)
    dt_instab = 130.0
    t_inst = np.arange(0.0, 1300.0 + 1e-9, dt_instab)
    ref_inst = compute_reference_solution_dop853(t_inst, i_fn, params, x0)
    sim_eu_inst = simulate_current_profile(t_inst, i_fn, params, ocv_fn, x0=x0, method="euler")
    sim_rk_inst = simulate_current_profile(t_inst, i_fn, params, ocv_fn, x0=x0, method="rk4")

    instability = {
        "euler_stability_limit_s": float(stab_lim),
        "test_dt_s": float(dt_instab),
        "time_s": t_inst,
        "vp_ref_V": ref_inst[:, 1],
        "vp_euler_V": sim_eu_inst["vp_V"],
        "vp_rk4_V": sim_rk_inst["vp_V"],
        "err_vp_euler_max_V": float(np.max(np.abs(sim_eu_inst["vp_V"] - ref_inst[:, 1]))),
        "err_vp_rk4_max_V": float(np.max(np.abs(sim_rk_inst["vp_V"] - ref_inst[:, 1]))),
    }
    return df, orders, instability


def benchmark_soc_profile_dt_sensitivity(
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    dt_max_values: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0),
) -> pd.DataFrame:
    """7 aşamalı şarj profilinde dt_max_s duyarlılığını Euler, RK4 ve solve_ivp için ölçer."""
    currents = np.array([7.0, 6.5, 6.0, 5.0, 4.0, 3.0, 2.0], dtype=np.float64)
    ref = simulate_soc_profile(currents, params, ocv_fn, method="solve_ivp", dt_max_s=0.25)

    rows: list[dict[str, Any]] = []
    for dt_max in dt_max_values:
        for method in ("euler", "rk4", "solve_ivp"):
            sim = simulate_soc_profile(currents, params, ocv_fn, method=method, dt_max_s=float(dt_max))
            rows.append(
                {
                    "dt_max_s": float(dt_max),
                    "method": method,
                    "final_time_s": sim["final_time_s"],
                    "max_voltage_V": sim["max_voltage_V"],
                    "max_temp_C": sim["max_temperature_C"],
                    "energy_loss_J": sim["total_energy_loss_J"],
                    "aging_D": sim["total_aging_D"],
                    "err_vmax_mV": abs(sim["max_voltage_V"] - ref["max_voltage_V"]) * 1000.0,
                    "err_tmax_mC": abs(sim["max_temperature_C"] - ref["max_temperature_C"]) * 1000.0,
                    "err_eloss_rel": abs(sim["total_energy_loss_J"] - ref["total_energy_loss_J"]) / ref["total_energy_loss_J"],
                    "runtime_ms": sim["runtime_s"] * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def benchmark_finite_difference_gradients(
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    h_values: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
) -> pd.DataFrame:
    """İleri ve merkezi sonlu fark gradyanlarının adım boyu (h) duyarlılığını analiz eder."""
    u0 = np.array([7.0, 6.5, 6.0, 5.0, 4.0, 3.0, 2.0], dtype=np.float64)
    n_stages = len(u0)
    dsoc = (0.90 - 0.20) / n_stages

    # Analitik gradyan: d(t_f)/dI_k = -3600 * Q * dSOC / (eta * I_k^2)
    grad_tf_exact = -(3600.0 * params.Q_Ah * dsoc) / (params.eta * (u0 ** 2))

    def _eval_tf_and_eloss(u_vec: np.ndarray) -> tuple[float, float]:
        sim = simulate_soc_profile(u_vec, params, ocv_fn, method="rk4", dt_max_s=1.0)
        return sim["final_time_s"], sim["total_energy_loss_J"]

    tf0, eloss0 = _eval_tf_and_eloss(u0)

    # Enerji kaybı için yüksek hassasiyetli merkezi referans (h = 1e-5)
    h_ref = 1e-5
    grad_eloss_ref = np.zeros(n_stages, dtype=np.float64)
    for k in range(n_stages):
        e_k = np.zeros(n_stages)
        e_k[k] = h_ref
        _, ep = _eval_tf_and_eloss(u0 + e_k)
        _, em = _eval_tf_and_eloss(u0 - e_k)
        grad_eloss_ref[k] = (ep - em) / (2.0 * h_ref)

    rows: list[dict[str, Any]] = []
    for h in h_values:
        g_fwd_tf = np.zeros(n_stages)
        g_cen_tf = np.zeros(n_stages)
        g_fwd_el = np.zeros(n_stages)
        g_cen_el = np.zeros(n_stages)

        for k in range(n_stages):
            e_k = np.zeros(n_stages)
            e_k[k] = float(h)
            tf_p, el_p = _eval_tf_and_eloss(u0 + e_k)
            tf_m, el_m = _eval_tf_and_eloss(u0 - e_k)
            g_fwd_tf[k] = (tf_p - tf0) / h
            g_cen_tf[k] = (tf_p - tf_m) / (2.0 * h)
            g_fwd_el[k] = (el_p - eloss0) / h
            g_cen_el[k] = (el_p - el_m) / (2.0 * h)

        rows.append(
            {
                "h_A": float(h),
                "forward_rel_err_tf": float(np.linalg.norm(g_fwd_tf - grad_tf_exact) / np.linalg.norm(grad_tf_exact)),
                "central_rel_err_tf": float(np.linalg.norm(g_cen_tf - grad_tf_exact) / np.linalg.norm(grad_tf_exact)),
                "forward_rel_err_eloss": float(np.linalg.norm(g_fwd_el - grad_eloss_ref) / np.linalg.norm(grad_eloss_ref)),
                "central_rel_err_eloss": float(np.linalg.norm(g_cen_el - grad_eloss_ref) / np.linalg.norm(grad_eloss_ref)),
            }
        )
    return pd.DataFrame(rows)


def _generate_phase5_figures(
    ode_df: pd.DataFrame,
    orders: dict[str, float],
    instability: dict[str, Any],
    fd_df: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    """Aşama 5 ODE yakınsama/kararlılık ve sonlu fark duyarlılık grafiklerini üretir."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    saved: list[str] = []

    # 1. ODE Yakınsama, Hata-Süre ve Kararlılık
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"euler": "#d62728", "rk4": "#1f77b4", "solve_ivp": "#2ca02c"}
    labels = {
        "euler": f"İleri Euler (eğim ≈ {orders['euler_order_vp']:.2f})",
        "rk4": f"RK4 (eğim ≈ {orders['rk4_order_vp']:.2f})",
        "solve_ivp": "SciPy solve_ivp (RK45)",
    }
    for method in ("euler", "rk4", "solve_ivp"):
        sub = ode_df[ode_df["method"] == method]
        axes[0].loglog(sub["dt_s"], sub["err_vp_max_V"], marker="o", linewidth=1.7, color=colors[method], label=labels[method])
        axes[1].loglog(sub["runtime_ms_mean"], sub["err_vp_max_V"], marker="s", linewidth=1.7, color=colors[method], label=method.upper())

    dt_ref = np.array([0.5, 20.0])
    axes[0].loglog(dt_ref, 5e-5 * dt_ref, "k--", alpha=0.5, label="O(h) referans")
    axes[0].loglog(dt_ref, 1e-11 * (dt_ref ** 4), "k:", alpha=0.6, label="O(h^4) referans")
    axes[0].set_title("Adım Boyuna Göre Vp Yakınsaması")
    axes[0].set_xlabel("Zaman Adımı Δt [s]")
    axes[0].set_ylabel("Maks. Vp Hatası [V]")
    axes[0].legend(fontsize=7.5)

    axes[1].set_title("Doğruluk – Hesaplama Süresi Ödünleşimi")
    axes[1].set_xlabel("Ortalama Çalışma Süresi [ms]")
    axes[1].set_ylabel("Maks. Vp Hatası [V]")
    axes[1].legend(fontsize=8)

    t_in = instability["time_s"] / 60.0
    axes[2].plot(t_in, instability["vp_ref_V"], "k-", linewidth=2.0, label="DOP853 Referans")
    axes[2].plot(t_in, instability["vp_rk4_V"], color="#1f77b4", linestyle="--", marker="o", markersize=4, label="RK4 (Kararlı)")
    axes[2].plot(t_in, instability["vp_euler_V"], color="#d62728", marker="x", markersize=5, label=f"Euler (dt={instability['test_dt_s']:.0f}s > {instability['euler_stability_limit_s']:.0f}s)")
    axes[2].set_title("Kararlılık Sınırı Üstü (Δt = 130 s)")
    axes[2].set_xlabel("Süre [dk]")
    axes[2].set_ylabel("Polarizasyon Gerilimi Vp [V]")
    axes[2].legend(fontsize=7.5)

    fig.tight_layout()
    p1 = figures_dir / "fig05_ode_convergence_and_runtime.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved.append(str(p1))

    # 2. Sonlu Fark Gradyan Adım Boyu Duyarlılığı
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].loglog(fd_df["h_A"], fd_df["forward_rel_err_tf"], "o-", color="#ff7f0e", label="İleri Fark O(h)")
    axes[0].loglog(fd_df["h_A"], np.maximum(fd_df["central_rel_err_tf"], 1e-16), "s-", color="#1f77b4", label="Merkezi Fark O(h²)")
    axes[0].set_title("Şarj Süresi Gradyanı (∇t_f) Bağıl Hatası")
    axes[0].set_xlabel("Sonlu Fark Adımı h [A]")
    axes[0].set_ylabel("Bağıl L2 Gradyan Hatası [-]")
    axes[0].legend(fontsize=8.5)

    axes[1].loglog(fd_df["h_A"], np.maximum(fd_df["forward_rel_err_eloss"], 1e-16), "o-", color="#ff7f0e", label="İleri Fark O(h)")
    axes[1].loglog(fd_df["h_A"], np.maximum(fd_df["central_rel_err_eloss"], 1e-16), "s-", color="#1f77b4", label="Merkezi Fark O(h²)")
    axes[1].set_title("Enerji Kaybı Gradyanı (∇E_loss) Duyarlılığı")
    axes[1].set_xlabel("Sonlu Fark Adımı h [A]")
    axes[1].set_ylabel("Referansa (h=1e-5) Göre Bağıl Fark [-]")
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    p2 = figures_dir / "fig05_fd_step_sensitivity.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved.append(str(p2))

    return saved


def run_phase5_pipeline(
    processed_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, Any]:
    """Aşama 5 ODE çözücü ve sonlu fark gradyan kıyaslama hattını çalıştırır."""
    proc_path = Path(processed_dir)
    tbl_path = Path(tables_dir)
    fig_path = Path(figures_dir)
    tbl_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    json_path = proc_path / "calibrated_parameters.json"
    params = BatteryParameters.from_calibrated_json(json_path)
    ocv_fn = OCVInterpolator.from_calibrated_json(json_path)

    ode_df, orders, instability = benchmark_ode_solvers(params, ocv_fn)
    ode_df.to_csv(tbl_path / "phase5_ode_solver_comparison.csv", index=False)

    soc_dt_df = benchmark_soc_profile_dt_sensitivity(params, ocv_fn)
    soc_dt_df.to_csv(tbl_path / "phase5_soc_dt_sensitivity.csv", index=False)

    fd_df = benchmark_finite_difference_gradients(params, ocv_fn)
    fd_df.to_csv(tbl_path / "phase5_fd_gradient_sensitivity.csv", index=False)

    saved_figs = _generate_phase5_figures(ode_df, orders, instability, fd_df, fig_path)

    return {
        "ode_benchmark": ode_df,
        "convergence_orders": orders,
        "instability": instability,
        "soc_dt_sensitivity": soc_dt_df,
        "fd_sensitivity": fd_df,
        "saved_figures": saved_figs,
    }
