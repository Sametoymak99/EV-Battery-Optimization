"""Aşama 7–9: Gradient Descent, Adam ve SciPy SLSQP şarj optimizasyonu algoritmaları."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.battery_model import (
    BatteryLimits,
    BatteryParameters,
    OCVInterpolator,
    simulate_soc_profile,
)
from src.cc_cv import simulate_cc_cv
from src.constraints import (
    compute_exterior_penalty,
    compute_stage_margins,
    evaluate_constraints,
    project_currents,
)
from src.objective_functions import ReferenceScales, evaluate_objectives


def finite_difference_gradient(
    func: Callable[[np.ndarray], float],
    u: np.ndarray,
    h: float = 1e-4,
    scheme: str = "forward",
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, int]:
    """İleri veya merkezi sonlu farklarla sayısal gradyan ve fonksiyon çağrı sayısını hesaplar."""
    n = len(u)
    grad = np.zeros(n, dtype=np.float64)
    nfev = 0

    if scheme == "forward":
        f0 = float(func(u))
        nfev += 1
        for k in range(n):
            step = float(h)
            if bounds is not None and u[k] + step > bounds[1]:
                step = -float(h)
            u_pert = u.copy()
            u_pert[k] += step
            fp = float(func(u_pert))
            nfev += 1
            grad[k] = (fp - f0) / step
    elif scheme == "central":
        for k in range(n):
            u_p = u.copy()
            u_m = u.copy()
            u_p[k] += h
            u_m[k] -= h
            grad[k] = (float(func(u_p)) - float(func(u_m))) / (2.0 * h)
            nfev += 2
    else:
        raise ValueError(f"Desteklenmeyen gradyan şeması: {scheme}")

    return grad, nfev


def make_initial_guess(
    n_stages: int = 10,
    seed: int | None = None,
    i_min_A: float = 0.50,
    i_max_A: float = 9.72,
) -> np.ndarray:
    """Deterministik veya sabit seed ile bozulmuş başlangıç akım profili üretir."""
    base = np.linspace(5.2, 2.8, n_stages, dtype=np.float64)
    if seed is None:
        return project_currents(base, i_min_A, i_max_A)
    rng = np.random.default_rng(int(seed))
    pert = rng.uniform(-0.9, 0.9, size=n_stages)
    return project_currents(base + pert, i_min_A, i_max_A)


def _finalize_result(
    name: str,
    u_opt: np.ndarray,
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    limits: BatteryLimits,
    ref_scales: ReferenceScales,
    weights: tuple[float, float, float],
    history: list[dict[str, Any]],
    n_iters: int,
    nfev: int,
    runtime_s: float,
    seed: int | None,
    i_min_A: float,
    i_max_A: float,
) -> dict[str, Any]:
    """Optimal akım profilini yüksek hassasiyetle (dt_max_s=1.0 s) yeniden simüle edip doğrular."""
    sim_hr = simulate_soc_profile(u_opt, params, ocv_fn, method="rk4", dt_max_s=1.0)
    objs = evaluate_objectives(sim_hr, ref_scales, weights=weights)
    cons = evaluate_constraints(sim_hr, limits, i_min_A=i_min_A, i_max_A=i_max_A)
    return {
        "algorithm": name,
        "u_opt_A": np.asarray(u_opt, dtype=np.float64),
        "objectives": objs,
        "constraints": cons,
        "simulation": sim_hr,
        "history": pd.DataFrame(history),
        "iterations": int(n_iters),
        "nfev": int(nfev),
        "runtime_s": float(runtime_s),
        "seed": seed,
    }


def optimize_gradient_descent(
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    limits: BatteryLimits,
    ref_scales: ReferenceScales,
    weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
    u0: np.ndarray | None = None,
    n_stages: int = 10,
    i_min_A: float = 0.50,
    i_max_A: float | None = None,
    max_iters_per_stage: int = 50,
    alpha0: float = 2.0,
    tol_step: float = 1e-4,
    dt_opt_s: float = 10.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Aşama 7: Projeksiyonlu ve 2-kademeli cezalı Gradient Descent optimizasyonu."""
    upper_i = limits.i_max_A if i_max_A is None else float(i_max_A)
    u = make_initial_guess(n_stages, seed=seed, i_min_A=i_min_A, i_max_A=upper_i) if u0 is None else project_currents(u0, i_min_A, upper_i)

    history: list[dict[str, Any]] = []
    nfev = 0
    best_feasible_u: np.ndarray | None = None
    best_feasible_j = float("inf")
    t_start = time.perf_counter()

    penalty_stages = [(500.0, 10.0), (5000.0, 100.0)]
    global_iter = 0

    for stage_idx, (rho_v, rho_t) in enumerate(penalty_stages, start=1):
        def _pen_obj(u_vec: np.ndarray) -> float:
            u_p = project_currents(u_vec, i_min_A, upper_i)
            sim = simulate_soc_profile(u_p, params, ocv_fn, method="rk4", dt_max_s=dt_opt_s)
            obj = evaluate_objectives(sim, ref_scales, weights=weights)["J_total"]
            pen = compute_exterior_penalty(sim, limits, rho_v=rho_v, rho_t=rho_t)
            return obj + pen

        for _ in range(max_iters_per_stage):
            sim_curr = simulate_soc_profile(u, params, ocv_fn, method="rk4", dt_max_s=dt_opt_s)
            nfev += 1
            objs = evaluate_objectives(sim_curr, ref_scales, weights=weights)
            cons = evaluate_constraints(sim_curr, limits, i_min_A=i_min_A, i_max_A=upper_i)
            pen_curr = compute_exterior_penalty(sim_curr, limits, rho_v=rho_v, rho_t=rho_t)
            f_curr = objs["J_total"] + pen_curr

            if cons["is_feasible"] and objs["J_total"] < best_feasible_j:
                best_feasible_j = objs["J_total"]
                best_feasible_u = u.copy()

            grad, g_fev = finite_difference_gradient(_pen_obj, u, h=1e-4, scheme="forward", bounds=(i_min_A, upper_i))
            nfev += g_fev

            # Geri izlemeli adım arama (Armijo benzeri azalma)
            alpha = alpha0
            u_next = u.copy()
            accepted = False
            for _ls in range(12):
                cand = project_currents(u - alpha * grad, i_min_A, upper_i)
                f_cand = _pen_obj(cand)
                nfev += 1
                if f_cand <= f_curr - 1e-5 * float(np.sum((cand - u) ** 2)):
                    u_next = cand
                    accepted = True
                    break
                alpha *= 0.5

            step_norm = float(np.linalg.norm(u_next - u))
            history.append(
                {
                    "iter": global_iter,
                    "stage": stage_idx,
                    "J_total": objs["J_total"],
                    "J_penalized": f_curr,
                    "J1": objs["J1"],
                    "J2": objs["J2"],
                    "J3": objs["J3"],
                    "max_voltage_V": cons["max_voltage_V"],
                    "max_temperature_C": cons["max_temperature_C"],
                    "feasible": cons["is_feasible"],
                    "grad_norm": float(np.linalg.norm(grad)),
                    "step_norm": step_norm,
                }
            )
            global_iter += 1
            u = u_next
            if not accepted or step_norm < tol_step:
                break

    sim_end = simulate_soc_profile(u, params, ocv_fn, method="rk4", dt_max_s=dt_opt_s)
    cons_end = evaluate_constraints(sim_end, limits, i_min_A=i_min_A, i_max_A=upper_i)
    objs_end = evaluate_objectives(sim_end, ref_scales, weights=weights)
    if cons_end["is_feasible"] and objs_end["J_total"] <= best_feasible_j:
        u_final = u
    elif best_feasible_u is not None:
        u_final = best_feasible_u
    else:
        u_final = u

    runtime_s = float(time.perf_counter() - t_start)
    return _finalize_result(
        "Gradient Descent",
        u_final,
        params,
        ocv_fn,
        limits,
        ref_scales,
        weights,
        history,
        global_iter,
        nfev,
        runtime_s,
        seed,
        i_min_A,
        upper_i,
    )


def optimize_adam(
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    limits: BatteryLimits,
    ref_scales: ReferenceScales,
    weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
    u0: np.ndarray | None = None,
    n_stages: int = 10,
    i_min_A: float = 0.50,
    i_max_A: float | None = None,
    max_iters_per_stage: int = 45,
    alpha: float = 0.22,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    dt_opt_s: float = 10.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Aşama 8: Projeksiyonlu ve 2-kademeli cezalı Adam optimizasyonu."""
    upper_i = limits.i_max_A if i_max_A is None else float(i_max_A)
    u = make_initial_guess(n_stages, seed=seed, i_min_A=i_min_A, i_max_A=upper_i) if u0 is None else project_currents(u0, i_min_A, upper_i)

    history: list[dict[str, Any]] = []
    nfev = 0
    best_feasible_u: np.ndarray | None = None
    best_feasible_j = float("inf")
    t_start = time.perf_counter()

    penalty_stages = [(500.0, 10.0), (5000.0, 100.0)]
    global_iter = 0

    for stage_idx, (rho_v, rho_t) in enumerate(penalty_stages, start=1):
        m = np.zeros(len(u), dtype=np.float64)
        v = np.zeros(len(u), dtype=np.float64)
        stage_alpha = alpha if stage_idx == 1 else 0.25 * alpha

        def _pen_obj(u_vec: np.ndarray) -> float:
            u_p = project_currents(u_vec, i_min_A, upper_i)
            sim = simulate_soc_profile(u_p, params, ocv_fn, method="rk4", dt_max_s=dt_opt_s)
            obj = evaluate_objectives(sim, ref_scales, weights=weights)["J_total"]
            pen = compute_exterior_penalty(sim, limits, rho_v=rho_v, rho_t=rho_t)
            return obj + pen

        for step_k in range(1, max_iters_per_stage + 1):
            sim_curr = simulate_soc_profile(u, params, ocv_fn, method="rk4", dt_max_s=dt_opt_s)
            nfev += 1
            objs = evaluate_objectives(sim_curr, ref_scales, weights=weights)
            cons = evaluate_constraints(sim_curr, limits, i_min_A=i_min_A, i_max_A=upper_i)
            f_curr = objs["J_total"] + compute_exterior_penalty(sim_curr, limits, rho_v=rho_v, rho_t=rho_t)

            if cons["is_feasible"] and objs["J_total"] < best_feasible_j:
                best_feasible_j = objs["J_total"]
                best_feasible_u = u.copy()

            grad, g_fev = finite_difference_gradient(_pen_obj, u, h=1e-4, scheme="forward", bounds=(i_min_A, upper_i))
            nfev += g_fev

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - (beta1 ** step_k))
            v_hat = v / (1.0 - (beta2 ** step_k))

            # Yumuşak öğrenme oranı azalımı
            lr_k = stage_alpha * (0.5 * (1.0 + np.cos(np.pi * (step_k - 1) / max_iters_per_stage)))
            lr_k = max(lr_k, 0.02 * stage_alpha)
            u_next = project_currents(u - lr_k * m_hat / (np.sqrt(v_hat) + eps), i_min_A, upper_i)
            step_norm = float(np.linalg.norm(u_next - u))

            history.append(
                {
                    "iter": global_iter,
                    "stage": stage_idx,
                    "J_total": objs["J_total"],
                    "J_penalized": f_curr,
                    "J1": objs["J1"],
                    "J2": objs["J2"],
                    "J3": objs["J3"],
                    "max_voltage_V": cons["max_voltage_V"],
                    "max_temperature_C": cons["max_temperature_C"],
                    "feasible": cons["is_feasible"],
                    "grad_norm": float(np.linalg.norm(grad)),
                    "step_norm": step_norm,
                }
            )
            global_iter += 1
            u = u_next

    sim_end = simulate_soc_profile(u, params, ocv_fn, method="rk4", dt_max_s=dt_opt_s)
    cons_end = evaluate_constraints(sim_end, limits, i_min_A=i_min_A, i_max_A=upper_i)
    objs_end = evaluate_objectives(sim_end, ref_scales, weights=weights)
    if cons_end["is_feasible"] and objs_end["J_total"] <= best_feasible_j:
        u_final = u
    elif best_feasible_u is not None:
        u_final = best_feasible_u
    else:
        u_final = u

    runtime_s = float(time.perf_counter() - t_start)
    return _finalize_result(
        "Adam",
        u_final,
        params,
        ocv_fn,
        limits,
        ref_scales,
        weights,
        history,
        global_iter,
        nfev,
        runtime_s,
        seed,
        i_min_A,
        upper_i,
    )


def optimize_slsqp(
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    limits: BatteryLimits,
    ref_scales: ReferenceScales,
    weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
    u0: np.ndarray | None = None,
    n_stages: int = 10,
    i_min_A: float = 0.50,
    i_max_A: float | None = None,
    maxiter: int = 100,
    ftol: float = 1e-8,
    dt_opt_s: float = 10.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Aşama 9: SciPy SLSQP ile doğrudan kısıtlı doğrusal olmayan optimizasyon."""
    upper_i = limits.i_max_A if i_max_A is None else float(i_max_A)
    u_init = make_initial_guess(n_stages, seed=seed, i_min_A=i_min_A, i_max_A=upper_i) if u0 is None else project_currents(u0, i_min_A, upper_i)

    cache: dict[bytes, dict[str, Any]] = {}
    eval_counter = {"nfev": 0}
    history: list[dict[str, Any]] = []

    def _sim_cached(u_vec: np.ndarray) -> dict[str, Any]:
        u_p = project_currents(u_vec, i_min_A, upper_i)
        key = u_p.tobytes()
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = simulate_soc_profile(u_p, params, ocv_fn, method="rk4", dt_max_s=dt_opt_s)
            eval_counter["nfev"] += 1
        return cache[key]

    def _obj(u_vec: np.ndarray) -> float:
        return evaluate_objectives(_sim_cached(u_vec), ref_scales, weights=weights)["J_total"]

    def _v_ineq(u_vec: np.ndarray) -> np.ndarray:
        v_m, _ = compute_stage_margins(_sim_cached(u_vec), limits, n_stages=len(u_vec))
        return v_m

    def _t_ineq(u_vec: np.ndarray) -> np.ndarray:
        _, t_m = compute_stage_margins(_sim_cached(u_vec), limits, n_stages=len(u_vec))
        return t_m

    def _callback(xk: np.ndarray) -> None:
        sim = _sim_cached(xk)
        objs = evaluate_objectives(sim, ref_scales, weights=weights)
        cons = evaluate_constraints(sim, limits, i_min_A=i_min_A, i_max_A=upper_i)
        history.append(
            {
                "iter": len(history),
                "stage": 1,
                "J_total": objs["J_total"],
                "J_penalized": objs["J_total"],
                "J1": objs["J1"],
                "J2": objs["J2"],
                "J3": objs["J3"],
                "max_voltage_V": cons["max_voltage_V"],
                "max_temperature_C": cons["max_temperature_C"],
                "feasible": cons["is_feasible"],
                "grad_norm": np.nan,
                "step_norm": np.nan,
            }
        )

    t_start = time.perf_counter()
    # Başlangıç durumunu kaydet
    _callback(u_init)
    res = minimize(
        _obj,
        u_init,
        method="SLSQP",
        bounds=[(i_min_A, upper_i)] * len(u_init),
        constraints=[
            {"type": "ineq", "fun": _v_ineq},
            {"type": "ineq", "fun": _t_ineq},
        ],
        callback=_callback,
        options={"maxiter": maxiter, "ftol": ftol, "eps": 1e-4},
    )
    runtime_s = float(time.perf_counter() - t_start)
    u_opt = project_currents(res.x, i_min_A, upper_i)

    out = _finalize_result(
        "SLSQP",
        u_opt,
        params,
        ocv_fn,
        limits,
        ref_scales,
        weights,
        history,
        int(res.nit),
        eval_counter["nfev"],
        runtime_s,
        seed,
        i_min_A,
        upper_i,
    )
    out["scipy_success"] = bool(res.success)
    out["scipy_message"] = str(res.message)
    return out


def _plot_single_optimizer(res: dict[str, Any], limits: BatteryLimits, title: str, out_path: Path) -> None:
    """Tek bir optimizasyon algoritması için yakınsama ve yörünge grafiği üretir."""
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    hist = res["history"]
    sim = res["simulation"]
    t_m = sim["time_s"] / 60.0

    axes[0, 0].plot(hist["iter"], hist["J_total"], "o-", markersize=3, linewidth=1.6, label="Amaç J")
    axes[0, 0].axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="1C CC-CV Referans (J=1)")
    axes[0, 0].set_title("Yakınsama Eğrisi")
    axes[0, 0].set_xlabel("İterasyon [-]")
    axes[0, 0].set_ylabel("Normalize Amaç J [-]")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].step(t_m, sim["current_A"], where="post", linewidth=1.8, color="#1f77b4")
    axes[0, 1].set_title(f"Optimal Akım Profili (t_f = {sim['final_time_s']/60:.2f} dk)")
    axes[0, 1].set_xlabel("Süre [dk]")
    axes[0, 1].set_ylabel("Akım [A]")

    axes[1, 0].plot(t_m, sim["terminal_voltage_V"], linewidth=1.8, color="#d62728")
    axes[1, 0].axhline(limits.v_max_V, color="gray", linestyle="--", linewidth=1.0, label=f"V_max = {limits.v_max_V} V")
    axes[1, 0].set_title(f"Terminal Gerilimi (V_max_sim = {sim['max_voltage_V']:.4f} V)")
    axes[1, 0].set_xlabel("Süre [dk]")
    axes[1, 0].set_ylabel("Gerilim [V]")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(t_m, sim["temperature_C"], linewidth=1.8, color="#ff7f0e")
    axes[1, 1].axhline(limits.t_max_C, color="gray", linestyle="--", linewidth=1.0, label=f"T_max = {limits.t_max_C} °C")
    axes[1, 1].set_title(f"Hücre Sıcaklığı (T_max_sim = {sim['max_temperature_C']:.2f} °C)")
    axes[1, 1].set_xlabel("Süre [dk]")
    axes[1, 1].set_ylabel("Sıcaklık [°C]")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_optimizer_comparison(
    cccv_sim: dict[str, Any],
    results: dict[str, dict[str, Any]],
    limits: BatteryLimits,
    out_path: Path,
) -> None:
    """CC-CV, Gradient Descent, Adam ve SLSQP algoritmalarını karşılaştıran 2x2 grafik."""
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    colors = {
        "CC-CV (1C)": "#7f7f7f",
        "Gradient Descent": "#1f77b4",
        "Adam": "#ff7f0e",
        "SLSQP": "#2ca02c",
    }

    # Panel 1: Yakınsama
    for name, res in results.items():
        hist = res["history"]
        axes[0, 0].plot(hist["iter"], hist["J_total"], linewidth=1.7, color=colors[name], label=f"{name} (J={res['objectives']['J_total']:.4f})")
    axes[0, 0].axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=1.1, label="CC-CV 1C (J=1.0000)")
    axes[0, 0].set_title("Algoritma Yakınsama Karşılaştırması")
    axes[0, 0].set_xlabel("İterasyon [-]")
    axes[0, 0].set_ylabel("Normalize Amaç J [-]")
    axes[0, 0].legend(fontsize=8)

    # Panel 2-4: Yörüngeler
    series_list = [("CC-CV (1C)", cccv_sim, "--")] + [(k, v["simulation"], "-") for k, v in results.items()]
    for label, sim, ls in series_list:
        t_m = sim["time_s"] / 60.0
        col = colors[label]
        axes[0, 1].plot(t_m, sim["current_A"], linestyle=ls, linewidth=1.7, color=col, label=label)
        axes[1, 0].plot(t_m, sim["terminal_voltage_V"], linestyle=ls, linewidth=1.7, color=col, label=label)
        axes[1, 1].plot(t_m, sim["temperature_C"], linestyle=ls, linewidth=1.7, color=col, label=label)

    axes[0, 1].set_title("Optimal Şarj Akımı Profilleri")
    axes[0, 1].set_xlabel("Süre [dk]")
    axes[0, 1].set_ylabel("Akım [A]")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("Terminal Gerilimi (Sınır: 4.20 V)")
    axes[1, 0].set_xlabel("Süre [dk]")
    axes[1, 0].set_ylabel("Gerilim [V]")
    axes[1, 0].axhline(limits.v_max_V, color="k", linestyle=":", linewidth=1.0)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Hücre Sıcaklığı (Sınır: 45.0 °C)")
    axes[1, 1].set_xlabel("Süre [dk]")
    axes[1, 1].set_ylabel("Sıcaklık [°C]")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Aşama 7–9 Karşılaştırması: CC-CV vs. Gradient Descent vs. Adam vs. SLSQP (w=[0.4, 0.2, 0.4])", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_phases_7_to_9_experiments(
    processed_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    outputs_dir: str | Path,
    weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
    seeds: tuple[int, ...] = (42, 123, 2026),
) -> dict[str, Any]:
    """GD, Adam ve SLSQP algoritmalarını çalıştırır, çoklu seed analizi yapar ve sonuçları kaydeder."""
    proc_path = Path(processed_dir)
    tbl_path = Path(tables_dir)
    fig_path = Path(figures_dir)
    out_path = Path(outputs_dir)
    for d in (tbl_path, fig_path, out_path):
        d.mkdir(parents=True, exist_ok=True)

    json_path = proc_path / "calibrated_parameters.json"
    params = BatteryParameters.from_calibrated_json(json_path)
    ocv_fn = OCVInterpolator.from_calibrated_json(json_path)
    limits = BatteryLimits()
    ref_scales = ReferenceScales.from_json(proc_path / "reference_scales.json")

    cccv_sim = simulate_cc_cv(params, ocv_fn, limits, i_cc_A=1.0 * params.Q_Ah, soc_init=0.20, soc_target=0.90, dt_s=1.0)
    cccv_objs = evaluate_objectives(cccv_sim, ref_scales, weights=weights)
    cccv_cons = evaluate_constraints(cccv_sim, limits, i_min_A=0.05, i_max_A=limits.i_max_A)

    res_gd = optimize_gradient_descent(params, ocv_fn, limits, ref_scales, weights=weights, seed=None)
    res_adam = optimize_adam(params, ocv_fn, limits, ref_scales, weights=weights, seed=None)
    res_slsqp = optimize_slsqp(params, ocv_fn, limits, ref_scales, weights=weights, seed=None)

    results_map = {"Gradient Descent": res_gd, "Adam": res_adam, "SLSQP": res_slsqp}

    comp_rows = [
        {
            "strategy": "CC-CV (1.0C Ref)",
            "tf_min": cccv_sim["final_time_s"] / 60.0,
            "energy_loss_Wh": cccv_sim["total_energy_loss_J"] / 3600.0,
            "aging_D": cccv_sim["total_aging_D"],
            "J1": cccv_objs["J1"],
            "J2": cccv_objs["J2"],
            "J3": cccv_objs["J3"],
            "J_total": cccv_objs["J_total"],
            "max_voltage_V": cccv_cons["max_voltage_V"],
            "max_temperature_C": cccv_cons["max_temperature_C"],
            "feasible": cccv_cons["is_feasible"],
            "iterations": 0,
            "nfev": 1,
            "runtime_s": cccv_sim["runtime_s"],
        }
    ]
    for name, r in results_map.items():
        sim = r["simulation"]
        objs = r["objectives"]
        cons = r["constraints"]
        comp_rows.append(
            {
                "strategy": name,
                "tf_min": sim["final_time_s"] / 60.0,
                "energy_loss_Wh": sim["total_energy_loss_J"] / 3600.0,
                "aging_D": sim["total_aging_D"],
                "J1": objs["J1"],
                "J2": objs["J2"],
                "J3": objs["J3"],
                "J_total": objs["J_total"],
                "max_voltage_V": cons["max_voltage_V"],
                "max_temperature_C": cons["max_temperature_C"],
                "feasible": cons["is_feasible"],
                "iterations": r["iterations"],
                "nfev": r["nfev"],
                "runtime_s": r["runtime_s"],
            }
        )

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(tbl_path / "phase9_optimizer_comparison.csv", index=False)

    # Çoklu sabit seed başlangıç koşulu duyarlılığı
    seed_raw_rows: list[dict[str, Any]] = []
    for sd in seeds:
        for name, fn in [
            ("Gradient Descent", optimize_gradient_descent),
            ("Adam", optimize_adam),
            ("SLSQP", optimize_slsqp),
        ]:
            r_sd = fn(params, ocv_fn, limits, ref_scales, weights=weights, seed=sd)
            seed_raw_rows.append(
                {
                    "algorithm": name,
                    "seed": int(sd),
                    "J_total": r_sd["objectives"]["J_total"],
                    "tf_min": r_sd["simulation"]["final_time_s"] / 60.0,
                    "energy_loss_Wh": r_sd["simulation"]["total_energy_loss_J"] / 3600.0,
                    "aging_D": r_sd["simulation"]["total_aging_D"],
                    "feasible": r_sd["constraints"]["is_feasible"],
                    "runtime_s": r_sd["runtime_s"],
                }
            )
    seed_raw_df = pd.DataFrame(seed_raw_rows)
    agg_rows: list[dict[str, Any]] = []
    for name, grp in seed_raw_df.groupby("algorithm"):
        agg_rows.append(
            {
                "algorithm": name,
                "seeds": ",".join(map(str, seeds)),
                "J_total_mean": float(grp["J_total"].mean()),
                "J_total_std": float(grp["J_total"].std(ddof=1)),
                "tf_min_mean": float(grp["tf_min"].mean()),
                "tf_min_std": float(grp["tf_min"].std(ddof=1)),
                "energy_loss_Wh_mean": float(grp["energy_loss_Wh"].mean()),
                "energy_loss_Wh_std": float(grp["energy_loss_Wh"].std(ddof=1)),
                "feasible_count": int(grp["feasible"].sum()),
                "runtime_s_mean": float(grp["runtime_s"].mean()),
            }
        )
    multiseed_df = pd.DataFrame(agg_rows)
    multiseed_df.to_csv(tbl_path / "phase9_multiseed_sensitivity.csv", index=False)

    # Optimal profilleri JSON'a kaydet
    profiles_payload = {
        "weights": list(weights),
        "n_stages": 10,
        "optimizers": {
            name: {
                "u_opt_A": r["u_opt_A"].tolist(),
                "objectives": r["objectives"],
                "constraints": r["constraints"],
                "iterations": r["iterations"],
                "nfev": r["nfev"],
                "runtime_s": r["runtime_s"],
            }
            for name, r in results_map.items()
        },
    }
    with open(out_path / "phase9_optimal_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles_payload, f, indent=2)

    # Grafikler
    f_gd = fig_path / "fig07_gradient_descent.png"
    f_adam = fig_path / "fig08_adam_optimization.png"
    f_comp = fig_path / "fig09_optimizer_comparison.png"
    _plot_single_optimizer(res_gd, limits, "Aşama 7: Gradient Descent Optimizasyon Sonuçları", f_gd)
    _plot_single_optimizer(res_adam, limits, "Aşama 8: Adam Optimizasyon Sonuçları", f_adam)
    _plot_optimizer_comparison(cccv_sim, results_map, limits, f_comp)

    return {
        "comparison_table": comp_df,
        "multiseed_table": multiseed_df,
        "multiseed_raw": seed_raw_df,
        "results": results_map,
        "saved_figures": [str(f_gd), str(f_adam), str(f_comp)],
    }
