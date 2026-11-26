"""Aşama 10–11: Çok amaçlı Pareto cephesi analizi ve Monte Carlo belirsizlik yayılımı."""

from __future__ import annotations

import json
from pathlib import Path
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
    simulate_soc_profile,
)
from src.cc_cv import simulate_cc_cv
from src.constraints import evaluate_constraints
from src.objective_functions import ReferenceScales
from src.optimizers import optimize_slsqp

DEFAULT_PARETO_WEIGHTS: list[tuple[float, float, float]] = [
    (0.90, 0.05, 0.05),
    (0.80, 0.10, 0.10),
    (0.70, 0.15, 0.15),
    (0.70, 0.25, 0.05),
    (0.70, 0.05, 0.25),
    (0.60, 0.20, 0.20),
    (0.55, 0.30, 0.15),
    (0.55, 0.15, 0.30),
    (0.50, 0.25, 0.25),
    (0.45, 0.35, 0.20),
    (0.40, 0.20, 0.40),
    (0.34, 0.33, 0.33),
    (0.30, 0.50, 0.20),
    (0.30, 0.20, 0.50),
    (0.25, 0.55, 0.20),
    (0.25, 0.15, 0.60),
    (0.15, 0.70, 0.15),
    (0.15, 0.15, 0.70),
    (0.05, 0.90, 0.05),
    (0.05, 0.05, 0.90),
]


def identify_pareto_front(
    df: pd.DataFrame,
    objective_cols: tuple[str, ...] = ("tf_min", "energy_loss_Wh", "aging_D"),
) -> np.ndarray:
    """Tüm amaçların minimize edildiği baskın olmayan (Pareto-optimal) çözüm maskesini döndürür."""
    vals = df[list(objective_cols)].to_numpy(dtype=np.float64)
    n = len(vals)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # Başka bir çözüm j, i'den tüm amaçlarda <= ve en az birinde < ise i baskındır (dominated)
        dominates_i = np.all(vals <= vals[i] + 1e-12, axis=1) & np.any(vals < vals[i] - 1e-12, axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            is_pareto[i] = False
    return is_pareto


def run_pareto_analysis(
    processed_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    outputs_dir: str | Path,
    weight_list: list[tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    """Aşama 10: Ağırlık taramasıyla çok amaçlı Pareto cephesini çıkarır ve CC-CV ile kıyaslar."""
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

    weights_grid = weight_list or DEFAULT_PARETO_WEIGHTS
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []

    for idx, w in enumerate(weights_grid):
        u0_init = np.linspace(5.5, 2.8, 10) if w[0] >= 0.40 else np.full(10, 2.5, dtype=np.float64)
        res = optimize_slsqp(
            params, ocv_fn, limits, ref_scales, weights=w,
            u0=u0_init, i_min_A=1.0, maxiter=40, ftol=1e-6, dt_opt_s=20.0,
        )
        if not res["constraints"]["is_feasible"]:
            # Daha muhafazakâr başlangıç noktasından yeniden dene
            res_alt = optimize_slsqp(
                params, ocv_fn, limits, ref_scales, weights=w,
                u0=np.full(10, 2.2, dtype=np.float64), i_min_A=1.0, maxiter=40, ftol=1e-6, dt_opt_s=20.0,
            )
            if res_alt["constraints"]["is_feasible"]:
                res = res_alt
        if (idx + 1) % 5 == 0 or idx + 1 == len(weights_grid):
            print(f"Pareto progress: {idx + 1}/{len(weights_grid)} weights done", flush=True)

        sim = res["simulation"]
        objs = res["objectives"]
        cons = res["constraints"]
        rows.append(
            {
                "case_id": idx + 1,
                "w1_time": w[0],
                "w2_energy": w[1],
                "w3_aging": w[2],
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
                "runtime_s": res["runtime_s"],
            }
        )
        profiles.append(
            {
                "case_id": idx + 1,
                "weights": list(w),
                "u_opt_A": res["u_opt_A"].tolist(),
                "feasible": cons["is_feasible"],
            }
        )

    df = pd.DataFrame(rows)
    feasible_mask = df["feasible"].to_numpy(dtype=bool)
    pareto_mask = np.zeros(len(df), dtype=bool)
    if np.any(feasible_mask):
        pareto_sub = identify_pareto_front(df.loc[feasible_mask])
        pareto_mask[feasible_mask] = pareto_sub
    df["is_pareto_optimal"] = pareto_mask

    df.to_csv(tbl_path / "phase10_pareto_solutions.csv", index=False)
    with open(out_path / "phase10_pareto_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    cccv_df = pd.read_csv(tbl_path / "phase6_cc_cv_comparison.csv").rename(columns={"final_time_min": "tf_min"})
    fig_out = fig_path / "fig10_pareto_analysis.png"
    _plot_pareto_figure(df, cccv_df, fig_out)

    return {
        "pareto_table": df,
        "n_feasible": int(feasible_mask.sum()),
        "n_pareto_optimal": int(pareto_mask.sum()),
        "figure_path": str(fig_out),
    }


def _plot_pareto_figure(df: pd.DataFrame, cccv_df: pd.DataFrame, out_path: Path) -> None:
    """3 ikili ödünleşim panelinde Pareto cephesini ve CC-CV referans eğrisini çizer."""
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    feas = df[df["feasible"]]
    par = feas[feas["is_pareto_optimal"]]
    dom = feas[~feas["is_pareto_optimal"]]
    knee = feas[(feas["w1_time"] == 0.40) & (feas["w2_energy"] == 0.20)]

    pairs = [
        ("tf_min", "aging_D", "Şarj Süresi [dk]", "Göreli Yaşlanma D [-]", "Süre – Yaşlanma Ödünleşimi"),
        ("tf_min", "energy_loss_Wh", "Şarj Süresi [dk]", "Enerji Kaybı [Wh]", "Süre – Enerji Kaybı Ödünleşimi"),
        ("energy_loss_Wh", "aging_D", "Enerji Kaybı [Wh]", "Göreli Yaşlanma D [-]", "Enerji Kaybı – Yaşlanma İlişkisi"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes, pairs):
        if not dom.empty:
            ax.scatter(dom[xcol], dom[ycol], c="#aec7e8", s=45, label="Baskılanmış (Dominated)")
        par_sorted = par.sort_values(xcol)
        ax.plot(par_sorted[xcol], par_sorted[ycol], "o-", color="#1f77b4", markersize=6, linewidth=1.6, label="Pareto Cephesi (SLSQP)")
        if not knee.empty:
            ax.scatter(knee[xcol], knee[ycol], c="#2ca02c", s=110, marker="*", zorder=5, label="Dengeli (w=[0.4,0.2,0.4])")
        ax.plot(cccv_df[xcol], cccv_df[ycol], "s--", color="#d62728", markersize=5.5, linewidth=1.5, label="CC-CV (0.5C–2.0C)")
        for _, r in cccv_df.iterrows():
            ax.annotate(f"{r['c_rate']:.1f}C", (r[xcol], r[ycol]), textcoords="offset points", xytext=(4, 4), fontsize=7.5, color="#d62728")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7.5)

    fig.suptitle("Aşama 10: Çok Amaçlı Optimal Şarj Pareto Analizi (SOC: 0.20 → 0.90)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def sample_uncertain_parameters(
    base_params: BatteryParameters,
    std_dict: dict[str, float],
    n_samples: int = 200,
    seed: int = 2026,
) -> pd.DataFrame:
    """R0, R1, C1, Q, Rth, Cth için sabit seed ile kesilmiş normal dağılım örnekleri üretir."""
    rng = np.random.default_rng(int(seed))
    specs = {
        "Q_Ah": (base_params.Q_Ah, std_dict.get("Q_Ah", 0.025), (4.20, 5.40)),
        "R0_Ohm": (base_params.R0_Ohm, std_dict.get("R0_Ohm", 0.0037), (0.012, 0.060)),
        "R1_Ohm": (base_params.R1_Ohm, std_dict.get("R1_Ohm", 0.0005), (0.005, 0.040)),
        "C1_F": (base_params.C1_F, std_dict.get("C1_F", 130.0), (1000.0, 8000.0)),
        "Rth_K_per_W": (base_params.Rth_K_per_W, std_dict.get("Rth_K_per_W", 0.64), (2.0, 12.0)),
        "Cth_J_per_K": (base_params.Cth_J_per_K, std_dict.get("Cth_J_per_K", 13.5), (60.0, 250.0)),
    }

    data: dict[str, np.ndarray] = {"sample_id": np.arange(n_samples, dtype=int)}
    for name, (mu, sigma, (lo, hi)) in specs.items():
        draws = rng.normal(loc=float(mu), scale=float(sigma), size=n_samples)
        data[name] = np.clip(draws, lo, hi)

    return pd.DataFrame(data)


def run_monte_carlo_analysis(
    processed_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    outputs_dir: str | Path,
    n_samples: int = 200,
    seed: int = 2026,
) -> dict[str, Any]:
    """Aşama 11: Belirsiz parametreler altında SLSQP optimal ve CC-CV 1C Monte Carlo simülasyonu."""
    proc_path = Path(processed_dir)
    tbl_path = Path(tables_dir)
    fig_path = Path(figures_dir)
    out_path = Path(outputs_dir)
    for d in (tbl_path, fig_path, out_path):
        d.mkdir(parents=True, exist_ok=True)

    json_path = proc_path / "calibrated_parameters.json"
    with open(json_path, "r", encoding="utf-8") as f:
        cal_payload = json.load(f)

    base_params = BatteryParameters.from_calibrated_json(json_path)
    ocv_fn = OCVInterpolator.from_calibrated_json(json_path)
    limits = BatteryLimits()
    std_dict = {k: float(v) for k, v in cal_payload["uncertainties_std"].items()}

    with open(out_path / "phase9_optimal_profiles.json", "r", encoding="utf-8") as f:
        prof_payload = json.load(f)
    u_slsqp = np.asarray(prof_payload["optimizers"]["SLSQP"]["u_opt_A"], dtype=np.float64)
    i_cc_1c = 1.0 * base_params.Q_Ah

    samples_df = sample_uncertain_parameters(base_params, std_dict, n_samples=n_samples, seed=seed)
    raw_rows: list[dict[str, Any]] = []

    for _, row in samples_df.iterrows():
        p_i = BatteryParameters(
            Q_Ah=float(row["Q_Ah"]),
            eta=base_params.eta,
            R0_Ohm=float(row["R0_Ohm"]),
            R1_Ohm=float(row["R1_Ohm"]),
            C1_F=float(row["C1_F"]),
            Rth_K_per_W=float(row["Rth_K_per_W"]),
            Cth_J_per_K=float(row["Cth_J_per_K"]),
            T_amb_C=base_params.T_amb_C,
            k0=base_params.k0,
            Ea_J_per_mol=base_params.Ea_J_per_mol,
            p_crate=base_params.p_crate,
            alpha_soc=base_params.alpha_soc,
            is_calibrated=True,
        )

        # 1. Açık döngü SLSQP optimal akım profili
        sim_opt = simulate_soc_profile(u_slsqp, p_i, ocv_fn, method="rk4", dt_max_s=5.0)
        cons_opt = evaluate_constraints(sim_opt, limits, tol_v=2e-3, tol_t=5e-2)
        raw_rows.append(
            {
                "strategy": "SLSQP Optimal",
                "sample_id": int(row["sample_id"]),
                "Q_Ah": p_i.Q_Ah,
                "R0_Ohm": p_i.R0_Ohm,
                "R1_Ohm": p_i.R1_Ohm,
                "C1_F": p_i.C1_F,
                "Rth_K_per_W": p_i.Rth_K_per_W,
                "Cth_J_per_K": p_i.Cth_J_per_K,
                "tf_min": sim_opt["final_time_s"] / 60.0,
                "energy_loss_Wh": sim_opt["total_energy_loss_J"] / 3600.0,
                "aging_D": sim_opt["total_aging_D"],
                "max_voltage_V": sim_opt["max_voltage_V"],
                "max_temperature_C": sim_opt["max_temperature_C"],
                "v_violation": sim_opt["max_voltage_V"] > limits.v_max_V + 2e-3,
                "t_violation": sim_opt["max_temperature_C"] > limits.t_max_C + 5e-2,
                "feasible": cons_opt["is_feasible"],
            }
        )

        # 2. Geleneksel CC-CV (1C) kapalı gerilim kontrollü referans
        sim_cc = simulate_cc_cv(p_i, ocv_fn, limits, i_cc_A=i_cc_1c, dt_s=2.0)
        cons_cc = evaluate_constraints(sim_cc, limits, i_min_A=0.05, tol_v=2e-3, tol_t=5e-2)
        raw_rows.append(
            {
                "strategy": "CC-CV (1.0C)",
                "sample_id": int(row["sample_id"]),
                "Q_Ah": p_i.Q_Ah,
                "R0_Ohm": p_i.R0_Ohm,
                "R1_Ohm": p_i.R1_Ohm,
                "C1_F": p_i.C1_F,
                "Rth_K_per_W": p_i.Rth_K_per_W,
                "Cth_J_per_K": p_i.Cth_J_per_K,
                "tf_min": sim_cc["final_time_s"] / 60.0,
                "energy_loss_Wh": sim_cc["total_energy_loss_J"] / 3600.0,
                "aging_D": sim_cc["total_aging_D"],
                "max_voltage_V": sim_cc["max_voltage_V"],
                "max_temperature_C": sim_cc["max_temperature_C"],
                "v_violation": sim_cc["max_voltage_V"] > limits.v_max_V + 2e-3,
                "t_violation": sim_cc["max_temperature_C"] > limits.t_max_C + 5e-2,
                "feasible": cons_cc["is_feasible"],
            }
        )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(out_path / "phase11_monte_carlo_raw.csv", index=False)

    # Özet istatistikler (ortalama ± standart sapma)
    summary_rows: list[dict[str, Any]] = []
    metrics = ["tf_min", "energy_loss_Wh", "aging_D", "max_voltage_V", "max_temperature_C"]
    for strat, grp in raw_df.groupby("strategy"):
        rec: dict[str, Any] = {
            "strategy": strat,
            "n_samples": int(len(grp)),
            "seed": int(seed),
            "feasible_pct": 100.0 * float(grp["feasible"].mean()),
            "v_violation_pct": 100.0 * float(grp["v_violation"].mean()),
            "t_violation_pct": 100.0 * float(grp["t_violation"].mean()),
        }
        for m in metrics:
            rec[f"{m}_mean"] = float(grp[m].mean())
            rec[f"{m}_std"] = float(grp[m].std(ddof=1))
            rec[f"{m}_p5"] = float(np.percentile(grp[m], 5))
            rec[f"{m}_p95"] = float(np.percentile(grp[m], 95))
        summary_rows.append(rec)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tbl_path / "phase11_monte_carlo_summary.csv", index=False)

    # Duyarlılık korelasyonları (SLSQP stratejisi)
    opt_df = raw_df[raw_df["strategy"] == "SLSQP Optimal"]
    param_cols = ["R0_Ohm", "R1_Ohm", "C1_F", "Q_Ah", "Rth_K_per_W", "Cth_J_per_K"]
    sens_rows: list[dict[str, Any]] = []
    for pcol in param_cols:
        sens_rows.append(
            {
                "parameter": pcol,
                "corr_tf_min": float(opt_df[pcol].corr(opt_df["tf_min"])),
                "corr_energy_loss_Wh": float(opt_df[pcol].corr(opt_df["energy_loss_Wh"])),
                "corr_aging_D": float(opt_df[pcol].corr(opt_df["aging_D"])),
                "corr_max_voltage_V": float(opt_df[pcol].corr(opt_df["max_voltage_V"])),
                "corr_max_temperature_C": float(opt_df[pcol].corr(opt_df["max_temperature_C"])),
            }
        )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(tbl_path / "phase11_monte_carlo_sensitivity.csv", index=False)

    fig_out = fig_path / "fig11_monte_carlo_uncertainty.png"
    _plot_monte_carlo_figure(raw_df, sens_df, limits, n_samples, seed, fig_out)

    return {
        "summary_table": summary_df,
        "sensitivity_table": sens_df,
        "raw_df": raw_df,
        "figure_path": str(fig_out),
    }


def _plot_monte_carlo_figure(
    raw_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    limits: BatteryLimits,
    n_samples: int,
    seed: int,
    out_path: Path,
) -> None:
    """Monte Carlo dağılım histogramları ve parametre duyarlılık çubuk grafiğini üretir."""
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    opt = raw_df[raw_df["strategy"] == "SLSQP Optimal"]
    cccv = raw_df[raw_df["strategy"] == "CC-CV (1.0C)"]

    axes[0, 0].hist(opt["energy_loss_Wh"], bins=22, alpha=0.72, color="#2ca02c", label=f"SLSQP ({opt['energy_loss_Wh'].mean():.3f} ± {opt['energy_loss_Wh'].std():.3f} Wh)")
    axes[0, 0].hist(cccv["energy_loss_Wh"], bins=22, alpha=0.60, color="#7f7f7f", label=f"CC-CV 1C ({cccv['energy_loss_Wh'].mean():.3f} ± {cccv['energy_loss_Wh'].std():.3f} Wh)")
    axes[0, 0].set_title("Enerji Kaybı Belirsizlik Dağılımı")
    axes[0, 0].set_xlabel("Enerji Kaybı [Wh]")
    axes[0, 0].set_ylabel("Frekans [-]")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(opt["aging_D"] * 1e4, bins=22, alpha=0.72, color="#1f77b4", label=f"SLSQP ({opt['aging_D'].mean()*1e4:.3f} ± {opt['aging_D'].std()*1e4:.3f})")
    axes[0, 1].hist(cccv["aging_D"] * 1e4, bins=22, alpha=0.60, color="#7f7f7f", label=f"CC-CV 1C ({cccv['aging_D'].mean()*1e4:.3f} ± {cccv['aging_D'].std()*1e4:.3f})")
    axes[0, 1].set_title("Göreli Yaşlanma Göstergesi (D × 10⁴)")
    axes[0, 1].set_xlabel("D × 10⁴ [-]")
    axes[0, 1].set_ylabel("Frekans [-]")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].hist(opt["max_voltage_V"], bins=22, alpha=0.75, color="#ff7f0e", label=f"SLSQP Açık Döngü ({opt['max_voltage_V'].mean():.3f} ± {opt['max_voltage_V'].std():.3f} V)")
    axes[1, 0].axvline(limits.v_max_V, color="#d62728", linestyle="--", linewidth=1.6, label=f"V_max = {limits.v_max_V:.2f} V")
    axes[1, 0].set_title("Maksimum Terminal Gerilimi Dağılımı (SLSQP Açık Döngü)")
    axes[1, 0].set_xlabel("Maks. Terminal Gerilimi [V]")
    axes[1, 0].set_ylabel("Frekans [-]")
    axes[1, 0].legend(fontsize=8)

    x = np.arange(len(sens_df))
    w = 0.27
    axes[1, 1].bar(x - w, sens_df["corr_energy_loss_Wh"], width=w, color="#2ca02c", label="Enerji Kaybı")
    axes[1, 1].bar(x, sens_df["corr_max_voltage_V"], width=w, color="#ff7f0e", label="Maks. Gerilim")
    axes[1, 1].bar(x + w, sens_df["corr_max_temperature_C"], width=w, color="#d62728", label="Maks. Sıcaklık")
    axes[1, 1].axhline(0.0, color="gray", linewidth=0.9)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(sens_df["parameter"], rotation=20, fontsize=8.5)
    axes[1, 1].set_title("Parametre Duyarlılığı (Pearson Korelasyonu)")
    axes[1, 1].set_ylabel("Korelasyon Katsayısı r [-]")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"Aşama 11: Monte Carlo Belirsizlik Analizi (N={n_samples}, seed={seed})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
