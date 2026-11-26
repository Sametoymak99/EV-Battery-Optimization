"""Aşama 4: Deneysel verilerden parametre tahmini, EIS çapraz doğrulaması ve bağımsız test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.signal import lfilter

from src.battery_model import BatteryParameters, OCVInterpolator
from src.data_loader import (
    STANFORD_CELL_CRATES,
    extract_stanford_diag_df,
    load_stanford_diagnostic,
)


def build_calibrated_ocv(
    ocv_csv_path: str | Path,
    cap_fade_csv_path: str | Path,
    calibration_cells: tuple[str, ...] = ("W4", "W8", "W9", "W10"),
) -> tuple[OCVInterpolator, pd.DataFrame, float, float]:
    """Kalibrasyon hücrelerinden ortalama Q [Ah] ve OCV(SOC) eğrisini (veri sızıntısız) oluşturur."""
    cap_df = pd.read_csv(cap_fade_csv_path)
    sub_cap = cap_df[(cap_df["diag_number"] == 1) & (cap_df["cell_id"].isin(calibration_cells))]
    if len(sub_cap) != len(calibration_cells):
        raise ValueError("Kalibrasyon hücrelerinin tümü Diag_1 kapasite tablosunda bulunamadı.")

    q_vals = sub_cap["capacity_Ah"].to_numpy(dtype=np.float64)
    q_mean = float(np.mean(q_vals))
    q_std = float(np.std(q_vals, ddof=1)) if len(q_vals) > 1 else 0.0

    ocv_raw = pd.read_csv(ocv_csv_path)
    soc_grid = np.linspace(0.0, 1.0, 201)
    curves: list[np.ndarray] = []
    for cid in calibration_cells:
        interp = OCVInterpolator.from_processed_csv(ocv_csv_path, cell_id=cid)
        curves.append(np.asarray(interp(soc_grid), dtype=np.float64))

    ocv_stack = np.vstack(curves)
    ocv_mean = np.maximum.accumulate(np.mean(ocv_stack, axis=0))
    ocv_std = np.std(ocv_stack, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(soc_grid)

    ocv_grid_df = pd.DataFrame(
        {
            "soc": soc_grid,
            "ocv_V": ocv_mean,
            "ocv_std_V": ocv_std,
        }
    )
    ocv_fn = OCVInterpolator(soc_grid, ocv_mean, cell_id="calibrated_mean")
    return ocv_fn, ocv_grid_df, q_mean, q_std


def simulate_1rc_voltage(
    time_s: np.ndarray,
    current_A: np.ndarray,
    ocv_V: np.ndarray,
    R0_Ohm: float,
    R1_Ohm: float,
    C1_F: float,
    vp0_V: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """1-RC polarizasyon gerilimini parça-sabit analitik adımla hesaplar ve (Vt, Vp) döndürür."""
    t = np.asarray(time_s, dtype=np.float64)
    curr = np.asarray(current_A, dtype=np.float64)
    tau = max(R1_Ohm * C1_F, 1e-6)
    dt = np.diff(t)

    if len(dt) > 0 and vp0_V == 0.0 and np.allclose(dt, dt[0], rtol=1e-4, atol=1e-4):
        alpha = float(np.exp(-dt[0] / tau))
        b = np.array([0.0, R1_Ohm * (1.0 - alpha)], dtype=np.float64)
        a = np.array([1.0, -alpha], dtype=np.float64)
        vp = lfilter(b, a, curr)
    else:
        vp = np.zeros_like(t)
        vp[0] = vp0_V
        for k in range(len(t) - 1):
            alpha = float(np.exp(-dt[k] / tau))
            vp[k + 1] = alpha * vp[k] + R1_Ohm * (1.0 - alpha) * curr[k]

    vt = np.asarray(ocv_V, dtype=np.float64) + R0_Ohm * curr + vp
    return vt, vp


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, maksimum mutlak hata ve R^2 metriklerini hesaplar."""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"rmse": rmse, "mae": mae, "max_abs_err": max_abs, "r2": r2}


def compute_hppc_soc(df_hppc: pd.DataFrame, q_ah: float, soc0: float = 1.0) -> np.ndarray:
    """Tam şarjdan başlayan HPPC testi için Coulomb sayımıyla SOC(t) hesaplar."""
    t = df_hppc["time_s"].to_numpy(dtype=np.float64)
    curr = df_hppc["current_A"].to_numpy(dtype=np.float64)
    q_cum = np.concatenate([[0.0], cumulative_trapezoid(curr, t) / 3600.0])
    return np.clip(soc0 + q_cum / float(q_ah), 0.0, 1.0)


def estimate_pulse_r0(df_hppc: pd.DataFrame, di_threshold: float = 0.5) -> float:
    """HPPC akım basamaklarındaki anlık dV/dI oranlarının medyanıyla R0 ön kestirimi yapar."""
    curr = df_hppc["current_A"].to_numpy(dtype=np.float64)
    volt = df_hppc["voltage_V"].to_numpy(dtype=np.float64)
    d_i = np.diff(curr)
    d_v = np.diff(volt)
    jumps = np.abs(d_i) > di_threshold
    if not np.any(jumps):
        return np.nan
    r_inst = d_v[jumps] / d_i[jumps]
    valid = r_inst[(r_inst > 0.002) & (r_inst < 0.2)]
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def estimate_electrical_params_hppc(
    df_hppc: pd.DataFrame,
    ocv_fn: OCVInterpolator,
    q_ah: float,
    soc_window: tuple[float, float] = (0.15, 0.95),
    initial_guess: tuple[float, float, float] = (0.026, 0.015, 2000.0),
) -> dict[str, Any]:
    """HPPC gerilim eğrisine doğrusal olmayan en küçük karelerle (R0, R1, C1) uydurur."""
    t = df_hppc["time_s"].to_numpy(dtype=np.float64)
    curr = df_hppc["current_A"].to_numpy(dtype=np.float64)
    v_meas = df_hppc["voltage_V"].to_numpy(dtype=np.float64)
    soc = compute_hppc_soc(df_hppc, q_ah=q_ah, soc0=1.0)
    ocv_vec = np.asarray(ocv_fn(soc), dtype=np.float64)

    mask = (soc >= soc_window[0]) & (soc <= soc_window[1])
    if int(mask.sum()) < 100:
        raise ValueError("HPPC SOC penceresinde yeterli veri noktası yok.")

    def _residuals(theta: np.ndarray) -> np.ndarray:
        r0, r1, c1 = theta
        vt_sim, _ = simulate_1rc_voltage(t, curr, ocv_vec, r0, r1, c1)
        return vt_sim[mask] - v_meas[mask]

    res = least_squares(
        _residuals,
        x0=np.array(initial_guess, dtype=np.float64),
        bounds=(np.array([0.005, 0.001, 50.0]), np.array([0.15, 0.15, 30000.0])),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
    )

    r0_fit, r1_fit, c1_fit = [float(x) for x in res.x]
    vt_full, _ = simulate_1rc_voltage(t, curr, ocv_vec, r0_fit, r1_fit, c1_fit)
    metrics = _regression_metrics(v_meas[mask], vt_full[mask])

    return {
        "R0_Ohm": r0_fit,
        "R1_Ohm": r1_fit,
        "C1_F": c1_fit,
        "tau1_s": r1_fit * c1_fit,
        "R0_pulse_Ohm": estimate_pulse_r0(df_hppc),
        "rmse_V": metrics["rmse"],
        "mae_V": metrics["mae"],
        "max_abs_err_V": metrics["max_abs_err"],
        "r2": metrics["r2"],
        "n_points": int(mask.sum()),
        "success": bool(res.success),
    }


def extract_eis_r0_summary(
    eis_mat_path: str | Path,
    cells: tuple[str, ...] = ("W8", "W9", "W10"),
    diag_number: int = 1,
) -> pd.DataFrame:
    """EIS spektrumlarında Im(Z) ≈ 0 kesişiminden yüksek frekans ohmik direncini (Re) çıkarır."""
    eis_data = load_stanford_diagnostic(eis_mat_path, test_kind="eis_test")
    rows: list[dict[str, Any]] = []
    for cid in cells:
        try:
            df_eis = extract_stanford_diag_df(eis_data, cid, diag_number)
        except ValueError:
            continue
        for soc_pct, grp in df_eis.groupby("soc_pct"):
            idx_min = int(np.argmin(np.abs(grp["im_z_Ohm"].to_numpy())))
            row = grp.iloc[idx_min]
            rows.append(
                {
                    "cell_id": cid,
                    "diag_number": int(diag_number),
                    "soc_pct": int(soc_pct),
                    "freq_Hz": float(row["freq_Hz"]),
                    "R0_eis_Ohm": float(row["re_z_Ohm"]),
                    "im_z_Ohm": float(row["im_z_Ohm"]),
                }
            )
    return pd.DataFrame(rows)


def simulate_lumped_thermal(
    time_s: np.ndarray,
    power_loss_W: np.ndarray,
    Rth_K_per_W: float,
    Cth_J_per_K: float,
    T_amb_C: float,
    T0_C: float,
) -> np.ndarray:
    """Birleştirilmiş termal ODE'yi parça-sabit analitik adımla çözer: Cth dT/dt = P - (T - Tamb)/Rth."""
    t = np.asarray(time_s, dtype=np.float64)
    p = np.asarray(power_loss_W, dtype=np.float64)
    tau_th = max(Rth_K_per_W * Cth_J_per_K, 1e-3)
    dt = np.diff(t)

    if len(dt) > 0 and np.allclose(dt, dt[0], rtol=1e-4, atol=1e-4):
        beta = float(np.exp(-dt[0] / tau_th))
        b = np.array([0.0, Rth_K_per_W * (1.0 - beta)], dtype=np.float64)
        a = np.array([1.0, -beta], dtype=np.float64)
        # T0_C - T_amb_C başlangıç farkının üstel sönümünü ekle
        steps = np.arange(len(t), dtype=np.float64)
        t_excess = lfilter(b, a, p) + (T0_C - T_amb_C) * (beta ** steps)
        return T_amb_C + t_excess

    temp = np.zeros_like(t)
    temp[0] = T0_C
    for k in range(len(t) - 1):
        beta = float(np.exp(-dt[k] / tau_th))
        steady = T_amb_C + Rth_K_per_W * p[k]
        temp[k + 1] = steady + (temp[k] - steady) * beta
    return temp


def estimate_thermal_params_w10(
    df_therm: pd.DataFrame,
    R0_Ohm: float,
    R1_Ohm: float,
    C1_F: float,
) -> dict[str, Any]:
    """W10 (3C şarj + 30 dk dinlenme) verisinden (Rth, Cth, T_amb) kestirimi yapar."""
    t = df_therm["time_s"].to_numpy(dtype=np.float64)
    curr = df_therm["current_A"].to_numpy(dtype=np.float64)
    t_meas = df_therm["temperature_C"].to_numpy(dtype=np.float64)

    _, vp = simulate_1rc_voltage(t, curr, np.zeros_like(t), R0_Ohm, R1_Ohm, C1_F)
    p_loss = (curr ** 2) * R0_Ohm + (vp ** 2) / R1_Ohm
    t0 = float(t_meas[0])

    def _residuals(theta: np.ndarray) -> np.ndarray:
        rth, cth, tamb = theta
        t_sim = simulate_lumped_thermal(t, p_loss, rth, cth, tamb, t0)
        return t_sim - t_meas

    res = least_squares(
        _residuals,
        x0=np.array([10.0, 65.0, 22.7], dtype=np.float64),
        bounds=(np.array([2.0, 25.0, 21.5]), np.array([35.0, 180.0, 24.0])),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
    )

    rth_fit, cth_fit, tamb_fit = [float(x) for x in res.x]
    t_sim = simulate_lumped_thermal(t, p_loss, rth_fit, cth_fit, tamb_fit, t0)
    full_metrics = _regression_metrics(t_meas, t_sim)

    # Faz bazlı değerlendirme (Step 7-8 hızlı 3C ısınma vs Step 9-11 soğuma)
    steps = df_therm["step_index"].to_numpy()
    heat_mask = np.isin(steps, [7, 8])
    cool_mask = np.isin(steps, [9, 10, 11])
    heat_metrics = _regression_metrics(t_meas[heat_mask], t_sim[heat_mask]) if np.any(heat_mask) else {"rmse": np.nan}
    cool_metrics = _regression_metrics(t_meas[cool_mask], t_sim[cool_mask]) if np.any(cool_mask) else {"rmse": np.nan}

    return {
        "Rth_K_per_W": rth_fit,
        "Cth_J_per_K": cth_fit,
        "T_amb_fit_C": tamb_fit,
        "tau_th_s": rth_fit * cth_fit,
        "rmse_C": full_metrics["rmse"],
        "mae_C": full_metrics["mae"],
        "max_abs_err_C": full_metrics["max_abs_err"],
        "r2": full_metrics["r2"],
        "heating_rmse_C": heat_metrics["rmse"],
        "cooling_rmse_C": cool_metrics["rmse"],
        "temperature_sim_C": t_sim,
        "power_loss_W": p_loss,
        "success": bool(res.success),
    }


def validate_model_independent(
    params: BatteryParameters,
    ocv_fn: OCVInterpolator,
    hppc_data: dict[str, Any],
    diag1_caps: dict[str, float],
    rep_chg_df: pd.DataFrame,
    calibration_cells: tuple[str, ...] = ("W4", "W8", "W9", "W10"),
    validation_cells_hppc: tuple[tuple[str, int], ...] = (("W5", 2), ("V4", 1)),
) -> pd.DataFrame:
    """Kalibre parametreleri eğitim ve bağımsız doğrulama hücreleri üzerinde test eder."""
    rows: list[dict[str, Any]] = []

    # 1. HPPC Doğrulama (Kalibrasyon Diag_1 + Görülmemiş Hücreler W5, V4)
    hppc_targets = [(cid, 1, "calibration") for cid in calibration_cells] + [
        (cid, dnum, "independent_validation") for cid, dnum in validation_cells_hppc
    ]
    for cid, dnum, split in hppc_targets:
        df_h = extract_stanford_diag_df(hppc_data, cid, dnum)
        q_cell = diag1_caps.get(cid, params.Q_Ah)
        soc = compute_hppc_soc(df_h, q_ah=q_cell, soc0=1.0)
        mask = (soc >= 0.15) & (soc <= 0.95)
        ocv_vec = np.asarray(ocv_fn(soc), dtype=np.float64)
        vt_sim, _ = simulate_1rc_voltage(
            df_h["time_s"].to_numpy(),
            df_h["current_A"].to_numpy(),
            ocv_vec,
            params.R0_Ohm,
            params.R1_Ohm,
            params.C1_F,
        )
        m = _regression_metrics(df_h["voltage_V"].to_numpy()[mask], vt_sim[mask])
        rows.append(
            {
                "source": f"HPPC_Diag{dnum}",
                "split": split,
                "cell_id": cid,
                "c_rate": STANFORD_CELL_CRATES[cid],
                "rmse_mV": m["rmse"] * 1000.0,
                "mae_mV": m["mae"] * 1000.0,
                "max_abs_err_mV": m["max_abs_err"] * 1000.0,
                "r2": m["r2"],
            }
        )

    # 2. Cycling_1 Çevrim 2 Şarj Eğrileri Üzerinde Gerilim Doğrulaması
    for cid in calibration_cells:
        sub = rep_chg_df[rep_chg_df["cell_id"] == cid].reset_index(drop=True)
        ocv_vec = np.asarray(ocv_fn(sub["soc"].to_numpy()), dtype=np.float64)
        vt_sim, _ = simulate_1rc_voltage(
            sub["time_s"].to_numpy(),
            sub["current_A"].to_numpy(),
            ocv_vec,
            params.R0_Ohm,
            params.R1_Ohm,
            params.C1_F,
        )
        m = _regression_metrics(sub["voltage_V"].to_numpy(), vt_sim)
        rows.append(
            {
                "source": "Cycling1_Cycle2_Charge",
                "split": "forward_protocol_check",
                "cell_id": cid,
                "c_rate": STANFORD_CELL_CRATES[cid],
                "rmse_mV": m["rmse"] * 1000.0,
                "mae_mV": m["mae"] * 1000.0,
                "max_abs_err_mV": m["max_abs_err"] * 1000.0,
                "r2": m["r2"],
            }
        )

    return pd.DataFrame(rows)


def _generate_phase4_figures(
    hppc_data: dict[str, Any],
    ocv_fn: OCVInterpolator,
    params: BatteryParameters,
    diag1_caps: dict[str, float],
    df_therm: pd.DataFrame,
    therm_res: dict[str, Any],
    rep_chg_df: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    """Aşama 4 kalibrasyon ve bağımsız doğrulama grafiklerini üretir."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    saved: list[str] = []

    # 1. HPPC Kalibrasyon (W8) ve Bağımsız Doğrulama (W5)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col_idx, (cid, dnum, ttl) in enumerate(
        [("W8", 1, "Kalibrasyon Hücresi: W8 (Diag_1)"), ("W5", 2, "Bağımsız Doğrulama: W5 (Diag_2)")]
    ):
        df_h = extract_stanford_diag_df(hppc_data, cid, dnum)
        soc = compute_hppc_soc(df_h, q_ah=diag1_caps.get(cid, params.Q_Ah), soc0=1.0)
        ocv_vec = np.asarray(ocv_fn(soc), dtype=np.float64)
        vt_sim, _ = simulate_1rc_voltage(
            df_h["time_s"].to_numpy(),
            df_h["current_A"].to_numpy(),
            ocv_vec,
            params.R0_Ohm,
            params.R1_Ohm,
            params.C1_F,
        )
        t_h = df_h["time_s"].to_numpy() / 3600.0
        err_mv = (vt_sim - df_h["voltage_V"].to_numpy()) * 1000.0
        mask = (soc >= 0.15) & (soc <= 0.95)
        rmse_mv = float(np.sqrt(np.mean(err_mv[mask] ** 2)))

        axes[0, col_idx].plot(t_h, df_h["voltage_V"], color="#1f77b4", linewidth=1.4, label="Deney (HPPC)")
        axes[0, col_idx].plot(t_h, vt_sim, color="#d62728", linestyle="--", linewidth=1.2, label="1-RC Model")
        axes[0, col_idx].set_title(f"{ttl} (RMSE={rmse_mv:.1f} mV)")
        axes[0, col_idx].set_ylabel("Terminal Gerilimi [V]")
        axes[0, col_idx].legend(fontsize=8)

        axes[1, col_idx].plot(t_h[mask], err_mv[mask], color="#2ca02c", linewidth=1.0)
        axes[1, col_idx].axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        axes[1, col_idx].set_xlabel("Süre [h]")
        axes[1, col_idx].set_ylabel("Gerilim Hatası (Model - Deney) [mV]")

    fig.tight_layout()
    p1 = figures_dir / "fig04_hppc_calibration_validation.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved.append(str(p1))

    # 2. Termal Kalibrasyon (W10 3C)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t_min = df_therm["time_s"].to_numpy() / 60.0
    t_meas = df_therm["temperature_C"].to_numpy()
    t_sim = therm_res["temperature_sim_C"]
    axes[0].plot(t_min, t_meas, color="#ff7f0e", linewidth=1.6, label="Deney (Aux_Temperature)")
    axes[0].plot(t_min, t_sim, color="#1f77b4", linestyle="--", linewidth=1.6, label="Lumped Termal Model")
    axes[0].set_title(
        f"W10 (3C Şarj + Dinlenme) Termal Kalibrasyon: Rth={therm_res['Rth_K_per_W']:.2f} K/W, "
        f"Cth={therm_res['Cth_J_per_K']:.1f} J/K, RMSE={therm_res['rmse_C']:.2f} °C"
    )
    axes[0].set_ylabel("Hücre Yüzey Sıcaklığı [°C]")
    axes[0].legend(fontsize=9)

    axes[1].plot(t_min, t_sim - t_meas, color="#9467bd", linewidth=1.2)
    axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("Sıcaklık Hatası [°C]")
    axes[1].set_xlabel("Süre [dk]")
    fig.tight_layout()
    p2 = figures_dir / "fig04_thermal_calibration_w10.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved.append(str(p2))

    # 3. Cycling_1 Çevrim 2 Şarj Gerilimi Doğrulaması
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, cid in zip(axes.ravel(), ["W4", "W8", "W9", "W10"]):
        sub = rep_chg_df[rep_chg_df["cell_id"] == cid].reset_index(drop=True)
        ocv_vec = np.asarray(ocv_fn(sub["soc"].to_numpy()), dtype=np.float64)
        vt_sim, _ = simulate_1rc_voltage(
            sub["time_s"].to_numpy(),
            sub["current_A"].to_numpy(),
            ocv_vec,
            params.R0_Ohm,
            params.R1_Ohm,
            params.C1_F,
        )
        rmse_mv = float(np.sqrt(np.mean((vt_sim - sub["voltage_V"].to_numpy()) ** 2)) * 1000.0)
        t_m = sub["time_s"].to_numpy() / 60.0
        ax.plot(t_m, sub["voltage_V"], color="#1f77b4", linewidth=1.6, label="Deney (Cycling_1)")
        ax.plot(t_m, vt_sim, color="#d62728", linestyle="--", linewidth=1.4, label="1-RC Model")
        ax.set_title(f"{cid} ({STANFORD_CELL_CRATES[cid]}) — RMSE={rmse_mv:.1f} mV")
        ax.set_xlabel("Süre [dk]")
        ax.set_ylabel("Terminal Gerilimi [V]")
        ax.legend(fontsize=8)

    fig.suptitle("Kalibre Edilmiş 1-RC Modelin Cycling_1 (Çevrim 2) Şarj Gerilimi Doğrulaması", fontsize=12.5)
    fig.tight_layout()
    p3 = figures_dir / "fig04_cycling_charge_validation.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved.append(str(p3))

    return saved


def run_phase4_pipeline(
    raw_dir: str | Path,
    processed_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    calibration_cells: tuple[str, ...] = ("W4", "W8", "W9", "W10"),
) -> dict[str, Any]:
    """Aşama 4 parametre kestirimi, EIS çapraz kontrolü ve bağımsız doğrulama hattını çalıştırır."""
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    tbl_path = Path(tables_dir)
    fig_path = Path(figures_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    tbl_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    cap_fade_csv = proc_path / "stanford_capacity_fade_all_cells.csv"
    ocv_csv = proc_path / "stanford_ocv_c20_diag1.csv"
    ocv_fn, ocv_grid_df, q_mean, q_std = build_calibrated_ocv(ocv_csv, cap_fade_csv, calibration_cells=calibration_cells)
    ocv_grid_df.to_csv(proc_path / "stanford_ocv_calibrated.csv", index=False)

    cap_df = pd.read_csv(cap_fade_csv)
    diag1_caps = {
        str(r["cell_id"]): float(r["capacity_Ah"])
        for _, r in cap_df[cap_df["diag_number"] == 1].iterrows()
    }

    # HPPC kalibrasyon hücreleri üzerinde R0, R1, C1 kestirimi
    hppc_data = load_stanford_diagnostic(raw_path / "stanford" / "diagnostic" / "HPPC_test.mat", test_kind="hppc_test")
    cell_fits: list[dict[str, Any]] = []
    for cid in calibration_cells:
        df_h = extract_stanford_diag_df(hppc_data, cid, 1)
        fit = estimate_electrical_params_hppc(df_h, ocv_fn=ocv_fn, q_ah=diag1_caps[cid])
        fit["cell_id"] = cid
        fit["c_rate"] = STANFORD_CELL_CRATES[cid]
        fit["Q_Ah"] = diag1_caps[cid]
        cell_fits.append(fit)

    fit_df = pd.DataFrame(cell_fits)
    r0_mean = float(fit_df["R0_Ohm"].mean())
    r0_std = float(fit_df["R0_Ohm"].std(ddof=1))
    r1_mean = float(fit_df["R1_Ohm"].mean())
    r1_std = float(fit_df["R1_Ohm"].std(ddof=1))
    c1_mean = float(fit_df["C1_F"].mean())
    c1_std = float(fit_df["C1_F"].std(ddof=1))

    # EIS R0 çapraz doğrulama
    eis_df = extract_eis_r0_summary(raw_path / "stanford" / "diagnostic" / "EIS_test.mat", cells=("W8", "W9", "W10"), diag_number=1)

    # Termal kalibrasyon (W10 3C)
    df_therm = pd.read_csv(proc_path / "stanford_w10_thermal_sample.csv")
    therm_res = estimate_thermal_params_w10(df_therm, R0_Ohm=r0_mean, R1_Ohm=r1_mean, C1_F=c1_mean)

    # Kalibre BatteryParameters nesnesi (Tamb = 23.0 C Stanford kabin nominal değeri)
    cal_params = BatteryParameters(
        Q_Ah=q_mean,
        eta=0.998,
        R0_Ohm=r0_mean,
        R1_Ohm=r1_mean,
        C1_F=c1_mean,
        Rth_K_per_W=therm_res["Rth_K_per_W"],
        Cth_J_per_K=therm_res["Cth_J_per_K"],
        T_amb_C=23.0,
        is_calibrated=True,
        parameter_provenance={
            "Q_Ah": "stanford_capacity_test_diag1_mean_W4_W8_W9_W10",
            "eta": "assumed_coulombic_efficiency_0.998",
            "R0_Ohm": "stanford_hppc_diag1_nonlinear_lsq_mean",
            "R1_Ohm": "stanford_hppc_diag1_nonlinear_lsq_mean",
            "C1_F": "stanford_hppc_diag1_nonlinear_lsq_mean",
            "Rth_K_per_W": "stanford_w10_channel_6_2_thermal_lsq",
            "Cth_J_per_K": "stanford_w10_channel_6_2_thermal_lsq",
            "T_amb_C": "stanford_chamber_23C",
            "aging": "literature_empirical_relative_indicator",
        },
    )

    rep_chg_df = pd.read_csv(proc_path / "stanford_rep_charge_profiles.csv")
    val_df = validate_model_independent(
        params=cal_params,
        ocv_fn=ocv_fn,
        hppc_data=hppc_data,
        diag1_caps=diag1_caps,
        rep_chg_df=rep_chg_df,
        calibration_cells=calibration_cells,
    )
    val_df.to_csv(tbl_path / "phase4_validation_metrics.csv", index=False)

    # Parametre özet tablosu (hem kalibrasyon hücreleri hem ortalama/kaynak ayrımı)
    summary_rows = [
        {"parameter": "Q_Ah", "value": q_mean, "std_or_uncertainty": q_std, "unit": "Ah", "source": "capacity_test_Diag1 (W4,W8,W9,W10)", "type": "data_estimated"},
        {"parameter": "R0_Ohm", "value": r0_mean, "std_or_uncertainty": r0_std, "unit": "Ohm", "source": "HPPC_test_Diag1 LSQ (W4,W8,W9,W10)", "type": "data_estimated"},
        {"parameter": "R0_EIS_Ohm", "value": float(eis_df["R0_eis_Ohm"].mean()), "std_or_uncertainty": float(eis_df["R0_eis_Ohm"].std(ddof=1)), "unit": "Ohm", "source": "EIS_test_Diag1 Im(Z)~0 (W8,W9,W10)", "type": "cross_check"},
        {"parameter": "R1_Ohm", "value": r1_mean, "std_or_uncertainty": r1_std, "unit": "Ohm", "source": "HPPC_test_Diag1 LSQ (W4,W8,W9,W10)", "type": "data_estimated"},
        {"parameter": "C1_F", "value": c1_mean, "std_or_uncertainty": c1_std, "unit": "F", "source": "HPPC_test_Diag1 LSQ (W4,W8,W9,W10)", "type": "data_estimated"},
        {"parameter": "Rth_K_per_W", "value": therm_res["Rth_K_per_W"], "std_or_uncertainty": therm_res["Rth_K_per_W"] * 0.15, "unit": "K/W", "source": "W10_Channel_6.2 3C thermal LSQ", "type": "data_estimated"},
        {"parameter": "Cth_J_per_K", "value": therm_res["Cth_J_per_K"], "std_or_uncertainty": therm_res["Cth_J_per_K"] * 0.10, "unit": "J/K", "source": "W10_Channel_6.2 3C thermal LSQ", "type": "data_estimated"},
        {"parameter": "T_amb_C", "value": 23.0, "std_or_uncertainty": therm_res["T_amb_fit_C"] - 23.0, "unit": "degC", "source": "Amerex chamber nominal (fit=T_amb_fit)", "type": "experimental_condition"},
        {"parameter": "eta", "value": cal_params.eta, "std_or_uncertainty": 0.002, "unit": "-", "source": "Assumed Coulombic efficiency", "type": "assumed_literature"},
        {"parameter": "k0", "value": cal_params.k0, "std_or_uncertainty": 0.0, "unit": "1/s", "source": "Relative aging index (not absolute SOH)", "type": "literature_empirical"},
        {"parameter": "Ea_J_per_mol", "value": cal_params.Ea_J_per_mol, "std_or_uncertainty": 0.0, "unit": "J/mol", "source": "Literature Arrhenius activation energy", "type": "literature_empirical"},
        {"parameter": "p_crate", "value": cal_params.p_crate, "std_or_uncertainty": 0.0, "unit": "-", "source": "Literature C-rate stress exponent", "type": "literature_empirical"},
    ]
    param_table = pd.DataFrame(summary_rows)
    param_table.to_csv(tbl_path / "phase4_parameter_estimation.csv", index=False)

    # JSON kaydet
    payload = {
        "parameters": {
            "Q_Ah": cal_params.Q_Ah,
            "eta": cal_params.eta,
            "R0_Ohm": cal_params.R0_Ohm,
            "R1_Ohm": cal_params.R1_Ohm,
            "C1_F": cal_params.C1_F,
            "Rth_K_per_W": cal_params.Rth_K_per_W,
            "Cth_J_per_K": cal_params.Cth_J_per_K,
            "T_amb_C": cal_params.T_amb_C,
            "k0": cal_params.k0,
            "Ea_J_per_mol": cal_params.Ea_J_per_mol,
            "p_crate": cal_params.p_crate,
            "alpha_soc": cal_params.alpha_soc,
            "Rg_J_per_mol_K": cal_params.Rg_J_per_mol_K,
            "is_calibrated": True,
        },
        "uncertainties_std": {
            "Q_Ah": q_std,
            "R0_Ohm": r0_std,
            "R1_Ohm": r1_std,
            "C1_F": c1_std,
            "Rth_K_per_W": therm_res["Rth_K_per_W"] * 0.15,
            "Cth_J_per_K": therm_res["Cth_J_per_K"] * 0.10,
        },
        "provenance": cal_params.parameter_provenance,
        "thermal_fit_metrics": {
            "T_amb_fit_C": therm_res["T_amb_fit_C"],
            "rmse_C": therm_res["rmse_C"],
            "mae_C": therm_res["mae_C"],
            "heating_rmse_C": therm_res["heating_rmse_C"],
            "cooling_rmse_C": therm_res["cooling_rmse_C"],
            "r2": therm_res["r2"],
        },
        "per_cell_hppc_fits": fit_df[
            ["cell_id", "c_rate", "Q_Ah", "R0_Ohm", "R0_pulse_Ohm", "R1_Ohm", "C1_F", "tau1_s", "rmse_V", "r2"]
        ].to_dict(orient="records"),
        "eis_r0_summary": eis_df.to_dict(orient="records"),
        "ocv_soc_grid": ocv_grid_df["soc"].tolist(),
        "ocv_V_grid": ocv_grid_df["ocv_V"].tolist(),
    }
    with open(proc_path / "calibrated_parameters.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    saved_figs = _generate_phase4_figures(
        hppc_data=hppc_data,
        ocv_fn=ocv_fn,
        params=cal_params,
        diag1_caps=diag1_caps,
        df_therm=df_therm,
        therm_res=therm_res,
        rep_chg_df=rep_chg_df,
        figures_dir=fig_path,
    )

    return {
        "calibrated_parameters": cal_params,
        "parameter_table": param_table,
        "per_cell_fits": fit_df,
        "eis_summary": eis_df,
        "validation_metrics": val_df,
        "thermal_fit": therm_res,
        "saved_figures": saved_figs,
    }
