"""Aşama 2: Veri temizleme, şarj/deşarj döngüsü ayrıştırma, SOC hesabı ve görselleştirme."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from src.data_loader import (
    STANFORD_CELL_CRATES,
    extract_nasa_cycle_df,
    extract_stanford_diag_df,
    load_nasa_mat,
    load_stanford_cycling,
    load_stanford_diagnostic,
    load_stanford_raw_xlsx,
)

CHARGE_STEPS = {2, 3, 7, 8, 9, 10}
AGING_CHARGE_STEPS = {7, 8, 9, 10}
REST_STEPS = {1, 4, 11}
DISCHARGE_STEPS = {13, 14}


def _strictly_increasing_mask(time_vals: np.ndarray) -> np.ndarray:
    """Zaman vektörünün kesin artan kalmasını sağlayan maskeyi döndürür."""
    n = len(time_vals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(n, dtype=bool)
    prev_max = np.maximum.accumulate(time_vals)
    mask[1:] = time_vals[1:] > prev_max[:-1]
    return mask


def clean_stanford_cycling(
    df: pd.DataFrame,
    v_bounds: tuple[float, float] = (2.50, 4.25),
    temp_col: str | None = None,
    t_bounds: tuple[float, float] = (0.0, 60.0),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stanford çevrim verisinden Step 0/0V artifaktlarını, NaN'ları ve monotonluk dışı satırları temizler."""
    n_initial = int(len(df))
    core_cols = ["time_s", "voltage_V", "current_A", "step_index"]
    if temp_col and temp_col in df.columns:
        core_cols.append(temp_col)

    nan_mask = df[core_cols].isna().any(axis=1).to_numpy()
    step0_mask = (df["step_index"].to_numpy() == 0)
    v_vals = df["voltage_V"].to_numpy()
    v_zero_mask = (v_vals <= 0.0)
    v_oob_mask = (v_vals < v_bounds[0]) | (v_vals > v_bounds[1])

    temp_oob_mask = np.zeros(n_initial, dtype=bool)
    if temp_col and temp_col in df.columns:
        t_vals = df[temp_col].to_numpy()
        temp_oob_mask = (t_vals < t_bounds[0]) | (t_vals > t_bounds[1])

    keep_mask = ~(nan_mask | step0_mask | v_oob_mask | temp_oob_mask)
    df_filtered = df.loc[keep_mask].copy()

    inc_mask = _strictly_increasing_mask(df_filtered["time_s"].to_numpy())
    n_non_mono = int((~inc_mask).sum())
    df_clean = df_filtered.loc[inc_mask].reset_index(drop=True)

    report = {
        "n_initial": n_initial,
        "removed_nan": int(nan_mask.sum()),
        "removed_step0": int(step0_mask.sum()),
        "removed_v_zero": int(v_zero_mask.sum()),
        "removed_v_oob": int(v_oob_mask.sum()),
        "removed_temp_oob": int(temp_oob_mask.sum()),
        "removed_non_monotonic": n_non_mono,
        "n_clean": int(len(df_clean)),
    }
    return df_clean, report


def segment_stanford_cycles(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step_Index geçişlerine göre çevrim numarası ve faz ('charge', 'rest', 'discharge') etiketler."""
    steps = df_clean["step_index"].to_numpy()
    is_step7_start = np.zeros(len(steps), dtype=np.int32)
    if len(steps) > 0:
        is_step7_start[0] = 1 if steps[0] == 7 else 0
        is_step7_start[1:] = ((steps[1:] == 7) & (steps[:-1] != 7)).astype(np.int32)
    cycle_nums = np.cumsum(is_step7_start, dtype=np.int32)

    phase = np.full(len(steps), "other", dtype=object)
    phase[np.isin(steps, list(CHARGE_STEPS))] = "charge"
    phase[np.isin(steps, list(REST_STEPS))] = "rest"
    phase[np.isin(steps, list(DISCHARGE_STEPS))] = "discharge"

    df_tagged = df_clean.copy()
    df_tagged["cycle_num"] = cycle_nums
    df_tagged["phase"] = phase

    time_arr = df_tagged["time_s"].to_numpy()
    curr_arr = df_tagged["current_A"].to_numpy()
    volt_arr = df_tagged["voltage_V"].to_numpy()

    summary_rows: list[dict[str, Any]] = []
    for cnum in np.unique(cycle_nums):
        if cnum == 0:
            continue
        cmask = (cycle_nums == cnum)
        chg_mask = cmask & np.isin(steps, list(AGING_CHARGE_STEPS))
        s7_mask = cmask & (steps == 7)
        s8_mask = cmask & (steps == 8)
        s9_mask = cmask & (steps == 9)
        s10_mask = cmask & (steps == 10)
        dis_mask = cmask & np.isin(steps, list(DISCHARGE_STEPS))

        t_chg = time_arr[chg_mask]
        i_chg = curr_arr[chg_mask]
        charge_ah = float(np.trapezoid(i_chg, t_chg) / 3600.0) if len(t_chg) > 1 else 0.0

        t_dis = time_arr[dis_mask]
        i_dis = curr_arr[dis_mask]
        discharge_ah = float(np.trapezoid(-i_dis, t_dis) / 3600.0) if len(t_dis) > 1 else 0.0

        def _dur(m: np.ndarray) -> float:
            tm = time_arr[m]
            return float(tm[-1] - tm[0]) if len(tm) > 1 else 0.0

        summary_rows.append(
            {
                "cycle_num": int(cnum),
                "charge_duration_s": _dur(chg_mask),
                "step7_cc_duration_s": _dur(s7_mask),
                "step8_cv_4v_duration_s": _dur(s8_mask),
                "step9_cc_c4_duration_s": _dur(s9_mask),
                "step10_cv_42v_duration_s": _dur(s10_mask),
                "discharge_duration_s": _dur(dis_mask),
                "charge_Ah": charge_ah,
                "discharge_Ah": discharge_ah,
                "i_max_A": float(np.max(i_chg)) if len(i_chg) > 0 else np.nan,
                "v_min_V": float(np.min(volt_arr[cmask])),
                "v_max_V": float(np.max(volt_arr[cmask])),
            }
        )

    return df_tagged, pd.DataFrame(summary_rows)


def extract_stanford_charge_cycle(
    df_tagged: pd.DataFrame,
    cycle_num: int,
    include_rest: bool = False,
    resample_dt_s: float | None = 1.0,
) -> pd.DataFrame:
    """Etiketli veriden tek bir yaşlandırma şarj çevrimini (Steps 7-10) çıkarır ve opsiyonel 1 Hz örnekler."""
    steps_target = set(AGING_CHARGE_STEPS) | ({11} if include_rest else set())
    mask = (df_tagged["cycle_num"] == cycle_num) & (df_tagged["step_index"].isin(steps_target))
    sub = df_tagged.loc[mask].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"Çevrim {cycle_num} için şarj adımı bulunamadı.")

    sub["time_s"] = sub["time_s"] - float(sub["time_s"].iloc[0])
    if resample_dt_s is None:
        return sub

    t_orig = sub["time_s"].to_numpy()
    t_grid = np.arange(0.0, float(t_orig[-1]) + 1e-9, float(resample_dt_s))
    out = pd.DataFrame({"time_s": t_grid})
    for col in ("current_A", "voltage_V", "charge_capacity_Ah", "temperature_C"):
        if col in sub.columns:
            out[col] = np.interp(t_grid, t_orig, sub[col].to_numpy())

    # step_index için en yakın önceki örnek eşlemesi
    idx = np.clip(np.searchsorted(t_orig, t_grid, side="right") - 1, 0, len(t_orig) - 1)
    out["step_index"] = sub["step_index"].to_numpy()[idx]
    out["cycle_num"] = int(cycle_num)
    return out


def compute_soc_from_capacity_test(df_cap: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """C/20 kapasite testinden kaydedilen ve Coulomb sayımıyla hesaplanan SOC eğrilerini karşılaştırır."""
    df = df_cap.copy().reset_index(drop=True)
    q_logged = float(df["capacity_Ah"].max())
    t = df["time_s"].to_numpy(dtype=np.float64)
    i_dis = -df["current_A"].to_numpy(dtype=np.float64)  # Deşarj akımı negatif olduğu için -I
    q_cum_coulomb = np.concatenate([[0.0], cumulative_trapezoid(i_dis, t) / 3600.0])
    q_coulomb = float(q_cum_coulomb[-1])

    soc_logged = np.clip(1.0 - df["capacity_Ah"].to_numpy(dtype=np.float64) / q_logged, 0.0, 1.0)
    soc_coulomb = np.clip(1.0 - q_cum_coulomb / q_coulomb, 0.0, 1.0)

    df["soc_logged"] = soc_logged
    df["soc_coulomb"] = soc_coulomb
    df["soc"] = soc_coulomb

    verification = {
        "Q_logged_Ah": q_logged,
        "Q_coulomb_Ah": q_coulomb,
        "Q_rel_diff_pct": 100.0 * abs(q_coulomb - q_logged) / max(q_logged, 1e-12),
        "max_abs_soc_diff": float(np.max(np.abs(soc_logged - soc_coulomb))),
        "rmse_soc_diff": float(np.sqrt(np.mean((soc_logged - soc_coulomb) ** 2))),
    }
    return df, verification


def compute_soc_charge_cycle(
    df_chg: pd.DataFrame,
    capacity_Ah: float,
    soc_end: float = 1.0,
    eta: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """CV sonu tam doluluk (soc_end=1.0) referansından geriye doğru Coulomb sayımıyla SOC(t) hesaplar."""
    df = df_chg.copy().reset_index(drop=True)
    t = df["time_s"].to_numpy(dtype=np.float64)
    curr = df["current_A"].to_numpy(dtype=np.float64)
    q_cum = np.concatenate([[0.0], cumulative_trapezoid(curr, t) / 3600.0])
    q_total = float(q_cum[-1])
    q_rem = q_total - q_cum
    soc = soc_end - (eta / float(capacity_Ah)) * q_rem
    soc_clipped = np.clip(soc, 0.0, 1.0)
    df["soc"] = soc_clipped

    stats = {
        "charge_Ah": q_total,
        "soc_start": float(soc_clipped[0]),
        "soc_end": float(soc_clipped[-1]),
        "delta_soc": float(soc_clipped[-1] - soc_clipped[0]),
        "soc_out_of_bounds_raw": int(np.sum((soc < -1e-4) | (soc > 1.0 + 1e-4))),
    }
    return df, stats


def clean_nasa_cycle(
    df_cycle: pd.DataFrame,
    v_bounds: tuple[float, float] = (1.80, 4.35),
    t_bounds: tuple[float, float] = (0.0, 65.0),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """NASA döngü verisinden NaN değerleri, gerilim/sıcaklık sıçramalarını ve monotonluk dışı zamanları temizler."""
    n_initial = int(len(df_cycle))
    core_cols = ["time_s", "voltage_V", "current_A", "temperature_C"]
    nan_mask = df_cycle[core_cols].isna().any(axis=1).to_numpy()
    v_vals = df_cycle["voltage_V"].to_numpy()
    t_vals = df_cycle["temperature_C"].to_numpy()
    v_oob_mask = (v_vals < v_bounds[0]) | (v_vals > v_bounds[1])
    t_oob_mask = (t_vals < t_bounds[0]) | (t_vals > t_bounds[1])

    keep_mask = ~(nan_mask | v_oob_mask | t_oob_mask)
    df_f = df_cycle.loc[keep_mask].copy()
    inc_mask = _strictly_increasing_mask(df_f["time_s"].to_numpy())
    df_clean = df_f.loc[inc_mask].reset_index(drop=True)

    return df_clean, {
        "n_initial": n_initial,
        "removed_nan": int(nan_mask.sum()),
        "removed_v_oob": int(v_oob_mask.sum()),
        "removed_temp_oob": int(t_oob_mask.sum()),
        "removed_non_monotonic": int((~inc_mask).sum()),
        "n_clean": int(len(df_clean)),
    }


def compute_nasa_charge_soc(
    df_chg_clean: pd.DataFrame,
    discharge_capacity_Ah: float,
    soc_end: float = 1.0,
) -> pd.DataFrame:
    """Temizlenmiş NASA şarj döngüsü için geriye doğru Coulomb sayımıyla SOC hesaplar."""
    df_out, _ = compute_soc_charge_cycle(df_chg_clean, capacity_Ah=discharge_capacity_Ah, soc_end=soc_end, eta=1.0)
    return df_out


def generate_phase2_figures(
    ocv_df: pd.DataFrame,
    cap_fade_df: pd.DataFrame,
    rep_chg_df: pd.DataFrame,
    w10_therm_df: pd.DataFrame,
    nasa_chg_df: pd.DataFrame,
    nasa_imp_cap_df: pd.DataFrame,
    figures_dir: str | Path,
) -> list[str]:
    """Aşama 2 için tüm birimli, yüksek çözünürlüklü PNG grafiklerini üretir."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    saved_paths: list[str] = []

    # 1. Stanford C/20 OCV ve Kapasite Yaşlanma Grafiği
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"W4": "#1f77b4", "W8": "#2ca02c", "W9": "#ff7f0e", "W10": "#d62728", "W5": "#9467bd", "V4": "#8c564b"}
    for cid in ["W4", "W8", "W9", "W10", "W5", "V4"]:
        sub = ocv_df[ocv_df["cell_id"] == cid]
        ls = "--" if cid in ("W5", "V4") else "-"
        lbl = f"{cid} ({STANFORD_CELL_CRATES[cid]})" + (" [Doğrulama]" if cid in ("W5", "V4") else "")
        axes[0].plot(sub["soc"], sub["voltage_V"], linestyle=ls, linewidth=1.8, color=colors[cid], label=lbl)
    axes[0].set_title("Stanford INR21700-M50T: C/20 (Diag_1) Gerilim–SOC Eğrisi")
    axes[0].set_xlabel("SOC [-]")
    axes[0].set_ylabel("Terminal Gerilimi [V]")
    axes[0].legend(fontsize=8)

    for cid, grp in cap_fade_df.groupby("cell_id"):
        axes[1].plot(grp["diag_number"], grp["capacity_Ah"], marker="o", markersize=4, linewidth=1.5, label=f"{cid} ({grp['c_rate'].iloc[0]})")
    axes[1].set_title("Stanford Tüm Hücreler: Tanı Testi Kapasite Düşüşü")
    axes[1].set_xlabel("Tanı Testi İndeksi (Diag_#)")
    axes[1].set_ylabel("Deşarj Kapasitesi [Ah]")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    p1 = fig_dir / "fig02_stanford_c20_ocv_and_aging.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved_paths.append(str(p1))

    # 2. Stanford CC-CV Şarj Protokolleri (W4, W8, W9, W10 - Çevrim 2)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for cid in ["W4", "W8", "W9", "W10"]:
        sub = rep_chg_df[rep_chg_df["cell_id"] == cid]
        t_min = sub["time_s"] / 60.0
        lbl = f"{cid} ({STANFORD_CELL_CRATES[cid]})"
        axes[0].plot(t_min, sub["current_A"], linewidth=1.7, color=colors[cid], label=lbl)
        axes[1].plot(t_min, sub["voltage_V"], linewidth=1.7, color=colors[cid], label=lbl)
        axes[2].plot(t_min, sub["soc"], linewidth=1.7, color=colors[cid], label=lbl)

    axes[0].set_title("Stanford Cycling_1 (Çevrim 2): Farklı C-Oranlarında Şarj Profilleri")
    axes[0].set_ylabel("Akım [A]")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].set_ylabel("Terminal Gerilimi [V]")
    axes[1].axhline(4.20, color="gray", linestyle=":", linewidth=1.0, label="V_max = 4.20 V")
    axes[2].set_ylabel("SOC [-]")
    axes[2].set_xlabel("Süre [dk]")
    fig.tight_layout()
    p2 = fig_dir / "fig02_stanford_charge_protocols.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved_paths.append(str(p2))

    # 3. Stanford W10 (3C) Ham Sıcaklık Ölçüm Eğrisi
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    t_min_w10 = (w10_therm_df["time_s"] - w10_therm_df["time_s"].iloc[0]) / 60.0
    axes[0].plot(t_min_w10, w10_therm_df["current_A"], color="#d62728", linewidth=1.4)
    axes[0].set_title("Stanford W10 (3C Şarj + 30 dk Dinlenme) Termal Davranış (Channel_6.2)")
    axes[0].set_ylabel("Akım [A]")
    axes[1].plot(t_min_w10, w10_therm_df["voltage_V"], color="#1f77b4", linewidth=1.4)
    axes[1].set_ylabel("Gerilim [V]")
    axes[2].plot(t_min_w10, w10_therm_df["temperature_C"], color="#ff7f0e", linewidth=1.8)
    axes[2].set_ylabel("Yüzey Sıcaklığı [°C]")
    axes[2].set_xlabel("Süre [dk]")
    fig.tight_layout()
    p3 = fig_dir / "fig02_stanford_w10_temperature.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved_paths.append(str(p3))

    # 4. NASA 18650 Bağımsız Yardımcı Veri Seti Grafiği
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    nasa_colors = {"B0005": "#1f77b4", "B0029": "#d62728", "B0053": "#17becf"}
    for cid, grp in nasa_chg_df.groupby("cell_id"):
        t_m = grp["time_s"] / 60.0
        lbl = f"{cid} (T_amb={int(grp['ambient_temperature_C'].iloc[0])} °C)"
        axes[0, 0].plot(t_m, grp["voltage_V"], linewidth=1.6, color=nasa_colors.get(cid, "k"), label=lbl)
        axes[0, 1].plot(t_m, grp["temperature_C"], linewidth=1.6, color=nasa_colors.get(cid, "k"), label=lbl)

    axes[0, 0].set_title("NASA 18650 (Bağımsız): Şarj Gerilimi")
    axes[0, 0].set_xlabel("Süre [dk]")
    axes[0, 0].set_ylabel("Gerilim [V]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("NASA 18650 (Bağımsız): Şarj Hücre Sıcaklığı")
    axes[0, 1].set_xlabel("Süre [dk]")
    axes[0, 1].set_ylabel("Sıcaklık [°C]")
    axes[0, 1].legend(fontsize=8)

    cap_sub = nasa_imp_cap_df[nasa_imp_cap_df["record_type"] == "capacity"]
    for cid, grp in cap_sub.groupby("cell_id"):
        axes[1, 0].plot(grp["cycle_index"], grp["value"], linewidth=1.5, label=cid)
    axes[1, 0].set_title("NASA 18650: Çevrim Bazlı Kapasite Kaybı")
    axes[1, 0].set_xlabel("Döngü İndeksi [-]")
    axes[1, 0].set_ylabel("Deşarj Kapasitesi [Ah]")
    axes[1, 0].legend(fontsize=8)

    imp_sub = nasa_imp_cap_df[(nasa_imp_cap_df["record_type"] == "Re") & (nasa_imp_cap_df["cell_id"].isin(["B0005", "B0006", "B0007", "B0018"]))]
    for cid, grp in imp_sub.groupby("cell_id"):
        axes[1, 1].plot(grp["cycle_index"], grp["value"], linewidth=1.4, label=f"{cid} Re")
    axes[1, 1].set_title("NASA 18650: Ohmik İç Direnç (Re) Artışı (24 °C)")
    axes[1, 1].set_xlabel("Döngü İndeksi [-]")
    axes[1, 1].set_ylabel("Re [Ohm]")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    p4 = fig_dir / "fig02_nasa_charge_thermal_aging.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved_paths.append(str(p4))

    return saved_paths


def run_phase2_pipeline(
    raw_dir: str | Path,
    processed_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, Any]:
    """Tüm Aşama 2 veri temizleme, ayrıştırma, SOC doğrulama ve görselleştirme hattını yürütür."""
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    tbl_path = Path(tables_dir)
    fig_path = Path(figures_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    tbl_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    cleaning_reports: list[dict[str, Any]] = []

    # 1. Stanford C/20 Kapasite Testleri (Diag_1..15 ve Diag_1 OCV)
    cap_diag = load_stanford_diagnostic(raw_path / "stanford" / "diagnostic" / "capacity_test.mat")
    cap_fade_rows: list[dict[str, Any]] = []
    ocv_frames: list[pd.DataFrame] = []
    diag1_capacities: dict[str, float] = {}

    for cid in cap_diag["cell_labels"]:
        for dnum in cap_diag["diag_numbers"]:
            try:
                df_c = extract_stanford_diag_df(cap_diag, cid, dnum)
            except ValueError:
                continue
            df_soc, ver = compute_soc_from_capacity_test(df_c)
            cap_fade_rows.append(
                {
                    "cell_id": cid,
                    "c_rate": STANFORD_CELL_CRATES[cid],
                    "diag_number": int(dnum),
                    "capacity_Ah": ver["Q_logged_Ah"],
                    "capacity_coulomb_Ah": ver["Q_coulomb_Ah"],
                    "soc_max_diff": ver["max_abs_soc_diff"],
                }
            )
            if dnum == 1:
                diag1_capacities[cid] = ver["Q_logged_Ah"]
                if cid in ("W4", "W8", "W9", "W10", "W5", "V4"):
                    ds = df_soc.iloc[::60][["time_s", "voltage_V", "current_A", "capacity_Ah", "soc", "soc_logged"]].copy()
                    ds["cell_id"] = cid
                    ocv_frames.append(ds)
                    cleaning_reports.append(
                        {
                            "dataset": "stanford",
                            "source": f"capacity_test_Diag1_{cid}",
                            "n_initial": len(df_c),
                            "n_clean": len(df_c),
                            "removed_total": 0,
                            "soc_max_diff": ver["max_abs_soc_diff"],
                            "notes": f"Q_logged={ver['Q_logged_Ah']:.4f} Ah, Q_coulomb={ver['Q_coulomb_Ah']:.4f} Ah",
                        }
                    )

    cap_fade_df = pd.DataFrame(cap_fade_rows)
    cap_fade_df.to_csv(proc_path / "stanford_capacity_fade_all_cells.csv", index=False)
    ocv_df = pd.concat(ocv_frames, ignore_index=True)
    ocv_df.to_csv(proc_path / "stanford_ocv_c20_diag1.csv", index=False)

    # 2. Stanford Cycling_1 Temizleme ve Şarj Döngüsü Ayrıştırma (W4, W8, W9, W10)
    cyc_summaries: list[pd.DataFrame] = []
    rep_chg_frames: list[pd.DataFrame] = []
    for cid in ("W4", "W8", "W9", "W10"):
        raw_cyc = load_stanford_cycling(raw_path / "stanford" / "cycling" / "Cycling_1" / f"{cid}.mat")
        df_clean, rep = clean_stanford_cycling(raw_cyc["dataframe"], v_bounds=(2.50, 4.25))
        df_tagged, csum = segment_stanford_cycles(df_clean)
        csum["cell_id"] = cid
        csum["c_rate"] = STANFORD_CELL_CRATES[cid]
        cyc_summaries.append(csum)

        # Çevrim 2 temiz 1 Hz şarj eğrisi ve SOC hesabı
        df_c2 = extract_stanford_charge_cycle(df_tagged, cycle_num=2, include_rest=False, resample_dt_s=1.0)
        df_c2_soc, soc_stats = compute_soc_charge_cycle(df_c2, capacity_Ah=diag1_capacities[cid], soc_end=1.0, eta=1.0)
        df_c2_soc["cell_id"] = cid
        df_c2_soc["c_rate"] = STANFORD_CELL_CRATES[cid]
        rep_chg_frames.append(df_c2_soc)

        cleaning_reports.append(
            {
                "dataset": "stanford",
                "source": f"Cycling_1_{cid}",
                "n_initial": rep["n_initial"],
                "n_clean": rep["n_clean"],
                "removed_total": rep["n_initial"] - rep["n_clean"],
                "soc_max_diff": np.nan,
                "notes": (
                    f"cycles={len(csum)}, step0_removed={rep['removed_step0']}, "
                    f"Cycle2 soc_start={soc_stats['soc_start']:.4f}, Q_chg={soc_stats['charge_Ah']:.4f} Ah"
                ),
            }
        )

    pd.concat(cyc_summaries, ignore_index=True).to_csv(proc_path / "stanford_charge_cycles_summary.csv", index=False)
    rep_chg_df = pd.concat(rep_chg_frames, ignore_index=True)
    rep_chg_df.to_csv(proc_path / "stanford_rep_charge_profiles.csv", index=False)

    # 3. Stanford Ham W10 Sıcaklık Verisi (Channel_6.2.xlsx)
    xlsx_path = raw_path / "stanford" / "raw_temp" / "INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.2.xlsx"
    w10_raw = load_stanford_raw_xlsx(xlsx_path, cell_id="W10")
    w10_clean, w10_rep = clean_stanford_cycling(w10_raw["dataframe"], v_bounds=(2.50, 4.25), temp_col="temperature_C", t_bounds=(0.0, 60.0))
    w10_tagged, _ = segment_stanford_cycles(w10_clean)
    # 3C şarj + 30 dk dinlenme segmenti (Steps 7-11) 1 Hz olarak dışa aktarılır
    w10_therm_1hz = extract_stanford_charge_cycle(w10_tagged, cycle_num=1, include_rest=True, resample_dt_s=1.0)
    w10_therm_1hz, w10_soc_stats = compute_soc_charge_cycle(w10_therm_1hz, capacity_Ah=diag1_capacities["W10"], soc_end=1.0, eta=1.0)
    w10_therm_1hz.to_csv(proc_path / "stanford_w10_thermal_sample.csv", index=False)
    cleaning_reports.append(
        {
            "dataset": "stanford",
            "source": "W10_Channel_6.2_thermal",
            "n_initial": w10_rep["n_initial"],
            "n_clean": w10_rep["n_clean"],
            "removed_total": w10_rep["n_initial"] - w10_rep["n_clean"],
            "soc_max_diff": np.nan,
            "notes": f"T=[{w10_therm_1hz['temperature_C'].min():.2f}, {w10_therm_1hz['temperature_C'].max():.2f}] C, 1Hz rows={len(w10_therm_1hz)}",
        }
    )

    # 4. NASA 18650 Temizleme ve Temsili Şarj Eğrileri
    nasa_imp_cap_rows: list[dict[str, Any]] = []
    nasa_rep_frames: list[pd.DataFrame] = []
    for fn in ("B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat", "B0029.mat", "B0053.mat"):
        nasa = load_nasa_mat(raw_path / "nasa" / fn)
        cid = nasa["cell_id"]
        for _, r in nasa["capacity_fade"].iterrows():
            nasa_imp_cap_rows.append({"cell_id": cid, "record_type": "capacity", "cycle_index": int(r["cycle_index"]), "value": float(r["capacity_Ah"])})
        for _, r in nasa["impedance_summary"].iterrows():
            nasa_imp_cap_rows.append({"cell_id": cid, "record_type": "Re", "cycle_index": int(r["cycle_index"]), "value": float(r["Re_Ohm"])})
            nasa_imp_cap_rows.append({"cell_id": cid, "record_type": "Rct", "cycle_index": int(r["cycle_index"]), "value": float(r["Rct_Ohm"])})

        removed_all = 0
        initial_all = 0
        for idx, cyc in enumerate(nasa["cycles"]):
            if cyc["type"] in ("charge", "discharge"):
                df_raw_c = extract_nasa_cycle_df(nasa, idx)
                _, crep = clean_nasa_cycle(df_raw_c)
                initial_all += crep["n_initial"]
                removed_all += crep["n_initial"] - crep["n_clean"]

        cleaning_reports.append(
            {
                "dataset": "nasa_18650",
                "source": cid,
                "n_initial": initial_all,
                "n_clean": initial_all - removed_all,
                "removed_total": removed_all,
                "soc_max_diff": np.nan,
                "notes": f"Q_init={nasa['capacity_fade']['capacity_Ah'].iloc[0]:.4f} Ah",
            }
        )

        if cid in ("B0005", "B0029", "B0053"):
            chg_indices = nasa["cycle_summary"].query("type == 'charge'")["cycle_index"].tolist()
            # İlk tam CC-CV şarjı al (B0005 için index 2 veya ilk >50 örnekli şarj)
            chosen_idx = next(i for i in chg_indices if len(extract_nasa_cycle_df(nasa, int(i))) > 50)
            df_n_chg, _ = clean_nasa_cycle(extract_nasa_cycle_df(nasa, int(chosen_idx)))
            q_ref = float(nasa["capacity_fade"]["capacity_Ah"].iloc[0])
            df_n_chg = compute_nasa_charge_soc(df_n_chg, discharge_capacity_Ah=q_ref, soc_end=1.0)
            df_n_chg["cell_id"] = cid
            df_n_chg["ambient_temperature_C"] = float(nasa["cycles"][int(chosen_idx)]["ambient_temperature"])
            nasa_rep_frames.append(df_n_chg)

    nasa_imp_cap_df = pd.DataFrame(nasa_imp_cap_rows)
    nasa_imp_cap_df.to_csv(proc_path / "nasa_capacity_impedance_clean.csv", index=False)
    nasa_chg_df = pd.concat(nasa_rep_frames, ignore_index=True)
    nasa_chg_df.to_csv(proc_path / "nasa_rep_charge_cycles.csv", index=False)

    report_df = pd.DataFrame(cleaning_reports)
    report_df.to_csv(tbl_path / "phase2_cleaning_report.csv", index=False)

    # 5. Grafikleri üret
    saved_figs = generate_phase2_figures(
        ocv_df=ocv_df,
        cap_fade_df=cap_fade_df,
        rep_chg_df=rep_chg_df,
        w10_therm_df=w10_therm_1hz,
        nasa_chg_df=nasa_chg_df,
        nasa_imp_cap_df=nasa_imp_cap_df,
        figures_dir=fig_path,
    )

    return {
        "cleaning_report": report_df,
        "saved_figures": saved_figs,
        "diag1_capacities": diag1_capacities,
    }
