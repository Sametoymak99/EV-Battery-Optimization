"""Stanford (INR21700-M50T) ve NASA (18650) batarya veri setleri için yükleme ve şema denetim modülü."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.io

# --- Veri Seti Sabitleri ve Doğrulanmış Meta Veriler ---

STANFORD_DATASET_ID = "stanford_inr21700_m50t"
NASA_DATASET_ID = "nasa_18650"

STANFORD_NOMINAL_CAPACITY_AH = 4.85
NASA_NOMINAL_CAPACITY_AH = 2.00

STANFORD_CELL_CRATES: dict[str, str] = {
    "W3": "3C",
    "W4": "C/4",
    "W5": "C/2",
    "W7": "C/4",
    "W8": "C/2",
    "W9": "1C",
    "W10": "3C",
    "G1": "3C",
    "V4": "C/4",
    "V5": "1C",
}

STANFORD_OSF_URL = "https://osf.io/qsabn/"
STANFORD_VIEW_ONLY_TOKEN = "2a03b6c78ef14922a3e244f3d549de78"
STANFORD_WATERBUTLER_BASE = "https://files.osf.io/v1/resources/qsabn/providers/dropbox"
STANFORD_DROPBOX_FALLBACK_URL = (
    "https://www.dropbox.com/scl/fo/3ss0age6ggfcm67okldhw/h"
    "?rlkey=tnczvb82gukfe2n4gol2uyo7x&dl=0"
)
NASA_S3_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

STANFORD_CYCLING_KEY_MAP: dict[str, str] = {
    "t_full_vec_M1_NMC25degC": "time_s",
    "I_full_vec_M1_NMC25degC": "current_A",
    "V_full_vec_M1_NMC25degC": "voltage_V",
    "Step_Index_full_vec_M1_NMC25degC": "step_index",
    "ch_cap_full_vec_M1_NMC25degC": "charge_capacity_Ah",
    "dis_cap_full_vec_M1_NMC25degC": "discharge_capacity_Ah",
}


def _convert_h5_group(group: h5py.Group) -> dict[str, Any]:
    """HDF5 (MATLAB v7.3) grubunu özyinelemeli olarak Python sözlüğüne dönüştürür."""
    out: dict[str, Any] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            out[key] = _convert_h5_group(item)
        else:
            out[key] = np.asarray(item)
    return out


def load_mat_file(filepath: str | Path) -> dict[str, Any]:
    """Klasik MATLAB (.mat) veya v7.3 HDF5 .mat dosyasını sözlük olarak yükler."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    try:
        raw = scipy.io.loadmat(str(path), simplify_cells=True)
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        with h5py.File(str(path), "r") as f:
            return _convert_h5_group(f)


def load_stanford_diagnostic(
    filepath: str | Path,
    test_kind: str | None = None,
) -> dict[str, Any]:
    """Stanford tanı test dosyasını (capacity_test.mat, HPPC_test.mat veya EIS_test.mat) yükler."""
    path = Path(filepath)
    raw = load_mat_file(path)

    if test_kind is None:
        name_lower = path.name.lower()
        if "eis" in name_lower or "re_z" in raw:
            test_kind = "eis_test"
        elif "hppc" in name_lower:
            test_kind = "hppc_test"
        elif "capacity" in name_lower:
            test_kind = "capacity_test"
        else:
            raise ValueError(f"Tanı test türü belirlenemedi: {path.name}")

    cell_labels = [str(c) for c in raw["col_cell_label"]]
    diag_numbers = [int(d) for d in np.atleast_1d(raw["row_diag_number"])]

    units = (
        {"freq": "Hz", "re_z": "Ohm", "im_z": "Ohm", "soc_level": "%"}
        if test_kind == "eis_test"
        else {"time": "s", "vcell": "V", "curr": "A", "cap": "Ah"}
    )

    return {
        "dataset": STANFORD_DATASET_ID,
        "filepath": str(path),
        "test_kind": test_kind,
        "nominal_capacity_Ah": STANFORD_NOMINAL_CAPACITY_AH,
        "current_sign_convention": "positive_charge_negative_discharge",
        "cell_labels": cell_labels,
        "diag_numbers": diag_numbers,
        "units": units,
        "raw": raw,
    }


def extract_stanford_diag_df(
    diag_data: dict[str, Any],
    cell_id: str,
    diag_number: int,
) -> pd.DataFrame:
    """Yüklenmiş Stanford tanı yapısından belirli bir hücre ve tanı numarası için DataFrame çıkarır."""
    labels = diag_data["cell_labels"]
    diags = diag_data["diag_numbers"]
    if cell_id not in labels:
        raise KeyError(f"Hücre bulunamadı: {cell_id}. Mevcutlar: {labels}")
    if diag_number not in diags:
        raise KeyError(f"Tanı numarası bulunamadı: {diag_number}. Mevcutlar: {diags}")

    ci = labels.index(cell_id)
    ri = diags.index(diag_number)
    raw = diag_data["raw"]
    kind = diag_data["test_kind"]

    if kind == "eis_test":
        re_arr = raw["re_z"][ri, ci]
        im_arr = raw["im_z"][ri, ci]
        freq_arr = raw["freq"][ri, ci]
        if not isinstance(re_arr, np.ndarray):
            raise ValueError(f"{cell_id} için Diag_{diag_number} EIS verisi mevcut değil.")
        soc_levels = np.atleast_1d(raw["soc_level"])
        frames = []
        for s_idx, soc_pct in enumerate(soc_levels):
            frames.append(
                pd.DataFrame(
                    {
                        "soc_pct": np.full(freq_arr.shape[0], int(soc_pct), dtype=int),
                        "freq_Hz": np.asarray(freq_arr[:, s_idx], dtype=float),
                        "re_z_Ohm": np.asarray(re_arr[:, s_idx], dtype=float),
                        "im_z_Ohm": np.asarray(im_arr[:, s_idx], dtype=float),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    v_arr = raw["vcell"][ri, ci]
    if not isinstance(v_arr, np.ndarray) or v_arr.size == 0:
        raise ValueError(f"{cell_id} için Diag_{diag_number} ({kind}) verisi mevcut değil.")

    return pd.DataFrame(
        {
            "time_s": np.asarray(raw["time"][ri, ci], dtype=float),
            "voltage_V": np.asarray(v_arr, dtype=float),
            "current_A": np.asarray(raw["curr"][ri, ci], dtype=float),
            "capacity_Ah": np.asarray(raw["cap"][ri, ci], dtype=float),
        }
    )


def load_stanford_cycling(
    filepath: str | Path,
    cell_id: str | None = None,
) -> dict[str, Any]:
    """Stanford işlenmiş çevrim dosyasını (örn. W8.mat) standartlaştırılmış DataFrame ile yükler."""
    path = Path(filepath)
    raw = load_mat_file(path)

    inferred_cell = cell_id or path.stem
    missing_keys = [k for k in STANFORD_CYCLING_KEY_MAP if k not in raw]
    if missing_keys:
        raise KeyError(f"{path.name} içinde beklenen anahtarlar eksik: {missing_keys}")

    df = pd.DataFrame(
        {
            std_name: np.asarray(raw[mat_name])
            for mat_name, std_name in STANFORD_CYCLING_KEY_MAP.items()
        }
    )
    # Sayısal sütunları tutarlı tiplere dönüştür
    for col in ("time_s", "current_A", "voltage_V", "charge_capacity_Ah", "discharge_capacity_Ah"):
        df[col] = df[col].astype(np.float64)
    df["step_index"] = df["step_index"].astype(np.int32)

    return {
        "dataset": STANFORD_DATASET_ID,
        "filepath": str(path),
        "cell_id": inferred_cell,
        "c_rate": STANFORD_CELL_CRATES.get(inferred_cell, "unknown"),
        "nominal_capacity_Ah": STANFORD_NOMINAL_CAPACITY_AH,
        "current_sign_convention": "positive_charge_negative_discharge",
        "temperature_included": False,
        "dataframe": df,
    }


def load_stanford_raw_xlsx(
    filepath: str | Path,
    cell_id: str | None = None,
) -> dict[str, Any]:
    """Stanford ham Arbin Excel dosyasını (Aux_Temperature dahil) standartlaştırılmış DataFrame ile yükler."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    xl = pd.ExcelFile(path, engine="openpyxl")
    channel_sheets = [s for s in xl.sheet_names if str(s).startswith("Channel")]
    if not channel_sheets:
        raise ValueError(f"{path.name} içinde 'Channel*' sayfası bulunamadı: {xl.sheet_names}")

    raw_df = xl.parse(channel_sheets[0])
    temp_cols = [c for c in raw_df.columns if "Aux_Temperature" in str(c) or "Temperature" in str(c)]
    if not temp_cols:
        raise KeyError(f"{path.name} içinde sıcaklık sütunu bulunamadı: {list(raw_df.columns)}")
    temp_col = temp_cols[0]

    col_map = {
        "Test_Time(s)": "time_s",
        "Step_Time(s)": "step_time_s",
        "Step_Index": "step_index",
        "Cycle_Index": "cycle_index",
        "Voltage(V)": "voltage_V",
        "Current(A)": "current_A",
        "Charge_Capacity(Ah)": "charge_capacity_Ah",
        "Discharge_Capacity(Ah)": "discharge_capacity_Ah",
        temp_col: "temperature_C",
    }
    missing = [k for k in col_map if k not in raw_df.columns]
    if missing:
        raise KeyError(f"{path.name} içinde eksik sütunlar: {missing}")

    df = raw_df[list(col_map.keys())].rename(columns=col_map).copy()
    for col in ("time_s", "step_time_s", "voltage_V", "current_A", "charge_capacity_Ah", "discharge_capacity_Ah", "temperature_C"):
        df[col] = df[col].astype(np.float64)
    for col in ("step_index", "cycle_index"):
        df[col] = df[col].astype(np.int32)

    inferred_cell = cell_id
    if inferred_cell is None:
        for cid in STANFORD_CELL_CRATES:
            if f"_{cid}_" in path.name or path.stem.endswith(cid):
                inferred_cell = cid
                break
        inferred_cell = inferred_cell or "unknown"

    return {
        "dataset": STANFORD_DATASET_ID,
        "filepath": str(path),
        "sheet_name": channel_sheets[0],
        "cell_id": inferred_cell,
        "c_rate": STANFORD_CELL_CRATES.get(inferred_cell, "unknown"),
        "nominal_capacity_Ah": STANFORD_NOMINAL_CAPACITY_AH,
        "current_sign_convention": "positive_charge_negative_discharge",
        "temperature_included": True,
        "dataframe": df,
    }


def load_nasa_mat(filepath: str | Path) -> dict[str, Any]:
    """NASA Li-ion Battery Aging (.mat) dosyasını döngü, kapasite ve empedans tablolarıyla yükler."""
    path = Path(filepath)
    raw = load_mat_file(path)
    top_keys = list(raw.keys())
    if not top_keys:
        raise ValueError(f"NASA dosyası boş: {path}")

    cell_id = top_keys[0]
    cycles = list(raw[cell_id]["cycle"])

    cycle_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    imp_rows: list[dict[str, Any]] = []

    for idx, cyc in enumerate(cycles):
        ctype = str(cyc["type"])
        amb_t = float(cyc["ambient_temperature"])
        d = cyc["data"]
        n_samples = len(d["Time"]) if isinstance(d, dict) and "Time" in d else 1
        cycle_rows.append(
            {
                "cycle_index": idx,
                "type": ctype,
                "ambient_temperature_C": amb_t,
                "n_samples": int(n_samples),
            }
        )
        if ctype == "discharge" and "Capacity" in d:
            cap_val = float(np.asarray(d["Capacity"]).squeeze())
            cap_rows.append(
                {
                    "cycle_index": idx,
                    "ambient_temperature_C": amb_t,
                    "capacity_Ah": cap_val,
                }
            )
        elif ctype == "impedance":
            imp_rows.append(
                {
                    "cycle_index": idx,
                    "ambient_temperature_C": amb_t,
                    "Re_Ohm": float(np.real(d["Re"])),
                    "Rct_Ohm": float(np.real(d["Rct"])),
                }
            )

    return {
        "dataset": NASA_DATASET_ID,
        "filepath": str(path),
        "cell_id": cell_id,
        "nominal_capacity_Ah": NASA_NOMINAL_CAPACITY_AH,
        "current_sign_convention": "positive_charge_negative_discharge",
        "cycle_summary": pd.DataFrame(cycle_rows),
        "capacity_fade": pd.DataFrame(cap_rows),
        "impedance_summary": pd.DataFrame(imp_rows),
        "cycles": cycles,
    }


def extract_nasa_cycle_df(nasa_data: dict[str, Any], cycle_index: int) -> pd.DataFrame:
    """NASA veri yapısından belirli bir şarj veya deşarj döngüsünün zaman serisini döndürür."""
    cyc = nasa_data["cycles"][cycle_index]
    ctype = str(cyc["type"])
    if ctype not in ("charge", "discharge"):
        raise ValueError(f"Döngü {cycle_index} zaman serisi değil (tip={ctype}).")

    d = cyc["data"]
    out = {
        "time_s": np.asarray(d["Time"], dtype=np.float64),
        "voltage_V": np.asarray(d["Voltage_measured"], dtype=np.float64),
        "current_A": np.asarray(d["Current_measured"], dtype=np.float64),
        "temperature_C": np.asarray(d["Temperature_measured"], dtype=np.float64),
    }
    if ctype == "charge":
        out["current_cmd_A"] = np.asarray(d["Current_charge"], dtype=np.float64)
        out["voltage_cmd_V"] = np.asarray(d["Voltage_charge"], dtype=np.float64)
    else:
        out["current_cmd_A"] = np.asarray(d["Current_load"], dtype=np.float64)
        out["voltage_cmd_V"] = np.asarray(d["Voltage_load"], dtype=np.float64)
    return pd.DataFrame(out)


def inspect_timeseries(
    df: pd.DataFrame,
    time_col: str = "time_s",
    voltage_col: str = "voltage_V",
    current_col: str = "current_A",
    temp_col: str | None = None,
    v_bounds: tuple[float, float] = (2.5, 4.25),
    t_bounds: tuple[float, float] = (0.0, 60.0),
) -> dict[str, Any]:
    """Zaman serisi üzerinde eksik değer, zaman monotonluğu, örnekleme ve fiziksel sınır denetimi yapar."""
    n_rows = int(len(df))
    nan_by_col = {col: int(df[col].isna().sum()) for col in df.columns}
    nan_total = int(sum(nan_by_col.values()))

    t_vals = df[time_col].to_numpy(dtype=np.float64)
    dt = np.diff(t_vals) if n_rows > 1 else np.array([], dtype=np.float64)
    valid_dt = dt[~np.isnan(dt)]
    non_pos_dt = int(np.sum(valid_dt <= 0)) if valid_dt.size > 0 else 0

    v_vals = df[voltage_col].to_numpy(dtype=np.float64)
    i_vals = df[current_col].to_numpy(dtype=np.float64)

    v_zero_count = int(np.sum(v_vals == 0.0))
    v_oob_count = int(np.sum((v_vals < v_bounds[0]) | (v_vals > v_bounds[1])))

    summary: dict[str, Any] = {
        "n_rows": n_rows,
        "nan_total": nan_total,
        "nan_pct": round(100.0 * nan_total / max(1, n_rows * len(df.columns)), 6),
        "nan_by_col": nan_by_col,
        "time_monotonic_inc": bool(valid_dt.size > 0 and non_pos_dt == 0),
        "non_pos_dt_count": non_pos_dt,
        "dt_min_s": float(np.min(valid_dt)) if valid_dt.size > 0 else np.nan,
        "dt_median_s": float(np.median(valid_dt)) if valid_dt.size > 0 else np.nan,
        "dt_max_s": float(np.max(valid_dt)) if valid_dt.size > 0 else np.nan,
        "voltage_min_V": float(np.nanmin(v_vals)),
        "voltage_max_V": float(np.nanmax(v_vals)),
        "voltage_zero_count": v_zero_count,
        "voltage_oob_count": v_oob_count,
        "current_min_A": float(np.nanmin(i_vals)),
        "current_max_A": float(np.nanmax(i_vals)),
    }

    if temp_col is not None and temp_col in df.columns:
        temp_vals = df[temp_col].to_numpy(dtype=np.float64)
        summary["temp_min_C"] = float(np.nanmin(temp_vals))
        summary["temp_max_C"] = float(np.nanmax(temp_vals))
        summary["temp_oob_count"] = int(
            np.sum((temp_vals < t_bounds[0]) | (temp_vals > t_bounds[1]))
        )
    else:
        summary["temp_min_C"] = np.nan
        summary["temp_max_C"] = np.nan
        summary["temp_oob_count"] = 0

    return summary


def summarize_stanford_directory(stanford_dir: str | Path) -> pd.DataFrame:
    """İndirilmiş Stanford tanı ve çevrim dosyalarını denetleyip özet tablo döndürür."""
    root = Path(stanford_dir)
    rows: list[dict[str, Any]] = []

    diag_dir = root / "diagnostic"
    for fname, tkind in [
        ("capacity_test.mat", "capacity_test"),
        ("HPPC_test.mat", "hppc_test"),
        ("EIS_test.mat", "eis_test"),
    ]:
        fpath = diag_dir / fname
        if not fpath.exists():
            continue
        diag = load_stanford_diagnostic(fpath, test_kind=tkind)
        for cell_id in diag["cell_labels"]:
            available_diags = []
            first_audit: dict[str, Any] | None = None
            for dnum in diag["diag_numbers"]:
                try:
                    df = extract_stanford_diag_df(diag, cell_id, dnum)
                    available_diags.append(dnum)
                    if first_audit is None and tkind != "eis_test":
                        first_audit = inspect_timeseries(df, v_bounds=(2.48, 4.25))
                except ValueError:
                    continue
            row = {
                "dataset": STANFORD_DATASET_ID,
                "category": "diagnostic",
                "file": fname,
                "cell_id": cell_id,
                "c_rate": STANFORD_CELL_CRATES.get(cell_id, "unknown"),
                "available_diags_count": len(available_diags),
                "available_diags": ",".join(map(str, available_diags)),
                "temperature_included": False,
                "n_rows_sample": first_audit["n_rows"] if first_audit else (len(available_diags) * 3 if tkind == "eis_test" else 0),
                "nan_total": first_audit["nan_total"] if first_audit else 0,
                "time_monotonic_inc": first_audit["time_monotonic_inc"] if first_audit else np.nan,
                "dt_median_s": first_audit["dt_median_s"] if first_audit else np.nan,
                "voltage_min_V": first_audit["voltage_min_V"] if first_audit else np.nan,
                "voltage_max_V": first_audit["voltage_max_V"] if first_audit else np.nan,
                "voltage_zero_count": first_audit["voltage_zero_count"] if first_audit else 0,
                "current_min_A": first_audit["current_min_A"] if first_audit else np.nan,
                "current_max_A": first_audit["current_max_A"] if first_audit else np.nan,
            }
            rows.append(row)

    cyc_dir = root / "cycling" / "Cycling_1"
    if cyc_dir.exists():
        for fpath in sorted(cyc_dir.glob("*.mat")):
            cyc = load_stanford_cycling(fpath)
            audit = inspect_timeseries(cyc["dataframe"], v_bounds=(2.5, 4.25))
            rows.append(
                {
                    "dataset": STANFORD_DATASET_ID,
                    "category": "cycling_1",
                    "file": fpath.name,
                    "cell_id": cyc["cell_id"],
                    "c_rate": cyc["c_rate"],
                    "available_diags_count": 1,
                    "available_diags": "Cycling_1",
                    "temperature_included": cyc["temperature_included"],
                    "n_rows_sample": audit["n_rows"],
                    "nan_total": audit["nan_total"],
                    "time_monotonic_inc": audit["time_monotonic_inc"],
                    "dt_median_s": audit["dt_median_s"],
                    "voltage_min_V": audit["voltage_min_V"],
                    "voltage_max_V": audit["voltage_max_V"],
                    "voltage_zero_count": audit["voltage_zero_count"],
                    "current_min_A": audit["current_min_A"],
                    "current_max_A": audit["current_max_A"],
                }
            )

    return pd.DataFrame(rows)


def summarize_nasa_directory(nasa_dir: str | Path) -> pd.DataFrame:
    """İndirilmiş NASA .mat dosyalarını denetleyip özet tablo döndürür."""
    root = Path(nasa_dir)
    rows: list[dict[str, Any]] = []
    for fpath in sorted(root.glob("B*.mat")):
        nasa = load_nasa_mat(fpath)
        csum = nasa["cycle_summary"]
        cap_df = nasa["capacity_fade"]
        imp_df = nasa["impedance_summary"]

        nan_total = 0
        non_mono_cycles = 0
        v_vals_all: list[float] = []
        t_vals_all: list[float] = []
        dt_meds: list[float] = []

        for idx, cyc in enumerate(nasa["cycles"]):
            if cyc["type"] in ("charge", "discharge"):
                df = extract_nasa_cycle_df(nasa, idx)
                audit = inspect_timeseries(df, temp_col="temperature_C", v_bounds=(2.0, 4.30))
                nan_total += audit["nan_total"]
                if not audit["time_monotonic_inc"]:
                    non_mono_cycles += 1
                v_vals_all.extend([audit["voltage_min_V"], audit["voltage_max_V"]])
                t_vals_all.extend([audit["temp_min_C"], audit["temp_max_C"]])
                if not np.isnan(audit["dt_median_s"]):
                    dt_meds.append(audit["dt_median_s"])

        amb_temps = sorted(csum["ambient_temperature_C"].unique().tolist())
        rows.append(
            {
                "dataset": NASA_DATASET_ID,
                "file": fpath.name,
                "cell_id": nasa["cell_id"],
                "total_cycles": int(len(csum)),
                "charge_cycles": int((csum["type"] == "charge").sum()),
                "discharge_cycles": int((csum["type"] == "discharge").sum()),
                "impedance_cycles": int((csum["type"] == "impedance").sum()),
                "ambient_temperatures_C": ",".join(map(lambda x: str(int(x)), amb_temps)),
                "initial_capacity_Ah": float(cap_df["capacity_Ah"].iloc[0]) if not cap_df.empty else np.nan,
                "final_capacity_Ah": float(cap_df["capacity_Ah"].iloc[-1]) if not cap_df.empty else np.nan,
                "Re_min_Ohm": float(imp_df["Re_Ohm"].min()) if not imp_df.empty else np.nan,
                "Re_max_Ohm": float(imp_df["Re_Ohm"].max()) if not imp_df.empty else np.nan,
                "Rct_min_Ohm": float(imp_df["Rct_Ohm"].min()) if not imp_df.empty else np.nan,
                "Rct_max_Ohm": float(imp_df["Rct_Ohm"].max()) if not imp_df.empty else np.nan,
                "nan_total": int(nan_total),
                "non_monotonic_cycles": int(non_mono_cycles),
                "dt_median_s": float(np.median(dt_meds)) if dt_meds else np.nan,
                "voltage_min_V": float(np.min(v_vals_all)) if v_vals_all else np.nan,
                "voltage_max_V": float(np.max(v_vals_all)) if v_vals_all else np.nan,
                "temp_min_C": float(np.min(t_vals_all)) if t_vals_all else np.nan,
                "temp_max_C": float(np.max(t_vals_all)) if t_vals_all else np.nan,
            }
        )

    return pd.DataFrame(rows)
