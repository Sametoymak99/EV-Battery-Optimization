"""Aşama 3: 1-RC Thevenin, birleştirilmiş termal ve ampirik yaşlanma batarya modeli ve ODE çözücüleri."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class BatteryLimits:
    """LG Chem INR21700-M50T hücre güvenlik ve çalışma sınırları."""

    v_min_V: float = 2.50
    v_max_V: float = 4.20
    t_max_C: float = 45.0
    i_min_A: float = 0.0
    i_max_A: float = 9.70  # Varsayılan 2C hızlı şarj üst sınırı (0.7C datasheet=3.40A, 3C=14.55A)
    soc_min: float = 0.0
    soc_max: float = 1.0


@dataclass
class BatteryParameters:
    """Batarya fiziksel parametreleri (Aşama 4 kalibrasyonu öncesi ön değerler açıkça etiketlidir)."""

    # Elektriksel parametreler (Aşama 4'te capacity_test ve HPPC_test'ten kestirilecek)
    Q_Ah: float = 4.85
    eta: float = 0.998
    R0_Ohm: float = 0.025
    R1_Ohm: float = 0.015
    C1_F: float = 2000.0

    # Termal parametreler (Aşama 4'te W10 Channel_6.2 verisinden kestirilecek)
    Rth_K_per_W: float = 12.0
    Cth_J_per_K: float = 65.0
    T_amb_C: float = 23.0

    # Ampirik göreli yaşlanma göstergesi (literatür / göreli indeks; mutlak SOH değildir)
    k0: float = 1.5e-2
    Ea_J_per_mol: float = 31500.0
    p_crate: float = 1.5
    alpha_soc: float = 0.5
    Rg_J_per_mol_K: float = 8.314462618

    is_calibrated: bool = False
    parameter_provenance: dict[str, str] = field(
        default_factory=lambda: {
            "Q_Ah": "datasheet_nominal_placeholder",
            "eta": "assumed_coulombic_efficiency",
            "R0_Ohm": "preliminary_placeholder",
            "R1_Ohm": "preliminary_placeholder",
            "C1_F": "preliminary_placeholder",
            "Rth_K_per_W": "preliminary_placeholder",
            "Cth_J_per_K": "preliminary_placeholder",
            "T_amb_C": "stanford_chamber_23C",
            "aging": "literature_empirical_relative_indicator",
        }
    )

    @classmethod
    def from_calibrated_json(cls, json_path: str | Path) -> "BatteryParameters":
        """Aşama 4 kalibre edilmiş parametre JSON dosyasından BatteryParameters yükler."""
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        p = payload["parameters"]
        return cls(
            Q_Ah=float(p["Q_Ah"]),
            eta=float(p["eta"]),
            R0_Ohm=float(p["R0_Ohm"]),
            R1_Ohm=float(p["R1_Ohm"]),
            C1_F=float(p["C1_F"]),
            Rth_K_per_W=float(p["Rth_K_per_W"]),
            Cth_J_per_K=float(p["Cth_J_per_K"]),
            T_amb_C=float(p["T_amb_C"]),
            k0=float(p["k0"]),
            Ea_J_per_mol=float(p["Ea_J_per_mol"]),
            p_crate=float(p["p_crate"]),
            alpha_soc=float(p["alpha_soc"]),
            Rg_J_per_mol_K=float(p["Rg_J_per_mol_K"]),
            is_calibrated=bool(p.get("is_calibrated", True)),
            parameter_provenance=dict(payload.get("provenance", {})),
        )


class OCVInterpolator:
    """C/20 verisinden oluşturulan şekil-korumalı PCHIP OCV(SOC) fonksiyonu."""

    def __init__(self, soc_grid: np.ndarray, ocv_V: np.ndarray, cell_id: str = "unknown") -> None:
        soc_arr = np.asarray(soc_grid, dtype=np.float64)
        ocv_arr = np.asarray(ocv_V, dtype=np.float64)
        if soc_arr.ndim != 1 or ocv_arr.ndim != 1 or len(soc_arr) != len(ocv_arr):
            raise ValueError("soc_grid ve ocv_V aynı uzunlukta 1B vektör olmalıdır.")
        order = np.argsort(soc_arr)
        soc_sorted = soc_arr[order]
        ocv_sorted = ocv_arr[order]
        # Tekrarlanan SOC noktalarını birleştir ve fiziksel monotonluk sağla
        uniq_soc, inv = np.unique(soc_sorted, return_inverse=True)
        uniq_ocv = np.zeros_like(uniq_soc)
        for idx in range(len(uniq_soc)):
            uniq_ocv[idx] = float(np.mean(ocv_sorted[inv == idx]))
        uniq_ocv = np.maximum.accumulate(uniq_ocv)

        self.soc_grid = uniq_soc
        self.ocv_V = uniq_ocv
        self.cell_id = cell_id
        self._interp = PchipInterpolator(self.soc_grid, self.ocv_V, extrapolate=False)
        self._d_interp = self._interp.derivative()

    @classmethod
    def from_processed_csv(cls, csv_path: str | Path, cell_id: str = "W8") -> "OCVInterpolator":
        """data/processed/stanford_ocv_c20_diag1.csv dosyasından OCV interpolatörü kurar."""
        df = pd.read_csv(csv_path)
        sub = df[df["cell_id"] == cell_id]
        if sub.empty:
            raise KeyError(f"{csv_path} içinde hücre bulunamadı: {cell_id}")
        return cls(sub["soc"].to_numpy(), sub["voltage_V"].to_numpy(), cell_id=cell_id)

    @classmethod
    def from_calibrated_json(cls, json_path: str | Path) -> "OCVInterpolator":
        """calibrated_parameters.json içinden kalibre OCV(SOC) eğrisini yükler."""
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload["ocv_soc_grid"], payload["ocv_V_grid"], cell_id="calibrated_mean")

    def __call__(self, soc: float | np.ndarray) -> float | np.ndarray:
        """Verilen SOC değeri için OCV [V] döndürür."""
        clipped = np.clip(soc, float(self.soc_grid[0]), float(self.soc_grid[-1]))
        val = self._interp(clipped)
        if np.ndim(soc) == 0:
            return float(val)
        return np.asarray(val, dtype=np.float64)

    def derivative(self, soc: float | np.ndarray) -> float | np.ndarray:
        """dOCV/dSOC [V] türevini döndürür."""
        clipped = np.clip(soc, float(self.soc_grid[0]), float(self.soc_grid[-1]))
        val = self._d_interp(clipped)
        if np.ndim(soc) == 0:
            return float(val)
        return np.asarray(val, dtype=np.float64)


def aging_soc_factor(soc: float | np.ndarray, alpha_soc: float = 0.5) -> float | np.ndarray:
    """Yaşlanma denklemi g(SOC) = 1 + alpha_soc * SOC^2 terimi."""
    s = np.clip(soc, 0.0, 1.0)
    return 1.0 + alpha_soc * (s ** 2)


def terminal_voltage(
    soc: float | np.ndarray,
    vp_V: float | np.ndarray,
    current_A: float | np.ndarray,
    params: BatteryParameters,
    ocv_fn: Callable[[float | np.ndarray], float | np.ndarray],
) -> float | np.ndarray:
    """Terminal gerilimi denklemi: Vt = OCV(SOC) + I * R0 + Vp."""
    return ocv_fn(soc) + current_A * params.R0_Ohm + vp_V


def instantaneous_power_loss_W(
    vp_V: float | np.ndarray,
    current_A: float | np.ndarray,
    params: BatteryParameters,
) -> float | np.ndarray:
    """Anlık direnç kaynaklı güç kaybı [W]: P_loss = I^2 * R0 + Vp^2 / R1."""
    return (current_A ** 2) * params.R0_Ohm + (vp_V ** 2) / params.R1_Ohm


def euler_stability_limit(params: BatteryParameters) -> float:
    """Açık Euler yöntemi için doğrusal kararlılık adım üst sınırı: 2 * min(R1*C1, Rth*Cth)."""
    tau_rc = params.R1_Ohm * params.C1_F
    tau_th = params.Rth_K_per_W * params.Cth_J_per_K
    return 2.0 * min(tau_rc, tau_th)


def ode_rhs(
    _t: float,
    y: np.ndarray,
    current_A: float,
    params: BatteryParameters,
) -> np.ndarray:
    """Artırılmış durum vektörü y = [SOC, Vp, T_C, D, E_loss_J] için türev vektörü dy/dt."""
    soc, vp_v, temp_c, _d, _e = y
    dsoc_dt = (params.eta * current_A) / (3600.0 * params.Q_Ah)
    dvp_dt = -vp_v / (params.R1_Ohm * params.C1_F) + current_A / params.C1_F
    p_loss = (current_A ** 2) * params.R0_Ohm + (vp_v ** 2) / params.R1_Ohm
    dtemp_dt = (p_loss - (temp_c - params.T_amb_C) / params.Rth_K_per_W) / params.Cth_J_per_K

    t_kelvin = max(temp_c + 273.15, 200.0)
    c_rate = abs(current_A) / params.Q_Ah
    g_soc = float(aging_soc_factor(float(soc), params.alpha_soc))
    dd_dt = (
        params.k0
        * np.exp(-params.Ea_J_per_mol / (params.Rg_J_per_mol_K * t_kelvin))
        * (c_rate ** params.p_crate)
        * g_soc
    )
    return np.array([dsoc_dt, dvp_dt, dtemp_dt, dd_dt, p_loss], dtype=np.float64)


def _resolve_current_series(
    t_eval: np.ndarray,
    current_input: Callable[[float], float] | np.ndarray,
) -> tuple[Callable[[float], float], np.ndarray]:
    """Akım girdisini hem skaler fonksiyon hem zaman dizisi olarak çözümler."""
    if callable(current_input):
        i_arr = np.array([float(current_input(float(t))) for t in t_eval], dtype=np.float64)
        return current_input, i_arr
    i_arr = np.asarray(current_input, dtype=np.float64)
    if i_arr.shape != t_eval.shape:
        raise ValueError(f"Akım dizisi boyutu {i_arr.shape}, t_eval {t_eval.shape} ile uyuşmuyor.")

    def _interp_i(t: float) -> float:
        return float(np.interp(t, t_eval, i_arr))

    return _interp_i, i_arr


def _integrate_euler(
    t_eval: np.ndarray,
    y0: np.ndarray,
    i_fn: Callable[[float], float],
    params: BatteryParameters,
) -> np.ndarray:
    """Açık İleri Euler ODE çözücüsü."""
    n = len(t_eval)
    y = np.zeros((n, len(y0)), dtype=np.float64)
    y[0] = y0
    for k in range(n - 1):
        dt = float(t_eval[k + 1] - t_eval[k])
        i_k = float(i_fn(float(t_eval[k])))
        y[k + 1] = y[k] + dt * ode_rhs(float(t_eval[k]), y[k], i_k, params)
    return y


def _integrate_rk4(
    t_eval: np.ndarray,
    y0: np.ndarray,
    i_fn: Callable[[float], float],
    params: BatteryParameters,
) -> np.ndarray:
    """Klasik 4. Derece Runge-Kutta (RK4) ODE çözücüsü."""
    n = len(t_eval)
    y = np.zeros((n, len(y0)), dtype=np.float64)
    y[0] = y0
    for k in range(n - 1):
        tk = float(t_eval[k])
        dt = float(t_eval[k + 1] - tk)
        tm = tk + 0.5 * dt
        tn = float(t_eval[k + 1])

        ik = float(i_fn(tk))
        im = float(i_fn(tm))
        inext = float(i_fn(tn))

        k1 = ode_rhs(tk, y[k], ik, params)
        k2 = ode_rhs(tm, y[k] + 0.5 * dt * k1, im, params)
        k3 = ode_rhs(tm, y[k] + 0.5 * dt * k2, im, params)
        k4 = ode_rhs(tn, y[k] + dt * k3, inext, params)
        y[k + 1] = y[k] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


def _integrate_solve_ivp(
    t_eval: np.ndarray,
    y0: np.ndarray,
    i_fn: Callable[[float], float],
    params: BatteryParameters,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """SciPy solve_ivp (uyarlanabilir RK45) ODE çözücüsü."""
    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, float(i_fn(float(t))), params),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp başarısız oldu: {sol.message}")
    return sol.y.T


def simulate_current_profile(
    t_eval: np.ndarray,
    current_input: Callable[[float], float] | np.ndarray,
    params: BatteryParameters,
    ocv_fn: Callable[[float | np.ndarray], float | np.ndarray],
    x0: np.ndarray | None = None,
    method: str = "rk4",
) -> dict[str, Any]:
    """Verilen zaman ızgarası ve akım profili için batarya durum yörüngesini simüle eder."""
    t_arr = np.asarray(t_eval, dtype=np.float64)
    if t_arr.ndim != 1 or len(t_arr) < 2 or not np.all(np.diff(t_arr) > 0):
        raise ValueError("t_eval en az 2 elemanlı kesin artan 1B dizi olmalıdır.")

    y0 = np.array([0.20, 0.0, params.T_amb_C, 0.0, 0.0], dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64)
    if len(y0) == 4:
        y0 = np.append(y0, 0.0)

    i_fn, i_arr = _resolve_current_series(t_arr, current_input)
    method_low = method.lower()

    t_wall0 = time.perf_counter()
    if method_low == "euler":
        states = _integrate_euler(t_arr, y0, i_fn, params)
    elif method_low == "rk4":
        states = _integrate_rk4(t_arr, y0, i_fn, params)
    elif method_low == "solve_ivp":
        states = _integrate_solve_ivp(t_arr, y0, i_fn, params)
    else:
        raise ValueError(f"Desteklenmeyen ODE yöntemi: {method}. Seçenekler: 'euler', 'rk4', 'solve_ivp'.")
    runtime_s = float(time.perf_counter() - t_wall0)

    soc = states[:, 0]
    vp_v = states[:, 1]
    temp_c = states[:, 2]
    aging_d = states[:, 3]
    e_loss_j = states[:, 4]

    vt_v = np.asarray(terminal_voltage(soc, vp_v, i_arr, params, ocv_fn), dtype=np.float64)
    p_loss_w = np.asarray(instantaneous_power_loss_W(vp_v, i_arr, params), dtype=np.float64)

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
        "final_time_s": float(t_arr[-1] - t_arr[0]),
        "final_soc": float(soc[-1]),
        "max_voltage_V": float(np.max(vt_v)),
        "max_temperature_C": float(np.max(temp_c)),
        "total_energy_loss_J": float(e_loss_j[-1]),
        "total_aging_D": float(aging_d[-1] - aging_d[0]),
        "method": method_low,
        "runtime_s": runtime_s,
    }


def simulate_soc_profile(
    stage_currents_A: np.ndarray | list[float],
    params: BatteryParameters,
    ocv_fn: Callable[[float | np.ndarray], float | np.ndarray],
    soc_init: float = 0.20,
    soc_target: float = 0.90,
    T_init_C: float | None = None,
    vp_init_V: float = 0.0,
    method: str = "rk4",
    dt_max_s: float = 2.0,
) -> dict[str, Any]:
    """SOC aralığını N eşit dilime bölen çok aşamalı sabit akım profili için tam şarj simülasyonu yapar."""
    currents = np.asarray(stage_currents_A, dtype=np.float64)
    if currents.ndim != 1 or len(currents) == 0:
        raise ValueError("stage_currents_A boş olmayan 1B dizi olmalıdır.")
    if np.any(currents <= 0.0):
        raise ValueError("Tüm şarj aşama akımları pozitif olmalıdır.")
    if not (0.0 <= soc_init < soc_target <= 1.0):
        raise ValueError("0 <= soc_init < soc_target <= 1 koşulu sağlanmalıdır.")

    n_stages = len(currents)
    dsoc = (soc_target - soc_init) / n_stages
    stage_durations = (3600.0 * params.Q_Ah * dsoc) / (params.eta * currents)

    t_init = params.T_amb_C if T_init_C is None else float(T_init_C)
    y_curr = np.array([soc_init, vp_init_V, t_init, 0.0, 0.0], dtype=np.float64)

    t_chunks: list[np.ndarray] = []
    i_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []

    t_cursor = 0.0
    t_wall0 = time.perf_counter()
    method_low = method.lower()

    for idx, (ik, dtk) in enumerate(zip(currents, stage_durations)):
        n_sub = max(2, int(np.ceil(dtk / dt_max_s)) + 1)
        t_local = np.linspace(t_cursor, t_cursor + float(dtk), n_sub)
        i_local = np.full(n_sub, float(ik), dtype=np.float64)

        def _const_i(_t: float, val: float = float(ik)) -> float:
            return val

        if method_low == "euler":
            st = _integrate_euler(t_local, y_curr, _const_i, params)
        elif method_low == "rk4":
            st = _integrate_rk4(t_local, y_curr, _const_i, params)
        elif method_low == "solve_ivp":
            st = _integrate_solve_ivp(t_local, y_curr, _const_i, params)
        else:
            raise ValueError(f"Desteklenmeyen ODE yöntemi: {method}.")

        y_curr = st[-1].copy()
        if idx > 0:
            t_local = t_local[1:]
            i_local = i_local[1:]
            st = st[1:]
        t_chunks.append(t_local)
        i_chunks.append(i_local)
        state_chunks.append(st)
        t_cursor += float(dtk)

    runtime_s = float(time.perf_counter() - t_wall0)
    t_all = np.concatenate(t_chunks)
    i_all = np.concatenate(i_chunks)
    states = np.vstack(state_chunks)

    soc = states[:, 0]
    vp_v = states[:, 1]
    temp_c = states[:, 2]
    aging_d = states[:, 3]
    e_loss_j = states[:, 4]

    vt_v = np.asarray(terminal_voltage(soc, vp_v, i_all, params, ocv_fn), dtype=np.float64)
    p_loss_w = np.asarray(instantaneous_power_loss_W(vp_v, i_all, params), dtype=np.float64)

    return {
        "time_s": t_all,
        "current_A": i_all,
        "soc": soc,
        "vp_V": vp_v,
        "temperature_C": temp_c,
        "aging_D": aging_d,
        "terminal_voltage_V": vt_v,
        "power_loss_W": p_loss_w,
        "energy_loss_J": e_loss_j,
        "energy_loss_Wh": e_loss_j / 3600.0,
        "stage_currents_A": currents,
        "stage_durations_s": stage_durations,
        "final_time_s": float(np.sum(stage_durations)),
        "final_soc": float(soc[-1]),
        "max_voltage_V": float(np.max(vt_v)),
        "max_temperature_C": float(np.max(temp_c)),
        "total_energy_loss_J": float(e_loss_j[-1]),
        "total_aging_D": float(aging_d[-1] - aging_d[0]),
        "method": method_low,
        "runtime_s": runtime_s,
    }
