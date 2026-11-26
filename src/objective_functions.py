"""Çok amaçlı optimizasyon için normalize edilmiş amaç fonksiyonları."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReferenceScales:
    """Amaç fonksiyonlarını boyutsuzlaştırmak için CC-CV referans ölçekleri."""

    t_ref_s: float
    E_ref_J: float
    D_ref: float

    @classmethod
    def from_json(cls, json_path: str | Path) -> "ReferenceScales":
        """data/processed/reference_scales.json dosyasından referans ölçeklerini yükler."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            t_ref_s=float(data["t_ref_s"]),
            E_ref_J=float(data["E_ref_J"]),
            D_ref=float(data["D_ref"]),
        )

    def to_json(self, json_path: str | Path) -> None:
        """Referans ölçeklerini JSON dosyasına kaydeder."""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "t_ref_s": self.t_ref_s,
                    "E_ref_J": self.E_ref_J,
                    "D_ref": self.D_ref,
                },
                f,
                indent=2,
            )


def evaluate_objectives(
    sim_result: dict[str, Any],
    ref_scales: ReferenceScales,
    weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
) -> dict[str, float]:
    """Normalize edilmiş J1 (süre), J2 (enerji kaybı), J3 (yaşlanma) ve ağırlıklı toplam J'yi hesaplar."""
    w1, w2, w3 = [float(w) for w in weights]
    if any(w < 0.0 for w in (w1, w2, w3)):
        raise ValueError("Tüm amaç ağırlıkları negatif olmayan değerler olmalıdır.")
    if abs((w1 + w2 + w3) - 1.0) > 1e-6:
        raise ValueError(f"Ağırlıklar toplamı 1 olmalıdır (verilen={w1 + w2 + w3:.6f}).")

    j1 = float(sim_result["final_time_s"]) / ref_scales.t_ref_s
    j2 = float(sim_result["total_energy_loss_J"]) / ref_scales.E_ref_J
    j3 = float(sim_result["total_aging_D"]) / ref_scales.D_ref
    j_total = w1 * j1 + w2 * j2 + w3 * j3

    return {
        "J1": j1,
        "J2": j2,
        "J3": j3,
        "J_total": float(j_total),
    }
