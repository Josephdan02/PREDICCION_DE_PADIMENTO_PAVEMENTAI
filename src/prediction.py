"""Predicción de PCI con el modelo entrenado.

Reglas del proyecto:
- Toda predicción proviene de ``LinearRegression.predict`` cargado desde Joblib.
- No se usan condicionales para fabricar valores de PCI.
- Los condicionales solo clasifican el estado y generan advertencias.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from src import config


class ModelNotTrainedError(RuntimeError):
    """Se lanza cuando no existe un modelo entrenado en disco."""


@lru_cache(maxsize=1)
def _cached_model(path_str: str, mtime: float):
    return joblib.load(path_str)


def load_model(path=None):
    """Carga el modelo Joblib (con caché invalidada si el archivo cambia)."""
    path = path or config.MODEL_PATH
    if not path.exists():
        raise ModelNotTrainedError(
            "No se encontró un modelo entrenado. Ejecuta `python -m src.train_model` "
            "o usa el botón de entrenamiento en la página Modelo ML."
        )
    return _cached_model(str(path), path.stat().st_mtime)


def model_is_available(path=None) -> bool:
    return (path or config.MODEL_PATH).exists()


def model_info(path=None) -> dict:
    """Intercepto, pendiente y ecuación del modelo cargado."""
    model = load_model(path)
    intercept = float(model.intercept_)
    slope = float(model.coef_[0])
    return {
        "intercepto": intercept,
        "pendiente": slope,
        "ecuacion": f"PCI = {intercept:.2f} {'+' if slope >= 0 else '−'} {abs(slope):.3f} × antigüedad",
    }


def predict_pci(antiguedad_anios: float, path=None) -> float:
    """PCI estimado (valor matemático original, sin recortar)."""
    if antiguedad_anios is None or not np.isfinite(float(antiguedad_anios)):
        raise ValueError("La antigüedad debe ser un número válido.")
    if float(antiguedad_anios) < 0:
        raise ValueError("La antigüedad no puede ser negativa.")
    model = load_model(path)
    return float(model.predict(np.array([[float(antiguedad_anios)]], dtype=float))[0])


def predict_many(antiguedades, path=None) -> np.ndarray:
    """Predicción vectorizada para varios años de antigüedad."""
    arr = np.asarray(antiguedad_anios_to_array(antiguedades), dtype=float).reshape(-1, 1)
    if (arr < 0).any():
        raise ValueError("La antigüedad no puede ser negativa.")
    return load_model(path).predict(arr)


def antiguedad_anios_to_array(values) -> list[float]:
    return [float(v) for v in values]


def clamp_pci(value: float) -> float:
    """Recorta el PCI al rango visual 0-100 conservando aparte el valor original."""
    return float(np.clip(value, config.PCI_MIN, config.PCI_MAX))


def predict_scenario(antiguedad_actual: float, horizonte_anios: int, metadata: dict | None = None) -> dict:
    """Ejecuta el escenario actual y futuro con el modelo entrenado.

    Devuelve valores originales y recortados, la variación y advertencias.
    """
    metadata = metadata if metadata is not None else config.load_metadata()
    antiguedad_futura = float(antiguedad_actual) + float(horizonte_anios)

    pci_actual_raw = predict_pci(antiguedad_actual)
    pci_futuro_raw = predict_pci(antiguedad_futura)

    warnings: list[str] = []
    rango = metadata.get("rango_antiguedad_entrenamiento") if metadata else None
    if rango:
        minimo, maximo = rango.get("min"), rango.get("max")
        if minimo is not None and antiguedad_actual < minimo:
            warnings.append(
                f"La antigüedad actual ({antiguedad_actual:.1f} años) está por debajo del rango "
                f"observado en entrenamiento ({minimo:.1f}–{maximo:.1f} años): la predicción es una extrapolación."
            )
        if maximo is not None and antiguedad_futura > maximo:
            warnings.append(
                f"El horizonte simulado alcanza {antiguedad_futura:.1f} años, fuera del rango observado "
                f"({minimo:.1f}–{maximo:.1f} años): la estimación extrapola el modelo lineal."
            )

    for etiqueta, valor in (("actual", pci_actual_raw), ("futuro", pci_futuro_raw)):
        if valor < config.PCI_MIN or valor > config.PCI_MAX:
            warnings.append(
                f"El PCI {etiqueta} matemático ({valor:.1f}) queda fuera de 0-100 y se mostrará recortado. "
                "El valor original se conserva para análisis."
            )

    estado_actual, color_actual, _ = config.classify_pci(clamp_pci(pci_actual_raw))
    estado_futuro, color_futuro, _ = config.classify_pci(clamp_pci(pci_futuro_raw))

    return {
        "antiguedad_actual": float(antiguedad_actual),
        "antiguedad_futura": antiguedad_futura,
        "horizonte_anios": int(horizonte_anios),
        "pci_actual": clamp_pci(pci_actual_raw),
        "pci_futuro": clamp_pci(pci_futuro_raw),
        "pci_actual_raw": pci_actual_raw,
        "pci_futuro_raw": pci_futuro_raw,
        "variacion": clamp_pci(pci_futuro_raw) - clamp_pci(pci_actual_raw),
        "variacion_raw": pci_futuro_raw - pci_actual_raw,
        "estado_actual": estado_actual,
        "estado_futuro": estado_futuro,
        "color_actual": color_actual,
        "color_futuro": color_futuro,
        "advertencias": warnings,
    }


def classify(value: float) -> str:
    """Etiqueta de condición visual para un PCI."""
    return config.classify_pci(value)[0]
