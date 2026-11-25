"""Evaluación del modelo de regresión lineal.

Calcula R², MAE y RMSE usando las funciones oficiales de scikit-learn.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred) -> dict:
    """Devuelve un diccionario con R², MAE y RMSE (valores reales, sin redondeo inventado)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("y_true e y_pred deben tener el mismo tamaño y no estar vacíos.")
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def residuals(y_true, y_pred) -> np.ndarray:
    """Residuos = valor real - valor predicho."""
    return np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)


def interpret_r2(r2: float) -> str:
    """Mensaje cualitativo honesto sobre el coeficiente de determinación."""
    if r2 >= 0.85:
        nivel = "muy alto"
    elif r2 >= 0.65:
        nivel = "moderado-alto"
    elif r2 >= 0.40:
        nivel = "moderado"
    elif r2 >= 0.0:
        nivel = "bajo"
    else:
        nivel = "nulo (el modelo no supera a la media)"
    return (
        f"El modelo explica aproximadamente {max(r2, 0) * 100:.1f}% de la variabilidad "
        f"del PCI observado (ajuste {nivel})."
    )


def equation_text(intercept: float, slope: float) -> str:
    """Ecuación de la recta en notación legible."""
    signo = "+" if slope >= 0 else "−"
    return f"PCI = {intercept:.2f} {signo} {abs(slope):.3f} × antigüedad"


def slope_explanation(slope: float) -> str:
    """Explica el significado de la pendiente en unidades de PCI por año."""
    if slope < 0:
        return (
            f"Por cada año adicional de antigüedad, el PCI estimado disminuye en promedio "
            f"{abs(slope):.2f} puntos. Un pendiente negativa es lo esperado en pavimentos que se degradan."
        )
    return (
        f"Por cada año adicional de antigüedad, el PCI estimado aumenta en promedio {slope:.2f} puntos. "
        "Una pendiente positiva no es lo habitual en pavimentos y sugiere revisar los datos."
    )


def metrics_table_rows(metrics: dict) -> list[dict]:
    """Filas listas para mostrar las métricas en una tabla."""
    return [
        {"Métrica": "R² (coef. de determinación)", "Valor": f"{metrics['r2']:.4f}"},
        {"Métrica": "MAE (error absoluto medio, puntos PCI)", "Valor": f"{metrics['mae']:.3f}"},
        {"Métrica": "RMSE (raíz del error cuadrático medio, puntos PCI)", "Valor": f"{metrics['rmse']:.3f}"},
    ]
