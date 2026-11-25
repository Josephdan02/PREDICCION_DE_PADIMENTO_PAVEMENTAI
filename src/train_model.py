"""Entrenamiento real del modelo de regresión lineal (PCI ~ antigüedad).

Uso:
    python -m src.train_model

- Divide los datos 80/20 con semilla reproducible.
- Usa una división AGRUPADA por ``codigo_tramo`` para que un mismo tramo no
  aparezca a la vez en entrenamiento y prueba.
- Entrena ``sklearn.linear_model.LinearRegression``.
- Calcula R², MAE y RMSE sobre el conjunto de prueba.
- Persiste el modelo con Joblib y las métricas en ``model_metadata.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src import config, data_loader, evaluation, preprocessing


def split_data(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Divide el dataframe en (train, test) y devuelve también el método usado.

    Si hay suficientes grupos (tramos) se usa ``GroupShuffleSplit``; en caso
    contrario se recurre a una división aleatoria simple.
    """
    n_grupos = df[config.GROUP_COLUMN].nunique() if config.GROUP_COLUMN in df.columns else 0
    if n_grupos >= 5 and len(df) >= 20:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df[config.GROUP_COLUMN]))
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
            "GroupShuffleSplit (agrupado por codigo_tramo)",
        )

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), "train_test_split (aleatorio simple)"


def train_model(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    """Entrena el modelo y devuelve un diccionario con modelo, métricas y metadatos."""
    config.ensure_directories()
    if df is None:
        df = data_loader.load_dataset()

    limpio = preprocessing.clean_dataframe(df)
    if len(limpio) < 10:
        raise ValueError(
            f"Se requieren al menos 10 registros limpios para entrenar; hay {len(limpio)}."
        )

    train_df, test_df, split_method = split_data(limpio)

    X_train = train_df[[config.FEATURE_COLUMN]].to_numpy(dtype=float)
    y_train = train_df[config.TARGET_COLUMN].to_numpy(dtype=float)
    X_test = test_df[[config.FEATURE_COLUMN]].to_numpy(dtype=float)
    y_test = test_df[config.TARGET_COLUMN].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics_train = evaluation.calculate_metrics(y_train, y_pred_train)
    metrics_test = evaluation.calculate_metrics(y_test, y_pred_test)

    intercept = float(model.intercept_)
    slope = float(model.coef_[0])

    metadata = {
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "algoritmo": "sklearn.linear_model.LinearRegression",
        "variable_independiente": config.FEATURE_COLUMN,
        "variable_dependiente": config.TARGET_COLUMN,
        "ecuacion": evaluation.equation_text(intercept, slope),
        "intercepto": round(intercept, 6),
        "pendiente": round(slope, 6),
        "metricas": {
            "r2": round(metrics_test["r2"], 6),
            "mae": round(metrics_test["mae"], 6),
            "rmse": round(metrics_test["rmse"], 6),
        },
        "metricas_entrenamiento": {
            "r2": round(metrics_train["r2"], 6),
            "mae": round(metrics_train["mae"], 6),
            "rmse": round(metrics_train["rmse"], 6),
        },
        "conjunto_prueba": {
            "n_registros": int(len(test_df)),
            "n_tramos": int(test_df[config.GROUP_COLUMN].nunique()),
        },
        "conjunto_entrenamiento": {
            "n_registros": int(len(train_df)),
            "n_tramos": int(train_df[config.GROUP_COLUMN].nunique()),
        },
        "total_registros_limpios": int(len(limpio)),
        "metodo_division": split_method,
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "rango_antiguedad_entrenamiento": {
            "min": float(train_df[config.FEATURE_COLUMN].min()),
            "max": float(train_df[config.FEATURE_COLUMN].max()),
        },
        "rango_pci_observado": {
            "min": float(limpio[config.TARGET_COLUMN].min()),
            "max": float(limpio[config.TARGET_COLUMN].max()),
        },
        "origen_datos": "Dataset sintético (tipo_registro = SIMULADO). No son inspecciones reales de Huánuco.",
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if save:
        config.ensure_directories()
        joblib.dump(model, config.MODEL_PATH)
        config.METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "model": model,
        "metadata": metadata,
        "train_df": train_df,
        "test_df": test_df,
        "y_pred_test": y_pred_test,
        "y_pred_train": y_pred_train,
    }


def load_training_bundle() -> dict:
    """Reconstruye el paquete de resultados para gráficos a partir del dataset y el modelo guardado."""
    result = train_model(save=False)
    return result


def main() -> None:  # pragma: no cover
    resultado = train_model(save=True)
    meta = resultado["metadata"]
    ecuacion_ascii = meta["ecuacion"].encode("ascii", "replace").decode("ascii")
    print("Modelo entrenado y guardado en:", config.MODEL_PATH)
    print("Ecuacion:", ecuacion_ascii)
    print(f"R2   = {meta['metricas']['r2']:.4f}")
    print(f"MAE  = {meta['metricas']['mae']:.3f}")
    print(f"RMSE = {meta['metricas']['rmse']:.3f}")
    print(f"División: {meta['metodo_division']}")
    print(
        f"Train: {meta['conjunto_entrenamiento']['n_registros']} registros / "
        f"Test: {meta['conjunto_prueba']['n_registros']} registros"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
