"""Pruebas del entrenamiento, métricas y persistencia del modelo."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src import config, data_loader, evaluation, prediction, train_model


def test_model_trains_and_returns_linear_regression(trained_model):
    modelo = trained_model["model"]
    assert isinstance(modelo, LinearRegression)
    assert modelo.coef_.shape == (1,)
    assert np.isfinite(modelo.intercept_)


def test_metrics_are_real_numbers(trained_model):
    metricas = trained_model["metadata"]["metricas"]
    for clave in ("r2", "mae", "rmse"):
        assert clave in metricas
        assert np.isfinite(metricas[clave])
    assert metricas["mae"] >= 0
    assert metricas["rmse"] >= 0
    # El RMSE nunca es menor que el MAE (propiedad matemática)
    assert metricas["rmse"] >= metricas["mae"] - 1e-9


def test_metrics_match_manual_computation(trained_model):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = trained_model["test_df"][config.TARGET_COLUMN].to_numpy()
    y_pred = trained_model["y_pred_test"]
    esperado = {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    for clave, valor in esperado.items():
        assert trained_model["metadata"]["metricas"][clave] == pytest.approx(valor, rel=1e-6)


def test_group_split_keeps_tramos_disjoint(trained_model):
    train = trained_model["train_df"]
    test = trained_model["test_df"]
    assert set(train["codigo_tramo"]).isdisjoint(set(test["codigo_tramo"])), (
        "Un tramo no debe aparecer en entrenamiento y prueba a la vez"
    )
    assert "Group" in trained_model["metadata"]["metodo_division"]


def test_split_sizes_are_about_80_20(trained_model):
    n_train = len(trained_model["train_df"])
    n_test = len(trained_model["test_df"])
    total = n_train + n_test
    assert 0.15 <= n_test / total <= 0.30


def test_model_file_and_metadata_are_saved(trained_model):
    assert config.MODEL_PATH.exists()
    assert config.METADATA_PATH.exists()
    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["algoritmo"] == "sklearn.linear_model.LinearRegression"
    assert metadata["variable_independiente"] == "antiguedad_anios"
    assert metadata["variable_dependiente"] == "pci_observado"
    assert "ecuacion" in metadata


def test_model_persists_and_reloads_identical(trained_model):
    cargado = joblib.load(config.MODEL_PATH)
    original = trained_model["model"]
    assert cargado.coef_[0] == pytest.approx(original.coef_[0])
    assert cargado.intercept_ == pytest.approx(original.intercept_)
    muestra = np.array([[5.0], [15.0]])
    np.testing.assert_allclose(cargado.predict(muestra), original.predict(muestra))


def test_slope_is_negative(trained_model):
    """El PCI debe disminuir con la antigüedad en el dataset generado."""
    assert trained_model["metadata"]["pendiente"] < 0


def test_model_beats_mean_baseline(trained_model):
    """R² > 0 implica que el modelo supera la predicción con la media."""
    assert trained_model["metadata"]["metricas"]["r2"] > 0


def test_equation_text_format(trained_model):
    meta = trained_model["metadata"]
    assert meta["ecuacion"].startswith("PCI = ")
    assert "antigüedad" in meta["ecuacion"]


def test_slope_explanation_mentions_units(trained_model):
    texto = evaluation.slope_explanation(trained_model["metadata"]["pendiente"])
    assert "año" in texto


def test_calculate_metrics_perfect_prediction():
    y = [10.0, 20.0, 30.0, 40.0]
    metricas = evaluation.calculate_metrics(y, y)
    assert metricas["r2"] == pytest.approx(1.0)
    assert metricas["mae"] == pytest.approx(0.0)
    assert metricas["rmse"] == pytest.approx(0.0)


def test_calculate_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluation.calculate_metrics([1, 2, 3], [1, 2])


def test_calculate_metrics_rejects_empty():
    with pytest.raises(ValueError):
        evaluation.calculate_metrics([], [])


def test_residuals_sign():
    residuos = evaluation.residuals([10.0, 20.0], [12.0, 18.0])
    assert list(residuos) == [-2.0, 2.0]


def test_training_requires_enough_records(synthetic_df):
    with pytest.raises(ValueError):
        train_model.train_model(synthetic_df.head(5), save=False)
