"""Pruebas de predicción, límites del PCI y simulador."""

from __future__ import annotations

import numpy as np
import pytest

from src import config, prediction


def test_model_is_available_after_training(trained_model):
    assert prediction.model_is_available()
    import joblib

    modelo = joblib.load(config.MODEL_PATH)
    info = prediction.model_info()
    assert info["pendiente"] == pytest.approx(float(modelo.coef_[0]))
    assert info["intercepto"] == pytest.approx(float(modelo.intercept_))


def test_missing_model_raises_clear_error(isolated_storage):
    prediction._cached_model.cache_clear()
    assert not prediction.model_is_available()
    with pytest.raises(prediction.ModelNotTrainedError):
        prediction.load_model()


def test_prediction_matches_sklearn(trained_model):
    import joblib

    modelo = joblib.load(config.MODEL_PATH)
    for edad in (0.0, 3.5, 12.0, 27.0):
        esperado = float(modelo.predict(np.array([[edad]]))[0])
        assert prediction.predict_pci(edad) == pytest.approx(esperado)


def test_prediction_uses_model_not_hardcoded_values(trained_model):
    """Cambiar la antigüedad debe cambiar la predicción (no hay valores fijos)."""
    valores = {prediction.predict_pci(e) for e in (5.0, 10.0, 20.0)}
    assert len(valores) == 3


def test_predict_many_vectorized(trained_model):
    resultado = prediction.predict_many([1.0, 2.0, 3.0])
    assert resultado.shape == (3,)
    assert resultado[0] > resultado[2]  # el PCI decae con la antigüedad


def test_negative_age_is_rejected(trained_model):
    with pytest.raises(ValueError):
        prediction.predict_pci(-1.0)


def test_invalid_age_is_rejected(trained_model):
    with pytest.raises(ValueError):
        prediction.predict_pci(float("nan"))


def test_clamp_pci_limits():
    assert prediction.clamp_pci(-15) == 0.0
    assert prediction.clamp_pci(150) == 100.0
    assert prediction.clamp_pci(63.4) == 63.4


def test_scenario_returns_expected_structure(trained_model):
    escenario = prediction.predict_scenario(10.0, 5)
    claves = {
        "antiguedad_actual", "antiguedad_futura", "horizonte_anios",
        "pci_actual", "pci_futuro", "pci_actual_raw", "pci_futuro_raw",
        "variacion", "estado_actual", "estado_futuro", "advertencias",
    }
    assert claves.issubset(escenario.keys())
    assert escenario["antiguedad_futura"] == pytest.approx(15.0)
    assert 0 <= escenario["pci_actual"] <= 100
    assert 0 <= escenario["pci_futuro"] <= 100


def test_scenario_variation_is_negative(trained_model):
    escenario = prediction.predict_scenario(8.0, 5)
    assert escenario["variacion"] < 0


def test_scenario_warns_about_extrapolation(trained_model):
    maximo = trained_model["metadata"]["rango_antiguedad_entrenamiento"]["max"]
    escenario = prediction.predict_scenario(maximo + 15, 5)
    assert any("fuera del rango" in aviso for aviso in escenario["advertencias"])


def test_scenario_keeps_raw_value_when_clipped(trained_model):
    """Con antigüedad muy alta el PCI matemático puede salir de 0-100."""
    escenario = prediction.predict_scenario(80.0, 10)
    if escenario["pci_futuro_raw"] < 0 or escenario["pci_futuro_raw"] > 100:
        assert escenario["pci_futuro"] in (0.0, 100.0)
        assert any("fuera de 0-100" in aviso for aviso in escenario["advertencias"])


def test_classify_matches_configured_ranges():
    assert prediction.classify(95) == "Excelente"
    assert prediction.classify(75) == "Muy bueno"
    assert prediction.classify(60) == "Bueno"
    assert prediction.classify(45) == "Regular"
    assert prediction.classify(30) == "Malo"
    assert prediction.classify(15) == "Muy malo"
    assert prediction.classify(3) == "Fallado"


def test_classification_boundaries():
    assert prediction.classify(85) == "Excelente"
    assert prediction.classify(84.9) == "Muy bueno"
    assert prediction.classify(70) == "Muy bueno"
    assert prediction.classify(0) == "Fallado"
    assert prediction.classify(100) == "Excelente"


def test_classification_clamps_out_of_range_values():
    assert prediction.classify(130) == "Excelente"
    assert prediction.classify(-5) == "Fallado"
