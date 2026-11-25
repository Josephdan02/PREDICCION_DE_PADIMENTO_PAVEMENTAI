"""Pruebas de generación, carga, validación y limpieza del dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data_loader, dataset_generator, preprocessing


# ---------------------------------------------------------------------------
# Generación y carga
# ---------------------------------------------------------------------------
def test_dataset_has_required_columns(synthetic_df: pd.DataFrame):
    assert list(synthetic_df.columns) == config.DATASET_COLUMNS


def test_dataset_has_at_least_50_records(synthetic_df: pd.DataFrame):
    assert len(synthetic_df) >= 50


def test_dataset_covers_all_pathologies_and_severities(synthetic_df: pd.DataFrame):
    assert set(synthetic_df["tipo_patologia"]) == set(config.PATOLOGIAS)
    assert set(synthetic_df["severidad"]) == set(config.SEVERIDADES)


def test_dataset_records_are_marked_as_simulated(synthetic_df: pd.DataFrame):
    assert synthetic_df["tipo_registro"].str.upper().eq(config.TIPO_REGISTRO_SIMULADO).all()


def test_dataset_has_multiple_inspections_per_tramo(synthetic_df: pd.DataFrame):
    conteo = synthetic_df.groupby("codigo_tramo").size()
    assert conteo.max() > 1, "Se requieren varios registros por tramo para la división agrupada"


def test_pci_within_valid_range(synthetic_df: pd.DataFrame):
    assert synthetic_df["pci_observado"].between(0, 100).all()


def test_generation_is_reproducible():
    a = dataset_generator.generate_dataset(n_tramos=20, seed=7)
    b = dataset_generator.generate_dataset(n_tramos=20, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_generation_has_variability(synthetic_df: pd.DataFrame):
    """La relación PCI-antigüedad no debe ser determinista (debe existir ruido)."""
    residuos = synthetic_df["pci_observado"] - np.polyval(
        np.polyfit(synthetic_df["antiguedad_anios"], synthetic_df["pci_observado"], 1),
        synthetic_df["antiguedad_anios"],
    )
    assert residuos.std() > 1.0


def test_pci_decreases_with_age_on_average(synthetic_df: pd.DataFrame):
    correlacion = synthetic_df["antiguedad_anios"].corr(synthetic_df["pci_observado"])
    assert correlacion < -0.5


def test_load_dataset_from_file(dataset_file):
    df = data_loader.load_dataset(dataset_file)
    assert len(df) >= 50
    assert pd.api.types.is_datetime64_any_dtype(df["fecha_inspeccion"])


def test_load_dataset_creates_file_when_missing(tmp_path, monkeypatch):
    destino = tmp_path / "nuevo.csv"
    df = data_loader.load_dataset(destino)
    assert destino.exists()
    assert len(df) >= 50


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def test_inspect_dataframe_flags_valid_dataset(synthetic_df: pd.DataFrame):
    reporte = preprocessing.inspect_dataframe(synthetic_df)
    assert reporte.es_valido
    assert reporte.total_nulos == 0
    assert reporte.duplicados == 0
    assert reporte.columnas_faltantes == []


def test_inspect_dataframe_detects_missing_columns(synthetic_df: pd.DataFrame):
    reporte = preprocessing.inspect_dataframe(synthetic_df.drop(columns=["severidad"]))
    assert "severidad" in reporte.columnas_faltantes
    assert not reporte.es_valido


def test_inspect_dataframe_detects_out_of_range_pci(synthetic_df: pd.DataFrame):
    alterado = synthetic_df.copy()
    alterado.loc[0, "pci_observado"] = 140
    reporte = preprocessing.inspect_dataframe(alterado)
    assert reporte.errores_rango
    assert not reporte.es_valido


def test_inspect_dataframe_detects_invalid_severity(synthetic_df: pd.DataFrame):
    alterado = synthetic_df.copy()
    alterado.loc[1, "severidad"] = "Extrema"
    reporte = preprocessing.inspect_dataframe(alterado)
    assert reporte.categorias_invalidas
    assert not reporte.es_valido


def test_inspect_dataframe_counts_duplicates(synthetic_df: pd.DataFrame):
    duplicado = pd.concat([synthetic_df, synthetic_df.head(3)], ignore_index=True)
    reporte = preprocessing.inspect_dataframe(duplicado)
    assert reporte.duplicados >= 3


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
def test_clean_dataframe_removes_duplicates(synthetic_df: pd.DataFrame):
    sucio = pd.concat([synthetic_df, synthetic_df.head(5)], ignore_index=True)
    limpio = preprocessing.clean_dataframe(sucio)
    assert len(limpio) == len(synthetic_df)


def test_clean_dataframe_handles_nulls(synthetic_df: pd.DataFrame):
    sucio = synthetic_df.copy()
    sucio.loc[0, "antiguedad_anios"] = np.nan
    sucio.loc[1, "volumen_transito"] = np.nan
    limpio = preprocessing.clean_dataframe(sucio)
    assert limpio["antiguedad_anios"].notna().all()
    assert limpio["volumen_transito"].notna().all()


def test_clean_dataframe_drops_rows_without_target(synthetic_df: pd.DataFrame):
    sucio = synthetic_df.copy()
    sucio.loc[2, "pci_observado"] = np.nan
    limpio = preprocessing.clean_dataframe(sucio)
    assert len(limpio) == len(synthetic_df) - 1


def test_clean_dataframe_clips_pci(synthetic_df: pd.DataFrame):
    sucio = synthetic_df.copy()
    sucio.loc[0, "pci_observado"] = 180.0
    sucio.loc[1, "pci_observado"] = -20.0
    limpio = preprocessing.clean_dataframe(sucio)
    assert limpio["pci_observado"].between(0, 100).all()


def test_clean_dataframe_drops_negative_age(synthetic_df: pd.DataFrame):
    sucio = synthetic_df.copy()
    sucio.loc[3, "antiguedad_anios"] = -4.0
    limpio = preprocessing.clean_dataframe(sucio)
    assert (limpio["antiguedad_anios"] >= 0).all()
    assert len(limpio) == len(synthetic_df) - 1


def test_prepare_training_data_shapes(synthetic_df: pd.DataFrame):
    X, y, limpio = preprocessing.prepare_training_data(synthetic_df)
    assert X.shape == (len(limpio), 1)
    assert y.shape == (len(limpio),)
    assert X.dtype == float and y.dtype == float


def test_clamp_pci():
    valores = preprocessing.clamp_pci([-10, 0, 55.5, 100, 130])
    assert list(valores) == [0.0, 0.0, 55.5, 100.0, 100.0]
