"""Preprocesamiento del dataset con Pandas y NumPy.

Responsabilidades:
- Revisar tipos de datos.
- Identificar valores nulos.
- Detectar duplicados.
- Validar rangos de las variables del dominio.
- Limpiar registros.
- Preparar X e y para el entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config

NUMERIC_COLUMNS = ["antiguedad_anios", "volumen_transito", "cantidad_fallas", "pci_observado"]


@dataclass
class ValidationReport:
    """Resumen legible del estado del dataset."""

    total_registros: int = 0
    nulos_por_columna: dict = field(default_factory=dict)
    total_nulos: int = 0
    duplicados: int = 0
    duplicados_por_id: int = 0
    errores_rango: list = field(default_factory=list)
    categorias_invalidas: list = field(default_factory=list)
    columnas_faltantes: list = field(default_factory=list)
    es_valido: bool = True
    observaciones: list = field(default_factory=list)


def inspect_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Revisa tipos, nulos, duplicados y rangos sin modificar los datos."""
    report = ValidationReport(total_registros=len(df))

    report.columnas_faltantes = [c for c in config.DATASET_COLUMNS if c not in df.columns]
    if report.columnas_faltantes:
        report.es_valido = False
        report.observaciones.append(f"Faltan columnas obligatorias: {', '.join(report.columnas_faltantes)}")

    serie_nulos = df.isna().sum()
    report.nulos_por_columna = {k: int(v) for k, v in serie_nulos.items() if int(v) > 0}
    report.total_nulos = int(serie_nulos.sum())

    report.duplicados = int(df.duplicated().sum())
    if "id_registro" in df.columns:
        report.duplicados_por_id = int(df["id_registro"].duplicated().sum())

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        valores = pd.to_numeric(df[col], errors="coerce")
        if col == "pci_observado":
            fuera = df[(valores < 0) | (valores > 100)]
        else:
            fuera = df[valores < 0]
        for idx in fuera.index[:20]:
            report.errores_rango.append(f"{col} fuera de rango en la fila {idx}")

    if "antiguedad_anios" in df.columns:
        fuera_anios = pd.to_numeric(df["antiguedad_anios"], errors="coerce")
        for idx in df[fuera_anios > 60].index[:20]:
            report.errores_rango.append(f"antiguedad_anios inusualmente alta en la fila {idx}")

    if "severidad" in df.columns:
        invalidas = sorted(set(df["severidad"].dropna().astype(str)) - set(config.SEVERIDADES))
        if invalidas:
            report.categorias_invalidas.append(f"severidad: {invalidas}")
    if "tipo_patologia" in df.columns:
        invalidas = sorted(set(df["tipo_patologia"].dropna().astype(str)) - set(config.PATOLOGIAS))
        if invalidas:
            report.categorias_invalidas.append(f"tipo_patologia: {invalidas}")

    if report.errores_rango or report.categorias_invalidas:
        report.es_valido = False
    return report


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset: tipos, nulos numéricos, duplicados y rangos del PCI.

    No elimina filas con PCI nulo de forma silenciosa sin antes descartar
    registros irrecuperables; devuelve una copia limpia.
    """
    limpio = df.copy()

    for col in NUMERIC_COLUMNS + ["id_registro", "codigo_tramo"]:
        if col in limpio.columns and col not in NUMERIC_COLUMNS:
            limpio[col] = limpio[col].astype(str).str.strip()

    if "fecha_inspeccion" in limpio.columns:
        limpio["fecha_inspeccion"] = pd.to_datetime(limpio["fecha_inspeccion"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in limpio.columns:
            limpio[col] = pd.to_numeric(limpio[col], errors="coerce")

    # Imputación con la mediana para antigüedad / volumen (valores estructurales)
    for col in ("antiguedad_anios", "volumen_transito"):
        if col in limpio.columns and limpio[col].isna().any():
            limpio[col] = limpio[col].fillna(limpio[col].median())

    # Registros sin PCI observado o sin tránsito no sirven para entrenar
    limpio = limpio.dropna(subset=[c for c in (config.TARGET_COLUMN, config.FEATURE_COLUMN) if c in limpio.columns])

    # Cantidad de fallas: mínimo 1 y entero
    if "cantidad_fallas" in limpio.columns:
        limpio["cantidad_fallas"] = limpio["cantidad_fallas"].fillna(1).clip(lower=1).round().astype(int)

    # Recorte de rango del PCI
    if config.TARGET_COLUMN in limpio.columns:
        limpio[config.TARGET_COLUMN] = limpio[config.TARGET_COLUMN].clip(config.PCI_MIN, config.PCI_MAX)

    # Antigüedad / volumen negativos no son físicamente posibles
    if config.FEATURE_COLUMN in limpio.columns:
        limpio = limpio[limpio[config.FEATURE_COLUMN] >= 0]

    # Duplicados exactos y por identificador
    limpio = limpio.drop_duplicates()
    if "id_registro" in limpio.columns:
        limpio = limpio.drop_duplicates(subset=["id_registro"], keep="first")

    # Tipos finales
    if "severidad" in limpio.columns:
        limpio["severidad"] = limpio["severidad"].fillna("Media").astype(str).str.strip().str.title()
    if "tipo_registro" in limpio.columns:
        limpio["tipo_registro"] = limpio["tipo_registro"].fillna(config.TIPO_REGISTRO_SIMULADO).astype(str).str.upper()

    return limpio.reset_index(drop=True)


def prepare_training_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Devuelve (X, y, dataframe_limpio) listos para ``LinearRegression``."""
    limpio = clean_dataframe(df)
    X = limpio[[config.FEATURE_COLUMN]].to_numpy(dtype=float)
    y = limpio[config.TARGET_COLUMN].to_numpy(dtype=float)
    return X, y, limpio


def prepare_dataframe_only(df: pd.DataFrame, path=None) -> pd.DataFrame:
    """Atajo para preparar y persistir el dataset limpio (usado por tests/CLI)."""
    limpio = clean_dataframe(df)
    if path is not None:
        limpio.to_csv(path, index=False, encoding="utf-8")
    return limpio


def clamp_pci(values) -> np.ndarray:
    """Limita valores de PCI al rango [0, 100] para su visualización."""
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, config.PCI_MIN, config.PCI_MAX)
