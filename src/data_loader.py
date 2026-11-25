"""Capa de acceso a datos (CSV locales).

Toda lectura/escritura de archivos CSV pasa por aquí. La persistencia es local
y sencilla: cada entidad vive en un CSV dentro de ``data/`` o ``storage/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src import config
from src.dataset_generator import write_dataset

ROAD_COLUMNS = [
    "codigo_tramo",
    "nombre_via",
    "distrito",
    "ubicacion",
    "longitud_m",
    "antiguedad_anios",
    "volumen_transito",
    "fecha_inspeccion",
    "fecha_registro",
]

PATHOLOGY_COLUMNS = [
    "id_patologia",
    "codigo_tramo",
    "tipo_patologia",
    "severidad",
    "cantidad_fallas",
    "descripcion",
    "fecha_inspeccion",
    "fotografia",
    "fecha_registro",
]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Escritura atómica: escribe en un temporal y reemplaza el original."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Dataset de inspecciones
# ---------------------------------------------------------------------------
def dataset_exists() -> bool:
    return config.DATASET_PATH.exists()


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Carga el dataset de inspecciones; lo genera si aún no existe."""
    path = path or config.DATASET_PATH
    if not path.exists():
        write_dataset(path)
    df = pd.read_csv(path, encoding="utf-8")
    if "fecha_inspeccion" in df.columns:
        df["fecha_inspeccion"] = pd.to_datetime(df["fecha_inspeccion"], errors="coerce")
    return df


def save_dataset(df: pd.DataFrame, path: Path | None = None) -> None:
    path = path or config.DATASET_PATH
    df.to_csv(path, index=False, encoding="utf-8")


def add_dataset_records(records: pd.DataFrame | dict | list[dict], path: Path | None = None) -> pd.DataFrame:
    """Añade registros de inspección al dataset principal y devuelve el resultado."""
    path = path or config.DATASET_PATH
    base = load_dataset(path)
    nuevos = pd.DataFrame(records)
    combinado = pd.concat([base, nuevos], ignore_index=True)
    combinado = combinado.drop_duplicates(subset=["id_registro"], keep="last")
    save_dataset(combinado, path)
    return combinado


# ---------------------------------------------------------------------------
# Vías / tramos
# ---------------------------------------------------------------------------
def load_roads(path: Path | None = None) -> pd.DataFrame:
    path = path or config.ROADS_PATH
    if not path.exists():
        return pd.DataFrame(columns=ROAD_COLUMNS)
    df = pd.read_csv(path, encoding="utf-8", dtype={"codigo_tramo": str})
    for col in ROAD_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[ROAD_COLUMNS]


def save_roads(df: pd.DataFrame, path: Path | None = None) -> None:
    path = path or config.ROADS_PATH
    _write_csv(df[ROAD_COLUMNS], path)


def upsert_road(road: dict, path: Path | None = None) -> pd.DataFrame:
    """Crea o actualiza un tramo identificado por ``codigo_tramo``."""
    df = load_roads(path)
    fila = {col: road.get(col, "") for col in ROAD_COLUMNS}
    codigo = str(fila["codigo_tramo"])
    if not df.empty and (df["codigo_tramo"].astype(str) == codigo).any():
        idx = df.index[df["codigo_tramo"].astype(str) == codigo][0]
        for col, value in fila.items():
            df.at[idx, col] = value
    else:
        df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    save_roads(df, path)
    return df


def delete_road(codigo_tramo: str, path: Path | None = None) -> pd.DataFrame:
    df = load_roads(path)
    df = df[df["codigo_tramo"].astype(str) != str(codigo_tramo)]
    save_roads(df, path)
    return df


# ---------------------------------------------------------------------------
# Patologías
# ---------------------------------------------------------------------------
def load_pathologies(path: Path | None = None) -> pd.DataFrame:
    path = path or config.PATHOLOGIES_PATH
    if not path.exists():
        return pd.DataFrame(columns=PATHOLOGY_COLUMNS)
    df = pd.read_csv(path, encoding="utf-8", dtype={"id_patologia": str, "codigo_tramo": str})
    for col in PATHOLOGY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[PATHOLOGY_COLUMNS]


def save_pathologies(df: pd.DataFrame, path: Path | None = None) -> None:
    path = path or config.PATHOLOGIES_PATH
    _write_csv(df[PATHOLOGY_COLUMNS], path)


def next_pathology_id(df: pd.DataFrame) -> str:
    if df.empty:
        return "PAT-0001"
    numeros = []
    for valor in df["id_patologia"].astype(str):
        sufijo = valor.split("-")[-1]
        if sufijo.isdigit():
            numeros.append(int(sufijo))
    return f"PAT-{(max(numeros) + 1 if numeros else 1):04d}"


def add_pathology(record: dict, path: Path | None = None) -> pd.DataFrame:
    df = load_pathologies(path)
    fila = {col: record.get(col, "") for col in PATHOLOGY_COLUMNS}
    fila["id_patologia"] = fila["id_patologia"] or next_pathology_id(df)
    df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    save_pathologies(df, path)
    return df


def delete_pathology(id_patologia: str, path: Path | None = None) -> pd.DataFrame:
    df = load_pathologies(path)
    df = df[df["id_patologia"].astype(str) != str(id_patologia)]
    save_pathologies(df, path)
    return df


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def all_known_tramos() -> list[str]:
    """Tramos conocidos: unión de vías registradas y dataset de inspecciones."""
    roads = load_roads()
    dataset = load_dataset()
    valores = set(roads["codigo_tramo"].astype(str).tolist()) | set(dataset["codigo_tramo"].astype(str).tolist())
    return sorted(v for v in valores if v and v.lower() != "nan")


def tramo_options() -> dict[str, str]:
    """Mapeo etiqueta legible -> código de tramo para los selectores."""
    roads = load_roads()
    opciones: dict[str, str] = {}
    if not roads.empty:
        for _, row in roads.iterrows():
            etiqueta = f"{row['codigo_tramo']} · {row['nombre_via']}" if str(row["nombre_via"]).strip() else str(row["codigo_tramo"])
            opciones[etiqueta] = str(row["codigo_tramo"])
    for codigo in all_known_tramos():
        if codigo not in opciones.values():
            opciones[codigo] = codigo
    return opciones
