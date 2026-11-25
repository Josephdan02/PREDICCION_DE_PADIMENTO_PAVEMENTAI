"""Configuración compartida de Pytest.

Añade la raíz del proyecto al ``sys.path`` y proporciona fixtures que usan
archivos temporales: las pruebas nunca sobrescriben los datos reales de
``data/``, ``storage/`` ni ``models/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config, data_loader, dataset_generator  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Dataset sintético reproducible generado en memoria."""
    return dataset_generator.generate_dataset(n_tramos=55, seed=config.RANDOM_STATE)


@pytest.fixture()
def dataset_file(tmp_path: Path) -> Path:
    """Archivo CSV temporal con el dataset sintético."""
    path = tmp_path / "pavimentos.csv"
    dataset_generator.generate_dataset(n_tramos=45).to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture()
def isolated_storage(tmp_path: Path, monkeypatch) -> dict:
    """Redirige las rutas de persistencia a un directorio temporal.

    Evita que las pruebas de registro modifiquen ``storage/`` del proyecto.
    """
    rutas = {
        "roads": tmp_path / "vias.csv",
        "pathologies": tmp_path / "patologias.csv",
        "uploads": tmp_path / "uploads",
        "dataset": tmp_path / "pavimentos.csv",
        "model": tmp_path / "model.joblib",
        "metadata": tmp_path / "metadata.json",
    }
    rutas["uploads"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "ROADS_PATH", rutas["roads"])
    monkeypatch.setattr(config, "PATHOLOGIES_PATH", rutas["pathologies"])
    monkeypatch.setattr(config, "PATHOLOGY_UPLOADS_DIR", rutas["uploads"])
    monkeypatch.setattr(config, "DATASET_PATH", rutas["dataset"])
    monkeypatch.setattr(config, "MODEL_PATH", rutas["model"])
    monkeypatch.setattr(config, "METADATA_PATH", rutas["metadata"])

    dataset_generator.generate_dataset(n_tramos=45).to_csv(rutas["dataset"], index=False, encoding="utf-8")
    return rutas


@pytest.fixture()
def trained_model(isolated_storage):
    """Entrena y guarda un modelo real en el almacenamiento temporal."""
    from src import train_model

    df = data_loader.load_dataset()
    resultado = train_model.train_model(df, save=True)
    from src import prediction

    prediction._cached_model.cache_clear()
    return resultado
