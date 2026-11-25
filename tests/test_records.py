"""Pruebas de registro de vías, patologías, persistencia y reportes."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import (
    data_loader,
    pathology_service,
    report_generator,
    road_service,
)


# ---------------------------------------------------------------------------
# Vías
# ---------------------------------------------------------------------------
def _via_valida(**overrides) -> dict:
    base = {
        "codigo_tramo": "hjnu-101",
        "nombre_via": "Av. Universitaria",
        "distrito": "Huánuco",
        "ubicacion": "Cuadra 1-8",
        "longitud_m": 850,
        "antiguedad_anios": 7,
        "volumen_transito": 4200,
        "fecha_inspeccion": "2025-03-15",
    }
    base.update(overrides)
    return base


def test_create_road_persists_and_normalizes(isolated_storage):
    vias = road_service.create_road(_via_valida())
    assert len(vias) == 1
    assert vias.iloc[0]["codigo_tramo"] == "HJNU-101"  # se normaliza a mayúsculas
    # Persistencia: se relee desde disco
    recargado = data_loader.load_roads()
    assert len(recargado) == 1
    assert recargado.iloc[0]["nombre_via"] == "Av. Universitaria"


def test_create_road_requires_mandatory_fields(isolated_storage):
    with pytest.raises(road_service.ValidationError):
        road_service.create_road(_via_valida(nombre_via=""))


def test_create_road_rejects_duplicate_code(isolated_storage):
    road_service.create_road(_via_valida())
    with pytest.raises(road_service.ValidationError):
        road_service.create_road(_via_valida(nombre_via="Otra vía"))


def test_create_road_validates_length(isolated_storage):
    with pytest.raises(road_service.ValidationError):
        road_service.create_road(_via_valida(longitud_m=0))


def test_create_road_validates_age_range(isolated_storage):
    with pytest.raises(road_service.ValidationError):
        road_service.create_road(_via_valida(antiguedad_anios=120))


def test_create_road_validates_traffic(isolated_storage):
    with pytest.raises(road_service.ValidationError):
        road_service.create_road(_via_valida(volumen_transito=-10))


def test_create_road_rejects_short_code(isolated_storage):
    with pytest.raises(road_service.ValidationError):
        road_service.create_road(_via_valida(codigo_tramo="A1"))


def test_update_road_changes_values(isolated_storage):
    road_service.create_road(_via_valida())
    vias = road_service.update_road("HJNU-101", _via_valida(longitud_m=1200))
    assert len(vias) == 1
    assert float(vias.iloc[0]["longitud_m"]) == 1200.0


def test_delete_road(isolated_storage):
    road_service.create_road(_via_valida())
    road_service.create_road(_via_valida(codigo_tramo="HJNU-102", nombre_via="Jr. Lima"))
    vias = road_service.delete_road("HJNU-101")
    assert len(vias) == 1
    assert data_loader.load_roads().iloc[0]["codigo_tramo"] == "HJNU-102"


def test_delete_unknown_road_is_safe(isolated_storage):
    vias = road_service.delete_road("NO-EXISTE")
    assert vias.empty


def test_filter_roads_by_text_and_district(isolated_storage):
    road_service.create_road(_via_valida())
    road_service.create_road(_via_valida(codigo_tramo="AMAR-001", nombre_via="Av. Perú", distrito="Amarilis"))
    todas = data_loader.load_roads()
    assert len(road_service.filter_roads(todas, texto="universitaria")) == 1
    assert len(road_service.filter_roads(todas, distrito="Amarilis")) == 1
    assert len(road_service.filter_roads(todas, texto="av")) == 2


def test_road_summary_counts(isolated_storage):
    road_service.create_road(_via_valida())
    road_service.create_road(_via_valida(codigo_tramo="AMAR-001", distrito="Amarilis", longitud_m=1150))
    resumen = road_service.road_summary()
    assert resumen["total_vias"] == 2
    assert resumen["distritos"] == 2
    assert resumen["longitud_total_km"] == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# Patologías
# ---------------------------------------------------------------------------
def _patologia_valida(**overrides) -> dict:
    base = {
        "codigo_tramo": "HJNU-101",
        "tipo_patologia": "Grietas longitudinales",
        "severidad": "Media",
        "cantidad_fallas": 12.5,
        "descripcion": "Fisuras paralelas al eje.",
        "fecha_inspeccion": "2025-04-02",
    }
    base.update(overrides)
    return base


def test_create_pathology_persists(isolated_storage):
    road_service.create_road(_via_valida())
    registros = pathology_service.create_pathology(_patologia_valida())
    assert len(registros) == 1
    assert registros.iloc[0]["id_patologia"] == "PAT-0001"
    assert len(data_loader.load_pathologies()) == 1


def test_pathology_ids_increment(isolated_storage):
    road_service.create_road(_via_valida())
    pathology_service.create_pathology(_patologia_valida())
    registros = pathology_service.create_pathology(_patologia_valida(severidad="Alta"))
    assert registros.iloc[1]["id_patologia"] == "PAT-0002"


def test_pathology_requires_existing_tramo(isolated_storage):
    with pytest.raises(pathology_service.ValidationError):
        pathology_service.create_pathology(_patologia_valida(codigo_tramo="FANTASMA-01"))


def test_pathology_accepts_tramo_from_dataset(isolated_storage):
    codigo = data_loader.load_dataset().iloc[0]["codigo_tramo"]
    registros = pathology_service.create_pathology(_patologia_valida(codigo_tramo=codigo))
    assert len(registros) == 1


def test_pathology_rejects_invalid_type(isolated_storage):
    road_service.create_road(_via_valida())
    with pytest.raises(pathology_service.ValidationError):
        pathology_service.create_pathology(_patologia_valida(tipo_patologia="Baches"))


def test_pathology_rejects_invalid_severity(isolated_storage):
    road_service.create_road(_via_valida())
    with pytest.raises(pathology_service.ValidationError):
        pathology_service.create_pathology(_patologia_valida(severidad="Crítica"))


def test_pathology_rejects_zero_quantity(isolated_storage):
    road_service.create_road(_via_valida())
    with pytest.raises(pathology_service.ValidationError):
        pathology_service.create_pathology(_patologia_valida(cantidad_fallas=0))


def test_pathology_validates_date(isolated_storage):
    road_service.create_road(_via_valida())
    with pytest.raises(pathology_service.ValidationError):
        pathology_service.create_pathology(_patologia_valida(fecha_inspeccion="no-es-fecha"))


def test_pathology_with_photo_is_saved(isolated_storage):
    road_service.create_road(_via_valida())
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    registros = pathology_service.create_pathology(
        _patologia_valida(), photo_bytes=png, photo_name="grieta.png"
    )
    ruta = registros.iloc[0]["fotografia"]
    assert ruta.endswith(".png")
    assert pathology_service.photo_path(ruta) is not None
    assert pathology_service.photo_path(ruta).exists()


def test_photo_rejects_invalid_extension(isolated_storage):
    road_service.create_road(_via_valida())
    with pytest.raises(pathology_service.ValidationError):
        pathology_service.create_pathology(
            _patologia_valida(), photo_bytes=b"data", photo_name="archivo.exe"
        )


def test_delete_pathology(isolated_storage):
    road_service.create_road(_via_valida())
    pathology_service.create_pathology(_patologia_valida())
    restantes = pathology_service.delete_pathology("PAT-0001")
    assert restantes.empty
    assert data_loader.load_pathologies().empty


def test_pathology_stats(isolated_storage):
    road_service.create_road(_via_valida())
    pathology_service.create_pathology(_patologia_valida())
    pathology_service.create_pathology(_patologia_valida(severidad="Alta", tipo_patologia="Losa dividida"))
    stats = pathology_service.pathology_stats()
    assert stats["total"] == 2
    assert stats["severidad_alta"] == 1


def test_filter_pathologies(isolated_storage):
    road_service.create_road(_via_valida())
    pathology_service.create_pathology(_patologia_valida())
    pathology_service.create_pathology(_patologia_valida(tipo_patologia="Escalonamiento", severidad="Alta"))
    df = data_loader.load_pathologies()
    assert len(pathology_service.filter_pathologies(df, severidad="Alta")) == 1
    assert len(pathology_service.filter_pathologies(df, patologia="Escalonamiento")) == 1
    assert len(pathology_service.filter_pathologies(df, tramo="HJNU-101")) == 2


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------
def test_pdf_report_is_generated(trained_model):
    pdf = report_generator.build_pdf_report(include_history=True, horizonte=5, antiguedad_escenario=12)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 5000


def test_pdf_report_saved_to_disk(trained_model, tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    nombre, contenido = report_generator.save_pdf_report(horizonte=3, antiguedad_escenario=10)
    assert nombre.endswith(".pdf")
    assert (tmp_path / nombre).exists()
    assert contenido.startswith(b"%PDF")


def test_pdf_report_without_model(isolated_storage):
    """Sin modelo entrenado el informe se genera igual, indicando el faltante."""
    pdf = report_generator.build_pdf_report(horizonte=5, antiguedad_escenario=10)
    assert pdf.startswith(b"%PDF")


def test_csv_export(trained_model):
    df = data_loader.load_dataset()
    csv = report_generator.export_csv(df.head(10))
    assert csv.startswith(b"\xef\xbb\xbf")  # BOM para Excel
    assert b"pci_observado" in csv


def test_json_export_payload(trained_model):
    payload = report_generator.build_export_payload()
    texto = report_generator.export_json(payload)
    datos = json.loads(texto.decode("utf-8"))
    assert datos["aplicacion"] == "PAVEMENT AI"
    assert "r2" in datos["metricas_modelo"]
    assert datos["indicadores"]["total_inspecciones"] > 0


def test_system_indicators_are_computed(trained_model):
    indicadores = report_generator.system_indicators()
    df = data_loader.load_dataset()
    assert indicadores["total_inspecciones"] == len(df)
    assert indicadores["total_tramos"] == df["codigo_tramo"].nunique()
    assert 0 <= indicadores["pci_promedio"] <= 100


def test_dataset_add_records_persists(isolated_storage):
    df = data_loader.load_dataset()
    nuevo = df.iloc[[0]].copy()
    nuevo["id_registro"] = "REG-TEST-01"
    resultado = data_loader.add_dataset_records(nuevo)
    assert "REG-TEST-01" in resultado["id_registro"].values
    assert "REG-TEST-01" in data_loader.load_dataset()["id_registro"].values
