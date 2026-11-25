"""Servicio de patologías: validación, fotografías y estadísticas.

IMPORTANTE: registrar una patología NO implica calcular un PCI oficial por
ASTM D6433. El PCI utilizado por el modelo es el ``pci_observado`` de una
evaluación técnica documentada (aquí, datos simulados).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src import config, data_loader

ALLOWED_IMAGE_TYPES = {"png", "jpg", "jpeg", "webp"}
MAX_IMAGE_MB = 10


class ValidationError(ValueError):
    """Error de validación de formulario, con mensajes legibles."""


def validate_pathology(data: dict, know_tramos: list[str] | None = None) -> dict:
    errores: list[str] = []

    codigo = str(data.get("codigo_tramo", "")).strip().upper()
    if not codigo:
        errores.append("Debes asociar la patología a un tramo.")
    else:
        tramos = know_tramos if know_tramos is not None else data_loader.all_known_tramos()
        if codigo not in tramos:
            errores.append(f"El tramo {codigo} no existe. Regístralo primero en 'Registro de vías'.")

    patologia = str(data.get("tipo_patologia", "")).strip()
    if patologia not in config.PATOLOGIAS:
        errores.append("Selecciona un tipo de patología válido.")

    severidad = str(data.get("severidad", "")).strip().title()
    if severidad not in config.SEVERIDADES:
        errores.append("Selecciona una severidad válida (Baja, Media o Alta).")

    try:
        cantidad = float(data.get("cantidad_fallas", 0))
        if cantidad <= 0:
            errores.append("La cantidad o extensión afectada debe ser mayor que 0.")
    except (TypeError, ValueError):
        errores.append("La cantidad o extensión afectada debe ser un número.")
        cantidad = 0.0

    descripcion = str(data.get("descripcion", "")).strip()
    if len(descripcion) > 500:
        errores.append("La descripción no debe superar los 500 caracteres.")

    fecha = data.get("fecha_inspeccion")
    if isinstance(fecha, (datetime, date)):
        fecha = fecha.strftime("%Y-%m-%d")
    try:
        fecha_dt = pd.to_datetime(fecha, errors="raise")
    except (ValueError, TypeError):
        errores.append("La fecha de inspección no es válida.")
        fecha_dt = pd.Timestamp.today()

    if errores:
        raise ValidationError(" ".join(errores))

    return {
        "codigo_tramo": codigo,
        "tipo_patologia": patologia,
        "severidad": severidad,
        "cantidad_fallas": cantidad,
        "descripcion": descripcion,
        "fecha_inspeccion": fecha_dt.strftime("%Y-%m-%d"),
    }


def save_photo(file_bytes: bytes, filename: str, codigo_tramo: str) -> str | None:
    """Guarda la fotografía en ``uploads/patologias`` y devuelve la ruta relativa."""
    if not file_bytes:
        return None
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_IMAGE_TYPES:
        raise ValidationError("Formato de imagen no admitido. Usa PNG, JPG, JPEG o WEBP.")
    if len(file_bytes) > MAX_IMAGE_MB * 1024 * 1024:
        raise ValidationError(f"La imagen supera el límite de {MAX_IMAGE_MB} MB.")

    config.ensure_directories()
    carpeta = config.PATHOLOGY_UPLOADS_DIR / str(codigo_tramo).upper()
    carpeta.mkdir(parents=True, exist_ok=True)
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = carpeta / f"{sello}_{Path(filename).stem[:40]}.{extension}"
    destino.write_bytes(file_bytes)
    try:
        return destino.relative_to(config.BASE_DIR).as_posix()
    except ValueError:
        # La carpeta de subidas no está dentro del proyecto (p. ej. en pruebas
        # con directorios temporales): se guarda la ruta absoluta.
        return destino.as_posix()


def create_pathology(data: dict, photo_bytes: bytes | None = None, photo_name: str | None = None) -> pd.DataFrame:
    """Valida y registra una patología; opcionalmente guarda la fotografía."""
    limpio = validate_pathology(data)
    foto = data.get("fotografia") or ""
    if photo_bytes:
        foto = save_photo(photo_bytes, photo_name or "foto.jpg", limpio["codigo_tramo"]) or foto
    limpio["fotografia"] = foto
    limpio["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return data_loader.add_pathology(limpio)


def delete_pathology(id_patologia: str) -> pd.DataFrame:
    if not str(id_patologia).strip():
        raise ValidationError("Debes indicar el identificador de la patología.")
    return data_loader.delete_pathology(id_patologia)


def photo_path(relative: str) -> Path | None:
    """Convierte la ruta guardada en ruta absoluta si el archivo existe."""
    if not relative or str(relative).lower() == "nan":
        return None
    ruta = Path(relative)
    if not ruta.is_absolute():
        ruta = config.BASE_DIR / ruta
    return ruta if ruta.exists() else None


def filter_pathologies(df: pd.DataFrame, tramo: str = "Todos", patologia: str = "Todas", severidad: str = "Todas") -> pd.DataFrame:
    resultado = df.copy()
    if tramo and tramo != "Todos":
        resultado = resultado[resultado["codigo_tramo"] == tramo]
    if patologia and patologia != "Todas":
        resultado = resultado[resultado["tipo_patologia"] == patologia]
    if severidad and severidad != "Todas":
        resultado = resultado[resultado["severidad"] == severidad]
    return resultado.reset_index(drop=True)


def pathology_stats(df: pd.DataFrame | None = None) -> dict:
    df = data_loader.load_pathologies() if df is None else df
    if df.empty:
        return {"total": 0, "severidad_alta": 0, "con_foto": 0, "patologia_frecuente": "—"}
    severidad = df["severidad"].value_counts()
    tipo = df["tipo_patologia"].value_counts()
    con_foto = df["fotografia"].fillna("").astype(str).str.strip().ne("").sum()
    return {
        "total": int(len(df)),
        "severidad_alta": int(severidad.get("Alta", 0)),
        "con_foto": int(con_foto),
        "patologia_frecuente": tipo.index[0] if len(tipo) else "—",
    }
