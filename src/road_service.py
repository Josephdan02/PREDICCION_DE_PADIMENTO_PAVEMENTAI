"""Servicio de vías / tramos: validación y operaciones de negocio."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from src import data_loader


class ValidationError(ValueError):
    """Error de validación de formulario, con mensajes legibles."""


def validate_road(data: dict, existing: pd.DataFrame | None = None, allow_same_code: bool = False) -> dict:
    """Valida y normaliza los datos de un tramo. Devuelve el diccionario limpio."""
    errores: list[str] = []

    codigo = str(data.get("codigo_tramo", "")).strip().upper()
    if not codigo:
        errores.append("El código del tramo es obligatorio.")
    elif len(codigo) < 3:
        errores.append("El código del tramo debe tener al menos 3 caracteres.")
    elif not all(ch.isalnum() or ch in "-_" for ch in codigo):
        errores.append("El código solo admite letras, números, guiones y guiones bajos.")

    nombre = str(data.get("nombre_via", "")).strip()
    if not nombre:
        errores.append("El nombre de la vía es obligatorio.")

    distrito = str(data.get("distrito", "")).strip()
    if not distrito:
        errores.append("Selecciona un distrito.")

    try:
        longitud = float(data.get("longitud_m", 0) or 0)
        if longitud <= 0:
            errores.append("La longitud debe ser mayor que 0 metros.")
    except (TypeError, ValueError):
        errores.append("La longitud debe ser un número.")
        longitud = 0.0

    try:
        antiguedad = float(data.get("antiguedad_anios", -1))
        if antiguedad < 0 or antiguedad > 60:
            errores.append("La antigüedad debe estar entre 0 y 60 años.")
    except (TypeError, ValueError):
        errores.append("La antigüedad debe ser un número.")
        antiguedad = 0.0

    try:
        volumen = int(float(data.get("volumen_transito", -1)))
        if volumen < 0:
            errores.append("El volumen de tránsito no puede ser negativo.")
    except (TypeError, ValueError):
        errores.append("El volumen de tránsito debe ser un número entero.")
        volumen = 0

    fecha = data.get("fecha_inspeccion")
    if isinstance(fecha, (datetime, date)):
        fecha = fecha.strftime("%Y-%m-%d")
    try:
        fecha_dt = pd.to_datetime(fecha, errors="raise")
    except (ValueError, TypeError):
        errores.append("La fecha de inspección no es válida.")
        fecha_dt = pd.Timestamp.today()

    if existing is not None and not existing.empty:
        if not allow_same_code and (existing["codigo_tramo"].astype(str) == codigo).any():
            errores.append(f"Ya existe un tramo con el código {codigo}.")

    if errores:
        raise ValidationError(" ".join(errores))

    return {
        "codigo_tramo": codigo,
        "nombre_via": nombre,
        "distrito": distrito,
        "ubicacion": str(data.get("ubicacion", "")).strip(),
        "longitud_m": round(longitud, 2),
        "antiguedad_anios": round(antiguedad, 1),
        "volumen_transito": volumen,
        "fecha_inspeccion": fecha_dt.strftime("%Y-%m-%d"),
        "fecha_registro": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def create_road(data: dict) -> pd.DataFrame:
    """Registra un tramo nuevo con validación."""
    existing = data_loader.load_roads()
    limpio = validate_road(data, existing=existing)
    return data_loader.upsert_road(limpio)


def update_road(codigo_original: str, data: dict) -> pd.DataFrame:
    """Actualiza un tramo existente."""
    existing = data_loader.load_roads()
    data = {**data, "codigo_tramo": str(codigo_original)}
    limpio = validate_road(data, existing=existing, allow_same_code=True)
    return data_loader.upsert_road(limpio)


def delete_road(codigo_tramo: str) -> pd.DataFrame:
    """Elimina un tramo por código (la confirmación se maneja en la UI)."""
    if not str(codigo_tramo).strip():
        raise ValidationError("Debes indicar el código del tramo a eliminar.")
    return data_loader.delete_road(codigo_tramo)


def filter_roads(df: pd.DataFrame, texto: str = "", distrito: str = "Todos") -> pd.DataFrame:
    """Búsqueda por texto libre (código, vía, ubicación) y filtro por distrito."""
    resultado = df.copy()
    if distrito and distrito != "Todos":
        resultado = resultado[resultado["distrito"] == distrito]
    texto = (texto or "").strip().lower()
    if texto:
        mascara = (
            resultado["codigo_tramo"].astype(str).str.lower().str.contains(texto, na=False)
            | resultado["nombre_via"].astype(str).str.lower().str.contains(texto, na=False)
            | resultado["ubicacion"].astype(str).str.lower().str.contains(texto, na=False)
        )
        resultado = resultado[mascara]
    return resultado.reset_index(drop=True)


def get_road(codigo_tramo: str) -> dict | None:
    df = data_loader.load_roads()
    fila = df[df["codigo_tramo"].astype(str) == str(codigo_tramo)]
    return None if fila.empty else fila.iloc[0].to_dict()


def road_summary() -> dict:
    df = data_loader.load_roads()
    return {
        "total_vias": int(len(df)),
        "distritos": int(df["distrito"].nunique()) if not df.empty else 0,
        "longitud_total_km": round(float(pd.to_numeric(df["longitud_m"], errors="coerce").sum()) / 1000, 2) if not df.empty else 0.0,
        "antiguedad_promedio": round(float(pd.to_numeric(df["antiguedad_anios"], errors="coerce").mean()), 1) if not df.empty else 0.0,
    }
