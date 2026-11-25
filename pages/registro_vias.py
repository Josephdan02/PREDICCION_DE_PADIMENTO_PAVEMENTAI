"""Registro de vías — CRUD de tramos de pavimento rígido."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src import config, data_loader, road_service, ui

LABELS = {
    "codigo_tramo": "Código",
    "nombre_via": "Vía",
    "distrito": "Distrito",
    "ubicacion": "Ubicación",
    "longitud_m": "Longitud (m)",
    "antiguedad_anios": "Antigüedad (años)",
    "volumen_transito": "Tránsito (veh/día)",
    "fecha_inspeccion": "Fecha inspección",
    "fecha_registro": "Registrado el",
}


def _form(road: dict | None = None, key_prefix: str = "nuevo") -> None:
    """Formulario de alta/edición. Con ``road`` se precarga para editar."""
    road = road or {}
    es_edicion = bool(road)
    with st.form(f"form_via_{key_prefix}", clear_on_submit=not es_edicion):
        col1, col2 = st.columns(2)
        with col1:
            codigo = st.text_input(
                "Código del tramo *",
                value=str(road.get("codigo_tramo", "")),
                placeholder="HJNU-101",
                disabled=es_edicion,
                help="Identificador único del tramo. En edición no se modifica.",
            )
            nombre = st.text_input("Nombre de la vía *", value=str(road.get("nombre_via", "")),
                                   placeholder="Av. Universitaria")
            distrito = st.selectbox(
                "Distrito *", config.DISTRITOS,
                index=config.DISTRITOS.index(road["distrito"]) if road.get("distrito") in config.DISTRITOS else 0,
            )
            ubicacion = st.text_input("Ubicación o referencia", value=str(road.get("ubicacion", "")),
                                      placeholder="Entre Jr. Puno y Jr. Cusco")
        with col2:
            longitud = st.number_input("Longitud aproximada (m) *", min_value=1.0, max_value=20000.0,
                                       value=float(road.get("longitud_m") or 500.0), step=10.0)
            antiguedad = st.number_input("Antigüedad (años) *", min_value=0.0, max_value=60.0,
                                         value=float(road.get("antiguedad_anios") or 5.0), step=0.5)
            volumen = st.number_input("Volumen de tránsito (veh/día) *", min_value=0, max_value=120000,
                                      value=int(road.get("volumen_transito") or 3000), step=100)
            fecha = st.date_input(
                "Fecha de inspección *",
                value=pd.to_datetime(road["fecha_inspeccion"]).date() if road.get("fecha_inspeccion") else date.today(),
                max_value=date.today(),
            )

        enviado = st.form_submit_button("Guardar cambios" if es_edicion else "Registrar tramo",
                                        type="primary", use_container_width=True)

    if enviado:
        datos = {
            "codigo_tramo": codigo, "nombre_via": nombre, "distrito": distrito, "ubicacion": ubicacion,
            "longitud_m": longitud, "antiguedad_anios": antiguedad, "volumen_transito": volumen,
            "fecha_inspeccion": fecha,
        }
        try:
            if es_edicion:
                road_service.update_road(road["codigo_tramo"], datos)
                st.success(f"Tramo {road['codigo_tramo']} actualizado correctamente.")
            else:
                road_service.create_road(datos)
                st.success(f"Tramo {codigo.strip().upper()} registrado correctamente.")
            st.rerun()
        except road_service.ValidationError as exc:
            st.error(f"No se pudo guardar: {exc}")


def main() -> None:
    ui.page_header(
        "Registro de vías",
        "Inventario de tramos de pavimento rígido con sus datos base de inspección. La información se guarda de forma permanente en `storage/vias.csv`.",
        eyebrow="Gestión de activos viales",
    )

    df = data_loader.load_roads()
    resumen = road_service.road_summary()

    ui.render_metric_grid([
        dict(label="Vías registradas", value=f"{resumen['total_vias']}", icon="▤", span=3),
        dict(label="Distritos", value=f"{resumen['distritos']}", icon="◆", span=3),
        dict(label="Longitud total", value=f"{resumen['longitud_total_km']}", unit="km", icon="◈", span=3),
        dict(label="Antigüedad media", value=f"{resumen['antiguedad_promedio']}", unit="años", icon="◉", span=3),
    ])

    tab_nuevo, tab_lista, tab_editar = st.tabs(["Registrar tramo", "Consultar inventario", "Editar / eliminar"])

    with tab_nuevo:
        if df.empty:
            st.info("Aún no hay tramos registrados. Completa el formulario para crear el primero.", icon="ℹ️")
        _form()

    with tab_lista:
        if df.empty:
            st.info("No hay tramos registrados todavía.", icon="📋")
        else:
            col1, col2 = st.columns([3, 2])
            with col1:
                texto = st.text_input("Buscar por código, vía o ubicación", placeholder="Av. Universitaria")
            with col2:
                distrito = st.selectbox("Filtrar por distrito", ["Todos"] + config.DISTRITOS)

            filtrado = road_service.filter_roads(df, texto, distrito)
            st.caption(f"{len(filtrado)} de {len(df)} tramos mostrados")
            ui.dataframe_display(
                filtrado.sort_values("codigo_tramo"),
                height=min(460, 90 + 35 * max(len(filtrado), 1)),
                column_labels=LABELS,
            )

    with tab_editar:
        if df.empty:
            st.info("No hay tramos para editar o eliminar.", icon="📋")
            return

        opciones = {f"{r['codigo_tramo']} · {r['nombre_via']}": r["codigo_tramo"] for _, r in df.iterrows()}
        seleccion = st.selectbox("Selecciona un tramo", list(opciones.keys()), key="via_editar")
        codigo = opciones[seleccion]
        tramo = road_service.get_road(codigo)

        col_edit, col_del = st.columns([3, 2], gap="medium")
        with col_edit:
            st.markdown("**Editar datos del tramo**")
            _form(tramo, key_prefix="edicion")
        with col_del:
            st.markdown("**Eliminar tramo**")
            st.warning(
                "La eliminación es permanente para el inventario. Las inspecciones históricas del dataset no se borran.",
                icon="⚠️",
            )
            confirmar = st.checkbox(f"Confirmo eliminar el tramo {codigo}", key="conf_del_via")
            if st.button("Eliminar tramo", disabled=not confirmar, use_container_width=True):
                road_service.delete_road(codigo)
                st.success(f"Tramo {codigo} eliminado.")
                st.rerun()


main()
