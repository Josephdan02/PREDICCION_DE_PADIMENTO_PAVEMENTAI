"""Dataset — exploración, calidad de datos, filtros y descarga."""

from __future__ import annotations

import streamlit as st

from src import charts, config, data_loader, preprocessing, report_generator, ui

LABELS = {
    "id_registro": "ID",
    "codigo_tramo": "Tramo",
    "distrito": "Distrito",
    "tipo_pavimento": "Pavimento",
    "fecha_inspeccion": "Fecha",
    "antiguedad_anios": "Antigüedad",
    "volumen_transito": "Tránsito",
    "tipo_patologia": "Patología",
    "severidad": "Severidad",
    "cantidad_fallas": "Fallas",
    "pci_observado": "PCI",
    "tipo_registro": "Origen",
}


def main() -> None:
    ui.page_header(
        "Dataset de inspecciones",
        "Consulta, filtra y descarga el conjunto de datos utilizado para entrenar el modelo. Todos los registros están marcados como SIMULADOS.",
        eyebrow="Datos · trazabilidad",
    )

    df = data_loader.load_dataset()
    if df.empty:
        st.warning("El dataset está vacío. Genera datos con `python -m src.dataset_generator`.")
        return

    reporte = preprocessing.inspect_dataframe(df)
    simulados = int(df["tipo_registro"].astype(str).str.upper().eq(config.TIPO_REGISTRO_SIMULADO).sum())

    ui.render_metric_grid([
        dict(label="Registros", value=f"{len(df)}", foot="inspecciones en el dataset", icon="▤", span=3),
        dict(label="Tramos únicos", value=f"{df['codigo_tramo'].nunique()}", icon="◆", span=3),
        dict(label="PCI promedio", value=f"{df['pci_observado'].mean():.1f}",
             foot=f"min {df['pci_observado'].min():.1f} · max {df['pci_observado'].max():.1f}", icon="◉", span=3),
        dict(label="Registros simulados", value=f"{simulados}",
             foot="<span class='chip chip-warn'>SIMULADO</span>", icon="▲", span=3),
    ])

    tab_datos, tab_calidad, tab_stats = st.tabs(["Explorar y descargar", "Calidad de datos", "Estadísticas"])

    # --- Exploración ----------------------------------------------------------
    with tab_datos:
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1.4])
        with col1:
            tramo = st.selectbox("Tramo", ["Todos"] + sorted(df["codigo_tramo"].unique().tolist()))
        with col2:
            distrito = st.selectbox("Distrito", ["Todos"] + sorted(df["distrito"].unique().tolist()))
        with col3:
            patologia = st.selectbox("Patología", ["Todas"] + list(config.PATOLOGIAS.keys()))
        with col4:
            severidad = st.selectbox("Severidad", ["Todas"] + config.SEVERIDADES)

        vista = df.copy()
        if tramo != "Todos":
            vista = vista[vista["codigo_tramo"] == tramo]
        if distrito != "Todos":
            vista = vista[vista["distrito"] == distrito]
        if patologia != "Todas":
            vista = vista[vista["tipo_patologia"] == patologia]
        if severidad != "Todas":
            vista = vista[vista["severidad"] == severidad]

        st.caption(f"{len(vista)} registros coinciden con los filtros.")
        ui.dataframe_display(
            vista.sort_values("fecha_inspeccion", ascending=False),
            height=min(480, 90 + 35 * max(len(vista), 1)),
            column_labels=LABELS,
        )

        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.download_button(
                "Descargar CSV filtrado",
                data=report_generator.export_csv(vista),
                file_name="pavement_ai_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_b:
            st.caption(
                "El CSV se exporta en UTF-8 con BOM para abrirse correctamente en Excel. "
                "Los registros son sintéticos y no representan vías reales de Huánuco."
            )

    # --- Calidad --------------------------------------------------------------
    with tab_calidad:
        col1, col2 = st.columns(2, gap="small")
        with col1:
            with ui.card("Resultado de la validación"):
                estado = "chip-soft" if reporte.es_valido else "chip-danger"
                texto_estado = "Sin observaciones" if reporte.es_valido else "Requiere revisión"
                st.markdown(
                    f'<span class="chip {estado}">{texto_estado}</span>'
                    f'<div class="bento-foot" style="margin-top:0.6rem;">'
                    f"Registros: <strong>{reporte.total_registros}</strong><br/>"
                    f"Valores nulos: <strong>{reporte.total_nulos}</strong><br/>"
                    f"Duplicados exactos: <strong>{reporte.duplicados}</strong><br/>"
                    f"Duplicados de ID: <strong>{reporte.duplicados_por_id}</strong><br/>"
                    f"Errores de rango: <strong>{len(reporte.errores_rango)}</strong><br/>"
                    f"Categorías inválidas: <strong>{len(reporte.categorias_invalidas)}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if reporte.observaciones:
                    for obs in reporte.observaciones:
                        st.warning(obs)
                if reporte.errores_rango:
                    st.caption(" · ".join(reporte.errores_rango[:8]))
                if reporte.categorias_invalidas:
                    st.caption(" · ".join(reporte.categorias_invalidas))

        with col2:
            with ui.card("Tipos de datos y nulos"):
                tipos = [
                    {"Columna": c, "Tipo": str(df[c].dtype), "Nulos": int(df[c].isna().sum()),
                     "Únicos": int(df[c].nunique())}
                    for c in df.columns
                ]
                st.dataframe(tipos, use_container_width=True, hide_index=True, height=380)

        st.markdown("**Rangos válidos esperados**")
        st.markdown(
            "- `pci_observado`: 0 – 100\n"
            "- `antiguedad_anios`: 0 – 60\n"
            "- `volumen_transito`: ≥ 0\n"
            "- `severidad`: Baja / Media / Alta\n"
            "- `tipo_patologia`: una de las seis patologías del catálogo"
        )

    # --- Estadísticas ---------------------------------------------------------
    with tab_stats:
        col1, col2 = st.columns(2, gap="small")
        with col1:
            with ui.card("Estadísticos descriptivos"):
                numericas = df[["antiguedad_anios", "volumen_transito", "cantidad_fallas", "pci_observado"]]
                resumen = numericas.describe().T.reset_index().rename(columns={"index": "Variable"})
                st.dataframe(resumen.round(2), use_container_width=True, hide_index=True)
        with col2:
            with ui.card():
                ui.plot(charts.severity_distribution_chart(df), key="ds_sev")

        col1, col2 = st.columns(2, gap="small")
        with col1:
            with ui.card():
                ui.plot(charts.pathology_frequency_chart(df), key="ds_pat")
        with col2:
            with ui.card("Conteo por distrito"):
                conteo = df["distrito"].value_counts().reset_index()
                conteo.columns = ["Distrito", "Inspecciones"]
                st.dataframe(conteo, use_container_width=True, hide_index=True, height=300)
                st.caption("Los datos son sintéticos: la distribución por distrito no refleja el estado real de las vías.")


main()
