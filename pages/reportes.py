"""Reportes — generación de PDF y exportaciones CSV/JSON."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from src import config, data_loader, report_generator, ui


def main() -> None:
    ui.page_header(
        "Reportes y exportación",
        "Genera el informe técnico en PDF con indicadores, métricas reales del modelo, gráficos y resultados de simulación.",
        eyebrow="Documentación del sistema",
    )

    indicadores = report_generator.system_indicators()
    metadata = config.load_metadata()

    ui.render_metric_grid([
        dict(label="Tramos", value=f"{indicadores['total_tramos']}", icon="◆", span=3),
        dict(label="Inspecciones", value=f"{indicadores['total_inspecciones']}", icon="◈", span=3),
        dict(label="Patologías registradas", value=f"{indicadores['total_patologias_registradas']}", icon="▲", span=3),
        dict(label="PCI promedio", value=f"{indicadores['pci_promedio']}",
             foot=f"Condición: <strong>{indicadores['condicion_promedio']}</strong>", icon="◉", span=3),
    ])

    col_form, col_info = st.columns([3, 2], gap="medium")

    with col_form:
        st.markdown("**Configuración del informe**")
        with st.container(border=True):
            incluir_historial = st.checkbox("Incluir historial de inspecciones (últimas 25)", value=True)

            tramos = data_loader.tramo_options()
            opciones = ["Escenario libre"] + list(tramos.keys())
            seleccion = st.selectbox("Tramo para el escenario de predicción", opciones)
            tramo_codigo = None if seleccion == "Escenario libre" else tramos[seleccion]

            horizonte = st.slider("Horizonte de predicción (años)", 1, 10, 5)

            antiguedad = None
            if tramo_codigo is None:
                antiguedad = st.number_input(
                    "Antigüedad para el escenario libre (años)", min_value=0.0, max_value=80.0, value=12.0, step=0.5
                )

            generar = st.button("Generar informe PDF", type="primary", use_container_width=True)

        if generar:
            with st.spinner("Componiendo el documento PDF…"):
                try:
                    nombre, pdf_bytes = report_generator.save_pdf_report(
                        include_history=incluir_historial,
                        tramo=tramo_codigo,
                        horizonte=horizonte,
                        antiguedad_escenario=antiguedad,
                    )
                    st.session_state["pdf_reciente"] = (nombre, pdf_bytes)
                    st.success(f"Informe generado: `reports/{nombre}`")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"No fue posible generar el informe: {exc}")

        if "pdf_reciente" in st.session_state:
            nombre, pdf_bytes = st.session_state["pdf_reciente"]
            st.download_button(
                f"Descargar {nombre}",
                data=pdf_bytes,
                file_name=nombre,
                mime="application/pdf",
                use_container_width=True,
            )

    with col_info:
        with ui.card("Contenido del informe"):
            st.markdown(
                "1. Resumen ejecutivo con indicadores del sistema\n"
                "2. Métricas reales del modelo (R², MAE, RMSE) y ecuación\n"
                "3. Gráfico antigüedad ↔ PCI con la recta ajustada\n"
                "4. Resultados de predicción del escenario\n"
                "5. Patologías registradas por el usuario\n"
                "6. Historial de inspecciones\n"
                "7. Limitaciones del estudio\n\n"
                "El PDF incluye un aviso visible sobre la naturaleza simulada de los datos."
            )
            if metadata:
                m = metadata["metricas"]
                st.caption(
                    f"Métricas vigentes: R² {m['r2']:.4f} · MAE {m['mae']:.2f} · RMSE {m['rmse']:.2f}"
                )
            else:
                st.warning("El modelo no está entrenado: el informe se generará sin métricas.", icon="⚠️")

    # --- Exportaciones --------------------------------------------------------
    st.markdown('<div class="bento-card-title" style="margin-top:0.5rem;">Exportaciones de datos</div>',
                unsafe_allow_html=True)
    df = data_loader.load_dataset()
    patologias = data_loader.load_pathologies()
    vias = data_loader.load_roads()

    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.download_button(
            "Dataset completo (CSV)",
            data=report_generator.export_csv(df),
            file_name="pavement_ai_dataset_completo.csv",
            mime="text/csv", use_container_width=True,
        )
    with col2:
        st.download_button(
            "Patologías (CSV)",
            data=report_generator.export_csv(patologias) if not patologias.empty else b"sin registros\n",
            file_name="pavement_ai_patologias.csv",
            mime="text/csv", use_container_width=True,
        )
    with col3:
        st.download_button(
            "Vías registradas (CSV)",
            data=report_generator.export_csv(vias) if not vias.empty else b"sin registros\n",
            file_name="pavement_ai_vias.csv",
            mime="text/csv", use_container_width=True,
        )
    with col4:
        st.download_button(
            "Resumen del sistema (JSON)",
            data=report_generator.export_json(report_generator.build_export_payload()),
            file_name=f"pavement_ai_resumen_{datetime.now():%Y%m%d}.json",
            mime="application/json", use_container_width=True,
        )

    # --- Historial de informes generados --------------------------------------
    with st.expander("Informes guardados en la carpeta reports/"):
        archivos = sorted(config.REPORTS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not archivos:
            st.caption("Todavía no se han generado informes.")
        else:
            for archivo in archivos[:10]:
                col_nombre, col_accion = st.columns([4, 1])
                fecha = datetime.fromtimestamp(archivo.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
                col_nombre.markdown(f"`{archivo.name}` · {archivo.stat().st_size / 1024:.0f} KB · {fecha}")
                with col_accion:
                    st.download_button(
                        "Descargar", data=archivo.read_bytes(), file_name=archivo.name,
                        mime="application/pdf", key=f"dl_{archivo.name}",
                    )

    st.info(
        "Los reportes reflejan exclusivamente los datos persistidos en el sistema. "
        "Si aún no has registrado vías o patologías, esos apartados aparecerán vacíos.",
        icon="ℹ️",
    )


main()
