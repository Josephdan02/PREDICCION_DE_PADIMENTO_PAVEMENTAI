"""Dashboard principal — vista Bento Grid del estado del pavimento."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts, config, data_loader, road_service, ui


def latest_per_tramo(df: pd.DataFrame) -> pd.DataFrame:
    """Última inspección de cada tramo (estado actual del pavimento)."""
    if df.empty:
        return df
    ordenado = df.sort_values("fecha_inspeccion")
    return ordenado.groupby("codigo_tramo", as_index=False).tail(1)


def recent_activity(df: pd.DataFrame, patologias: pd.DataFrame, vias: pd.DataFrame) -> list[dict]:
    """Mezcla las últimas patologías, vías e inspecciones en una lista de actividad."""
    eventos: list[dict] = []
    if not patologias.empty:
        for _, r in patologias.tail(3).iloc[::-1].iterrows():
            eventos.append({
                "icon": "!",
                "title": f"Patología {r['tipo_patologia']} · {r['codigo_tramo']}",
                "meta": f"Severidad {r['severidad']} · {r['fecha_inspeccion']}",
            })
    if not vias.empty:
        for _, r in vias.tail(2).iloc[::-1].iterrows():
            eventos.append({
                "icon": "+",
                "title": f"Tramo registrado {r['codigo_tramo']}",
                "meta": f"{r['nombre_via']} · {r['distrito']}",
            })
    if not df.empty:
        ultima = df.sort_values("fecha_inspeccion").iloc[-1]
        eventos.append({
            "icon": "~",
            "title": f"Inspección en {ultima['codigo_tramo']}",
            "meta": (
                f"PCI {float(ultima['pci_observado']):.1f} · "
                f"{pd.to_datetime(ultima['fecha_inspeccion']).strftime('%d/%m/%Y')}"
            ),
        })
    return eventos[:6]


def main() -> None:
    ui.page_header(
        "Panel de control",
        "Estado del pavimento rígido, indicadores del sistema y el desempeño real del modelo de regresión lineal.",
        eyebrow="Observatorio de infraestructura vial · Huánuco",
        back_to="",
    )

    df = data_loader.load_dataset()
    patologias = data_loader.load_pathologies()
    vias = data_loader.load_roads()
    metadata = config.load_metadata()
    actuales = latest_per_tramo(df)

    if df.empty:
        st.warning("El dataset está vacío. Ejecuta `python -m src.dataset_generator` para generarlo.")
        return

    total_tramos = int(df["codigo_tramo"].nunique())
    total_inspecciones = int(len(df))
    total_patologias = int(len(patologias)) + int(len(df))
    pci_promedio = float(actuales[config.TARGET_COLUMN].mean()) if not actuales.empty else 0.0
    etiqueta, color, texto = config.classify_pci(pci_promedio)
    criticos = actuales[actuales[config.TARGET_COLUMN] < config.PCI_CRITICAL_THRESHOLD]
    simulados = int(df["tipo_registro"].astype(str).str.upper().eq("SIMULADO").sum())

    # --- Fila 1: KPIs ---------------------------------------------------------
    ui.render_metric_grid([
        dict(label="Tramos", value=f"{total_tramos}",
             foot=f"{df['distrito'].nunique()} distritos cubiertos", icon="◆", span=3),
        dict(label="Inspecciones", value=f"{total_inspecciones}",
             foot=f"Registros simulados: <strong>{simulados}</strong>", icon="◈", span=3),
        dict(label="Patologías", value=f"{total_patologias}",
             foot=f"Registradas por usuario: <strong>{len(patologias)}</strong>", icon="▲", span=3),
        dict(label="Tramos críticos", value=f"{len(criticos)}", unit=f"/{total_tramos}",
             foot="PCI actual &lt; 40 puntos", icon="■", span=3,
             value_color="#E98D7C" if len(criticos) else config.COLORS["lime"]),
    ])

    # --- Fila 2: PCI promedio, estado del modelo, inventario -------------------
    pci_foot = (
        f'<span class="chip" style="background:{color}; color:{texto};">{etiqueta}</span>'
        "&nbsp; Condición promedio de la última inspección de cada tramo"
    )
    if metadata:
        metricas = metadata.get("metricas", {})
        modelo_foot = (
            '<span class="chip chip-soft">Entrenado</span>&nbsp;'
            f'R² <strong>{metricas.get("r2", 0):.3f}</strong> · '
            f'MAE <strong>{metricas.get("mae", 0):.2f}</strong> · '
            f'RMSE <strong>{metricas.get("rmse", 0):.2f}</strong>'
        )
        modelo_valor = "LinearRegression"
    else:
        modelo_valor = "Sin entrenar"
        modelo_foot = "Entrena el modelo desde la página Modelo ML."

    vias_resumen = road_service.road_summary()
    ui.render_metric_grid([
        dict(label="PCI promedio", value=f"{pci_promedio:.1f}", foot=pci_foot, span=5, hero=True, icon="◉"),
        dict(label="Estado del modelo", value=modelo_valor, foot=modelo_foot, span=4, icon="⚙", value_sm=True),
        dict(label="Inventario de vías", value=f"{vias_resumen['total_vias']}",
             foot=(f"{vias_resumen['longitud_total_km']} km registrados · "
                   f"antigüedad media {vias_resumen['antiguedad_promedio']} años"),
             span=3, icon="▤"),
    ])

    # --- Tramo prioritario ----------------------------------------------------
    if not actuales.empty:
        peor = actuales.sort_values(config.TARGET_COLUMN).iloc[0]
        st.markdown(
            f'<div class="bento" style="margin-bottom:0.9rem;">'
            f'<div class="bento-label">Tramo prioritario</div>'
            f'<div style="display:flex; align-items:center; gap:0.8rem; margin-top:0.35rem; flex-wrap:wrap;">'
            f'<span style="font-size:1.15rem; font-weight:800;">{peor["codigo_tramo"]}</span>'
            f'{ui.pci_badge(float(peor[config.TARGET_COLUMN]), large=True)}'
            f'<span class="bento-foot" style="margin:0;">Distrito de {peor["distrito"]} · '
            f'PCI {float(peor[config.TARGET_COLUMN]):.1f} · {peor["tipo_patologia"]} '
            f'({peor["severidad"]}) · antigüedad {float(peor["antiguedad_anios"]):.1f} años</span>'
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # --- Gráficos -------------------------------------------------------------
    col_izq, col_der = st.columns([7, 5], gap="small")
    with col_izq:
        with ui.card():
            ui.plot(charts.condition_distribution_chart(actuales), key="dash_cond")
    with col_der:
        with ui.card():
            ui.plot(charts.pathology_frequency_chart(df), key="dash_pat")

    col_izq, col_der = st.columns([7, 5], gap="small")
    with col_izq:
        with ui.card():
            intercept = metadata.get("intercepto") if metadata else None
            slope = metadata.get("pendiente") if metadata else None
            ui.plot(charts.age_vs_pci_chart(df, intercept, slope), key="dash_age")
    with col_der:
        with ui.card():
            ui.plot(charts.pci_by_tramo_chart(actuales, top=12), key="dash_tramo")

    col_izq, col_der = st.columns([7, 5], gap="small")
    with col_izq:
        with ui.card():
            ui.plot(charts.inspections_timeline_chart(df), key="dash_time")
    with col_der:
        with ui.card("Actividad reciente"):
            st.markdown(ui.activity_list(recent_activity(df, patologias, vias)), unsafe_allow_html=True)

    with st.expander("Leyenda de condiciones del PCI (referencia visual)"):
        ui.legend()
        st.caption(
            "Estos rangos se usan como referencia visual dentro de la aplicación. No sustituyen una evaluación "
            "técnica del pavimento ni la metodología oficial PCI (ASTM D6433)."
        )


main()
