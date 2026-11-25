"""Simulador PCI — proyección del PCI con el modelo de regresión entrenado."""

from __future__ import annotations

import streamlit as st

from src import charts, config, data_loader, prediction, ui

HORIZONTES = {"Corto plazo · 1 año": 1, "Mediano plazo · 3 años": 3, "Largo plazo · 5 años": 5}


def _scenario_card(titulo: str, resultado: dict, descripcion: str) -> str:
    """Tarjeta Bento con el resultado de un escenario."""
    delta = resultado["variacion"]
    color_delta = "#E98D7C" if delta < 0 else config.COLORS["lime"]
    return (
        f'<div class="bento bento-span-4">'
        f'<div><div class="bento-head"><span class="bento-label">{titulo}</span>'
        f'<span class="bento-icon">▸</span></div>'
        f'<div class="bento-value">{resultado["pci_futuro"]:.1f}'
        f'<span class="bento-unit">PCI</span></div>'
        f'<div style="margin-top:0.45rem;">{ui.pci_badge(resultado["pci_futuro"], large=True)}</div></div>'
        f'<div class="bento-foot">{descripcion}<br/>'
        f'Variación: <strong style="color:{color_delta};">{delta:+.2f} pts</strong> · '
        f'antigüedad {resultado["antiguedad_futura"]:.1f} años</div>'
        f"</div>"
    )


def main() -> None:
    ui.page_header(
        "Simulador PCI",
        "Proyecta el Índice de Condición del Pavimento usando el modelo de regresión lineal entrenado con la antigüedad del tramo.",
        eyebrow="Análisis predictivo",
    )

    if not prediction.model_is_available():
        st.error(
            "No hay un modelo entrenado disponible. Entrena el modelo en la página **Modelo ML** "
            "o ejecuta `python -m src.train_model`.",
            icon="⚠️",
        )
        return

    metadata = config.load_metadata()
    info = prediction.model_info()
    df = data_loader.load_dataset()
    tramos = data_loader.tramo_options()

    # --- Selección de escenario ----------------------------------------------
    with st.container(border=True):
        col1, col2, col3 = st.columns([2.2, 1.6, 1.6])
        with col1:
            modo = st.radio("Punto de partida", ["Tramo registrado", "Escenario nuevo"], horizontal=True)
        with col2:
            if modo == "Tramo registrado" and tramos:
                seleccion = st.selectbox("Tramo", list(tramos.keys()))
                codigo = tramos[seleccion]
                filas = df[df["codigo_tramo"].astype(str) == codigo]
                antiguedad_sugerida = (
                    float(filas.sort_values("fecha_inspeccion").iloc[-1][config.FEATURE_COLUMN])
                    if not filas.empty else 0.0
                )
            else:
                codigo = None
                antiguedad_sugerida = 10.0
            antiguedad = st.number_input(
                "Antigüedad actual (años)", min_value=0.0, max_value=80.0,
                value=float(antiguedad_sugerida), step=0.5,
                help="Edad del pavimento en el momento de la consulta.",
            )
        with col3:
            horizonte_label = st.selectbox("Horizonte de simulación", list(HORIZONTES.keys()), index=2)
            horizonte = HORIZONTES[horizonte_label]
            st.caption(f"Antigüedad futura: **{antiguedad + horizonte:.1f} años**")

    resultado = prediction.predict_scenario(antiguedad, horizonte, metadata)

    # --- Advertencias ---------------------------------------------------------
    for advertencia in resultado["advertencias"]:
        st.warning(advertencia, icon="⚠️")

    # --- Resultado principal --------------------------------------------------
    actual_html = (
        f'<div class="bento bento-span-6 bento-hero">'
        f'<div><div class="bento-head"><span class="bento-label">PCI estimado actual</span>'
        f'<span class="bento-icon">◉</span></div>'
        f'<div class="pci-block"><span class="pci-value" style="color:{resultado["color_actual"]};">'
        f'{resultado["pci_actual"]:.1f}</span>{ui.pci_badge(resultado["pci_actual"], large=True)}</div>'
        f'<div class="bento-foot">Antigüedad de {resultado["antiguedad_actual"]:.1f} años · '
        f'PCI matemático: <span class="mono">{resultado["pci_actual_raw"]:.2f}</span></div></div>'
        f"{ui.pci_scale_bar(resultado['pci_actual'])}</div>"
    )
    futuro_html = (
        f'<div class="bento bento-span-6">'
        f'<div><div class="bento-head"><span class="bento-label">PCI estimado en {ui.years_label(horizonte)}</span>'
        f'<span class="bento-icon">◈</span></div>'
        f'<div class="pci-block"><span class="pci-value" style="color:{resultado["color_futuro"]};">'
        f'{resultado["pci_futuro"]:.1f}</span>{ui.pci_badge(resultado["pci_futuro"], large=True)}</div>'
        f'<div class="bento-foot">Antigüedad de {resultado["antiguedad_futura"]:.1f} años · '
        f'PCI matemático: <span class="mono">{resultado["pci_futuro_raw"]:.2f}</span></div></div>'
        f"{ui.pci_scale_bar(resultado['pci_futuro'])}</div>"
    )
    st.markdown(f'<div class="bento-grid">{actual_html}{futuro_html}</div>', unsafe_allow_html=True)

    # --- Tres escenarios ------------------------------------------------------
    st.markdown('<div class="bento-card-title" style="margin-top:0.4rem;">Escenarios de simulación</div>',
                unsafe_allow_html=True)
    escenarios = {etiqueta: prediction.predict_scenario(antiguedad, h, metadata) for etiqueta, h in HORIZONTES.items()}
    tarjetas = "".join(
        _scenario_card(etiqueta, res, f"Proyección a {ui.years_label(res['horizonte_anios'])} desde hoy")
        for etiqueta, res in escenarios.items()
    )
    st.markdown(f'<div class="bento-grid">{tarjetas}</div>', unsafe_allow_html=True)

    # --- Gráfico y ecuación ---------------------------------------------------
    col_graf, col_info = st.columns([8, 4], gap="small")
    with col_graf:
        with ui.card("Recta del modelo y puntos consultados", "joblib · LinearRegression"):
            ui.plot(
                charts.simulator_chart(
                    df, info["intercepto"], info["pendiente"],
                    resultado["antiguedad_actual"], resultado["antiguedad_futura"],
                    resultado["pci_actual"], resultado["pci_futuro"],
                    metadata.get("rango_antiguedad_entrenamiento"),
                ),
                key="sim_chart",
            )

    with col_info:
        with ui.card("Ecuación utilizada"):
            st.markdown(
                f'<div class="mono" style="font-size:1.02rem; color:{config.COLORS["lime"]}; '
                f'background:rgba(163,230,53,0.07); border:1px solid rgba(163,230,53,0.25); '
                f'border-radius:10px; padding:0.7rem 0.8rem; margin-bottom:0.6rem;">{info["ecuacion"]}</div>'
                f'<div class="bento-foot">Intercepto <strong>{info["intercepto"]:.3f}</strong> · '
                f'pendiente <strong>{info["pendiente"]:.4f}</strong> puntos PCI por año.</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "El valor mostrado se calcula exclusivamente con `LinearRegression.predict` cargado desde Joblib. "
                "El recorte a 0–100 es solo visual; el valor matemático se conserva arriba."
            )
            if metadata.get("metricas"):
                metricas = metadata["metricas"]
                st.markdown("**Desempeño del modelo (conjunto de prueba)**")
                st.markdown(
                    f"- R² = `{metricas['r2']:.4f}`\n- MAE = `{metricas['mae']:.2f}` puntos\n"
                    f"- RMSE = `{metricas['rmse']:.2f}` puntos"
                )

    # --- Tabla resumen --------------------------------------------------------
    with st.expander("Detalle numérico de los escenarios"):
        filas = []
        for etiqueta, res in escenarios.items():
            filas.append({
                "Escenario": etiqueta,
                "Años": res["horizonte_anios"],
                "Antigüedad futura": f"{res['antiguedad_futura']:.1f}",
                "PCI futuro (0-100)": f"{res['pci_futuro']:.2f}",
                "PCI matemático": f"{res['pci_futuro_raw']:.2f}",
                "Variación": f"{res['variacion']:+.2f}",
                "Condición": res["estado_futuro"],
            })
        st.dataframe(filas, use_container_width=True, hide_index=True)

    st.caption(
        "Las proyecciones son estimaciones estadísticas basadas en un único predictor (antigüedad) y en un dataset "
        "simulado. No constituyen una evaluación oficial del pavimento ni una recomendación de intervención."
    )


main()
