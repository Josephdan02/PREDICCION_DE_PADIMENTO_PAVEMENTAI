"""Modelo ML — entrenamiento, métricas reales y diagnóstico del modelo."""

from __future__ import annotations

import streamlit as st

from src import charts, config, data_loader, evaluation, prediction, train_model, ui


def _render_metrics(metadata: dict) -> None:
    metricas = metadata.get("metricas", {})
    ui.render_metric_grid([
        dict(label="R² (prueba)", value=f"{metricas.get('r2', 0):.4f}",
             foot="Coeficiente de determinación", icon="◆", span=4, hero=True),
        dict(label="MAE (prueba)", value=f"{metricas.get('mae', 0):.2f}", unit="pts PCI",
             foot="Error absoluto medio", icon="◈", span=4),
        dict(label="RMSE (prueba)", value=f"{metricas.get('rmse', 0):.2f}", unit="pts PCI",
             foot="Raíz del error cuadrático medio", icon="◉", span=4),
    ])


def main() -> None:
    ui.page_header(
        "Modelo de regresión lineal",
        "Entrenamiento real con scikit-learn, evaluación con R²/MAE/RMSE y diagnóstico gráfico del ajuste.",
        eyebrow="Machine learning · LinearRegression",
    )

    df = data_loader.load_dataset()
    metadata = config.load_metadata()

    # --- Entrenamiento --------------------------------------------------------
    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        st.markdown(
            f"**Configuración:** variable independiente `{config.FEATURE_COLUMN}`, "
            f"variable dependiente `{config.TARGET_COLUMN}`, división 80/20 con semilla {config.RANDOM_STATE} "
            "y separación agrupada por `codigo_tramo`."
        )
        st.caption(
            "La división agrupada evita que inspecciones del mismo tramo aparezcan simultáneamente en "
            "entrenamiento y prueba, lo que inflaría artificialmente las métricas."
        )
    with col2:
        if st.button("Entrenar / reentrenar modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando LinearRegression…"):
                try:
                    resultado = train_model.train_model(df, save=True)
                    st.session_state["ultimo_entrenamiento"] = resultado["metadata"]
                    prediction._cached_model.cache_clear()
                    st.success("Modelo entrenado y guardado en `models/linear_regression.joblib`.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001 - se muestra al usuario
                    st.error(f"No se pudo entrenar el modelo: {exc}")

    if not metadata:
        st.warning("Aún no existe un modelo entrenado. Pulsa 'Entrenar / reentrenar modelo'.", icon="⚠️")
        return

    _render_metrics(metadata)

    # --- Ecuación -------------------------------------------------------------
    intercepto = metadata["intercepto"]
    pendiente = metadata["pendiente"]
    st.markdown(
        f'<div class="bento-card">'
        f'<div class="bento-label">Ecuación ajustada</div>'
        f'<div class="mono" style="font-size:1.25rem; font-weight:600; color:{config.COLORS["lime"]}; margin:0.35rem 0 0.5rem;">'
        f'{metadata["ecuacion"]}</div>'
        f'<div class="bento-foot">{evaluation.slope_explanation(pendiente)}<br/>'
        f'{evaluation.interpret_r2(metadata["metricas"]["r2"])}</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("¿Cómo interpretar la pendiente?"):
        st.markdown(
            f"""
La pendiente **{pendiente:.4f}** indica cuántos puntos de PCI se pierden (o ganan) por cada año de antigüedad.
En este modelo, un tramo con **1 año más** tendría un PCI estimado de
`{intercepto + pendiente * 11:.1f}` frente a `{intercepto + pendiente * 10:.1f}` con 10 años, es decir
una diferencia de `{pendiente:.2f}` puntos.

- **Intercepto ({intercepto:.2f})**: PCI estimado para un pavimento de 0 años según el modelo. Es una
  referencia matemática; el dataset no contiene tramos recién construidos, por lo que no debe interpretarse
  como el PCI real de una vía nueva.
- **Pendiente ({pendiente:.4f})**: tasa media de degradación. Una pendiente negativa confirma que el PCI
  disminuye con la edad; su magnitud resume la velocidad de deterioro promedio de la muestra.
- **R² ({metadata["metricas"]["r2"]:.4f})**: proporción de la variabilidad del PCI explicada por la antigüedad.
  El resto se debe a factores no incluidos (tránsito, drenaje, calidad de construcción, tipo de patología).
            """
        )

    # --- Gráficos -------------------------------------------------------------
    try:
        bundle = train_model.train_model(df, save=False)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No fue posible reconstruir los conjuntos de entrenamiento/prueba: {exc}")
        return

    col1, col2 = st.columns(2, gap="small")
    with col1:
        with ui.card():
            ui.plot(charts.scatter_fit_chart(bundle["train_df"], bundle["test_df"], intercepto, pendiente), key="ml_fit")
    with col2:
        with ui.card():
            ui.plot(charts.predicted_vs_actual_chart(bundle["test_df"][config.TARGET_COLUMN], bundle["y_pred_test"]), key="ml_pred")

    col1, col2 = st.columns([1, 1], gap="small")
    with col1:
        with ui.card():
            ui.plot(charts.residuals_chart(bundle["test_df"][config.TARGET_COLUMN], bundle["y_pred_test"]), key="ml_res")
    with col2:
        with ui.card("Detalle del entrenamiento"):
            train_info = metadata["conjunto_entrenamiento"]
            test_info = metadata["conjunto_prueba"]
            rango = metadata["rango_antiguedad_entrenamiento"]
            st.dataframe(
                [
                    {"Aspecto": "Algoritmo", "Valor": metadata["algoritmo"]},
                    {"Aspecto": "Método de división", "Valor": metadata["metodo_division"]},
                    {"Aspecto": "Registros de entrenamiento", "Valor": f"{train_info['n_registros']} ({train_info['n_tramos']} tramos)"},
                    {"Aspecto": "Registros de prueba", "Valor": f"{test_info['n_registros']} ({test_info['n_tramos']} tramos)"},
                    {"Aspecto": "Registros limpios totales", "Valor": metadata["total_registros_limpios"]},
                    {"Aspecto": "Rango de antigüedad (train)", "Valor": f"{rango['min']:.1f} – {rango['max']:.1f} años"},
                    {"Aspecto": "R² entrenamiento", "Valor": f"{metadata['metricas_entrenamiento']['r2']:.4f}"},
                    {"Aspecto": "Fecha de entrenamiento", "Valor": metadata["fecha_entrenamiento"]},
                ],
                use_container_width=True, hide_index=True, height=300,
            )

    st.info(
        f"Origen de los datos: {metadata.get('origen_datos', 'no documentado')}",
        icon="ℹ️",
    )


main()
