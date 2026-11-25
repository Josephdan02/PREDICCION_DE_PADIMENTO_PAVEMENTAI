"""Acerca del proyecto — alcance, metodología, limitaciones y criterios académicos."""

from __future__ import annotations

import streamlit as st

from src import config, data_loader, ui


def main() -> None:
    ui.page_header(
        "Acerca del proyecto",
        "Alcance académico, metodología, procedencia de los datos y limitaciones del sistema.",
        eyebrow="PAVEMENT AI · SENATI",
    )

    metadata = config.load_metadata()
    df = data_loader.load_dataset()

    ui.render_metric_grid([
        dict(label="Aplicación", value=config.APP_NAME, foot=f"Versión {config.APP_VERSION}", span=4),
        dict(label="Modelo", value="LinearRegression", foot="scikit-learn · persistencia Joblib", icon="⚙", span=4,
             value_sm=True),
        dict(label="Registros", value=f"{len(df)}", foot="dataset sintético de inspecciones", icon="▤", span=4),
    ])

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown("### Objetivo")
        st.markdown(
            "Desarrollar una aplicación web académica para una municipalidad de Huánuco que permita **registrar "
            "patologías de pavimentos rígidos de concreto** y **estimar el Índice de Condición del Pavimento (PCI)** "
            "mediante un modelo de **regresión lineal simple**, como actividad del curso de Inteligencia Artificial "
            "de SENATI."
        )

        st.markdown("### ¿Qué hace el sistema?")
        st.markdown(
            "- **Registrar tramos** de pavimento rígido con sus datos base de inspección.\n"
            "- **Documentar patologías** (grietas, losa dividida, despostillamiento, escalonamiento) con severidad y fotografía.\n"
            "- **Entrenar un modelo real** de regresión lineal: `PCI = f(antigüedad)`.\n"
            "- **Simular la evolución del PCI** a 1, 3 y 5 años usando el modelo entrenado.\n"
            "- **Generar informes PDF** con indicadores, métricas y gráficos."
        )

        st.markdown("### Modelo y evaluación")
        if metadata:
            m = metadata["metricas"]
            st.markdown(
                f"- Algoritmo: `{metadata['algoritmo']}`\n"
                f"- Variables: `X = {metadata['variable_independiente']}`, `y = {metadata['variable_dependiente']}`\n"
                f"- División: {metadata['metodo_division']}, semilla {metadata['random_state']}\n"
                f"- Ecuación: `{metadata['ecuacion']}`\n"
                f"- **R² = {m['r2']:.4f}**, **MAE = {m['mae']:.2f}**, **RMSE = {m['rmse']:.2f}** (conjunto de prueba)\n"
                f"- Entrenado el {metadata['fecha_entrenamiento'][:10]}"
            )
        else:
            st.warning("El modelo aún no ha sido entrenado.")

        st.markdown("### Interpretación del PCI (referencia visual)")
        ui.legend()
        st.caption(
            "Estos rangos son una **referencia visual** adoptada por la aplicación. No sustituyen una evaluación "
            "técnica del pavimento ni el procedimiento oficial de la metodología PCI (ASTM D6433), que requiere la "
            "medición de densidad y severidad de cada patología sobre unidades de muestreo."
        )

    with col2:
        st.markdown("### Procedencia de los datos")
        st.markdown(
            "El dataset (`data/pavimentos.csv`) es **sintético**. Se generó de forma controlada para que el modelo "
            "pueda entrenarse y evaluarse, y **todos los registros están marcados como `SIMULADO`**."
        )
        st.markdown(
            """<div class="bento" style="margin:0.6rem 0 1rem;">
            <div class="bento-label">Aviso</div>
            <div style="font-size:0.9rem; color:#D6E5B5; margin-top:4px; line-height:1.5;">
            Los datos <strong>no</strong> provienen de inspecciones reales verificadas en Huánuco y
            <strong>no</strong> deben presentarse como tal. Cualquier uso real requiere una campaña de
            inspección técnica documentada.
            </div></div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### Limitaciones")
        st.markdown(
            "- Un solo predictor (antigüedad): el tránsito, el drenaje y el tipo de patología no entran al modelo.\n"
            "- La relación PCI–antigüedad del dataset es aproximadamente lineal por diseño; en campo es no lineal.\n"
            "- El registro de patologías **no** calcula un PCI oficial.\n"
            "- Las predicciones fuera del rango de antigüedad entrenado son extrapolaciones.\n"
            "- El recorte a 0–100 es visual; el valor matemático se conserva y se muestra."
        )

        st.markdown("### Tecnologías")
        st.markdown(
            "Python 3.11 · Pandas · NumPy · scikit-learn · Matplotlib · Plotly · Streamlit · Joblib · "
            "ReportLab · Pytest"
        )

        st.markdown("### Estructura")
        st.code(
            "PAVEMENT_AI/\n├── app.py\n├── data/          dataset sintético\n├── models/        modelo .joblib + metadata\n"
            "├── src/           lógica de negocio\n├── pages/         páginas Streamlit\n├── assets/        estilos\n"
            "├── storage/       vías y patologías (CSV)\n├── uploads/       fotografías\n├── reports/       PDFs\n"
            "└── tests/         pruebas Pytest",
            language="text",
        )

    st.markdown("### Enlaces útiles")
    st.markdown(
        "- [Documentación de scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)\n"
        "- [Streamlit Docs](https://docs.streamlit.io)\n"
        "- [FHWA — distresses de pavimentos rígidos](https://www.fhwa.dot.gov/pavement/)"
    )


main()
