"""Registro de patologías — hallazgos por tramo con fotografía opcional."""

from __future__ import annotations

from datetime import date

import streamlit as st

from src import config, data_loader, pathology_service, ui

LABELS = {
    "id_patologia": "ID",
    "codigo_tramo": "Tramo",
    "tipo_patologia": "Patología",
    "severidad": "Severidad",
    "cantidad_fallas": "Cantidad",
    "descripcion": "Descripción",
    "fecha_inspeccion": "Fecha",
    "fotografia": "Fotografía",
    "fecha_registro": "Registrado el",
}


def main() -> None:
    ui.page_header(
        "Registro de patologías",
        "Documenta las fallas observadas en cada tramo: tipo, severidad, extensión, descripción y fotografía.",
        eyebrow="Inspección de campo",
    )

    patologias = data_loader.load_pathologies()
    stats = pathology_service.pathology_stats(patologias)

    patologia_frecuente = stats["patologia_frecuente"].replace("Grietas ", "Grietas ").replace("Despostillamiento", "Despostillam.")
    ui.render_metric_grid([
        dict(label="Patologías registradas", value=f"{stats['total']}", icon="▲", span=3),
        dict(label="Severidad alta", value=f"{stats['severidad_alta']}", icon="■", span=3,
             value_color="#E98D7C" if stats["severidad_alta"] else None),
        dict(label="Con fotografía", value=f"{stats['con_foto']}", icon="◈", span=3),
        dict(label="Más frecuente", value=patologia_frecuente, icon="◉", span=3, value_sm=True),
    ])

    st.info(
        "Registrar patologías documenta el deterioro del tramo, pero **no** calcula un PCI oficial. "
        "El PCI observado que usa el modelo proviene de una evaluación técnica documentada.",
        icon="ℹ️",
    )

    tab_registro, tab_consulta = st.tabs(["Nueva patología", "Patologías registradas"])

    with tab_registro:
        tramos = data_loader.tramo_options()
        if not tramos:
            st.warning("No hay tramos disponibles. Registra primero una vía en 'Registro de vías'.", icon="⚠️")
            return

        with st.form("form_patologia", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                tramo_etiqueta = st.selectbox("Tramo asociado *", list(tramos.keys()))
                tipo = st.selectbox("Tipo de patología *", list(config.PATOLOGIAS.keys()))
                severidad = st.radio("Severidad *", config.SEVERIDADES, horizontal=True, index=1)
            with col2:
                cantidad = st.number_input(
                    "Cantidad o extensión afectada *", min_value=0.1, max_value=10000.0, value=1.0, step=0.5,
                    help="Número de fallas, metros lineales o metros cuadrados afectados.",
                )
                fecha = st.date_input("Fecha de inspección *", value=date.today(), max_value=date.today())
                foto = st.file_uploader("Fotografía (opcional)", type=["png", "jpg", "jpeg", "webp"])

            descripcion = st.text_area(
                "Descripción", placeholder="Observaciones de campo, ubicación exacta, condiciones del entorno…",
                max_chars=500, height=90,
            )
            st.caption(f"Referencia: {config.PATOLOGIAS[tipo]}" if tipo else "")
            enviado = st.form_submit_button("Registrar patología", type="primary", use_container_width=True)

        if enviado:
            datos = {
                "codigo_tramo": tramos[tramo_etiqueta],
                "tipo_patologia": tipo,
                "severidad": severidad,
                "cantidad_fallas": cantidad,
                "descripcion": descripcion,
                "fecha_inspeccion": fecha,
            }
            try:
                pathology_service.create_pathology(
                    datos,
                    photo_bytes=foto.getvalue() if foto else None,
                    photo_name=foto.name if foto else None,
                )
                st.success(f"Patología de {tipo.lower()} registrada en el tramo {tramos[tramo_etiqueta]}.")
                st.rerun()
            except pathology_service.ValidationError as exc:
                st.error(f"No se pudo registrar: {exc}")

    with tab_consulta:
        if patologias.empty:
            st.info("Todavía no hay patologías registradas por el usuario.", icon="📋")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            tramos_disponibles = ["Todos"] + sorted(patologias["codigo_tramo"].unique().tolist())
            f_tramo = st.selectbox("Tramo", tramos_disponibles)
        with col2:
            f_tipo = st.selectbox("Patología", ["Todas"] + list(config.PATOLOGIAS.keys()))
        with col3:
            f_sev = st.selectbox("Severidad", ["Todas"] + config.SEVERIDADES)

        vistas = pathology_service.filter_pathologies(
            patologias.sort_values("fecha_registro", ascending=False), f_tramo, f_tipo, f_sev
        )

        ui.dataframe_display(vistas.drop(columns=["fotografia"]), height=min(420, 90 + 35 * max(len(vistas), 1)),
                             column_labels=LABELS)

        st.markdown("#### Tarjetas de patologías")
        for _, r in vistas.head(12).iterrows():
            with st.container(border=True):
                col_info, col_foto = st.columns([3, 1])
                with col_info:
                    chip = {"Baja": "chip-soft", "Media": "chip-warn", "Alta": "chip-danger"}.get(r["severidad"], "chip-neutral")
                    st.markdown(
                        f'<div class="bento-head"><span class="bento-label">{r["id_patologia"]} · {r["codigo_tramo"]}</span>'
                        f'<span class="chip {chip}">{r["severidad"]}</span></div>'
                        f'<div style="font-weight:700; font-size:1.02rem;">{r["tipo_patologia"]}</div>'
                        f'<div class="bento-foot">Extensión/cantidad: <strong>{float(r["cantidad_fallas"]):.1f}</strong> · '
                        f"Inspección: {r['fecha_inspeccion']}</div>",
                        unsafe_allow_html=True,
                    )
                    if str(r["descripcion"]).strip():
                        st.caption(r["descripcion"])
                with col_foto:
                    ruta = pathology_service.photo_path(r["fotografia"])
                    if ruta:
                        st.image(str(ruta), use_container_width=True)
                    else:
                        st.caption("Sin fotografía")

                confirmar = st.checkbox("Confirmar eliminación", key=f"del_pat_{r['id_patologia']}")
                if st.button("Eliminar", key=f"btn_del_{r['id_patologia']}", disabled=not confirmar):
                    pathology_service.delete_pathology(r["id_patologia"])
                    st.success(f"Patología {r['id_patologia']} eliminada.")
                    st.rerun()

        if len(vistas) > 12:
            st.caption(f"Se muestran las 12 patologías más recientes de {len(vistas)} resultados.")


main()
