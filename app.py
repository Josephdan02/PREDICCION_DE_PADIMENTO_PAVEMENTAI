"""PAVEMENT AI — punto de entrada de la aplicación Streamlit.

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src import config, ui

# Configuración global (una sola vez por sesión)
st.set_page_config(
    page_title=f"{config.APP_NAME} · Gestión de pavimentos rígidos",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

config.ensure_directories()
ui.inject_css()

# ---------------------------------------------------------------------------
# Navegación
# ---------------------------------------------------------------------------
# Los iconos son SVG (``src/nav_icons.py``): se pasan como HTML a ``st.page_link``
# en lugar de emojis o de Material Symbols.
ATRIBUTOS = config.nav_link_attributes()


def crear_pagina(ruta: str, predeterminada: bool = False) -> st.Page:
    """Construye un ``st.Page`` con su título e icono declarados en config.

    El icono se declara con un nombre Material único (Streamlit lo exige); el
    SVG definitivo se inyecta con CSS en la barra lateral.
    """
    atributos = ATRIBUTOS[ruta]
    return st.Page(
        ruta,
        title=atributos["label"],
        icon=atributos["slot"],
        url_path=atributos["data-icon"],
        default=predeterminada,
    )


PAGINAS = {
    "Análisis": [
        crear_pagina("pages/dashboard.py", predeterminada=True),
        crear_pagina("pages/simulador_pci.py"),
        crear_pagina("pages/modelo_ml.py"),
    ],
    "Gestión": [
        crear_pagina("pages/registro_vias.py"),
        crear_pagina("pages/patologias.py"),
        crear_pagina("pages/dataset.py"),
    ],
    "Sistema": [
        crear_pagina("pages/reportes.py"),
        crear_pagina("pages/acerca.py"),
    ],
}

# Marca y estado del modelo se dibujan antes que la navegación.
ui.sidebar_brand()

navegacion = st.navigation(PAGINAS, position="sidebar")
navegacion.run()

ui.page_footer()
