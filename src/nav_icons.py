# PAVEMENT AI - iconos SVG de la navegación lateral
#
# Segunda parte del icono: los SVG son vectores `currentColor` sobre una retícula
# de 24x24. Streamlit solo acepta emojis o nombres de Material Symbols en
# ``st.Page(icon=...)`` (rechaza HTML, URL e imágenes), por lo que el icono se
# declara con un nombre Material y ``assets/styles.css`` lo sustituye por el SVG
# de la página correspondiente (regla del atributo ``href``).
#
# IMPORTANTE: los nombres de Material se repiten entre páginas porque el motor
# de navegación los deduplica para detectar «todas las páginas comparten icono»
# y, en ese caso, deja de renderizar el icono de cada enlace. La deduplicación
# se anula con un comentario invisible y distinto por página en ``icon_html``.

SVG_STROKE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
)

# Mapa página -> SVG. La clave es el slug de la URL de la página (ver data-icon).
NAV_ICONS: dict[str, str] = {
    # Dashboard: retícula de indicadores (Bento)
    "dashboard": (
        SVG_STROKE
        + '<rect x="3" y="3" width="7.5" height="7.5" rx="1.6"/>'
        '<rect x="13.5" y="3" width="7.5" height="4.5" rx="1.6"/>'
        '<rect x="13.5" y="10.5" width="7.5" height="10.5" rx="1.6"/>'
        '<rect x="3" y="13.5" width="7.5" height="7.5" rx="1.6"/></svg>'
    ),
    # Simulador PCI: tendencia con proyección y puntos de consulta
    "simulador_pci": (
        SVG_STROKE
        + '<path d="M3 17.5 8.5 12l3.5 3 6.5-8"/>'
        '<path d="M15 7h3.5v3.5"/>'
        '<circle cx="8.5" cy="12" r="1.6"/>'
        '<circle cx="12" cy="15" r="1.6"/></svg>'
    ),
    # Modelo ML: nodos en capas conectados
    "modelo_ml": (
        SVG_STROKE
        + '<circle cx="5" cy="12" r="2.2"/>'
        '<circle cx="12" cy="6" r="2.2"/>'
        '<circle cx="12" cy="18" r="2.2"/>'
        '<circle cx="19" cy="12" r="2.2"/>'
        '<path d="M7 11.2 10 7.1M7 12.8l3 4.1M14 7.1l3 4.1M14 16.9l3-4.1"/></svg>'
    ),
    # Registro de vías: calzada en perspectiva con eje central
    "registro_vias": (
        SVG_STROKE
        + '<path d="M8.4 3h7.2l3.3 18H5.1z"/>'
        '<path d="M12 5.4v2.4M12 10.6v2.8M12 16.4v2.2"/></svg>'
    ),
    # Patologías: advertencia con señal de fisura
    "patologias": (
        SVG_STROKE
        + '<path d="M10.6 3.9a1.6 1.6 0 0 1 2.8 0l7.7 13.6a1.6 1.6 0 0 1-1.4 2.4H4.3a1.6 1.6 0 0 1-1.4-2.4z"/>'
        + '<path d="M12 9v3.6"/><path d="M12 16.2h.01"/></svg>'
    ),
    # Dataset: capas apiladas
    "dataset": (
        SVG_STROKE
        + '<path d="M12 3.2 21 7.7 12 12.2 3 7.7z"/>'
        '<path d="M3 12.2l9 4.5 9-4.5"/>'
        '<path d="M3 16.6l9 4.5 9-4.5"/></svg>'
    ),
    # Reportes: documento con líneas de texto
    "reportes": (
        SVG_STROKE
        + '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 3v5h5"/>'
        '<path d="M8.6 12.5h6.8M8.6 16h4.8"/></svg>'
    ),
    # Acerca del proyecto: círculo de información
    "acerca": (
        SVG_STROKE
        + '<circle cx="12" cy="12" r="8.6"/>'
        '<path d="M12 11.2v5"/><path d="M12 8h.01"/></svg>'
    ),
}

# Nombre Material de relleno por página: nunca se muestra (el CSS oculta el glifo
# y pinta el SVG). Se usan nombres DISTINTOS porque el motor de navegación
# deduplica iconos y, si todas las páginas comparten uno, deja de renderizarlo.
_PLACEHOLDER_ICONS = {
    "dashboard": ":material/dashboard:",
    "simulador_pci": ":material/insights:",
    "modelo_ml": ":material/model_training:",
    "registro_vias": ":material/add_road:",
    "patologias": ":material/warning:",
    "dataset": ":material/table_view:",
    "reportes": ":material/description:",
    "acerca": ":material/info:",
}


def nav_icon(slug: str) -> str:
    """SVG del icono de navegación; devuelve una cadena vacía si no hay icono."""
    return NAV_ICONS.get(slug, "")


def icon_html(slug: str) -> str:
    """HTML normalizado del icono (sin espacios ni saltos de línea).

    Cada SVG es distinto, por lo que Streamlit no puede deduplicarlos al
    comparar los iconos de todas las páginas.
    """
    return " ".join(nav_icon(slug).split())


def icon_slot(slug: str) -> str:
    """Valor para ``st.Page(icon=...)``: nombre Material válido y distinto por página."""
    return _PLACEHOLDER_ICONS[slug]
