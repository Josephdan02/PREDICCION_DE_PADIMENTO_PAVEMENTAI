"""Componentes y utilidades de interfaz para PAVEMENT AI (Streamlit).

Concentra el CSS, los encabezados de página y las tarjetas Bento para que las
páginas solo declaren contenido.

Nota técnica: Streamlit envuelve cada ``st.markdown`` en su propio contenedor,
por lo que un ``<div>`` abierto en una llamada no puede envolver contenido de
otra. Por eso:
- las cuadrículas de tarjetas KPI se emiten como **un solo bloque HTML**;
- las tarjetas que contienen gráficos o tablas usan ``st.container(border=True)``,
  estilizado con CSS como tarjeta.
"""

from __future__ import annotations

import html
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from src import charts, config


# ---------------------------------------------------------------------------
# Tema
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Inserta la hoja de estilos personalizada una sola vez por sesión."""
    css_path = config.STYLE_PATH
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def setup_page(title: str, icon: str = "🛣️") -> None:
    """Configuración común de cada página."""
    st.set_page_config(
        page_title=f"{title} · {config.APP_NAME}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    config.ensure_directories()
    inject_css()


def page_header(title: str, description: str = "", eyebrow: str = "", back_to: str = "pages/dashboard.py") -> None:
    """Encabezado jerárquico de página.

    Se usan ``div`` (no ``h1``) para que Streamlit no convierta el bloque en un
    componente de encabezado con su propio espaciado; así el estilo es predecible.

    ``back_to`` dibuja un enlace nativo (``st.page_link``, con su icono SVG) de
    vuelta al dashboard, ya que la barra lateral solo contiene el menú.
    """
    if back_to:
        atributos = config.nav_link_attributes().get(back_to)
        if atributos:
            # ``page_link`` requiere que la página esté registrada en st.navigation;
            # al ejecutar la página de forma aislada (pruebas, AppTest) se omite.
            try:
                # Tampoco admite SVG: se declara el nombre Material y el CSS lo
                # sustituye por el icono de retorno.
                st.page_link(back_to, label="Panel de control", icon=atributos["slot"])
            except Exception:  # noqa: BLE001 - navegación no registrada
                pass

    eyebrow_html = f'<div class="page-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    desc_html = f'<div class="page-desc-color">{html.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="page-head"><div class="page-head-inner">{eyebrow_html}'
        f'<div class="page-title" role="heading" aria-level="1">{html.escape(title)}</div>{desc_html}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    """Marca del producto + estado del modelo (bloque superior de la barra lateral).

    Se emiten en una sola llamada porque Streamlit agrupa en un único bloque
    todos los elementos que se dibujan antes de ``st.navigation``; así la
    cuadrícula de la barra lateral los coloca juntos por encima del menú.
    """
    meta = config.load_metadata()
    if meta:
        r2 = meta.get("metricas", {}).get("r2", 0)
        estado = (
            '<div class="bento model-card" style="padding:0.7rem 0.8rem; min-height:auto; margin-top:0.55rem;">'
            '<div class="bento-label">Modelo cargado</div>'
            '<div style="font-size:0.85rem; font-weight:700; color:#A3E635; margin-top:2px;">LinearRegression</div>'
            f'<div class="bento-foot" style="margin-top:4px;">R² = {r2:.3f} · joblib</div>'
            "</div>"
        )
    else:
        estado = (
            '<div class="model-card" style="margin-top:0.55rem; font-size:0.78rem; '
            'color:#E3C05A;">Modelo sin entrenar</div>'
        )

    st.sidebar.markdown(
        f"""
        <div class="brand">
            <div class="brand-mark">PA</div>
            <div>
                <div class="brand-name">PAVEMENT AI</div>
                <div class="brand-sub">Pavimentos rígidos</div>
            </div>
        </div>
        {estado}
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    """Pie de la barra lateral (se conserva por compatibilidad de API)."""
    return


def page_footer() -> None:
    st.markdown(
        '<div class="app-footer">'
        f"<span>{config.APP_NAME} · {config.APP_SUBTITLE}</span>"
        "<span>Referencia visual PCI · no sustituye la metodología oficial ASTM D6433</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Componentes Bento
# ---------------------------------------------------------------------------
def bento_metric(label: str, value: str, foot: str = "", icon: str = "", span: int = 3,
                 hero: bool = False, unit: str = "", value_color: str | None = None,
                 extra_html: str = "", value_sm: bool = False) -> str:
    """HTML de una tarjeta KPI del Bento Grid.

    ``value_sm`` reduce el tamaño del valor para textos largos (p. ej. nombres
    de algoritmo) que de otro modo desbordarían la tarjeta.
    """
    classes = f"bento bento-span-{span}" + (" bento-hero" if hero else "")
    value_classes = "bento-value bento-value-sm" if value_sm else "bento-value"
    unit_html = f'<span class="bento-unit">{html.escape(unit)}</span>' if unit else ""
    color_style = f' style="color:{value_color};"' if value_color else ""
    icon_html = f'<span class="bento-icon">{icon}</span>' if icon else ""
    foot_html = f'<div class="bento-foot">{foot}</div>' if foot else ""
    return (
        f'<div class="{classes}">'
        f'<div><div class="bento-head"><span class="bento-label">{html.escape(label)}</span>{icon_html}</div>'
        f'<div class="{value_classes}"{color_style}>{value}{unit_html}</div>{extra_html}</div>'
        f"{foot_html}</div>"
    )


def render_metric_grid(cards: list[dict]) -> None:
    """Renderiza una cuadrícula Bento completa en un único bloque HTML."""
    tarjetas = "".join(bento_metric(**card) for card in cards)
    st.markdown(f'<div class="bento-grid">{tarjetas}</div>', unsafe_allow_html=True)


@contextmanager
def card(title: str = "", right: str = ""):
    """Tarjeta contenedora real (envuelve gráficos y tablas)."""
    with st.container(border=True):
        if title:
            right_html = f'<span class="chip chip-neutral">{html.escape(right)}</span>' if right else ""
            st.markdown(
                f'<div class="bento-card-title"><span>{html.escape(title)}</span>{right_html}</div>',
                unsafe_allow_html=True,
            )
        yield


def section_card(title: str, right: str = "") -> None:
    """Alias histórico mantenido por compatibilidad (usar ``card``)."""
    st.markdown(
        f'<div class="bento-card-title"><span>{html.escape(title)}</span>'
        f'{"<span class=chip chip-neutral>" + html.escape(right) + "</span>" if right else ""}</div>',
        unsafe_allow_html=True,
    )


def end_card() -> None:
    """No-op: se conserva para no romper llamadas existentes."""


def pci_badge(value: float, large: bool = False) -> str:
    """Chip HTML con la condición del pavimento."""
    etiqueta, color, texto = config.classify_pci(value)
    size = "chip-lg" if large else ""
    return f'<span class="chip {size}" style="background:{color}; color:{texto};">{etiqueta}</span>'


def pci_scale_bar(value: float) -> str:
    """Barra proporcional de la escala completa del PCI con marcador de posición."""
    value = max(0.0, min(100.0, float(value)))
    anchos = [(high - low + 1) for low, high, *_ in config.PCI_RANGES]
    total = sum(anchos)
    segmentos = "".join(
        f'<span style="width:{w / total * 100:.2f}%; background:{r[3]};"></span>'
        for w, r in zip(anchos, config.PCI_RANGES)
    )
    return (
        f'<div class="pci-scale-wrap"><div class="pci-scale">{segmentos}</div>'
        f'<span class="pci-marker" style="left:{value:.1f}%;"></span></div>'
        f'<div class="bento-foot">PCI consultado: <strong>{value:.1f}</strong> / 100</div>'
    )


def years_label(n: int) -> str:
    """Etiqueta de años con concordancia singular/plural."""
    return "1 año" if int(n) == 1 else f"{int(n)} años"


def legend() -> None:
    """Leyenda de condiciones del PCI."""
    items = "".join(
        f'<div class="legend-item" style="border-left:4px solid {bg};">'
        f'<div class="legend-range">{low}-{high}</div>'
        f'<div class="legend-name" style="color:{fg};">{label}</div></div>'
        for low, high, label, bg, fg in config.PCI_RANGES
    )
    st.markdown(f'<div class="legend-grid">{items}</div>', unsafe_allow_html=True)


def plot(fig, key: str | None = None) -> None:
    """Renderiza un gráfico Plotly con configuración común."""
    st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG, key=key)


def activity_list(rows: list[dict]) -> str:
    """Lista de actividad reciente."""
    if not rows:
        return '<div class="bento-foot">Sin actividad registrada todavía.</div>'
    filas = "".join(
        f'<div class="activity-row"><div class="activity-icon">{r.get("icon", "•")}</div>'
        f'<div class="activity-main"><div class="activity-title">{html.escape(str(r.get("title", "")))}</div>'
        f'<div class="activity-meta">{html.escape(str(r.get("meta", "")))}</div></div></div>'
        for r in rows
    )
    return f'<div class="activity">{filas}</div>'


# ---------------------------------------------------------------------------
# Utilidades de datos
# ---------------------------------------------------------------------------
def dataframe_display(df: pd.DataFrame, height: int = 320, column_labels: dict | None = None) -> None:
    """Tabla Streamlit con etiquetas legibles y fechas en formato dd/mm/aaaa."""
    vista = _format_dates(df)
    if column_labels:
        vista = vista.rename(columns=column_labels)
    st.dataframe(vista, use_container_width=True, hide_index=True, height=height)


def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de fecha a texto dd/mm/aaaa para su visualización."""
    vista = df.copy()
    for col in vista.columns:
        if "fecha" in str(col).lower():
            vista[col] = pd.to_datetime(vista[col], errors="coerce").dt.strftime("%d/%m/%Y")
            vista[col] = vista[col].fillna("—")
    return vista


def relative_time(dt) -> str:
    """Texto amigable para una fecha."""
    if dt is None or pd.isna(dt):
        return "—"
    return pd.Timestamp(dt).strftime("%d/%m/%Y")
