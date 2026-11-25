"""Gráficos interactivos (Plotly) y estáticos (Matplotlib) de PAVEMENT AI.

Todos los gráficos comparten la paleta grafito / verde lima y se construyen a
partir de datos reales del dataset, las patologías o el modelo entrenado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config

GRAPHITE = config.COLORS["graphite"]
CARBON = config.COLORS["carbon"]
LIME = config.COLORS["lime"]
SAGE = config.COLORS["sage"]
BONE = config.COLORS["bone"]
MUTED = config.COLORS["muted"]
LINE = config.COLORS["line"]

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _base_layout(title: str = "", height: int = 320) -> dict:
    return {
        "title": {"text": title, "font": {"size": 14, "color": BONE}, "x": 0.01, "xanchor": "left"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": SAGE, "family": "Inter, Segoe UI, sans-serif", "size": 12},
        "margin": {"l": 48, "r": 18, "t": 44 if title else 16, "b": 40},
        "height": height,
        "hoverlabel": {"bgcolor": CARBON, "font": {"color": BONE}, "bordercolor": LINE},
        "legend": {"orientation": "h", "y": -0.22, "font": {"size": 11}},
    }


def _axis(title: str = "") -> dict:
    return {
        "title": {"text": title, "font": {"size": 11, "color": MUTED}},
        "gridcolor": "rgba(214,229,181,0.10)",
        "zerolinecolor": "rgba(214,229,181,0.18)",
        "linecolor": LINE,
        "tickfont": {"size": 11, "color": MUTED},
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def condition_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Cantidad de inspecciones por condición de pavimento (según pci_observado)."""
    etiquetas = [r[2] for r in config.PCI_RANGES]
    colores = {r[2]: r[3] for r in config.PCI_RANGES}
    if df.empty:
        conteo = pd.Series(0, index=etiquetas)
    else:
        condiciones = df[config.TARGET_COLUMN].apply(lambda v: config.classify_pci(float(v))[0])
        conteo = condiciones.value_counts().reindex(etiquetas, fill_value=0)

    fig = go.Figure(
        go.Bar(
            x=list(conteo.index),
            y=list(conteo.values),
            marker_color=[colores[e] for e in conteo.index],
            marker_line_color=GRAPHITE,
            marker_line_width=1,
            text=list(conteo.values),
            textposition="outside",
            textfont={"color": BONE, "size": 11},
            hovertemplate="%{x}: %{y} inspecciones<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Distribución de condiciones del pavimento", 300))
    fig.update_xaxes(**_axis("Condición"))
    fig.update_yaxes(**_axis("Inspecciones"))
    return fig


def pathology_frequency_chart(df: pd.DataFrame) -> go.Figure:
    """Patologías más frecuentes en el dataset de inspecciones."""
    conteo = df["tipo_patologia"].value_counts() if not df.empty else pd.Series(dtype=int)
    conteo = conteo.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=list(conteo.values),
            y=[p.replace("Grietas ", "Grietas<br>") for p in conteo.index],
            orientation="h",
            marker_color=LIME,
            marker_line_width=0,
            text=list(conteo.values),
            textposition="outside",
            textfont={"color": BONE, "size": 11},
            hovertemplate="%{y}: %{x} registros<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Patologías más frecuentes", 300))
    fig.update_xaxes(**_axis("Registros"))
    fig.update_yaxes(**_axis(""))
    return fig


def pci_by_tramo_chart(df: pd.DataFrame, top: int = 12) -> go.Figure:
    """PCI promedio por tramo, ordenado de peor a mejor condición."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout("PCI promedio por tramo", 320))
        return fig

    agrupado = (
        df.groupby("codigo_tramo")[config.TARGET_COLUMN]
        .agg(["mean", "count"])
        .sort_values("mean")
        .head(top)
        .reset_index()
    )
    colores = [config.classify_pci(v)[1] for v in agrupado["mean"]]
    fig = go.Figure(
        go.Bar(
            x=agrupado["codigo_tramo"],
            y=agrupado["mean"].round(1),
            marker_color=colores,
            marker_line_width=0,
            customdata=agrupado["count"],
            hovertemplate="%{x}<br>PCI promedio: %{y}<br>Inspecciones: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(f"PCI promedio por tramo (los {top} más críticos)", 320))
    fig.update_xaxes(**_axis("Tramo"), tickangle=-35)
    fig.update_yaxes(**_axis("PCI promedio"), range=[0, 100])
    return fig


def age_vs_pci_chart(df: pd.DataFrame, intercept: float | None = None, slope: float | None = None) -> go.Figure:
    """Dispersión antigüedad vs PCI con la recta de regresión si se conoce."""
    fig = go.Figure()
    if not df.empty:
        colores = [config.classify_pci(float(v))[1] for v in df[config.TARGET_COLUMN]]
        fig.add_trace(
            go.Scatter(
                x=df[config.FEATURE_COLUMN],
                y=df[config.TARGET_COLUMN],
                mode="markers",
                marker={"color": colores, "size": 8, "line": {"width": 1, "color": GRAPHITE}},
                name="Inspecciones",
                customdata=df["codigo_tramo"],
                hovertemplate="Tramo %{customdata}<br>Antigüedad: %{x:.1f} años<br>PCI: %{y:.1f}<extra></extra>",
            )
        )
        x_min, x_max = float(df[config.FEATURE_COLUMN].min()), float(df[config.FEATURE_COLUMN].max())
    else:
        x_min, x_max = 0.0, 30.0

    if intercept is not None and slope is not None:
        xs = np.linspace(max(0, x_min - 1), x_max + 2, 60)
        ys = np.clip(intercept + slope * xs, 0, 100)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": LIME, "width": 2.5},
                name="Regresión lineal",
                hovertemplate="Antigüedad: %{x:.1f}<br>PCI estimado: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(**_base_layout("Relación entre antigüedad y PCI", 340))
    fig.update_xaxes(**_axis("Antigüedad (años)"))
    fig.update_yaxes(**_axis("PCI observado"), range=[0, 102])
    return fig


def inspections_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Evolución mensual del número de inspecciones y del PCI promedio."""
    fig = go.Figure()
    if df.empty or "fecha_inspeccion" not in df.columns:
        fig.update_layout(**_base_layout("Evolución de inspecciones", 300))
        return fig

    temporal = df.copy()
    temporal["fecha_inspeccion"] = pd.to_datetime(temporal["fecha_inspeccion"], errors="coerce")
    temporal = temporal.dropna(subset=["fecha_inspeccion"])
    if temporal.empty:
        fig.update_layout(**_base_layout("Evolución de inspecciones", 300))
        return fig

    temporal["periodo"] = temporal["fecha_inspeccion"].dt.to_period("Q").astype(str)
    agrupado = temporal.groupby("periodo").agg(inspecciones=("id_registro", "count"), pci=(config.TARGET_COLUMN, "mean")).reset_index()

    fig.add_trace(
        go.Bar(
            x=agrupado["periodo"],
            y=agrupado["inspecciones"],
            name="Inspecciones",
            marker_color=CARBON,
            marker_line={"color": LINE, "width": 1},
            hovertemplate="%{x}: %{y} inspecciones<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agrupado["periodo"],
            y=agrupado["pci"].round(1),
            name="PCI promedio",
            yaxis="y2",
            mode="lines+markers",
            line={"color": LIME, "width": 2.5},
            marker={"size": 7},
            hovertemplate="%{x}: PCI %{y}<extra></extra>",
        )
    )
    layout = _base_layout("Evolución de inspecciones y PCI promedio", 320)
    layout["yaxis2"] = {
        "overlaying": "y",
        "side": "right",
        "range": [0, 100],
        "showgrid": False,
        "title": {"text": "PCI promedio", "font": {"size": 11, "color": MUTED}},
        "tickfont": {"size": 11, "color": MUTED},
    }
    fig.update_layout(**layout)
    fig.update_xaxes(**_axis("Trimestre"), tickangle=-35)
    fig.update_yaxes(**_axis("Inspecciones"))
    return fig


# ---------------------------------------------------------------------------
# Modelo ML
# ---------------------------------------------------------------------------
def scatter_fit_chart(train_df: pd.DataFrame, test_df: pd.DataFrame, intercept: float, slope: float) -> go.Figure:
    """Puntos de entrenamiento y prueba con la recta ajustada."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_df[config.FEATURE_COLUMN], y=train_df[config.TARGET_COLUMN],
            mode="markers", name="Entrenamiento",
            marker={"color": SAGE, "size": 8, "opacity": 0.75},
            hovertemplate="Antigüedad %{x:.1f}<br>PCI %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_df[config.FEATURE_COLUMN], y=test_df[config.TARGET_COLUMN],
            mode="markers", name="Prueba",
            marker={"color": LIME, "size": 10, "symbol": "diamond", "line": {"width": 1, "color": GRAPHITE}},
            hovertemplate="Antigüedad %{x:.1f}<br>PCI %{y:.1f}<extra></extra>",
        )
    )
    x_min = float(min(train_df[config.FEATURE_COLUMN].min(), test_df[config.FEATURE_COLUMN].min()))
    x_max = float(max(train_df[config.FEATURE_COLUMN].max(), test_df[config.FEATURE_COLUMN].max()))
    xs = np.linspace(max(0, x_min - 1), x_max + 2, 80)
    fig.add_trace(
        go.Scatter(
            x=xs, y=np.clip(intercept + slope * xs, 0, 100), mode="lines", name="Recta ajustada",
            line={"color": LIME, "width": 2.5},
            hovertemplate="Antigüedad %{x:.1f}<br>PCI estimado %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Datos reales y línea de regresión", 380))
    fig.update_xaxes(**_axis("Antigüedad (años)"))
    fig.update_yaxes(**_axis("PCI observado"), range=[0, 102])
    return fig


def predicted_vs_actual_chart(y_true, y_pred) -> go.Figure:
    """Valores predichos frente a reales, con la diagonal ideal."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true, y=y_pred, mode="markers", name="Registros de prueba",
            marker={"color": LIME, "size": 9, "line": {"width": 1, "color": GRAPHITE}},
            hovertemplate="Real %{x:.1f}<br>Predicho %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Predicción perfecta",
                   line={"color": MUTED, "width": 1.5, "dash": "dash"}),
    )
    fig.update_layout(**_base_layout("Predicho vs. real (conjunto de prueba)", 340))
    fig.update_xaxes(**_axis("PCI real"), range=[0, 102])
    fig.update_yaxes(**_axis("PCI predicho"), range=[0, 102])
    return fig


def residuals_chart(y_true, y_pred) -> go.Figure:
    """Residuos frente al valor predicho."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=res, mode="markers", name="Residuos",
            marker={"color": SAGE, "size": 9, "line": {"width": 1, "color": GRAPHITE}},
            hovertemplate="Predicho %{x:.1f}<br>Residuo %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": LIME, "width": 1.5, "dash": "dash"})
    fig.update_layout(**_base_layout("Residuos del modelo", 340))
    fig.update_xaxes(**_axis("PCI predicho"), range=[0, 102])
    fig.update_yaxes(**_axis("Residuo (real − predicho)"))
    return fig


def simulator_chart(df: pd.DataFrame, intercept: float, slope: float, antiguedad_actual: float,
                    antiguedad_futura: float, pci_actual: float, pci_futuro: float,
                    rango_entrenamiento: dict | None = None) -> go.Figure:
    """Recta del modelo con el punto actual y el proyectado."""
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df[config.FEATURE_COLUMN], y=df[config.TARGET_COLUMN], mode="markers",
                name="Inspecciones (dataset)",
                marker={"color": "rgba(214,229,181,0.45)", "size": 7},
                hovertemplate="Antigüedad %{x:.1f}<br>PCI %{y:.1f}<extra></extra>",
            )
        )

    if rango_entrenamiento:
        fig.add_vrect(
            x0=rango_entrenamiento.get("min", 0), x1=rango_entrenamiento.get("max", 0),
            fillcolor="rgba(163,230,53,0.06)", line_width=0,
            annotation_text="rango entrenado", annotation_position="top left",
            annotation_font={"size": 10, "color": MUTED},
        )

    x_min = 0.0
    x_max = max(antiguedad_futura + 3, 10)
    xs = np.linspace(x_min, x_max, 120)
    fig.add_trace(
        go.Scatter(x=xs, y=intercept + slope * xs, mode="lines", name="Modelo lineal",
                   line={"color": LIME, "width": 2.5},
                   hovertemplate="Antigüedad %{x:.1f}<br>PCI %{y:.1f}<extra></extra>")
    )
    fig.add_trace(
        go.Scatter(x=[antiguedad_actual], y=[pci_actual], mode="markers+text", name="PCI actual",
                   marker={"color": BONE, "size": 13, "symbol": "circle", "line": {"width": 2, "color": GRAPHITE}},
                   text=["Actual"], textposition="top center", textfont={"color": BONE, "size": 11})
    )
    fig.add_trace(
        go.Scatter(x=[antiguedad_futura], y=[pci_futuro], mode="markers+text", name="PCI proyectado",
                   marker={"color": "#E3C05A", "size": 13, "symbol": "diamond", "line": {"width": 2, "color": GRAPHITE}},
                   text=["Proyectado"], textposition="top center", textfont={"color": BONE, "size": 11})
    )
    fig.update_layout(**_base_layout("Proyección del PCI con el modelo entrenado", 400))
    fig.update_xaxes(**_axis("Antigüedad (años)"))
    fig.update_yaxes(**_axis("PCI"), range=[0, 105])
    return fig


def severity_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Distribución de severidades registradas en patologías."""
    orden = config.SEVERIDADES
    colores = {"Baja": SAGE, "Media": "#E3C05A", "Alta": "#C85B4A"}
    conteo = df["severidad"].value_counts().reindex(orden, fill_value=0) if not df.empty else pd.Series(0, index=orden)
    fig = go.Figure(
        go.Pie(
            labels=list(conteo.index), values=list(conteo.values), hole=0.62,
            marker={"colors": [colores[i] for i in conteo.index], "line": {"color": GRAPHITE, "width": 2}},
            textinfo="label+percent", textfont={"color": BONE, "size": 11},
            hovertemplate="%{label}: %{value} registros<extra></extra>",
        )
    )
    layout = _base_layout("Severidad de patologías registradas", 300)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Matplotlib (para el PDF)
# ---------------------------------------------------------------------------
def matplotlib_scatter(df: pd.DataFrame, intercept: float, slope: float):
    """Gráfico estático para el informe PDF (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.scatter(df[config.FEATURE_COLUMN], df[config.TARGET_COLUMN], s=22, color="#8FA36A",
               edgecolor="#3F4A32", linewidth=0.5, label="Inspecciones", zorder=3)

    xs = np.linspace(0, max(30, float(df[config.FEATURE_COLUMN].max()) + 2), 100)
    ax.plot(xs, np.clip(intercept + slope * xs, 0, 100), color="#5F8B1F", linewidth=2, label="Regresión lineal", zorder=4)

    ax.set_xlabel("Antigüedad (años)", fontsize=9, color="#333333")
    ax.set_ylabel("PCI observado", fontsize=9, color="#333333")
    ax.set_ylim(0, 102)
    ax.set_title("Antigüedad vs. PCI con recta ajustada", fontsize=10, color="#171A1C", loc="left")
    ax.grid(color="#E2E5DD", linewidth=0.6)
    ax.tick_params(labelsize=8, colors="#555555")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
