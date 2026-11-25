"""Generación de informes PDF (ReportLab) y exportaciones CSV/JSON."""

from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import config, charts, data_loader, evaluation, pathology_service, prediction, road_service

GRAPHITE = colors.HexColor("#171A1C")
CARBON = colors.HexColor("#252B2D")
LIME = colors.HexColor("#A3E635")
SAGE = colors.HexColor("#D6E5B5")
BONE = colors.HexColor("#F1F3EF")
GREY = colors.HexColor("#5A6663")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
                                textColor=GRAPHITE, alignment=TA_LEFT, spaceAfter=2, leading=23),
        "subtitle": ParagraphStyle("S", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
                                   textColor=GREY, spaceAfter=10, leading=13),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
                             textColor=GRAPHITE, spaceBefore=12, spaceAfter=5, leading=15),
        "body": ParagraphStyle("B", parent=base["Normal"], fontName="Helvetica", fontSize=9,
                               textColor=colors.HexColor("#2B3234"), leading=13, alignment=TA_JUSTIFY),
        "small": ParagraphStyle("SM", parent=base["Normal"], fontName="Helvetica", fontSize=7.5,
                                textColor=GREY, leading=10.5),
        "cell": ParagraphStyle("C", parent=base["Normal"], fontName="Helvetica", fontSize=8,
                               textColor=colors.HexColor("#2B3234"), leading=10),
        "head": ParagraphStyle("H", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8,
                               textColor=colors.HexColor("#F1F3EF"), leading=10),
        "cellb": ParagraphStyle("CB", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8,
                                textColor=colors.HexColor("#2B3234"), leading=10),
    }


def _table(headers: list[str], rows: list[list], col_widths=None, highlight_first_col: bool = True) -> Table:
    st = _styles()
    data = [[Paragraph(f"<b>{h}</b>", st["head"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), st["cell"]) for c in row])
    tabla = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), CARBON),
        ("TEXTCOLOR", (0, 0), (-1, 0), BONE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8DDD4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F8F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if highlight_first_col:
        estilo.append(("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#EEF3E1")))
    tabla.setStyle(TableStyle(estilo))
    return tabla


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(GRAPHITE)
    canvas.rect(0, height - 14 * mm, width, 14 * mm, stroke=0, fill=1)
    canvas.setFillColor(LIME)
    canvas.rect(0, height - 15.4 * mm, width, 1.4 * mm, stroke=0, fill=1)
    canvas.setFillColor(BONE)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(18 * mm, height - 9.2 * mm, "PAVEMENT AI")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SAGE)
    canvas.drawRightString(width - 18 * mm, height - 9.2 * mm, "Informe técnico del sistema · SENATI")
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 10 * mm, "Los registros del dataset son SIMULADOS. No constituyen una evaluación oficial ASTM D6433.")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Página {doc.page}")
    canvas.restoreState()


def system_indicators() -> dict:
    """Indicadores generales calculados a partir de los datos persistidos."""
    df = data_loader.load_dataset()
    patologias = data_loader.load_pathologies()
    vias = data_loader.load_roads()
    total = len(df)
    pci_prom = float(df[config.TARGET_COLUMN].mean()) if total else 0.0
    criticos = df[df[config.TARGET_COLUMN] < config.PCI_CRITICAL_THRESHOLD]["codigo_tramo"].nunique() if total else 0
    return {
        "total_tramos": int(df["codigo_tramo"].nunique()) if total else 0,
        "total_inspecciones": int(total),
        "total_patologias_dataset": int(len(df)),
        "total_patologias_registradas": int(len(patologias)),
        "total_vias_registradas": int(len(vias)),
        "pci_promedio": round(pci_prom, 1),
        "condicion_promedio": config.classify_pci(pci_prom)[0],
        "tramos_criticos": int(criticos),
        "distritos": int(df["distrito"].nunique()) if total else 0,
    }


def build_pdf_report(include_history: bool = True, tramo: str | None = None,
                     horizonte: int = 5, antiguedad_escenario: float | None = None) -> bytes:
    """Construye el informe PDF completo y devuelve los bytes."""
    metadata = config.load_metadata()
    indicadores = system_indicators()
    df = data_loader.load_dataset()
    patologias = data_loader.load_pathologies()
    st = _styles()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=22 * mm, bottomMargin=18 * mm,
        title="PAVEMENT AI - Informe técnico", author="PAVEMENT AI",
    )
    story = []

    # --- Portada / encabezado -------------------------------------------------
    story.append(Paragraph("Informe técnico del sistema", st["title"]))
    story.append(Paragraph(
        f"PAVEMENT AI · {config.APP_SUBTITLE}<br/>"
        f"Generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')} · Versión {config.APP_VERSION}",
        st["subtitle"],
    ))

    datos_simulados = bool(indicadores["total_inspecciones"]) and (
        df["tipo_registro"].astype(str).str.upper().eq(config.TIPO_REGISTRO_SIMULADO).all()
    )
    if datos_simulados:
        aviso = Table(
            [[Paragraph(
                "<b>Aviso sobre los datos.</b> El dataset utilizado en este informe corresponde a registros "
                "<b>SIMULADOS</b> con fines académicos. No proviene de inspecciones reales verificadas en la "
                "provincia de Huánuco ni de una evaluación oficial del PCI (ASTM D6433). Los valores de PCI "
                "observado deben obtenerse de una evaluación técnica documentada.",
                st["body"])]],
            colWidths=[174 * mm],
        )
        aviso.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7E9")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#A3E635")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(aviso)

    # --- 1. Resumen -------------------------------------------------------------
    story.append(Paragraph("1. Resumen ejecutivo", st["h2"]))
    story.append(Paragraph(
        "PAVEMENT AI permite registrar tramos de pavimento rígido, documentar sus patologías y estimar el "
        "Índice de Condición del Pavimento (PCI) mediante un modelo de regresión lineal simple entrenado con "
        "la antigüedad de los tramos como variable independiente. Este informe resume los indicadores del "
        "sistema, las métricas reales del modelo, los resultados de la simulación solicitada y el historial "
        "de inspecciones disponibles.",
        st["body"],
    ))
    story.append(Spacer(1, 6))
    story.append(_table(
        ["Indicador", "Valor"],
        [
            ["Tramos en el dataset", indicadores["total_tramos"]],
            ["Inspecciones registradas", indicadores["total_inspecciones"]],
            ["Vías registradas por el usuario", indicadores["total_vias_registradas"]],
            ["Patologías registradas por el usuario", indicadores["total_patologias_registradas"]],
            ["PCI promedio", f"{indicadores['pci_promedio']}"],
            ["Condición promedio", indicadores["condicion_promedio"]],
            ["Tramos en condición crítica (PCI < 40)", indicadores["tramos_criticos"]],
            ["Distritos cubiertos", indicadores["distritos"]],
        ],
        col_widths=[110 * mm, 64 * mm],
    ))

    # --- 2. Modelo --------------------------------------------------------------
    story.append(Paragraph("2. Modelo de regresión lineal", st["h2"]))
    if metadata:
        metricas = metadata.get("metricas", {})
        story.append(Paragraph(
            f"Algoritmo: <b>{metadata.get('algoritmo', 'LinearRegression')}</b>. "
            f"Variable independiente: <b>{metadata.get('variable_independiente')}</b>; "
            f"variable dependiente: <b>{metadata.get('variable_dependiente')}</b>. "
            f"División: {metadata.get('metodo_division')}, semilla {metadata.get('random_state')}. "
            f"Entrenamiento: {metadata.get('conjunto_entrenamiento', {}).get('n_registros', '—')} registros; "
            f"prueba: {metadata.get('conjunto_prueba', {}).get('n_registros', '—')} registros.",
            st["body"],
        ))
        story.append(Spacer(1, 5))
        ecuacion = metadata.get("ecuacion", "")
        story.append(Paragraph(
            f"Ecuación ajustada: <b>{ecuacion.replace('−', '-')}</b><br/>"
            f"{evaluation.slope_explanation(metadata.get('pendiente', 0.0))}<br/>"
            f"{evaluation.interpret_r2(metricas.get('r2', 0.0))}",
            st["body"],
        ))
        story.append(Spacer(1, 6))
        story.append(_table(
            ["Métrica (conjunto de prueba)", "Valor"],
            [
                ["R²", f"{metricas.get('r2', 0):.4f}"],
                ["MAE (puntos PCI)", f"{metricas.get('mae', 0):.3f}"],
                ["RMSE (puntos PCI)", f"{metricas.get('rmse', 0):.3f}"],
            ],
            col_widths=[110 * mm, 64 * mm],
        ))
    else:
        story.append(Paragraph("El modelo aún no ha sido entrenado. Ejecute <i>python -m src.train_model</i>.", st["body"]))

    if metadata and not df.empty:
        fig = charts.matplotlib_scatter(df, metadata["intercepto"], metadata["pendiente"])
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", facecolor="white")
        img.seek(0)
        story.append(Spacer(1, 8))
        story.append(Image(img, width=170 * mm, height=85 * mm))

    story.append(PageBreak())

    # --- 3. Escenario -----------------------------------------------------------
    story.append(Paragraph("3. Resultados de predicción", st["h2"]))
    try:
        escenario = None
        if antiguedad_escenario is not None:
            escenario = prediction.predict_scenario(float(antiguedad_escenario), horizonte, metadata)
        elif tramo:
            fila = df[df["codigo_tramo"].astype(str) == str(tramo)]
            if not fila.empty:
                antig = float(fila.sort_values("fecha_inspeccion").iloc[-1][config.FEATURE_COLUMN])
                escenario = prediction.predict_scenario(antig, horizonte, metadata)

        if escenario is None and not df.empty:
            antig = float(df.sort_values("fecha_inspeccion").iloc[-1][config.FEATURE_COLUMN])
            escenario = prediction.predict_scenario(antig, horizonte, metadata)

        if escenario:
            story.append(Paragraph(
                f"Escenario: tramo <b>{tramo or 'libre'}</b>, antigüedad actual {escenario['antiguedad_actual']:.1f} años, "
                f"horizonte de <b>{escenario['horizonte_anios']} año(s)</b> (antigüedad futura "
                f"{escenario['antiguedad_futura']:.1f} años).",
                st["body"],
            ))
            story.append(Spacer(1, 5))
            story.append(_table(
                ["Concepto", "Valor"],
                [
                    ["PCI estimado actual", f"{escenario['pci_actual']:.2f} ({escenario['estado_actual']})"],
                    ["PCI estimado futuro (recortado 0-100)", f"{escenario['pci_futuro']:.2f} ({escenario['estado_futuro']})"],
                    ["PCI futuro matemático (sin recortar)", f"{escenario['pci_futuro_raw']:.2f}"],
                    ["Variación del PCI", f"{escenario['variacion']:+.2f} puntos"],
                ],
                col_widths=[110 * mm, 64 * mm],
            ))
            for advertencia in escenario["advertencias"]:
                story.append(Spacer(1, 3))
                story.append(Paragraph(f"<i>Advertencia: {advertencia}</i>", st["small"]))
        else:
            story.append(Paragraph("No hay registros suficientes para calcular un escenario de predicción.", st["body"]))
    except prediction.ModelNotTrainedError as exc:
        story.append(Paragraph(f"No fue posible generar predicciones: {exc}", st["body"]))

    # --- 4. Patologías ----------------------------------------------------------
    story.append(Paragraph("4. Patologías registradas por el usuario", st["h2"]))
    if patologias.empty:
        story.append(Paragraph("Aún no se han registrado patologías desde el módulo de patologías.", st["body"]))
    else:
        resumen = (
            patologias.groupby("tipo_patologia")
            .agg(registros=("id_patologia", "count"), extension=("cantidad_fallas", "sum"))
            .sort_values("registros", ascending=False)
        )
        filas = [[tipo, int(fila.registros), f"{float(fila.extension):.1f}"] for tipo, fila in resumen.iterrows()]
        story.append(Paragraph(
            f"Se han registrado {len(patologias)} patologías en {patologias['codigo_tramo'].nunique()} tramo(s). "
            "El registro de patologías documenta el deterioro, pero no calcula automáticamente un PCI oficial.",
            st["body"],
        ))
        story.append(Spacer(1, 5))
        story.append(_table(["Tipo de patología", "Registros", "Extensión / cantidad total"], filas, col_widths=[96 * mm, 36 * mm, 42 * mm]))
        recientes = patologias.tail(8)
        story.append(Spacer(1, 6))
        story.append(_table(
            ["ID", "Tramo", "Patología", "Severidad", "Fecha"],
            [[r["id_patologia"], r["codigo_tramo"], r["tipo_patologia"], r["severidad"], r["fecha_inspeccion"]]
             for _, r in recientes.iterrows()],
            col_widths=[20 * mm, 28 * mm, 62 * mm, 30 * mm, 34 * mm],
        ))

    # --- 5. Historial -----------------------------------------------------------
    if include_history and not df.empty:
        story.append(Paragraph("5. Historial de inspecciones (últimas 25)", st["h2"]))
        historial = df.sort_values("fecha_inspeccion", ascending=False).head(25)
        filas = []
        for _, r in historial.iterrows():
            fecha = pd.to_datetime(r["fecha_inspeccion"]).strftime("%d/%m/%Y") if pd.notna(r["fecha_inspeccion"]) else "—"
            filas.append([r["id_registro"], r["codigo_tramo"], fecha, f"{r['antiguedad_anios']:.1f}",
                          r["tipo_patologia"], f"{r['pci_observado']:.1f}"])
        story.append(_table(
            ["ID", "Tramo", "Fecha", "Antig.", "Patología", "PCI"],
            filas, col_widths=[22 * mm, 28 * mm, 24 * mm, 16 * mm, 60 * mm, 24 * mm],
        ))
        condiciones = pd.Series([config.classify_pci(float(v))[0] for v in df[config.TARGET_COLUMN]]).value_counts()
        story.append(Spacer(1, 8))
        story.append(Paragraph("Distribución de condiciones del pavimento", st["h2"]))
        story.append(_table(
            ["Condición", "Rango PCI", "Inspecciones"],
            [[etiqueta, f"{low}-{high}", int(condiciones.get(etiqueta, 0))] for low, high, etiqueta, _c, _f in config.PCI_RANGES],
            col_widths=[60 * mm, 57 * mm, 57 * mm],
        ))

    # --- 6. Limitaciones --------------------------------------------------------
    story.append(Paragraph("6. Limitaciones del estudio", st["h2"]))
    story.append(Paragraph(
        "• Los registros del dataset son sintéticos y no representan la realidad física de las vías de Huánuco.<br/>"
        "• La regresión lineal simple usa solo la antigüedad; no incorpora tránsito, drenaje, tipo de patología ni espesor de losa.<br/>"
        "• La relación PCI-antigüedad es aproximadamente lineal por construcción del dataset; en campo la degradación es no lineal.<br/>"
        "• Las predicciones más allá del rango de antigüedad observado son extrapolaciones y deben interpretarse con cautela.<br/>"
        "• Los rangos de condición mostrados son una referencia visual y no sustituyen la metodología oficial PCI (ASTM D6433).",
        st["body"],
    ))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return buffer.getvalue()


def save_pdf_report(**kwargs) -> tuple[str, bytes]:
    """Genera el PDF, lo guarda en ``reports/`` y devuelve (ruta, bytes)."""
    config.ensure_directories()
    pdf_bytes = build_pdf_report(**kwargs)
    nombre = f"informe_pavement_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    ruta = config.REPORTS_DIR / nombre
    ruta.write_bytes(pdf_bytes)
    return ruta.name, pdf_bytes


def export_csv(df: pd.DataFrame) -> bytes:
    """Exporta un dataframe a CSV (UTF-8 con BOM para Excel)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def export_json(payload: dict) -> bytes:
    """Exporta un diccionario a JSON legible."""
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def build_export_payload() -> dict:
    """Paquete JSON con indicadores, métricas y metadatos del sistema."""
    return {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "aplicacion": config.APP_NAME,
        "version": config.APP_VERSION,
        "origen_datos": "SIMULADO - dataset académico",
        "indicadores": system_indicators(),
        "metricas_modelo": config.load_metadata().get("metricas", {}),
        "metadata_modelo": config.load_metadata(),
        "vias": road_service.road_summary(),
        "patologias": pathology_service.pathology_stats(),
    }
