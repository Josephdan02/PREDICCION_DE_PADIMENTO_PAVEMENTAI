"""Configuración central de PAVEMENT AI.

Este módulo concentra rutas, constantes del dominio y la paleta de colores.
Ningún otro módulo debe definir rutas por su cuenta: todas se importan desde aquí.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
UPLOADS_DIR = BASE_DIR / "uploads"
PATHOLOGY_UPLOADS_DIR = UPLOADS_DIR / "patologias"
REPORTS_DIR = BASE_DIR / "reports"
STORAGE_DIR = BASE_DIR / "storage"
TESTS_DIR = BASE_DIR / "tests"

DATASET_PATH = DATA_DIR / "pavimentos.csv"
ROADS_PATH = STORAGE_DIR / "vias.csv"
PATHOLOGIES_PATH = STORAGE_DIR / "patologias.csv"
MODEL_PATH = MODELS_DIR / "linear_regression.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
STYLE_PATH = ASSETS_DIR / "styles.css"

# ---------------------------------------------------------------------------
# Constantes del dominio
# ---------------------------------------------------------------------------
APP_NAME = "PAVEMENT AI"
APP_SUBTITLE = "Gestión de pavimentos rígidos · Municipalidad Provincial de Huánuco"
APP_VERSION = "1.0.0"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Variable independiente y dependiente del modelo (regresión lineal simple)
FEATURE_COLUMN = "antiguedad_anios"
TARGET_COLUMN = "pci_observado"
GROUP_COLUMN = "codigo_tramo"

# Columnas obligatorias del dataset
DATASET_COLUMNS = [
    "id_registro",
    "codigo_tramo",
    "distrito",
    "tipo_pavimento",
    "fecha_inspeccion",
    "antiguedad_anios",
    "volumen_transito",
    "tipo_patologia",
    "severidad",
    "cantidad_fallas",
    "pci_observado",
    "tipo_registro",
]

DISTRITOS = ["Huánuco", "Amarilis", "Pillco Marca", "Santa María del Valle"]

TIPOS_PAVIMENTO = ["Rígido"]  # El alcance del sistema son pavimentos rígidos de concreto

PATOLOGIAS = {
    "Grietas longitudinales": "Fisuras paralelas al eje de la vía, normalmente por contracción del concreto.",
    "Grietas transversales": "Fisuras perpendiculares al eje de la vía, asociadas a juntas o retracción.",
    "Grietas de esquina": "Fisuras que intersectan las juntas formando una pieza angular en la losa.",
    "Losa dividida": "Losa fracturada en cuatro o más pedazos por fisuras superpuestas.",
    "Despostillamiento": "Pérdida de material en bordes o juntas de la losa.",
    "Escalonamiento": "Desnivel entre losas adyacentes en una junta.",
}

SEVERIDADES = ["Baja", "Media", "Alta"]

PESO_SEVERIDAD = {"Baja": 1.0, "Media": 2.0, "Alta": 3.5}

# Rangos de referencia visual del PCI (no sustituyen la metodología oficial ASTM D6433)
PCI_RANGES = [
    (85, 100, "Excelente", "#A3E635", "#171A1C"),
    (70, 84, "Muy bueno", "#C6E887", "#171A1C"),
    (55, 69, "Bueno", "#D6E5B5", "#171A1C"),
    (40, 54, "Regular", "#E3C05A", "#171A1C"),
    (25, 39, "Malo", "#DB8E4A", "#171A1C"),
    (10, 24, "Muy malo", "#C85B4A", "#F1F3EF"),
    (0, 9, "Fallado", "#8E2F26", "#F1F3EF"),
]

PCI_MIN = 0.0
PCI_MAX = 100.0

# Tramos en condición crítica (PCI < 40 => Regular hacia abajo)
PCI_CRITICAL_THRESHOLD = 40.0

# ---------------------------------------------------------------------------
# Navegación
# ---------------------------------------------------------------------------
# Orden de las páginas y su icono SVG (ver ``src/nav_icons.py``). La clave es el
# slug de la URL de la página; se usa tanto para construir la navegación como
# para resolver el icono en la barra lateral.
NAV_SECTIONS: dict[str, list[tuple[str, str, str]]] = {
    "Análisis": [
        ("pages/dashboard.py", "Dashboard", "dashboard"),
        ("pages/simulador_pci.py", "Simulador PCI", "simulador_pci"),
        ("pages/modelo_ml.py", "Modelo ML", "modelo_ml"),
    ],
    "Gestión": [
        ("pages/registro_vias.py", "Registro de vías", "registro_vias"),
        ("pages/patologias.py", "Patologías", "patologias"),
        ("pages/dataset.py", "Dataset", "dataset"),
    ],
    "Sistema": [
        ("pages/reportes.py", "Reportes", "reportes"),
        ("pages/acerca.py", "Acerca del proyecto", "acerca"),
    ],
}


def nav_link_attributes() -> dict[str, dict[str, str]]:
    """Atributos ``st.page_link`` para cada página, con su icono SVG."""
    from src import nav_icons

    atributos: dict[str, dict[str, str]] = {}
    for entradas in NAV_SECTIONS.values():
        for ruta, titulo, slug in entradas:
            atributos[ruta] = {
                "page": ruta,
                "label": titulo,
                "icon": nav_icons.icon_html(slug),
                "slot": nav_icons.icon_slot(slug),
                "data-icon": slug,
            }
    return atributos


# ---------------------------------------------------------------------------
# Paleta visual
# ---------------------------------------------------------------------------
COLORS = {
    "graphite": "#171A1C",
    "carbon": "#252B2D",
    "lime": "#A3E635",
    "sage": "#D6E5B5",
    "bone": "#F1F3EF",
    "muted": "#8B9694",
    "line": "#3A4245",
}

TIPO_REGISTRO_SIMULADO = "SIMULADO"


def ensure_directories() -> None:
    """Crea las carpetas de trabajo si no existen (idempotente)."""
    for directory in (
        DATA_DIR,
        MODELS_DIR,
        IMAGES_DIR,
        PATHOLOGY_UPLOADS_DIR,
        REPORTS_DIR,
        STORAGE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def classify_pci(value: float) -> tuple[str, str, str]:
    """Devuelve (etiqueta, color_fondo, color_texto) para un valor de PCI.

    Es una función de *clasificación visual*: nunca se usa para calcular PCI.
    """
    if value is None:
        return "Sin dato", COLORS["muted"], COLORS["bone"]
    value = float(max(PCI_MIN, min(PCI_MAX, value)))
    for low, high, label, bg, fg in PCI_RANGES:
        if value >= low:
            return label, bg, fg
    return "Fallado", "#8E2F26", COLORS["bone"]


def pci_scale_records() -> list[dict]:
    """Leyenda del PCI lista para renderizar en tablas."""
    return [
        {"Rango": f"{low} - {high}", "Condición": label, "Color": bg}
        for low, high, label, bg, _fg in PCI_RANGES
    ]


def load_metadata() -> dict:
    """Lee model_metadata.json; devuelve {} si aún no se ha entrenado."""
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
