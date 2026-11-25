"""Generador del dataset sintético de inspecciones de pavimento rígido.

ADVERTENCIA IMPORTANTE
======================
Los registros generados aquí son SINTÉTICOS. No provienen de inspecciones
reales realizadas en Huánuco ni de una evaluación oficial ASTM D6433.
Todos los registros se marcan con ``tipo_registro = "SIMULADO"``.

El objetivo es disponer de un conjunto de datos coherente, con variabilidad
realista, para entrenar y demostrar un modelo de regresión lineal en el
contexto académico de SENATI.

Modelo generativo (simplificado):
    pci = 100 - tasa_anual * antiguedad - impacto_patologia + ruido

donde ``tasa_anual`` depende del volumen de tránsito del tramo y el ruido
aporta variabilidad para que la relación NO sea determinista.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

PATOLOGIAS_LISTA = list(config.PATOLOGIAS.keys())


def _slug(texto: str) -> str:
    """Convierte un texto en un identificador simple sin acentos ni espacios."""
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n", "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ñ": "N"}
    limpio = "".join(reemplazos.get(ch, ch) for ch in texto)
    return "".join(ch for ch in limpio if ch.isalnum()).upper()


def _tasa_degradacion(volumen_transito: int, rng: np.random.Generator) -> float:
    """Tasa de pérdida de PCI por año, mayor con más tránsito y con variabilidad."""
    factor_transito = np.clip(volumen_transito / 9000.0, 0.25, 1.6)
    base = 1.55 + 1.35 * factor_transito
    # Variabilidad estructural entre tramos (calidad de construcción, drenaje, etc.)
    return float(base * rng.uniform(0.82, 1.18))


def generate_dataset(n_tramos: int = 55, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Genera el dataset sintético de inspecciones.

    Cada tramo tiene entre 1 y 3 inspecciones a lo largo del tiempo, de modo que
    existen varios registros por ``codigo_tramo`` (lo que permite la división
    agrupada entrenamiento/prueba).
    """
    rng = np.random.default_rng(seed)
    registros: list[dict] = []
    id_registro = 1
    hoy = pd.Timestamp("2025-11-30")

    for i in range(1, n_tramos + 1):
        distrito = str(rng.choice(config.DISTRITOS))
        codigo_tramo = f"{_slug(distrito)[:4]}-{i:03d}"

        volumen_transito = int(np.clip(rng.normal(6500, 3200), 600, 18000) // 50 * 50)
        antiguedad_actual = float(np.clip(rng.normal(13, 7), 0.5, 32))
        tasa = _tasa_degradacion(volumen_transito, rng)
        desviacion_tramo = float(rng.normal(0, 3.5))  # condición particular del tramo

        # Patología dominante del tramo (se repite entre inspecciones del mismo tramo)
        patologia_tramo = str(rng.choice(PATOLOGIAS_LISTA))
        severidad_tramo = str(rng.choice(config.SEVERIDADES, p=[0.35, 0.40, 0.25]))
        peso_sev = config.PESO_SEVERIDAD[severidad_tramo]

        n_inspecciones = int(rng.choice([1, 2, 3], p=[0.35, 0.40, 0.25]))
        # Desfase temporal hacia atrás (años) para cada inspección previa: se
        # recortan para no retroceder más allá del año 0 de antigüedad.
        desfases = [d for d in rng.integers(2, 5, size=n_inspecciones - 1).tolist() if d < antiguedad_actual]
        desfases = sorted(set(desfases), reverse=True)

        for desfase in desfases + [0]:
            antiguedad = round(max(0.3, antiguedad_actual - desfase), 1)
            fecha = hoy - pd.Timedelta(days=int(round(desfase * 365 + rng.integers(0, 45))))

            cantidad_fallas = int(max(1, rng.poisson(lam=4 + antiguedad * 0.9)))
            impacto_patologia = peso_sev * np.log1p(cantidad_fallas) * rng.uniform(0.9, 1.5)

            pci = 100.0 - tasa * antiguedad - impacto_patologia + desviacion_tramo
            pci += float(rng.normal(0, 4.2))  # ruido: la relación no es determinista
            pci = float(np.clip(round(pci, 1), config.PCI_MIN, config.PCI_MAX))

            registros.append(
                {
                    "id_registro": f"REG-{id_registro:04d}",
                    "codigo_tramo": codigo_tramo,
                    "distrito": distrito,
                    "tipo_pavimento": "Rígido",
                    "fecha_inspeccion": fecha.strftime("%Y-%m-%d"),
                    "antiguedad_anios": antiguedad,
                    "volumen_transito": volumen_transito,
                    "tipo_patologia": patologia_tramo,
                    "severidad": severidad_tramo,
                    "cantidad_fallas": cantidad_fallas,
                    "pci_observado": pci,
                    "tipo_registro": config.TIPO_REGISTRO_SIMULADO,
                }
            )
            id_registro += 1

    df = pd.DataFrame(registros, columns=config.DATASET_COLUMNS)
    return df.sort_values(["codigo_tramo", "fecha_inspeccion"]).reset_index(drop=True)


def write_dataset(path=None, n_tramos: int = 55, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Genera el dataset y lo escribe en disco (sobrescribe)."""
    path = path or config.DATASET_PATH
    df = generate_dataset(n_tramos=n_tramos, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":  # pragma: no cover
    data = write_dataset()
    print(f"Dataset sintético generado: {len(data)} registros en {config.DATASET_PATH}")
