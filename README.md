# PAVEMENT AI

**Sistema académico de gestión de pavimentos rígidos y estimación del Índice de Condición del Pavimento (PCI) mediante regresión lineal.**

Proyecto desarrollado para la actividad de **Inteligencia Artificial — SENATI**.
Contexto: municipalidad de la provincia de Huánuco.

---

## 1. Descripción

PAVEMENT AI es una aplicación web que permite a un área de obras públicas:

- **Registrar tramos** de pavimento rígido de concreto con sus datos base de inspección.
- **Documentar las patologías** encontradas (grietas longitudinales, transversales, de esquina, losa dividida, despostillamiento y escalonamiento) con severidad, extensión, descripción y **fotografía opcional**.
- **Entrenar un modelo real de regresión lineal** (`sklearn.linear_model.LinearRegression`) que relaciona la **antigüedad** del tramo con el **PCI observado**.
- **Simular la evolución del PCI** a corto (1 año), mediano (3 años) y largo plazo (5 años) usando el modelo entrenado.
- **Generar informes PDF**, exportaciones CSV/JSON y consultar el dataset con filtros y estadísticas.

La interfaz sigue el concepto **Bento Grid**: tarjetas de distintos tamaños sobre una retícula de 12 columnas, con una identidad visual propia (grafito + verde lima).

---

## 2. Objetivo

Demostrar, en un caso de infraestructura vial, el flujo completo de un proyecto de aprendizaje automático:

1. Datos → 2. limpieza y validación → 3. entrenamiento → 4. evaluación → 5. persistencia → 6. uso del modelo en una aplicación → 7. generación de reportes → 8. pruebas automatizadas.

El objetivo **no** es implementar la metodología oficial PCI (ASTM D6433), sino **estimar** el PCI a partir de un único predictor (antigüedad) y explicar sus limitaciones.

---

## 3. Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ |
| Datos | Pandas, NumPy |
| Machine Learning | scikit-learn (`LinearRegression`, `GroupShuffleSplit`) |
| Persistencia del modelo | Joblib |
| Visualización | Plotly (interactivo), Matplotlib (informe PDF) |
| Interfaz web | Streamlit |
| Reportes | ReportLab |
| Pruebas | Pytest (+ `streamlit.testing.AppTest`) |

---

## 4. Estructura del proyecto

```
PAVEMENT_AI/
├── app.py                        # Punto de entrada (navegación)
├── requirements.txt
├── README.md
├── .gitignore
├── pytest.ini
├── .streamlit/
│   └── config.toml               # Tema y configuración del servidor
├── data/
│   ├── pavimentos.csv            # Dataset (98 registros SIMULADOS)
│   └── README_DATASET.md         # Diccionario y procedencia de los datos
├── models/
│   ├── linear_regression.joblib  # Modelo entrenado
│   └── model_metadata.json       # Métricas y metadatos del entrenamiento
├── src/
│   ├── config.py                 # Rutas, constantes, paleta y rangos PCI
│   ├── dataset_generator.py      # Generador del dataset sintético
│   ├── data_loader.py            # Acceso a CSV (dataset, vías, patologías)
│   ├── preprocessing.py          # Validación, limpieza y preparación de X/y
│   ├── train_model.py            # Entrenamiento, división agrupada, persistencia
│   ├── evaluation.py             # R², MAE, RMSE e interpretación
│   ├── prediction.py             # Carga del modelo y simulaciones
│   ├── road_service.py           # Lógica y validación de vías
│   ├── pathology_service.py      # Lógica, validación y fotos de patologías
│   ├── charts.py                 # Gráficos Plotly y Matplotlib
│   ├── report_generator.py       # Informe PDF y exportaciones CSV/JSON
│   └── ui.py                     # CSS, componentes Bento y utilidades de UI
├── pages/
│   ├── dashboard.py
│   ├── registro_vias.py
│   ├── patologias.py
│   ├── dataset.py
│   ├── modelo_ml.py
│   ├── simulador_pci.py
│   ├── reportes.py
│   └── acerca.py
├── assets/
│   └── styles.css                # Identidad visual (Bento Grid)
├── storage/                      # vias.csv y patologias.csv (persistencia local)
├── uploads/patologias/           # Fotografías subidas
├── reports/                      # PDFs generados
└── tests/
    ├── conftest.py
    ├── test_preprocessing.py
    ├── test_model.py
    ├── test_prediction.py
    ├── test_records.py
    └── test_pages.py
```

**Separación de responsabilidades:** las páginas solo componen la interfaz; toda la lógica (validación, persistencia, entrenamiento, predicción, reportes) vive en `src/`. Así las pruebas pueden verificar el comportamiento sin depender de Streamlit.

---

## 5. Instalación

```powershell
# 1. Crear y activar un entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt
```

> Requiere Python 3.11 o superior.

---

## 6. Ejecución

```powershell
streamlit run app.py
```

La aplicación queda disponible en `http://localhost:8501`.

Si `data/pavimentos.csv` o `models/linear_regression.joblib` no existieran, se regeneran con:

```powershell
python -m src.dataset_generator   # crea el dataset sintético
python -m src.train_model         # entrena y guarda el modelo
```

---

## 7. Entrenamiento del modelo

```powershell
python -m src.train_model
```

También puede hacerse desde la página **Modelo ML → Entrenar / reentrenar modelo**.

### Configuración

| Aspecto | Valor |
|---|---|
| Variable independiente (`X`) | `antiguedad_anios` |
| Variable dependiente (`y`) | `pci_observado` |
| Algoritmo | `sklearn.linear_model.LinearRegression` |
| División | 80/20 con `GroupShuffleSplit` agrupado por `codigo_tramo` |
| Semilla | 42 (reproducible) |
| Registros limpios | 98 (55 tramos) |
| Entrenamiento | 75 registros / 44 tramos |
| Prueba | 23 registros / 11 tramos |

La **división agrupada** es clave: como un mismo tramo tiene varias inspecciones en el tiempo, una división aleatoria pondría registros casi idénticos en entrenamiento y prueba, inflando artificialmente las métricas. Con `GroupShuffleSplit` ningún tramo aparece en ambos conjuntos.

### Métricas reales obtenidas (conjunto de prueba)

| Métrica | Valor |
|---|---|
| **R²** | **0.7541** |
| **MAE** | **8.03 puntos PCI** |
| **RMSE** | **9.02 puntos PCI** |

Métricas de referencia en entrenamiento: R² = 0.7625 · MAE = 6.85 · RMSE = 8.44.

### Ecuación ajustada

```
PCI = 93.72 − 2.625 × antigüedad
```

- **Intercepto (93.72):** PCI estimado para 0 años según el modelo. Es una referencia matemática, no el PCI de una vía nueva real.
- **Pendiente (−2.625):** por cada año adicional de antigüedad el PCI estimado baja en promedio **2.63 puntos**. La pendiente negativa confirma la degradación esperada.
- **R² (0.754):** la antigüedad explica ~75 % de la variabilidad del PCI observado. El resto depende de tránsito, drenaje, calidad constructiva y tipo de patología, que el modelo no incorpora.

> Las métricas anteriores son las realmente calculadas por `train_model.py` y guardadas en `models/model_metadata.json`. No se garantiza ninguna precisión futura sobre datos reales.

---

## 8. Funcionamiento del simulador

Página **Simulador PCI**:

1. Se elige un **tramo registrado** (toma su antigüedad de la última inspección) o un **escenario nuevo** (antigüedad manual).
2. Se ingresa la **antigüedad actual** y se elige el **horizonte**: 1, 3 o 5 años (también hasta 10 años).
3. El sistema ejecuta `LinearRegression.predict` dos veces (antigüedad actual y futura) con el modelo cargado desde Joblib.
4. Se muestran: PCI actual, PCI futuro, variación (puntos), condición visual, ecuación usada y el gráfico de la recta con el punto consultado.
5. Se presentan tres tarjetas de escenario: corto, mediano y largo plazo.

Reglas implementadas:

- **Todas las predicciones provienen del modelo**; no hay condicionales que fabriquen valores.
- Los condicionales se usan únicamente para **validar entradas, generar advertencias y clasificar** la condición del pavimento.
- Si la antigüedad consultada está **fuera del rango observado en entrenamiento (0.6 – 25.9 años)** se muestra una advertencia de extrapolación.
- El PCI mostrado se **recorta a 0–100** para la escala visual, pero el **valor matemático original** se conserva y se muestra explícitamente.

---

## 9. Interpretación del PCI (referencia visual)

| Rango PCI | Condición |
|---|---|
| 85 – 100 | Excelente |
| 70 – 84 | Muy bueno |
| 55 – 69 | Bueno |
| 40 – 54 | Regular |
| 25 – 39 | Malo |
| 10 – 24 | Muy malo |
| 0 – 9 | Fallado |

Estos rangos se utilizan como **referencia visual** dentro de la aplicación (colores, etiquetas, conteos). **No sustituyen** una evaluación técnica del pavimento ni la metodología oficial PCI (ASTM D6433), que exige medir densidad y severidad de cada patología en unidades de muestreo.

Un tramo se considera **crítico** en el dashboard cuando su PCI < 40.

---

## 10. Procedencia del dataset

**El dataset es sintético.** `data/pavimentos.csv` se genera con
`src/dataset_generator.py` y **todos los registros están marcados con
`tipo_registro = "SIMULADO"`**.

- **No** proviene de inspecciones reales verificadas en Huánuco.
- **No** debe presentarse como datos oficiales de la municipalidad.
- Su objetivo es exclusivamente académico: disponer de un conjunto coherente, con variabilidad, para entrenar y evaluar el modelo.

Modelo generativo simplificado:

```
PCI = 100 − tasa_año × antigüedad − impacto_patología + efecto_tramo + ruido
```

donde `tasa_año` crece con el volumen de tránsito y el ruido (`σ ≈ 4.2`) evita que la relación PCI–antigüedad sea determinista. El detalle completo está en [`data/README_DATASET.md`](data/README_DATASET.md).

---

## 11. Persistencia

Todo es local y transparente:

| Dato | Archivo |
|---|---|
| Dataset de inspecciones | `data/pavimentos.csv` |
| Vías registradas | `storage/vias.csv` |
| Patologías registradas | `storage/patologias.csv` |
| Fotografías | `uploads/patologias/<CODIGO_TRAMO>/` |
| Modelo | `models/linear_regression.joblib` |
| Metadatos del modelo | `models/model_metadata.json` |
| Informes PDF | `reports/` |

Las escrituras usan archivos temporales + reemplazo atómico para evitar corrupción. La información persiste al reiniciar Streamlit.

---

## 12. Reportes

La página **Reportes** genera un **PDF** con: indicadores generales, métricas reales del modelo y ecuación, gráfico antigüedad ↔ PCI, resultados del escenario de predicción, patologías registradas, historial de inspecciones y limitaciones. Incluye un **aviso visible** sobre la naturaleza simulada de los datos.

También se exportan: dataset completo (CSV), patologías (CSV), vías (CSV) y un resumen del sistema (JSON).

---

## 13. Pruebas

```powershell
python -m pytest -q
```

Estado actual: **102 pruebas aprobadas**.

| Archivo | Cobertura |
|---|---|
| `test_preprocessing.py` | Generación/carga del dataset, columnas obligatorias, ≥ 50 registros, cobertura de patologías y severidades, variabilidad, validación de rangos, tratamiento de nulos, duplicados, limpieza y preparación de X/y |
| `test_model.py` | Entrenamiento real, `LinearRegression`, métricas verificadas contra scikit-learn, división agrupada sin fuga de tramos, persistencia con Joblib, recarga idéntica, ecuación y pendiente |
| `test_prediction.py` | Predicción igual a `model.predict`, ausencia de valores fijos, rechazo de entradas inválidas, estructura del escenario, advertencias de extrapolación, límites 0–100 y clasificación del PCI |
| `test_records.py` | Alta/edición/eliminación y validaciones de vías, alta/eliminación de patologías, guardado y rechazo de fotografías, estadísticas, filtros, generación de PDF (con y sin modelo) y exportaciones CSV/JSON |
| `test_pages.py` | Las 8 páginas de Streamlit se ejecutan sin excepciones (`AppTest`) y renderizan contenido |

Las pruebas usan directorios temporales: **nunca** sobrescriben los datos reales del proyecto.

---

## 14. Limitaciones

- Los datos son **sintéticos**; el desempeño del modelo no es extrapolable a vías reales.
- La regresión **solo usa la antigüedad**: no incorpora tránsito, drenaje, espesor de losa, tipo o severidad de patologías.
- El modelo es **lineal por diseño de los datos**; la degradación real de un pavimento es no lineal.
- El registro de patologías **no calcula** un PCI oficial: documenta el deterioro.
- Las predicciones fuera del rango 0.6–25.9 años son **extrapolaciones** y deben interpretarse con cautela.
- No incluye autenticación, base de datos remota, mapas ni APIs externas (fuera del alcance académico).

---

## 15. Autoría y créditos

Actividad académica de Inteligencia Artificial · SENATI.
Datos del dataset: generados para fines demostrativos. Referencias metodológicas: FHWA (distresses en pavimentos rígidos) y ASTM D6433 (condición del pavimento).
