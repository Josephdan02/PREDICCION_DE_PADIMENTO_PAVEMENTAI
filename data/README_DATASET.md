# Dataset de inspecciones de pavimento rígido

## Advertencia principal

**Todos los registros de `pavimentos.csv` son SINTÉTICOS y están marcados con
`tipo_registro = "SIMULADO"`.**

- No provienen de inspecciones reales realizadas en la provincia de Huánuco.
- No constituyen una evaluación oficial del PCI (ASTM D6433).
- Su única finalidad es académica: permitir el entrenamiento, la evaluación y la
  demostración del modelo de regresión lineal del proyecto PAVEMENT AI.

Cualquier uso con decisiones reales de conservación vial requiere una campaña de
inspección técnica documentada y una evaluación oficial del PCI.

---

## Archivo

| Elemento | Valor |
|---|---|
| Ruta | `data/pavimentos.csv` |
| Registros | 98 |
| Tramos únicos | 55 |
| Inspecciones por tramo | 1 a 3 |
| Rango de antigüedad | 0.6 – 25.9 años |
| Rango de PCI observado | 26.4 – 100.0 |
| Codificación | UTF-8 |
| Generador | `src/dataset_generator.py` (semilla 42) |

Regeneración:

```powershell
python -m src.dataset_generator
```

La generación es **reproducible**: con la misma semilla se obtiene exactamente el
mismo archivo.

---

## Diccionario de variables

| Columna | Tipo | Descripción | Rango / valores |
|---|---|---|---|
| `id_registro` | texto | Identificador de la inspección | `REG-0001` … `REG-0098` |
| `codigo_tramo` | texto | Código del tramo (agrupa inspecciones sucesivas) | p. ej. `HUAN-014` |
| `distrito` | texto | Distrito donde se ubica el tramo | Huánuco, Amarilis, Pillco Marca, Santa María del Valle |
| `tipo_pavimento` | texto | Tipo de superficie | Rígido |
| `fecha_inspeccion` | fecha | Fecha de la inspección (AAAA-MM-DD) | 2021-10-19 – 2025-11-30 |
| `antiguedad_anios` | numérico | Edad del pavimento al momento de la inspección | 0 – 60 |
| `volumen_transito` | entero | Tránsito promedio diario estimado | ≥ 0 veh/día |
| `tipo_patologia` | texto | Patología dominante del tramo | ver catálogo |
| `severidad` | texto | Severidad de la patología | Baja, Media, Alta |
| `cantidad_fallas` | entero | Número de fallas o extensión afectada | ≥ 1 |
| `pci_observado` | numérico | PCI registrado en la inspección | 0 – 100 |
| `tipo_registro` | texto | Origen del dato | SIMULADO |

### Catálogo de patologías

1. Grietas longitudinales
2. Grietas transversales
3. Grietas de esquina
4. Losa dividida
5. Despostillamiento
6. Escalonamiento

### Pesos de severidad usados por el generador

| Severidad | Peso |
|---|---|
| Baja | 1.0 |
| Media | 2.0 |
| Alta | 3.5 |

---

## Modelo generativo

```
PCI = 100 − tasa_anual × antigüedad − peso_severidad × ln(1 + cantidad_fallas) × k
      + efecto_tramo + ruido
```

- `tasa_anual` crece con el volumen de tránsito: `≈ (1.55 + 1.35 × clip(tránsito/9000, 0.25, 1.6))`, multiplicado por un factor aleatorio uniforme en [0.82, 1.18] que representa diferencias de calidad constructiva y drenaje entre tramos.
- `efecto_tramo` ~ N(0, 3.5): condición particular de cada tramo, constante entre sus inspecciones.
- `ruido` ~ N(0, 4.2): garantiza que la relación PCI–antigüedad **no sea determinista**.
- El resultado se recorta a [0, 100].

Esto produce una correlación negativa fuerte (≈ −0.87) entre antigüedad y PCI, con
dispersión realista, que es precisamente lo que permite que una regresión lineal
simple alcance un ajuste razonable sin ser perfecto.

---

## Limitaciones

1. Los datos no describen ninguna vía real; la distribución por distrito es aleatoria.
2. La patología y la severidad son **atributos del tramo**, no se miden densidades por unidad de muestreo.
3. La relación PCI–antigüedad es aproximadamente lineal **por construcción**; en campo la degradación es no lineal e influida por muchos factores.
4. El PCI es un valor sintético: no fue deducido mediante el procedimiento oficial de deduct values.
5. Las fechas son aproximadas y se generan hacia atrás desde una fecha de referencia.

---

## Uso en el proyecto

- `src/preprocessing.py` valida tipos, nulos, duplicados y rangos antes de entrenar.
- `src/train_model.py` entrena con `X = antiguedad_anios` e `y = pci_observado`, con división agrupada por `codigo_tramo` (80/20, semilla 42).
- La página **Dataset** permite filtrar por tramo, distrito, patología y severidad, consultar estadísticas y descargar el CSV.
