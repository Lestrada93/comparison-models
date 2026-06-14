# Dataset: SST-2 (Stanford Sentiment Treebank v2)

## Origen

| Campo | Detalle |
|-------|---------|
| **Nombre completo** | Stanford Sentiment Treebank v2 |
| **Autores** | Richard Socher et al., Stanford NLP Group |
| **Publicación** | "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank" (EMNLP 2013) |
| **Fuente en Hugging Face** | `stanfordnlp/sst2` |
| **Licencia** | Mixed (uso académico y de investigación) |
| **Idioma** | Inglés |
| **Dominio** | Críticas y reseñas de películas |

El dataset SST-2 es un benchmark estándar para clasificación binaria de sentimientos en NLP.
Forma parte del benchmark GLUE (General Language Understanding Evaluation) y es ampliamente
utilizado para evaluar modelos de lenguaje en tareas de clasificación de texto.

---

## Splits disponibles

| Split | Ejemplos | Uso en esta actividad |
|-------|----------|-----------------------|
| `train` | 67,349 | No utilizado |
| `validation` | 872 | ✅ Evaluación de modelos |
| `test` | 1,821 | No utilizado (etiquetas ocultas) |

En esta actividad se utilizó una **submuestra aleatoria de 200 ejemplos** del split de
validación, seleccionados con semilla fija (`random.seed(42)`) para garantizar reproducibilidad.

---

## Estructura de campos

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `idx` | `int32` | Índice único del ejemplo en el dataset | `0`, `1`, `2`, ... |
| `sentence` | `string` | Texto de la reseña o fragmento de crítica cinematográfica | `"it 's a charming and often affecting journey ."` |
| `label` | `int32` | Etiqueta de sentimiento binaria: `0` = negativo, `1` = positivo | `0` o `1` |

### Notas sobre los datos
- Los textos ya están **tokenizados con espacios** (pre-tokenización del Penn Treebank).
- Las oraciones son fragmentos cortos extraídos de reseñas completas de Rotten Tomatoes.
- La longitud promedio es de ~20 tokens por oración.
- El dataset está **balanceado**: ~50% positivos y ~50% negativos en el split de validación.

---

## Distribución de etiquetas (split de validación)

| Etiqueta | Valor | Cantidad | Porcentaje |
|----------|-------|----------|------------|
| Negativo | 0 | 428 | 49.1% |
| Positivo | 1 | 444 | 50.9% |

---

## Ejemplos representativos

| Sentimiento | Texto |
|-------------|-------|
| ✅ POSITIVO | `"it 's a charming and often affecting journey ."` |
| ✅ POSITIVO | `"allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker ."` |
| ❌ NEGATIVO | `"unflinchingly bleak and desperate ."` |
| ❌ NEGATIVO | `"the film is a ponderous and ultimately empty exercise in imagery ."` |

---

## Cómo se obtuvo en esta actividad

```python
from datasets import load_dataset
import random

# Carga desde Hugging Face Hub
dataset = load_dataset('stanfordnlp/sst2')

# Split de validación
test_data = dataset['validation']  # 872 ejemplos

# Submuestra reproducible de 200 ejemplos
random.seed(42)
indices = random.sample(range(len(test_data)), 200)
sample  = test_data.select(indices)

texts  = sample['sentence']
labels = sample['label']
```

Para reproducir exactamente los mismos 200 ejemplos, ejecuta `prepare_dataset.py`
o la celda correspondiente del cuaderno Colab.
