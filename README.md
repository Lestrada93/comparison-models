# Análisis Comparativo de Modelos Preentrenados de Hugging Face

> **Gestión de Proyectos de Inteligencia Artificial** — Actividad 4 | Módulo 2 / Semana 5  
> Universidad Tecmilenio

---

## 📋 Descripción del Proyecto

Este proyecto implementa y compara tres modelos preentrenados del ecosistema **Hugging Face** aplicados al problema de **análisis de sentimientos en reseñas de productos**, utilizando Google Colab con GPU como entorno de prototipado.

El caso de uso pertenece a la rama de **Procesamiento de Lenguaje Natural (NLP)** y consiste en clasificar automáticamente el sentimiento de una reseña textual como **positivo** o **negativo**.

### Modelos evaluados

| # | Modelo | Arquitectura | Parámetros |
|---|--------|--------------|------------|
| 1 | `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT | 66M |
| 2 | `textattack/roberta-base-SST-2` | RoBERTa-base | 125M |
| 3 | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Twitter-RoBERTa | 125M |

### Dataset

- **Nombre:** SST-2 (Stanford Sentiment Treebank v2)
- **Fuente:** `stanfordnlp/sst2` en Hugging Face Hub
- **Split utilizado:** Validación (submuestra de 200 ejemplos, `seed=42`)

---

## 📊 Resultados

| Modelo | Accuracy | Precision | Recall | F1-score | Latencia (ms) |
|--------|----------|-----------|--------|----------|---------------|
| DistilBERT-SST2 | 0.8850 | 0.8750 | 0.9159 | 0.8950 | 2.03 |
| **RoBERTa-SST2** | **0.9150** | **0.9245** | **0.9159** | **0.9202** | 4.44 |
| Twitter-RoBERTa | 0.8000 | 0.9467 | 0.6636 | 0.7802 | 3.87 |

🏆 **Modelo recomendado:** `textattack/roberta-base-SST-2` por mayor F1-score (0.9202)  
⚡ **Alternativa eficiente:** `distilbert-base-uncased-finetuned-sst-2-english` para entornos con restricción de latencia

---

## 🗂️ Estructura del Repositorio

```
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── notebooks/
│   └── actividad4_sentiment_analysis.ipynb   # Cuaderno principal de Colab
├── data/
│   ├── dataset_info.md                # Descripción y estructura del dataset
│   ├── prepare_dataset.py             # Script reproducible de preparación
│   ├── sst2_sample.csv                # Submuestra de 200 ejemplos (generada)
│   └── sst2_full.csv                  # Split de validación completo (generado)
├── results/
│   └── comparacion_modelos.png        # Gráfica comparativa de métricas
└── src/
    └── (scripts auxiliares)
```

---

## ⚙️ Entorno de Ejecución

| Componente | Versión |
|------------|---------|
| Python | 3.12 |
| transformers | 4.x |
| datasets | 3.x |
| evaluate | 0.4.x |
| scikit-learn | 1.x |
| torch | 2.x (CUDA) |
| pandas | 2.x |
| matplotlib | 3.x |
| GPU | NVIDIA T4 (Google Colab) |

---

## 🚀 Pasos de Ejecución

### Opción A — Google Colab (recomendado)

1. Abre el notebook en Google Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

2. Habilita la GPU:  
   `Runtime > Change runtime type > T4 GPU`

3. *(Opcional)* Agrega tu token de Hugging Face en Secrets:  
   Panel izquierdo 🔑 → **Add new secret** → Nombre: `HF_TOKEN`

4. Ejecuta todas las celdas:  
   `Runtime > Run all`

### Opción B — Ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Preparar el dataset
python data/prepare_dataset.py

# 4. Abrir el notebook
jupyter notebook notebooks/actividad4_sentiment_analysis.ipynb
```

---

## 📦 requirements.txt

```
transformers>=4.40.0
datasets>=2.19.0
evaluate>=0.4.0
scikit-learn>=1.4.0
torch>=2.2.0
pandas>=2.0.0
matplotlib>=3.8.0
accelerate>=0.29.0
huggingface_hub>=0.22.0
jupyter>=1.0.0
```

---

## 📝 Conclusiones

1. **RoBERTa-SST2** es el modelo más adecuado para producción en el dominio de reseñas de productos, con F1-score de **0.9202** y accuracy de **0.9150**.

2. **DistilBERT-SST2** representa la alternativa óptima cuando la latencia es crítica, siendo **54% más rápido** (2.03 ms/muestra) que RoBERTa a costa de ~2.5% de F1.

3. **Twitter-RoBERTa** muestra degradación de rendimiento (F1: 0.7802) en texto formal por desalineación de dominio, pero es la elección correcta para análisis de redes sociales.

4. La selección del modelo debe considerar no solo las métricas de benchmark, sino también el **alineamiento del dominio** de entrenamiento con el caso de uso y las **restricciones del entorno de despliegue**.

---

## 🔁 Reproducibilidad

Todos los resultados son reproducibles ejecutando el notebook con:
- GPU T4 habilitada en Google Colab
- Semilla aleatoria fija: `seed = 42`
- Tamaño de muestra: `200 ejemplos`

---

## 📚 Referencias

- Socher, R. et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. EMNLP 2013.
- Wolf, T. et al. (2020). *HuggingFace's Transformers: State-of-the-art Natural Language Processing*. EMNLP 2020.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT*. arXiv:1910.01108.
- Loureiro, D. et al. (2022). *TimeLMs: Diachronic Language Models from Twitter*. ACL 2022.
