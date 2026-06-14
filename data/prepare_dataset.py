"""
prepare_dataset.py
==================
Script reproducible para descarga y preparación del dataset SST-2.

Uso:
    python prepare_dataset.py

Requisitos:
    pip install datasets pandas

Salida:
    data/sst2_sample.csv   — 200 ejemplos usados en la evaluación
    data/sst2_full.csv     — split de validación completo (872 ejemplos)
"""

import random
import os
import pandas as pd
from datasets import load_dataset

# ── Configuración ──────────────────────────────────────────────────────────────
SEED        = 42
SAMPLE_SIZE = 200
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))  # misma carpeta /data

# ── 1. Descarga del dataset ────────────────────────────────────────────────────
print("📥 Descargando dataset SST-2 desde Hugging Face Hub...")
dataset   = load_dataset("stanfordnlp/sst2")
val_data  = dataset["validation"]

print(f"   Split de validación: {len(val_data)} ejemplos")
print(f"   Columnas: {val_data.column_names}")

# ── 2. Guardar split completo ──────────────────────────────────────────────────
full_path = os.path.join(OUTPUT_DIR, "sst2_full.csv")
df_full   = val_data.to_pandas()
df_full["sentiment"] = df_full["label"].map({0: "NEGATIVE", 1: "POSITIVE"})
df_full.to_csv(full_path, index=False, encoding="utf-8")
print(f"\n✅ Dataset completo guardado: {full_path}  ({len(df_full)} filas)")

# ── 3. Submuestra reproducible de 200 ejemplos ────────────────────────────────
random.seed(SEED)
indices = random.sample(range(len(val_data)), SAMPLE_SIZE)
sample  = val_data.select(indices)

df_sample = sample.to_pandas()
df_sample["sentiment"] = df_sample["label"].map({0: "NEGATIVE", 1: "POSITIVE"})

sample_path = os.path.join(OUTPUT_DIR, "sst2_sample.csv")
df_sample.to_csv(sample_path, index=False, encoding="utf-8")
print(f"✅ Submuestra guardada     : {sample_path}  ({len(df_sample)} filas)")

# ── 4. Resumen estadístico ─────────────────────────────────────────────────────
print("\n── Distribución de etiquetas (submuestra) ──────────────────────────")
print(df_sample["sentiment"].value_counts().to_string())

print("\n── Estadísticas de longitud de texto ───────────────────────────────")
df_sample["num_words"] = df_sample["sentence"].str.split().str.len()
print(df_sample["num_words"].describe().round(2).to_string())

print("\n── Ejemplos de la submuestra ────────────────────────────────────────")
for _, row in df_sample.head(5).iterrows():
    icon = "✅" if row["label"] == 1 else "❌"
    print(f"  {icon} [{row['sentiment']}] {row['sentence'][:80]}...")

print("\n✅ Preparación completada. Archivos listos en:", OUTPUT_DIR)
print("   • sst2_full.csv    — validación completa (872 ejemplos)")
print("   • sst2_sample.csv  — submuestra evaluación (200 ejemplos, seed=42)")
