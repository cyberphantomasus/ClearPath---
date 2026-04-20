"""
ClearPath — AI Training Pipeline
=================================
Google Colab ready. Run this entire file to:
  1. Generate 800-row synthetic labeled dataset (200 per class)
  2. Engineer all 12 behavioral signals
  3. Train 3 Random Forest classifiers
  4. Evaluate on held-out test set
  5. Export all 3 models to TFLite format

Usage:
  python train.py
  
  Or paste into Google Colab and run all cells.
  GPU is NOT required — Random Forest runs on CPU fine.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score
)
import json
import os

# TFLite conversion — requires tensorflow
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
    print("TensorFlow available — will export TFLite models")
except ImportError:
    TFLITE_AVAILABLE = False
    print("TensorFlow not found — skipping TFLite export (install: pip install tensorflow)")

# ─── Config ────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_PER_CLASS = 200       # 200 typical + 200 per condition = 800 total
N_ESTIMATORS = 200
MAX_DEPTH = 8
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "mean_response_time",        # ms — higher = slower
    "response_time_variance",    # CoV — higher = more variable
    "error_rate",                # 0..1 — higher = more errors
    "error_pattern_score",       # 0..1 — 1=systematic errors (dyslexia)
    "retry_rate",                # 0..1 — retries after wrong answer
    "attention_drift_index",     # negative = drifting attention (ADHD)
    "impulsivity_score",         # 0..1 — false taps / total taps
    "recovery_speed",            # 1=normal, >1=slower after distractor
    "emotion_accuracy",          # 0..1 — emotion recognition (autism)
    "social_hesitation_time",    # ms — pause before social answer
    "sequence_memory_score",     # 0..1 — gesture sequence memory
    "engagement_decay_rate",     # positive = engagement dropping
]

# ─── Synthetic Data Generation ─────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)

def generate_typical_child(n: int) -> pd.DataFrame:
    """Typical developing child — all signals in normal ranges."""
    return pd.DataFrame({
        "mean_response_time":       rng.normal(900, 180, n).clip(400, 2500),
        "response_time_variance":   rng.normal(0.20, 0.06, n).clip(0, 0.6),
        "error_rate":               rng.normal(0.12, 0.05, n).clip(0, 0.4),
        "error_pattern_score":      rng.normal(0.10, 0.06, n).clip(0, 0.5),
        "retry_rate":               rng.normal(0.45, 0.12, n).clip(0, 1),
        "attention_drift_index":    rng.normal(-0.02, 0.08, n).clip(-0.4, 0.3),
        "impulsivity_score":        rng.normal(0.08, 0.04, n).clip(0, 0.3),
        "recovery_speed":           rng.normal(1.05, 0.12, n).clip(0.7, 1.8),
        "emotion_accuracy":         rng.normal(0.78, 0.09, n).clip(0.3, 1.0),
        "social_hesitation_time":   rng.normal(950, 200, n).clip(400, 2500),
        "sequence_memory_score":    rng.normal(0.72, 0.12, n).clip(0.2, 1.0),
        "engagement_decay_rate":    rng.normal(0.08, 0.05, n).clip(-0.2, 0.4),
        # Labels — all negative
        "dyslexia_label": np.zeros(n, dtype=int),
        "adhd_label":     np.zeros(n, dtype=int),
        "autism_label":   np.zeros(n, dtype=int),
    })


def generate_dyslexia_child(n: int) -> pd.DataFrame:
    """
    Dyslexia markers:
    - Very high error pattern score (systematic, not random errors)
    - High error rate — especially on visually similar letters
    - Slower response time (reading processing difficulty)
    - Low retry rate (avoidance behavior)
    """
    return pd.DataFrame({
        "mean_response_time":       rng.normal(1600, 300, n).clip(700, 4000),
        "response_time_variance":   rng.normal(0.25, 0.07, n).clip(0, 0.7),
        "error_rate":               rng.normal(0.42, 0.10, n).clip(0.15, 0.85),
        "error_pattern_score":      rng.normal(0.68, 0.12, n).clip(0.35, 1.0),  # KEY marker
        "retry_rate":               rng.normal(0.22, 0.09, n).clip(0, 0.5),     # avoidance
        "attention_drift_index":    rng.normal(-0.04, 0.09, n).clip(-0.5, 0.3),
        "impulsivity_score":        rng.normal(0.10, 0.05, n).clip(0, 0.35),
        "recovery_speed":           rng.normal(1.08, 0.15, n).clip(0.7, 2.0),
        "emotion_accuracy":         rng.normal(0.75, 0.10, n).clip(0.3, 1.0),   # mostly normal
        "social_hesitation_time":   rng.normal(980, 210, n).clip(400, 2500),
        "sequence_memory_score":    rng.normal(0.68, 0.13, n).clip(0.2, 1.0),
        "engagement_decay_rate":    rng.normal(0.18, 0.07, n).clip(-0.1, 0.5),  # tires from reading
        "dyslexia_label": np.ones(n, dtype=int),
        "adhd_label":     np.zeros(n, dtype=int),
        "autism_label":   np.zeros(n, dtype=int),
    })


def generate_adhd_child(n: int) -> pd.DataFrame:
    """
    ADHD markers:
    - High impulsivity score (acts before thinking)
    - Negative attention drift (worse over time)
    - High response time variance (inconsistent)
    - High engagement decay rate (boredom/fatigue)
    - Low retry rate (gives up quickly after errors)
    """
    return pd.DataFrame({
        "mean_response_time":       rng.normal(750, 200, n).clip(300, 2000),    # often fast but wrong
        "response_time_variance":   rng.normal(0.52, 0.11, n).clip(0.25, 0.9), # KEY — very inconsistent
        "error_rate":               rng.normal(0.28, 0.09, n).clip(0.05, 0.6),
        "error_pattern_score":      rng.normal(0.14, 0.07, n).clip(0, 0.5),    # random, not systematic
        "retry_rate":               rng.normal(0.20, 0.09, n).clip(0, 0.5),    # gives up
        "attention_drift_index":    rng.normal(-0.24, 0.10, n).clip(-0.7, 0.1),# KEY — drifts badly
        "impulsivity_score":        rng.normal(0.38, 0.10, n).clip(0.15, 0.75),# KEY — high false taps
        "recovery_speed":           rng.normal(1.35, 0.20, n).clip(0.8, 2.5),  # slow after distractors
        "emotion_accuracy":         rng.normal(0.70, 0.12, n).clip(0.3, 1.0),
        "social_hesitation_time":   rng.normal(880, 220, n).clip(300, 2500),
        "sequence_memory_score":    rng.normal(0.55, 0.15, n).clip(0.1, 0.9),  # working memory affected
        "engagement_decay_rate":    rng.normal(0.38, 0.10, n).clip(0.1, 0.8),  # KEY — engagement drops fast
        "dyslexia_label": np.zeros(n, dtype=int),
        "adhd_label":     np.ones(n, dtype=int),
        "autism_label":   np.zeros(n, dtype=int),
    })


def generate_autism_child(n: int) -> pd.DataFrame:
    """
    Autism Spectrum markers:
    - Low emotion recognition accuracy
    - Very high social hesitation time
    - Low sequence memory score (social gesture sequences)
    - Attention drift tends positive (hyper-focus on mechanical tasks)
    - Impulsivity tends lower (more rigid, methodical)
    """
    return pd.DataFrame({
        "mean_response_time":       rng.normal(1100, 250, n).clip(400, 3000),
        "response_time_variance":   rng.normal(0.22, 0.07, n).clip(0, 0.6),
        "error_rate":               rng.normal(0.20, 0.08, n).clip(0, 0.5),
        "error_pattern_score":      rng.normal(0.12, 0.06, n).clip(0, 0.4),
        "retry_rate":               rng.normal(0.55, 0.13, n).clip(0.1, 0.9),
        "attention_drift_index":    rng.normal(0.05, 0.08, n).clip(-0.3, 0.4), # hyper-focus tendency
        "impulsivity_score":        rng.normal(0.06, 0.03, n).clip(0, 0.2),    # more careful/rigid
        "recovery_speed":           rng.normal(1.10, 0.14, n).clip(0.7, 2.0),
        "emotion_accuracy":         rng.normal(0.38, 0.12, n).clip(0.05, 0.75),# KEY — poor emotion recognition
        "social_hesitation_time":   rng.normal(2100, 400, n).clip(800, 5000),  # KEY — very slow social response
        "sequence_memory_score":    rng.normal(0.35, 0.12, n).clip(0.05, 0.7), # KEY — social gesture memory poor
        "engagement_decay_rate":    rng.normal(0.05, 0.06, n).clip(-0.2, 0.3), # may not decay (restricted interest)
        "dyslexia_label": np.zeros(n, dtype=int),
        "adhd_label":     np.zeros(n, dtype=int),
        "autism_label":   np.ones(n, dtype=int),
    })


# ─── Build Dataset ─────────────────────────────────────────────────────────────

print("=" * 60)
print("ClearPath — Model Training Pipeline")
print("=" * 60)
print(f"\n[1/5] Generating synthetic dataset ({N_PER_CLASS * 4} children)...")

df = pd.concat([
    generate_typical_child(N_PER_CLASS),
    generate_dyslexia_child(N_PER_CLASS),
    generate_adhd_child(N_PER_CLASS),
    generate_autism_child(N_PER_CLASS),
], ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"  Dataset shape: {df.shape}")
print(f"  Label distribution:")
print(f"    Dyslexia positive: {df['dyslexia_label'].sum()} / {len(df)}")
print(f"    ADHD positive:     {df['adhd_label'].sum()} / {len(df)}")
print(f"    Autism positive:   {df['autism_label'].sum()} / {len(df)}")

# Save dataset
df.to_csv(f"{OUTPUT_DIR}/training_data.csv", index=False)
print(f"  Saved to {OUTPUT_DIR}/training_data.csv")


# ─── Feature Stats ──────────────────────────────────────────────────────────────

print("\n[2/5] Feature statistics:")
print(df[FEATURE_NAMES].describe().round(3).to_string())


# ─── Train / Test Split ────────────────────────────────────────────────────────

X = df[FEATURE_NAMES].values
y_dyslexia = df["dyslexia_label"].values
y_adhd = df["adhd_label"].values
y_autism = df["autism_label"].values

# 80/20 split stratified per target
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X, y_dyslexia, test_size=0.2, random_state=RANDOM_SEED, stratify=y_dyslexia)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X, y_adhd, test_size=0.2, random_state=RANDOM_SEED, stratify=y_adhd)
X_train_au, X_test_au, y_train_au, y_test_au = train_test_split(
    X, y_autism, test_size=0.2, random_state=RANDOM_SEED, stratify=y_autism)

print(f"\n  Train: {len(X_train_d)} | Test: {len(X_test_d)}")


# ─── Train Models ──────────────────────────────────────────────────────────────

print("\n[3/5] Training 3 Random Forest classifiers...")

def train_rf(X_train, y_train, label: str) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=3,
        class_weight="balanced",  # important for real data with unbalanced classes
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Cross-validation on training set
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1")
    print(f"  {label}: 5-fold CV F1 = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return clf


clf_dyslexia = train_rf(X_train_d, y_train_d, "Dyslexia")
clf_adhd     = train_rf(X_train_a, y_train_a, "ADHD   ")
clf_autism   = train_rf(X_train_au, y_train_au, "Autism ")


# ─── Evaluate ─────────────────────────────────────────────────────────────────

print("\n[4/5] Evaluation on held-out test set:")

def evaluate(clf, X_test, y_test, label: str) -> dict:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n  ── {label} ──")
    print(f"    Precision: {prec:.3f}")
    print(f"    Recall:    {rec:.3f}")
    print(f"    F1:        {f1:.3f}")
    print(f"    AUC-ROC:   {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])}")

    # Feature importances
    fi = pd.Series(clf.feature_importances_, index=FEATURE_NAMES)
    top = fi.nlargest(5)
    print(f"  Top 5 features:")
    for feat, importance in top.items():
        print(f"    {feat:<28} {importance:.3f}")

    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc}


results = {
    "dyslexia": evaluate(clf_dyslexia, X_test_d, y_test_d, "DYSLEXIA"),
    "adhd":     evaluate(clf_adhd, X_test_a, y_test_a, "ADHD"),
    "autism":   evaluate(clf_autism, X_test_au, y_test_au, "AUTISM SPECTRUM"),
}

# Save metrics
with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved metrics to {OUTPUT_DIR}/metrics.json")

# Check targets
print("\n  ── Target Check ──")
for condition, metrics in results.items():
    prec_ok = metrics["precision"] >= 0.80
    rec_ok  = metrics["recall"] >= 0.75
    status  = "✓ PASS" if prec_ok and rec_ok else "✗ FAIL"
    print(f"  {condition:<12} Precision {metrics['precision']:.2f} ≥ 0.80: {'✓' if prec_ok else '✗'} | "
          f"Recall {metrics['recall']:.2f} ≥ 0.75: {'✓' if rec_ok else '✗'}  → {status}")


# ─── TFLite Export ─────────────────────────────────────────────────────────────

print("\n[5/5] Exporting to TFLite format...")

if not TFLITE_AVAILABLE:
    print("  Skipping TFLite export — TensorFlow not installed.")
    print("  Install with: pip install tensorflow")
    print("  Then re-run this script.")
else:
    def rf_to_tflite(clf: RandomForestClassifier, output_path: str, label: str):
        """
        Convert sklearn Random Forest to TFLite.
        Strategy: wrap in a TF Keras model that calls tf.raw_ops to simulate
        the RF, OR convert via a simple neural net trained to mimic the RF.
        
        We use the mimic approach: train a small neural net to match RF predictions,
        then export that to TFLite. This is reliable cross-platform.
        """
        # Get RF predictions on training data for mimic training
        X_all = np.vstack([X_train_d, X_test_d]).astype(np.float32)
        y_prob_rf = clf.predict_proba(X_all)[:, 1].astype(np.float32)

        # Normalize features (important for neural net)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all).astype(np.float32)

        # Save scaler params for use in Flutter
        scaler_params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }
        scaler_path = output_path.replace(".tflite", "_scaler.json")
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f)

        # Build mimic neural net
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(12,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ], name=f"clearpath_{label.lower()}")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train mimic on RF probability outputs
        y_classes = (y_prob_rf > 0.5).astype(int)
        model.fit(
            X_scaled, y_classes,
            epochs=40,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
        )

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"  {label}: {output_path} ({size_kb:.1f} KB)")
        return scaler_params

    scalers = {}
    scalers["dyslexia"] = rf_to_tflite(
        clf_dyslexia, f"{OUTPUT_DIR}/dyslexia.tflite", "Dyslexia")
    scalers["adhd"] = rf_to_tflite(
        clf_adhd, f"{OUTPUT_DIR}/adhd.tflite", "ADHD")
    scalers["autism"] = rf_to_tflite(
        clf_autism, f"{OUTPUT_DIR}/autism.tflite", "Autism")

    # Save all scalers to one file for Flutter to load
    with open(f"{OUTPUT_DIR}/scalers.json", "w") as f:
        json.dump(scalers, f, indent=2)

    print(f"\n  Saved scalers to {OUTPUT_DIR}/scalers.json")
    print("\n  ── Copy these files to your Flutter app ──")
    print(f"  cp {OUTPUT_DIR}/dyslexia.tflite  <flutter_project>/assets/models/")
    print(f"  cp {OUTPUT_DIR}/adhd.tflite       <flutter_project>/assets/models/")
    print(f"  cp {OUTPUT_DIR}/autism.tflite     <flutter_project>/assets/models/")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
