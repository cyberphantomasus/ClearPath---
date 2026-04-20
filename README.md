# ClearPath — AI Child Learning Disability Screener

> University of Nahrain · Department of AI & Robotics Engineering · 2026  
> Mahdi Ahmed · Anmar Nihad · Abdullah Sharhabel

The first Arabic-language AI child developmental screening app built for Iraq.  
**Free. Offline. 10 minutes. Any Android phone.**

---

## What It Does

ClearPath screens children (ages 4–12) for three developmental conditions using a 10-minute interactive session. The child plays three mini-games. Hidden behind the games, the AI measures 12 precise behavioral signals. At the end, the app generates a detailed report — entirely offline, no internet required.

| Game | Screens For | Key Signals |
|------|-------------|-------------|
| Letter Matching | Dyslexia | Error pattern score, response time, systematic errors |
| Focus Task | ADHD | Impulsivity, attention drift, response time variance |
| Social Recognition | Autism Spectrum | Emotion accuracy, social hesitation, sequence memory |

---

## Project Structure

```
clearpath/
├── lib/
│   ├── main.dart               # App entry, theme, routing
│   ├── models/
│   │   └── session_model.dart  # Session data model, InteractionEvent
│   ├── screens/
│   │   ├── home_screen.dart    # Child name/age input, start session
│   │   ├── game1_screen.dart   # Arabic letter matching (Dyslexia)
│   │   ├── game2_screen.dart   # Attention/focus task (ADHD)
│   │   ├── game3_screen.dart   # Social pattern recognition (Autism)
│   │   └── report_screen.dart  # Results + PDF generation
│   └── services/
│       ├── logger.dart         # SQLite interaction logger (10ms precision)
│       ├── features.dart       # Compute 12 behavioral signals
│       ├── inference.dart      # TFLite model runner (3 conditions)
│       ├── tts_service.dart    # Arabic TTS game instructions
│       └── pdf_service.dart    # PDF report generator
├── assets/
│   └── models/                 # dyslexia.tflite, adhd.tflite, autism.tflite
│                               # (copy here after running python/train.py)
├── python/
│   ├── train.py                # Full training pipeline — run on Google Colab
│   └── features.py             # Feature extraction — mirrors Dart logic exactly
└── pubspec.yaml
```

---

## Setup Instructions

### Flutter App

**Requirements:** Flutter 3.x, Android Studio, Android device or emulator

```bash
# 1. Install dependencies
flutter pub get

# 2. Run on connected Android device
flutter run

# 3. Build release APK
flutter build apk --release
```

**First run:** The app works immediately using heuristic fallback scores.  
Once TFLite models are trained and copied to `assets/models/`, it will use the real ML classifiers.

---

### Python AI Pipeline

**Requirements:** Python 3.9+, Google Colab (recommended) or local machine

```bash
# Install dependencies
pip install numpy pandas scikit-learn tensorflow

# Run full training pipeline
python python/train.py
```

This will:
1. Generate 800-row synthetic dataset (200 typical + 200 per condition)
2. Train 3 Random Forest classifiers (target: precision >0.80, recall >0.75)
3. Export 3 TFLite models to `models/` folder
4. Print evaluation metrics

**After training, copy models to Flutter:**
```bash
cp models/dyslexia.tflite  assets/models/
cp models/adhd.tflite      assets/models/
cp models/autism.tflite    assets/models/
```

Then rebuild: `flutter build apk --release`

---

### Feature Extraction Validation

Run the Python feature extractor to verify it matches the Dart implementation:
```bash
python python/features.py
# Should print all 12 signals and "✓ All checks passed"
```

---

## The 12 Behavioral Signals

| # | Signal | Formula | Condition |
|---|--------|---------|-----------|
| 1 | Mean Response Time | mean(response_time_ms) correct answers | All |
| 2 | Response Time Variance | std / mean (CoV) | ADHD |
| 3 | Error Rate | wrong / total answers | All |
| 4 | Error Pattern Score | repeated same-pair errors / total errors | Dyslexia |
| 5 | Retry Rate | retries_after_wrong / total_wrong | ADHD |
| 6 | Attention Drift Index | accuracy_last_quarter − accuracy_first_quarter | ADHD |
| 7 | Impulsivity Score | false_taps / (false_taps + correct) | ADHD |
| 8 | Recovery Speed | mean(rt_after_distractor) / baseline_rt | ADHD |
| 9 | Emotion Accuracy | correct_emotion / total_emotion_tasks | Autism |
| 10 | Social Hesitation Time | mean(rt) on social scenario questions | Autism |
| 11 | Sequence Memory Score | longest_correct_seq / max_possible | Memory |
| 12 | Engagement Decay Rate | (taps/min first half − second half) / first half | ADHD |

---

## Technology Stack

| Layer | Tool | Cost |
|-------|------|------|
| Mobile framework | Flutter 3.x | Free |
| Game engine | Flame 1.x | Free |
| Interaction logging | SQLite (sqflite) | Free |
| Feature extraction | Python + Pandas | Free |
| ML classifiers | Scikit-learn Random Forest | Free |
| Model training | Google Colab | Free |
| On-device inference | TensorFlow Lite | Free |
| PDF report | pdf Flutter package | Free |
| Arabic TTS | flutter_tts | Free |
| **Total** | | **$0** |

---

## Sprint Schedule

| Days | Phase | Goal |
|------|-------|------|
| 1–2 | Setup & Games | All 3 games playable, logger capturing all events |
| 3–4 | AI + Integration | TFLite models running offline, PDF auto-generates |
| 5–7 | Polish & Demo | Validated on 10 real children, demo rehearsed |

---

## Team

| Name | Role | Responsibilities |
|------|------|-----------------|
| Mahdi Ahmed | Team Lead | Flutter architecture, all 3 game UIs, PDF report, demo |
| Anmar Nihad | AI Lead | 12 signals, 3 classifiers, TFLite export, technical report |
| Abdullah Sharhabel | Software | SQLite logger, feature pipeline, TFLite integration, TTS |

**Rotation rule:** Every student writes AI code, builds app features, and runs user tests.

---

## Notes

- All Arabic text in the app uses RTL layout via Flutter's `Directionality.rtl`
- The app runs fully offline — `flutter build apk` with no internet permissions required
- The SQLite database stores all sessions locally, never transmitted
- PDF generation uses the `pdf` Flutter package with Cairo font for Arabic rendering
- TFLite models run inference in <100ms on mid-range Android devices

---

*Early detection changes lives. كشف مبكر — يغيّر المسار*
