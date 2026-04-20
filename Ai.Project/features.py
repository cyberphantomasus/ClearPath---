"""
ClearPath — Feature Extraction (Python)
=========================================
Mirrors the Dart lib/services/features.dart logic EXACTLY.
Use this to verify features match between training and on-device inference.

Input:  A list of interaction event dicts (from SQLite or CSV)
Output: Dict of 12 signal values
"""

import numpy as np
from typing import List, Dict, Optional
import pandas as pd


# ─── Event structure ──────────────────────────────────────────────────────────

# Events have these fields (same as Dart InteractionEvent):
# {
#   "game_id": int,           # 1, 2, or 3
#   "action_type": str,       # 'correct', 'wrong', 'retry', 'false_tap',
#                             #  'emotion_select', 'social_response',
#                             #  'sequence_tap', 'miss'
#   "value": float,           # response time in ms (or 0/1 for binary events)
#   "metadata": str,          # optional — letter pair, emotion label, gesture
#   "timestamp": int,         # unix timestamp ms
# }


def extract_features(events: List[Dict]) -> Dict[str, float]:
    """
    Main entry point. Takes a list of raw interaction events,
    returns the 12-signal feature dict.
    """
    g1 = [e for e in events if e["game_id"] == 1]
    g2 = [e for e in events if e["game_id"] == 2]
    g3 = [e for e in events if e["game_id"] == 3]

    return {
        "mean_response_time":      _mean_response_time(events),
        "response_time_variance":  _response_time_variance(events),
        "error_rate":              _error_rate(events),
        "error_pattern_score":     _error_pattern_score(g1),
        "retry_rate":              _retry_rate(events),
        "attention_drift_index":   _attention_drift_index(events),
        "impulsivity_score":       _impulsivity_score(g2),
        "recovery_speed":          _recovery_speed(g2),
        "emotion_accuracy":        _emotion_accuracy(g3),
        "social_hesitation_time":  _social_hesitation_time(g3),
        "sequence_memory_score":   _sequence_memory_score(g3),
        "engagement_decay_rate":   _engagement_decay_rate(events),
    }


def features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dict to ordered list for model input."""
    return [
        features["mean_response_time"],
        features["response_time_variance"],
        features["error_rate"],
        features["error_pattern_score"],
        features["retry_rate"],
        features["attention_drift_index"],
        features["impulsivity_score"],
        features["recovery_speed"],
        features["emotion_accuracy"],
        features["social_hesitation_time"],
        features["sequence_memory_score"],
        features["engagement_decay_rate"],
    ]


# ─── Signal Implementations ────────────────────────────────────────────────────

def _mean_response_time(events: List[Dict]) -> float:
    """Signal 1: mean(response_time_ms) across all correct answers."""
    correct_times = [e["value"] for e in events if e["action_type"] == "correct"]
    return float(np.mean(correct_times)) if correct_times else 1500.0


def _response_time_variance(events: List[Dict]) -> float:
    """Signal 2: std / mean (coefficient of variation) on response times."""
    correct_times = [e["value"] for e in events if e["action_type"] == "correct"]
    if len(correct_times) < 2:
        return 0.0
    mean = np.mean(correct_times)
    if mean == 0:
        return 0.0
    return float(np.std(correct_times) / mean)


def _error_rate(events: List[Dict]) -> float:
    """Signal 3: wrong_answers / total_answers."""
    taps = [e for e in events if e["action_type"] in ("correct", "wrong")]
    if not taps:
        return 0.0
    wrong = sum(1 for e in taps if e["action_type"] == "wrong")
    return wrong / len(taps)


def _error_pattern_score(g1_events: List[Dict]) -> float:
    """
    Signal 4: count(repeated same error pair) / total_errors.
    Score near 1.0 = same mistake made repeatedly = systematic = dyslexia marker.
    """
    wrong = [e for e in g1_events if e["action_type"] == "wrong"]
    if len(wrong) < 2:
        return 0.0

    pair_counts: Dict[str, int] = {}
    for e in wrong:
        key = e.get("metadata") or "unknown"
        pair_counts[key] = pair_counts.get(key, 0) + 1

    repeated = sum(count for count in pair_counts.values() if count > 1)
    return repeated / len(wrong)


def _retry_rate(events: List[Dict]) -> float:
    """Signal 5: retries_after_wrong / total_wrong_answers."""
    wrong = sum(1 for e in events if e["action_type"] == "wrong")
    if wrong == 0:
        return 0.0
    retries = sum(1 for e in events if e["action_type"] == "retry")
    return retries / wrong


def _attention_drift_index(events: List[Dict]) -> float:
    """
    Signal 6: accuracy_last_quarter - accuracy_first_quarter.
    Negative value = accuracy declining = attention drifting (ADHD).
    """
    taps = [e for e in events if e["action_type"] in ("correct", "wrong")]
    if len(taps) < 8:
        return 0.0

    q = len(taps) // 4
    first = taps[:q]
    last = taps[-q:]

    def accuracy(subset):
        if not subset:
            return 0.0
        return sum(1 for e in subset if e["action_type"] == "correct") / len(subset)

    return accuracy(last) - accuracy(first)


def _impulsivity_score(g2_events: List[Dict]) -> float:
    """Signal 7: false_taps / (false_taps + correct_taps) from Game 2."""
    false_taps = sum(1 for e in g2_events if e["action_type"] == "false_tap")
    correct = sum(1 for e in g2_events if e["action_type"] == "correct")
    total = false_taps + correct
    return false_taps / total if total > 0 else 0.0


def _recovery_speed(g2_events: List[Dict]) -> float:
    """Signal 8: mean(response_time after distractor) / baseline."""
    correct = [e for e in g2_events if e["action_type"] == "correct"]
    if not correct:
        return 1.0
    baseline = np.mean([e["value"] for e in correct])
    if baseline == 0:
        return 1.0

    post_distractor = []
    for i in range(len(g2_events) - 1):
        if (g2_events[i]["action_type"] == "false_tap" and
                g2_events[i + 1]["action_type"] == "correct"):
            post_distractor.append(g2_events[i + 1]["value"])

    if not post_distractor:
        return 1.0
    return float(np.mean(post_distractor) / baseline)


def _emotion_accuracy(g3_events: List[Dict]) -> float:
    """Signal 9: correct_emotion_matches / total_emotion_tasks (Game 3)."""
    emotion_taps = [e for e in g3_events if e["action_type"] == "emotion_select"]
    if not emotion_taps:
        return 0.5
    correct = sum(1 for e in emotion_taps if e["value"] == 1.0)
    return correct / len(emotion_taps)


def _social_hesitation_time(g3_events: List[Dict]) -> float:
    """Signal 10: mean response time on social scenario questions only."""
    social = [e for e in g3_events if e["action_type"] == "social_response"]
    if not social:
        return 1500.0
    return float(np.mean([e["value"] for e in social]))


def _sequence_memory_score(g3_events: List[Dict]) -> float:
    """Signal 11: longest_correct_sequence / max_possible_sequence."""
    seq_events = [e for e in g3_events if e["action_type"] == "sequence_tap"]
    if not seq_events:
        return 0.0

    longest = 0
    current = 0
    total = len(seq_events)

    for e in seq_events:
        if e["value"] == 1.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest / total if total > 0 else 0.0


def _engagement_decay_rate(events: List[Dict]) -> float:
    """
    Signal 12: (taps/min first half - taps/min second half) / taps/min first half.
    Positive = engagement dropping.
    """
    taps = [e for e in events if e["action_type"] in ("correct", "wrong")]
    if len(taps) < 4:
        return 0.0

    mid = len(taps) // 2
    first_half = taps[:mid]
    second_half = taps[mid:]

    def taps_per_min(half: List[Dict]) -> float:
        if len(half) < 2:
            return 1.0
        duration_ms = half[-1]["timestamp"] - half[0]["timestamp"]
        duration_min = max(duration_ms / 60000.0, 0.001)
        return len(half) / duration_min

    first_rate = taps_per_min(first_half)
    second_rate = taps_per_min(second_half)

    if first_rate == 0:
        return 0.0
    return (first_rate - second_rate) / first_rate


# ─── CSV pipeline ─────────────────────────────────────────────────────────────

def process_session_csv(csv_path: str) -> Dict[str, float]:
    """
    Load a session CSV exported from the Flutter app's SQLite log
    and extract all 12 features.
    """
    df = pd.read_csv(csv_path)
    events = df.to_dict("records")
    return extract_features(events)


def process_all_sessions(events_csv: str) -> pd.DataFrame:
    """
    Process a CSV containing events from multiple sessions.
    Returns a DataFrame with one row per session and all 12 features.
    """
    df = pd.read_csv(events_csv)
    results = []
    for session_id, group in df.groupby("session_id"):
        events = group.to_dict("records")
        features = extract_features(events)
        features["session_id"] = session_id
        results.append(features)
    return pd.DataFrame(results)


# ─── Validation ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test with synthetic events
    import random
    random.seed(42)

    def make_event(game_id, action_type, value, metadata="", ts_offset=0):
        return {
            "game_id": game_id,
            "action_type": action_type,
            "value": value,
            "metadata": metadata,
            "timestamp": 1700000000000 + ts_offset * 500,
        }

    # Build a fake session
    test_events = []
    for i in range(20):
        rt = random.gauss(900, 200)
        correct = random.random() > 0.15
        test_events.append(make_event(1, "correct" if correct else "wrong", rt,
                                       "ب_ن" if not correct else "ب", i))

    for i in range(30):
        rt = random.gauss(750, 300)
        correct = random.random() > 0.25
        if random.random() < 0.1:
            test_events.append(make_event(2, "false_tap", 0.0, "", 20 + i))
        else:
            test_events.append(make_event(2, "correct" if correct else "wrong", rt, "", 20 + i))

    for i in range(16):
        test_events.append(make_event(
            3, "emotion_select", 1.0 if random.random() > 0.4 else 0.0, "", 50 + i))

    features = extract_features(test_events)
    print("Feature extraction smoke test:")
    for k, v in features.items():
        print(f"  {k:<28}: {v:.4f}")

    vector = features_to_vector(features)
    print(f"\nFeature vector length: {len(vector)} (expected 12)")
    assert len(vector) == 12, "Feature vector must have 12 elements!"
    print("✓ All checks passed")
