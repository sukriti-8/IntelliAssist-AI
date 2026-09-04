from typing import List, Dict


# ---------------------------------------------------------
# Evaluation data
# ---------------------------------------------------------

questions = [
    {"id": 1, "score": 0.5416, "answerable": True},
    {"id": 2, "score": 0.9970, "answerable": True},
    {"id": 3, "score": 0.3017, "answerable": True},
    {"id": 4, "score": 0.9565, "answerable": True},
    {"id": 5, "score": 0.7660, "answerable": True},
    {"id": 6, "score": 0.7280, "answerable": True},
    {"id": 7, "score": 0.6495, "answerable": True},
    {"id": 8, "score": 0.1110, "answerable": True},
    {"id": 9, "score": 0.0000, "answerable": False},
    {"id": 10, "score": 0.0000, "answerable": False},
]


# ---------------------------------------------------------
# Evaluate a threshold
# ---------------------------------------------------------

def evaluate_threshold(threshold: float) -> Dict:

    true_accept = 0
    false_accept = 0
    true_refusal = 0
    false_refusal = 0

    for question in questions:

        predicted_answerable = question["score"] >= threshold
        actual_answerable = question["answerable"]

        if predicted_answerable and actual_answerable:
            true_accept += 1

        elif predicted_answerable and not actual_answerable:
            false_accept += 1

        elif not predicted_answerable and not actual_answerable:
            true_refusal += 1

        elif not predicted_answerable and actual_answerable:
            false_refusal += 1

    return {
        "threshold": threshold,
        "true_accept": true_accept,
        "false_accept": false_accept,
        "true_refusal": true_refusal,
        "false_refusal": false_refusal,
    }


# ---------------------------------------------------------
# Test multiple thresholds
# ---------------------------------------------------------

thresholds = [
    0.00,
    0.05,
    0.10,
    0.11,
    0.12,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


print("=" * 80)
print("CONFIDENCE THRESHOLD CALIBRATION")
print("=" * 80)

for threshold in thresholds:

    result = evaluate_threshold(threshold)

    print(
        f"Threshold: {result['threshold']:.2f} | "
        f"True Accept: {result['true_accept']} | "
        f"False Accept: {result['false_accept']} | "
        f"True Refusal: {result['true_refusal']} | "
        f"False Refusal: {result['false_refusal']}"
    )