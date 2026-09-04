from typing import List, Dict


# ---------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------

HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.10


# ---------------------------------------------------------
# Calculate evidence confidence
# ---------------------------------------------------------

def calculate_confidence(results: List[Dict]) -> float:
    """
    Calculate confidence from reranked retrieval results.

    The strongest retrieved chunk is treated as the main
    evidence signal.
    """

    if not results:
        return 0.0

    # Results should already be sorted by reranker score.
    top_score = float(
        results[0].get("reranker_score", 0.0)
    )

    # Keep confidence within [0, 1].
    confidence = max(0.0, min(1.0, top_score))

    return confidence


# ---------------------------------------------------------
# Classify confidence
# ---------------------------------------------------------

def classify_confidence(confidence: float) -> str:
    """
    Convert numeric confidence into a human-readable level.
    """

    if confidence >= HIGH_THRESHOLD:
        return "HIGH"

    if confidence >= MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------
# Complete confidence decision
# ---------------------------------------------------------

def assess_evidence(results: List[Dict]) -> Dict:
    """
    Assess whether retrieved evidence is strong enough
    to support an answer.
    """

    confidence = calculate_confidence(results)
    level = classify_confidence(confidence)

    if level == "HIGH":
        decision = "ANSWER"

    elif level == "MEDIUM":
        decision = "ANSWER_WITH_CAUTION"

    else:
        decision = "INSUFFICIENT_EVIDENCE"

    return {
        "confidence": confidence,
        "level": level,
        "decision": decision,
    }