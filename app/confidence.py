from typing import List, Dict


# Confidence thresholds
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.10

# Minimum score for a chunk to be considered meaningful
SUPPORTING_EVIDENCE_THRESHOLD = 0.07


def calculate_confidence(results: List[Dict]) -> float:
    """
    Calculate evidence confidence from multiple reranked results.

    The strongest result remains the primary signal, but additional
    sufficiently strong results can increase confidence when a question
    requires information from multiple chunks.
    """

    if not results:
        return 0.0

    scores = [
        max(
            0.0,
            min(
                1.0,
                float(result.get("reranker_score", 0.0)),
            ),
        )
        for result in results[:3]
    ]

    if not scores:
        return 0.0

    # Strongest evidence is the primary signal.
    confidence = scores[0]

    # Additional supporting evidence contributes less than the strongest
    # result so that multiple weak chunks cannot easily override the gate.
    if len(scores) >= 2:
        if scores[1] >= SUPPORTING_EVIDENCE_THRESHOLD:
            confidence += 0.35 * scores[1]

    if len(scores) >= 3:
        if scores[2] >= SUPPORTING_EVIDENCE_THRESHOLD:
            confidence += 0.15 * scores[2]

    return max(
        0.0,
        min(1.0, confidence),
    )


def classify_confidence(confidence: float) -> str:
    """
    Convert numeric confidence into a human-readable level.
    """

    if confidence >= HIGH_THRESHOLD:
        return "HIGH"

    if confidence >= MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "LOW"


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