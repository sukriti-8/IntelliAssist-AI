import json
from typing import Dict

from google import genai


MODEL_NAME = "gemini-3.6-flash"


def analyze_sentiment_intent(
    text: str,
    client: genai.Client,
) -> Dict:
    """
    Analyze business text for sentiment and intent.

    Returns:
        {
            "sentiment": "Positive | Neutral | Negative",
            "intent": "...",
            "sentiment_confidence": 0.0-1.0,
            "intent_confidence": 0.0-1.0,
            "reason": "short explanation"
        }
    """

    text = str(text).strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    prompt = f"""
You are a business text analysis system.

Analyze the following text for:

1. Sentiment:
   - Positive
   - Neutral
   - Negative

2. Intent:
   Choose the most appropriate intent:
   - Complaint
   - Refund Request
   - Purchase Inquiry
   - Service Issue
   - Information Request
   - Cancellation
   - Feedback
   - Other

3. Confidence:
   Give a confidence score from 0.0 to 1.0 for both sentiment and intent.

4. Reason:
   Give a short explanation based only on the provided text.

Return ONLY valid JSON using exactly this structure:

{{
  "sentiment": "Positive",
  "intent": "Complaint",
  "sentiment_confidence": 0.95,
  "intent_confidence": 0.90,
  "reason": "The customer expresses dissatisfaction with the service."
}}

Text to analyze:
\"\"\"
{text}
\"\"\"
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    raw_text = response.text.strip()

    # Handle accidental markdown code fences from the model.
    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "", 1)
        raw_text = raw_text.replace("```", "", 1)
        raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON: {raw_text}"
        ) from exc

    required_fields = {
        "sentiment",
        "intent",
        "sentiment_confidence",
        "intent_confidence",
        "reason",
    }

    missing_fields = required_fields - set(result.keys())

    if missing_fields:
        raise ValueError(
            f"Analysis response is missing fields: {sorted(missing_fields)}"
        )

    valid_sentiments = {"Positive", "Neutral", "Negative"}

    if result["sentiment"] not in valid_sentiments:
        raise ValueError(
            f"Invalid sentiment: {result['sentiment']}"
        )

    confidence_fields = [
        "sentiment_confidence",
        "intent_confidence",
    ]

    for field in confidence_fields:
        try:
            result[field] = float(result[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid confidence value for {field}"
            ) from exc

        result[field] = max(0.0, min(1.0, result[field]))

    result["intent"] = str(result["intent"]).strip()
    result["reason"] = str(result["reason"]).strip()

    return result