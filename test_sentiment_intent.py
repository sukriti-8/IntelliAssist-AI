import os
from dotenv import load_dotenv
from google import genai

from app.sentiment_intent import analyze_sentiment_intent

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

tests = [
    "I am extremely happy with the service. The support team was excellent.",
    "I am disappointed with the service and want my money refunded.",
    "Can you provide information about your enterprise pricing?",
    "The application keeps crashing whenever I try to upload a document.",
]

for text in tests:
    print("\n" + "=" * 70)
    print("TEXT:", text)

    result = analyze_sentiment_intent(text, client)

    print("SENTIMENT:", result["sentiment"])
    print("INTENT:", result["intent"])
    print("SENTIMENT CONFIDENCE:", result["sentiment_confidence"])
    print("INTENT CONFIDENCE:", result["intent_confidence"])
    print("REASON:", result["reason"])