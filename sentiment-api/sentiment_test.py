from transformers import pipeline

# Load a pre-trained sentiment-analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Test sentences
texts = [
    "I love working with AWS and Terraform!",
    "This deployment process is frustrating.",
    "The movie was okay, not great but not terrible."
]

for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}")
    print(f"→ Label: {result['label']}, Confidence: {result['score']:.4f}\n")

