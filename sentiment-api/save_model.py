from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Choose a small, fast model
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print("Downloading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Save locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Model and tokenizer saved to ./model/")

