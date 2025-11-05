# Sentiment Analysis Model

This project demonstrates a simple **sentiment analysis model** using the Hugging Face `transformers` library.
It predicts whether a given text expresses **positive**, **negative**, or **neutral** sentiment.

---

## Overview

Sentiment analysis is a Natural Language Processing (NLP) technique used to determine the emotional tone of text.
This project uses a **pre-trained Transformer model** to analyze short sentences and return sentiment predictions.

---

##	Example Output

```bash
Text: I love working with AWS and Terraform!
→ Label: POSITIVE, Confidence: 0.9993

Text: This deployment process is frustrating.
→ Label: NEGATIVE, Confidence: 0.9981

Text: The movie was okay, not great but not terrible.
→ Label: NEUTRAL, Confidence: 0.7652
```
---

##	Requirements
Python 3.9 or later
transformers
torch
