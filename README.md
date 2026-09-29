# AI-Based Automotive Review and Customer Sentiment Analytics

## 📝 Overview

This repository contains an end-to-end Natural Language Processing (NLP) pipeline that performs Aspect-Based Sentiment Analysis (ABSA) on automotive customer reviews. Standard sentiment analysis models assign a single polarity to a full review, which fails to capture nuanced feedback where a customer might praise the engine but criticize the battery range. This system uses Transformer-based architectures to extract specific vehicle aspects (e.g., mileage, safety, infotainment) and assigns an independent sentiment score to each.

**Academic Context:** Developed as a college project for the Department of Computer Science and Engineering (AI & ML) at UEM Kolkata.

## ✨ Key Features

* **Aspect Term Extraction:** Automatically identifies specific vehicle components mentioned in unstructured text.


* **Polarity Classification:** Classifies the contextual sentiment (Positive, Negative, Neutral) tied strictly to the extracted aspect.


* **Pre-trained Transformer Backbone:** Leverages DeBERTa-v3/BERT for high-accuracy contextual embeddings.


* **Batch Processing:** Parses CSV files containing bulk reviews and outputs a structured tabular analysis mapping aspects to sentiments.



## ⚙️ System Architecture

1. **Text Ingestion & Preprocessing:** Cleans and formats raw review text or batch datasets.


2. **Subword Tokenization:** Maps string inputs to positional IDs suitable for transformer consumption.


3. **Local Context Focus (LCF) Layer:** Highlights adjectives and sentiment-bearing words immediately surrounding the target vehicle aspect to decouple mixed opinions within a single sentence.


4. **Joint Prediction:** Outputs Sequence Labeling (IOB tags) for extraction and polarity classification with confidence probabilities.



## 🛠️ Tech Stack

* **Language:** Python


* **Framework:** PyTorch, PyABSA


* **Transformers:** Hugging Face `transformers`

* **Data Manipulation:** Pandas


* **Data Visualization:** Matplotlib, Seaborn


* **Environment:** Google Colab (T4 GPU) / Databricks



## 🚀 Getting Started

### Prerequisites

Ensure you have a GPU-enabled environment (like Google Colab) to handle the transformer weights.

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/automotive-sentiment-analytics.git
cd automotive-sentiment-analytics
pip install pyabsa -U
pip install "transformers<=4.29.0"
pip install pandas matplotlib seaborn

```

## 💻 Usage

### Single Sentence Prediction

Load the pre-trained fast-LCF-ATEPC model and run inference on a single string:

```python
from transformers import PretrainedConfig
if not hasattr(PretrainedConfig, 'is_decoder'):
    PretrainedConfig.is_decoder = False

from pyabsa import AspectTermExtraction as ATEPC

# Initialize the Aspect Extractor
aspect_extractor = ATEPC.AspectExtractor('english', auto_device=True)

# Run prediction
reviews = ["The steering wheel is very responsive, but the braking system feels slightly soft."]
results = aspect_extractor.predict(text=reviews, print_result=True)

```

### Batch Processing from CSV

Process a full dataset of automotive reviews and export the structured analysis:

```python
import pandas as pd
from pyabsa import AspectTermExtraction as ATEPC

aspect_extractor = ATEPC.AspectExtractor('english', auto_device=True)
df = pd.read_csv('sample_reviews.csv')
batch_reviews = df['review_text'].tolist()

batch_results = aspect_extractor.predict(text=batch_reviews, print_result=False)

```

## 📊 Sample Output

The model successfully decouples conflicting sentiments within the same review:

| Review Text | Aspect Term | Sentiment | Confidence |
| --- | --- | --- | --- |
| *"The battery range is excellent, but the charging time is too long."*<br> | battery range

 | Positive

 | 0.8212

 |
|  | charging time

 | Negative

 | 0.6033

 |
| *"Exceptional mileage and high-quality materials on the dashboard."*<br> | mileage

 | Positive

 | 0.9928

 |
|  | dashboard

 | Positive

 | 0.9957

 |
