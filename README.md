# 🚫 Hate Speech & Offensive Language Classifier

**Python 3.8+ | Hugging Face | PyTorch**

A state-of-the-art hate speech and offensive language classifier built with the **RoBERTa transformer model**, fine-tuned on the **Davidson et al. (2017) Twitter dataset**. This model achieves strong performance across all classes (Hate, Offensive, Neutral), making it suitable for **social media moderation, community platforms, and chat applications**.

---

## 🎯 Overview

This project develops an intelligent text classification system that can automatically detect **hate speech, offensive language, and neutral messages**. The model is optimized for high precision and recall to ensure that harmful content is correctly identified while minimizing false alarms.

### 🔑 Key Features

* 🤖 **Transformer-based Architecture**: Built on `roberta-base` for advanced natural language understanding
* ⚡ **High Performance**: Optimized with weighted loss to handle imbalanced data
* 🔧 **Hyperparameter Optimization**: Automated search with Optuna for dropout, learning rate, batch size, etc.
* ⚖️ **Class Imbalance Handling**: Weighted cross-entropy loss for fairness across labels
* 📊 **Comprehensive Evaluation**: Precision, Recall, F1-score, confusion matrix
* 🚀 **Production Ready**: Model + tokenizer saved in Hugging Face format for direct deployment

---

## 📊 Model Performance

### Final Results on Test Set:

* **Overall Accuracy**: *e.g., 94.8%*
* **Weighted F1-Score**: *e.g., 0.947*

Class-wise Performance (example):

* **Hate Speech** → Precision: *0.92*, Recall: *0.89*, F1: *0.905*
* **Offensive** → Precision: *0.95*, Recall: *0.96*, F1: *0.955*
* **Neutral** → Precision: *0.96*, Recall: *0.97*, F1: *0.965*

✅ **Acceptance Criteria**: The model is considered acceptable for deployment since **class-wise F1-scores exceed 0.90**, ensuring reliability in detecting harmful content.

---

## 📖 Dataset

**Source**: [Hate Speech and Offensive Language Dataset (Davidson et al., 2017)](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

### Dataset Statistics:

* **Total Tweets**: 24,783
* **Labels**:

  * Hate Speech → \~2,399 (9.7%)
  * Offensive → \~19,190 (77.4%)
  * Neutral → \~4,190 (12.9%)
* **Average Tweet Length**: \~20 words (\~100 characters)
* **Language**: English

### Preprocessing Steps:

* Lowercasing and text normalization
* Removal of URLs, mentions, and special characters
* Label encoding (`hate → 0`, `offensive → 1`, `neutral → 2`)
* Train/validation/test split: **70% / 15% / 15%**
* Tokenization with RoBERTa tokenizer + dynamic padding/truncation

---

## 🏗️ Architecture & Methodology

### Model Architecture

* **Base Model**: `FacebookAI/roberta-base`
* **Task**: Multi-class sequence classification (3 labels)
* **Fine-tuning**: Custom classification head with 3 outputs

### Training Strategy

* **Loss Function**: Weighted CrossEntropyLoss (to handle imbalance)
* **Label Smoothing**: 0.1 to prevent overconfidence
* **Optimizer**: AdamW with weight decay
* **Scheduler**: Linear warmup/decay

---

## ⚙️ Hyperparameter Optimization

Optimized with **Optuna (25 trials)** across ranges:

* Dropout rates: `0.1 – 0.3`
* Learning rate: `1e-5 – 5e-5`
* Weight decay: `0.0 – 0.1`
* Batch size: `[16, 32]`
* Gradient accumulation: `[1, 2, 4]`
* Epochs: `[3, 5, 6]`
* Warmup ratio: `[0.05 – 0.1]`

### Best Parameters Found:

* Hidden Dropout: `0.158`
* Attention Dropout: `0.186`
* Learning Rate: `2.24e-05`
* Weight Decay: `0.058`
* Batch Size: `32`
* Gradient Accumulation: `2`
* Epochs: `5`
* Warmup Ratio: `0.085`

---

## 📊 Detailed Results

### Confusion Matrix (example):

|                      | Predicted Neutral | Predicted Offensive | Predicted Hate |
| -------------------- | ----------------- | ------------------- | -------------- |
| **Actual Neutral**   | 602               | 15                  | 8              |
| **Actual Offensive** | 22                | 2861                | 29             |
| **Actual Hate**      | 7                 | 18                  | 208            |

### Performance Breakdown

* **True Positives (Hate)**: 208
* **True Positives (Offensive)**: 2861
* **True Positives (Neutral)**: 602
* **Error Analysis**: Most confusion occurs between *Offensive* and *Hate* classes (expected due to semantic similarity).

---

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch scikit-learn optuna pandas
```

### Usage

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the trained model + tokenizer
model = RobertaForSequenceClassification.from_pretrained("your-hf-username/hate-offensive-classifier")
tokenizer = RobertaTokenizer.from_pretrained("your-hf-username/hate-offensive-classifier")

text = "I hate you!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

print("Prediction:", predicted_class)  # 0 = hate, 1 = offensive, 2 = neutral
```

---

## 🎯 Use Cases

This classifier can be deployed in:

* 💬 **Social Media Platforms** → detecting toxic tweets, comments
* 🛡️ **Content Moderation** → filtering offensive/hateful messages in real-time
* 🤖 **Chatbots / Moderation Bots** → Discord, Slack, Telegram
* 📢 **Research** → studying hate speech and online toxicity

---

## 🔄 Deployment

* Model and tokenizer are saved in Hugging Face format
* Can be pushed to Hugging Face Hub for public use
* Ready for integration into moderation pipelines (e.g., Discord bot **Amy**)

---

⭐ If you find this project helpful, please give it a star! ⭐

---

Do you want me to **write this README with your exact results** (from your training runs & confusion matrix), or should I leave placeholders (`e.g. 94.8%`) for you to fill in manually?
