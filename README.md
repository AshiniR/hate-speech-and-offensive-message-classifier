# 🚫 Hate Speech & Offensive Message Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/AshiniR/hate-speech-and-offensive-message-classifier)

A state-of-the-art hate speech and offensive message classifier built with the **RoBERTa transformer model**, fine-tuned on the **Davidson et al. (2017) Twitter dataset**.  This model achieves exceptional performance with 0.9774 F1-score for Hate speech and offencive message detection and 96.23% overall accuracy, making it suitable for **social media moderation, community platforms, and chat applications**.

---

## 🎯 Overview

This project develops an intelligent text classification system that can automatically detect **hate speech, offensive language, and netural messages**. The model is optimized for high precision and recall to ensure that harmful content is correctly identified while minimizing false alarms.

### 🔑 Key Features

* 🤖 **Transformer-based Architecture**: Built on `roberta-base` for advanced natural language understanding
* ⚡ **High Performance**: 0.9774 F1-score for hate/offensive detection, 96.23% overall accuracy
* 🔧 **Hyperparameter Optimization**: Automated tuning using Optuna framework
* ⚖️ **Class Imbalance Handling**: Weighted cross-entropy loss for fairness across labels
* 📊 **Comprehensive Evaluation**: Precision, Recall, F1-score, confusion matrix
* 🚀 **Production Ready**: Model + tokenizer saved in Hugging Face format for direct deployment

---

## 📊 Model Performance

### Final Results on Test Set:

* **Overall Accuracy**: *96.23%*
* **Weighted F1-Score**: *0.9621*
* **Offensive/Hate** F1-Score: 0.9774 ✅ (Exceeds 0.90 acceptance threshold)
* **Offensive/Hate** Precision: 97.49% 
* **Offensive/Hate** Recall: 98% (High hate/offensive detection rate)
* **Neither** Precision: 89.82%
* **Neither** Recall: 87.52%


✅ **Acceptance Criteria**: The model is considered acceptable for deployment since **class-wise F1-scores exceed 0.90**, ensuring reliability in detecting harmful content.

---
Generalizability
📊 Strong Generalization: All performance metrics are evaluated on a completely unseen test set (15% of data, 3718 messages) that was never used during training or hyperparameter tuning, ensuring robust real-world performance and preventing overfitting.

---
## 📖 Dataset

**Source**: [Hate Speech and Offensive Language Dataset (Davidson et al., 2017)](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

### Dataset Statistics:

* **Total Tweets**: 24,783
* **Hate Speech / Offensive**: 20620
* **Neutral**: 4163
* **Average Tweet Length**: ~86 characters
* **Language**: English

### Dataset Split:
* Training Set: 70% (17,348 tweets) – model training
* Validation Set: 15% (3,717 tweets) – hyperparameter tuning
* Test Set: 15% (3,718 tweets) – final evaluation on unseen data

### Preprocessing Steps:
* Label mapping: 0 = Neither, 1 = Hate/Offensive.
* Text cleaning.
* Train/validation/test split.
* Tokenization with RoBERTa tokenizer.
* Dynamic padding and truncation.

---

## 🏗️ Architecture & Methodology

### Model Architecture

* **Base Model**: `roberta-base` (Hugging Face Transformers)
* **Task**: Multi-class sequence classification (2 labels)
* **Fine-tuning**: Custom classification head with 2 outputs
* **Tokenization**: RoBERTa tokenizer with optimal sequence length

### Training Strategy

1. Data Preprocessing: Hate/offencive message cleaning and label encoding
2. Tokenization: Dynamic padding with optimal max length
3. Class Balancing: Weighted loss function to handle imbalanced dataset
4. Hyperparameter Optimization: Optuna-based automated tuning
5. Evaluation: Comprehensive metrics on held-out test set

---

## ⚙️ Hyperparameter Optimization

Optimized with **Optuna (25 trials)** across ranges:

* Dropout rates: Hidden dropout (0.1-0.3), Attention dropout (0.1-0.2)
* Learning rate: 1e-5 to 5e-5 range
* Weight decay: 0.0 to 0.1 regularization
* Batch size: 8, 16, or 32 samples
* Gradient accumulation steps: 1 to 4
* Training epochs: 2 to 5 epochs
* Warmup ratio: 0.05 to 0.1 for learning rate scheduling

### Best Parameters Found:

* Hidden Dropout: `0.13034059066330464`
* Attention Dropout: `0.1935379847495239`
* Learning Rate: `1.031409901695853e-05`
* Weight Decay: `0.03606621145317628`
* Batch Size: `16`
* Gradient Accumulation: `1`
* Epochs: `2`
* Warmup Ratio: `0.0718442228846798`

---
## 📁 Project Structure

```plaintext
amy-bot-toxic-message-identifier/
│
├── Data/
│   └── labeled_data.csv             
│
├── notebooks/
│   └── hate_speech_and_offensive_message_classifier.ipynb  
│
├── README.md                         
├── .gitignore
```
---

## 🛠️ Technical Implementation
### Key Technologies
* 🤗 **Transformers**: Hugging Face transformers library
* 🔥 **PyTorch**: Deep learning framework
* 📊 **Scikit-learn**: Evaluation metrics and preprocessing
* 🎯 **Optuna**: Hyperparameter optimization
* 📈 **Matplotlib/Seaborn**: Data visualization
* 🐼 **Pandas**: Data manipulation

### Custom Features
* Weighted Loss Function: Handles class imbalance effectively
* Label Smoothing: 0.1 to prevent overconfidence
* Custom Metrics: Specialized  hate/offensive massege detection metrics
* Confusion Matrix Analysis: Detailed error analysis
* Class-specific Performance: Separate metrics for natural and hate/offensive

## 📊 Detailed Results

### Confusion Matrix :

|                     | Predicted Neither | Predicted Offensive/Hate |
|---------------------|-------------------|--------------------------|
| **Actual Neither**  | 547               | 78                       |
| **Actual Offensive**| 62                | 3031                     |
### Performance Breakdown

* **True Positives (Hate/Offensive correctly identified)**: 3031
* **True Negatives (Neutral correctly identified)**: 547
* **False Positives (Neutral incorrectly flagged)**: 78
* **False Negatives (Hate/offensive missed)**: 62

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
model = RobertaForSequenceClassification.from_pretrained("AshiniR/hate-speech-and-offensive-message-classifier")
tokenizer = RobertaTokenizer.from_pretrained("AshiniR/hate-speech-and-offensive-message-classifier")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_inference(text: str) -> list:
    """Returns prediction results in [{'label': str, 'score': float}, ...] format."""
    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
    
    # Convert to label format
    labels = ["neither", "hate/offensive"]
    results = []
    for i, prob in enumerate(probabilities[0]):
        results.append({
            "label": labels[i],
            "score": prob.item()
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

# Example usage
text = "I hate you!"
predictions = get_inference(text)
print(f"Text: '{text}'")
print(f"Predictions: {predictions}")
```

---

## 🎯 Use Cases
This hate/offensive massege classifier is ideal for:

### 💬 Messaging Platforms
* Discord bot moderation (Primary use case)
* SMS filtering systems
* Chat application content filtering
### 🛡️ Content Moderation
* Social media platforms
* Comment section filtering
* User-generated content screening

---

## 🔄 Deployment

### 🤖 Integration with Amy Discord Bot

This model serves as the **core hate & offensive message detection component** for **Amy**, an intelligent Discord moderation bot that:

* Detects **hate speech** and **offensive messages** in real time  
* Flags or removes harmful content automatically  
* Helps maintain a **safe and respectful server environment**  
* Supports server admins with actionable insights on community health 

---

⭐ If you find this project helpful, please give it a star! ⭐

---
