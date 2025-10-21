# 🚀 Sentiment Analysis Web Application

A powerful Streamlit web application for analyzing sentiment in text using machine learning and deep learning models. Supports single-text analysis and batch CSV processing, with multi-model support, confidence scoring, and visualized results.

---

## 📋 Table of Contents
- [Features](#-features)  
- [Demo](#-demo)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Project Structure](#-project-structure)  
- [Models](#-models)  
- [Configuration](#-configuration)  
- [Future Enhancements](#-future-enhancements)  


---

## ✨ Features

### 🔍 Core Analysis
- **Single Text Analysis** — Real-time sentiment for individual texts.  
- **Batch Processing** — Analyze multiple texts from CSV files.  
- **Multi-Model Support** — Select between AI model (LSTM), Rule-based fallback, or Auto.  
- **Confidence Scoring** — Get confidence/probability for each prediction.

### 🛠 Technical Features
- Text preprocessing: cleaning, tokenization, stopword removal, lemmatization.  
- **LSTM Model** for deep-learning-based sentiment classification.  
- **Rule-based Fallback**: keyword scoring when models are unavailable.  
- Auto-detection of text column names in CSV uploads.

### 📊 Visualization & UI
- Color-coded sentiment display.  
- Progress bars for batch processing.  
- Exportable CSV results.

---

## 🎯 Demo

### Quick Start Examples
- 😊 **Positive**: `I love this product! It's amazing and works perfectly!`  
- 😞 **Negative**: `This is terrible service, very disappointed`  
- 😐 **Neutral**: `The package arrived on time as described`

### Live Demo
Add your deployed app link here: `[Deployed App URL](#)`

---

## 🚀 Installation

### Prerequisites
- Python 3.8+  
- `pip`

### Step-by-step
```bash
# Clone the repo
git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app

# (Recommended) Create virtual environment
python -m venv sentiment_env
source sentiment_env/bin/activate   # On Windows: sentiment_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Place model files
Ensure these files are in the project root:
- `minimal_model.h5` — Trained LSTM model  
- `tokenizer.pickle` — Text tokenizer

---

## 💻 Usage

### Start the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

### Single Text Analysis
1. Enter/paste text in the input box.  
2. Choose Method: `Auto`, `AI Model`, or `Rule-based`.  
3. Click **Analyze Sentiment**.  
4. View label, confidence, and detailed insights.

### Batch File Analysis
1. Prepare a CSV with a text column.  
2. Upload the CSV in Batch Analysis.  
3. Select the detected text column.  
4. Click **Analyze All Texts**.  
5. Download results as CSV.

### Supported CSV column names (auto-detection)
`text`, `tweet`, `review`, `comment`, `message`, `content`

---

## 📁 Project Structure
```
sentiment-analysis-app/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── minimal_model.h5       # Trained LSTM model
├── tokenizer.pickle       # Text tokenizer
├── twitter dataset.csv    # Training data (if available)
│
├── examples/              # Sample data files
│   ├── sample_tweets.csv
│   ├── product_reviews.csv
│   └── customer_feedback.csv
│
└── README.md              # This file
```

---

## 🤖 Models

### LSTM Neural Network
- **Architecture:** Embedding → LSTM → Dense layers  
- **Training Data:** Twitter sentiment dataset (example)  
- **Input:** Preprocessed text sequences  
- **Output:** Sentiment probability (0–1)  
- **Accuracy:** *Add your model accuracy here*

### Rule-based Analysis
- **Method:** Keyword matching + scoring  
- **Use case:** Fallback when ML models unavailable  
- **Keywords** (example):
  - Positive: `good`, `great`, `amazing`, `love`, `excellent`  
  - Negative: `bad`, `terrible`, `awful`, `hate`, `worst`

---

## ⚙️ Configuration

### Analysis Methods
- **Auto:** Automatically selects the best method.  
- **AI Model:** Uses the trained LSTM.  
- **Rule-Based:** Uses keyword scoring for speed/fallback.

### Text Preprocessing Steps
- Remove `@mentions` and URLs  
- Convert to lowercase  
- Remove numbers and punctuation  
- Tokenization and stopword removal  
- Lemmatization

---

## 📊 API Documentation

### `predict_sentiment`
```python
def predict_sentiment(text, method='auto'):
    """
    Analyze sentiment of input text

    Args:
        text (str): Input text to analyze
        method (str): Analysis method ('auto', 'ai', 'rule')

    Returns:
        dict: {
            'label': 'Positive' | 'Negative' | 'Neutral',
            'score': float (0-1),
            'method': str,
            'processed_text': str
        }
    """
```

---

## 🎪 Examples

### Example 1 — Product Review
Input:
```
"This phone has an amazing camera and battery life!"
```
Output:
- Label: 😊 **Positive**  
- Score: `0.87`  
- Confidence: High

### Example 2 — Customer Complaint
Input:
```
"Terrible service, waited 2 hours and food was cold"
```
Output:
- Label: 😞 **Negative**  
- Score: `0.15`  
- Confidence: High

### Example 3 — CSV Batch Processing
Input file:
```csv
text,user
"Great product, highly recommend!",user1
"Poor quality, would not buy again",user2
"Average product, does the job",user3
```
Output:
```csv
text,user,sentiment,score
"Great product, highly recommend!",user1,Positive,0.82
"Poor quality, would not buy again",user2,Negative,0.23
"Average product, does the job",user3,Neutral,0.55
```

---

## 🔧 Troubleshooting

### Common Issues

**ModuleNotFoundError**
```bash
# Install missing package
pip install package_name
```

**Model Loading Errors**
- Ensure `minimal_model.h5` and `tokenizer.pickle` are in the project root.  
- Check file permissions.

**CSV Column Detection**
- Rename your text column to `text`, or use auto-detection.

**Memory Issues**
- Reduce batch size or use rule-based analysis for large batches.

### Performance Tips
- Use **AI Model** for single-text accuracy.  
- Use **Rule-based** for speed on large batches.  
- Close other applications to free memory.

---

## 🙏 Acknowledgments
- Built with **Streamlit**  
- ML via **TensorFlow / Keras**  
- Text processing with **NLTK**  
- Data handling with **Pandas**

---

## 📈 Future Enhancements
- Multi-language sentiment analysis  
- Real-time social media monitoring  
- Advanced emotion detection (joy, anger, sadness...)  
- Production-ready API endpoints  
- Mobile app version

---
