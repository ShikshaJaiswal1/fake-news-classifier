# 📰 Fake News Classifier

A machine learning project for detecting fake news using **Natural Language Processing (NLP)** and multiple classifiers.  
This project compares the performance of **Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting** to identify whether a news article is **real** or **fake**.

---

## 🚀 Features
- Text preprocessing (tokenization, stopword removal, vectorization with TF-IDF/CountVectorizer)
- Implementation of multiple ML models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
- Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

---

## 📂 Project Structure
.
├── True.csv # Dataset (true news samples)


├── Fake.csv # Dataset (fake news samples)


├── fakenews.ipynb # Jupyter Notebook with code and results


├── requirements.txt # Python dependencies


└── README.md # Project documentation





---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/fake-news-classifier.git
   cd fake-news-classifier

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

3. Install dependencies:
   pip install -r requirements.txt









