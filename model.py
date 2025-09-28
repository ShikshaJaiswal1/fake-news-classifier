# model.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load and prepare data
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true['label'] = 1
fake['label'] = 0

news = pd.concat([fake, true], axis=0)
news = news.drop(["title","subject","date"], axis=1)
news = news.sample(frac=1).reset_index(drop=True)

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

news['text'] = news['text'].apply(clean_text)

# Train/test split
X = news['text']
y = news['label']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Train models
LR = LogisticRegression().fit(xv_train, y_train)
DTC = DecisionTreeClassifier().fit(xv_train, y_train)
RFC = RandomForestClassifier().fit(xv_train, y_train)
GBC = GradientBoostingClassifier().fit(xv_train, y_train)

def predict_label(model, text):
    text = clean_text(text)
    vect_text = vectorizer.transform([text])
    pred = model.predict(vect_text)[0]
    return "Genuine News" if pred == 1 else "Fake News"
