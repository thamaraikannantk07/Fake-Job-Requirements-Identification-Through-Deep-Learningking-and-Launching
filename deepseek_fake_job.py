import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
import re
import joblib

nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict_job(clf, vectorizer):
    print("\n=== Fake Job Detector ===")
    while True:
        job_text = input("\nEnter job description (or 'exit' to quit): ").strip()
        if job_text.lower() == 'exit':
            break
        processed_text = preprocess_text(job_text)
        text_tfidf = vectorizer.transform([processed_text])
        prediction = clf.predict(text_tfidf)[0]
        print(f"\nPrediction: {'⚠️ FAKE JOB' if prediction == 1 else '✅ REAL JOB'}")

def load_or_train_model():
    try:
        clf = joblib.load('model/fake_job_classifier.pkl')
        vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        print("Pre-trained model loaded.")
    except:
        print("No pre-trained model found. Training a minimal model...")
        data = {
            "text": [
                "Hiring experienced developers for full-time roles",
                "Earn $5000/day from home with no experience!"
            ],
            "label": [0, 1]
        }
        df = pd.DataFrame(data)
        df['text'] = df['text'].apply(preprocess_text)
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['text'])
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, df['label'])
        os.makedirs("model", exist_ok=True)
        joblib.dump(clf, 'model/fake_job_classifier.pkl')
        joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
        print("Minimal model trained and saved.")
    return clf, vectorizer

def main():
    clf, vectorizer = load_or_train_model()
    predict_job(clf, vectorizer)

if __name__ == "__main__":
    main()