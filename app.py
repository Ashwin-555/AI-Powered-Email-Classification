import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data=pd.read_csv(r'C:\Users\ashwi\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\F6306026B635FE70DB280CDB5CA3D995B768EB90\transfers\2026-08\merged_data.csv')
sia = SentimentIntensityAnalyzer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def assign_urgency(text, category):
    text = clean_text(text)
    score = 0

    high_keywords = ["urgent", "immediately", "asap", "emergency","critical", "today", "right now"]
    medium_keywords = ["issue", "problem", "help", "support","request", "soon"]
    low_keywords = ["suggestion", "feedback", "thank you","inquiry", "information"]
    negations = ["no rush", "not urgent", "whenever possible"]
    action_verbs = ["approve", "grant", "enable", "unlock","reset", "activate", "escalate"]
    for word in high_keywords:
        if word in text:
            score += 2
    for word in medium_keywords:
        if word in text:
            score += 1
    for word in low_keywords:
        if word in text:
            score -= 1
    for word in negations:
        if word in text:
            score -= 2
    sentiment = sia.polarity_scores(text)
    compound = sentiment["compound"]
    if compound < -0.5:
        score += 2
    elif compound < -0.2:
        score += 1
    if category == "Complaint":
        score += 1
    elif category == "Request":
        for verb in action_verbs:
            if verb in text:
                score += 1
    elif category == "Feedback":
        score -= 1
    elif category == "Spam":
        return "Low"

    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"
    
data["urgency"] = data.apply(lambda row: assign_urgency(row["text"], row["label"]),axis=1)
data=pd.DataFrame(data)
X = data["text"]
y_category = data["label"]
y_urgency = data["urgency"]


X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(X, y_category, y_urgency,test_size=0.2,random_state=42,stratify=y_category)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

cat_model = LogisticRegression(max_iter=1000)
cat_model.fit(X_train_vec, y_cat_train)


urg_model = LogisticRegression(max_iter=1000)
urg_model.fit(X_train_vec, y_urg_train)

cat_pred = cat_model.predict(X_test_vec)

print("\n===== CATEGORY MODEL =====")
print("Accuracy:", round(accuracy_score(y_cat_test, cat_pred)*100, 2), "%")
print(classification_report(y_cat_test, cat_pred))

# -------------------------
# Evaluate Urgency Model
# -------------------------
urg_pred = urg_model.predict(X_test_vec)

print("\n===== URGENCY MODEL =====")
print("Accuracy:", round(accuracy_score(y_urg_test, urg_pred)*100, 2), "%")
print(classification_report(y_urg_test, urg_pred))

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(cat_model, open("category_model.pkl", "wb"))
pickle.dump(urg_model, open("urgency_model.pkl", "wb"))

print("\nModels saved successfully!")