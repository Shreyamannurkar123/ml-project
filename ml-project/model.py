# model.py (Base Model)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data
texts = ["good news", "bad news", "fake news", "real news"]
texts = [t.lower() for t in texts]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Model
model = LogisticRegression()
model.fit(X, labels)

print("Base Model Trained")