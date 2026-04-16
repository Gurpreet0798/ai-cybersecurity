import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
df = pd.read_csv('data/phishing_emails.csv')

# Split data
X = df['text']
y = df['label']

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_vec, y)

# Create model folder if not exists
os.makedirs('model', exist_ok=True)

# Save model + vectorizer
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("✅ Model trained and saved!")