import pickle

# Load model
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

def predict_email(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "PHISHING 🚨" if result == 1 else "LEGIT ✅"

# Test cases
if __name__ == "__main__":
    test_email = input("Enter email text: ")
    print("Prediction:", predict_email(test_email))