import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Sample data: student free responses with labels
data = {
    "response": [
        "I feel confident using AI tools for my studies",
        "AI makes studying easier and more interesting",
        "I'm unsure if AI helps my learning",
        "AI is confusing and hard to use",
        "I don't think AI impacts my grades",
        "AI tools have improved my academic performance"
    ],
    "label": ["Positive", "Positive", "Neutral", "Negative", "Neutral", "Positive"]
}

df = pd.DataFrame(data)

# Text vectorization with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['response'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save model and vectorizer for reuse
joblib.dump(clf, 'rf_student_feedback_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Function to classify new responses
def classify_response(text):
    vect_text = vectorizer.transform([text])
    prediction = clf.predict(vect_text)
    return prediction[0]

# Command line interface
if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Input: {text}")
        print(f"Predicted sentiment: {classify_response(text)}")
    else:
        print("Please provide a text input to classify. Example:")
        print("python student_feedback_classifier.py 'AI helps me study better'")
