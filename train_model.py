import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

print("1. Loading the dataset...")
data = pd.read_csv(r"C:\Users\HP\Documents\test\bug_reports.csv", sep='|')

X = data["Report_Text"]
y = data["Label"]

print("2. Splitting data into Training and Testing batches...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Building the Machine Learning Pipeline...")
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression())

print("4. Training the AI Model...")
model.fit(X_train, y_train)

print("5. Evaluating Accuracy...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("6. Saving the model for Streamlit...")
joblib.dump(model, r"C:\Users\HP\Documents\test\bug_model.pkl")
print("Model saved successfully as bug_model.pkl!")