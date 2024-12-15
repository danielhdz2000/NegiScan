import nltk
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Step 1: Setup - Install necessary libraries (if not installed)
# !pip install nltk pandas scikit-learn kaggle

# Step 2: Download dataset from Kaggle if not already present
if not os.path.exists('jigsaw-toxic-comment-classification-challenge.zip'):
    os.system('kaggle competitions download -c jigsaw-toxic-comment-classification-challenge')

if not os.path.exists('jigsaw_data'):
    os.makedirs('jigsaw_data')

# Extract the files from the zip only if train.csv doesn't exist
if not os.path.exists('jigsaw_data/train.csv'):
    import zipfile
    with zipfile.ZipFile('jigsaw-toxic-comment-classification-challenge.zip', 'r') as zip_ref:
        zip_ref.extractall('jigsaw_data')
    
    # Unzip the individual csv files if they are still in zip format
    for zip_file in ['jigsaw_data/train.csv.zip', 'jigsaw_data/test.csv.zip', 'jigsaw_data/test_labels.csv.zip']:
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('jigsaw_data')

# Step 3: Load the Jigsaw Toxic Comment Classification dataset
current_directory = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_directory, 'jigsaw_data', 'train.csv')

# Load the train.csv file
print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)

# Step 4: Data Cleaning
nltk.download("stopwords")

def preprocess_text(text):
    """Tokenize, clean, and remove stopwords from the text."""
    text = re.sub(r"[^a-zA-Z]", " ", str(text))  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words("english"))  # Get stopwords
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Apply preprocessing to the 'comment_text' column
df['cleaned_comment'] = df['comment_text'].apply(preprocess_text)

# We only need the toxic column (toxic = 1, non-toxic = 0)
X = df['cleaned_comment']
y = df['toxic']

# Convert text to feature vectors using CountVectorizer
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(X)

# Step 5: Split Data (Training and Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Choice (Switch between Naive Bayes and Logistic Regression)
model_choice = 'naive_bayes'  # Change to 'logistic_regression' if desired
if model_choice == 'naive_bayes':
    model = MultinomialNB()
else:
    model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Step 7: Prediction Function
def predict_sentiment(text):
    """Predict if the given text is negative or non-negative."""
    cleaned_text = preprocess_text(text)  # Clean the input text
    text_vector = vectorizer.transform([cleaned_text])  # Vectorize the cleaned text
    prediction = model.predict(text_vector)  # Predict using the trained model
    return "Negative" if prediction[0] == 1 else "Non-Negative"

# Test the prediction function with sample texts
print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
print(predict_sentiment("It was a terrible film. I hated it."))
print(predict_sentiment("You are the worst person on this planet!"))
print(predict_sentiment("Thank you for being so kind and thoughtful."))
