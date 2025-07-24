import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from generative import generate_suggestion
from sklearn.utils import resample

def preprocess_text(text):
    # Handle non-string values
    if not isinstance(text, str) or pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)
      
          
# Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load and preprocess data
data = pd.read_csv('reviews.csv', encoding='ISO-8859-1', low_memory=False)
data['cleaned_review'] = data['Review'].apply(preprocess_text)
# Map sentiments and balance dataset
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
data['Sentiment'] = data['Sentiment'].str.lower().map(sentiment_map).fillna(-1)
data = data[data['Sentiment'] != -1]
print("Initial sentiment distribution:", data['Sentiment'].value_counts())

# Determine available counts and sample
min_count = min(data['Sentiment'].value_counts())
data_positive = data[data['Sentiment'] == 2].sample(n=min_count, random_state=42)
data_negative = data[data['Sentiment'] == 0].sample(n=min_count, random_state=42)
data_neutral = data[data['Sentiment'] == 1].sample(n=min_count, random_state=42)
data_balanced = pd.concat([data_positive, data_negative, data_neutral])
print("Balanced sentiment distribution:", data_balanced['Sentiment'].value_counts())

# vectorize and train
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data_balanced['cleaned_review']).toarray()
y = data_balanced['Sentiment']

# Train a Classifier using Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'sentiment_model_new.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_new.pkl')

# Manual test
# test_reviews = ["The product was great!", "Terrible experience", "The product was okay"]
# test_cleaned = [preprocess_text(r) for r in test_reviews]
# test_X = vectorizer.transform(test_cleaned).toarray()
# test_pred = model.predict(test_X)
# print("Manual test predictions:", test_pred)
