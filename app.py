from flask import Flask, request, jsonify, render_template
import joblib
import os
import pandas as pd
from data_preprocessing import preprocess_text


# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Lazy load model and vectorizer on first request
model = None
vectorizer = None

def load_models():
    global model, vectorizer
    if model is None or vectorizer is None:
        model = joblib.load('sentiment_model_new.pkl')
        vectorizer = joblib.load('tfidf_vectorizer_new.pkl')
        print(f"Loaded model: {type(model).__name__}, Vectorizer: {type(vectorizer).__name__}")

def process_reviews_in_chunks(reviews, chunk_size=1000):
    """Process reviews in chunks to manage memory."""
    review_list = [r.strip() for r in reviews.split(',') if r.strip()] if isinstance(reviews, str) else reviews
    from data_preprocessing import preprocess_text  # Lazy import
    cleaned_reviews = [preprocess_text(r) for r in review_list]
    for i in range(0, len(cleaned_reviews), chunk_size):
        chunk = cleaned_reviews[i:i + chunk_size]
        yield vectorizer.transform(chunk).toarray()  # Vectorize only when needed

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        print("Received form data:", dict(request.form))  # Debug print
        print("Received files:", request.files)  # Debug print
        reviews = request.form.get('reviews', '').strip()  # Changed from 'review' to 'reviews'
        file = request.files.get('file')
        
        if file and file.filename:
            print(f"File uploaded: {file.filename}, size: {file.content_length} bytes")
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file, encoding='ISO-8859-1', nrows=10000)
                    print(f"CSV columns: {df.columns.tolist()}")
                    if 'Review' not in df.columns:
                        return render_template('index.html', error="No 'Review' column found. Available columns: " + str(df.columns.tolist()), result=None)
                    reviews = ','.join(df['Review'].dropna().astype(str))
                    print(f"Extracted reviews: {reviews[:200]}..." if reviews else "No reviews extracted")
                    if not reviews:
                        return render_template('index.html', error="File is empty or no reviews in 'Review' column.", result=None)
                elif file.filename.endswith('.txt'):
                    encodings = ['utf-8', 'utf-16', 'ISO-8859-1', 'latin1']
                    for enc in encodings:
                        try:
                            content = file.read().decode(enc)
                            reviews = content.replace('\n', ',')
                            print(f"Successfully decoded with {enc}: {reviews[:200]}...")
                            break
                        except UnicodeDecodeError:
                            print(f"Failed to decode with {enc}")
                            if enc == encodings[-1]:
                                return render_template('index.html', error="Unable to decode file. Try saving as UTF-8 or ISO-8859-1.", result=None)
                    if not reviews.strip():
                        return render_template('index.html', error="File is empty or contains no valid text.", result=None)
                else:
                    return render_template('index.html', error="Unsupported file type. Use CSV or TXT.", result=None)
            except Exception as e:
                return render_template('index.html', error=f"Error reading file: {str(e)}", result=None)
        elif not reviews:
            return render_template('index.html', error="No review or file provided.", result=None)
        
        print(f"Processing reviews: {len([r for r in reviews.split(',') if r.strip()])} reviews")
        sentiments = []
        try:
            load_models()
            from generative import generate_suggestion
            for chunk_predictions in process_reviews_in_chunks(reviews):
                sentiments.extend(model.predict(chunk_predictions))
            suggestions = [generate_suggestion(sent) for sent in sentiments]
        except Exception as e:
            return render_template('index.html', error=f"Error processing reviews: {str(e)}", result=None)
        
        review_list = [r.strip() for r in reviews.split(',') if r.strip()]
        results = [{'review': rev, 'sentiment': int(sent), 'suggestion': sugg} 
                   for rev, sent, sugg in zip(review_list, sentiments, suggestions)]
        return render_template('index.html', error=None, result=results)
    
    return render_template('index.html', error=None, result=None)


@app.route('/analyze', methods=['POST'])
def analyze_review():
    data = request.get_json()
    reviews = data.get('reviews', data.get('review_text', '')).strip()
    if not reviews:
        return jsonify({'error': 'No review provided'}), 400
    
    sentiments = []
    try:
        load_models()  # Load models on demand
        from generative import generate_suggestion  # Lazy import
        from data_preprocessing import preprocess_text  # Lazy import
        for chunk_predictions in process_reviews_in_chunks(reviews):
            sentiments.extend(model.predict(chunk_predictions))
        suggestions = [generate_suggestion(sent) for sent in sentiments]
    except Exception as e:
        return jsonify({'error': f"Error processing reviews: {str(e)}"}), 500
    
    review_list = [r.strip() for r in reviews.split(',') if r.strip()]
    results = [{'review': rev, 'sentiment': int(sent), 'suggestion': sugg} 
               for rev, sent, sugg in zip(review_list, sentiments, suggestions)]
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)