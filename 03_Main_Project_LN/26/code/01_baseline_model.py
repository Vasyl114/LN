import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK data for tokenization (only needed the first time)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def preprocess_text(text):
    """
    Applies lowercasing and stemming to a given text.
    This matches the baseline requirements from the project description.
    """
    # Handle missing values (NaNs)
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase the text
    text = text.lower()
    
    # 2. Tokenize (split text into individual words)
    tokens = word_tokenize(text)
    
    # 3. Stemming (reduce words to their root form, e.g., 'running' -> 'run')
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    # Join the words back into a single string
    return " ".join(stemmed_tokens)

def main():
    # --- STEP 1: LOAD THE DATA ---
    # The data is in the StudentsPack folder. We use sep=';' and quotechar='"' 
    # so pandas handles the multiline text correctly, just like opening it in Excel.
    data_path = '../../StudentsPack/train.csv'
    print(f"Loading data from {data_path}...")
    
    try:
        df = pd.read_csv(data_path, sep=';', quotechar='"', engine='python')
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Make sure the path is correct.")
        return
    
    # --- STEP 2: SELECT FEATURES AND LABELS ---
    # For the baseline model, we only use the 'description' field as our input (X)
    # and the 'medical_specialty' as our target label (y).
    # Filter out rows with very long labels (parsing errors) or empty labels
    df = df[df['medical_specialty'].str.len() < 50]
    
    # Filter out classes that have only 1 sample (cannot stratify split these)
    counts = df['medical_specialty'].value_counts()
    valid_classes = counts[counts > 1].index
    df = df[df['medical_specialty'].isin(valid_classes)]
    
    X_raw = df['description'].fillna('')
    y = df['medical_specialty']
    
    # --- STEP 3: PREPROCESS THE TEXT ---
    print("Preprocessing text (lowercasing and stemming)... this might take a few seconds.")
    X_processed = X_raw.apply(preprocess_text)
    
    # --- STEP 4: SPLIT INTO TRAIN AND VALIDATION SETS ---
    # We keep 20% of the data to evaluate how good our model is.
    # stratify=y ensures the 80/20 split keeps the same class proportions (crucial for imbalanced data!).
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- STEP 5: CONVERT TEXT TO NUMBERS (TF-IDF) ---
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer()
    
    # fit_transform learns the vocabulary from the training set and transforms it.
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # transform only converts the validation set using the already learned vocabulary.
    X_val_tfidf = vectorizer.transform(X_val)
    
    # --- STEP 6: TRAIN THE MODEL ---
    print("Training the Baseline Model (Multinomial Naive Bayes)...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # --- STEP 7: EVALUATE THE MODEL ---
    print("Evaluating the model on the validation set...")
    y_pred = model.predict(X_val_tfidf)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print("\n" + "="*40)
    print(f"OVERALL ACCURACY: {accuracy * 100:.2f}%")
    print("="*40 + "\n")
    
    # Show detailed metrics per medical specialty
    # zero_division=0 prevents warnings if a class is never predicted
    print("Detailed Classification Report (Pay attention to the 'f1-score' column):")
    print(classification_report(y_val, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
