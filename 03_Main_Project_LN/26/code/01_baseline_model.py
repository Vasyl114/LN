import sys
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
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_tokens)

def get_gold_validation_set():
    """
    Always returns the exact same 20% of the original training data.
    This prevents 'data leakage' when testing augmented datasets.
    """
    orig_path = '../../StudentsPack/train.csv'
    try:
        df = pd.read_csv(orig_path, sep=';', quotechar='"', engine='python')
    except FileNotFoundError:
        print(f"Error: Could not find {orig_path}.")
        sys.exit(1)
        
    df = df[df['medical_specialty'].str.len() < 50]
    mask = ~((df['medical_specialty'] == 'Neurosurgery') &
             (df['description'].fillna('').apply(lambda x: len(str(x).split())) > 200))
    df = df[mask].copy()
    
    counts = df['medical_specialty'].value_counts()
    valid_classes = counts[counts > 1].index
    df = df[df['medical_specialty'].isin(valid_classes)]
    
    _, dev_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['medical_specialty'])
    return dev_df

def main():
    # Allow passing a different dataset via terminal (e.g. python3 01_baseline_model.py train_augmented.csv)
    data_path = sys.argv[1] if len(sys.argv) > 1 else '../../StudentsPack/train.csv'
    print(f"Loading training data from {data_path}...")
    
    try:
        train_df = pd.read_csv(data_path, sep=';', quotechar='"', engine='python')
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}.")
        return
        
    # Clean the training data (same rules as validation set) to prevent errors
    train_df = train_df[train_df['medical_specialty'].str.len() < 50]
    mask = ~((train_df['medical_specialty'] == 'Neurosurgery') &
             (train_df['description'].fillna('').apply(lambda x: len(str(x).split())) > 200))
    train_df = train_df[mask].copy()
    counts = train_df['medical_specialty'].value_counts()
    valid_classes = counts[counts > 1].index
    train_df = train_df[train_df['medical_specialty'].isin(valid_classes)]
    
    # Get the fixed, pure validation set
    val_df = get_gold_validation_set()
    
    # If using an augmented dataset, we must remove any rows from training that are 
    # exact copies of the validation records to prevent data leakage.
    if data_path != '../../StudentsPack/train.csv':
        val_descs = set(val_df['description'].fillna('').tolist())
        train_df = train_df[~train_df['description'].isin(val_descs)]
    else:
        # If running on original data, just do the normal split so train doesn't overlap val
        train_df, _ = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['medical_specialty'])
        
    print(f"Training on {len(train_df)} records...")
    print(f"Validating on {len(val_df)} original pure records...")
    
    X_train_raw = train_df['description'].fillna('')
    y_train = train_df['medical_specialty']
    
    X_val_raw = val_df['description'].fillna('')
    y_val = val_df['medical_specialty']
    
    print("Preprocessing text (lowercasing and stemming)... this might take a few seconds.")
    X_train_processed = X_train_raw.apply(preprocess_text)
    X_val_processed = X_val_raw.apply(preprocess_text)
    
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_processed)
    X_val_tfidf = vectorizer.transform(X_val_processed)
    
    print("Training the Baseline Model (Multinomial Naive Bayes)...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating the model on the validation set...")
    y_pred = model.predict(X_val_tfidf)
    
    accuracy = accuracy_score(y_val, y_pred)
    print("\n" + "="*40)
    print(f"OVERALL ACCURACY: {accuracy * 100:.2f}%")
    print("="*40 + "\n")
    
    print("Detailed Classification Report (Pay attention to the 'f1-score' column):")
    print(classification_report(y_val, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
