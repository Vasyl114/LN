"""
03_data_augmentation.py
========================
Phase 2 — Data Augmentation for Group 26 NLP Project.

WHAT THIS DOES:
    1. Loads and cleans the original training data
    2. Creates a held-out development set (from original data only)
    3. Evaluates a baseline model on that dev set BEFORE augmentation
    4. Generates augmented records for the 4 minority classes using 15 techniques
       (beneficial + creative + intentional noise)
    5. Evaluates the same model on the same dev set AFTER clean augmentation
    6. Evaluates again with noisy augmentation (for comparison)
    7. Prints a full summary table
    8. Saves two output files

OUTPUT FILES (saved to the same folder as this script):
    train_augmented.csv              -- original + clean augmented records
    train_augmented_with_noise.csv   -- original + all augmented (including noise)

HOW TO RUN:
    From the 26/code/ directory:
        python3 03_data_augmentation.py
"""

import pandas as pd
import numpy as np
import random
import re
import os
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data (only runs once)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_PATH   = '../../StudentsPack/train.csv'
RANDOM_SEED = 42
TARGET_COUNT = 400   # Target samples per minority class after augmentation

# ─────────────────────────────────────────────────────────────
# MEDICAL SYNONYM MAP
# Curated pairs of clinically equivalent terms.
# Safer than raw WordNet for medical text — we never swap a medical
# procedure name for something incorrect.
# ─────────────────────────────────────────────────────────────
MEDICAL_SYNONYMS = {
    'patient':      ['individual', 'subject', 'person'],
    'patients':     ['individuals', 'subjects', 'persons'],
    'physician':    ['doctor', 'clinician', 'practitioner', 'provider'],
    'physicians':   ['doctors', 'clinicians', 'practitioners'],
    'examination':  ['evaluation', 'assessment', 'workup'],
    'examined':     ['evaluated', 'assessed'],
    'normal':       ['unremarkable', 'intact', 'within normal limits'],
    'abnormal':     ['irregular', 'atypical', 'aberrant'],
    'history':      ['background', 'clinical history', 'past history'],
    'diagnosis':    ['impression', 'clinical impression'],
    'complaint':    ['presenting concern', 'chief concern'],
    'procedure':    ['intervention', 'operation'],
    'treatment':    ['management', 'therapy'],
    'admitted':     ['hospitalised', 'brought in'],
    'symptoms':     ['manifestations', 'clinical features'],
    'symptom':      ['manifestation', 'clinical feature'],
    'chronic':      ['longstanding', 'persistent', 'ongoing'],
    'acute':        ['sudden-onset', 'new-onset'],
    'severe':       ['significant', 'marked', 'pronounced'],
    'mild':         ['slight', 'minor', 'modest'],
    'moderate':     ['intermediate', 'medium-grade'],
    'significant':  ['notable', 'considerable', 'meaningful'],
    'discussed':    ['reviewed', 'addressed'],
    'performed':    ['conducted', 'carried out', 'undertaken'],
    'revealed':     ['demonstrated', 'showed', 'indicated'],
    'noted':        ['observed', 'documented', 'recorded'],
    'developed':    ['experienced', 'exhibited'],
    'presented':    ['arrived', 'came in'],
}

# Filler words that can be removed without changing clinical meaning
FILLER_WORDS = {
    'very', 'quite', 'generally', 'typically', 'usually', 'often',
    'commonly', 'particularly', 'essentially', 'basically', 'simply',
    'certainly', 'clearly', 'obviously', 'indeed', 'actually',
}

# Medical abbreviation expansions
ABBREVIATIONS = {
    r'\bpt\b':    'patient',
    r'\bpts\b':   'patients',
    r'\bdx\b':    'diagnosis',
    r'\bhx\b':    'history',
    r'\brx\b':    'prescription',
    r'\bcc\b':    'chief complaint',
    r'\bsx\b':    'symptoms',
    r'\bwnl\b':   'within normal limits',
    r'\bprn\b':   'as needed',
    r'\bbid\b':   'twice daily',
    r'\btid\b':   'three times daily',
    r'\bpo\b':    'by mouth',
}

# Small numbers → words
NUMBER_TO_WORD = {
    '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
}


# ─────────────────────────────────────────────────────────────
# HELPER: Identify medical terms that must never be modified
# ─────────────────────────────────────────────────────────────
def is_protected(word):
    """
    Returns True if a word should never be modified.
    Protects: ALL-CAPS abbreviations, numeric tokens, dosages,
    spinal levels (C3-C4), and hyphenated compound terms.
    """
    if word.isupper() and len(word) > 1:
        return True
    if re.match(r'^[\d.,/%]+$', word):
        return True
    if re.match(r'^\d+[a-z]+$', word.lower()):  # e.g. 30mg, 5cc
        return True
    if re.match(r'^[CLSTclst]\d[-\u2013][CLSTclst]\d$', word):  # C3-C4
        return True
    if '-' in word and len(word) > 6:  # hyphenated medical compound
        return True
    return False


# ─────────────────────────────────────────────────────────────
# AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def synonym_replace(text, ratio=0.10, seed=42):
    """
    Replace ~ratio of replaceable words with clinical synonyms.
    Uses the curated medical synonym map first, then WordNet for general words.
    Protected words (medical terms, abbreviations) are never touched.
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return text
    rng = random.Random(seed)
    words = text.split()
    n_replace = max(1, int(len(words) * ratio))

    # Find words eligible for medical synonym replacement
    med_candidates = [i for i, w in enumerate(words)
                      if not is_protected(w) and w.lower() in MEDICAL_SYNONYMS]

    # Find words eligible for WordNet replacement (general English)
    wn_candidates = [i for i, w in enumerate(words)
                     if not is_protected(w)
                     and w.lower() not in MEDICAL_SYNONYMS
                     and w.isalpha() and len(w) > 3]

    changed = set()

    # Apply medical synonyms first (highest quality)
    for idx in rng.sample(med_candidates, min(n_replace, len(med_candidates))):
        synonyms = MEDICAL_SYNONYMS[words[idx].lower()]
        replacement = rng.choice(synonyms)
        if words[idx][0].isupper():
            replacement = replacement.capitalize()
        words[idx] = replacement
        changed.add(idx)

    # Fill remaining quota with WordNet synonyms
    remaining = n_replace - len(changed)
    if remaining > 0 and wn_candidates:
        for idx in rng.sample(wn_candidates, min(remaining, len(wn_candidates))):
            synsets = wordnet.synsets(words[idx].lower())
            candidates = []
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name.lower() != words[idx].lower() and name.isalpha():
                        candidates.append(name)
            if candidates:
                replacement = rng.choice(candidates[:5])
                if words[idx][0].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement

    return ' '.join(words)


def random_deletion(text, ratio=0.10, seed=42):
    """
    Remove a random fraction of non-protected words.
    Very short texts (≤5 words) are never touched.
    Applied mainly to the transcription field.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return text
    rng = random.Random(seed)
    words = text.split()
    if len(words) <= 5:
        return text
    n_delete = max(1, int(len(words) * ratio))
    deletable = [i for i, w in enumerate(words) if not is_protected(w)]
    to_delete = set(rng.sample(deletable, min(n_delete, len(deletable))))
    return ' '.join(w for i, w in enumerate(words) if i not in to_delete)


def random_swap(text, n=1, seed=42):
    """Swap n pairs of adjacent words. Creates light word-order variation."""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return text
    rng = random.Random(seed)
    words = text.split()
    if len(words) < 3:
        return text
    for _ in range(n):
        i = rng.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
    return ' '.join(words)


def compress_text(text):
    """
    Remove filler words to create a shorter, more telegraphic version.
    E.g.: "The patient is very likely to..." → "The patient is likely to..."
    """
    if not isinstance(text, str):
        return text
    words = text.split()
    compressed = [w for w in words if w.lower() not in FILLER_WORDS]
    return ' '.join(compressed) if compressed else text


def expand_abbreviations(text):
    """
    Expand common medical shorthand.
    E.g.: "pt presents with dx of..." → "patient presents with diagnosis of..."
    """
    if not isinstance(text, str):
        return text
    for pattern, expansion in ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def case_lower(text):
    """Lowercase the entire text (simulates hastily typed note)."""
    return text.lower() if isinstance(text, str) else text


def case_no_period(text):
    """Remove trailing period (a common real-world variation)."""
    if isinstance(text, str) and text.endswith('.'):
        return text[:-1]
    return text


def number_variation(text):
    """
    Replace standalone small numbers with their word equivalents.
    E.g.: "3 lesions" → "three lesions"
    """
    if not isinstance(text, str):
        return text
    for num, word in NUMBER_TO_WORD.items():
        text = re.sub(r'\b' + re.escape(num) + r'\b', word, text)
    return text


def sentence_deletion(text, seed=42):
    """
    Delete 1–2 random sentences from a longer text.
    Only applied to the transcription field. Texts with ≤3 sentences are untouched.
    Simulates an incomplete or truncated clinical note.
    """
    if not isinstance(text, str):
        return text
    sentences = re.split(r'(?<=[.!?,])\s+', text)
    if len(sentences) <= 3:
        return text
    rng = random.Random(seed)
    n_delete = 1 if len(sentences) < 8 else 2
    keep = sorted(rng.sample(range(len(sentences)), len(sentences) - n_delete))
    return ' '.join(sentences[i] for i in keep)


def keyword_subset(keywords, keep_ratio=0.75, seed=42):
    """
    Keep a random subset of keywords, shuffled.
    Empty keyword fields are left as-is.
    """
    if not isinstance(keywords, str) or len(keywords.strip()) < 3:
        return keywords
    rng = random.Random(seed)
    kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
    if len(kw_list) <= 2:
        return keywords
    n_keep = max(2, int(len(kw_list) * keep_ratio))
    kept = rng.sample(kw_list, n_keep)
    rng.shuffle(kept)
    return ', '.join(kept)


def heavy_noise(text, seed=42):
    """
    Intentional heavy noise for the noise experiment:
    - 20% word deletion
    - 3 random word swaps
    - Lowercase entire text
    This deliberately degrades the text to test model robustness.
    NOT used in the clean augmented dataset.
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return text
    text = random_deletion(text, ratio=0.20, seed=seed)
    text = random_swap(text, n=3, seed=seed + 1)
    text = text.lower()
    return text


# ─────────────────────────────────────────────────────────────
# AUGMENTATION VARIANT DEFINITIONS
# 15 distinct transformations — each applied to create one new record
# from one original record.
# ─────────────────────────────────────────────────────────────
def get_clean_variants(seed_base=RANDOM_SEED):
    """
    Returns a list of variant configs. Each is a dict with:
      name:     label for this variant
      desc_fn:  transform to apply to the description field
      trans_fn: transform to apply to the transcription field
      kw_fn:    transform to apply to the keywords field
    """
    s = seed_base
    return [
        # ── Synonym-based (beneficial) ──
        {'name': 'syn_light',
         'desc_fn':  lambda t: synonym_replace(t, ratio=0.10, seed=s),
         'trans_fn': lambda t: synonym_replace(t, ratio=0.10, seed=s+1),
         'kw_fn':    lambda k: k},

        {'name': 'syn_medium',
         'desc_fn':  lambda t: synonym_replace(t, ratio=0.20, seed=s+2),
         'trans_fn': lambda t: synonym_replace(t, ratio=0.20, seed=s+3),
         'kw_fn':    lambda k: k},

        # ── Deletion-based (applied only to transcription to protect description) ──
        {'name': 'del_light_trans',
         'desc_fn':  lambda t: t,
         'trans_fn': lambda t: random_deletion(t, ratio=0.10, seed=s+4),
         'kw_fn':    lambda k: keyword_subset(k, keep_ratio=0.85, seed=s)},

        {'name': 'del_medium_trans',
         'desc_fn':  lambda t: t,
         'trans_fn': lambda t: random_deletion(t, ratio=0.20, seed=s+5),
         'kw_fn':    lambda k: keyword_subset(k, keep_ratio=0.70, seed=s)},

        # ── Swap-based (light word-order variation) ──
        {'name': 'swap_light',
         'desc_fn':  lambda t: random_swap(t, n=1, seed=s+6),
         'trans_fn': lambda t: random_swap(t, n=2, seed=s+7),
         'kw_fn':    lambda k: k},

        # ── Combination: synonym + swap ──
        {'name': 'combo_syn_swap',
         'desc_fn':  lambda t: synonym_replace(random_swap(t, n=1, seed=s+8), ratio=0.10, seed=s+9),
         'trans_fn': lambda t: synonym_replace(t, ratio=0.10, seed=s+10),
         'kw_fn':    lambda k: keyword_subset(k, keep_ratio=0.80, seed=s)},

        # ── Combination: synonym + deletion ──
        {'name': 'combo_syn_del',
         'desc_fn':  lambda t: compress_text(synonym_replace(t, ratio=0.15, seed=s+11)),
         'trans_fn': lambda t: random_deletion(synonym_replace(t, ratio=0.15, seed=s+12), ratio=0.10, seed=s+13),
         'kw_fn':    lambda k: keyword_subset(k, keep_ratio=0.75, seed=s)},

        # ── Lowercase (simulates rushed or informal note entry) ──
        {'name': 'case_lower',
         'desc_fn':  lambda t: case_lower(synonym_replace(t, ratio=0.10, seed=s+14)),
         'trans_fn': lambda t: case_lower(t),
         'kw_fn':    lambda k: case_lower(k)},

        # ── Abbreviation expansion (longer, more verbose version) ──
        {'name': 'abbrev_expand',
         'desc_fn':  lambda t: expand_abbreviations(t),
         'trans_fn': lambda t: expand_abbreviations(t),
         'kw_fn':    lambda k: k},

        # ── Compressed version (shorter, telegraphic) ──
        {'name': 'compress',
         'desc_fn':  lambda t: compress_text(t),
         'trans_fn': lambda t: compress_text(random_deletion(t, ratio=0.05, seed=s+15)),
         'kw_fn':    lambda k: keyword_subset(k, keep_ratio=0.70, seed=s)},

        # ── Number-to-word variation ──
        {'name': 'num_to_word',
         'desc_fn':  lambda t: number_variation(t),
         'trans_fn': lambda t: t,
         'kw_fn':    lambda k: k},

        # ── Sentence deletion from transcription ──
        {'name': 'sent_del',
         'desc_fn':  lambda t: t,
         'trans_fn': lambda t: sentence_deletion(t, seed=s+16),
         'kw_fn':    lambda k: keyword_subset(k, keep_ratio=0.80, seed=s)},

        # ── No trailing period on description (punctuation variation) ──
        {'name': 'no_period',
         'desc_fn':  lambda t: case_lower(case_no_period(t)),
         'trans_fn': lambda t: synonym_replace(t, ratio=0.10, seed=s+17),
         'kw_fn':    lambda k: k},

        # ── Synonym + sentence deletion (combined creative transform) ──
        {'name': 'syn_sent_del',
         'desc_fn':  lambda t: synonym_replace(t, ratio=0.15, seed=s+18),
         'trans_fn': lambda t: sentence_deletion(synonym_replace(t, ratio=0.10, seed=s+19), seed=s+20),
         'kw_fn':    lambda k: keyword_subset(k, keep_ratio=0.65, seed=s)},

        # ── Compress + swap (shorter and reordered) ──
        {'name': 'compress_swap',
         'desc_fn':  lambda t: random_swap(compress_text(t), n=1, seed=s+21),
         'trans_fn': lambda t: random_deletion(compress_text(t), ratio=0.10, seed=s+22),
         'kw_fn':    lambda k: keyword_subset(k, keep_ratio=0.75, seed=s)},
        # ── Noise variant (intentional) ──
        {'name': 'noise_heavy',
         'desc_fn':  lambda t: heavy_noise(t, seed=s+99),
         'trans_fn': lambda t: heavy_noise(t, seed=s+100),
         'kw_fn':    lambda k: case_lower(k)},
    ]


# ─────────────────────────────────────────────────────────────
# AUGMENTATION ENGINE
# ─────────────────────────────────────────────────────────────
def augment_class(df, class_name, target_count):
    """
    Generate new records for class_name until target_count is reached.
    Cycles through all variants (including noise).
    If the class is already at or above target_count, returns empty DataFrame.
    """
    class_df = df[df['medical_specialty'] == class_name].copy().reset_index(drop=True)
    n_original = len(class_df)
    n_needed = target_count - n_original

    if n_needed <= 0:
        print(f"  [{class_name}] Already at {n_original} records. Skipping.")
        return pd.DataFrame()

    print(f"  [{class_name}] {n_original} → {target_count} | Generating {n_needed} new records...")

    variants = get_clean_variants()

    augmented_rows = []
    row_index = 0
    variant_index = 0

    while len(augmented_rows) < n_needed:
        row = class_df.iloc[row_index % n_original]
        v = variants[variant_index % len(variants)]
        try:
            kw_val = row.get('keywords')
            kw_val_str = '' if pd.isna(kw_val) else str(kw_val)
            new_row = {
                'medical_specialty': row['medical_specialty'],
                'description':       v['desc_fn'](str(row.get('description', ''))),
                'sample_name':       row.get('sample_name', ''),
                'transcription':     v['trans_fn'](str(row.get('transcription', ''))),
                'keywords':          v['kw_fn'](kw_val_str) if kw_val_str else np.nan,
            }
            # Skip if the description became empty or too short
            if len(str(new_row['description']).split()) >= 3:
                augmented_rows.append(new_row)
        except Exception:
            pass  # Silently skip any row that errors

        row_index += 1
        if row_index % n_original == 0:
            variant_index += 1  # Move to next variant after cycling all original rows

    return pd.DataFrame(augmented_rows[:n_needed])


# ─────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────
def evaluate_on_dev(train_df, dev_df, label=''):
    """
    Train a Logistic Regression + TF-IDF model on train_df,
    evaluate on dev_df, and print accuracy + per-class F1 for priority classes.
    This is the same kind of model as Model 2 in the project.
    """
    def combine(df):
        return (df['description'].fillna('') + ' ' +
                df['sample_name'].fillna('') + ' ' +
                df['keywords'].fillna(''))

    X_train = combine(train_df)
    y_train = train_df['medical_specialty']
    X_dev   = combine(dev_df)
    y_dev   = dev_df['medical_specialty']

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_tr_tfidf = vec.fit_transform(X_train)
    X_dv_tfidf = vec.transform(X_dev)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                             class_weight='balanced')
    clf.fit(X_tr_tfidf, y_train)
    y_pred = clf.predict(X_dv_tfidf)

    acc    = accuracy_score(y_dev, y_pred)
    report = classification_report(y_dev, y_pred, zero_division=0, output_dict=True)

    print(f"  {label}")
    print(f"    Overall Accuracy : {acc*100:.2f}%")
    print(f"    Macro F1         : {report['macro avg']['f1-score']*100:.2f}%")
    for cls in ['Dermatology', 'Psychiatry-Psychology', 'Ophthalmology', 'Neurosurgery']:
        f1  = report.get(cls, {}).get('f1-score', 0.0)
        sup = report.get(cls, {}).get('support', 0)
        print(f"    {cls:<30} F1={f1*100:.1f}%  (support={sup})")

    return acc, report


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 2 — DATA AUGMENTATION")
    print("=" * 60)

    # ── Load and clean ──
    df = pd.read_csv(DATA_PATH, sep=';', quotechar='"', engine='python')
    df = df[df['medical_specialty'].str.len() < 50]

    # Remove Neurosurgery rows where description leaked transcription content
    mask = ~((df['medical_specialty'] == 'Neurosurgery') &
             (df['description'].fillna('').apply(lambda x: len(str(x).split())) > 200))
    df = df[mask].copy()

    # Remove classes with only 1 sample (can't stratify)
    counts = df['medical_specialty'].value_counts()
    df = df[df['medical_specialty'].isin(counts[counts > 1].index)].copy()
    print(f"\nTotal records after cleaning : {len(df)}")
    print(f"Number of classes            : {df['medical_specialty'].nunique()}")

    # ── Dev split (20% from original data only — never augmented) ──
    train_orig, dev_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED,
        stratify=df['medical_specialty']
    )
    print(f"Train (original)             : {len(train_orig)}")
    print(f"Dev (held out, never touched): {len(dev_df)}")

    # ── BASELINE evaluation (no augmentation) ──
    print("\n" + "=" * 60)
    print("STEP 1: Baseline (no augmentation)")
    print("=" * 60)
    acc_base, _ = evaluate_on_dev(train_orig, dev_df, label='No augmentation:')

    # ── Generate AUGMENTED records ──
    print("\n" + "=" * 60)
    print("STEP 2: Generating augmented records for all classes < 400")
    print("=" * 60)
    
    # Augment all classes that are below the TARGET_COUNT
    classes_to_augment = df['medical_specialty'].value_counts()
    classes_to_augment = classes_to_augment[classes_to_augment < TARGET_COUNT].index.tolist()
    
    aug_parts = []
    for cls in classes_to_augment:
        aug = augment_class(train_orig, cls, TARGET_COUNT)
        if not aug.empty:
            aug_parts.append(aug)

    aug_df = pd.concat(aug_parts, ignore_index=True) if aug_parts else pd.DataFrame()
    train_aug = (pd.concat([train_orig, aug_df], ignore_index=True)
                 .sample(frac=1, random_state=RANDOM_SEED)
                 .reset_index(drop=True))

    # ── Evaluate with augmentation ──
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation with all augmentation variants")
    print("=" * 60)
    acc_aug, _ = evaluate_on_dev(train_aug, dev_df, label='With augmentation:')

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY — DEV SET COMPARISON")
    print("=" * 60)
    print(f"  No augmentation : {acc_base*100:.2f}%")
    print(f"  Augmentation    : {acc_aug*100:.2f}%   (Δ {(acc_aug - acc_base)*100:+.2f}%)")
    print()

    # ── Class distribution before/after ──
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION — BEFORE vs AFTER")
    print("=" * 60)
    before = train_orig['medical_specialty'].value_counts()

    # The final file combines ALL original data (not just 80%) + augmented
    full_aug = pd.concat([df, aug_df], ignore_index=True)

    after = full_aug['medical_specialty'].value_counts()
    for cls in after.sort_values(ascending=False).index:
        b = before.get(cls, 0)
        a = after.get(cls, 0)
        bar = '█' * min(30, a // 20)
        print(f"  {cls:<30} {b:>4} → {a:>4}  {bar}")

    # ── Save output files ──
    print("\n" + "=" * 60)
    print("SAVING OUTPUT FILES")
    print("=" * 60)

    full_aug = full_aug.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    full_aug.to_csv('train_augmented.csv', sep=';', quotechar='"',
                      quoting=1, index=False)
    print(f"  Saved: train_augmented.csv          ({len(full_aug)} total records)")
    print("\nDone. Use train_augmented.csv for all model training in Phases 3–4.")


if __name__ == '__main__':
    main()
