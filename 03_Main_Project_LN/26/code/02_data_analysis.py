"""
02_data_analysis.py
====================
Phase 1 — Data Analysis for Group 26 NLP Project.

PURPOSE:
    This script analyses the training dataset and produces:
      - A class distribution table (printed and saved as a figure)
      - A text length analysis per class (printed and saved as a figure)
      - A top-words analysis for 3 selected classes
      - A count of noisy/incomplete rows

    All numbers printed here are the raw material for the
    "Data Analysis" subsection of the paper (26.tex).

HOW TO RUN:
    From the 26/code/ directory:
        python3 02_data_analysis.py

OUTPUT FILES:
    ../figures/fig1_class_distribution.png
    ../figures/fig2_text_lengths.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Run without a display (for servers/terminals)
from sklearn.feature_extraction.text import CountVectorizer
import os

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_PATH    = '../../StudentsPack/train.csv'
FIGURES_DIR  = '../figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=';', quotechar='"', engine='python')

# Apply the same cleaning as the baseline model:
# Remove rows where the label is very long (parsing errors)
df = df[df['medical_specialty'].str.len() < 50]

# Remove classes that have only 1 sample
counts_raw = df['medical_specialty'].value_counts()
valid_classes = counts_raw[counts_raw > 1].index
df = df[df['medical_specialty'].isin(valid_classes)]

print(f"  Total rows after cleaning : {len(df)}")
print(f"  Number of classes         : {df['medical_specialty'].nunique()}")
print(f"  Columns                   : {list(df.columns)}\n")


# ─────────────────────────────────────────────────────────────
# STEP 2: CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Class distribution")
print("=" * 60)

# Count samples per class and compute percentage
class_counts = df['medical_specialty'].value_counts()
class_pct    = (class_counts / len(df) * 100).round(1)

dist_table = pd.DataFrame({
    'Count'  : class_counts,
    'Percent': class_pct
})

print(dist_table.to_string())
print(f"\nMost common class  : {class_counts.idxmax()} ({class_counts.max()} samples, {class_pct.max()}%)")
print(f"Least common class : {class_counts.idxmin()} ({class_counts.min()} samples, {class_pct.min()}%)")
print(f"Imbalance ratio    : {class_counts.max() / class_counts.min():.1f}:1\n")

# ── Figure 1: Bar chart of class distribution ──
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#d62728' if v == class_counts.max() else '#1f77b4' for v in class_counts.values]
bars = ax.barh(class_counts.index[::-1], class_counts.values[::-1], color=colors[::-1])

# Add count labels on each bar
for bar, count in zip(bars, class_counts.values[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            str(count), va='center', fontsize=9)

ax.set_xlabel('Number of Samples')
ax.set_title('Class Distribution of Training Set\n(red = majority class)')
ax.set_xlim(0, class_counts.max() + 80)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'fig1_class_distribution.png')
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"  [SAVED] {fig_path}\n")


# ─────────────────────────────────────────────────────────────
# STEP 3: TEXT LENGTH ANALYSIS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Text length per class (word count)")
print("=" * 60)

def word_count(text):
    """Count words in a string. Returns 0 for missing values."""
    if not isinstance(text, str):
        return 0
    return len(text.split())

# Compute word counts for description and transcription fields
df['desc_wc']  = df['description'].apply(word_count)
df['trans_wc'] = df['transcription'].apply(word_count)

# Group by class and compute mean
length_table = df.groupby('medical_specialty')[['desc_wc', 'trans_wc']].mean().round(1)
length_table.columns = ['Avg Words (description)', 'Avg Words (transcription)']
length_table = length_table.sort_values('Avg Words (transcription)', ascending=False)

print(length_table.to_string())
print(f"\nOverall avg description length  : {df['desc_wc'].mean():.1f} words")
print(f"Overall avg transcription length: {df['trans_wc'].mean():.1f} words\n")

# ── Figure 2: Grouped bar chart of text lengths ──
fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(length_table))
width = 0.4

ax.bar([i - width/2 for i in x], length_table['Avg Words (description)'],
       width=width, label='Description', color='#1f77b4')
ax.bar([i + width/2 for i in x], length_table['Avg Words (transcription)'],
       width=width, label='Transcription', color='#ff7f0e', alpha=0.8)

ax.set_xticks(list(x))
ax.set_xticklabels(length_table.index, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Average Word Count')
ax.set_title('Average Text Length per Field and Specialty')
ax.legend()
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'fig2_text_lengths.png')
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"  [SAVED] {fig_path}\n")


# ─────────────────────────────────────────────────────────────
# STEP 4: TOP WORDS PER CLASS
# (Identify discriminating vocabulary and noise words)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Top 15 words per class (in 'description' field)")
print("  → Look for: class-specific terms vs. words that appear everywhere")
print("=" * 60)

# Classes of interest: biggest, smallest, and one "confusing" pair
classes_of_interest = ['Surgery', 'Dermatology', 'Radiology', 'Neurosurgery']

for cls in classes_of_interest:
    subset = df[df['medical_specialty'] == cls]['description'].fillna('').tolist()
    combined_text = ' '.join(subset).lower()

    # Use CountVectorizer to get word frequencies, ignoring very common English words
    vectorizer = CountVectorizer(
        max_features=15,
        stop_words='english',   # Remove 'the', 'a', 'is', etc.
        token_pattern=r'[a-zA-Z]{3,}'  # Only words with 3+ letters
    )
    try:
        X = vectorizer.fit_transform([combined_text])
        freqs = dict(zip(vectorizer.get_feature_names_out(),
                         X.toarray()[0]))
        top = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        words_str = ', '.join([f"{w} ({c})" for w, c in top])
        print(f"\n  [{cls}] ({len(subset)} samples)")
        print(f"  Top words: {words_str}")
    except Exception as e:
        print(f"  [{cls}] Could not compute: {e}")

print()


# ─────────────────────────────────────────────────────────────
# STEP 5: NOISE AND MISSING DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Noisy and incomplete rows")
print("=" * 60)

# Rows with empty transcription
empty_trans = df['transcription'].isna().sum() + (df['transcription'] == '').sum()

# Rows with very short description (under 5 words)
short_desc = (df['desc_wc'] < 5).sum()

print(f"  Rows with empty/missing transcription : {empty_trans}")
print(f"  Rows with description under 5 words  : {short_desc}")
print(f"  Total rows potentially noisy         : {empty_trans + short_desc}")
print()

print("=" * 60)
print("DATA ANALYSIS COMPLETE")
print(f"Figures saved to: {os.path.abspath(FIGURES_DIR)}")
print("Use the numbers above to verify/update the values in 26.tex")
print("=" * 60)
