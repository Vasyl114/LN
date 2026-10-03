"""
02_data_analysis.py
====================
Phase 1 — Data Analysis for Group 26 NLP Project.

PURPOSE:
    This script analyses the training dataset and produces:
      - A class distribution table (printed and saved as a figure)
      - A text length analysis per class (printed and saved as a figure)
      - A top-words analysis for 4 selected classes
      - A count of noisy/incomplete rows
      - The numbers behind the two observations reported in the paper
        (notes listed under several specialties, label inside the keywords)

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
import matplotlib
matplotlib.use('Agg')  # Run without a display (for servers/terminals)
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import os

from data_utils import load_train

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
FIGURES_DIR  = '../figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

# The shared loader repairs the rows whose description cell absorbed the
# records that followed it (see data_utils.py). The given file is not modified.
df = load_train(verbose=True)

print(f"\n  Total rows after cleaning : {len(df)}")
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

# Group by class and compute mean.
# Empty transcriptions are left out of the transcription average
# (counting them as 0 words would pull the average down).
has_trans = df['transcription'].notna()
length_table = pd.DataFrame({
    'Avg Words (description)'  : df.groupby('medical_specialty')['desc_wc'].mean(),
    'Avg Words (transcription)': df[has_trans].groupby('medical_specialty')['trans_wc'].mean(),
}).round(1)
length_table = length_table.sort_values('Avg Words (transcription)', ascending=False)

print(length_table.to_string())
print(f"\nOverall avg description length  : {df['desc_wc'].mean():.1f} words "
      f"(median {df['desc_wc'].median():.0f}, max {df['desc_wc'].max()})")
print(f"Overall avg transcription length: {df.loc[has_trans, 'trans_wc'].mean():.1f} words "
      f"(median {df.loc[has_trans, 'trans_wc'].median():.0f}, non-empty rows only)\n")

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

# Rows with empty transcription / keywords (the loader turns blank cells into NaN)
empty_trans = df['transcription'].isna().sum()
empty_kw    = df['keywords'].isna().sum()

# Rows with very short description (under 5 words)
short_desc = (df['desc_wc'] < 5).sum()

print(f"  Rows with empty/missing transcription : {empty_trans}")
print(f"  Rows with empty/missing keywords      : {empty_kw} ({empty_kw / len(df) * 100:.1f}%)")
print(f"  Rows with description under 5 words  : {short_desc}")
print()


# ─────────────────────────────────────────────────────────────
# STEP 6: THE SAME NOTE LISTED UNDER SEVERAL SPECIALTIES
# (Observation 1 in the paper)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: Notes listed under more than one specialty")
print("=" * 60)

with_trans = df[has_trans]
per_note   = with_trans.groupby('transcription')['medical_specialty'].agg(['nunique', 'count'])
multi_note = per_note[per_note['nunique'] > 1]
in_multi   = with_trans['transcription'].isin(multi_note.index)

print(f"  Rows with a transcription             : {len(with_trans)}")
print(f"  Distinct transcriptions               : {len(per_note)}")
print(f"  Transcriptions under >1 specialty     : {len(multi_note)}")
print(f"  Rows involved                         : {in_multi.sum()} "
      f"({in_multi.sum() / len(with_trans) * 100:.1f}% of rows with a transcription)")

# Do the copies of a note differ in the other fields?
copies = with_trans[in_multi].groupby('transcription')
same_desc = sum(g['description'].nunique(dropna=False) == 1 for _, g in copies)
same_name = sum(g['sample_name'].nunique(dropna=False) == 1 for _, g in copies)
print(f"  ...with identical description         : {same_desc} of {len(multi_note)}")
print(f"  ...with identical sample_name         : {same_name} of {len(multi_note)}")

# Which pairs of specialties share the most notes
pair_counts = Counter()
for _, g in copies:
    labels = sorted(g['medical_specialty'].unique())
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            pair_counts[(labels[a], labels[b])] += 1
print("\n  Most frequent pairs (same transcription, different label):")
for (a, b), n in pair_counts.most_common(6):
    print(f"    {n:>4}  {a} / {b}")

# Share of each class whose note also exists under another label
print("\n  Share of each class also listed under another specialty:")
share = (with_trans.assign(shared=in_multi)
         .groupby('medical_specialty')['shared'].agg(['sum', 'count']))
share['pct'] = (share['sum'] / share['count'] * 100).round(0)
for cls, r in share.sort_values('pct', ascending=False).iterrows():
    print(f"    {cls:<26} {int(r['sum']):>4} of {int(r['count']):>4}  ({r['pct']:.0f}%)")
print()


# ─────────────────────────────────────────────────────────────
# STEP 7: THE LABEL INSIDE THE KEYWORDS FIELD
# (Observation 2 in the paper)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 7: First keyword vs. label")
print("=" * 60)

with_kw  = df[df['keywords'].notna()]
first_kw = with_kw['keywords'].str.split(',').str[0].str.strip().str.lower()
# The keywords write the specialty as e.g. "cardiovascular / pulmonary"
label_as_kw = with_kw['medical_specialty'].str.lower().str.replace('-', ' / ')
n_match = (first_kw == label_as_kw).sum()

print(f"  Rows with keywords                    : {len(with_kw)}")
print(f"  First keyword equals the label        : {n_match} ({n_match / len(with_kw) * 100:.1f}%)")
print(f"  Rows without keywords                 : {empty_kw} ({empty_kw / len(df) * 100:.1f}%)")
example = with_kw.iloc[0]
print(f"  Example: [{example['medical_specialty']}] {example['keywords'][:70]}...")
print()

print("=" * 60)
print("DATA ANALYSIS COMPLETE")
print(f"Figures saved to: {os.path.abspath(FIGURES_DIR)}")
print("Use the numbers above to verify/update the values in 26.tex")
print("=" * 60)
