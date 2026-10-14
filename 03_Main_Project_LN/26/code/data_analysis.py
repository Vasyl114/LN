"""
data_analysis.py: statistics of the training data (Data Analysis section of the paper).

Prints:  class distribution; text lengths per class; top words of four classes; empty and short fields;
         notes listed under several specialties (Observation 1); first keyword versus label (Observation 2).
Reads:   train.csv.
Writes:  ../figures/fig1_text_lengths.png.
Run:     python3 data_analysis.py        (seconds)
"""

import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from data_utils import load_train

FIGURES_DIR = '../figures'


def banner(title, *notes):
    print("=" * 60)
    print(title)
    for note in notes:
        print(note)
    print("=" * 60)


def word_count(text):
    """Number of words in a string; 0 for missing values."""
    return len(text.split()) if isinstance(text, str) else 0


def class_distribution(df):
    banner("STEP 2: Class distribution")
    counts = df['medical_specialty'].value_counts()
    pct = (counts / len(df) * 100).round(1)
    print(pd.DataFrame({'Count': counts, 'Percent': pct}).to_string())
    print(f"\nMost common class  : {counts.idxmax()} ({counts.max()} samples, {pct.max()}%)")
    print(f"Least common class : {counts.idxmin()} ({counts.min()} samples, {pct.min()}%)")
    print(f"Imbalance ratio    : {counts.max() / counts.min():.1f}:1\n")


def text_lengths(df, has_trans):
    """Average words per class for description and transcription; saves the figure."""
    banner("STEP 3: Text length per class (word count)")
    # Empty transcriptions are left out of the transcription average (counting them as 0 would lower it)
    table = pd.DataFrame({
        'Avg Words (description)':   df.groupby('medical_specialty')['desc_wc'].mean(),
        'Avg Words (transcription)': df[has_trans].groupby('medical_specialty')['trans_wc'].mean(),
    }).round(1)
    table = table.sort_values('Avg Words (transcription)', ascending=False)

    print(table.to_string())
    print(f"\nOverall avg description length  : {df['desc_wc'].mean():.1f} words "
          f"(median {df['desc_wc'].median():.0f}, max {df['desc_wc'].max()})")
    print(f"Overall avg transcription length: {df.loc[has_trans, 'trans_wc'].mean():.1f} words "
          f"(median {df.loc[has_trans, 'trans_wc'].median():.0f}, non-empty rows only)\n")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(table))
    width = 0.4
    ax.bar([i - width / 2 for i in x], table['Avg Words (description)'],
           width=width, label='Description', color='#1f77b4')
    ax.bar([i + width / 2 for i in x], table['Avg Words (transcription)'],
           width=width, label='Transcription', color='#ff7f0e', alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(table.index, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Average Word Count')
    ax.set_title('Average Text Length per Field and Specialty')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_text_lengths.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  [SAVED] {path}\n")


def top_words(df):
    """Most frequent description words of the largest and smallest classes and a confusing pair."""
    banner("STEP 4: Top 15 words per class (in 'description' field)",
           "  → Look for: class-specific terms vs. words that appear everywhere")
    for cls in ['Surgery', 'Dermatology', 'Radiology', 'Neurosurgery']:
        descriptions = df[df['medical_specialty'] == cls]['description'].fillna('').tolist()
        text = ' '.join(descriptions).lower()
        vectorizer = CountVectorizer(max_features=15, stop_words='english',
                                     token_pattern=r'[a-zA-Z]{3,}')   # words of 3+ letters
        try:
            counts = vectorizer.fit_transform([text]).toarray()[0]
            ranked = sorted(zip(vectorizer.get_feature_names_out(), counts), key=lambda x: x[1], reverse=True)
            print(f"\n  [{cls}] ({len(descriptions)} samples)")
            print(f"  Top words: {', '.join(f'{w} ({c})' for w, c in ranked)}")
        except Exception as e:
            print(f"  [{cls}] Could not compute: {e}")
    print()


def noisy_rows(df):
    banner("STEP 5: Noisy and incomplete rows")
    empty_trans = df['transcription'].isna().sum()
    empty_kw = df['keywords'].isna().sum()
    short_desc = (df['desc_wc'] < 5).sum()
    print(f"  Rows with empty/missing transcription : {empty_trans}")
    print(f"  Rows with empty/missing keywords      : {empty_kw} ({empty_kw / len(df) * 100:.1f}%)")
    print(f"  Rows with description under 5 words  : {short_desc}")
    print()


def shared_notes(df, has_trans):
    """Observation 1: the same note listed under more than one specialty."""
    banner("STEP 6: Notes listed under more than one specialty")
    with_trans = df[has_trans]
    per_note = with_trans.groupby('transcription')['medical_specialty'].agg(['nunique', 'count'])
    multi_note = per_note[per_note['nunique'] > 1]
    in_multi = with_trans['transcription'].isin(multi_note.index)

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

    # Pairs of specialties that share the most notes
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


def first_keyword_vs_label(df):
    """Observation 2: the first keyword is the specialty itself."""
    banner("STEP 7: First keyword vs. label")
    with_kw = df[df['keywords'].notna()]
    first_kw = with_kw['keywords'].str.split(',').str[0].str.strip().str.lower()
    # The keywords write the specialty as e.g. "cardiovascular / pulmonary"
    label_as_kw = with_kw['medical_specialty'].str.lower().str.replace('-', ' / ')
    n_match = (first_kw == label_as_kw).sum()
    empty_kw = df['keywords'].isna().sum()

    print(f"  Rows with keywords                    : {len(with_kw)}")
    print(f"  First keyword equals the label        : {n_match} ({n_match / len(with_kw) * 100:.1f}%)")
    print(f"  Rows without keywords                 : {empty_kw} ({empty_kw / len(df) * 100:.1f}%)")
    example = with_kw.iloc[0]
    print(f"  Example: [{example['medical_specialty']}] {example['keywords'][:70]}...")
    print()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    banner("STEP 1: Loading data")
    df = load_train(verbose=True)   # repairs the rows that swallowed the following records (see data_utils.py)
    print(f"\n  Total rows after cleaning : {len(df)}")
    print(f"  Number of classes         : {df['medical_specialty'].nunique()}")
    print(f"  Columns                   : {list(df.columns)}\n")

    df['desc_wc'] = df['description'].apply(word_count)
    df['trans_wc'] = df['transcription'].apply(word_count)
    has_trans = df['transcription'].notna()

    class_distribution(df)
    text_lengths(df, has_trans)
    top_words(df)
    noisy_rows(df)
    shared_notes(df, has_trans)
    first_keyword_vs_label(df)

    banner("DATA ANALYSIS COMPLETE", f"Figures saved to: {os.path.abspath(FIGURES_DIR)}")


if __name__ == '__main__':
    main()
