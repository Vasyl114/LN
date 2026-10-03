"""
data_utils.py
=============
Shared data loading and splitting for Group 26 NLP Project.

PURPOSE:
    Every script must work on exactly the same records and on exactly the
    same split, so both live here and are imported by the other scripts.

    The file given by the professors is never modified. It contains 7 rows
    whose 'description' cell absorbed the records that followed it (an opening
    quote that only closes several lines later). Three of those cells are cut
    at 32,759 characters and the rest of the cut record spills into the next
    row. load_train() repairs this in memory:
      - the first line of the cell stays as that row's description
      - every absorbed line becomes a record of its own again
      - a cut record is rejoined with its tail (one character is lost at the join)

    THE SPLIT (split_train_heldout):
      - the original records are divided once: 80% for training, 20% held out
      - the division is stratified: every specialty keeps its proportion
      - the random seed is fixed, so the split is identical in every script
      - the held-out records are only used to measure a model, never to train
        it and never as a source for data augmentation

HOW TO USE:
    from data_utils import load_train, split_train_heldout
    df = load_train()
    train_df, heldout_df = split_train_heldout(df)
"""

import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_PATH   = '../../StudentsPack/train.csv'
RANDOM_SEED  = 42
HELDOUT_SIZE = 0.20   # Share of the original records kept aside for evaluation

COLUMNS = ['medical_specialty', 'description', 'sample_name',
           'transcription', 'keywords']

VALID_LABELS = [
    'Surgery', 'Cardiovascular-Pulmonary', 'Orthopedic', 'Radiology',
    'General Medicine', 'Gastroenterology', 'Neurology',
    'Obstetrics-Gynecology', 'Neurosurgery', 'Ophthalmology',
    'Psychiatry-Psychology', 'Dermatology',
]


def _clean_record(fields):
    """Turn a list of up to 5 field values into a record dict (empty → NaN)."""
    fields = (list(fields) + [np.nan] * 5)[:5]
    record = dict(zip(COLUMNS, fields))
    record['medical_specialty'] = record['medical_specialty'].strip()
    for col in COLUMNS[1:]:
        if isinstance(record[col], str) and record[col].strip() == '':
            record[col] = np.nan
    return record


def load_train(path=TRAIN_PATH, verbose=False):
    """
    Load train.csv and return a clean DataFrame with one record per row.
    Rows are kept in file order. With verbose=True, prints what was repaired.
    Any other file with the same five columns (e.g. train_augmented.csv)
    can be loaded the same way; a file without broken rows is returned as is.
    """
    raw = pd.read_csv(path, sep=';', quotechar='"', engine='python')
    rows = raw.to_dict('records')

    records = []
    n_broken, n_recovered, n_rejoined, n_dropped = 0, 0, 0, 0

    i = 0
    while i < len(rows):
        row = rows[i]
        desc = row['description']

        # A tail that could not be attached to any record
        if row['medical_specialty'] not in VALID_LABELS:
            n_dropped += 1
            i += 1
            continue

        # Ordinary row: keep as is
        if not (isinstance(desc, str) and '\n' in desc):
            records.append(_clean_record([row[c] for c in COLUMNS]))
            i += 1
            continue

        # Broken row: the description cell holds several lines
        n_broken += 1
        pieces = [p.rstrip('\r') for p in desc.split('\n')]
        records.append(_clean_record([row['medical_specialty'], pieces[0]]))

        for j, piece in enumerate(pieces[1:], start=1):
            fields = next(csv.reader([piece], delimiter=';', quotechar='"'))

            if j == len(pieces) - 1:
                # The last absorbed record is incomplete. Its remaining fields
                # are either in the other cells of this row, or in the next row.
                rest = [row[c] for c in COLUMNS[2:] if isinstance(row[c], str)]
                if rest:
                    # the cell was closed by a quote inside this record
                    if fields[-1].count('"') % 2 == 1:
                        fields[-1] = fields[-1].rstrip('"')
                    fields = fields + rest
                elif (i + 1 < len(rows)
                      and rows[i + 1]['medical_specialty'] not in VALID_LABELS):
                    tail = rows[i + 1]
                    fields[-1] = fields[-1] + tail['medical_specialty']
                    if isinstance(tail['description'], str):
                        fields.append(tail['description'])
                    n_rejoined += 1
                    i += 1  # the tail row is consumed

            if fields[0].strip() in VALID_LABELS and len(fields) <= 5:
                records.append(_clean_record(fields))
                n_recovered += 1
            else:
                n_dropped += 1
        i += 1

    df = pd.DataFrame(records, columns=COLUMNS)

    if verbose:
        print(f"  Rows as parsed from the file      : {len(raw)}")
        print(f"  Broken rows (cell absorbed others): {n_broken}")
        print(f"  Records recovered from those cells: {n_recovered}")
        print(f"  Cut records rejoined with a tail  : {n_rejoined}")
        print(f"  Pieces that could not be used     : {n_dropped}")
        print(f"  Records after repair              : {len(df)}")

    return df


def split_train_heldout(df):
    """
    Split the original records into a training portion and a held-out portion.
    Stratified by label, fixed seed. Returns (train_df, heldout_df).
    """
    return train_test_split(df, test_size=HELDOUT_SIZE, random_state=RANDOM_SEED,
                            stratify=df['medical_specialty'])


def remove_heldout(df, heldout_df):
    """
    Return df without the records of the held-out portion.
    A record is removed when all five fields are equal to a held-out record.
    Used to train on a file that contains the whole original data
    (e.g. train_augmented.csv) without training on the evaluation records.
    """
    def as_key(d):
        return d[COLUMNS].fillna('').astype(str).agg('\x1f'.join, axis=1)

    return df[~as_key(df).isin(set(as_key(heldout_df)))]


if __name__ == '__main__':
    # Running this file directly writes the repaired data to train_clean.csv,
    # so the result of the repair can be inspected. The scripts do not need
    # this file: they call load_train() themselves.
    clean = load_train(verbose=True)
    clean.to_csv('train_clean.csv', sep=';', quotechar='"', quoting=1, index=False)
    print(f"  Saved: train_clean.csv ({len(clean)} records)")
