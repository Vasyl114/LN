"""
data_utils.py: loading, repair and splitting of the data, shared by every script.

The given CSV files are never modified. Seven rows of train.csv have a description cell that swallowed
the records that followed it; load_train() repairs them in memory (2,616 parsed rows become 2,669 records):
the first line of the cell stays as that row's description, every other line becomes a record again, and a
record cut in two is rejoined with its tail.

Functions:
    find_data_file         locate train.csv / test_no_labels.csv (code/data/ or StudentsPack/)
    load_train, load_test  read the two given files (the test file has no header and no labels)
    split_train_heldout    stratified 80/20 split with seed 42 (534 held-out records)
    remove_heldout         drop the held-out records from a training file
    first_keyword_label    specialty named by the first keyword of each record ('none' if there is none)

The held-out records only measure a model: they never train it and never source augmentation.
Run `python3 data_utils.py` to print what the repair of train.csv does.
"""

import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# The two files given by the teachers are looked for next to the scripts (data/, the layout of the
# submitted zip) and in StudentsPack/ (the layout of the project folder).
DATA_FOLDERS = ['data', '../../StudentsPack', '../StudentsPack']


def find_data_file(name):
    """Path of a file given by the teachers; if it is not found, the path where it is expected."""
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, folder, name) for folder in DATA_FOLDERS]
    return next((p for p in paths if os.path.exists(p)), paths[0])


TRAIN_PATH   = find_data_file('train.csv')
TEST_PATH    = find_data_file('test_no_labels.csv')   # clean test set published by the teachers (2026-10-09)
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


def load_test(path=TEST_PATH):
    """
    Load the clean test file: 4 columns (description, sample_name, transcription,
    keywords), NO header row and NO label. Returns one row per line of the file,
    in file order, so that row N is line N of results.txt.
    Only for producing predictions; nothing may ever be trained on it.
    """
    test = pd.read_csv(path, sep=';', quotechar='"', engine='python', header=None,
                       names=COLUMNS[1:])
    with open(path, 'rb') as f:
        data = f.read()
    n_lines = data.count(b'\n') + (0 if data.endswith(b'\n') else 1)
    assert len(test) == n_lines, f"{len(test)} records but {n_lines} lines: the file has multi-line records"
    return test


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


# The keywords write the specialty as e.g. "cardiovascular / pulmonary"
KEYWORD_TO_LABEL = {label.lower().replace('-', ' / '): label for label in VALID_LABELS}


def first_keyword_label(df):
    """
    The specialty named by the first keyword of each record, or 'none' when the
    record has no keywords (or the first keyword is not a specialty name).
    """
    first = df['keywords'].fillna('').str.split(',').str[0].str.strip().str.lower()
    return first.map(KEYWORD_TO_LABEL).fillna('none')


if __name__ == '__main__':
    # Running this file directly prints what the repair of train.csv does.
    load_train(verbose=True)
