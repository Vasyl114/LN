"""
data_utils.py
=============
Shared data loading for Group 26 NLP Project.

PURPOSE:
    Every script must work on exactly the same records, so the loading and
    cleaning of train.csv lives here and is imported by the other scripts.

    The file given by the professors is never modified. It contains 7 rows
    whose 'description' cell absorbed the records that followed it (an opening
    quote that only closes several lines later). Three of those cells are cut
    at 32,759 characters and the rest of the cut record spills into the next
    row. load_train() repairs this in memory:
      - the first line of the cell stays as that row's description
      - every absorbed line becomes a record of its own again
      - a cut record is rejoined with its tail (one character is lost at the join)

HOW TO USE:
    from data_utils import load_train
    df = load_train()
"""

import csv
import numpy as np
import pandas as pd

TRAIN_PATH = '../../StudentsPack/train.csv'

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


if __name__ == '__main__':
    # Running this file directly writes the repaired data to train_clean.csv,
    # so the result of the repair can be inspected. The scripts do not need
    # this file: they call load_train() themselves.
    clean = load_train(verbose=True)
    clean.to_csv('train_clean.csv', sep=';', quotechar='"', quoting=1, index=False)
    print(f"  Saved: train_clean.csv ({len(clean)} records)")
