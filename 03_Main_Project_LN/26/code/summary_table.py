"""
summary_table.py: held-out result of every model in one table (Table 3 of the paper).

Reads:   results/*.json written by the model scripts (no model is run).
Writes:  results/summary_table.csv (accuracy, macro-F1 and lowest class F1, in %).
Run:     python3 summary_table.py        (after M1, M2, M3 and M4_final_system.py)
"""

import json
import os
import pandas as pd

from model_utils import RESULTS_DIR

MODELS = [('M1 baseline', 'm1_baseline_original'),
          ('M2 classical, all fields', 'm2'),
          ('M3 language model, description', 'm3'),
          ('M4 = M3 + first keyword', 'm4'),
          ('ablation: M2 + first keyword', 'm2_first_keyword')]


def main():
    rows = []
    for label, name in MODELS:
        path = os.path.join(RESULTS_DIR, f'{name}.json')
        assert os.path.exists(path), f"missing {path}: run the script of that model first"
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        rows.append({'model': label,
                     'accuracy_%': round(r['accuracy'] * 100, 2),
                     'macro_f1_%': round(r['macro_f1'] * 100, 2),
                     'lowest_class_f1_%': round(r['lowest_class_f1'] * 100, 2),
                     'lowest_class': r['lowest_class']})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, 'summary_table.csv'), index=False)
    print(table.to_string(index=False))
    print("\n  Saved results/summary_table.csv")


if __name__ == '__main__':
    main()
